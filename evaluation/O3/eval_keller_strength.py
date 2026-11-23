# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = DREAM pairs
#   T = atoms   K = Keller molecules (479)   S = Keller stimuli (molecule x dilution)
# ---------------------------------------------------------------------------------------------
"""Molecule strength learned from Keller & Vosshall 2016 measured intensity, read from the strongest atom, then used
in the DREAM readout (user 2026-11-20: "train on it and see the improvement"; "yes and also with dream, and see").
  backbone: plain O3 (o3m_w64), frozen, GS-LF singles only; atom vectors h_a (mean of h_a = v)
  strength module (as C34 / eval_ma_atom_strength): log-strength(m) = tau log mean_a exp(q . h_a / tau)   (q 64 + tau 1)
  Keller fit: standardized intensity_s = c0 + c1 log10 dilution_s + c2 log-strength(m_s); MSE + 1e-3 |q|^2;
      Adam 500 steps. c0, c1, c2 are Keller-only numbers (dilution is a Keller nuisance, never used on DREAM).
  Keller molecules that appear in any DREAM mixture (157) are REMOVED from all Keller training (DREAM molecules unseen).
  DREAM, frozen strength: shares ~ exp(beta * standardized log-strength + kappa * cos(v, others)); unit merge;
      angle; line. Fitted on DREAM: beta, kappa, a, b = 4 numbers.
  DREAM, joint: unit merge + atom soft max (C34, 68 DREAM numbers) with the SAME q, tau also fitting Keller:
      loss = DREAM MSE / var(y_train) + Keller MSE (standardized intensity, 3 Keller-only numbers); weight 1 fixed
      before running, not tuned.
Checks: (1) Keller, 5 unseen-molecule folds: RMSE / r of predicted intensity, dilution only vs + atom strength, and
r on 1/1000 stimuli only (no dilution difference). (2) Shuffled-Keller control: molecule identities permuted (seed 0).
Rows (fixed before running), 3 members averaged:
  O1.2b (reference, 27) | unit merge, equal weights + kappa (3) | unit merge + atom soft max fitted on DREAM (68; C34)
  | unit merge + Keller-learned atom strength (4) | unit merge + shuffled-Keller atom strength (4)
  | unit merge + atom soft max, joint DREAM + Keller (68 + 3 Keller-only)
CV: development pairs 50 x 5, six subsets; change from O1.2b and from DREAM-fitted atom soft max (same folds;
predictions saved). --heldout: T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_keller_strength.py [--heldout]
Writes:  results/eval_keller_strength/<date_time>.{txt,json} (+ _cv.pkl)
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_atom_attn import atom_vectors  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_ma_atom_strength import FixedStrength, log_strength  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
LAM = 1e-3
SEEDS = (0, 1, 2)
REF, K_EQ, K_DR, K_KE, K_SH, K_JO = ("O1.2b (reference, 27)", "unit merge, equal weights + kappa (3)",
                                     "unit merge + atom soft max fitted on DREAM (68)", "unit merge + Keller-learned atom strength (4)",
                                     "unit merge + shuffled-Keller atom strength (4)", "unit merge + atom soft max, joint DREAM + Keller (68)")


class KellerGraphs(o2.Graphs):
    """features/keller_graphs.npz (export_keller_graphs.py) with o2.Graphs batching."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.cids, self.in_dream = z["cids"], z["in_dream"]
        self.node, self.edge, self.edge_index = z["node"], z["edge"], z["edge_index"]
        self.node_off, self.edge_off = z["node_off"], z["edge_off"]
        ok = np.isfinite(z["I_S"])
        self.mol_S, self.logdil_S, self.I_S = z["mol_S"][ok], z["logdil_S"][ok].astype(float), z["I_S"][ok].astype(float)


class KellerData:
    """Keller stimuli restricted to molecules ``keep_K``; torch tensors; intensity standardized on these stimuli."""

    def __init__(self, KG, H, mol, keep_K, perm=None):
        keepS = keep_K[KG.mol_S]; ms = KG.mol_S[keepS]
        if perm is not None:
            ms = perm[ms]
        self.H, self.mol, self.n = torch.as_tensor(H), torch.as_tensor(mol), int(mol.max()) + 1
        self.ms = torch.as_tensor(ms); self.ld = torch.as_tensor(KG.logdil_S[keepS] + 3.0)   # centred at 1/1000
        I = KG.I_S[keepS]; self.mu, self.sd = I.mean(), I.std(); self.I = torch.as_tensor((I - self.mu) / self.sd)

    def loss(self, q, t, c):
        ls = log_strength(self.H, self.mol, self.n, q, t)
        return ((c[0] + c[1] * self.ld + c[2] * ls[self.ms] - self.I) ** 2).mean()


def fit_keller(kd, steps=500):
    q = torch.zeros(kd.H.shape[1], dtype=F64, requires_grad=True)
    t = torch.tensor(float(np.log(kd.H.std().item() * 3)), dtype=F64, requires_grad=True)
    c = torch.tensor([0.0, 0.0, 1.0], dtype=F64, requires_grad=True)
    opt = torch.optim.Adam([q, t, c], lr=0.05)
    for _ in range(steps):
        loss = kd.loss(q, t, c) + LAM * (q ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return q.detach(), t.detach(), c.detach()


def predict_keller(KG, H, mol, q, t, c, mu, sd, stim):
    with torch.no_grad():
        ls = log_strength(torch.as_tensor(H), torch.as_tensor(mol), int(mol.max()) + 1, q, t).numpy()
    return mu + sd * (c[0].item() + c[1].item() * (KG.logdil_S[stim] + 3.0) + c[2].item() * ls[KG.mol_S[stim]])


class JointAtom(UnitAtom):
    """C34 readout; q, tau also fit Keller intensity (DREAM MSE / var(y) + Keller MSE)."""

    def __init__(self, V, H, mol, A, B, kd):
        super().__init__(V, H, mol, A, B); self.kd = kd

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float)); vy = y.var()
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        t = torch.tensor(self.t0, dtype=F64, requires_grad=True); b0 = torch.zeros((), dtype=F64)
        c = torch.tensor([0.0, 0.0, 0.0], dtype=F64, requires_grad=True)
        with torch.no_grad():
            self.lw = self.mol_lw(q, t); d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, q, kappa, t, c], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.mol_lw(q, t)
            loss = (((lin[0] * self.distance(kappa, b0, idx) + lin[1] - y) ** 2).mean() / vy + self.kd.loss(q, t, c)
                    + self.lam * ((q ** 2).sum() + kappa ** 2))
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.mol_lw(q, t).detach()
            self.keller_mse = self.kd.loss(q, t, c).item()
        self.q, self.kappa, self.t, self.b, self.lin, self.c = q.detach(), kappa.detach(), t.detach(), b0, lin.detach(), c.detach()
        return self


def setup(dev, rep):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); KG = KellerGraphs(features.features_dir() / "keller_graphs.npz")
    _, tr, _, _ = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    nK = len(KG.cids); keep = ~KG.in_dream
    # shuffle among kept molecules only (shuffled control keeps the same molecule set)
    kidx = np.where(keep)[0]; perm = np.arange(nK); perm[kidx] = np.random.default_rng(0).permutation(kidx)
    rng = np.random.default_rng(0); fold_K = rng.integers(0, 5, nK)
    rep.line(f"Keller: {nK} molecules, {len(KG.mol_S)} stimuli with ratings; used for training: {int(keep.sum())} molecules "
             f"({int(keep[KG.mol_S].sum())} stimuli) not in any DREAM mixture; 5 unseen-molecule folds")
    out, chk = [], {k: [] for k in ("dilution only", "+ atom strength", "+ atom strength, shuffled")}
    chk1000 = {k: [] for k in chk}
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        HD, molD = atom_vectors(n, G, rows_D, dev); HD = HD / scale
        VD = np.zeros((len(rows_D), HD.shape[1])); np.add.at(VD, molD, HD); VD /= np.bincount(molD)[:, None]
        HK, molK = atom_vectors(n, KG, np.arange(nK), dev); HK = HK / scale
        # (1) unseen-molecule folds on Keller (kept molecules only)
        pred = {k: np.full(len(KG.mol_S), np.nan) for k in chk}
        for f in range(5):
            trK = keep & (fold_K != f); teS = keep[KG.mol_S] & (fold_K[KG.mol_S] == f)
            for nm, pm in (("+ atom strength", None), ("+ atom strength, shuffled", perm)):
                kd = KellerData(KG, HK, molK, trK, pm); q, t, c = fit_keller(kd)
                pred[nm][teS] = predict_keller(KG, HK, molK, q, t, c, kd.mu, kd.sd, teS)
            kd = KellerData(KG, HK, molK, trK); X = np.c_[np.ones(len(kd.ld)), kd.ld.numpy()]
            w = np.linalg.lstsq(X, kd.I.numpy(), rcond=None)[0]
            pred["dilution only"][teS] = kd.mu + kd.sd * (w[0] + w[1] * (KG.logdil_S[teS] + 3.0))
        ev = keep[KG.mol_S]; e1000 = ev & np.isclose(KG.logdil_S, -3.0)
        for k in chk:
            chk[k].append((metrics.rmse(pred[k][ev], KG.I_S[ev]), metrics.pearson(pred[k][ev], KG.I_S[ev])))
            chk1000[k].append(metrics.pearson(pred[k][e1000], KG.I_S[e1000]))
        # (2) final modules on all kept Keller molecules -> log-strength of DREAM molecules
        lsD = {}
        for nm, pm in (("real", None), ("shuffled", perm)):
            kd = KellerData(KG, HK, molK, keep, pm); q, t, c = fit_keller(kd)
            with torch.no_grad():
                lsD[nm] = log_strength(torch.as_tensor(HD), torch.as_tensor(molD), len(VD), q, t).numpy()
            rep.line(f"  member {s}, Keller {nm}: c (intercept, dilution, strength) {c.numpy().round(3)}; "
                     f"tau / SD of atom scores {np.exp(t.item()) / (kd.H @ q).std().item():.2f}; DREAM log-strength SD {lsD[nm].std():.3f}")
        out.append({"V": VD, "H": HD, "mol": molD, "ls": lsD, "kd": KellerData(KG, HK, molK, keep)})
    rep.section("Keller check: unseen Keller molecules (5 folds), predicted intensity (0-100 scale); mean over 3 members")
    rep.table(["model", "RMSE", "r (all stimuli)", "r (1/1000 stimuli only)"],
              [[k, f"{np.mean([a for a, _ in v]):.2f}", f"{np.mean([b for _, b in v]):.3f}", f"{np.mean(chk1000[k]):.3f}"] for k, v in chk.items()])
    rep.line(f"1/1000 stimuli evaluated: {int((keep[KG.mol_S] & np.isclose(KG.logdil_S, -3.0)).sum())}")
    return G, out


def models(BB, q_op, X, A, B):
    def ens(make):
        return lambda tr, te, y: np.mean([make(m).fit(tr, y[tr]).predict(te) for m in BB], 0)
    zero = np.zeros(len(BB[0]["V"]))
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            K_EQ: ens(lambda m: FixedStrength(m["V"], A, B, zero)),
            K_DR: ens(lambda m: UnitAtom(m["V"], m["H"], m["mol"], A, B)),
            K_KE: ens(lambda m: FixedStrength(m["V"], A, B, m["ls"]["real"])),
            K_SH: ens(lambda m: FixedStrength(m["V"], A, B, m["ls"]["shuffled"])),
            K_JO: ens(lambda m: JointAtom(m["V"], m["H"], m["mol"], A, B, m["kd"]))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_keller_strength", {"lam": LAM, "seeds": SEEDS, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    G, BB = setup(dev, rep); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "same-size share none": (k_P == 0) & (gap <= 1),
               "setting 2 (share >= 1)": k_P >= 1, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        res = {}
        for nm, fp in models(BB, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y))
            rep.line(f"done: {nm}; non-finite predictions {int((~np.isfinite(res[nm].pred_N)).sum())}")
        pk = ROOT / "results/eval_keller_strength" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv.pkl"
        pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"CV predictions saved: {pk.name}")

        def sc(r_, sub, against):
            m = SUB[sub]; m = np.isfinite(r_.pred_N) if m is None else (m[r_.pair_N] & np.isfinite(r_.pred_N))
            if against is not None:
                m = m & np.isfinite(against.pred_N)
            return evaluate.score_cv(P, r_, reference=against, mask_N=m)
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[K_DR])) if nm not in (REF, K_DR) else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from DREAM-fitted atom soft max"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; dr = preds[K_DR]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = c2 = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                if nm not in (REF, K_DR):
                    cr, _ = paired(p, dr, y)
                    c2 = f"{s['rmse'] - metrics.rmse(dr, y):+.4f} / {s['r'] - metrics.pearson(dr, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c, c2])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]", "change from DREAM-fitted soft max"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
