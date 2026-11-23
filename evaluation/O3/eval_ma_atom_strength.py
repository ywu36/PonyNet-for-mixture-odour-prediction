# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = DREAM pairs
#   T = atoms   R = Ma mixtures (two molecules each)   K = backbone members
# ---------------------------------------------------------------------------------------------
"""Molecule strength learned OUTSIDE DREAM, from Ma's "which molecule dominates", read from the strongest atom
(user go-ahead 2026-11-20; clues C1, C16, C29, C32, C33).
  backbone: plain O3 (o3m_w64), frozen, GS-LF singles only; atom vectors h_a (mean of h_a = v)
  strength module: atom score q . h_a; molecule log-strength = tau log mean_a exp(q . h_a / tau)   (q 64 + tau 1)
  training (Ma only): two-molecule mixture z = (w_A u_A + w_B u_B) / (w_A + w_B), u = v / |v| (unit merge);
      d = cos(z, u_A) - cos(z, u_B); loss = softplus(-(2y - 1) d / T) + 1e-3 |q|^2, y = [A dominates] (sign of L)
      Adam 500 steps (as C1 / eval_o3w.fit_weights); T learned.
  DREAM: q, tau frozen; shares ~ exp(beta * log-strength + kappa * cos(v, others)); unit merge; angle; line.
      Fitted on DREAM: beta, kappa, a, b = 4 numbers.
Checks: (1) Ma, 5 unseen-molecule folds: AUROC of d (strength trained without the fold's molecules) vs equal weights.
(2) Shuffled-Ma control: same module trained on shuffled dominance labels (seed 0), then the same 4-number DREAM fit.
Rows (fixed before running), 3 members averaged:
  O1.2b (reference, 27) | unit merge, equal weights + kappa (3) | unit merge + atom soft max fitted on DREAM (68; C34)
  | unit merge + Ma-learned atom strength (4) | unit merge + shuffled-Ma atom strength (4)
CV: development pairs 50 x 5, six subsets incl. same-size share none (share none, size gap <= 1); change from O1.2b and
from equal weights (same folds; predictions saved). --heldout: T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_ma_atom_strength.py [--heldout]
Writes:  results/eval_ma_atom_strength/<date_time>.{txt,json} (+ _cv.pkl)
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_atom_attn import atom_vectors  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
LAM = 1e-3
SEEDS = (0, 1, 2)
REF, K_EQ, K_DR, K_MA, K_SH = ("O1.2b (reference, 27)", "unit merge, equal weights + kappa (3)",
                               "unit merge + atom soft max fitted on DREAM (68)", "unit merge + Ma-learned atom strength (4)",
                               "unit merge + shuffled-Ma atom strength (4)")


def log_strength(H, mol, n_mol, q, t):
    """tau log mean_a exp(q . h_a / tau) per molecule (tau = exp(t), floored)."""
    tau = torch.exp(t).clamp_min(1e-4); z = (H @ q) / tau
    mx = torch.full((n_mol,), -1e30, dtype=F64).scatter_reduce(0, mol, z, reduce="amax", include_self=True)
    se = torch.zeros(n_mol, dtype=F64).index_add(0, mol, torch.exp(z - mx[mol]))
    return tau * (mx + torch.log(se / torch.bincount(mol, minlength=n_mol).double()))


def fit_ma(H, mol, U, a, b, y, steps=500):
    """Strength module on Ma two-molecule mixtures (a, b: molecule indices into U; y: 1 if a dominates)."""
    H, mol, U = torch.as_tensor(H), torch.as_tensor(mol), torch.as_tensor(U); a, b = torch.as_tensor(a), torch.as_tensor(b)
    y = torch.as_tensor(y, dtype=F64); n = len(U)
    q = torch.zeros(H.shape[1], dtype=F64, requires_grad=True)
    t = torch.tensor(float(np.log(H.std().item() * 3)), dtype=F64, requires_grad=True)
    lT = torch.tensor(np.log(0.1), dtype=F64, requires_grad=True)
    opt = torch.optim.Adam([q, t, lT], lr=0.05)
    for _ in range(steps):
        d = ma_d(H, mol, U, a, b, q, t, n)
        loss = torch.nn.functional.softplus(-(2 * y - 1) * d / torch.exp(lT)).mean() + LAM * (q ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return q.detach(), t.detach()


def ma_d(H, mol, U, a, b, q, t, n):
    ls = log_strength(H, mol, n, q, t); wa, wb = torch.exp(ls[a] - torch.maximum(ls[a], ls[b])), torch.exp(ls[b] - torch.maximum(ls[a], ls[b]))
    z = (wa[:, None] * U[a] + wb[:, None] * U[b]) / (wa + wb)[:, None]; c = torch.nn.functional.cosine_similarity
    return c(z, U[a], dim=1) - c(z, U[b], dim=1)


class FixedStrength(ecd.ContextReadoutQ):
    """Unit merge; shares ~ exp(beta * fixed log-strength + kappa * cos(v, others)); fits beta, kappa, line (4)."""

    def __init__(self, V, A, B, ls0):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=LAM)
        self.U = self.V / self.V.norm(dim=1, keepdim=True); ls0 = torch.as_tensor(ls0, dtype=F64)
        self.ls0 = (ls0 - ls0.mean()) / ls0.std().clamp_min(1e-12) if ls0.abs().sum() > 0 else ls0

    def distance(self, kappa, b, idx):
        def out(Mx, C):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(lw); z = (W @ self.U) / W.sum(1, keepdim=True); return z / z.norm(dim=1, keepdim=True)
        return torch.sqrt(((out(self.A, self.CA) - out(self.B, self.CB)) ** 2).sum(1) + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        beta = torch.zeros((), dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True); b0 = torch.zeros((), dtype=F64)
        with torch.no_grad():
            self.lw = beta * self.ls0; d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True); opt = torch.optim.LBFGS([lin, beta, kappa], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = beta * self.ls0
            loss = ((lin[0] * self.distance(kappa, b0, idx) + lin[1] - y) ** 2).mean() + self.lam * kappa ** 2
            loss.backward(); return loss
        opt.step(closure)
        self.beta, self.kappa, self.b, self.lin = beta.detach(), kappa.detach(), b0, lin.detach(); self.lw = (self.beta * self.ls0).detach()
        return self


def setup(dev, rep):
    """Per member: DREAM (V, H, mol), Ma-trained log-strength on DREAM molecules (real, shuffled); Ma fold AUROC."""
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    _, tr, _, _ = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    ma_mols = np.unique(np.r_[S.ma_a, S.ma_b]); ia = np.searchsorted(ma_mols, S.ma_a); ib = np.searchsorted(ma_mols, S.ma_b)
    y_ma = (S.ma_L > 0).astype(float); y_sh = np.random.default_rng(0).permutation(y_ma)
    rep.line(f"Ma: {len(y_ma)} two-molecule mixtures, {len(ma_mols)} molecules, A dominates in {y_ma.mean():.2f}; 5 unseen-molecule folds")
    out, auc = [], {"equal weights": [], "Ma-learned atom strength": []}
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        HD, molD = atom_vectors(n, G, rows_D, dev); HD = HD / scale
        VD = np.zeros((len(rows_D), HD.shape[1])); np.add.at(VD, molD, HD); VD /= np.bincount(molD)[:, None]
        HM, molM = atom_vectors(n, S, ma_mols, dev); HM = HM / scale
        VM = np.zeros((len(ma_mols), HM.shape[1])); np.add.at(VM, molM, HM); VM /= np.bincount(molM)[:, None]
        UM = VM / np.linalg.norm(VM, axis=1, keepdims=True)
        # (1) Ma unseen-molecule folds
        pred, pred0 = np.zeros(len(y_ma)), np.zeros(len(y_ma))
        for f in range(5):
            trf, tef = S.ma_split(f); q, t = fit_ma(HM, molM, UM, ia[trf], ib[trf], y_ma[trf])
            with torch.no_grad():
                pred[tef] = ma_d(torch.as_tensor(HM), torch.as_tensor(molM), torch.as_tensor(UM), torch.as_tensor(ia[tef]),
                                 torch.as_tensor(ib[tef]), q, t, len(UM)).numpy()
                pred0[tef] = ma_d(torch.as_tensor(HM), torch.as_tensor(molM), torch.as_tensor(UM), torch.as_tensor(ia[tef]),
                                  torch.as_tensor(ib[tef]), torch.zeros_like(q), t, len(UM)).numpy()
        auc["Ma-learned atom strength"].append(roc_auc_score(y_ma, pred)); auc["equal weights"].append(roc_auc_score(y_ma, pred0))
        # (2) final modules on all Ma (real, shuffled) -> log-strength of DREAM molecules
        lsD = {}
        for nm, yy in (("real", y_ma), ("shuffled", y_sh)):
            q, t = fit_ma(HM, molM, UM, ia, ib, yy)
            with torch.no_grad():
                lsD[nm] = log_strength(torch.as_tensor(HD), torch.as_tensor(molD), len(VD), q, t).numpy()
            rep.line(f"  member {s}, Ma {nm}: tau / SD of atom scores {np.exp(t.item()) / (torch.as_tensor(HM) @ q).std().item():.2f}; "
                     f"DREAM log-strength SD {lsD[nm].std():.3f}")
        out.append({"V": VD, "H": HD, "mol": molD, "ls": lsD})
    rep.line("Ma unseen-molecule AUROC (equal weights pred = 0.5 by construction if symmetric): " +
             "; ".join(f"{k} {np.mean(v):.3f} ({', '.join(f'{x:.3f}' for x in v)})" for k, v in auc.items()))
    return G, out


def models(BB, q_op, X, A, B):
    def ens(make):
        return lambda tr, te, y: np.mean([make(m).fit(tr, y[tr]).predict(te) for m in BB], 0)
    zero = np.zeros(len(BB[0]["V"]))
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            K_EQ: ens(lambda m: FixedStrength(m["V"], A, B, zero)),
            K_DR: ens(lambda m: UnitAtom(m["V"], m["H"], m["mol"], A, B)),
            K_MA: ens(lambda m: FixedStrength(m["V"], A, B, m["ls"]["real"])),
            K_SH: ens(lambda m: FixedStrength(m["V"], A, B, m["ls"]["shuffled"]))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_ma_atom_strength", {"lam": LAM, "seeds": SEEDS, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
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
        fits = [FixedStrength(m["V"], A, B, m["ls"]["real"]).fit(np.arange(len(P)), y) for m in BB]
        rep.line("Ma strength on DREAM, fitted on all development pairs: beta " + ", ".join(f"{m.beta.item():+.3f}" for m in fits))
        res = {}
        for nm, fp in models(BB, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y))
            rep.line(f"done: {nm}; non-finite predictions {int((~np.isfinite(res[nm].pred_N)).sum())}")
        pk = ROOT / "results/eval_ma_atom_strength" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv.pkl"
        pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"CV predictions saved: {pk.name}")

        def sc(r_, sub, against):
            m = SUB[sub]; m = np.isfinite(r_.pred_N) if m is None else (m[r_.pair_N] & np.isfinite(r_.pred_N))
            return evaluate.score_cv(P, r_, reference=against, mask_N=m)
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[K_EQ])) if nm not in (REF, K_EQ) else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from equal weights"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
