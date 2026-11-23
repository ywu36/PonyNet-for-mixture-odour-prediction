# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   T = DREAM atoms   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Does the atom soft max readout (68 DREAM numbers) improve when we add what we learned? (user, 2026-11-20: "ok do it")
Every row: frozen network, molecules merged as unit directions, molecule log-weight = soft max over its atoms' scores
(sharpness tau fitted) + odd-one-out, rating = slope x angle + intercept. Rows, fixed before running:
  O1.2b (27)                                     reference
  current best (68)                              plain O3 atoms, atom score = q . atom vector, penalty 1e-3 (= UnitAtom)
  + agreement (69)                               + b x difference of log |merged vector| (coherent vs cancelled blend; C32)
  penalty 0.01 (68) / penalty 0.1 (68)           C38: the 64-number scorer is fragile under rescaling
  O5 own scores (4)                              O5 network, learned strength, no DREAM-fitted scorer (finding 7)
  O5 equal weights (3)                           O5 network, equal molecule weights (+ odd-one-out): value of O5's scores
  O5 prior + correction (69)                     O5 atoms; atom score = c x O5's learned atom score (standardised) +
                                                 q . atom vector, q pulled to 0 by the penalty: DREAM adjusts what single
                                                 molecules taught instead of learning the scorer from scratch
Non-finite predictions (any row) are masked in every row and counted; predictions are pickled after each model
(re-score without refitting: --from-pkl <file>). Added 2026-11-20 after a NaN crash lost eval_attpool's CV.
Seeds from O5_SEEDS (default "0 1 2"), predictions averaged over members. CV: 514 development pairs, 50 x 5, all
subsets; --heldout: fit on 820 training pairs, T46 then V50. No overlap term, no routing, no group weights.

Env:     mist
Usage:   [O5_SEEDS="0 1 2"] python evaluation/O3/eval_atom_integrate.py [--heldout] [--smoke]
Writes:  results/eval_atom_integrate/<date_time>[_heldout].{txt,json}
"""
import argparse
import pickle
import sys
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
from mixdist.report import Report, results_dir  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_unit_merge import LAM  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_atom_attn import atom_vectors  # noqa: E402
from eval_attpool import attpool_atoms, LearnedStrength, REF, CUR, SEEDS  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
O5 = "O5 own scores (4)"


def logmeanexp(z_T, mol_T, M):
    mx = torch.full((M,), -1e30, dtype=F64).scatter_reduce(0, mol_T, z_T, reduce="amax", include_self=True)
    se = torch.zeros(M, dtype=F64).index_add(0, mol_T, torch.exp(z_T - mx[mol_T]))
    return mx + torch.log(se / torch.bincount(mol_T, minlength=M).double())


class AtomReadout(ecd.ContextReadoutQ):
    """Unit merge; molecule log-weight = tau x log mean exp((c x prior_a + q . H_a) / tau) + odd-one-out; optional agreement."""

    def __init__(self, V, H, mol, A, B, lam=LAM, agree=False, prior=None):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=lam); self.agree = agree
        self.U = self.V / self.V.norm(dim=1, keepdim=True)
        self.H = torch.as_tensor(H, dtype=F64); self.mol = torch.as_tensor(mol); self.M = len(V)
        self.P = None if prior is None else torch.as_tensor((prior - prior.mean()) / prior.std(), dtype=F64)
        rq = torch.as_tensor(np.random.default_rng(0).normal(size=H.shape[1]) / np.sqrt(H.shape[1]), dtype=F64)
        self.t0 = float(np.log(10 * (self.H @ rq).std().item()))   # same start as eval_atom_attn.AtomAttn

    def mol_lw(self, q, c, t):
        s = self.H @ q + (c * self.P if self.P is not None else 0)
        tau = torch.exp(t).clamp_min(1e-4)
        return tau * logmeanexp(s / tau, self.mol, self.M)

    def _out(self, kappa, Mx, C, idx):
        lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
        W = Mx[idx] * torch.exp(lw); z = (W @ self.U) / W.sum(1, keepdim=True); n = z.norm(dim=1, keepdim=True)
        return z / n, torch.log(n[:, 0])

    def dist(self, kappa, b, idx):
        uA, hA = self._out(kappa, self.A, self.CA, idx); uB, hB = self._out(kappa, self.B, self.CB, idx)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + (b ** 2 * ((hA - hB) / self.s) ** 2 if self.agree else 0) + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        t = torch.tensor(self.t0, dtype=F64, requires_grad=True); c = torch.zeros((), dtype=F64, requires_grad=self.P is not None)
        b = torch.tensor(0.1, dtype=F64, requires_grad=self.agree)
        with torch.no_grad():
            self.lw = self.mol_lw(q, c, t)
            _, hA = self._out(kappa, self.A, self.CA, idx); _, hB = self._out(kappa, self.B, self.CB, idx)
            self.s = torch.cat([hA, hB]).std().clamp_min(1e-6); d = self.dist(kappa, b, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        params = [lin, q, kappa, t] + ([c] if self.P is not None else []) + ([b] if self.agree else [])
        opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.mol_lw(q, c, t)
            loss = ((lin[0] * self.dist(kappa, b, idx) + lin[1] - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2 + c ** 2)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.mol_lw(q, c, t).detach()
        self.q, self.kappa, self.t, self.c, self.bb, self.lin = q.detach(), kappa.detach(), t.detach(), c.detach(), b.detach(), lin.detach()
        return self

    def predict(self, idx):
        with torch.no_grad():
            return (self.lin[0] * self.dist(self.kappa, self.bb, torch.as_tensor(np.asarray(idx))) + self.lin[1]).numpy()


def backbones(dev):
    """plain O3: (V, H, mol); O5: (V, H, mol, atom scores, molecule strength). Same scaling as eval_attpool.backbones."""
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = {"plain": [], "o5": []}
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        H, mol = atom_vectors(n, G, rows_D, dev); H = H / scale
        V = np.zeros((len(rows_D), H.shape[1])); np.add.at(V, mol, H); V /= np.bincount(mol)[:, None]
        out["plain"].append((V, H, mol))
        n = train_o3m.load(ROOT / f"checkpoints/o3m_attpool/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
            H, sc, mol, V = attpool_atoms(n, G, rows_D, dev)
        st = logmeanexp(torch.as_tensor(sc), torch.as_tensor(mol), len(V)).numpy()
        out["o5"].append((V / scale, H / scale, mol, sc, st))
    return out


def models(BB, q_op, X, A, B):
    def ens(k, make):
        return lambda tr, te, y: np.mean([make(*b).fit(tr, y[tr]).predict(te) for b in BB[k]], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            CUR: ens("plain", lambda V, H, mol: UnitAtom(V, H, mol, A, B)),
            "current best + agreement (69)": ens("plain", lambda V, H, mol: AtomReadout(V, H, mol, A, B, agree=True)),
            "current best, penalty 0.01 (68)": ens("plain", lambda V, H, mol: AtomReadout(V, H, mol, A, B, lam=1e-2)),
            "current best, penalty 0.1 (68)": ens("plain", lambda V, H, mol: AtomReadout(V, H, mol, A, B, lam=1e-1)),
            O5: ens("o5", lambda V, H, mol, sc, st: LearnedStrength(V, st, A, B, True)),
            "O5 equal weights (3)": ens("o5", lambda V, H, mol, sc, st: LearnedStrength(V, st, A, B, False)),
            "O5 prior + correction (69)": ens("o5", lambda V, H, mol, sc, st: AtomReadout(V, H, mol, A, B, prior=sc))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-pkl", default="")
    a = ap.parse_args(); ecd.DIRECTION_ONLY = True; dev = pick_device()
    BB = backbones(dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    if a.smoke:   # AtomReadout with defaults must reproduce UnitAtom (the current best)
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M); tr = np.arange(400); te = np.arange(400, len(P))
        V, H, mol = BB["plain"][0]
        p1 = UnitAtom(V, H, mol, A, B).fit(tr, y[tr]).predict(te); p2 = AtomReadout(V, H, mol, A, B).fit(tr, y[tr]).predict(te)
        print("max |UnitAtom - AtomReadout|", np.abs(p1 - p2).max())
        m = AtomReadout(*BB["o5"][0][:3], A, B, prior=BB["o5"][0][3]).fit(tr, y[tr])
        print("O5 prior fit: c", m.c.item(), "|q|", m.q.norm().item(), "tau", torch.exp(m.t).item(), "r", metrics.pearson(m.predict(te), y[te]))
        m = AtomReadout(V, H, mol, A, B, agree=True).fit(tr, y[tr]); print("agree b", m.bb.item(), "r", metrics.pearson(m.predict(te), y[te]))
        return
    rep = Report("eval_atom_integrate", {"seeds": SEEDS, "lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
               "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "setting 2 (share >= 1)": k_P >= 1,
               "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        if a.from_pkl:
            res = pickle.load(open(a.from_pkl, "rb")); rep.line(f"predictions from {a.from_pkl}")
        else:
            res = {}; pk = results_dir() / "eval_atom_integrate" / f"{rep.time:%Y-%m-%d_%H%M%S}_cv.pkl"; pk.parent.mkdir(parents=True, exist_ok=True)
            for nm, fp in models(BB, q_op, X, A, B).items():
                res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")
                pickle.dump(res, open(pk, "wb"))
            rep.line(f"predictions saved to {pk}")
        fin_N = np.all([np.isfinite(r_.pred_N) for r_ in res.values()], 0)
        for nm, r_ in res.items():
            rep.line(f"non-finite predictions: {nm}: {(~np.isfinite(r_.pred_N)).sum()} of {len(r_.pred_N)}")
        rep.line(f"masked in every row: {(~fin_N).sum()} of {len(fin_N)}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=fin_N if m is None else fin_N & m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), "" if nm in (REF, CUR) else ch(sc(r_, sub, res[CUR]))])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from current best"], rows)
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
                    cr, ce = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE [CI] / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
