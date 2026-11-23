# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   T = atoms
# ---------------------------------------------------------------------------------------------
"""O5's single-molecule atom scores combined with the strongest-atom soft max (user go-ahead 2026-11-20: "yes u can go
with it"; finding 6 sec. 12, C34, C36, C38). 10 members of each backbone, readout fitted per member, predictions averaged.
  O5 = plain O3 with attention pooling over atoms (checkpoints/o3m_attpool; atom score s_a learned by the word loss on
  GS-LF singles). s_a divided by the SD over DREAM molecules of O5's own strength (log mean_a exp s_a), so at tau = 1
  row (a) equals the 4-number row.
  (a) O5 own atom scores, soft max: log-weight = beta * tau log mean_a exp(s_a / tau); fits beta, tau, kappa, line = 5.
  (b) O5 atom soft max with a DREAM query shrunk toward O5's own scorer: atom score = h_a . (beta q0 + delta), q0 = least
      squares of s_a on O5's hidden atom vectors h_a (DREAM atoms); penalty 1e-3 |delta|^2 (C34's penalty, but toward
      O5's scorer instead of toward 0); fits delta 64, beta, tau, kappa, line = 69.
Rows (fixed before running), unit merge and angle everywhere:
  O1.2b (reference, 27)
  plain O3: unit merge + atom soft max fitted on DREAM (68)   [current best, C34]
  O5: own strength + odd-one-out (4)                          [C36]
  O5: own atom scores, soft max (5)                           (a)
  O5: atom soft max fitted on DREAM, shrunk toward 0 (68)      (separates the prior from the backbone)
  O5: atom soft max fitted on DREAM, shrunk toward own scorer (69)   (b)
CV: development pairs 50 x 5, six subsets; change from O1.2b and from the current best (same folds; predictions saved).
--heldout: T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_o5_softmax.py [--heldout]
         CV in parallel parts: --rows 0,2,3 | --rows 1 | --rows 4 | --rows 5, then --merge <part pickles>
Writes:  results/eval_o5_softmax/<date_time>.{txt,json} (+ _cv.pkl)
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
from eval_attpool import LearnedStrength, attpool_atoms  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
LAM = 1e-3
SEEDS = tuple(int(x) for x in __import__("os").environ.get("O5_SEEDS", " ".join(map(str, range(10)))).split())
REF, CUR, O4, OA, O0, OB = ("O1.2b (reference, 27)", "plain O3: unit merge + atom soft max fitted on DREAM (68) [current best]",
                            "O5: own strength + odd-one-out (4)", "O5: own atom scores, soft max (5)",
                            "O5: atom soft max fitted on DREAM, shrunk toward 0 (68)",
                            "O5: atom soft max fitted on DREAM, shrunk toward own scorer (69)")


def backbones(dev, rep):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = {"plain": [], "o5": []}
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        H, mol = atom_vectors(n, G, rows_D, dev); H = H / scale
        V = np.zeros((len(rows_D), H.shape[1])); np.add.at(V, mol, H); V /= np.bincount(mol)[:, None]
        out["plain"].append({"V": V, "H": H, "mol": mol})
        n = train_o3m.load(ROOT / f"checkpoints/o3m_attpool/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        H, sc, mol, V = attpool_atoms(n, G, rows_D, dev); H, V = H / scale, V / scale
        cnt = np.bincount(mol); mx = np.full(len(cnt), -np.inf); np.maximum.at(mx, mol, sc)
        se = np.zeros(len(cnt)); np.add.at(se, mol, np.exp(sc - mx[mol])); strength = mx + np.log(se / cnt)
        s_atoms = sc / strength.std()
        Xa = np.c_[H, np.ones(len(H))]; w = np.linalg.lstsq(Xa, s_atoms, rcond=None)[0]; q0 = w[:-1]
        r2 = 1 - ((Xa @ w - s_atoms) ** 2).sum() / ((s_atoms - s_atoms.mean()) ** 2).sum()
        rep.line(f"member {s}: O5 strength SD {strength.std():.3f}; own scorer read from hidden atom vectors R^2 {r2:.3f}")
        out["o5"].append({"V": V, "H": H, "mol": mol, "strength": strength, "s": s_atoms, "q0": q0})
    return G, out


class PriorAtom(UnitAtom):
    """Unit merge; molecule log-weight = tau log mean_a exp(score_a / tau);
    mode "own": score = beta * s_a (5 numbers); mode "shrink": score = h_a . (beta q0 + delta), penalty on delta (69)."""

    def __init__(self, V, H, mol, A, B, s_atoms, q0, mode):
        super().__init__(V, H, mol, A, B)
        self.s = torch.as_tensor(s_atoms, dtype=F64); self.q0 = torch.as_tensor(q0, dtype=F64); self.mode = mode

    def smax(self, z, t):
        tau = torch.exp(t).clamp_min(1e-4); z = z / tau
        mx = torch.full((self.M,), -1e30, dtype=F64).scatter_reduce(0, self.mol, z, reduce="amax", include_self=True)
        se = torch.zeros(self.M, dtype=F64).index_add(0, self.mol, torch.exp(z - mx[self.mol]))
        return tau * (mx + torch.log(se / self.n))

    def score(self, beta, delta):
        return beta * self.s if self.mode == "own" else self.H @ (beta * self.q0 + delta)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        beta = torch.zeros((), dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        t = torch.zeros((), dtype=F64, requires_grad=True); b0 = torch.zeros((), dtype=F64)
        delta = torch.zeros(self.H.shape[1], dtype=F64, requires_grad=self.mode == "shrink")
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t); d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        ps = [lin, beta, kappa, t] + ([delta] if self.mode == "shrink" else [])
        opt = torch.optim.LBFGS(ps, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.smax(self.score(beta, delta), t)
            loss = (((lin[0] * self.distance(kappa, b0, idx) + lin[1] - y) ** 2).mean()
                    + self.lam * (beta ** 2 + kappa ** 2 + (delta ** 2).sum()))
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t).detach()
        self.beta, self.kappa, self.t, self.delta, self.b, self.lin = beta.detach(), kappa.detach(), t.detach(), delta.detach(), b0, lin.detach()
        return self


def models(BB, q_op, X, A, B):
    def ens(k, make):
        return lambda tr, te, y: np.mean([make(m).fit(tr, y[tr]).predict(te) for m in BB[k]], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            CUR: ens("plain", lambda m: UnitAtom(m["V"], m["H"], m["mol"], A, B)),
            O4: ens("o5", lambda m: LearnedStrength(m["V"], m["strength"], A, B, True)),
            OA: ens("o5", lambda m: PriorAtom(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], "own")),
            O0: ens("o5", lambda m: UnitAtom(m["V"], m["H"], m["mol"], A, B)),
            OB: ens("o5", lambda m: PriorAtom(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], "shrink"))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--rows", default="", help="CV only: comma list of row numbers (0-5) to fit; saves a part pickle, no scoring")
    ap.add_argument("--merge", nargs="*", default=None, help="CV only: part pickles to merge and score (no fitting)")
    a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_o5_softmax", {"lam": LAM, "seeds": SEEDS, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    G, BB = (None, None) if a.merge else backbones(dev, rep); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    assert G is None or list(G.dream_cids) == [int(c) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "same-size share none": (k_P == 0) & (gap <= 1),
               "setting 2 (share >= 1)": k_P >= 1, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        if a.merge:
            res = {}
            for f in a.merge:
                res.update(pickle.load(open(f, "rb"))["res"])
            res = {nm: res[nm] for nm in (REF, CUR, O4, OA, O0, OB) if nm in res}; rep.line(f"merged rows: {list(res)}")
        fa = [] if a.merge else [PriorAtom(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], "own").fit(np.arange(len(P)), y) for m in BB["o5"]]
        if fa:
            rep.line("(a) fitted on all development pairs: beta " + ", ".join(f"{m.beta.item():+.2f}" for m in fa)
                     + "; tau " + ", ".join(f"{np.exp(m.t.item()):.2f}" for m in fa))
        mdl = {} if a.merge else models(BB, q_op, X, A, B)
        if a.rows:
            keep = [list(mdl)[int(i)] for i in a.rows.split(",")]; mdl = {k: mdl[k] for k in keep}
        if not a.merge:
            res = {}
        for nm, fp in mdl.items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y))
            rep.line(f"done: {nm}; non-finite predictions {int((~np.isfinite(res[nm].pred_N)).sum())}")
        if not a.merge:
            tag = f"_rows{a.rows.replace(',', '-')}" if a.rows else ""
            pk = ROOT / "results/eval_o5_softmax" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv{tag}.pkl"
            pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"CV predictions saved: {pk.name}")
        if a.rows:
            rep.save(); return

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
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[CUR])) if nm not in (REF, CUR) else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from current best"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; cur = preds[CUR]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = c2 = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                if nm not in (REF, CUR):
                    cr, _ = paired(p, cur, y)
                    c2 = f"{s['rmse'] - metrics.rmse(cur, y):+.4f} / {s['r'] - metrics.pearson(cur, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c, c2])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]", "change from current best"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
