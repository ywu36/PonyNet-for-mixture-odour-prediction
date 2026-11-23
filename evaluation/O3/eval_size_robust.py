# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   T = atoms
# ---------------------------------------------------------------------------------------------
"""Size-robust (scale-invariant) mixture comparison on model (b) (user 2026-11-20: "average will pull at the shared but
should still be scale invariant; check how to achieve it with our O5 prior + DREAM (10 members), and loss function").
Finding 6 sec. 19 / C48: our distance shrinks with the number of molecules (r -0.37 on 81 same-size share-none pairs;
label +0.09) because averaging more unit directions pulls every mixture toward the shared direction.
Base: (b) = O5 (o3m_attpool, 10 members, frozen, GS-LF singles) + unit merge + strongest-atom soft max with the DREAM
atom query shrunk toward O5's own attention + odd-one-out + angle + line (69 numbers; eval_o5_softmax.PriorAtom).
Fixes (all label-free except the loss row; mixture size = number of molecules, known at test time):
  centre        subtract the mean unit direction of the 235 DREAM molecules before merging, renormalise each molecule
                (mixtures then average toward 0, not toward the shared direction).
  size-neutral  divide each pair's distance by the expected distance between random mixtures of the same two sizes
                (equal weights, drawn from the 235 DREAM molecules, 200 draws per size pair, seed 0).
  size loss     fit with an extra penalty 1.0 x corr(distance, mean log size)^2 over the training pairs (weight fixed
                before running), so the fitted distance cannot follow mixture size.
Rows (fixed before running), 10 members averaged: O1.2b (27) | (b) (69) | (b) + centre | (b) + size-neutral |
  (b) + centre + size-neutral | (b) + size loss.
Diagnostics (no fitting): 0-number O5 distances with each fix: r with mixture size and with the label on the 81
same-size share-none development pairs.
CV: development pairs 50 x 5, six subsets; change from O1.2b and from (b) (same folds). --heldout: T46 then V50.
CV in parallel parts: --rows 0,1 | --rows 2 | ... then --merge <part pickles>.

Env:     mist
Usage:   python evaluation/O3/eval_size_robust.py [--heldout | --rows i,j | --merge pkl ...]
Writes:  results/eval_size_robust/<date_time>.{txt,json} (+ _cv[_rows..].pkl)
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
from mixdist import data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402

F64 = torch.float64
LAM = 1e-3
SIZE_LOSS = 1.0
REF, RB = "O1.2b (reference, 27)", "(b) (69)"
VARIANTS = {RB: (False, False, False), "(b) + centre": (True, False, False), "(b) + size-neutral": (False, True, False),
            "(b) + centre + size-neutral": (True, True, False), "(b) + size loss": (False, False, True)}


def centred_units(V):
    U = V / np.linalg.norm(V, axis=1, keepdims=True); U = U - U.mean(0); return U / np.linalg.norm(U, axis=1, keepdims=True)


def null_distance(U, sizes, draws=200, seed=0):
    """Expected distance between random equal-weight mixtures of sizes (na, nb) from the molecules U (rows unit)."""
    M = len(U); out = {}
    for na, nb in sorted(set(sizes)):
        rng = np.random.default_rng([seed, int(na), int(nb)])
        d = []
        for _ in range(draws):
            ia = rng.choice(M, na, replace=False); ib = rng.choice(np.setdiff1d(np.arange(M), ia), nb, replace=False)
            za, zb = U[ia].mean(0), U[ib].mean(0); d.append(np.linalg.norm(za / np.linalg.norm(za) - zb / np.linalg.norm(zb)))
        out[(na, nb)] = float(np.mean(d))
    return out


_NULL = {}


def cached_null(key, U, sizes):
    """null_distance per (member vectors, centre) key; computed once per size pair."""
    tab = _NULL.setdefault(key, {}); new = sorted(set(sizes) - set(tab))
    if new:
        tab.update(null_distance(U, new))
    return tab


class SizeRobust(eo.PriorAtom):
    def __init__(self, V, H, mol, A, B, s_atoms, q0, centre, neutral, size_loss):
        super().__init__(V, H, mol, A, B, s_atoms, q0, "shrink"); self.size_loss = size_loss
        if centre:
            self.U = torch.as_tensor(centred_units(np.asarray(V)), dtype=F64)
        na = self.A.sum(1).long().numpy(); nb = self.B.sum(1).long().numpy()
        self.S = torch.as_tensor((np.log(na) + np.log(nb)) / 2, dtype=F64)
        if neutral:
            nd = cached_null((id(V), centre), self.U.numpy(), list(zip(na, nb))); self.F = torch.as_tensor([nd[(a, b)] for a, b in zip(na, nb)], dtype=F64)
        else:
            self.F = torch.ones(len(na), dtype=F64)

    def distance(self, kappa, b, idx):
        return super().distance(kappa, b, idx) / self.F[idx]

    def fit(self, idx, y, max_iter=300):
        if not self.size_loss:
            return super().fit(idx, y, max_iter)
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float)); s = self.S[idx]
        beta = torch.zeros((), dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        t = torch.zeros((), dtype=F64, requires_grad=True); b0 = torch.zeros((), dtype=F64)
        delta = torch.zeros(self.H.shape[1], dtype=F64, requires_grad=True)
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t); d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, beta, kappa, t, delta], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.smax(self.score(beta, delta), t); d = self.distance(kappa, b0, idx)
            dc, sc = d - d.mean(), s - s.mean(); corr = (dc * sc).sum() / (dc.norm() * sc.norm() + 1e-12)
            loss = (((lin[0] * d + lin[1] - y) ** 2).mean() + SIZE_LOSS * y.var() * corr ** 2
                    + self.lam * (beta ** 2 + kappa ** 2 + (delta ** 2).sum()))
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t).detach()
        self.beta, self.kappa, self.t, self.delta, self.b, self.lin = beta.detach(), kappa.detach(), t.detach(), delta.detach(), b0, lin.detach()
        return self


def models(BB, q_op, X, A, B):
    def ens(c, n, l):
        return lambda tr, te, y: np.mean([SizeRobust(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], c, n, l).fit(tr, y[tr]).predict(te) for m in BB], 0)
    out = {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)}
    out.update({nm: ens(*v) for nm, v in VARIANTS.items()}); return out


def diagnostics(rep, BB, op):
    from eval_zero_param import mix_dist
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    k = np.array([len(set(p.a) & set(p.b)) for p in P]); na = A.sum(1).astype(int); nb = B.sum(1).astype(int)
    m = (k == 0) & (np.abs(na - nb) <= 1); sz = (np.log(na) + np.log(nb)) / 2
    rows = []
    for nm, (c, n) in {"none (unit merge)": (False, False), "centre": (True, False), "size-neutral": (False, True), "centre + size-neutral": (True, True)}.items():
        ds = []
        for mm in BB:
            U = centred_units(mm["V"]) if c else mm["V"] / np.linalg.norm(mm["V"], axis=1, keepdims=True)
            d = mix_dist(U, A, B, np.exp(mm["strength"] - mm["strength"].max()))
            if n:
                nd = null_distance(U, list(zip(na, nb))); d = d / np.array([nd[(a, b)] for a, b in zip(na, nb)])
            ds.append(d)
        d = np.mean(ds, 0)
        rows.append([nm, f"{metrics.pearson(d[m], sz[m]):+.3f}", f"{metrics.pearson(d[m], y[m]):+.3f}", f"{metrics.pearson(d[k == 0], y[k == 0]):+.3f}",
                     f"{metrics.pearson(d, y):+.3f}"])
    rep.section("Diagnostics, 0 DREAM numbers (O5 + own attention weights, 10 members); label r with size on the 81 pairs +0.085")
    rep.table(["fix", "r with mixture size (81)", "r with label (81 same-size share none)", "r share none (186)", "r all (514)"], rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--rows", default=""); ap.add_argument("--merge", nargs="*", default=None); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_size_robust", {"lam": LAM, "size_loss": SIZE_LOSS, "members": eo.SEEDS, "heldout": a.heldout, "rows": a.rows,
                                      "repeats": evaluate.CV_REPEATS, "device": dev}, tag="heldout" if a.heldout else (f"rows{a.rows}" if a.rows else ""))
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    BB = None if a.merge else eo.backbones(dev, rep)[1]["o5"]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if a.heldout:
        diagnostics(rep, BB, op)
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rb = preds[RB]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = c2 = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y); c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                if nm not in (REF, RB):
                    cr, _ = paired(p, rb, y); c2 = f"{s['rmse'] - metrics.rmse(rb, y):+.4f} / {s['r'] - metrics.pearson(rb, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c, c2])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]", "change from (b)"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
        rep.save(); return
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    if a.merge:
        res = {}
        for f in a.merge:
            res.update(pickle.load(open(f, "rb"))["res"])
        order = [REF] + list(VARIANTS); res = {nm: res[nm] for nm in order if nm in res}; rep.line(f"merged rows: {list(res)}")
    else:
        mdl = models(BB, q_op, X, A, B)
        if a.rows:
            mdl = {k_: mdl[k_] for k_ in [list(mdl)[int(i)] for i in a.rows.split(",")]}
        res = {}
        for nm, fp in mdl.items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y))
            rep.line(f"done: {nm}; non-finite predictions {int((~np.isfinite(res[nm].pred_N)).sum())}")
        tag = f"_rows{a.rows.replace(',', '-')}" if a.rows else ""
        pk = ROOT / "results/eval_size_robust" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv{tag}.pkl"
        pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"CV predictions saved: {pk.name}")
        if a.rows:
            rep.save(); return
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "same-size share none": (k_P == 0) & (gap <= 1),
           "share none, sizes differ": (k_P == 0) & (gap >= 2), "setting 2 (share >= 1)": k_P >= 1,
           "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}

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
                         "" if nm == REF else ch(s_), ch(sc(r_, sub, res[RB])) if nm not in (REF, RB) and RB in res else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from (b)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
