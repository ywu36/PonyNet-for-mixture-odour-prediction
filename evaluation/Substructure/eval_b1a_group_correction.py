# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   P = training pairs   M = molecules in the open-POM table   G = functional groups   F = features (2G)
# ---------------------------------------------------------------------------------------------
"""Finding 3, experiment E2: B1a + a correction from shared / differing functional groups.

    prediction = B1a (line on the open-POM baseline distance)  +  ridge(X_PxF)
    X: for each of the 24 RDKit functional groups g present in >= 10 training molecules (fixed
       before looking at any residual), with phi_g(A) = fraction of A's molecules containing g:
         "differ in g" |phi_g(A) - phi_g(B)|   and   "share g" min(phi_g(A), phi_g(B))

Steps (finding_3.html, E2), training pairs only:
  1. CV (10 x 5 zero-overlap folds): in each training fold fit B1a's line, then ridge on its residual
     (penalty chosen by inner zero-overlap folds of the training fold). Scored against B1a on the
     same folds (paired bootstrap), on all pairs and on pairs sharing no molecule.
  2. Chance level: the same CV with the group labels shuffled among molecules (each molecule keeps
     its number of groups), --perms times.
  3. Stability: L1 fit in every outer fold; a feature is stable if selected in >= 80% of folds with
     one sign. Dominance predicts "share g" < 0 and "differ in g" > 0.
  4. Molecule split, stable features only: the molecules carrying g are split in half 20 times; the
     coefficient is fitted on pairs whose g comes from one half and tested on the other half.
Pass (fixed in advance): pairs sharing no molecule, r change > 0 with 95% paired CI above 0, RMSE not
worse, all pairs not worse, and above the 95th percentile of the shuffled runs. V50/T46 only if it passes.

Env:     mist
Usage:   python evaluation/Substructure/eval_b1a_group_correction.py [--perms 100] [--quick]
Writes:  results/eval_b1a_group_correction/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import cv, data, evaluate, features, metrics, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402

MIN_MOLECULES = 10
ALPHAS = np.logspace(-1, 4, 20)


def group_table(smiles_M, train_rows):
    """Presence of every RDKit fr_* group per molecule; keep groups in >= MIN_MOLECULES training molecules."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    fns = {n.removeprefix("fr_"): f for n, f in Descriptors.descList if n.startswith("fr_")}
    has_MxG = np.array([[f(Chem.MolFromSmiles(s)) > 0 for f in fns.values()] for s in smiles_M])
    keep = has_MxG[train_rows].sum(0) >= MIN_MOLECULES
    return [n for n, k in zip(fns, keep) if k], has_MxG[:, keep]


def pair_features(pairs, row_of, has_MxG):
    """-> X_PxF: [differ in g for all g, share g for all g]."""
    def phi(mix):
        return has_MxG[[row_of[c] for c in mix]].mean(0)
    A = np.stack([phi(p.a) for p in pairs]); B = np.stack([phi(p.b) for p in pairs])
    return np.hstack([np.abs(A - B), np.minimum(A, B)])


class GroupCorrection:
    """fit_predict for evaluate.cross_validate; inner zero-overlap folds cached per training fold."""

    def __init__(self, pairs, dist_P, y_P, X_PxF, seed=0):
        self.pairs, self.d, self.y, self.X, self.seed = pairs, dist_P, y_P, X_PxF, seed
        self.inner, self.coefs = {}, []

    def inner_folds(self, train_idx):
        key = hash(train_idx.tobytes())
        if key not in self.inner:
            fs = cv.mixture_holdout_folds([self.pairs[i] for i in train_idx], k=5, repeats=1, seed=self.seed)
            self.inner[key] = [(np.array(f.train), np.array(f.test)) for f in fs if len(f.test)]
        return self.inner[key]

    def __call__(self, train_idx, test_idx, l1=False):
        a, c = evaluate.fit_line(self.d[train_idx], self.y[train_idx])
        resid = self.y[train_idx] - (a * self.d[train_idx] + c)
        sc = StandardScaler().fit(self.X[train_idx])
        Xtr, Xte = sc.transform(self.X[train_idx]), sc.transform(self.X[test_idx])
        model = RidgeCV(alphas=ALPHAS, cv=self.inner_folds(train_idx)).fit(Xtr, resid)
        if l1:
            alpha = LassoCV(cv=self.inner_folds(train_idx), n_alphas=30, max_iter=5000).fit(Xtr, resid).alpha_
            self.coefs.append(Lasso(alpha=alpha, max_iter=5000).fit(Xtr, resid).coef_)
        return a * self.d[test_idx] + c + model.predict(Xte)


def changes(result, ref):
    """[(subset, score with paired diff vs B1a)] on all pairs and pairs sharing no molecule."""
    out = []
    for name, m in [("all pairs", np.ones(len(result.pred_N), bool)), ("share no molecule", result.no_overlap_N)]:
        out.append((name, metrics.score(result.pred_N[m], result.truth_N[m], reference=ref.pred_N[m], n_boot=2000)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="2 repeats, 5 shuffles")
    args = ap.parse_args()
    if args.quick:
        args.perms, args.repeats = 5, 2
    rep = Report("eval_b1a_group_correction", vars(args), tag="quick" * args.quick)

    rep.section("1. Data and features")
    train = data.load_train()
    y_P = evaluate.labels(train)
    fs = features.load("openpom")
    row_of = {int(c): i for i, c in enumerate(fs.cids_M)}
    mol_tr = sorted(set().union(*[m for p in train for m in (p.a, p.b)]))
    train_rows = np.array([row_of[c] for c in mol_tr])
    gnames, has_MxG = group_table(fs.smiles_M, train_rows)
    fnames = [f"differ {g}" for g in gnames] + [f"share {g}" for g in gnames]
    X_PxF = pair_features(train, row_of, has_MxG)
    dist_P = mixture.paper_distance(train, fs.cids_M, fs.probs_MxW)
    rep.line(f"{len(train)} training pairs; {len(gnames)} groups in >= {MIN_MOLECULES} of {len(mol_tr)} training molecules: {', '.join(gnames)}")
    rep.line(f"{X_PxF.shape[1]} pair features; ridge penalty from inner zero-overlap folds, grid {ALPHAS[0]:g}..{ALPHAS[-1]:g}")

    rep.section(f"2. CV ({args.repeats} x 5 folds): B1a + group correction vs B1a on the same folds")
    t0 = time.time()
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(dist_P, y_P, np.zeros(len(train))), k=5, repeats=args.repeats)
    gc = GroupCorrection(train, dist_P, y_P, X_PxF)
    res = evaluate.cross_validate(train, lambda tr, te: gc(tr, te, l1=True), k=5, repeats=args.repeats)
    rows = changes(res, ref)
    rep.table(["subset", "n", "B1a RMSE / r", "change RMSE", "change r", "95% CI of r change", "p"],
              [[n, s["n"], f"{metrics.rmse(ref.pred_N[m], ref.truth_N[m]):.3f} / {s['r_ref']:.3f}",
                f"{s['rmse'] - metrics.rmse(ref.pred_N[m], ref.truth_N[m]):+.3f}", f"{s['r'] - s['r_ref']:+.3f}",
                f"[{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]", f"{s['p_vs_ref']:.3f}"]
               for (n, s), m in zip(rows, [np.ones(len(res.pred_N), bool), res.no_overlap_N])])
    gain = {n: s["r"] - s["r_ref"] for n, s in rows}
    rep.line(f"{time.time() - t0:.0f} s")

    rep.section(f"3. Chance level: group labels shuffled among molecules, {args.perms} times")
    rng = np.random.default_rng(0)
    null = {"all pairs": [], "share no molecule": []}
    for i in range(args.perms):
        perm = has_MxG.copy()
        perm[train_rows] = has_MxG[rng.permutation(train_rows)]
        gp = GroupCorrection(train, dist_P, y_P, pair_features(train, row_of, perm))
        gp.inner = gc.inner
        r = evaluate.cross_validate(train, gp, k=5, repeats=args.repeats)
        for n, m in [("all pairs", np.ones(len(r.pred_N), bool)), ("share no molecule", r.no_overlap_N)]:
            null[n].append(metrics.pearson(r.pred_N[m], r.truth_N[m]) - metrics.pearson(ref.pred_N[m], ref.truth_N[m]))
    for n in null:
        nb = np.array(null[n])
        rep.line(f"{n}: real r change {gain[n]:+.3f}; shuffled 95th percentile {np.percentile(nb, 95):+.3f}, "
                 f"median {np.median(nb):+.3f}; fraction of shuffles >= real {np.mean(nb >= gain[n]):.2f}")

    rep.section("4. Stability: L1 fit in every outer fold")
    C = np.array(gc.coefs)
    sel = (C != 0).mean(0); pos = (C > 0).mean(0); neg = (C < 0).mean(0)
    order = np.argsort(-sel)
    rep.table(["feature", "selected", "positive", "negative", "expected if dominant"],
              [[fnames[f], f"{sel[f]:.0%}", f"{pos[f]:.0%}", f"{neg[f]:.0%}",
                "> 0 (farther)" if fnames[f].startswith("differ") else "< 0 (closer)"] for f in order[:15]])
    stable = [f for f in range(len(fnames)) if max(pos[f], neg[f]) >= 0.8]
    rep.line("stable (>= 80% of folds, one sign): " + (", ".join(fnames[f] for f in stable) or "none"))

    rep.section("5. Molecule split for stable features: fitted on half the molecules carrying g, tested on the other half")
    n_P = np.bincount(ref.pair_N, minlength=len(train))
    oof_P = np.bincount(ref.pair_N, weights=ref.pred_N, minlength=len(train)) / np.maximum(n_P, 1)
    eps_P = np.where(n_P > 0, y_P - oof_P, np.nan)
    split_out = {}
    for f in stable:
        g = f % len(gnames)
        carriers = [r for r in train_rows if has_MxG[r, g]]
        rs = []
        for s in range(20):
            half = set(np.random.default_rng(s).permutation(carriers)[: len(carriers) // 2])
            side = []
            for p in train:
                rows_g = [row_of[c] for c in p.a | p.b if has_MxG[row_of[c], g]]
                side.append(0 if not rows_g else 1 if all(r in half for r in rows_g) else 2 if not any(r in half for r in rows_g) else 3)
            side = np.array(side)
            fit, test = (side == 1) & ~np.isnan(eps_P), (side == 2) & ~np.isnan(eps_P)
            if fit.sum() < 15 or test.sum() < 15:
                continue
            coef = np.polyfit(X_PxF[fit, f], eps_P[fit], 1)
            rs.append(metrics.pearson(np.polyval(coef, X_PxF[test, f]), eps_P[test]))
        rs = np.array(rs)
        split_out[fnames[f]] = rs.tolist()
        rep.line(f"{fnames[f]}: {len(rs)} usable splits; test r mean {np.nanmean(rs):+.3f}, positive in {np.mean(rs > 0):.0%}"
                 if len(rs) else f"{fnames[f]}: too few pairs per half")

    rep.section("6. Ridge coefficients on all training pairs (standardised features; for reading only)")
    sc = StandardScaler().fit(X_PxF)
    a, c = evaluate.fit_line(dist_P, y_P)
    full = RidgeCV(alphas=ALPHAS, cv=gc.inner_folds(np.arange(len(train)))).fit(sc.transform(X_PxF), y_P - (a * dist_P + c))
    o = np.argsort(-np.abs(full.coef_))
    rep.line(f"penalty {full.alpha_:g}; largest: " + ", ".join(f"{fnames[f]} {full.coef_[f]:+.4f}" for f in o[:10]))

    rep.value("gain", gain)
    rep.value("null", null)
    rep.value("selection", {fnames[f]: {"selected": float(sel[f]), "pos": float(pos[f]), "neg": float(neg[f])} for f in range(len(fnames))})
    rep.value("molecule_split", split_out)
    rep.save()


if __name__ == "__main__":
    main()
