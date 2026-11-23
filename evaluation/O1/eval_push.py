# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   P = pairs (820)   N = pooled out-of-fold predictions
# ---------------------------------------------------------------------------------------------
"""Does a standout UNSHARED molecule push the pair apart? (user question 2026-11-18), declared before running.

Why: every B1 / O1.2 model averages profiles, so a molecule that stands out from its own mixture is diluted by 1/n,
yet people may still notice it (Laing: 3-4 components identifiable). A noticeable component in A with no counterpart in
B should make the pair more different than the averaged profiles say. This acts on every pair, including pairs sharing
no molecule (setting 1, the V50/T46 setting). Earlier: shared standouts pull (B1b, setting 2 only); same group in
different standout molecules pushes weakly (+0.032 per sd; B1b+ = B1b). The plain unshared-standout push was not tested.

Per unshared molecule m of side X (other side Y), theta-free B1a profiles q:
  c(m)  = |q_m - mean q of X's other molecules|  (standout from its company; 0 if X has one molecule)
  nn(m) = min over y in Y of |q_m - q_y|           (how far its nearest counterpart on the other side is)
Per side, then summed over the two sides (0 for a side with no unshared molecule):
  max    max c                              the most noticeable unique component (main form)
  sum    sum c                              all of them (grows with size)
  top3   sum of the 3 largest c             Laing's 3-4 identifiable components
  disp   mean c                             control: mixture heterogeneity, not one standout
  gap    max c - mean c, + disp             standout beyond heterogeneity
  glob   max |q_m - mean q of 235|          control: distinct in general, no company
  unm    max c * nn                         standout AND unmatched on the other side
  size   max, + max * log(mean size / 10)   push fades in big mixtures (expected < 0)
  shuf   max permuted within (nA, nB) size strata (seed 0)   null: keeps size, breaks content
Each set is added (z-scored over pairs) as head covariates to two bases: B1a++ (headline reference) and O1.2b (best on
V50/T46). Joint fit, theta lam 1e-3, standard CV 50 x 5 on the same folds as results/eval_o12c (bases reused).
Expected sign: > 0 (push apart). Subsets: all pairs (820), share no molecule (325, setting 1), share >= 1 (495).
Audit first: r(base OOF error, feature) before any fit. Lead = r change vs its own base with CI > 0 on share no
molecule or all pairs, RMSE not worse, and above the shuffled null. Only a lead goes to V50/T46 (scored once).
CPU: small L-BFGS models on 820 pairs (single-threaded, parallel runs).

Env:     mist
Usage:   python evaluation/O1/eval_push.py --base b1a2|o12b --variant NAME ; --audit ; --report
Writes:  results/eval_o12c/cv_push_<base>_<variant>.pkl ; report results/eval_push/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import evaluate, features  # noqa: E402
from mixdist.b1a_plus import B1aPlus  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402

CV = ROOT / "results/eval_o12c"
VARIANTS = {"max": "max standout", "sum": "sum standout", "top3": "top-3 standout", "disp": "control: mean standout (heterogeneity)",
            "gap": "max - mean standout, + heterogeneity", "glob": "control: global distinctiveness",
            "unm": "standout x unmatched", "size": "max standout + x log(size/10)", "shuf": "null: shuffled max within size strata"}
TERMS = {"max": ["max"], "sum": ["sum"], "top3": ["top3"], "disp": ["mean"], "gap": ["gap", "mean"], "glob": ["glob"],
         "unm": ["unm"], "size": ["max", "max x size"], "shuf": ["shuffled"]}
BASES = {"b1a2": "B1a++", "o12b": "O1.2b"}


def raw_features(train, q, cids):
    row = {int(c): i for i, c in enumerate(cids)}; gmean = q.mean(0)
    names = ("max", "sum", "top3", "mean", "gap", "glob", "unm")
    R = {n: np.zeros(len(train)) for n in names}
    for p_, pr in enumerate(train):
        for X, Y in ((pr.a, pr.b), (pr.b, pr.a)):
            xs = [row[c] for c in X]; ys = [row[c] for c in Y]; un = [row[c] for c in X - Y]
            if not un or len(xs) < 2:
                continue
            tot = q[xs].sum(0); n = len(xs)
            c = np.array([np.linalg.norm(q[m] - (tot - q[m]) / (n - 1)) for m in un])
            nn = np.array([np.linalg.norm(q[ys] - q[m], axis=1).min() for m in un])
            cs = np.sort(c)[::-1]
            R["max"][p_] += cs[0]; R["sum"][p_] += c.sum(); R["top3"][p_] += cs[:3].sum(); R["mean"][p_] += c.mean()
            R["gap"][p_] += cs[0] - c.mean(); R["glob"][p_] += max(np.linalg.norm(q[m] - gmean) for m in un)
            R["unm"][p_] += (c * nn).max()
    lsize = np.log(np.array([(len(p.a) + len(p.b)) / 2 for p in train]) / 10)
    R["max x size"] = R["max"] * lsize
    strata = {}
    for p_, pr in enumerate(train):
        strata.setdefault(tuple(sorted((len(pr.a), len(pr.b)))), []).append(p_)
    rng = np.random.default_rng(0); sh = R["max"].copy()
    for ix in strata.values():
        sh[ix] = R["max"][rng.permutation(ix)]
    R["shuffled"] = sh
    return R


def feature_sets(train, q, cids):
    R = raw_features(train, q, cids)
    z = lambda v: (v - v.mean()) / v.std()
    return {v: np.column_stack([z(R[t]) for t in TERMS[v]]) for v in VARIANTS}, R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base"); ap.add_argument("--variant")
    ap.add_argument("--audit", action="store_true"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup(); op = features.load("openpom")
    F, R = feature_sets(train, q, op.cids_M)
    if a.variant:
        store = []
        if a.base == "b1a2":
            Sx = np.c_[S, F[a.variant]]; off = 4

            def fp(tr, te):
                f = B1aPlus(q, X, A, B, Sx, LAM).fit(tr, y[tr]); store.append(f.lin.numpy()[off:].copy()); return f.predict(te)
        else:
            Fx = F[a.variant]; off = 2

            def fp(tr, te):
                f = O12b(q, X, A, B, True, True, size_PxS=Fx, lam=LAM).fit(tr, y[tr]); store.append(f.lin.numpy()[off:].copy())
                return f.predict(te)
        res = evaluate.cross_validate(train, fp)
        pickle.dump({"res": res, "g": store}, open(CV / f"cv_push_{a.base}_{a.variant}.pkl", "wb")); print("saved", a.base, a.variant); return

    rep = Report("eval_push", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    nsh = np.array([len(p.a & p.b) for p in train]); s0 = nsh == 0
    base = {b: pickle.load(open(CV / f"cv_{b}.pkl", "rb"))["res"] for b in BASES}
    rep.section("0  Audit: features and base errors (before any fit)")
    lsz = np.log(np.array([(len(p.a) + len(p.b)) / 2 for p in train]))
    mism = np.abs(np.log([len(p.a) for p in train]) - np.log([len(p.b) for p in train]))
    arow = []
    for t in ("max", "sum", "top3", "mean", "gap", "glob", "unm", "shuffled"):
        cells = [t, f"{R[t][s0].mean():.2f} ({(R[t][s0] > 0).sum()} > 0)", f"{np.corrcoef(R[t], lsz)[0, 1]:+.2f}",
                 f"{np.corrcoef(R[t], mism)[0, 1]:+.2f}"]
        for b, r in base.items():
            num = np.bincount(r.pair_N, weights=r.pred_N, minlength=len(train)); den = np.bincount(r.pair_N, minlength=len(train))
            err = y - num / np.maximum(den, 1)
            cells += [f"{np.corrcoef(err, R[t])[0, 1]:+.3f}", f"{np.corrcoef(err[s0], R[t][s0])[0, 1]:+.3f}"]
        arow.append(cells)
    rep.table(["feature", "share-0 mean (pairs > 0)", "r with log size", "r with size mismatch",
               "r(B1a++ error), all", "r(B1a++ error), share 0", "r(O1.2b error), all", "r(O1.2b error), share 0"], arow)
    rep.line("error = label - pooled OOF prediction; r > 0 = base under-predicts the distance where the feature is high (push missing).")
    if a.audit:
        rep.save(); return

    run = {(b, v): pickle.load(open(CV / f"cv_push_{b}_{v}.pkl", "rb")) for b in BASES for v in VARIANTS}
    ref = base["b1a2"]
    assert all(np.array_equal(r["res"].pair_N, ref.pair_N) for r in run.values()), "folds differ"
    nN = nsh[ref.pair_N]
    fmt = lambda s, kk: (f"{s[kk + 'rmse_change']:+.4f} [{s[kk + 'rmse_change_ci'][0]:+.4f}, {s[kk + 'rmse_change_ci'][1]:+.4f}] / "
                         f"{s[kk + 'r_change']:+.3f} [{s[kk + 'r_change_ci'][0]:+.3f}, {s[kk + 'r_change_ci'][1]:+.3f}]")
    for b in BASES:
        rep.section(f"1-{b}  Standard CV ({evaluate.CV_REPEATS} x 5), on {BASES[b]}: change from B1a++ and from {BASES[b]} (same pairs)")
        rows = []
        for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1)]:
            sb = evaluate.score_cv(train, base[b], reference=ref, mask_N=mask)
            rows.append([BASES[b], subset, sb["pairs"], f"{sb['rmse']:.4f}", f"{sb['r']:.3f}",
                         "" if b == "b1a2" else fmt(sb, ""), ""])
            for v in VARIANTS:
                r = run[(b, v)]["res"]
                s1 = evaluate.score_cv(train, r, reference=ref, mask_N=mask)
                s2 = evaluate.score_cv(train, r, reference=base[b], mask_N=mask)
                rows.append([f"+ {VARIANTS[v]}", subset, s1["pairs"], f"{s1['rmse']:.4f}", f"{s1['r']:.3f}", fmt(s1, ""), fmt(s2, "")])
        rep.table(["model", "subset", "pairs", "RMSE", "r", "change from B1a++: RMSE [CI] / r [CI]",
                   f"change from {BASES[b]}: RMSE [CI] / r [CI]"], rows)
    rep.section("2  Push weights per sd, 250 CV fits (median [2.5, 97.5%]); > 0 = pushes apart")
    grow = []
    for b in BASES:
        for v in VARIANTS:
            G = np.array(run[(b, v)]["g"])
            for j, lb in enumerate(TERMS[v]):
                grow.append([BASES[b], VARIANTS[v], lb, f"{np.median(G[:, j]):+.4f} [{np.percentile(G[:, j], 2.5):+.4f}, {np.percentile(G[:, j], 97.5):+.4f}]",
                             f"{(G[:, j] > 0).mean():.2f}"])
    rep.table(["base", "variant", "term", "weight", "fraction of fits > 0"], grow)
    rep.save()


if __name__ == "__main__":
    main()
