# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   P = pairs (820)   I = (pair, shared molecule) instances
# ---------------------------------------------------------------------------------------------
"""Salience of shared molecules, deeper (follow-up to eval_common.py), declared before running.

Known (eval_common, change from B1a++, share >= 1): + k (shared count) +0.026; + F_mol + k +0.039 with
g_Fmol -0.080 and g_k +0.051 in every CV fit; + mean salience + k +0.025 (g ~0). So each shared molecule s contributes
(-b c_s + d): a salient one pulls the pair closer, a bland one pushes (relative to B1a++). Base here: + F_mol + k.
c_X(s) = |q_s - mean q of X's unshared molecules| (company-relative standout; 0 if X has no unshared molecule).
Rows, all on top of B1a++ with + k, change from B1a++ on the same folds:
  min side (base: noticed in both)  F = sum_s min(c_A(s), c_B(s))
  max side (noticed in either)      F = sum_s max(c_A(s), c_B(s))
  global distinctiveness (no context) F = sum_s |q_s - mean q over the 235 molecules|
  base + size interaction           + sum_s min(c_A, c_B) * log(mean size / 10)   (identity fades in large mixtures -> > 0)
  shuffled salience (null)          min-side values permuted across all (pair, shared molecule) instances (seed 0)
Reading: min >= max -> must be noticed in both; company > global -> context, not a fixed trait; interaction > 0 ->
salience matters more in small mixtures; shuffled ~ k alone. Subsets as eval_common. CPU (small L-BFGS model).

Env:     mist
Usage:   python evaluation/O1/eval_salience.py --variant min|max|global|size|shuf ; then --report
Writes:  results/eval_o12c/cv_sal_<variant>.pkl ; report results/eval_salience/<date_time>.{txt,json}
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
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402
from eval_o15 import pair_contrast  # noqa: E402

CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "common_k": "+ k only (reference)", "sal_min": "+ k + salience, min side (base)",
         "sal_max": "+ k + salience, max side", "sal_global": "+ k + global distinctiveness (no context)",
         "sal_size": "+ k + salience + salience x log(size/10)", "sal_shuf": "+ k + shuffled salience (null)"}
TERMS = {"sal_min": ["salience", "k"], "sal_max": ["salience", "k"], "sal_global": ["distinct", "k"],
         "sal_size": ["salience", "k", "salience x size"], "sal_shuf": ["shuffled", "k"], "common_k": ["k"]}


def features_all(train, q, cids):
    row = {int(c): i for i, c in enumerate(cids)}; gmean = q.mean(0)
    P_I, cmin, cmax, glob = [], [], [], []
    for p_, pr in enumerate(train):
        sh = pr.a & pr.b
        for s in sh:
            cs = []
            for mine in (pr.a, pr.b):
                un = [row[c] for c in mine - sh]
                cs.append(np.linalg.norm(q[row[s]] - q[un].mean(0)) if un else 0.0)
            P_I.append(p_); cmin.append(min(cs)); cmax.append(max(cs)); glob.append(np.linalg.norm(q[row[s]] - gmean))
    P_I, cmin, cmax, glob = map(np.array, (P_I, cmin, cmax, glob))
    shuf = np.random.default_rng(0).permutation(cmin)
    agg = lambda v: np.bincount(P_I, weights=v, minlength=len(train))
    k = np.array([len(p.a & p.b) for p in train], float)
    lsize = np.log(np.array([(len(p.a) + len(p.b)) / 2 for p in train]) / 10)
    z = lambda v: (v - v.mean()) / v.std()
    Fmin = agg(cmin)
    return {"sal_min": np.c_[z(Fmin), z(k)], "sal_max": np.c_[z(agg(cmax)), z(k)], "sal_global": np.c_[z(agg(glob)), z(k)],
            "sal_size": np.c_[z(Fmin), z(k), z(Fmin * lsize)], "sal_shuf": np.c_[z(agg(shuf)), z(k)], "common_k": z(k)[:, None]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup(); op = features.load("openpom")
    F = features_all(train, q, op.cids_M)
    if a.variant:
        v = f"sal_{a.variant}"; Sx = np.c_[S, F[v]]; store = []

        def fp(tr, te):
            f = B1aPlus(q, X, A, B, Sx, LAM).fit(tr, y[tr]); store.append(f.lin.numpy()[4:].copy())
            return f.predict(te)
        res = evaluate.cross_validate(train, fp)
        pickle.dump({"res": res, "g": store}, open(CV / f"cv_{v}.pkl", "wb")); print("saved", v); return

    rep = Report("eval_salience", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    run = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nsh = np.array([len(p.a & p.b) for p in train]); nN = nsh[ref.pair_N]
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1), ("share 1-2 (audit)", (nN >= 1) & (nN <= 2))]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a2"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in list(NAMES)[1:]:
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask)
            rows.append([NAMES[k_], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE [CI] / r [CI]"], rows)
    rep.section("1b  Each variant vs the base form (salience, min side), same folds: change RMSE [CI] / r [CI]")
    vrow = []
    for subset, mask in [("all pairs", None), ("share >= 1", nN >= 1)]:
        for k_ in ("sal_max", "sal_global", "sal_size", "sal_shuf", "common_k"):
            s = evaluate.score_cv(train, res[k_], reference=res["sal_min"], mask_N=mask)
            vrow.append([NAMES[k_], subset, f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / "
                         f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["variant", "subset", "change from base: RMSE [CI] / r [CI]"], vrow)
    rep.section("2  Head weights per sd, 250 CV fits (median [2.5, 97.5%]); < 0 = closer")
    grow = []
    for k_ in list(NAMES)[1:]:
        G = np.array(run[k_]["g"])
        for j, lb in enumerate(TERMS[k_]):
            grow.append([NAMES[k_], lb, f"{np.median(G[:, j]):+.4f} [{np.percentile(G[:, j], 2.5):+.4f}, {np.percentile(G[:, j], 97.5):+.4f}]"])
    rep.table(["model", "term", "weight"], grow)
    con_P, _, _ = pair_contrast(train, q, op.cids_M); sh = nsh >= 1; crow = []
    for k_, r in res.items():
        num = np.bincount(r.pair_N, weights=r.pred_N, minlength=len(train)); den = np.bincount(r.pair_N, minlength=len(train))
        crow.append([NAMES[k_], f"{np.corrcoef((y - num / np.maximum(den, 1))[sh], con_P[sh])[0, 1]:+.3f}"])
    rep.table(["model", "r(OOF error, b1a+ contrast), share >= 1"], crow)
    rep.save()


if __name__ == "__main__":
    main()
