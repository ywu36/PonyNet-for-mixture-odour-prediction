# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   P = pairs (820)   N = pooled out-of-fold predictions
# ---------------------------------------------------------------------------------------------
"""B1b+ = B1b + related-component push (mixdist/b1b_plus.py), declared before running.

Rows, standard CV, change from B1a++ on the same folds (B1a++ reused from results/eval_o12c):
  B1a++ (baseline) | B1b (b1a+'s class, rerun here on the same folds) |
  B1b+ group: push = salience of the same RDKit group carried by different unshared molecules (min side, summed)
  B1b+ note:  push = F_note (noticeable notes present in both mixtures)
Key subset: share no molecule (B1b = B1a++ there; only the push can act); also all pairs, share >= 1, and
B1b+ vs B1b paired. Expected (joint-head runs, rows 5g-5h): about 0. Reported: push weight h over 250 CV fits.
CPU (small L-BFGS + OLS stages).

Env:     mist
Usage:   python evaluation/O1/eval_b1b_plus.py --variant b1b | group | note ; then --report
Writes:  results/eval_o12c/cv_b1bp_<variant>.pkl ; report results/eval_b1b_plus/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import evaluate, features  # noqa: E402
from mixdist.b1b import B1b  # noqa: E402
from mixdist.b1b_plus import B1bPlus  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402
from eval_common import common_features  # noqa: E402
from eval_salience_units import unit_terms  # noqa: E402

CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "b1bp_b1b": "B1b", "b1bp_group": "B1b+ (push: same group, different molecules)",
         "b1bp_note": "B1b+ (push: noticeable shared notes)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup(); op = features.load("openpom")
    if a.variant:
        store = []
        if a.variant == "b1b":
            fp = lambda tr, te: B1b(q, X, A, B, S, LAM).fit(tr, y[tr]).predict(te)
        else:
            Fp = unit_terms(train, q, op.cids_M, X > 0, True)[0] if a.variant == "group" else common_features(train, q, op.cids_M)[2]

            def fp(tr, te):
                f = B1bPlus(q, X, A, B, S, Fp, LAM).fit(tr, y[tr]); store.append(f.h[1]); return f.predict(te)
        res = evaluate.cross_validate(train, fp)
        pickle.dump({"res": res, "h": store}, open(CV / f"cv_b1bp_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_b1b_plus", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    run = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nN = np.array([len(p.a & p.b) for p in train])[ref.pair_N]
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1)]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a2"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in list(NAMES)[1:]:
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask)
            rows.append([NAMES[k_], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE [CI] / r [CI]"], rows)
    rep.section("2  B1b+ vs B1b (paired, same folds)")
    vrow = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1)]:
        for k_ in ("b1bp_group", "b1bp_note"):
            s = evaluate.score_cv(train, res[k_], reference=res["b1bp_b1b"], mask_N=mask)
            vrow.append([NAMES[k_], subset, f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "change from B1b: RMSE [CI] / r [CI]"], vrow)
    rep.table(["model", "push weight h per sd (250 fits) median [2.5, 97.5%]"],
              [[NAMES[k_], f"{np.median(run[k_]['h']):+.4f} [{np.percentile(run[k_]['h'], 2.5):+.4f}, {np.percentile(run[k_]['h'], 97.5):+.4f}]"] for k_ in ("b1bp_group", "b1bp_note")])
    rep.save()


if __name__ == "__main__":
    main()
