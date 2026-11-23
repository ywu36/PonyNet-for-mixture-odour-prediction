# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""O1.3 metric variants (mixdist/o13.py; design/o1_2_design.md "Next: the metric side"), standard CV.

All with the 24 RDKit group gains (lam 1e-3) and a line; metric parameters unpenalised, starts in o13.INIT.
Rows, declared before running, all reported, change from B1a on the same folds:
  B1a (baseline) | O1.2b gain + length (current best) | M1 angle | M2 pfe (uncertainty) | M3 hyper (Poincare) |
  M4 white (factor out the common direction) | M5 margin (angle threshold) | reference B1a++ (groups + size covariates)
Subsets: all pairs, share no molecule; share 1 as audit (chosen after seeing results). Full-data parameters reported.
Run one variant per process (parallel), then --report scores all saved runs.
Note found before the CV (in-sample, theta = 0, grid): M2 peaks at r ~0.21 and M3 at ~0.35, both below B1a (0.43).

Env:     mist
Usage:   python evaluation/O1/eval_o13.py --variant NAME   (NAME in b1a, o12b, angle, pfe, hyper, white, margin, b1a2)
         python evaluation/O1/eval_o13.py --report
Writes:  results/eval_o13/cv_<NAME>.pkl ; report results/eval_o13/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.o13 import O13  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3
OUT = ROOT / "results/eval_o13"
NAMES = {"b1a": "B1a (baseline)", "o12b": "O1.2b gain + length", "angle": "M1 angle + length", "pfe": "M2 uncertainty (PFE)",
         "hyper": "M3 hyperbolic", "white": "M4 factor out white", "margin": "M5 angle threshold + length", "b1a2": "reference: B1a++"}


def setup():
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(train, op.cids_M); S = size_features(train)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    return train, y, op, q, A, B, S, X


def make(name, y, op, train, q, A, B, S, X):
    if name == "b1a":
        return evaluate.linear_calibration(mixture.paper_distance(train, op.cids_M, op.probs_MxW), y, np.zeros(len(train)))
    if name == "o12b":
        return lambda tr, te: O12b(q, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)
    if name == "b1a2":
        return lambda tr, te: B1aPlus(q, X, A, B, S, LAM).fit(tr, y[tr]).predict(te)
    return lambda tr, te: O13(q, X, A, B, name, LAM).fit(tr, y[tr]).predict(te)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, op, q, A, B, S, X = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.variant:
        res = evaluate.cross_validate(train, make(a.variant, y, op, train, q, A, B, S, X))
        pickle.dump(res, open(OUT / f"cv_{a.variant}.pkl", "wb")); print("saved", a.variant); return
    rep = Report("eval_o13", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    res = {k: pickle.load(open(OUT / f"cv_{k}.pkl", "rb")) for k in NAMES}
    ref = res["b1a"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    share1 = np.array([len(p.a & p.b) == 1 for p in train])[ref.pair_N]
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", ref.no_overlap_N), ("share 1 (audit)", share1)]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k in NAMES:
            if k == "b1a":
                continue
            s = evaluate.score_cv(train, res[k], reference=ref, mask_N=mask)
            rows.append([NAMES[k], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a: RMSE / r [95% CI]"], rows)
    rep.section("2  Full-data metric parameters (start -> fitted)")
    allp = np.arange(len(train)); prow = []
    for k in ("angle", "pfe", "hyper", "white", "margin"):
        f = O13(q, X, A, B, k, LAM).fit(allp, y)
        prow.append([NAMES[k], str(np.round(f.e.numpy(), 3).tolist()), f"{np.corrcoef(f.predict(allp), y)[0, 1]:.3f}"])
    rep.table(["model", "e (see mixdist/o13.py)", "in-sample r"], prow)
    rep.save()


if __name__ == "__main__":
    main()
