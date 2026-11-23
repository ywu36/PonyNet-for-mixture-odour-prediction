"""B1a under the CV protocol of 2026-11-17 (evaluation/CV.md): 50 x 5 folds, every pair counted
once, CI by resampling mixtures. Shown next to the earlier protocol (10 x 5, every prediction
counted, bootstrap over predictions) so the difference is visible. Training pairs only.

Env:     mist
Usage:   python evaluation/eval_b1a_cv.py
Writes:  results/eval_b1a_cv/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data, evaluate, features, metrics, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    rep = Report("eval_b1a_cv", {"repeats": evaluate.CV_REPEATS})
    train = data.load_train()
    y_P = evaluate.labels(train)
    fs = features.load("openpom")
    fp = evaluate.linear_calibration(mixture.paper_distance(train, fs.cids_M, fs.probs_MxW), y_P, np.zeros(len(train)))

    rows = []
    old = evaluate.cross_validate(train, fp, k=5, repeats=10)
    for name, m in [("all pairs", np.ones(len(old.pred_N), bool)), ("share no molecule", old.no_overlap_N)]:
        s = metrics.score(old.pred_N[m], old.truth_N[m], n_boot=2000)
        rows.append(["earlier: 10 x 5, per prediction", name, s["n"], f"{s['rmse']:.3f}", f"{s['r']:.3f}", f"[{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]"])
    new = evaluate.cross_validate(train, fp, k=5, repeats=evaluate.CV_REPEATS)
    for name, m in [("all pairs", None), ("share no molecule", new.no_overlap_N)]:
        s = evaluate.score_cv(train, new, mask_N=m)
        rows.append([f"new: {evaluate.CV_REPEATS} x 5, per pair, mixture CI", name, s["pairs"], f"{s['rmse']:.3f}", f"{s['r']:.3f}", f"[{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]"])
    rep.table(["protocol", "subset", "n (predictions / pairs)", "RMSE", "r", "95% CI of r"], rows)
    rep.save()


if __name__ == "__main__":
    main()
