# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = training pairs (820)
# ---------------------------------------------------------------------------------------------
"""B1a ensemble: mean of the B1a distances from open-POM and from MIST, then one line (2 numbers).

Both encoders frozen; each distance is B1a's recipe (per-word quantile, mean per mixture, length 1,
Euclidean). Scored by zero-overlap CV on the training pairs against B1a open-POM on the same folds
(paired bootstrap); B1a MIST shown as a second reference row. No V50/T46 (scored only if it passes
finding 3's pre-set criterion).

Env:     mist
Usage:   python evaluation/B1/eval_b1a_openpom_mist_mean.py [--repeats 10]
Writes:  results/eval_b1a_openpom_mist_mean/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import data, evaluate, features, metrics, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    rep = Report("eval_b1a_openpom_mist_mean", vars(args))
    train = data.load_train()
    y_P = evaluate.labels(train)
    d = {}
    for m in ("openpom", "mist"):
        fs = features.load(m)
        d[m] = mixture.paper_distance(train, fs.cids_M, fs.probs_MxW)
    d["mean"] = (d["openpom"] + d["mist"]) / 2
    rep.line(f"Pearson between the two B1a distances on the training pairs: {metrics.pearson(d['openpom'], d['mist']):.3f}")
    res = {k: evaluate.cross_validate(train, evaluate.linear_calibration(v, y_P, np.zeros(len(train))), k=5, repeats=args.repeats)
           for k, v in d.items()}
    ref = res["openpom"]
    rows = []
    for k in ("mist", "mean"):
        for name, m in [("all pairs", np.ones(len(ref.pred_N), bool)), ("share no molecule", ref.no_overlap_N)]:
            s = metrics.score(res[k].pred_N[m], ref.truth_N[m], reference=ref.pred_N[m], n_boot=2000)
            rows.append([f"B1a {k}", name, f"{metrics.rmse(ref.pred_N[m], ref.truth_N[m]):.3f} / {s['r_ref']:.3f}",
                         f"{s['rmse'] - metrics.rmse(ref.pred_N[m], ref.truth_N[m]):+.3f}", f"{s['r'] - s['r_ref']:+.3f}",
                         f"[{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "B1a open-POM RMSE / r", "change RMSE", "change r", "95% CI of r change"], rows)
    rep.save()


if __name__ == "__main__":
    main()
