# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = pairs (train + held-out)
# ---------------------------------------------------------------------------------------------
"""Final held-out test (user go-ahead 2026-11-18): every model fitted once on all 820 training pairs, scored on T46 and V50.

Both sets contain only pairs sharing no molecule (setting 1), so they test whether a model understands the smell of a
mixture; shared-molecule terms (B1b) cannot act (B1b = B1a++ there, shown as a check).
Rows: B1a open-POM (open-POM baseline) | B1a MIST (MIST baseline) | B1a++ (groups + size) | O1.2b (gain + length) | B1b.
Metrics: RMSE, Pearson r with 95% bootstrap CI over pairs (metrics.score), paired bootstrap change from B1a open-POM (RMSE and
r), bias = mean prediction - mean label (diagnostic only). T46 first. Published noise ceiling for V50 (test-retest r).
Check: B1a open-POM and B1a MIST must reproduce results/check_baselines/2026-11-16_000651 (T46 0.117 / 0.434, 0.111 / 0.410;
V50 0.108 / 0.555, 0.115 / 0.549).

Env:     mist
Usage:   python evaluation/O1/final_heldout.py
Writes:  results/final_heldout/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.b1b import B1b  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3


def paired(pred, ref, y, n_boot=10_000, seed=0):
    idx = metrics.bootstrap_indices(len(y), n_boot, seed)
    dr = metrics.bootstrap_r(pred, y, idx) - metrics.bootstrap_r(ref, y, idx)
    de = np.sqrt(((pred[idx] - y[idx]) ** 2).mean(1)) - np.sqrt(((ref[idx] - y[idx]) ** 2).mean(1))
    return np.nanpercentile(dr, [2.5, 97.5]), np.percentile(de, [2.5, 97.5])


def main():
    rep = Report("final_heldout", {"lam": LAM})
    train = data.load_train(); op = features.load("openpom"); mi = features.load("mist")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
        y_all = np.r_[evaluate.labels(train), hs.y_P]
        A, B = membership(allp, op.cids_M); S = size_features(allp)
        lin = lambda d: evaluate.linear_calibration(d, y_all, np.zeros(len(allp)))(tr, te)
        preds = {
            "B1a open-POM (open-POM baseline)": lin(mixture.paper_distance(allp, op.cids_M, op.probs_MxW)),
            "B1a MIST (MIST baseline)": lin(mixture.paper_distance(allp, mi.cids_M, mi.probs_MxW)),
            "B1a++ (groups + size)": B1aPlus(q, X, A, B, S, LAM).fit(tr, y_all[tr]).predict(te),
            "O1.2b (gain + length)": O12b(q, X, A, B, True, True, lam=LAM).fit(tr, y_all[tr]).predict(te),
            "B1b (= B1a++ on these pairs)": B1b(q, X, A, B, S, LAM).fit(tr, y_all[tr]).predict(te),
        }
        y = hs.y_P; ref = preds["B1a open-POM (open-POM baseline)"]
        rep.section(f"{hs.name}: {len(y)} pairs, all sharing no molecule; mean label {y.mean():.3f}"
                    + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
        rows = []
        for nm, p in preds.items():
            s = metrics.score(p, y, ceiling=hs.ceiling)
            ch = ""
            if p is not ref:
                cr, ce = paired(p, ref, y)
                ch = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
            rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", ch])
        rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from B1a open-POM: RMSE [CI] / r [CI]"], rows)
        rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
