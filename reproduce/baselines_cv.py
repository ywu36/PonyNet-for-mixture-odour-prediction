"""open-POM and MIST baselines (B1a recipe: quantile per word, mean, length 1, Euclidean, one line) on the same CV folds/subsets as
eval_o5_softmax (2026-11-20_171657), compared with O1.2b and (b) from the saved CV predictions"""
import sys, pickle, glob
import numpy as np
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path[:0] = [ROOT, ROOT + "/evaluation/O3", ROOT + "/training"]
from mixdist import data, evaluate, features, metrics, mixture
from train_o2d_dev import dev_pairs
import eval_o5_softmax as eo
op = features.load("openpom")
def b1a(pairs, tr, te, y):
    d = mixture.paper_distance(pairs, op.cids_M, op.probs_MxW); c = np.polyfit(d[tr], y[tr], 1); return np.polyval(c, d[te])
P = dev_pairs(); y = evaluate.labels(P)
res = {}
for f in sorted(glob.glob(ROOT + "/results/eval_o5_softmax/*_cv_rows*.pkl")):
    res.update(pickle.load(open(f, "rb"))["res"])
res = {"O1.2b": res[eo.REF], "(b)": res[eo.OB]}
res["open-POM baseline"] = evaluate.cross_validate(P, lambda tr, te: b1a(P, tr, te, y))
mi = features.load("mist"); assert list(mi.cids_M) == list(op.cids_M)
res["MIST baseline"] = evaluate.cross_validate(P, lambda tr, te: (lambda d: np.polyval(np.polyfit(d[tr], y[tr], 1), d[te]))(mixture.paper_distance(P, mi.cids_M, mi.probs_MxW)))
k = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
SUB = {"all": None, "setting 1": k == 0, "same-size s1": (k == 0) & (gap <= 1), "setting 2": k >= 1, "share 1-2": (k >= 1) & (k <= 2), "share >=3": k >= 3}
print("CV (50 x 5, 514 dev pairs): RMSE / r; change from open-POM baseline r [CI]")
for sub, m0 in SUB.items():
    for nm, r_ in res.items():
        m = np.isfinite(r_.pred_N) if m0 is None else (m0[r_.pair_N] & np.isfinite(r_.pred_N))
        ref = None if nm == "open-POM baseline" else res["open-POM baseline"]
        s = evaluate.score_cv(P, r_, reference=ref, mask_N=m)
        c = "" if ref is None else f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"
        print(f"{sub:14s} {nm:20s} {s['rmse']:.4f} / {s['r']:.3f}  {c}", flush=True)
