"""B1a size-term ablation (user 2026-11-21): b1 = total size (log n_A + log n_B), b2 = size mismatch |log n_A - log n_B|,
each alone and together, without and with the group weights. Based on the factor ablation (user 2026-11-21): B1a open-POM, + factor 1 only (mixture size: log n_A + log n_B and
|log n_A - log n_B| in the readout), + factor 2 only (molecule weight exp(theta . 24 RDKit groups) in the mean),
B1a++ (both). lam 1e-3 (as B1a++). T46 / V50 (fit on all 820 training pairs; paired CI vs B1a) and CV 50 x 5 on the
514 development pairs (same folds as eval_o5_softmax)."""
import sys
import numpy as np
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path[:0] = [ROOT, ROOT + "/evaluation/B1aplus", ROOT + "/evaluation/O1", ROOT + "/training"]
from mixdist import data, evaluate, features, metrics, mixture, pushpull as pp
from mixdist.b1a_plus import B1aPlus, membership, size_features
from eval_b1a_plus import piece_matrix
from final_heldout import paired
from train_o2d_dev import dev_pairs
LAM = 1e-3
op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]; q = mixture.quantile_per_word(op.probs_MxW)
X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
X0 = np.zeros((len(ids), 0))
def models(P, y):
    A, B = membership(P, op.cids_M); S = size_features(P); d = mixture.paper_distance(P, op.cids_M, op.probs_MxW)
    out = {"B1a": lambda tr, te: np.polyval(np.polyfit(d[tr], y[tr], 1), d[te])}
    for gn, Xg in (("", X0), ("groups", X)):
        for sn, cols in (("", []), ("b1", [0]), ("b2", [1]), ("b1 + b2", [0, 1])):
            if not gn and not sn:
                continue
            nm = "B1a + " + " + ".join(t for t in (gn, sn) if t)
            out[nm] = (lambda Xg, cols: lambda tr, te: B1aPlus(q, Xg, A, B, S[:, cols] if cols else None, LAM).fit(tr, y[tr]).predict(te))(Xg, cols)
    return out


train = data.load_train()
for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
    allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp)); y_all = np.r_[evaluate.labels(train), hs.y_P]
    preds = {nm: f(tr, te) for nm, f in models(allp, y_all).items()}; y = hs.y_P; ref = preds["B1a"]
    print(f"== {hs.name} ({len(y)} pairs): RMSE / r; change from B1a: RMSE / r [CI]")
    for nm, p in preds.items():
        c = "" if p is ref else (lambda cr: f"{metrics.rmse(p, y) - metrics.rmse(ref, y):+.4f} / {metrics.pearson(p, y) - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]")(paired(p, ref, y)[0])
        print(f"   {nm:26s} {metrics.rmse(p, y):.4f} / {metrics.pearson(p, y):.3f}   {c}", flush=True)
P = dev_pairs(); y = evaluate.labels(P); k = np.array([len(set(p.a) & set(p.b)) for p in P])
SUB = {"all": None, "setting 1 (share none)": k == 0, "setting 2 (share >= 1)": k >= 1}
res = {nm: evaluate.cross_validate(P, f) for nm, f in models(P, y).items()}
print("== CV 50 x 5, 514 development pairs: RMSE / r; change from B1a r [CI]")
for sub, m0 in SUB.items():
    for nm, r_ in res.items():
        m = np.isfinite(r_.pred_N) if m0 is None else (m0[r_.pair_N] & np.isfinite(r_.pred_N))
        ref = None if nm == "B1a" else res["B1a"]; s = evaluate.score_cv(P, r_, reference=ref, mask_N=m)
        c = "" if ref is None else f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"
        print(f"   {sub:24s} {nm:26s} {s['rmse']:.4f} / {s['r']:.3f}  {c}", flush=True)
