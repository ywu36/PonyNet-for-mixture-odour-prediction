# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = pairs sharing exactly 1 molecule (126)   M = molecules (235)
#   G = RDKit groups (24)   W = open-POM words (138)   F = candidate features   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""Setting 2, pairs sharing exactly 1 molecule: what explains the current model's error, at group level?

Current model = B1a + size + groups (B1a+ RDKit groups + size, lam 1e-3), standard CV on all 820 pairs;
its held-out prediction per pair (mean over repeats) gives the error e = y - prediction.
On the pairs sharing exactly one molecule s, candidates fixed before running (inputs only):
  structure (groups)
    S1   relative dominance of s:  d_s - mean d over the other molecules of A and B,
         d = theta . x (group log-weight), theta fitted on SETTING 1 ONLY (never saw these pairs)
    S1b  absolute dominance d_s
    S2   group overlap of the unshared parts: Jaccard of the group sets of A minus s and B minus s
  semantic (open-POM quantile profiles q)
    P1   smell stand-out of s: 1 - mean over A, B of cos(q_s, mean q of the other molecules)
    P2   smell similarity of the unshared parts: cos(mean q of A minus s, mean q of B minus s)
  control
    C    share of s: (1/nA + 1/nB) / 2
1. Per candidate: slope of e on the candidate (per sd), 95% CI and p by resampling mixtures (2000),
   BH q over the 6. Negative slope = pairs rated closer than the model predicts.
2. CV check per candidate: current prediction + a line in the candidate fitted on the training folds of
   these pairs (5 x 50, whole mixtures held out); change in RMSE and r from the current model.

Env:     mist
Usage:   python evaluation/B1aplus/eval_setting2_share1.py
Writes:  results/eval_setting2_share1/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
sys.path.insert(0, str(ROOT / "findings"))
from pair_signal_by_sharing import bh  # noqa: E402
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3
N_BOOT = 2000


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    rep = Report("eval_setting2_share1", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    row = {int(c): i for i, c in enumerate(op.cids_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    X_MxG, names = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    cur = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, X_MxG, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    w = evaluate.cv_weights(cur)
    pred_P = np.bincount(cur.pair_N, weights=w * cur.pred_N, minlength=len(train)) / np.bincount(cur.pair_N, weights=w, minlength=len(train))
    s1_idx = np.array([i for i, p in enumerate(train) if p.jaccard == 0])
    theta1 = B1aPlus(q, X_MxG, A, B, S, LAM).fit(s1_idx, y[s1_idx]).theta.numpy()
    d_M = X_MxG @ theta1
    print("  done current model", flush=True)

    idx1 = np.array([i for i, p in enumerate(train) if len(p.a & p.b) == 1])
    pairs1 = [train[i] for i in idx1]
    e_P = y[idx1] - pred_P[idx1]
    feats = {k: [] for k in ["S1 relative dominance of shared", "S1b dominance of shared", "S2 group overlap of unshared parts",
                             "P1 smell stand-out of shared", "P2 smell similarity of unshared parts", "C share of shared"]}
    for p in pairs1:
        s = next(iter(p.a & p.b)); rs = row[s]
        ra = [row[c] for c in p.a if c != s]; rb = [row[c] for c in p.b if c != s]
        feats["S1 relative dominance of shared"].append(d_M[rs] - d_M[ra + rb].mean())
        feats["S1b dominance of shared"].append(d_M[rs])
        ga, gb = X_MxG[ra].max(0) > 0, X_MxG[rb].max(0) > 0
        feats["S2 group overlap of unshared parts"].append((ga & gb).sum() / max((ga | gb).sum(), 1))
        feats["P1 smell stand-out of shared"].append(1 - 0.5 * (cos(q[rs], q[ra].mean(0)) + cos(q[rs], q[rb].mean(0))))
        feats["P2 smell similarity of unshared parts"].append(cos(q[ra].mean(0), q[rb].mean(0)))
        feats["C share of shared"].append(0.5 * (1 / len(p.a) + 1 / len(p.b)))
    F_PxF = np.array([feats[k] for k in feats]).T
    Z_PxF = (F_PxF - F_PxF.mean(0)) / F_PxF.std(0)

    rep.section(f"0. {len(pairs1)} pairs share exactly 1 molecule; {len({next(iter(p.a & p.b)) for p in pairs1})} distinct shared molecules")
    s_cur = evaluate.score_cv(train, cur, mask_N=np.isin(cur.pair_N, idx1), n_boot=N_BOOT)
    rep.line(f"current model on these pairs: RMSE {s_cur['rmse']:.4f}, r {s_cur['r']:.3f} [{s_cur['r_ci'][0]:.3f}, {s_cur['r_ci'][1]:.3f}]; "
             f"mean error (rating - prediction) {e_P.mean():+.3f}, sd {e_P.std():.3f}")
    C = np.corrcoef(F_PxF.T)
    rep.table(["candidate"] + [k.split()[0] for k in feats],
              [[k] + [f"{C[i, j]:+.2f}" for j in range(len(feats))] for i, k in enumerate(feats)])
    rep.line("correlations between candidates over the pairs")

    mixes = sorted({m for p in pairs1 for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ia = np.array([im[p.a] for p in pairs1]); ib = np.array([im[p.b] for p in pairs1])
    c = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    w_BxP = (c[:, ia] * c[:, ib]).astype(float)

    def slope(z, e, wt):
        X = np.c_[np.ones(len(z)), z]; Xw = X * wt[:, None]
        try:
            return np.linalg.solve(X.T @ Xw, Xw.T @ e)[1]
        except np.linalg.LinAlgError:
            return np.nan
    res = []
    for f, k in enumerate(feats):
        b = slope(Z_PxF[:, f], e_P, np.ones(len(e_P)))
        bb = np.array([slope(Z_PxF[:, f], e_P, wb) for wb in w_BxP]); bb = bb[np.isfinite(bb)]
        p = max(min(1, 2 * min((bb <= 0).mean(), (bb >= 0).mean())), 1 / N_BOOT)
        res.append((k, b, *np.percentile(bb, [2.5, 97.5]), p))
    qv = bh([r[4] for r in res])

    # CV check: current prediction + line in the candidate, fitted on training folds of these pairs
    yy = y[idx1]; base_off = pred_P[idx1]
    ref = evaluate.cross_validate(pairs1, lambda tr, te: base_off[te])
    rows = []
    for f, (k, b, lo, hi, p) in enumerate(res):
        def fp(tr, te, f=f):
            X = np.c_[np.ones(len(tr)), Z_PxF[tr, f]]
            coef = np.linalg.lstsq(X, yy[tr] - base_off[tr], rcond=None)[0]
            return base_off[te] + coef[0] + coef[1] * Z_PxF[te, f]
        r_ = evaluate.cross_validate(pairs1, fp)
        s = evaluate.score_cv(pairs1, r_, reference=ref, n_boot=N_BOOT)
        rows.append([k, f"{b:+.4f} [{lo:+.4f}, {hi:+.4f}]", f"{p:.3f}", f"{qv[f]:.2f}",
                     f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    s_ref = evaluate.score_cv(pairs1, ref, n_boot=N_BOOT)
    rep.section("1. What explains the current model's error on pairs sharing exactly 1 molecule")
    rep.table(["candidate", "slope of error per sd [95% CI]", "p", "q", "CV: change from current model, RMSE / r [95% CI]"], rows)
    rep.line(f"current model inside this CV: RMSE {s_ref['rmse']:.4f}, r {s_ref['r']:.3f}. Slope < 0: higher candidate -> rated closer than predicted. "
             f"Label sd here {yy.std():.3f}.")
    rep.value("result", [{"candidate": r[0], "slope": r[1], "lo": r[2], "hi": r[3], "p": r[4], "q": float(qv[i])} for i, r in enumerate(res)])
    rep.value("theta_setting1", {"groups": names, "theta": theta1.tolist()})
    rep.save()


if __name__ == "__main__":
    main()
