# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = pairs (820)   N = pooled out-of-fold predictions   Z = size columns
# ---------------------------------------------------------------------------------------------
"""Audit: how well does content predict once mixture size is taken out? (step 1 of the share-1 question)

Question: is share-1's low r (B1a++ 0.07) special, or is it just what content does without the size help it gets on
pairs sharing no molecule? Declared before running:
  Content models (no size covariate): B1a, O1.2 gain only (theta). Reference: B1a++ (has size covariates).
  Size Z = [log nA + log nB, |log nA - log nB|] (B1a++'s size terms).
  Per subset: (a) raw r; (b) partial r = r of out-of-fold prediction and target after both are regressed on [1, Z]
  within the subset (weights: each pair once), CI by resampling mixtures (regression held fixed: approximate);
  (c) RMSE of a line on Z alone vs a line on Z + prediction, within the subset (in-sample on out-of-fold
  predictions), and size-alone r.
  Reading: partial r on share 0 ~ 0.07-0.10 -> share 1 is not special (content is weak everywhere);
  clearly higher -> share 1 is special.
Subsets: share no molecule; share 1 and share 2 (audit, defined by the input). Same folds as eval_o12c.

Env:     mist
Usage:   python evaluation/O1/audit_content_no_size.py  (needs results/eval_o12c/cv_{theta,b1a2}.pkl, eval_o13/cv_b1a.pkl)
Writes:  results/audit_content_no_size/<date_time>.{txt,json}
"""
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, evaluate  # noqa: E402
from mixdist.b1a_plus import size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402


def wls_resid(Z, v, w):
    sw = np.sqrt(w); coef = np.linalg.lstsq(Z * sw[:, None], v * sw, rcond=None)[0]
    return v - Z @ coef


def main():
    rep = Report("audit_content_no_size", {"repeats": evaluate.CV_REPEATS})
    train = data.load_train(); S_PxZ = size_features(train)
    res = {"B1a (content only)": pickle.load(open(ROOT / "results/eval_o13/cv_b1a.pkl", "rb")),
           "O1.2 gain only (content only)": pickle.load(open(ROOT / "results/eval_o12c/cv_theta.pkl", "rb"))["res"],
           "B1a++ (reference, has size terms)": pickle.load(open(ROOT / "results/eval_o12c/cv_b1a2.pkl", "rb"))["res"]}
    ref = next(iter(res.values()))
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nsh_N = np.array([len(p.a & p.b) for p in train])[ref.pair_N]
    w_N = evaluate.cv_weights(ref)
    rows, srow = [], []
    for subset, m in [("share no molecule", nsh_N == 0), ("share 1 (audit)", nsh_N == 1), ("share 2 (audit)", nsh_N == 2)]:
        Z = np.c_[np.ones(m.sum()), S_PxZ[ref.pair_N[m]]]; w = w_N[m]; t = ref.truth_N[m]
        t_res = wls_resid(Z, t, w)
        rmse_size = np.sqrt(np.average(t_res ** 2, weights=w))
        r_size = evaluate._weighted_r(t - t_res, t, w)          # each pair once
        srow.append([subset, int(len(np.unique(ref.pair_N[m]))), f"{np.sqrt(np.cov(S_PxZ[ref.pair_N[m], 0], aweights=w)):.2f}",
                     f"{r_size:.3f}", f"{rmse_size:.4f}"])
        for name, r in res.items():
            p = r.pred_N[m]
            raw = evaluate.score_cv(train, r, mask_N=m)
            fake = replace(r, pred_N=r.pred_N.copy(), truth_N=r.truth_N.copy())
            fake.pred_N[m] = wls_resid(Z, p, w); fake.truth_N[m] = t_res
            pr = evaluate.score_cv(train, fake, mask_N=m)
            Zp = np.c_[Z, p]; e = wls_resid(Zp, t, w); rmse_both = np.sqrt(np.average(e ** 2, weights=w))
            rows.append([name, subset, f"{raw['r']:.3f}", f"{pr['r']:.3f} [{pr['r_ci'][0]:.3f}, {pr['r_ci'][1]:.3f}]",
                         f"{rmse_size:.4f} -> {rmse_both:.4f} ({rmse_both - rmse_size:+.4f})"])
    rep.section("1  Size alone (line on Z, in-sample)")
    rep.table(["subset", "pairs", "sd(log nA + log nB)", "size-alone r", "size-alone RMSE"], srow)
    rep.section("2  Content once size is taken out")
    rep.table(["model", "subset", "raw r", "partial r (size removed) [95% CI]", "RMSE: size line -> size + prediction line"], rows)
    rep.save()


if __name__ == "__main__":
    main()
