"""Transfer test: does the class dominance measured directly in Ma et al. 2021 improve DREAM setting 1?

Ma 2021 (two-molecule mixtures) measures which component wins; the class weights theta_Ma were fitted
there by the other session (no DREAM rating used) and read from a JSON file given on the command line.
Here theta_Ma is FIXED; only its overall scale c (plus B1a's line and size) is fitted on DREAM folds:
    w_i = exp(c * sum_k theta_Ma,k * x_ik)     x_ik = 1 if molecule i carries class k (acid: 0, not in Ma)
c > 0 keeps Ma's direction; c = 0 is B1a + size.  lam = 0 (one parameter; no penalty, scale-free).
Fixed before running, same test and rule as eval_b1a_plus_setting1_claim.py, plus c > 0:
  base   B1a + size;  transfer  B1a+ Ma-direction + size
  1. setting 1: change in RMSE and r, 95% CI by resampling mixtures
  2. study-confound check (evaluation only): within-study r averaged
  claim "Ma's class dominance transfers to DREAM setting 1" only if both CIs > 0 and c (all 820 pairs) > 0.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_ma_transfer.py MA_THETA_JSON
Writes:  results/eval_b1a_plus_ma_transfer/<date_time>.{txt,json}
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_b1a_plus_setting1_claim import within_study_r  # noqa: E402

N_BOOT = 2000
MIN_STUDY_PAIRS = 10


def main():
    ma = json.load(open(sys.argv[1]))
    rep = Report("eval_b1a_plus_ma_transfer", {"ma_theta_file": sys.argv[1], "ma_theta": ma, "lam": 0.0,
                                               "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    smiles = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    Xc, nc = piece_matrix(pp.units_smarts(smiles, pp.CLASSES), ids)
    th_ma = np.array([dict(zip(ma["classes"], ma["theta"])).get(n, 0.0) for n in nc])
    x1 = (Xc @ th_ma)[:, None]                                   # Ma log-weight direction per molecule
    rep.table(["class", "DREAM molecules", "theta_Ma"], [[n, int(Xc[:, k].sum()), f"{th_ma[k]:+.3f}"] for k, n in enumerate(nc)])

    def fp(X, lam):
        def fit_predict(tr, te):
            return B1aPlus(q, X, A, B, S, lam).fit(tr, y[tr]).predict(te)
        return fit_predict

    base = evaluate.cross_validate(train, fp(np.zeros((len(ids), 0)), 0.0)); print("  done base", flush=True)
    tr_ = evaluate.cross_validate(train, fp(x1, 0.0)); print("  done transfer", flush=True)
    c_all = float(B1aPlus(q, x1, A, B, S, 0.0).fit(np.arange(len(train)), y).theta[0])
    s1 = base.no_overlap_N
    label = "B1a+ Ma-direction + size"

    rep.section("1. Setting 1 (pairs sharing no molecule): Ma's class direction vs B1a + size, same folds")
    s0 = evaluate.score_cv(train, base, mask_N=s1, n_boot=N_BOOT)
    s = evaluate.score_cv(train, tr_, reference=base, mask_N=s1, n_boot=N_BOOT)
    rep.table(["model", "pairs", "RMSE", "r [95% CI]", "change from B1a + size: RMSE [95% CI] / r [95% CI]"],
              [["B1a + size", s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""],
               [label, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / "
                f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"]])
    rep.line(f"scale c fitted on all 820 pairs: {c_all:+.2f} (> 0: DREAM agrees with Ma's direction); "
             f"log-weight range over DREAM molecules {c_all * x1.min():+.2f} to {c_all * x1.max():+.2f}")
    sa = evaluate.score_cv(train, tr_, reference=base, n_boot=N_BOOT)
    rep.line(f"for context, all 820 pairs: RMSE change {sa['rmse_change']:+.4f}, r change {sa['r_change']:+.3f} "
             f"[{sa['r_change_ci'][0]:+.3f}, {sa['r_change_ci'][1]:+.3f}]")

    rep.section(f"2. Study-confound check (evaluation only): within-study r, averaged (studies with >= {MIN_STUDY_PAIRS} setting-1 pairs)")
    w = evaluate.cv_weights(base)[s1]
    pairN = base.pair_N[s1]; study = base.study_N[s1]
    n_by = {st: len(np.unique(pairN[study == st])) for st in np.unique(study)}
    studies = [st for st, n in n_by.items() if n >= MIN_STUDY_PAIRS]
    keep = np.isin(study, studies)
    w, pairN, study = w[keep], pairN[keep], study[keep]
    pb, pt, t = base.pred_N[s1][keep], tr_.pred_N[s1][keep], base.truth_N[s1][keep]
    mixes = sorted({x for p in train for x in (p.a, p.b)}, key=sorted); idx = {x: i for i, x in enumerate(mixes)}
    ia = np.array([idx[train[i].a] for i in pairN]); ib = np.array([idx[train[i].b] for i in pairN])
    rb0 = within_study_r(pb, t, w[None], study, studies)[0]; rt0 = within_study_r(pt, t, w[None], study, studies)[0]
    rng = np.random.default_rng(0); d = []
    for start in range(0, N_BOOT, 100):
        c = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=100)
        wb = w * c[:, ia] * c[:, ib]
        d.append(within_study_r(pt, t, wb, study, studies) - within_study_r(pb, t, wb, study, studies))
    d = np.concatenate(d); d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    rep.table(["model", "within-study r (averaged)", "change [95% CI]"],
              [["B1a + size", f"{rb0:.3f}", ""], [label, f"{rt0:.3f}", f"{rt0 - rb0:+.3f} [{lo:+.3f}, {hi:+.3f}]"]])
    ok = s["r_change_ci"][0] > 0 and lo > 0 and c_all > 0
    rep.line(f"claim rule: r change CI > 0: {s['r_change_ci'][0] > 0}; within-study CI > 0: {lo > 0}; c > 0: {c_all > 0} -> claim {'MET' if ok else 'NOT met'}")
    rep.value("result", {"c": c_all, "r_change": s["r_change"], "r_change_ci": list(s["r_change_ci"]), "rmse_change": s["rmse_change"],
                         "within_change": rt0 - rb0, "within_ci": [lo, hi], "met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
