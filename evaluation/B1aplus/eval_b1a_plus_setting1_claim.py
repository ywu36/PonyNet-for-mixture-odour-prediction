"""Pre-declared claim test: do sub-molecular dominance weights (theta) improve setting-1 prediction?

Setting 1 = training pairs whose two mixtures share no molecule (325 of 820).
Inputs are only those available for a new test pair (molecules, their open-POM words, mixture sizes);
no study offsets.

Models (fixed before running, both trained on all pairs of each training fold, standard CV
evaluation/CV.md: 50 x 5, whole mixtures held out):
  base   B1a + size                    line on B1a distance + log nA + log nB + |log nA - log nB|
  theta  B1a+ classes + size, lam 1e-3 same + weight exp(sum_k theta_k x_ik) per molecule, 9 SMARTS classes
Scored on setting-1 pairs only:
  1. change in RMSE and r (theta - base), 95% CI by resampling mixtures (2000)
  2. study-confound check, evaluation only: r computed within each study (studies with >= 10 setting-1
     pairs), averaged with weights = pairs per study; change theta - base, same mixture resampling.
     Studies never enter the model and no study is reported on its own.
Claim rule (fixed before running): claim "sub-molecular dominance weights improve setting-1
prediction" only if the CI of the r change in (1) is above 0 AND the change in (2) has CI above 0.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_setting1_claim.py [classes|groups]
         groups = RDKit fr_ groups carried by >= 5 of the 235 molecules (merged), same model and rule
         (added 2026-11-17 after the classes run; its setting-1 point estimate had been seen before)
Writes:  results/eval_b1a_plus_setting1_claim/<date_time>.{txt,json}
"""
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

LAM = 1e-3
N_BOOT = 2000
MIN_STUDY_PAIRS = 10


def within_study_r(p, t, w_BxN, study_N, studies):
    """Pair-count-weighted mean of within-study r -> r_B."""
    tot, acc = 0.0, 0.0
    for s in studies:
        m = study_N == s
        n = w_BxN[:, m].sum(1)
        acc = acc + n * np.nan_to_num(evaluate._weighted_r(p[m], t[m], w_BxN[:, m]))
        tot = tot + n
    return acc / tot


def main():
    units = sys.argv[1] if len(sys.argv) > 1 else "classes"
    rep = Report("eval_b1a_plus_setting1_claim", {"units": units, "lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT,
                                                  "min_study_pairs": MIN_STUDY_PAIRS})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    smiles = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    Xc, nc = piece_matrix(pp.units_smarts(smiles, pp.CLASSES) if units == "classes"
                          else pp.merge_identical(pp.units_rdkit_groups(smiles)), ids)
    label = f"B1a+ {units} + size, lam 1e-3"
    rep.line(f"{units} used ({len(nc)}): {', '.join(nc)}")

    def fp(X, lam):
        def fit_predict(tr, te):
            return B1aPlus(q, X, A, B, S, lam).fit(tr, y[tr]).predict(te)
        return fit_predict

    base = evaluate.cross_validate(train, fp(np.zeros((len(ids), 0)), 0.0)); print("  done base", flush=True)
    th = evaluate.cross_validate(train, fp(Xc, LAM)); print("  done theta", flush=True)
    s1 = base.no_overlap_N

    rep.section("1. Setting 1 (pairs sharing no molecule): theta vs B1a + size, same folds")
    s0 = evaluate.score_cv(train, base, mask_N=s1, n_boot=N_BOOT)
    s = evaluate.score_cv(train, th, reference=base, mask_N=s1, n_boot=N_BOOT)
    rep.table(["model", "pairs", "RMSE", "r [95% CI]", "change from B1a + size: RMSE [95% CI] / r [95% CI]"],
              [["B1a + size", s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""],
               [label, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / "
                f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"]])
    sa = evaluate.score_cv(train, th, reference=base, n_boot=N_BOOT)
    rep.line(f"for context, all 820 pairs: RMSE change {sa['rmse_change']:+.4f} [{sa['rmse_change_ci'][0]:+.4f}, {sa['rmse_change_ci'][1]:+.4f}], "
             f"r change {sa['r_change']:+.3f} [{sa['r_change_ci'][0]:+.3f}, {sa['r_change_ci'][1]:+.3f}]")

    rep.section(f"2. Study-confound check (evaluation only): within-study r, averaged (studies with >= {MIN_STUDY_PAIRS} setting-1 pairs)")
    w = evaluate.cv_weights(base)[s1]
    pairN = base.pair_N[s1]; study = base.study_N[s1]
    n_by = {st: len(np.unique(pairN[study == st])) for st in np.unique(study)}
    studies = [st for st, n in n_by.items() if n >= MIN_STUDY_PAIRS]
    keep = np.isin(study, studies)
    w, pairN, study = w[keep], pairN[keep], study[keep]
    pb, pt, t = base.pred_N[s1][keep], th.pred_N[s1][keep], base.truth_N[s1][keep]
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
    rep.line(f"{len(studies)} studies, {len(np.unique(pairN))} setting-1 pairs (of {s['pairs']}); studies with fewer pairs left out: {len(n_by) - len(studies)}")
    rep.table(["model", "within-study r (averaged)", "change [95% CI]"],
              [["B1a + size", f"{rb0:.3f}", ""], [label, f"{rt0:.3f}", f"{rt0 - rb0:+.3f} [{lo:+.3f}, {hi:+.3f}]"]])

    ok1 = s["r_change_ci"][0] > 0; ok2 = lo > 0
    rep.section("3. Claim rule (fixed before running)")
    rep.line(f"r change CI above 0: {ok1}; within-study change CI above 0: {ok2} -> claim {'MET' if ok1 and ok2 else 'NOT met'}")
    rep.value("claim", {"r_change": s["r_change"], "r_change_ci": list(s["r_change_ci"]), "rmse_change": s["rmse_change"],
                        "rmse_change_ci": list(s["rmse_change_ci"]), "within_change": rt0 - rb0, "within_ci": [lo, hi], "met": bool(ok1 and ok2)})
    rep.save()


if __name__ == "__main__":
    main()
