"""Smell-level dominance: does a molecule's smell (open-POM words) decide how much it counts in a mixture?

Hypothesis: in a mixture, molecules with certain smells dominate (or are masked) whatever their
chemistry. Same model as B1a+ (mixdist/b1a_plus.py), but the dominance input is the molecule's smell:
    w_i = exp(sum_w theta_w * q_iw),  q_iw = B1a's per-word quantile score (standard-normal scale),
    138 open-POM words; theta_w > 0: molecules strong in word w dominate their mixture.
Inputs available for a new test pair only (molecules, open-POM words, mixture sizes); no study offsets.

Fixed before running (same test and claim rule as eval_b1a_plus_setting1_claim.py):
  base   B1a + size
  words  B1a+ words + size, lam 1e-3
  1. setting 1 (pairs sharing no molecule): change in RMSE and r, 95% CI by resampling mixtures
  2. study-confound check, evaluation only: within-study r averaged (studies with >= 10 setting-1 pairs)
  claim "smell-level dominance improves setting-1 prediction" only if both CIs are above 0.
  3. back to sub-molecules (reported whatever 1-2 show, as description): fit on all 820 pairs; per
     chemical class, mean log-weight (theta . q_i) of molecules carrying it minus those not carrying it,
     95% CI by resampling mixtures (100 refits); top / bottom 10 words by theta.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_words.py
Writes:  results/eval_b1a_plus_words/<date_time>.{txt,json}
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
from eval_b1a_plus_setting1_claim import within_study_r  # noqa: E402

LAM = 1e-3
N_BOOT = 2000
N_BOOT_ATTR = 100
MIN_STUDY_PAIRS = 10


def main():
    rep = Report("eval_b1a_plus_words", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT,
                                         "n_boot_attr": N_BOOT_ATTR, "min_study_pairs": MIN_STUDY_PAIRS})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    words = [str(w) for w in op.words_W]
    ids = [str(int(c)) for c in op.cids_M]
    smiles = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    Xc, nc = piece_matrix(pp.units_smarts(smiles, pp.CLASSES), ids)
    rep.line(f"dominance input: {q.shape[1]} open-POM word scores per molecule (mean {q.mean():.2f}, sd {q.std():.2f})")

    def fp(X, lam):
        def fit_predict(tr, te):
            return B1aPlus(q, X, A, B, S, lam).fit(tr, y[tr]).predict(te)
        return fit_predict

    base = evaluate.cross_validate(train, fp(np.zeros((len(ids), 0)), 0.0)); print("  done base", flush=True)
    wd = evaluate.cross_validate(train, fp(q, LAM)); print("  done words", flush=True)
    s1 = base.no_overlap_N

    rep.section("1. Setting 1 (pairs sharing no molecule): smell-level dominance vs B1a + size, same folds")
    s0 = evaluate.score_cv(train, base, mask_N=s1, n_boot=N_BOOT)
    s = evaluate.score_cv(train, wd, reference=base, mask_N=s1, n_boot=N_BOOT)
    rep.table(["model", "pairs", "RMSE", "r [95% CI]", "change from B1a + size: RMSE [95% CI] / r [95% CI]"],
              [["B1a + size", s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""],
               ["B1a+ words + size, lam 1e-3", s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / "
                f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"]])
    sa = evaluate.score_cv(train, wd, reference=base, n_boot=N_BOOT)
    rep.line(f"for context, all 820 pairs: RMSE change {sa['rmse_change']:+.4f} [{sa['rmse_change_ci'][0]:+.4f}, {sa['rmse_change_ci'][1]:+.4f}], "
             f"r change {sa['r_change']:+.3f} [{sa['r_change_ci'][0]:+.3f}, {sa['r_change_ci'][1]:+.3f}]")

    rep.section(f"2. Study-confound check (evaluation only): within-study r, averaged (studies with >= {MIN_STUDY_PAIRS} setting-1 pairs)")
    w = evaluate.cv_weights(base)[s1]
    pairN = base.pair_N[s1]; study = base.study_N[s1]
    n_by = {st: len(np.unique(pairN[study == st])) for st in np.unique(study)}
    studies = [st for st, n in n_by.items() if n >= MIN_STUDY_PAIRS]
    keep = np.isin(study, studies)
    w, pairN, study = w[keep], pairN[keep], study[keep]
    pb, pt, t = base.pred_N[s1][keep], wd.pred_N[s1][keep], base.truth_N[s1][keep]
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
    rep.line(f"{len(studies)} studies, {len(np.unique(pairN))} setting-1 pairs (of {s['pairs']})")
    rep.table(["model", "within-study r (averaged)", "change [95% CI]"],
              [["B1a + size", f"{rb0:.3f}", ""], ["B1a+ words + size, lam 1e-3", f"{rt0:.3f}", f"{rt0 - rb0:+.3f} [{lo:+.3f}, {hi:+.3f}]"]])
    ok1 = s["r_change_ci"][0] > 0; ok2 = lo > 0
    rep.line(f"claim rule: r change CI above 0: {ok1}; within-study CI above 0: {ok2} -> claim {'MET' if ok1 and ok2 else 'NOT met'}")

    rep.section("3. Back to sub-molecules (description): fit on all 820 pairs")
    allp = np.arange(len(train))
    mixes_all = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes_all)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes_all), np.full(len(mixes_all), 1 / len(mixes_all)), size=N_BOOT_ATTR)
    th = B1aPlus(q, q, A, B, S, LAM).fit(allp, y).theta.numpy()
    th_b = np.array([B1aPlus(q, q, A, B, S, LAM).fit(allp, y, w_pairs=(cnt[b, ja] * cnt[b, jb]).astype(float) + 1e-12).theta.numpy()
                     for b in range(N_BOOT_ATTR)])
    in_train = np.zeros(len(ids), bool); in_train[np.where(np.stack([A, B]).max((0, 1)) > 0)[0]] = True

    def class_gap(theta):
        lw = q @ theta
        return np.array([lw[in_train & (Xc[:, k] > 0)].mean() - lw[in_train & (Xc[:, k] == 0)].mean() for k in range(len(nc))])
    g = class_gap(th); gb = np.array([class_gap(t_) for t_ in th_b]); glo, ghi = np.percentile(gb, [2.5, 97.5], 0)
    lw = q @ th
    rep.line(f"log-weight theta . q_i over the training molecules: sd {lw[in_train].std():.2f} (weight factor exp: 1 sd = x{np.exp(lw[in_train].std()):.2f})")
    rep.table(["class", "training molecules", "mean log-weight: carriers - others [95% CI]", "weight factor"],
              [[n, int((in_train & (Xc[:, k] > 0)).sum()), f"{g[k]:+.2f} [{glo[k]:+.2f}, {ghi[k]:+.2f}]", f"x{np.exp(g[k]):.2f}"] for k, n in enumerate(nc)])
    rep.line("> 0: molecules of this class smell like the dominant words (count more in their mixture); < 0: smell like masked words.")
    tlo, thi = np.percentile(th_b, [2.5, 97.5], 0)
    o = np.argsort(-th)
    rep.table(["word", "theta [95% CI]"], [[words[k], f"{th[k]:+.3f} [{tlo[k]:+.3f}, {thi[k]:+.3f}]"] for k in list(o[:10]) + list(o[-10:])])
    rep.line("top 10 (dominant) then bottom 10 (masked) words by theta.")
    rep.value("result", {"r_change": s["r_change"], "r_change_ci": list(s["r_change_ci"]), "rmse_change": s["rmse_change"],
                         "rmse_change_ci": list(s["rmse_change_ci"]), "within_change": rt0 - rb0, "within_ci": [lo, hi],
                         "met": bool(ok1 and ok2), "classes": nc, "class_gap": g.tolist(), "class_gap_ci": [glo.tolist(), ghi.tolist()],
                         "words": words, "theta": th.tolist()})
    rep.save()


if __name__ == "__main__":
    main()
