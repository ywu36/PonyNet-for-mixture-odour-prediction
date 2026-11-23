"""Setting 2, step 1: audit, then size and group dominance (from setting 1) applied to setting 2.

Setting 2 = training pairs whose two mixtures share >= 1 molecule (495 of 820).
0. Audit (inputs only, no model): pairs, studies, mixture sizes, shared molecules per pair, overlap
   (Jaccard), how often a molecule is shared, which molecules are shared most.
1. Standard CV (evaluation/CV.md), models trained on all pairs, scored on setting-2 pairs:
     B1a (open-POM baseline, the reference)
     B1a + size                      (as in setting 1)
     B1a+ groups + size, lam 1e-3    (as in setting 1)
   change in RMSE and r with 95% CI by resampling mixtures; within-study r check (evaluation only)
   for each step against the one before. Also by number of shared molecules (1, 2, >= 3): these are
   input-defined subsets, not studies.
Inputs available for a new test pair only; no study offsets.

Env:     mist
Usage:   python evaluation/B1aplus/eval_setting2_step1.py
Writes:  results/eval_setting2_step1/<date_time>.{txt,json}
"""
import sys
from collections import Counter
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
MIN_STUDY_PAIRS = 10


def within_change(train, ref, res, mask, n_boot=N_BOOT):
    w = evaluate.cv_weights(ref)[mask]; pairN = ref.pair_N[mask]; study = ref.study_N[mask]
    n_by = {st: len(np.unique(pairN[study == st])) for st in np.unique(study)}
    studies = [st for st, n in n_by.items() if n >= MIN_STUDY_PAIRS]
    k = np.isin(study, studies)
    w, pairN, study = w[k], pairN[k], study[k]
    pb, pt, t = ref.pred_N[mask][k], res.pred_N[mask][k], ref.truth_N[mask][k]
    mixes = sorted({x for p in train for x in (p.a, p.b)}, key=sorted); idx = {x: i for i, x in enumerate(mixes)}
    ia = np.array([idx[train[i].a] for i in pairN]); ib = np.array([idx[train[i].b] for i in pairN])
    d0 = within_study_r(pt, t, w[None], study, studies)[0] - within_study_r(pb, t, w[None], study, studies)[0]
    rng = np.random.default_rng(0); d = []
    for _ in range(n_boot // 100):
        c = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=100)
        wb = w * c[:, ia] * c[:, ib]
        d.append(within_study_r(pt, t, wb, study, studies) - within_study_r(pb, t, wb, study, studies))
    d = np.concatenate(d); d = d[np.isfinite(d)]
    return d0, np.percentile(d, [2.5, 97.5]), len(studies)


def main():
    rep = Report("eval_setting2_step1", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    X_MxG, names = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    s2 = [p for p in train if p.jaccard > 0]
    rep.section(f"0. Audit: setting 2 = {len(s2)} of {len(train)} pairs share >= 1 molecule")
    nsh = np.array([len(p.a & p.b) for p in s2])
    na = np.array([len(p.a) for p in s2]); nb = np.array([len(p.b) for p in s2])
    jac = np.array([p.jaccard for p in s2])
    rep.line("studies (pairs): " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(p.dataset for p in s2).items())) + "  [audit only; never reported alone]")
    rep.line(f"shared molecules per pair: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(np.minimum(nsh, 10)).items())) + " (10 = >= 10)")
    rep.line(f"mixture size: median {int(np.median(np.r_[na, nb]))}, range {np.r_[na, nb].min()}-{np.r_[na, nb].max()}; "
             f"same size in both mixtures: {np.mean(na == nb):.0%}; overlap (Jaccard) median {np.median(jac):.2f}, IQR {np.percentile(jac, 25):.2f}-{np.percentile(jac, 75):.2f}")
    sh_cnt = Counter(c for p in s2 for c in p.a & p.b)
    rep.line(f"distinct molecules ever shared: {len(sh_cnt)}; shared in >= 10 pairs: {sum(v >= 10 for v in sh_cnt.values())}; "
             f"top 5 by pairs: " + ", ".join(f"CID {c} ({v})" for c, v in sh_cnt.most_common(5)))
    rep.line(f"label y: setting 2 mean {np.mean([p.y for p in s2]):.3f} (sd {np.std([p.y for p in s2]):.3f}); setting 1 mean {np.mean([p.y for p in train if p.jaccard == 0]):.3f}")

    def fp(X, size, lam):
        def f(tr, te):
            return B1aPlus(q, X, A, B, S if size else None, lam).fit(tr, y[tr]).predict(te)
        return f
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(mixture.paper_distance(train, op.cids_M, op.probs_MxW), y, np.zeros(len(train))))
    res = {"B1a + size": evaluate.cross_validate(train, fp(np.zeros((len(ids), 0)), True, 0.0))}
    res["B1a+ groups + size"] = evaluate.cross_validate(train, fp(X_MxG, True, LAM))
    print("  done CV", flush=True)
    m2 = ~ref.no_overlap_N
    nshN = np.array([len(train[i].a & train[i].b) for i in ref.pair_N])

    rep.section("1. Setting 2: size and group dominance, trained on all pairs (standard CV)")
    rows = []
    for subset, mask in [("setting 2 (all)", m2), ("  1 shared molecule", nshN == 1), ("  2 shared", nshN == 2), ("  >= 3 shared", nshN >= 3)]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask, n_boot=N_BOOT)
        rows.append(["B1a (reference)", subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", "", ""])
        prev = ref
        for k in res:
            s = evaluate.score_cv(train, res[k], reference=prev, mask_N=mask, n_boot=N_BOOT)
            wc = ""
            if subset == "setting 2 (all)":
                d0, (lo, hi), ns = within_change(train, prev, res[k], mask)
                wc = f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}] ({ns} studies)"
            rows.append([k, subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]", wc])
            prev = res[k]
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from the row above: RMSE / r [95% CI]", "within-study r change [CI]"], rows)
    rep.line("Each change is against the previous model (B1a -> + size -> + groups), same folds.")
    rep.save()


if __name__ == "__main__":
    main()
