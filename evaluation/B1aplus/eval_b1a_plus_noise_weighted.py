"""B1a++ trained with each pair weighted by 1 / (its label's noise variance); prediction unchanged.

Noise per label as in eval_noise_ceiling_counts.py: count labels (Bushdid k/26, Ravia 4 k/48|56|64)
p(1-p)/N; rating labels the V50 panel-mean noise var(session 1 - session 2)/4.  Fixed before running: standard CV, change from B1a++ on the same folds;
primary setting 1; claim only if r-change CI > 0 and within-study CI > 0. Setting 2 and all pairs reported.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_noise_weighted.py
Writes:  results/eval_b1a_plus_noise_weighted/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402
from eval_noise_ceiling_counts import COUNT_N, lowest_n  # noqa: E402

LAM = 1e-3; N_BOOT = 2000


def main():
    rep = Report("eval_b1a_plus_noise_weighted", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    val = data.load_validation()
    noise = np.full(P, float(np.var(np.array([p.extra["rep1"] for p in val]) - np.array([p.extra["rep2"] for p in val])) / 4))
    for i, p in enumerate(train):
        if p.dataset in COUNT_N:
            n0 = lowest_n(p.y); noise[i] = p.y * (1 - p.y) / next(n for n in COUNT_N[p.dataset] if n % n0 == 0)
    wt = 1 / noise
    cnt_lab = np.array([p.dataset in COUNT_N for p in train])
    rep.line(f"weights: rating labels {wt[~cnt_lab][0]:.0f}; count labels median {np.median(wt[cnt_lab]):.0f} (range {wt[cnt_lab].min():.0f}-{wt[cnt_lab].max():.0f})")
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr], w_pairs=wt[tr]).predict(te))
    print("  done CV", flush=True)
    s1 = base.no_overlap_N; rows = []; ok = False
    for subset, mask in [("setting 1 (primary)", s1), ("setting 2", ~s1), ("all pairs", np.ones(len(s1), bool))]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        d0, (lo, hi), _ = within_change(train, base, res, mask)
        if subset.startswith("setting 1"):
            ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"])
    rep.section("1. Noise-weighted training vs B1a++ (same folds)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 1): {'MET' if ok else 'NOT met'}")
    rep.value("result", {"claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
