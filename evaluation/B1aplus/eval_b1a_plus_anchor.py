"""Anchor discount: is the part two mixtures share tuned out, so that differences relative to it stand out?

In a pair (A, B), molecules in both mixtures form the anchor. Model (one unpenalised number alpha on B1a++):
    log w_i = theta . groups_i + alpha * [i is in both A and B]        B1a++ = B1a + size + groups, lam 1e-3
alpha < 0: the shared anchor counts less in both mixtures (discounted), so the unshared parts drive the
distance; alpha > 0: the anchor counts more (pulls the pair together beyond its share); alpha = 0: B1a++.
Pairs sharing no molecule are unchanged by construction.
Fixed before running (rule as in eval_b1a_plus_setting1_claim.py, on setting 2):
  1. setting 2 (share >= 1): change from B1a++ on the same folds, RMSE and r, 95% CI by resampling
     mixtures; within-study check (evaluation only). Claim "the shared part is weighted differently from
     its share" only if both CIs > 0. Audit by molecules shared (1, 2, >= 3) and setting 1 (should be ~0).
  2. alpha on all 820 pairs, 95% CI from 100 mixture-resample refits.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_anchor.py
Writes:  results/eval_b1a_plus_anchor/<date_time>.{txt,json}
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
from eval_b1a_plus_group_graph import B1aPlusGraph  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402

LAM = 1e-3
N_BOOT = 2000
N_BOOT_ALPHA = 100


def main():
    rep = Report("eval_b1a_plus_anchor", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_alpha": N_BOOT_ALPHA})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    shared_PxM = A * B                                         # 1 if the molecule is in both mixtures of the pair

    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: B1aPlusGraph(q, Xg, A, B, S, LAM, shared_PxM, shared_PxM).fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)

    s1 = base.no_overlap_N
    nsh = np.array([len(train[i].a & train[i].b) for i in base.pair_N])
    rep.section("1. + anchor weight (alpha) vs B1a++, same folds")
    rows = []
    ok = False
    for subset, mask in [("setting 2 (share >= 1)", ~s1), ("  share 1", nsh == 1), ("  share 2", nsh == 2), ("  share >= 3", nsh >= 3),
                         ("setting 1 (unchanged by construction)", s1), ("all pairs", np.ones(len(nsh), bool))]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        wc = ""
        if subset.startswith("setting 2") or subset == "all pairs":
            d0, (lo, hi), ns = within_change(train, base, res, mask)
            wc = f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            if subset.startswith("setting 2"):
                ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]", wc])
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "+ anchor RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 2): {'MET' if ok else 'NOT met'}. Subsets by molecules shared are an audit.")

    allp = np.arange(len(train))
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_ALPHA)
    alpha = float(B1aPlusGraph(q, Xg, A, B, S, LAM, shared_PxM, shared_PxM).fit(allp, y).theta[-1])
    ab = np.array([float(B1aPlusGraph(q, Xg, A, B, S, LAM, shared_PxM, shared_PxM).fit(allp, y, w_pairs=(cnt[b, ja] * cnt[b, jb]).astype(float) + 1e-12).theta[-1])
                   for b in range(N_BOOT_ALPHA)])
    rep.section("2. alpha (all 820 pairs)")
    rep.line(f"alpha = {alpha:+.3f} [{np.percentile(ab, 2.5):+.3f}, {np.percentile(ab, 97.5):+.3f}]: a shared molecule counts x{np.exp(alpha):.2f} "
             f"[x{np.exp(np.percentile(ab, 2.5)):.2f}, x{np.exp(np.percentile(ab, 97.5)):.2f}] relative to an unshared one. < 1: anchor discounted.")
    rep.value("result", {"alpha": alpha, "alpha_ci": [float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))], "claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
