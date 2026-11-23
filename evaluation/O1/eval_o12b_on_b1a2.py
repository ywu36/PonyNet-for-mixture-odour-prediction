# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = pairs (820)   B = draws
# ---------------------------------------------------------------------------------------------
"""Profile length on top of B1a++: does keeping log|p| (how specific vs "white" a mixture is) add to B1a++?

B1a++ = B1a + size covariates + 24 RDKit group gains (B1a+ groups + size, lam 1e-3; mixdist/b1a_plus.py).
New (mixdist/o12.py O12b, same gains, same size covariates, same lam on theta, beta unpenalised):
    d = sqrt(|u(A) - u(B)|^2 + beta (h(A) - h(B))^2),  u = p/|p|, h = log|p|, p = gain-weighted mean profile
    beta = 0 gives B1a++. beta = b^2, b starts at 0.1.
Fixed before running (rule of evaluation/B1aplus/eval_b1a_plus_setting1_claim.py):
  1. Change from B1a++ on the same folds (standard CV 50 x 5, whole mixtures held out), RMSE and r with 95% CI by
     resampling mixtures; within-study r change (evaluation only; within_change from eval_setting2_step1.py).
     Setting 1 (pairs sharing no molecule) and setting 2 (sharing >= 1), all pairs alongside.
     Claim "profile length improves setting s" only if both the r-change CI and the within-study CI are > 0.
  2. beta on all 820 pairs, 95% CI from 100 mixture-resample refits.

Env:     mist
Usage:   python evaluation/O1/eval_o12b_on_b1a2.py
Writes:  results/eval_o12b_on_b1a2/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402

LAM, N_BOOT, N_BOOT_BETA = 1e-3, 2000, 100


def main():
    rep = Report("eval_o12b_on_b1a2", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_beta": N_BOOT_BETA})
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: O12b(q, Xg, A, B, True, True, size_PxS=S, lam=LAM).fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)

    rep.section("1. B1a++ + profile length vs B1a++, same folds")
    rows, claims = [], {}
    s1 = base.no_overlap_N
    for subset, mask in [("setting 1 (share no molecule)", s1), ("setting 2 (share >= 1)", ~s1), ("all pairs", None)]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        mm = np.ones(len(base.pred_N), bool) if mask is None else mask
        d0, (lo, hi), ns = within_change(train, base, res, mm)
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / "
                     f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}] ({ns} studies)"])
        claims[subset] = bool(s["r_change_ci"][0] > 0 and lo > 0)
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "+ length RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line("claim rule: " + "; ".join(f"{k}: {'MET' if v else 'NOT met'}" for k, v in claims.items()))

    allp = np.arange(len(train))
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_BETA)
    full = O12b(q, Xg, A, B, True, True, size_PxS=S, lam=LAM).fit(allp, y)
    bb = np.array([O12b(q, Xg, A, B, True, True, size_PxS=S, lam=LAM).fit(allp, y, w_pairs=(cnt[b, ja] * cnt[b, jb]).astype(float) + 1e-12).beta
                   for b in range(N_BOOT_BETA)])
    rep.section("2. beta (all 820 pairs)")
    rep.line(f"beta = {full.beta:.3f} [{np.percentile(bb, 2.5):.3f}, {np.percentile(bb, 97.5):.3f}] ({N_BOOT_BETA} mixture-resample refits); "
             f"size covariates in the same fit: {full.lin.numpy()[2:].round(4).tolist()} (B1a++ alone: "
             f"{B1aPlus(q, Xg, A, B, S, LAM).fit(allp, y).lin.numpy()[2:].round(4).tolist()})")
    rep.value("result", {"beta": full.beta, "beta_ci": np.percentile(bb, [2.5, 97.5]).tolist(), "claims": claims})
    rep.save()


if __name__ == "__main__":
    main()
