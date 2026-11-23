"""Semantic salience: do molecules with an unusual smell dominate their mixture?

Salience sigma_i = length of molecule i's open-POM quantile profile q_i (138 words; B1a's quantile step
centres each word at 0 over the 235 molecules, so this is the distance from the average molecule's smell),
standardised over the 235 molecules. Model (one extra number beta on top of B1a++):
    log w_i = theta . groups_i + beta * sigma_i          B1a++ = B1a + size + groups (lam 1e-3)
beta > 0: smell-distinctive molecules dominate; < 0: they are masked. beta carries the same tiny
penalty lam 1e-3 as theta. Inputs available for a new test pair only.
Fixed before running (rule as in eval_b1a_plus_setting1_claim.py):
  1. setting 1: change from B1a++ (same folds), RMSE and r with 95% CI by resampling mixtures; within-study
     check (evaluation only). Claim "semantic salience improves setting-1 prediction" only if both CIs > 0.
     Setting 2 and all pairs reported alongside.
  2. beta on all 820 pairs, 95% CI from 100 mixture-resample refits.
  3. Sulfur check (description): salience of molecules with a thiol or sulfide vs others; theta of SH and
     sulfide with and without beta; the 10 most salient molecules.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_salience.py
Writes:  results/eval_b1a_plus_salience/<date_time>.{txt,json}
"""
import csv
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
from eval_setting2_step1 import within_change  # noqa: E402

LAM = 1e-3
N_BOOT = 2000
N_BOOT_BETA = 100


def main():
    rep = Report("eval_b1a_plus_salience", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_beta": N_BOOT_BETA})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    Xg, names = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    short = [n.split(" (= ")[0] for n in names]
    sal = np.linalg.norm(q, axis=1); sal_z = (sal - sal.mean()) / sal.std()
    Xs = np.c_[Xg, sal_z]

    def fp(X):
        return lambda tr, te: B1aPlus(q, X, A, B, S, LAM).fit(tr, y[tr]).predict(te)
    base = evaluate.cross_validate(train, fp(Xg)); res = evaluate.cross_validate(train, fp(Xs))
    print("  done CV", flush=True)

    rep.section("1. + salience (beta) vs B1a++, same folds")
    rows = []
    s1 = base.no_overlap_N
    for subset, mask in [("setting 1 (share no molecule)", s1), ("setting 2 (share >= 1)", ~s1), ("all pairs", None)]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        mm = np.ones(len(base.pred_N), bool) if mask is None else mask
        d0, (lo, hi), ns = within_change(train, base, res, mm)
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"])
        if subset.startswith("setting 1"):
            ok = s["r_change_ci"][0] > 0 and lo > 0
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "+ salience RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 1): {'MET' if ok else 'NOT met'}")

    allp = np.arange(len(train))
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_BETA)
    full = B1aPlus(q, Xs, A, B, S, LAM).fit(allp, y)
    beta = float(full.theta[-1])
    bb = np.array([float(B1aPlus(q, Xs, A, B, S, LAM).fit(allp, y, w_pairs=(cnt[b, ja] * cnt[b, jb]).astype(float) + 1e-12).theta[-1]) for b in range(N_BOOT_BETA)])
    th0 = B1aPlus(q, Xg, A, B, S, LAM).fit(allp, y).theta.numpy()
    rep.section("2. beta (all 820 pairs)")
    rep.line(f"beta = {beta:+.3f} [{np.percentile(bb, 2.5):+.3f}, {np.percentile(bb, 97.5):+.3f}] per sd of salience "
             f"(weight x{np.exp(beta):.2f} per sd); salience sd {sal.std():.2f}, range {sal.min():.1f}-{sal.max():.1f}")

    rep.section("3. Sulfur check and the most salient molecules")
    in_tr = np.stack([A, B]).max((0, 1)) > 0
    for g in ["SH", "sulfide"]:
        k = short.index(g)
        car = in_tr & (Xg[:, k] > 0); oth = in_tr & (Xg[:, k] == 0)
        rep.line(f"{g}: {int(car.sum())} training molecules, salience z {sal_z[car].mean():+.2f} vs others {sal_z[oth].mean():+.2f}; "
                 f"theta without beta {th0[k]:+.3f}, with beta {float(full.theta[k]):+.3f}")
    names_by = {}
    try:
        for r in csv.DictReader(open(ROOT / "external/pnas_repo/SOTA/data/raw/openpom_ensemble_predictions_results.csv")):
            names_by[int(r["CID"])] = r["IUPAC Name"]
    except Exception:
        pass
    o = [i for i in np.argsort(-sal_z) if in_tr[i]][:10]
    words = [str(w) for w in op.words_W]
    rep.table(["molecule", "salience z", "top 3 words (quantile)"],
              [[names_by.get(int(op.cids_M[i]), str(op.cids_M[i]))[:40], f"{sal_z[i]:+.2f}", ", ".join(words[j] for j in np.argsort(-q[i])[:3])] for i in o])
    rep.value("result", {"beta": beta, "beta_ci": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))], "claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
