"""Does team D2Smell's model already carry the setting-2 shared-molecule pull (B1b's F_mol term)?

D2Smell (XGBoost) has three ways to see a shared molecule: identity columns per side (a tree can split on
"molecule j in mixture 1" and "molecule j in mixture 2"), odour differences from a pair-blend network, and its
augmentation (per shared molecule, a copy of the pair with it removed from mixture 1 only, label + 0.05).
Test, no retraining: its saved cross-validation predictions (results/eval_d2smell_ablation/2026-11-16_001424_final.json,
5 folds x 3 repeats on the team's 780 pairs, whole mixtures held out; NOT the same folds as B1a++) are matched to our
820 pairs. On the matched setting-2 pairs, regress each model's error (label - CV prediction, mean over repeats) on
[1, z(k), z(F_mol)] (the B1b stage-2 form). g_F < 0 = the pull is still left in that model's errors; g_F ~ 0 = the
model already has it. Same regression on B1a++ CV errors (same pairs) for comparison. 95% CI: resample mixtures
(weight count(A) x count(B)), 2000 draws. Scores on the matched pairs are descriptive (folds differ).
Never split by study.

Env:     mist
Usage:   python evaluation/B1b/check_d2smell_shared.py
Writes:  results/check_d2smell_shared/<date_time>.{txt,json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, evaluate, features, mixture  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.b1b import shared_features  # noqa: E402
from mixdist.report import Report  # noqa: E402

N_BOOT = 2000
SRC = ROOT / "results/eval_d2smell_ablation/2026-11-16_001424_final.json"
VARIANTS = [("D2Smell full", "full model (the team's)"), ("odour only", "A odour differences only"),
            ("identity only", "A identity only"), ("identity + odour", "A identity + odour (ratings columns removed)"),
            ("full, no augmentation", "B no augmentation (780 rows)")]


def wfit(X, e, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], e * sw, rcond=None)[0]


def wr(a, b, w):
    w = w / w.sum(); a = a - (w * a).sum(); b = b - (w * b).sum()
    return (w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum())


def main():
    rep = Report("check_d2smell_shared", {"source": str(SRC.relative_to(ROOT)), "n_boot": N_BOOT})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); q = np.asarray(mixture.quantile_per_word(op.probs_MxW), float)
    A, B = membership(train, op.cids_M); F, k = shared_features(q, A, B)
    b1a = np.load(ROOT / "results/eval_setting2_shared_fraction/b1a_pp_cv_pred.npz")["pred_P"]

    # the team's 780 original pairs, as ablation.load_team_data builds them (first rows of its gt file)
    team = data.locate("olfboost_repo") / "data/processed"
    gt = pd.read_csv(team / "gt_with_dataset_V2_augmented_dataset.csv"); defs = pd.read_csv(team / "Mixure_Definitions_augmented_dataset.csv")
    cid_cols = [c for c in defs.columns if c.startswith("CID_")]
    mix = {(ds, lab): frozenset(int(c) for c in cids if c == c and c != 0)
           for ds, lab, *cids in defs[["Dataset", "Mixture Label"] + cid_cols].itertuples(index=False)}
    n_orig = int(gt["augmentation_Action"].isna().sum())
    ours = {}
    for i, p in enumerate(train):
        ours.setdefault(frozenset([p.a, p.b]), []).append(i)
    team_to_ours = np.full(n_orig, -1)
    for j, (ds, m1, m2, yy) in enumerate(gt[["Dataset", "Mixture 1", "Mixture 2", "Experimental Values"]].iloc[:n_orig].itertuples(index=False)):
        cand = [i for i in ours.get(frozenset([mix[(ds, m1)], mix[(ds, m2)]]), []) if abs(y[i] - yy) < 1e-6]
        if cand:
            team_to_ours[j] = cand[0]
    cv = json.load(open(SRC))
    cv = next(it["value"] for it in cv["items"] if isinstance(it, dict) and it.get("key") == "cv")
    pairN = np.array(cv["pair_N"])
    pred = {}
    for short, lab in VARIANTS:
        pN = np.array(cv["pred_N"][lab]); s = np.bincount(pairN, pN, n_orig) / np.bincount(pairN, minlength=n_orig)
        v = np.full(P, np.nan); ok = team_to_ours >= 0; v[team_to_ours[ok]] = s[ok]; pred[short] = v
    pred = {"B1a++": b1a, **pred}
    m = np.isfinite(pred["D2Smell full"])
    rep.line(f"team pairs {n_orig}; matched to ours {int((team_to_ours >= 0).sum())}; our pairs covered {int(m.sum())} of {P} "
             f"(setting 1 {int((m & (k == 0)).sum())}, setting 2 {int((m & (k > 0)).sum())}: share 1-2 {int((m & (k >= 1) & (k <= 2)).sum())}, share >= 3 {int((m & (k >= 3)).sum())})")

    mixes = sorted({x for p in train for x in (p.a, p.b)}, key=sorted); im = {x: j for j, x in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    rng = np.random.default_rng(0); cnt = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    W = (cnt[:, ja] * cnt[:, jb]).astype(float)

    rep.section("1. Scores on the matched pairs (descriptive: D2Smell folds 5x3 on its 780 pairs, B1a++ 50x5 on 820)")
    rows = []
    for sub, mk in [("all matched", m), ("setting 1", m & (k == 0)), ("setting 2", m & (k > 0)), ("  share 1-2", m & (k >= 1) & (k <= 2)), ("  share >= 3", m & (k >= 3))]:
        idx = np.where(mk)[0]
        for name, v in pred.items():
            r0 = wr(v[idx], y[idx], np.ones(len(idx))); e0 = np.sqrt(np.mean((v[idx] - y[idx]) ** 2))
            if name == "B1a++":
                rows.append([sub, len(idx), name, f"{e0:.4f} / {r0:.3f}", ""])
            else:
                d = [wr(v[idx], y[idx], W[b, idx] + 1e-12) - wr(b1a[idx], y[idx], W[b, idx] + 1e-12) for b in range(N_BOOT)]
                d = np.array(d); d = d[np.isfinite(d)]
                rows.append([sub, "", name, f"{e0:.4f} / {r0:.3f}", f"{r0 - wr(b1a[idx], y[idx], np.ones(len(idx))):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"])
    rep.table(["subset", "pairs", "model", "RMSE / r", "r minus B1a++ [CI]"], rows)

    rep.section("2. Is the pull left in each model's errors? err ~ 1 + z(k) + z(F_mol), matched setting-2 pairs")
    s2 = np.where(m & (k > 0))[0]
    zk = (k[s2] - k[s2].mean()) / k[s2].std(); zF = (F[s2] - F[s2].mean()) / F[s2].std()
    X = np.c_[np.ones(len(s2)), zk, zF]
    rows, out = [], {}
    for name, v in pred.items():
        e = y[s2] - v[s2]
        g = wfit(X, e, np.ones(len(s2))); gb = np.array([wfit(X, e, W[b, s2] + 1e-12) for b in range(N_BOOT)])
        ci = lambda j: f"{g[j]:+.4f} [{np.percentile(gb[:, j], 2.5):+.4f}, {np.percentile(gb[:, j], 97.5):+.4f}]"  # noqa: E731
        rows.append([name, len(s2), ci(1), ci(2)]); out[name] = {"g": g.tolist(), "gF_ci": list(np.percentile(gb[:, 2], [2.5, 97.5]))}
    rep.table(["model whose errors", "pairs", "g_k (number shared) [CI]", "g_F (standing-out shared) [CI]"], rows)
    rep.line("g_F < 0: pairs whose shared molecules stand out are rated closer than that model predicts (pull missing).")
    rep.value("result", out)
    rep.save()


if __name__ == "__main__":
    main()
