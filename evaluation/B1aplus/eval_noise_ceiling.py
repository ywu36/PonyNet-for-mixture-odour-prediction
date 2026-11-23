"""How much of the rated distance can any model predict? (noise ceiling)

Reliability R = share of the label's variance that is reproducible. No model can exceed r = sqrt(R).
Two estimates, both description only:
  1. Training data: 44 pairs whose two mixtures were rated in two different experiments (same molecules).
     r between the two panel means. This mixes rating noise WITH experiment differences (other panel,
     other session, other context pairs), so 1 - r overstates pure noise: R_train is a LOWER bound.
     Also: mean difference between the two experiments (systematic shift) and r after removing it.
  2. V50 (16 subjects x 2 sessions, same panel, same pairs): r(session 1 mean, session 2 mean) is the
     reliability of a 16-person half; Spearman-Brown to both sessions: R_v = 2 r / (1 + r).
     Labels used only to estimate noise, never to fit anything.
CI: 95% bootstrap over pairs (44 / 50 pairs; mixtures barely repeat within these sets).
Then, for setting 1 (325 pairs) and share 1-2 (179): r reached by size alone and by B1a++ (CV), vs the ceiling
sqrt(R); share of reachable explained variance (r^2 / R) reached so far.

Env:     mist
Usage:   python evaluation/B1aplus/eval_noise_ceiling.py
Writes:  results/eval_noise_ceiling/<date_time>.{txt,json}
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, evaluate  # noqa: E402
from mixdist.b1a_plus import size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402

N_BOOT = 5000


def boot_r(a, b, rng):
    i = rng.integers(0, len(a), (N_BOOT, len(a)))
    return np.percentile([np.corrcoef(a[j], b[j])[0, 1] for j in i], [2.5, 97.5])


def main():
    rep = Report("eval_noise_ceiling", {"n_boot": N_BOOT})
    rng = np.random.default_rng(0)
    train = data.load_train()

    rep.section("1. Training: same pair rated in two experiments")
    g = defaultdict(list)
    for p in train:
        g[frozenset((p.a, p.b))].append(p)
    rr = [v for v in g.values() if len(v) > 1]
    y1 = np.array([v[0].y for v in rr]); y2 = np.array([v[1].y for v in rr])
    combos = defaultdict(list)
    for v, a, b in zip(rr, y1, y2):
        combos[f"{v[0].dataset} vs {v[1].dataset}"].append(a - b)
    r_tr = np.corrcoef(y1, y2)[0, 1]; ci = boot_r(y1, y2, rng)
    rep.line(f"{len(rr)} pairs; r = {r_tr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]; mean |diff| {np.abs(y1 - y2).mean():.3f}; label sd {np.r_[y1, y2].std():.3f}")
    rows = [[k, len(d), f"{np.mean(d):+.3f}", f"{np.std(d):.3f}"] for k, d in combos.items()]
    rep.table(["experiments (first - second)", "pairs", "mean diff", "sd diff"], rows)
    # remove each experiment-combination's mean shift, then correlate
    y2s = y2.copy()
    for k, d in combos.items():
        idx = [i for i, v in enumerate(rr) if f"{v[0].dataset} vs {v[1].dataset}" == k]
        y2s[idx] += np.mean(d)
    r_tr2 = np.corrcoef(y1, y2s)[0, 1]
    rep.line(f"after removing each combination's mean shift: r = {r_tr2:.3f} (shift-free agreement; uses the shift, description only)")
    rep.line(f"most of these pairs share molecules: {sum(v[0].jaccard > 0 for v in rr)} of {len(rr)}")

    rep.section("2. V50: same panel, same pairs, two sessions")
    val = data.load_validation()
    r1 = np.array([p.extra["rep1"] for p in val]); r2 = np.array([p.extra["rep2"] for p in val])
    r_v = np.corrcoef(r1, r2)[0, 1]; ci_v = boot_r(r1, r2, rng)
    R_v = 2 * r_v / (1 + r_v); R_v_ci = 2 * ci_v / (1 + ci_v)
    rep.line(f"{len(val)} pairs; r(session 1, session 2) = {r_v:.3f} [{ci_v[0]:.3f}, {ci_v[1]:.3f}]; "
             f"reliability of the 2-session mean R = {R_v:.3f} [{R_v_ci[0]:.3f}, {R_v_ci[1]:.3f}] -> ceiling r = {np.sqrt(R_v):.3f}")
    rep.line(f"label sd V50 {np.array([p.y for p in val]).std():.3f}; mean |session diff| {np.abs(r1 - r2).mean():.3f}")

    rep.section("3. How far below the ceiling are we? (r^2 / R = share of reachable variance explained)")
    y = evaluate.labels(train); S = size_features(train)
    sh = np.array([len(p.a & p.b) for p in train])
    pred = np.load(ROOT / "results/eval_setting2_shared_fraction/b1a_pp_cv_pred.npz")["pred_P"]   # B1a++ standard CV, mean over repeats
    rows = []
    for name, m in [("setting 1", sh == 0), ("share 1-2", (sh >= 1) & (sh <= 2)), ("all pairs", np.ones(len(sh), bool))]:
        X = np.c_[np.ones(m.sum()), S[m]]
        r_size = np.corrcoef(X @ np.linalg.lstsq(X, y[m], rcond=None)[0], y[m])[0, 1]
        r_pp = np.corrcoef(pred[m], y[m])[0, 1]
        cells = [name, int(m.sum()), f"{r_size:.2f} (in-sample)", f"{r_pp:.2f}"]
        for lab, R in [("train", r_tr), ("V50", R_v)]:
            cells.append(f"{np.sqrt(R):.2f} | {r_pp ** 2 / R:.0%}")
        rows.append(cells)
    rep.table(["subset", "pairs", "r size only", "r B1a++ (CV)", "ceiling | reached (R = train 0.56)", "ceiling | reached (R = V50)"], rows)
    rep.line("Caveat: R from other pairs. Ceilings assume the subset's labels are as reliable as those pairs'.")
    rep.value("result", {"r_train": r_tr, "r_train_ci": list(ci), "r_train_shift_free": r_tr2, "r_v50": r_v, "R_v50": R_v})
    rep.save()


if __name__ == "__main__":
    main()
