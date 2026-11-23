"""Noise ceiling for the TRAINING labels, from how each label was measured.

Count labels: a label that is exactly k/N is a fraction of correct answers (triangle test), with counting
noise var = p (1 - p) / N, p = label. Detected from the data: every Bushdid label is k/26; every Ravia 4
label is k/N with N in {48, 56, 64} (lowest-terms N is a divisor; the true N is taken as the smallest of
48, 56, 64 that the lowest-terms N divides; 26 for Bushdid). Other studies' labels are not exact
fractions (or are means of integer ratings): rating labels.
Rating labels: noise of a panel mean taken from V50 (one panel, two sessions): var(session 1 - session 2) / 4
is the noise variance of the two-session mean. Assumes the training panels are as large as V50's
(16 subjects x 2); stated as a caveat.
Reliability of a subset R = 1 - mean(noise var) / var(labels); ceiling r = sqrt(R); share reached = r^2 / R.
Pooled over each subset (setting 1, share 1-2, all pairs); no study is reported on its own. Description only.

Env:     mist
Usage:   python evaluation/B1aplus/eval_noise_ceiling_counts.py
Writes:  results/eval_noise_ceiling_counts/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, evaluate  # noqa: E402
from mixdist.b1a_plus import size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402

N_BOOT = 2000
COUNT_N = {"Bushdid": [26], "Ravia 4": [48, 56, 64]}


def lowest_n(v, top=120):
    return next((n for n in range(1, top + 1) if abs(v * n - round(v * n)) < 1e-6), None)


def main():
    rep = Report("eval_noise_ceiling_counts", {"n_boot": N_BOOT, "count_N": COUNT_N})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    val = data.load_validation()
    r1 = np.array([p.extra["rep1"] for p in val]); r2 = np.array([p.extra["rep2"] for p in val])
    var_rating = float(np.var(r1 - r2) / 4)

    noise = np.full(P, var_rating); kind = np.array(["rating"] * P, dtype=object); bad = 0
    for i, p in enumerate(train):
        if p.dataset in COUNT_N:
            n0 = lowest_n(p.y)
            N = next((n for n in COUNT_N[p.dataset] if n0 and n % n0 == 0), None)
            if N is None:
                bad += 1; continue
            noise[i] = p.y * (1 - p.y) / N; kind[i] = "count"
    rep.section("1. Noise per label")
    rep.line(f"rating labels: noise sd of a panel mean {np.sqrt(var_rating):.3f} (from V50 sessions); count labels: "
             f"sd sqrt(p(1-p)/N), mean {np.sqrt(noise[kind == 'count']).mean():.3f}; count labels not matched to an N: {bad}")

    sh = np.array([len(p.a & p.b) for p in train])
    pred = np.load(ROOT / "results/eval_setting2_shared_fraction/b1a_pp_cv_pred.npz")["pred_P"]
    S = size_features(train)
    rng = np.random.default_rng(0)
    rep.section("2. Ceiling and share reached (pooled per subset)")
    rows, out = [], {}
    for name, m in [("setting 1", sh == 0), ("share 1-2", (sh >= 1) & (sh <= 2)), ("all pairs", np.ones(P, bool))]:
        idx = np.where(m)[0]
        R = 1 - noise[idx].mean() / y[idx].var()
        r_pp = np.corrcoef(pred[idx], y[idx])[0, 1]
        X = np.c_[np.ones(len(idx)), S[idx]]; ry = y[idx] - X @ np.linalg.lstsq(X, y[idx], rcond=None)[0]
        rp = pred[idx] - X @ np.linalg.lstsq(X, pred[idx], rcond=None)[0]
        R_c = 1 - noise[idx].mean() / ry.var()                    # reliability of the part size does not explain
        r_c = np.corrcoef(rp, ry)[0, 1]
        bs = []
        for _ in range(N_BOOT):
            j = rng.choice(idx, len(idx))
            bs.append(1 - noise[j].mean() / y[j].var())
        ci = np.percentile(bs, [2.5, 97.5])
        rows.append([name, len(idx), f"{(kind[idx] == 'count').mean():.0%}", f"{y[idx].std():.3f}", f"{np.sqrt(noise[idx].mean()):.3f}",
                     f"{R:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", f"{np.sqrt(R):.2f}", f"{r_pp:.2f} -> {r_pp ** 2 / R:.0%}",
                     f"{R_c:.2f}", f"{r_c:.2f} -> {r_c ** 2 / R_c:.0%}"])
        out[name] = {"R": R, "R_ci": list(ci), "r": r_pp, "R_content": R_c, "r_content": r_c}
    rep.table(["subset", "pairs", "count labels", "label sd", "noise sd", "reliability R [CI]", "ceiling r",
               "B1a++ r -> share reached", "R after size removed", "B1a++ content r -> reached"], rows)
    rep.line("CI resamples pairs (description). 'after size removed': size terms partialled from labels and prediction, in-sample.")
    rep.value("result", out)
    rep.save()


if __name__ == "__main__":
    main()
