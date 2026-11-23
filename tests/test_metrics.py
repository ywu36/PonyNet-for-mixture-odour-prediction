"""mixdist.metrics: Pearson, bootstrap CI, paired comparison, bootstrap medians.

Run with pytest, or directly:  python tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import metrics  # noqa: E402

RNG = np.random.default_rng(0)
Y = RNG.random(60)
PRED = Y + 0.3 * RNG.standard_normal(60)


def test_pearson_matches_numpy():
    assert abs(metrics.pearson(PRED, Y) - np.corrcoef(PRED, Y)[0, 1]) < 1e-12


def test_score_fields_and_ci_contains_r():
    s = metrics.score(PRED, Y, n_boot=2000)
    assert {"n", "r", "rmse", "r_ci"} <= set(s) and s["n"] == 60
    assert s["r_ci"][0] < s["r"] < s["r_ci"][1]


def test_paired_comparison_with_itself_is_null():
    s = metrics.score(PRED, Y, reference=PRED, ceiling=0.8, n_boot=2000)
    assert s["r_ref"] == s["r"] and s["diff_ci"] == (0.0, 0.0) and s["p_vs_ref"] == 1.0
    assert abs(s["pct_ceiling"] - 100 * s["r"] / 0.8) < 1e-12


def test_better_model_wins_the_paired_test():
    s = metrics.score(Y + 0.01 * RNG.standard_normal(60), Y, reference=PRED, n_boot=2000)
    assert s["diff_ci"][0] > 0 and s["p_vs_ref"] < 0.01


def test_bootstrap_medians_deterministic_and_close_to_plain():
    a = metrics.bootstrap_medians(PRED, Y, n_boot=2000)
    assert a == metrics.bootstrap_medians(PRED, Y, n_boot=2000)
    assert abs(a[1] - metrics.pearson(PRED, Y)) < 0.05 and abs(a[0] - metrics.rmse(PRED, Y)) < 0.02


def test_auroc_known_cases():
    assert metrics.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert metrics.auroc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert metrics.auroc([0, 1], [0.5, 0.5]) == 0.5          # tie counts half
    assert np.isnan(metrics.auroc([1, 1], [0.1, 0.2]))      # one class only


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
