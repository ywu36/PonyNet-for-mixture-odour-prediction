"""mixdist.evaluate: calibration lines and the pooled cross-validation loop.

Run with pytest, or directly:  python tests/test_evaluate.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data, evaluate  # noqa: E402

TRAIN = data.load_train()


def test_linear_calibration_fits_each_group_separately():
    x_P = np.linspace(0, 1, 40)
    group_P = np.arange(40) % 2 == 0
    y_P = np.where(group_P, 2 * x_P + 1, -x_P + 3)          # a different exact line per group
    fit_predict = evaluate.linear_calibration(x_P, y_P, group_P)
    train_idx, test_idx = np.arange(0, 30), np.arange(30, 40)
    assert np.allclose(fit_predict(train_idx, test_idx), y_P[test_idx])


def test_linear_calibration_single_group_is_one_line():
    x_P = np.linspace(0, 1, 40)
    y_P = 0.5 * x_P + 0.2
    fit_predict = evaluate.linear_calibration(x_P, y_P, np.zeros(40, int))
    assert np.allclose(fit_predict(np.arange(0, 30), np.arange(30, 40)), y_P[30:])


def test_calibration_groups():
    shared = evaluate.calibration_groups(TRAIN, "shared")
    bush = evaluate.calibration_groups(TRAIN, "bushdid")
    typ = evaluate.calibration_groups(TRAIN, "type")
    assert set(shared) == {0}
    assert bush.sum() == sum(p.dataset == "Bushdid" for p in TRAIN)
    assert typ.sum() == sum(p.dataset in ("Bushdid", "Ravia 4") for p in TRAIN)
    # held-out sets: T46 is a triangle test, V50 a similarity rating
    assert evaluate.calibration_group("T46", "type") == 1
    assert evaluate.calibration_group("V50", "type") == 0
    assert evaluate.calibration_group("T46", "bushdid") == 0


def test_cross_validate_pools_aligned_predictions():
    oracle = lambda train_idx, test_idx: evaluate.labels(TRAIN)[test_idx]   # noqa: E731
    res = evaluate.cross_validate(TRAIN, oracle, k=5, repeats=2, seed=0)
    assert np.array_equal(res.pred_N, res.truth_N)
    assert all(TRAIN[i].dataset == s for i, s in zip(res.pair_N, res.study_N))
    rows = dict(res.score_rows())
    assert abs(rows["all pairs"]["r"] - 1.0) < 1e-12
    assert rows["all pairs"]["n"] == len(res.pair_N)
    assert set(rows) == {"all pairs", "  of which share no molecule"}


def test_held_out_sets():
    v50, t46 = evaluate.held_out_sets()
    assert (v50.name, len(v50.pairs), len(v50.reference_P)) == ("V50", 50, 50)
    assert (t46.name, len(t46.pairs), t46.ceiling) == ("T46", 46, None)
    assert abs(v50.ceiling - 0.74154776) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)


def test_overlap_calibration_groups():
    """Overlap groups come from the mixture definitions: 1 = the two mixtures share no molecule."""
    from mixdist.data import Pair
    pairs = [Pair("Snitz 1", "1", "2", frozenset({1, 2}), frozenset({2, 3}), 0.5),
             Pair("Snitz 1", "3", "4", frozenset({1, 2}), frozenset({3, 4}), 0.7)]
    assert list(evaluate.calibration_groups(pairs, "overlap")) == [0, 1]
