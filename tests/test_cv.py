"""Step 2 checks: the zero-overlap splitter never leaks a held-out mixture into training.

Run with pytest, or directly:  python3 tests/test_cv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import cv, data  # noqa: E402

PAIRS = data.load_train()


def test_no_mixture_leaks_and_partition_is_complete():
    for f in cv.mixture_holdout_folds(PAIRS, k=5, repeats=3, seed=0):
        train_mixes = {m for i in f.train for m in (PAIRS[i].a, PAIRS[i].b)}
        test_mixes = {m for i in f.test for m in (PAIRS[i].a, PAIRS[i].b)}
        assert not train_mixes & test_mixes
        assert sorted(f.train + f.test + f.half) == list(range(len(PAIRS)))
        assert f.test and f.train


def test_each_pair_tested_at_most_once_per_repeat():
    folds = cv.mixture_holdout_folds(PAIRS, k=5, repeats=2, seed=1)
    for rep in (0, 1):
        tested = [i for f in folds if f.repeat == rep for i in f.test]
        assert len(tested) == len(set(tested))


def test_single_pair_mixtures_are_always_tested():
    # Bushdid mixtures occur in exactly one pair each, so every Bushdid pair is tested each repeat.
    folds = cv.mixture_holdout_folds(PAIRS, k=5, repeats=1, seed=0)
    tested = {i for f in folds for i in f.test}
    assert all(i in tested for i, p in enumerate(PAIRS) if p.dataset == "Bushdid")


def test_deterministic():
    a = cv.mixture_holdout_folds(PAIRS, k=5, repeats=2, seed=7)
    assert a == cv.mixture_holdout_folds(PAIRS, k=5, repeats=2, seed=7)
    assert a != cv.mixture_holdout_folds(PAIRS, k=5, repeats=2, seed=8)


def test_zero_jaccard_subset():
    f = cv.mixture_holdout_folds(PAIRS, k=5)[0]
    z = cv.zero_jaccard(PAIRS, f.test)
    assert set(z) <= set(f.test) and all(PAIRS[i].jaccard == 0 for i in z)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
