"""mixdist.mixture: mixture profiles and cosine distance on a hand-made example.

Run with pytest, or directly:  python tests/test_mixture.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import mixture  # noqa: E402
from mixdist.data import Pair  # noqa: E402

CIDS_M = np.array([10, 20, 30])
FEAT_MxD = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
PAIRS = [Pair("toy", "A", "B", frozenset({10, 20}), frozenset({30}), 0.5),
         Pair("toy", "A", "C", frozenset({10}), frozenset({20}), 0.9)]


def test_mean_profiles():
    a_PxD, b_PxD = mixture.mean_profiles(PAIRS, CIDS_M, FEAT_MxD)
    assert np.allclose(a_PxD, [[0.5, 0.5], [1.0, 0.0]])
    assert np.allclose(b_PxD, [[1.0, 1.0], [0.0, 1.0]])


def test_cosine_distance():
    dist_P = mixture.mean_cosine_distance(PAIRS, CIDS_M, FEAT_MxD)
    assert np.allclose(dist_P, [0.0, 1.0])                  # same direction / orthogonal


def test_distance_is_symmetric():
    swapped = [Pair(p.dataset, p.id_b, p.id_a, p.b, p.a, p.y) for p in PAIRS]
    assert np.allclose(mixture.mean_cosine_distance(PAIRS, CIDS_M, FEAT_MxD),
                       mixture.mean_cosine_distance(swapped, CIDS_M, FEAT_MxD))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
