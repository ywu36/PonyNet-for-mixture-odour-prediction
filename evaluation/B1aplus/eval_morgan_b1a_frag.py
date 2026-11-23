"""Helper: fragment-overlap distance between two mixtures (same definition as evaluation/B1/eval_morgan_b1a.py).

1 - |A and B| / |A or B| over the set of Morgan fragment bins (2048 folded, radius r) present in any
molecule of each mixture.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation/B1"))
from eval_morgan_b1a import morgan, tanimoto_distance  # noqa: E402


def frag_distance(pairs, cids_M, radius=1, smiles_M=None):
    from mixdist import features
    if smiles_M is None:
        op = features.load("openpom")
        assert np.array_equal(op.cids_M, cids_M)
        smiles_M = [str(s) for s in op.smiles_M]
    return tanimoto_distance(pairs, cids_M, morgan(smiles_M, radius, False))
