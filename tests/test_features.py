"""mixdist.features: every cached feature file follows the standard layout and covers every
molecule in the data. A model whose file has not been computed yet is skipped.

Run with pytest, or directly:  python tests/test_features.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data, features  # noqa: E402

PRESENT = [m for m in features.MODELS if features.path(m).exists()]


def test_standard_layout():
    for model in PRESENT:
        fs = features.load(model)
        m = len(fs.cids_M)
        assert fs.probs_MxW.shape == (m, len(fs.words_W)) and len(fs.emb_MxE) == m == len(fs.smiles_M)
        if fs.probs_MxW.size:                                 # embedding-only models (Uni-Mol 2) have no odour words
            assert 0.0 <= fs.probs_MxW.min() and fs.probs_MxW.max() <= 1.0
        assert len(set(fs.cids_M.tolist())) == m             # one row per molecule


def test_every_molecule_has_features():
    needed = set(data.load_smiles())
    for model in PRESENT:
        assert needed <= set(features.load(model).cids_M.tolist()), model


def test_openpom_mean_of_members():
    if "openpom" in PRESENT:
        fs = features.load("openpom")
        assert np.allclose(fs.extra["probs_per_model"].mean(axis=0), fs.probs_MxW, atol=1e-6)


if __name__ == "__main__":
    print("feature files present:", ", ".join(PRESENT) or "none")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
