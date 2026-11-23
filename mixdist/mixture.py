# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = molecules        D = feature columns (odour words or embedding)
#   P = pairs            K = molecules in one mixture
# ---------------------------------------------------------------------------------------------
"""From per-molecule features to one number per pair of mixtures.

    mean_profiles     mixture = plain mean of its molecules' feature rows
    cosine_distance   1 - cosine similarity between the A and B profiles of each pair
    paper_distance    the paper's open-POM baseline recipe: quantile per word, length 1, Euclidean
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import QuantileTransformer


def mean_profiles(pairs, cids_M, feat_MxD: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of molecule rows for the A and B mixture of every pair -> (mixA_PxD, mixB_PxD)."""
    row_of_cid = {int(c): i for i, c in enumerate(cids_M)}

    def mean_of(mix) -> np.ndarray:                         # mix: frozenset of K CIDs
        idx_K = [row_of_cid[c] for c in sorted(mix)]
        return feat_MxD[idx_K].mean(axis=0)                  # K x D -> D

    mixA_PxD = np.stack([mean_of(p.a) for p in pairs])
    mixB_PxD = np.stack([mean_of(p.b) for p in pairs])
    return mixA_PxD, mixB_PxD


def cosine_distance(a_PxD: np.ndarray, b_PxD: np.ndarray) -> np.ndarray:
    """Row-wise cosine distance -> dist_P."""
    dot_P = (a_PxD * b_PxD).sum(axis=1)
    norms_P = np.linalg.norm(a_PxD, axis=1) * np.linalg.norm(b_PxD, axis=1)
    return 1.0 - dot_P / norms_P


def mean_cosine_distance(pairs, cids_M, feat_MxD: np.ndarray) -> np.ndarray:
    """Cosine distance between the mean profiles of A and B -> dist_P."""
    return cosine_distance(*mean_profiles(pairs, cids_M, feat_MxD))


def quantile_per_word(feat_MxD: np.ndarray) -> np.ndarray:
    """Each column mapped to a standard normal by its rank over the M molecules (the paper's open-POM
    notebook: QuantileTransformer(output_distribution='normal', n_quantiles=100)). No labels used."""
    return QuantileTransformer(output_distribution="normal", n_quantiles=100).fit_transform(feat_MxD)


def unit_euclidean_distance(a_PxD: np.ndarray, b_PxD: np.ndarray) -> np.ndarray:
    """Each profile scaled to length 1, then Euclidean distance (= sqrt(2 * cosine distance))."""
    a_PxD = a_PxD / np.linalg.norm(a_PxD, axis=1, keepdims=True)
    b_PxD = b_PxD / np.linalg.norm(b_PxD, axis=1, keepdims=True)
    return np.linalg.norm(a_PxD - b_PxD, axis=1)


def paper_distance(pairs, cids_M, feat_MxD: np.ndarray) -> np.ndarray:
    """The paper's open-POM baseline distance: quantile per word over molecules, mean per mixture,
    length 1, Euclidean -> dist_P."""
    return unit_euclidean_distance(*mean_profiles(pairs, cids_M, quantile_per_word(feat_MxD)))
