"""Read training files and hold out whole mixtures.

Molecule file (Stage 1): a SMILES column and one 0/1 column per odour word; other columns (e.g. free text) are ignored.
Pair file (Stage 2): columns mixture_a, mixture_b, rating; a mixture is its SMILES joined by ';'.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def read_molecules(path: str | Path, smiles_column: str = "smiles"):
    """-> (smiles list, labels molecules x words as 0/1 array, word list). Word columns are those holding only 0 and 1."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or smiles_column not in rows[0]:
        raise ValueError(f"{path}: no column named {smiles_column!r}")
    is01 = lambda v: v.strip() in ("0", "1", "0.0", "1.0")  # noqa: E731
    words = [c for c in rows[0] if c != smiles_column and all(is01(r[c]) for r in rows)]
    if not words:
        raise ValueError(f"{path}: no 0/1 word columns found")
    labels = np.array([[float(r[w]) for w in words] for r in rows], np.float32)
    return [r[smiles_column] for r in rows], labels, words


def read_pairs(path: str | Path, sep: str = ";"):
    """-> (mixtures_a, mixtures_b, ratings); each mixture is a list of SMILES."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for c in ("mixture_a", "mixture_b", "rating"):
        if not rows or c not in rows[0]:
            raise ValueError(f"{path}: needs columns mixture_a, mixture_b, rating")
    split = lambda s: [x.strip() for x in s.split(sep) if x.strip()]  # noqa: E731
    return ([split(r["mixture_a"]) for r in rows], [split(r["mixture_b"]) for r in rows],
            np.array([float(r["rating"]) for r in rows]))


def holdout_mixtures(mixtures_a: Sequence, mixtures_b: Sequence, fraction: float = 0.2, seed: int = 0) -> np.ndarray:
    """Hold out a random ``fraction`` of the distinct mixtures; returns a True/False mask over pairs that is True for
    every pair containing a held-out mixture (the test pairs). The fit then never sees a test mixture."""
    key = lambda m: ";".join(sorted(m))  # noqa: E731
    mixtures = sorted({key(m) for m in list(mixtures_a) + list(mixtures_b)})
    held = set(np.random.default_rng(seed).choice(mixtures, max(1, round(fraction * len(mixtures))), replace=False))
    return np.array([key(a) in held or key(b) in held for a, b in zip(mixtures_a, mixtures_b)])


def subset(items: Sequence, mask) -> list:
    """Items where mask is True."""
    return [x for x, k in zip(items, mask) if k]
