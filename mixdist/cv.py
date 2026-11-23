"""Zero-overlap cross-validation: folds hold out whole mixtures (PLAN.md, step 2).

A mixture is identified by its molecule content, so the same mixture appearing in two studies
(6 Ravia 1 / Snitz 2 pairs do) is one mixture here and cannot leak across folds.

For each fold, with H = the held-out mixtures:
    test   pairs with BOTH mixtures in H      (the model has seen neither mixture)
    train  pairs with NO mixture in H
    half   pairs with exactly one mixture in H (used for neither; kept for optional reporting)

Mixtures are assigned to folds by connected component of the pair graph. A small component
(Bushdid: every mixture occurs in exactly one pair) goes to one fold whole, so its pairs are
all tested. A large component (Snitz and Ravia studies compare every mixture with many others)
has its mixtures spread evenly over the folds; only the pairs that fall inside one fold are
tested, so use several repeats to cover more pairs.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train: tuple          # indices into the pair list
    test: tuple
    half: tuple


def _components(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        parent[find(p.a)] = find(p.b)
    comps = defaultdict(list)
    for m in parent:
        comps[find(m)].append(m)
    # deterministic order, independent of hash seeds
    return sorted((sorted(c, key=sorted) for c in comps.values()), key=lambda c: (len(c), sorted(c[0])))


def mixture_holdout_folds(pairs, k: int = 5, repeats: int = 1, seed: int = 0,
                          max_whole_component: int = 4) -> list[Fold]:
    """Return ``k * repeats`` folds over ``pairs`` (objects with frozenset attributes ``a``, ``b``)."""
    comps = _components(pairs)
    out = []
    for rep in range(repeats):
        rng = random.Random(seed * 1000 + rep)
        fold_of = {}
        small = [c for c in comps if len(c) <= max_whole_component]
        rng.shuffle(small)
        for i, comp in enumerate(small):
            for m in comp:
                fold_of[m] = i % k
        for comp in comps:
            if len(comp) > max_whole_component:
                comp = comp[:]
                rng.shuffle(comp)
                start = rng.randrange(k)
                for i, m in enumerate(comp):
                    fold_of[m] = (start + i) % k
        for f in range(k):
            tr, te, half = [], [], []
            for i, p in enumerate(pairs):
                n = (fold_of[p.a] == f) + (fold_of[p.b] == f)
                (tr, half, te)[n].append(i)
            out.append(Fold(rep, f, tuple(tr), tuple(te), tuple(half)))
    return out


def zero_jaccard(pairs, idx) -> tuple:
    """Subset of ``idx`` whose two mixtures share no molecule (the V50-like condition)."""
    return tuple(i for i in idx if not (pairs[i].a & pairs[i].b))


# DREAM lockbox (evaluation/CV.md rule 7, fixed 2026-11-18 before any model was scored on it).
LOCKBOX_K = 4             # 1/4 of the mixtures held out
LOCKBOX_SEED = 20261118   # chosen once (the date); not tuned


def lockbox_split(pairs) -> Fold:
    """Fixed lockbox over the 820 DREAM training pairs: ``test`` = lockbox pairs (both mixtures held out),
    ``train`` = development pairs (no lockbox mixture; all development CV runs on these only),
    ``half`` = pairs with one lockbox mixture (used for neither)."""
    return mixture_holdout_folds(pairs, k=LOCKBOX_K, repeats=1, seed=LOCKBOX_SEED)[0]
