"""Push/pull tool checks on synthetic data with a known answer.

Planted: unit "pull" lowers the rating by 0.10 when both mixtures carry it, unit "push" raises it by
0.10 when exactly one does, unit "null" does nothing. Given mixture effects these are one quantity,
gamma (both-sided minus twice one-sided): -0.10 for "pull", -0.20 for "push", 0 for "null". Plus random per-mixture effects and noise.
Run with pytest, or directly:  python3 tests/test_pushpull.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import pushpull  # noqa: E402


def synthetic(seed=0, n_mix=60, n_pairs=600):
    rng = np.random.default_rng(seed)
    names = ["pull", "push", "null"]
    mols = [f"m{i}" for i in range(300)]
    has = {m: rng.random(3) < 0.25 for m in mols}
    mixes = [frozenset(rng.choice(mols, rng.integers(2, 6), replace=False)) for _ in range(n_mix)]
    eff = rng.normal(0, 0.05, n_mix)
    a, b, y = [], [], []
    while len(y) < n_pairs:
        i, j = rng.choice(n_mix, 2, replace=False)
        if mixes[i] & mixes[j]:
            continue
        inA = np.any([has[m] for m in mixes[i]], 0); inB = np.any([has[m] for m in mixes[j]], 0)
        v = 0.5 + eff[i] + eff[j] - 0.10 * (inA[0] & inB[0]) + 0.10 * (inA[1] ^ inB[1]) + rng.normal(0, 0.05)
        a.append(mixes[i]); b.append(mixes[j]); y.append(v)
    units = pushpull.units_table(names, {m: has[m].astype(int) for m in mols})
    return pushpull.Pairs(a, b, y), units


def test_planted_pull_push_found_and_null_not():
    pairs, units = synthetic()
    rows = {r["unit"]: r for r in pushpull.analyse(pairs, units, mode="presence", control="none", n_boot=200)}
    assert abs(rows["pull"]["gamma"] + 0.10) < 0.03 and rows["pull"]["q"] < 0.05
    assert abs(rows["push"]["gamma"] + 0.20) < 0.04 and rows["push"]["q"] < 0.05
    assert rows["null"]["q"] > 0.05


def test_overlap_control_removes_general_sharing():
    """Every unit pulls a little when shared (a general rule): without control all look like
    pulls; with the overlap control no single unit should stand out."""
    rng = np.random.default_rng(1)
    names = [f"u{k}" for k in range(8)]
    mols = [f"m{i}" for i in range(300)]
    has = {m: rng.random(8) < 0.3 for m in mols}
    mixes = [frozenset(rng.choice(mols, rng.integers(2, 6), replace=False)) for _ in range(60)]
    eff = rng.normal(0, 0.05, 60); a, b, y = [], [], []
    while len(y) < 600:
        i, j = rng.choice(60, 2, replace=False)
        if mixes[i] & mixes[j]:
            continue
        inA = np.any([has[m] for m in mixes[i]], 0); inB = np.any([has[m] for m in mixes[j]], 0)
        o = (inA & inB).sum() / max((inA | inB).sum(), 1)
        a.append(mixes[i]); b.append(mixes[j]); y.append(0.5 + eff[i] + eff[j] - 0.3 * o + rng.normal(0, 0.05))
    pairs, units = pushpull.Pairs(a, b, y), pushpull.units_table(names, {m: has[m].astype(int) for m in mols})
    gen = pushpull.general_sharing(pairs, units, n_boot=100)
    assert gen["hi"] < 0
    raw = pushpull.analyse(pairs, units, control="none", n_boot=100)
    ctl = pushpull.analyse(pairs, units, control="overlap", n_boot=100)
    assert sum(r["gamma"] < 0 for r in raw) >= 7
    assert all(r["q"] > 0.05 for r in ctl)


def test_shared_setting_ignores_shared_molecules():
    m = ["x", "y", "z", "w"]
    units = pushpull.units_table(["u"], {"x": [1], "y": [0], "z": [0], "w": [1]})
    pairs = pushpull.Pairs([["x", "y"]], [["x", "z"]], [0.3])
    sA, sB = pushpull._shares(pairs, units, exclude_shared=True)
    assert sA[0, 0] == 0 and sB[0, 0] == 0          # u only on the shared molecule x -> not counted


if __name__ == "__main__":
    test_planted_pull_push_found_and_null_not(); test_overlap_control_removes_general_sharing(); test_shared_setting_ignores_shared_molecules(); print("ok")
