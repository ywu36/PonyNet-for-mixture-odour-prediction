"""Data loader for the DREAM olfactory mixture-distance task.

Standard library only, so it runs before any ML environment exists.

Where things are looked up (first place that has them wins), see ``locate``:
  1. the folder named by the ``OSMO_DATA`` environment variable (a fast local copy),
     then its ``external/`` subfolder (other teams' repos, open-POM),
  2. ``external/`` in this repository: the reference repos pnas_repo, pommix_repo, mist_repo
     (git-ignored),
  3. ``../dream2_data``: the MIST model weights and the remaining downloads.

Cleaning rules implemented here (dream2_data/PLAN.md, "Rules"):
  2. PostChallenge training file: rows with Dataset in {random, manual, angle} are the test set
     and are never loaded as training data.
  3. Mixture definitions come from pnas_repo (corrected Bushdid), never from olfboost's old file.
  4. Validation set: mixture names are whitespace-stripped, pairs are unordered, the two pairs
     with undefined mixtures are dropped, label = mean rating / 100.
"""
from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TEST_ROWS_IN_POSTCHALLENGE = {"random", "manual", "angle"}
EXTRA_TRAIN_DATASETS = ("Ravia 5", "Ravia 6", "Metamer")


REPO = Path(__file__).resolve().parents[1]


def locate(name: str, root: Path | None = None, needs: str = "") -> Path:
    """Path of a reference repo or model folder, e.g. "pnas_repo", "pommix_repo", "mist_odour", "openpom".

    ``needs``: a file or folder inside it that must exist, to skip incomplete copies (the local
    mirror's pnas_repo, for instance, has the data but not the SOTA/ baselines).
    """
    if root is not None:
        return Path(root) / name
    env = os.environ.get("OSMO_DATA")
    places = ([Path(env), Path(env) / "external"] if env else []) + [REPO / "external", REPO.parent / "dream2_data"]
    for place in places:
        if (place / name / needs).exists():
            return place / name
    raise FileNotFoundError(f"{name}/{needs} not found in: " + ", ".join(str(x) for x in places))


@dataclass(frozen=True)
class Pair:
    """One labelled pair of mixtures. A mixture is a frozenset of PubChem CIDs."""

    dataset: str                 # source study, e.g. "Snitz 1", "Bushdid", "T46", "V50"
    id_a: str                    # mixture label within its dataset
    id_b: str
    a: frozenset
    b: frozenset
    y: float                     # perceptual distance in [0, 1] (Bushdid: fraction correct)
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def jaccard(self) -> float:
        return len(self.a & self.b) / len(self.a | self.b)


def _rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.reader(f) if any(c.strip() for c in r)]


def _cids(cells) -> frozenset:
    out = set()
    for c in cells:
        c = c.strip()
        if c and float(c) != 0:
            out.add(int(float(c)))      # some files store CIDs as "6654.0"
    return frozenset(out)


def load_smiles(root: Path | None = None) -> dict[int, str]:
    rows = _rows(locate("pnas_repo", root) / "PostChallenge_Model/Dataset/cid_to_smiles.csv")
    return {int(r[0]): r[1] for r in rows[1:]}


def _train_defs(path: Path) -> dict[tuple[str, str], frozenset]:
    return {(r[0].strip(), r[1].strip()): _cids(r[2:]) for r in _rows(path)[1:]}


def _train_pairs(dist_path: Path, defs: dict, keep=None) -> list[Pair]:
    out = []
    for ds, m1, m2, y in _rows(dist_path)[1:]:
        ds = ds.strip()
        if ds in TEST_ROWS_IN_POSTCHALLENGE or (keep is not None and ds not in keep):
            continue
        m1, m2 = m1.strip(), m2.strip()
        out.append(Pair(ds, m1, m2, defs[(ds, m1)], defs[(ds, m2)], float(y)))
    return out


def load_train(root: Path | None = None, extra: bool = True) -> list[Pair]:
    """Challenge training set (730 rows) plus, if ``extra``, Ravia 5/6 and Metamer (90 rows).

    The 46 leaderboard pairs (dream2_data/olfboost_repo/data/raw/forms/LeaderboardData_mixturedist.csv) are
    already among the 730 rows with identical labels, so they are not added again.
    """
    pnas = locate("pnas_repo", root)
    base = pnas / "Training_Dataset"
    pairs = _train_pairs(base / "TrainingData_mixturedist.csv",
                         _train_defs(base / "Mixure_Definitions_Training_set.csv"))
    if extra:
        post = pnas / "PostChallenge_Model/Dataset"
        pairs += _train_pairs(post / "TrainingData_mixturedist.csv",
                              _train_defs(post / "Mixure_Definitions_Training_set.csv"),
                              keep=set(EXTRA_TRAIN_DATASETS))
    return pairs


def load_test(root: Path | None = None) -> list[Pair]:
    """T46: the official challenge test set, 46 pairs. Report on it once; never tune on it."""
    base = locate("pnas_repo", root) / "Test_Dataset"
    defs = {r[0].strip(): _cids(r[1:]) for r in _rows(base / "Test_set_Mixure_Definitions.csv")[1:]}
    out = []
    for name, m1, m2, y in _rows(base / "Test_set_mixturedist.csv")[1:]:
        m1, m2 = m1.strip(), m2.strip()
        out.append(Pair("T46", m1, m2, defs[m1], defs[m2], float(y), {"pair_name": name.strip()}))
    return out


def load_validation(root: Path | None = None) -> list[Pair]:
    """V50: 50 pairs, 16 subjects x 2 repetitions. Primary test; never tune on it.

    ``y`` is the mean of all 32 ratings / 100 (the source repo's ``ExpMean_combined``).
    ``extra`` holds ``rep1`` and ``rep2`` (panel mean of each repetition) and ``ratings``
    (subject -> {rep: rating}), all on the 0-1 scale.
    """
    base = locate("pnas_repo", root) / "Validation_Dataset"
    defs = {r[1].strip(): _cids(r[2:]) for r in _rows(base / "Mixure_Definitions_Validation_set.csv")[1:]}
    ratings = defaultdict(lambda: defaultdict(dict))
    for subj, m1, m2, rep, d in _rows(base / "Dream_validation_TestRetest.csv")[1:]:
        key = tuple(sorted((m1.strip(), m2.strip())))       # file lists each pair in both orders
        ratings[key][subj.strip()][int(rep)] = float(d) / 100.0
    out = []
    for (m1, m2), by_subj in sorted(ratings.items()):
        if m1 not in defs or m2 not in defs:                # A_11, B_67, B_70 are undefined
            continue
        rep = {k: statistics.fmean(s[k] for s in by_subj.values() if k in s) for k in (1, 2)}
        allr = [x for s in by_subj.values() for x in s.values()]
        out.append(Pair("V50", m1, m2, defs[m1], defs[m2], statistics.fmean(allr),
                        {"rep1": rep[1], "rep2": rep[2],
                         "ratings": {s: dict(r) for s, r in by_subj.items()}}))
    return out


def load_baseline_predictions(which: str, root: Path | None = None) -> dict[str, dict[tuple[str, str], float]]:
    """Published baseline predictions. ``which`` is "V50" or "T46".

    Returns {model name: {(id_a, id_b): prediction}}; for V50 the key is the sorted mixture-label
    pair, for T46 it is the (Mixture 1, Mixture 2) labels as in the test file.
    """
    pred = locate("pnas_repo", root) / "Predictions"
    fname = {"V50": "Validation_set_Prediction_top6_Teams.csv",
             "T46": "Test_set_Prediction_top6_Teams.csv"}[which]
    rows = _rows(pred / fname)
    head = [h.strip() for h in rows[0]]
    i1, i2 = head.index("Mixture 1"), head.index("Mixture 2")
    out = {}
    for j, h in enumerate(head):
        if not h.startswith("Predicted_Values_"):
            continue
        col = {}
        for r in rows[1:]:
            key = (r[i1].strip(), r[i2].strip())
            col[tuple(sorted(key)) if which == "V50" else key] = float(r[j])
        out[h[len("Predicted_Values_"):]] = col
    return out


def load_t46_sota_predictions(root: Path | None = None) -> dict[str, dict[tuple[str, str], float]]:
    """POM, Snitz angle, semantic, pair and ensemble model predictions on T46.

    The SOTA file has no mixture labels; its rows are in the same order as the top-6 teams file,
    which is checked here through the experimental values both files carry.
    """
    pred = locate("pnas_repo", root) / "Predictions"
    ids = _rows(pred / "Test_set_Prediction_top6_Teams.csv")
    sota = _rows(pred / "Test_set_SOTA_Ensemble_Post_Challenge_Predictions.csv")
    hi, hs = [h.strip() for h in ids[0]], [h.strip() for h in sota[0]]
    i1, i2, ie = hi.index("Mixture 1"), hi.index("Mixture 2"), hi.index("Experimental_Values")
    se = hs.index("Experimental_Values")
    assert len(ids) == len(sota)
    out = {h[len("Predicted_Values_"):]: {} for h in hs if h.startswith("Predicted_Values_")}
    for ri, rs in zip(ids[1:], sota[1:]):
        assert abs(float(ri[ie]) - float(rs[se])) < 1e-6, "SOTA and team files are not row-aligned"
        for j, h in enumerate(hs):
            if h.startswith("Predicted_Values_"):
                out[h[len("Predicted_Values_"):]][(ri[i1].strip(), ri[i2].strip())] = float(rs[j])
    return out


def pearson(x, y) -> float:
    return statistics.correlation(list(x), list(y))
