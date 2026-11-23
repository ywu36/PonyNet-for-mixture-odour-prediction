# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes
#   P = pairs in the set being processed
#   N = CV predictions pooled over folds and repeats
# ---------------------------------------------------------------------------------------------
"""The evaluation protocol shared by all experiments (PLAN.md rules 5 and 6).

1. Cross-validation on the training set only, holding out whole mixtures (mixdist.cv). Any model
   plugs in as a ``fit_predict(train_idx, test_idx) -> predictions`` function.
2. Held-out sets V50 and T46, scored once with the model fitted on all training pairs, against
   the published POM predictions (paired bootstrap) and, for V50, the test-retest noise ceiling.

Calibration schemes (rule 5), for models that end in a line y = a*x + c:
  shared   one line for all studies (default; what the organisers and D2Smell did)
  bushdid  Bushdid gets its own line; V50 and T46 use the line of the other studies
  type     one line for triangle-test studies, one for similarity-rating studies; T46 (triangle)
           and V50 (rating) each use the line of their own type
  overlap  one line for pairs whose two mixtures share no molecule, one for pairs that share at
           least one; decided per pair from the mixture definitions (inputs, no labels). Every V50
           and T46 pair shares no molecule
For a held-out set whose pairs all use one line, Pearson r is the same under every scheme; only
RMSE changes.
The CV table pools all pairs; the only subset shown is pairs sharing no molecule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import cv, data, metrics

FitPredict = Callable[[np.ndarray, np.ndarray], np.ndarray]
CV_REPEATS = 50          # standard CV (evaluation/CV.md): smallest of 10/20/50 that holds every pair out at least once


def labels(pairs) -> np.ndarray:
    """Human perceptual distance per pair -> y_P."""
    return np.array([p.y for p in pairs], dtype=float)


CALIBRATIONS = ("shared", "bushdid", "type", "overlap")
# SI Table S2. Ravia 5, Ravia 6 and Metamer are not listed there and count as rating.
TRIANGLE_STUDIES = frozenset({"Bushdid", "Ravia 4", "T46"})


def calibration_group(dataset: str, scheme: str) -> int:
    """Which calibration line a study (training study name, or "V50" / "T46") uses. Not for
    "overlap", which is decided per pair (calibration_groups)."""
    if scheme == "shared":
        return 0
    if scheme == "bushdid":
        return int(dataset == "Bushdid")
    if scheme == "type":
        return int(dataset in TRIANGLE_STUDIES)
    raise ValueError(f"unknown calibration scheme {scheme!r}")


def calibration_groups(pairs, scheme: str) -> np.ndarray:
    """Calibration group per pair -> group_P (overlap: 1 = the two mixtures share no molecule)."""
    if scheme == "overlap":
        return np.array([int(p.jaccard == 0) for p in pairs])
    return np.array([calibration_group(p.dataset, scheme) for p in pairs])


def fit_line(x_P, y_P) -> tuple[float, float]:
    """Least-squares y = a*x + c -> (a, c)."""
    a, c = np.polyfit(x_P, y_P, 1)
    return float(a), float(c)


def linear_calibration(x_P, y_P, group_P) -> FitPredict:
    """fit_predict for a one-number-per-pair model (e.g. a distance): one line y = a*x + c per
    group label in group_P, fitted on the training indices of that group."""
    x_P, y_P, group_P = map(np.asarray, (x_P, y_P, group_P))

    def fit_predict(train_idx, test_idx):
        pred = np.empty(len(test_idx))
        for g in np.unique(group_P):
            fit_idx = train_idx[group_P[train_idx] == g]
            in_g = group_P[test_idx] == g
            if in_g.any():
                a, c = fit_line(x_P[fit_idx], y_P[fit_idx])
                pred[in_g] = a * x_P[test_idx[in_g]] + c
        return pred

    return fit_predict


@dataclass
class CVResult:
    """Pooled test-fold predictions; entry n belongs to training pair pair_N[n]."""
    pair_N: np.ndarray
    pred_N: np.ndarray
    truth_N: np.ndarray
    study_N: np.ndarray
    no_overlap_N: np.ndarray                                 # the pair's two mixtures share no molecule
    k: int
    repeats: int
    seed: int

    def describe(self) -> str:
        return f"{self.k} folds x {self.repeats} repeats, whole mixtures held out, seed {self.seed}"

    def score_rows(self) -> list[tuple[str, dict]]:
        """Standard CV table: all pooled test pairs, and the subset whose two mixtures share no
        molecule (the kind of pair V50 and T46 contain; decided from the inputs). No study is
        reported on its own: subsets picked after seeing results invite look-ahead bias."""
        subsets = [("all pairs", np.ones(len(self.pred_N), bool)),
                   ("  of which share no molecule", self.no_overlap_N)]
        return [(name, metrics.score(self.pred_N[m], self.truth_N[m])) for name, m in subsets]


def cross_validate(pairs, fit_predict: FitPredict, k: int = 5, repeats: int = CV_REPEATS, seed: int = 0) -> CVResult:
    """Run fit_predict on every zero-overlap fold and pool the test predictions."""
    y_P = labels(pairs)
    pair_idx, pred = [], []
    for fold in cv.mixture_holdout_folds(pairs, k=k, repeats=repeats, seed=seed):
        train_idx, test_idx = np.array(fold.train), np.array(fold.test)
        pair_idx.append(test_idx)
        pred.append(np.asarray(fit_predict(train_idx, test_idx), dtype=float))
    pair_N = np.concatenate(pair_idx)
    return CVResult(pair_N, np.concatenate(pred), y_P[pair_N],
                    np.array([pairs[i].dataset for i in pair_N]),
                    np.array([pairs[i].jaccard == 0 for i in pair_N]), k, repeats, seed)


@dataclass
class HeldOutSet:
    name: str
    pairs: list
    y_P: np.ndarray
    reference_P: np.ndarray                                  # published POM predictions
    ceiling: float | None                                    # test-retest Pearson (V50 only)
    note: str = ""


def held_out_sets() -> list[HeldOutSet]:
    """V50 (primary) and T46. Load only when scoring once at the end (rule 6)."""
    v50 = data.load_validation()
    pom_v = data.load_baseline_predictions("V50")["POM"]
    t46 = data.load_test()
    pom_t = data.load_t46_sota_predictions()["POM"]
    return [
        HeldOutSet("V50", v50, labels(v50), np.array([pom_v[(p.id_a, p.id_b)] for p in v50]),
                   metrics.pearson([p.extra["rep1"] for p in v50], [p.extra["rep2"] for p in v50]),
                   "published POM column = POM design targets, only its r is meaningful"),
        HeldOutSet("T46", t46, labels(t46), np.array([pom_t[(p.id_a, p.id_b)] for p in t46]), None),
    ]


def score_held_out(hs: HeldOutSet, pred_P) -> dict:
    return metrics.score(pred_P, hs.y_P, reference=hs.reference_P, ceiling=hs.ceiling)


# ---------------------------------------------------------------------------------------------
# CV scoring, protocol of 2026-11-17 (evaluation/CV.md). Added alongside the older pooled scoring
# above, which stays for reproducing earlier results.
# ---------------------------------------------------------------------------------------------


def cv_weights(res: CVResult) -> np.ndarray:
    """Weight per pooled prediction = 1 / (times its pair was held out) -> w_N; every pair counts once."""
    times = np.bincount(res.pair_N)
    return 1.0 / times[res.pair_N]


def _weighted_r(p, t, w):
    """Weighted Pearson r; p, t: (N,), w: (B, N) or (N,) -> r_B or scalar."""
    w = np.atleast_2d(w)
    s = w.sum(1, keepdims=True)
    pm, tm = (w * p).sum(1, keepdims=True) / s, (w * t).sum(1, keepdims=True) / s
    pc, tc = p - pm, t - tm
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (w * pc * tc).sum(1) / np.sqrt((w * pc * pc).sum(1) * (w * tc * tc).sum(1))
    return r if r.size > 1 else float(r[0])


def _weighted_rmse(p, t, w):
    w = np.atleast_2d(w)
    v = np.sqrt((w * (p - t) ** 2).sum(1) / w.sum(1))
    return v if v.size > 1 else float(v[0])


def score_cv(pairs, res: CVResult, reference: CVResult | None = None, mask_N=None,
             n_boot: int = 2000, seed: int = 0) -> dict:
    """r and RMSE with every pair weighted once; 95% CI by resampling mixtures (pairs that share a
    mixture are not independent): a draw takes |mixtures| mixtures with replacement and weights a
    pair by count(A) * count(B). With ``reference`` (same folds), the paired change in r and RMSE."""
    m = np.ones(len(res.pred_N), bool) if mask_N is None else np.asarray(mask_N, bool)
    w = cv_weights(res)[m]
    pN, tN, pairN = res.pred_N[m], res.truth_N[m], res.pair_N[m]
    mixes = sorted({x for p in pairs for x in (p.a, p.b)}, key=sorted)
    idx = {x: i for i, x in enumerate(mixes)}
    ia = np.array([idx[pairs[i].a] for i in pairN]); ib = np.array([idx[pairs[i].b] for i in pairN])
    out = {"pairs": int(len(np.unique(pairN))), "r": _weighted_r(pN, tN, w), "rmse": _weighted_rmse(pN, tN, w)}
    if reference is not None:
        assert np.array_equal(reference.pair_N, res.pair_N), "reference must use the same folds"
        rN = reference.pred_N[m]
        out["r_ref"], out["rmse_ref"] = _weighted_r(rN, tN, w), _weighted_rmse(rN, tN, w)
    rng = np.random.default_rng(seed)
    r_b, d_b, e_b = [], [], []
    for start in range(0, n_boot, 100):
        c = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=min(100, n_boot - start))
        wb = w * c[:, ia] * c[:, ib]
        r = _weighted_r(pN, tN, wb); r_b.append(r)
        if reference is not None:
            d_b.append(r - _weighted_r(rN, tN, wb))
            e_b.append(_weighted_rmse(pN, tN, wb) - _weighted_rmse(rN, tN, wb))
    out["r_ci"] = tuple(np.nanpercentile(np.concatenate(r_b), [2.5, 97.5]))
    if reference is not None:
        d = np.concatenate(d_b); d = d[np.isfinite(d)]
        out["r_change"], out["r_change_ci"] = out["r"] - out["r_ref"], tuple(np.percentile(d, [2.5, 97.5]))
        out["rmse_change"], out["rmse_change_ci"] = out["rmse"] - out["rmse_ref"], tuple(np.percentile(np.concatenate(e_b), [2.5, 97.5]))
        out["p_vs_ref"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out
