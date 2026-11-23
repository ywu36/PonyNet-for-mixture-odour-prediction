# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = pairs scored   B = bootstrap samples (10,000)
# ---------------------------------------------------------------------------------------------
"""Scoring per PLAN.md rule 6: Pearson with a 10,000-sample bootstrap CI, a paired bootstrap
comparison against a reference model (POM), and the score as a share of the noise ceiling.

    score              the standard result dict used by every script (see mixdist.report)
    bootstrap_medians  median RMSE and r over bootstrap samples, as the paper reports them
    auroc              area under the ROC curve (odour-label checks)
"""
from __future__ import annotations

import numpy as np


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def rmse(x, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, float) - np.asarray(y, float)) ** 2)))


def bootstrap_indices(n: int, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Resampled pair indices -> idx_BxP."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_r(pred_P, y_P, idx_BxP) -> np.ndarray:
    """Pearson r on every bootstrap sample -> r_B (nan where a sample is constant)."""
    p, t = pred_P[idx_BxP], y_P[idx_BxP]                      # B x P
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    den = np.sqrt((p * p).sum(axis=1) * (t * t).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (p * t).sum(axis=1) / den


def score(pred, y, reference=None, ceiling: float | None = None, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Pearson r, RMSE, 95% percentile bootstrap CI over pairs; optional paired test vs ``reference``.

    The paired test resamples the same pairs for both models; ``p_vs_ref`` is two-sided.
    """
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    idx = bootstrap_indices(len(y), n_boot, seed)
    b = bootstrap_r(pred, y, idx)
    out = {"n": len(y), "r": pearson(pred, y), "rmse": rmse(pred, y),
           "r_ci": tuple(np.nanpercentile(b, [2.5, 97.5]))}
    if ceiling is not None:
        out["pct_ceiling"] = 100.0 * out["r"] / ceiling
    if reference is not None:
        reference = np.asarray(reference, float)
        d = b - bootstrap_r(reference, y, idx)
        d = d[np.isfinite(d)]
        out["r_ref"] = pearson(reference, y)
        out["diff_ci"] = tuple(np.percentile(d, [2.5, 97.5]))
        out["p_vs_ref"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return out


def auroc(y_true, score) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties counted half); nan if only one class."""
    y_true, score = np.asarray(y_true, bool), np.asarray(score, float)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):                                    # average ranks over tied scores
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_medians(pred, y, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """(median RMSE, median Pearson r) over bootstrap samples of the pairs (paper, SI Table S4)."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    idx = bootstrap_indices(len(y), n_boot, seed)
    rmse_B = np.sqrt(((pred[idx] - y[idx]) ** 2).mean(axis=1))
    return float(np.median(rmse_B)), float(np.nanmedian(bootstrap_r(pred, y, idx)))


def fmt(name: str, s: dict) -> str:
    """One-line summary of a score dict (tables are printed by mixdist.report)."""
    t = f"{name}: r = {s['r']:.3f} [95% CI {s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}], RMSE {s['rmse']:.3f}, n = {s['n']}"
    if "pct_ceiling" in s:
        t += f", {s['pct_ceiling']:.0f}% of ceiling"
    if "r_ref" in s:
        t += (f" | reference r = {s['r_ref']:.3f}, difference CI [{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}],"
              f" p = {s['p_vs_ref']:.3f}")
    return t
