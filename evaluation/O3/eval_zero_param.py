# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   W = words (138)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Backbones alone on DREAM: NOTHING fitted on DREAM (user 2026-11-20: "show me backbone only performance on dream, no
parameter trained on dream"). Prediction = distance between the two mixture vectors; r needs no fitted number.
Backbones are frozen and trained on GS-LF single molecules only.
  open-POM baseline (B1a): 138 word probabilities, quantile per word (over the 235 molecules, no labels), mean over the
      mixture, length 1, Euclidean.
  plain O3 (o3m_w64), 10 members: unit merge (molecule vectors length 1), equal weights, mixture length 1, Euclidean;
      distance averaged over members.
  O5 (o3m_attpool), 10 members: same, equal weights | weights = exp(O5's own molecule strength), no scale fitted.
r: Pearson of distance with the label (0 numbers). RMSE: needs a line (slope, offset); fitted on the 820 training
pairs only, shown separately (2 numbers). Pairs: T46, V50, and the 514 development pairs (no fitting, so no folds).

Env:     mist
Usage:   python evaluation/O3/eval_zero_param.py
Writes:  results/eval_zero_param/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402


def mix_dist(U, A, B, w=None):
    """U: molecule vectors (M x D); A, B: membership (P x M); w: molecule weights (M) or None = equal."""
    w = np.ones(len(U)) if w is None else w
    def z(Mx):
        Z = (Mx * w) @ U / (Mx * w).sum(1, keepdims=True); return Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return np.linalg.norm(z(A) - z(B), axis=1)


def main():
    rep = Report("eval_zero_param", {"members": len(eo.SEEDS), "device": pick_device()})
    _, BB = eo.backbones(pick_device(), rep); op = features.load("openpom")
    Q = mixture.quantile_per_word(op.probs_MxW)
    unit = lambda V: V / np.linalg.norm(V, axis=1, keepdims=True)  # noqa: E731
    models = {
        "open-POM baseline (B1a)": lambda A, B: mix_dist(Q, A, B),
        "plain O3, equal weights": lambda A, B: np.mean([mix_dist(unit(m["V"]), A, B) for m in BB["plain"]], 0),
        "O5, equal weights": lambda A, B: np.mean([mix_dist(unit(m["V"]), A, B) for m in BB["o5"]], 0),
        "O5, own atom attention as weights": lambda A, B: np.mean(
            [mix_dist(unit(m["V"]), A, B, np.exp(m["strength"] - m["strength"].max())) for m in BB["o5"]], 0)}
    train = data.load_train(); At, Bt = membership(train, op.cids_M); yt = evaluate.labels(train)
    sets = [(h.name, list(h.pairs), h.y_P) for h in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46")]
    P = dev_pairs(); sets.append(("514 development pairs", P, evaluate.labels(P)))
    rows = []
    for nm, f in models.items():
        dt = f(At, Bt); slope, off = np.polyfit(dt, yt, 1)
        for sn, pairs, y in sets:
            A, B = membership(pairs, op.cids_M); d = f(A, B); s = metrics.score(d, y)
            rows.append([nm, sn, len(y), f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{metrics.rmse(slope * d + off, y):.4f}"])
    rep.section("No number fitted on DREAM: r of mixture distance with the label; RMSE after a line fitted on the 820 training pairs")
    rep.table(["backbone", "pairs", "n", "r [95% CI] (0 DREAM numbers)", "RMSE (2-number line)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
