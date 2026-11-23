"""Frozen-encoder benchmark: B1a recipe on open-POM, MIST and Uni-Mol2, nothing trained but the line.

Each row: per-molecule vector (odour-word probabilities, or embedding) -> per-dimension quantile ->
mean per mixture -> length 1 -> Euclidean -> one line (2 numbers). Uni-Mol2 is frozen (public 84M
weights) and has no odour words, so only its embedding is used.
  1. Standard CV on the 820 training pairs (evaluation/CV.md): all pairs and pairs sharing no
     molecule; change from B1a open-POM on the same folds, 95% CI by resampling mixtures.
  2. V50 and T46 (requested by the user, 2026-11-17): line fitted on all 820 training pairs, scored
     once; r with 95% CI, RMSE, and the paired comparison with the published POM predictions.

Env:     mist
Usage:   python evaluation/B1/eval_unimol2_b1a.py
Writes:  results/eval_unimol2_b1a/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import data, evaluate, features, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    rep = Report("eval_unimol2_b1a", {"repeats": evaluate.CV_REPEATS})
    train = data.load_train()
    y_P = evaluate.labels(train)
    zero = np.zeros(len(train))
    fs = {m: features.load(m) for m in ("openpom", "mist", "unimol2")}
    inputs = {"B1a open-POM words": ("openpom", "probs"), "B1a open-POM embedding": ("openpom", "emb"),
              "B1a MIST words": ("mist", "probs"), "B1a MIST embedding": ("mist", "emb"),
              "B1a Uni-Mol2 embedding": ("unimol2", "emb")}
    feat = {k: (fs[m].cids_M, fs[m].probs_MxW if kind == "probs" else fs[m].emb_MxE) for k, (m, kind) in inputs.items()}

    rep.section(f"1. Standard CV ({evaluate.CV_REPEATS} x 5, every pair once, CI by resampling mixtures)")
    res = {k: evaluate.cross_validate(train, evaluate.linear_calibration(mixture.paper_distance(train, c, X), y_P, zero))
           for k, (c, X) in feat.items()}
    ref = res["B1a open-POM words"]
    rows = []
    for subset, m in [("all pairs", None), ("share no molecule", ref.no_overlap_N)]:
        for k in inputs:
            s = evaluate.score_cv(train, res[k], reference=None if k == "B1a open-POM words" else ref, mask_N=m)
            rows.append([k, subset, s["pairs"], f"{s['rmse']:.3f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         "" if k == "B1a open-POM words" else f"{s['rmse_change']:+.3f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a open-POM: RMSE / r [95% CI]"], rows)

    rep.section("2. V50 and T46: line fitted on all 820 training pairs, scored once")
    rows = []
    for hs in evaluate.held_out_sets():
        for k, (c, X) in feat.items():
            a, b = evaluate.fit_line(mixture.paper_distance(train, c, X), y_P)
            pred = a * mixture.paper_distance(hs.pairs, c, X) + b
            s = evaluate.score_held_out(hs, pred)
            rows.append([hs.name, k, s["n"], f"{s['rmse']:.3f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['r_ref']:.3f}", f"[{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]"])
    rep.table(["set", "model", "pairs", "RMSE", "r [95% CI]", "published POM r", "r - POM r (95% CI)"], rows)
    rep.line("V50: the published POM column is the POM design target; only its r is meaningful.")
    rep.save()


if __name__ == "__main__":
    main()
