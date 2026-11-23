"""Check our data loading and scoring against the paper's published scores.

Recomputes the published models' scores from the authors' prediction files and prints them next
to the paper's values (Satarifard et al., PNAS 2026, SI Table S4: medians over 10,000 bootstrap
samples). Nothing is trained or run: this only tests our loaders and metrics. It does not prove
the published models work (their predictions are taken as given).

Env:     mist (numpy only)
Usage:   python paper/check_paper_scores.py
Writes:  results/check_paper_scores/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data, evaluate, metrics  # noqa: E402
from mixdist.report import Report  # noqa: E402

# (set, model): (median RMSE, median Pearson r) from SI Table S4
PAPER = {
    ("V50", "POM"): (0.150, 0.501), ("V50", "Ensemble"): (0.126, 0.475),
    ("T46", "POM"): (0.114, 0.397), ("T46", "Ensemble"): (0.078, 0.569),
    ("T46", "Snitz Angle"): (0.124, 0.340), ("T46", "Semantic Model"): (0.120, 0.268),
    ("T46", "Pair Model"): (0.107, 0.415),
    ("T46", "D2Smell"): (0.081, 0.469), ("T46", "UMich CASI"): (0.088, 0.511),
    ("T46", "ChemSenSim Lab"): (0.092, 0.512), ("T46", "belfaction"): (0.088, 0.428),
    ("T46", "PL21"): (0.086, 0.346), ("T46", "Tamarin"): (0.094, 0.402),
}
PAPER_V50_CEILING = 0.742


def main():
    rep = Report("check_paper_scores")
    v50, t46 = data.load_validation(), data.load_test()
    sets = {"V50": (v50, data.load_baseline_predictions("V50")),
            "T46": (t46, {**data.load_baseline_predictions("T46"), **data.load_t46_sota_predictions()})}

    rep.section("Published predictions rescored (bootstrap medians) vs SI Table S4")
    rows, worst = [], 0.0
    for (name, model), (paper_rmse, paper_r) in PAPER.items():
        pairs, preds = sets[name]
        pred_P = [preds[model][(p.id_a, p.id_b)] for p in pairs]
        rmse, r = metrics.bootstrap_medians(pred_P, evaluate.labels(pairs))
        diff = max(abs(rmse - paper_rmse), abs(r - paper_r))
        worst = max(worst, diff)
        rows.append([f"{name} {model}", rmse, paper_rmse, r, paper_r, diff])
    rep.table(["set and model", "RMSE ours", "paper", "r ours", "paper", "max |diff|"], rows)

    ceiling = metrics.pearson([p.extra["rep1"] for p in v50], [p.extra["rep2"] for p in v50])
    rep.line()
    rep.line(f"V50 test-retest Pearson: ours {ceiling:.3f}, paper {PAPER_V50_CEILING:.3f}")
    rep.line(f"largest difference from the paper over all rows: {worst:.3f}")
    rep.value("largest_difference", worst)
    rep.save()


if __name__ == "__main__":
    main()
