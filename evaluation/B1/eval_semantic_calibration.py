"""B1 calibration comparison (rule 5): one line for all studies, a separate Bushdid line, or one
line per measurement type (triangle test / similarity rating), for each frozen model.
Cross-validation on the training set only; V50 and T46 are never scored here.

Shared steps in evaluation/B1/semantic.py; schemes in mixdist.evaluate.CALIBRATIONS.

Env:     mist
Usage:   python evaluation/B1/eval_semantic_calibration.py
Writes:  results/eval_semantic_calibration/<date_time>.{txt,json}
"""
import semantic
from mixdist import evaluate, metrics
from mixdist.report import Report

MODELS = {"mist": "semantic_mist_odour", "openpom": "semantic_openpom"}


def main():
    args = semantic.parse_args(__doc__, final=False)
    rep = Report("eval_semantic_calibration", vars(args))
    rows = []
    for model, name in MODELS.items():
        _, train, dist_P, y_P = semantic.load(model, rep, args.recompute_embeddings)
        for calibration in evaluate.CALIBRATIONS:
            res = semantic.cross_validate(train, dist_P, y_P, calibration)
            nb_N = res.study_N != "Bushdid"
            rows.append([name, calibration,
                         metrics.pearson(res.pred_N[nb_N], res.truth_N[nb_N]), metrics.rmse(res.pred_N[nb_N], res.truth_N[nb_N]),
                         metrics.pearson(res.pred_N[~nb_N], res.truth_N[~nb_N]), metrics.rmse(res.pred_N[~nb_N], res.truth_N[~nb_N])])
    rep.section(f"2. Cross-validation on the training set ({res.describe()})")
    rep.table(["run", "calibration", "r non-Bushdid", "RMSE non-Bushdid", "r Bushdid", "RMSE Bushdid"], rows)
    rep.line()
    rep.line("shared = one line; bushdid = Bushdid own line; type = triangle (Bushdid, Ravia 4) vs rating")
    rep.save()


if __name__ == "__main__":
    main()
