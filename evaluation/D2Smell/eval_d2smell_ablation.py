# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes
#   O = original training pairs (780)      N = cross-validation predictions pooled over folds and repeats
#   T = T46 pairs (46)                     S = seeds (10)
# ---------------------------------------------------------------------------------------------
"""Ablation study of team D2Smell's mixture model: which of its parts carry the result.

Each variant changes one part of the team's model: which columns it sees (A), how the rows and the
odour columns are made (B), the same steps for a model that sees only the odour differences (C),
and no trained model at all (D). Variants and the checks of the rebuilt baseline are in
evaluation/D2Smell/ablation.py. Two scores, each compared with the full model on the same pairs
(paired bootstrap):
  cross-validation   on the team's 780 training pairs, whole mixtures held out (mixdist.cv); this
                     is the score to draw conclusions from
  T46 (--final)      trained as the team did, all rows and 10 seeds; 46 pairs, so wide intervals

Env:     d2smell (the team's pinned environment, see evaluation/check_d2smell.py)
Usage:   python evaluation/D2Smell/eval_d2smell_ablation.py            cross-validation only
         python evaluation/D2Smell/eval_d2smell_ablation.py --final    also T46 (do this once)
         One worker per GPU with 4 GB free (about 1 h for each of the two parts on two GPUs);
         set CUDA_VISIBLE_DEVICES to choose the GPUs.
Writes:  results/eval_d2smell_ablation/<date_time>[_final].{txt,json}
"""
import argparse
import os

import ablation
import numpy as np
from mixdist import data, evaluate, metrics, teams
from mixdist.data import locate
from mixdist.report import Report

K, REPEATS = 5, 3
REFERENCE = "full"                                           # every other row is compared with this variant
TEAM_FILES = {"team's file: full model": "Test_set_Submission_form_ensemble.csv",
              "team's file: 'non-semantic' (class1)": "Test_set_Submission_form_ensemble_class1.csv",
              "team's file: 'semantic' (class23)": "Test_set_Submission_form_ensemble_class23.csv"}


def describe(rep: Report, source, td: ablation.TeamData, settings_S: list, gpus: list) -> None:
    rep.section("1. Data, checks and settings")
    n_aug = int((td.action_R != ablation.NONE).sum())
    rep.line(f"team code  {source}")
    rep.line(f"rows       {len(td.pairs_O)} original pairs + {n_aug} augmented copies; T46 {len(td.keys_T)} pairs")
    rep.line("columns    " + ", ".join(f"{name} {hi - lo}" for name, (lo, hi) in ablation.BLOCKS.items()))
    difference = ablation.check_features(source, td)
    assert difference < 1e-4, f"the rebuilt T46 matrix does not reproduce the team's predictions ({difference:.1e})"
    rep.line(f"check      the team's 10 saved models, run on the T46 matrix built here, differ from the team's "
             f"prediction file by at most {difference:.1e}")
    for note in td.notes:
        rep.line(f"check      {note}")
    rep.line(f"labels     augmented rows: 'add' -{ablation.ADD_SHIFT}, 'remove' +{ablation.REMOVE_SHIFT} (the team tuned "
             "these per seed and did not save them; middle of its ranges)")
    rep.line(f"GPUs       {', '.join(gpus)}")
    rep.line()
    rep.line("settings of the team's saved models, used here without a new search (cross-validation repeat i uses row i):")
    short = {"n_estimators": "trees", "learning_rate": "rate", "max_depth": "depth", "min_child_weight": "min child",
             "colsample_bytree": "colsample", "reg_lambda": "lambda", "reg_alpha": "alpha"}
    rep.table(["seed"] + [short.get(k, k) for k in ablation.TUNED],
              [[str(seed)] + [s[k] if isinstance(s[k], float) else str(s[k]) for k in ablation.TUNED]
               for seed, s in zip(ablation.SEEDS, settings_S)])
    rep.value("settings", dict(zip(map(str, ablation.SEEDS), settings_S)))


def score_rows(rows: list, y, keep=slice(None)) -> list:
    """rows: [(label, predictions)], the first one is the reference -> [(label, score dict)]."""
    reference = rows[0][1][keep]
    return [(label, metrics.score(pred[keep], y[keep], reference=None if i == 0 or not np.ptp(pred[keep]) else reference))
            for i, (label, pred) in enumerate(rows)]


def report_cv(rep: Report, td: ablation.TeamData, pred_by_key: dict) -> None:
    y_O = evaluate.labels(td.pairs_O)

    def cv(fit_predict) -> evaluate.CVResult:
        return evaluate.cross_validate(td.pairs_O, fit_predict, k=K, repeats=REPEATS)

    base = cv(ablation.constant_fit_predict(y_O))            # also gives the pair, study and overlap of every prediction
    rows = [(ablation.BY_KEY[REFERENCE].label, pred_by_key[REFERENCE])]
    rows += [(v.label, pred_by_key[v.key]) for v in ablation.VARIANTS if v.key != REFERENCE]
    one_line = np.zeros(len(y_O), int)
    rows += [(f"D {label}", cv(evaluate.linear_calibration(dist_O, y_O, one_line)).pred_N)
             for label, (dist_O, _) in ablation.distances(td).items()]
    rows.append(("D constant (mean label of the training pairs)", base.pred_N))

    not_bushdid_N = base.study_N != "Bushdid"
    rep.section(f"3. Cross-validation on the team's {len(y_O)} training pairs ({base.describe()})")
    rep.line("reference = the full model on the same pairs (paired bootstrap). Bushdid is reported apart.")
    for title, keep_N in (("3a. all pairs except Bushdid", not_bushdid_N),
                          ("3b. pairs with no shared molecule, except Bushdid", not_bushdid_N & base.no_overlap_N),
                          ("3c. Bushdid", ~not_bushdid_N)):
        rep.section(title)
        rep.scores(score_rows(rows, base.truth_N, keep_N), label="variant")
    rep.value("cv", {"pair_N": base.pair_N.tolist(), "truth_N": base.truth_N.tolist(), "study_N": base.study_N.tolist(),
                     "no_overlap_N": base.no_overlap_N.tolist(), "pred_N": {label: p.tolist() for label, p in rows}})


def report_t46(rep: Report, source, td: ablation.TeamData, pred_by_key: dict) -> None:
    rep.section(f"4. T46, trained as the team did: all rows, one model per seed, mean of {len(ablation.SEEDS)}")
    earlier = rep.previous_runs(final=True)
    if earlier:
        rep.line(f"NOTE: --final was already run at {', '.join(earlier)}")
    t46 = data.load_test()
    y_T, y_O = evaluate.labels(t46), evaluate.labels(td.pairs_O)
    order_T = [td.keys_T.index((p.id_a, p.id_b)) for p in t46]              # T rows are in the order of the team's form
    seeds_SxT = pred_by_key[REFERENCE][:, order_T]

    rows = [(ablation.BY_KEY[REFERENCE].label, seeds_SxT.mean(axis=0))]
    rows += [(v.label, pred_by_key[v.key][:, order_T].mean(axis=0)) for v in ablation.VARIANTS if v.key != REFERENCE]
    for label, (dist_O, dist_T) in ablation.distances(td).items():
        if dist_T is not None:
            a, c = evaluate.fit_line(dist_O, y_O)
            rows.append((f"D {label}", a * dist_T[order_T] + c))
    rows.append(("D constant (mean label of the training pairs)", np.full(len(y_T), y_O.mean())))
    rows += [(label, teams.read_submission(source / "final_results" / name, t46)) for label, name in TEAM_FILES.items()]

    rep.scores(score_rows(rows, y_T), label="variant")
    rep.line("reference = the full model retrained here (paired bootstrap). Molecule overlap is left out: no T46 pair shares a molecule.")
    rep.section("4b. T46 as the paper reports it: medians over 10,000 bootstrap samples (SI Table S4)")
    rep.table(["variant", "RMSE", "r"], [[label, *metrics.bootstrap_medians(pred, y_T)] for label, pred in rows])

    rep.section("4c. The full model retrained here next to the team's own")
    team_T = dict(rows)["team's file: full model"]
    rep.line(f"our mean of {len(seeds_SxT)} vs the team's file: r between the two sets of predictions "
             f"{metrics.pearson(rows[0][1], team_T):.3f}, largest difference {np.abs(rows[0][1] - team_T).max():.3f}")
    single = np.array([[metrics.pearson(p, y_T), metrics.rmse(p, y_T)] for p in seeds_SxT])
    rep.table(["", "r", "RMSE"], [["mean of the 10 models", metrics.pearson(rows[0][1], y_T), metrics.rmse(rows[0][1], y_T)],
                                 ["single model: lowest r", *single[single[:, 0].argmin()]],
                                 ["single model: median", *np.median(single, axis=0)],
                                 ["single model: highest r", *single[single[:, 0].argmax()]]])
    rep.value("t46_predictions", {label: np.asarray(pred).tolist() for label, pred in rows})   # in the order of data.load_test()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", action="store_true", help="also train as the team did and score T46 (do this once)")
    args = ap.parse_args()
    rep = Report("eval_d2smell_ablation", {"k": K, "repeats": REPEATS, **vars(args)}, tag="final" * args.final)
    source = locate("olfboost_repo", needs="output")
    gpus = ablation.free_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]             # this process only runs the team's saved models

    td = ablation.load_team_data(source)
    describe(rep, source, td, ablation.saved_settings(source), gpus)
    protocols = ("cv", "t46") if args.final else ("cv",)
    rep.section(f"2. Training runs: {len(ablation.VARIANTS)} variants x {len(protocols)} protocol(s)")
    pred = ablation.run_variants(source, protocols, K, REPEATS, gpus, rep.line)
    report_cv(rep, td, pred["cv"])
    if args.final:
        report_t46(rep, source, td, pred["t46"])
    else:
        rep.line()
        rep.line("T46 not scored (pass --final).")
    rep.save()


if __name__ == "__main__":
    main()
