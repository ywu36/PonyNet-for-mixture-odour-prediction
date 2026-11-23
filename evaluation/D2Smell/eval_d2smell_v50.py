# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   R = training rows (20,578)        O = original training pairs (780)     V = V50 pairs (50)
#   T = T46 pairs (46)                X = mixtures of the team's profile file (20,740)
#   M = V50 mixtures (76)             W = odour words kept by the team (25)  S = seeds (10)
# ---------------------------------------------------------------------------------------------
"""Team D2Smell's model on V50: every row of findings/finding_2.html, as a change from the full model.

Variants are those of evaluation/D2Smell/ablation.py that finding 2 shows, trained as the team
trained for T46 (all rows, one model per seed on the last of its ten 90% splits, mean of 10 seeds),
plus three rows without training of a new model: one seed instead of the mean of ten (median over
the 10 seeds), and the network profiles through B1a and through B1 with one line on the 780 pairs.
V50 rows are built with the team's own feature functions, as its T46 rows. The odour profiles of
the V50 mixtures come from evaluation/D2Smell/odorpair_profiles.py (the team's network, not in its
file). The team's quantile transform (twice, over its 20,740 mixtures, which include T46) is fitted
on those mixtures only and applied to V50; the team refitted it with its test mixtures included.

Models are saved to checkpoints/d2smell_ablation/<variant>/ and reused by later runs, so training
never sees a V50 label. Without --final, random numbers stand in for the V50 labels (dry run).

Env:     d2smell (the team's pinned environment, see evaluation/check_d2smell.py); GPU
Usage:   python evaluation/D2Smell/odorpair_profiles.py               first, env odorpair
         python evaluation/D2Smell/eval_d2smell_v50.py                 train or load, dry-run scores
         python evaluation/D2Smell/eval_d2smell_v50.py --final         score V50 (do this once)
         --retrain trains again even if checkpoints exist (about 1.5 h on one GPU for all variants)
         --variants full odour ...   only these (to train on two GPUs at once, before one --final run)
Writes:  checkpoints/d2smell_ablation/<variant>/seed_<seed>.json,
         results/eval_d2smell_v50/<date_time>[_final].{txt,json}
"""
import argparse
import csv
import json
import os

import ablation
import numpy as np
import pandas as pd
from mixdist import data, evaluate, metrics, mixture
from mixdist.data import locate
from mixdist.report import REPO, Report, results_dir

LABEL = {                                                    # key of ablation.BY_KEY: row of finding 2, in its order
    "full": "full model (baseline)",
    "no_ratings": "A ratings removed",
    "identity": "A identity only (no odour)",
    "ratings": "A ratings only",
    "odour": "A odour differences only",
    "team_semantic": "A the team's 'semantic' split",
    "no_aug": "B no augmentation",
    "aug_same_label": "B augmented rows keep the parent's label",
    "no_quantile": "B no quantile transform",
    "words_33": "B all 33 odour words",
    "default_settings": "B default XGBoost settings",
    "odour_linear": "C least squares on the odour differences"}
VARIANTS = tuple(LABEL)
ONE_SEED = "B one seed, not the mean of ten (median over seeds)"
B1A, B1 = "C no XGBoost: B1a on the profiles + line", "C no XGBoost: B1 (cosine) on the profiles + line"
TEAM = "team's published V50 predictions"
T46_LABEL = {**{LABEL[k]: ablation.BY_KEY[k].label for k in VARIANTS},
             B1A: "D network profile, quantile, length 1, Euclidean + line (B1a)",
             B1: "D network profile, cosine + line (B1)", TEAM: "team's file: full model"}
CHECKPOINTS = REPO / "checkpoints/d2smell_ablation"
PROFILES = REPO / "features/odorpair_profiles.csv"


def quantile_twice(fit_XxW: np.ndarray):
    """The team's applying_2QuantileTransformer as two fitted transforms -> function(x) for new rows."""
    from sklearn.preprocessing import QuantileTransformer
    make = lambda: QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=0)  # noqa: E731
    q1 = make().fit(fit_XxW)
    q2 = make().fit(q1.transform(fit_XxW))
    return lambda x: q2.transform(q1.transform(x))


def v50_pairs(pnas) -> tuple:
    """(Mixture 1, Mixture 2) in the published file's order, both mixtures defined, and the team's
    published prediction for each. Label columns of that file are not read."""
    with open(pnas / "Predictions/Validation_set_Prediction_top6_Teams.csv", newline="") as f:
        rows = [(r["Mixture 1"].strip(), r["Mixture 2"].strip(), float(r["Predicted_Values_D2Smell"])) for r in csv.DictReader(f)]
    return [(a, b) for a, b, _ in rows], np.array([p for _, _, p in rows])


def build_v50(rep: Report, source, td: ablation.TeamData, keys_V: list) -> tuple:
    """V50 feature rows (V x 8240, blocks as ablation.BLOCKS) and the profiles of each pair:
    quantile (for B1a) and raw (for B1), {"a_q", "b_q", "a_raw", "b_raw"}: V x W each."""
    team_utils, team_train = ablation._team_code(source)
    prof = pd.read_csv(source / "data/processed/predictions_separated_mean_33_Augmentation_Dataset.csv")
    words = [c for c in prof.columns if c.startswith("Prediction_")]
    kept = [c for c in words if int(c.split("_")[1]) not in ablation.DROPPED_WORDS]
    transform, transform_33 = quantile_twice(prof[kept].values), quantile_twice(prof[words].values)
    team_q_XxW = team_utils.applying_2QuantileTransformer(prof[["Dataset", "Mixture Label"] + kept].copy())[kept].values
    rep.line(f"check  quantile transform refitted here vs the team's on its {len(prof)} mixtures: largest difference "
             f"{np.abs(transform(prof[kept].values) - team_q_XxW).max():.1e}")

    ours = pd.read_csv(PROFILES, dtype={"Mixture Label": str})
    test = ours[ours["Dataset"] == "Test"].set_index("Mixture Label")[kept]
    team_test = prof[prof["Dataset"] == "Test"].astype({"Mixture Label": str}).set_index("Mixture Label")[kept].loc[test.index]
    rep.line(f"check  T46 profiles recomputed with the network vs the team's file: largest difference "
             f"{np.abs(test.values - team_test.values).max():.1e}")
    v50 = ours[ours["Dataset"] == "V50"].reset_index(drop=True)
    raw_MxW = v50[kept].values
    q_MxW = transform(raw_MxW)
    percept = pd.concat([pd.DataFrame({"Dataset": "Test", "Mixture Label": v50["Mixture Label"]}),
                         pd.DataFrame(q_MxW, columns=kept)], axis=1)          # the team's code looks up 'Test'
    defs = pd.read_csv(locate("pnas_repo") / "Validation_Dataset/Mixure_Definitions_Validation_set.csv")
    form = pd.DataFrame({"Mixture_1": [a for a, _ in keys_V], "Mixture_2": [b for _, b in keys_V]})
    all_cids = sorted(pd.read_csv(source / "data/processed/CID.csv")["CID"].astype(int).tolist())
    ratings = pd.read_csv(source / "output/percept_scaled.csv")
    with ablation._quiet():
        x_Vx433 = team_train.prepare_test_data(form, percept, defs, all_cids)
    x_Vx8240 = np.zeros((len(keys_V), ablation.BLOCKS["odour_33"][1]), np.float32)
    ablation._expand_into(x_Vx8240, np.arange(len(keys_V)), team_utils, x_Vx433, ratings, list(range(1, ratings.shape[1])))

    row_of = {m: i for i, m in enumerate(v50["Mixture Label"])}
    ia_V, ib_V = np.array([row_of[a] for a, _ in keys_V]), np.array([row_of[b] for _, b in keys_V])
    lo, hi = ablation.BLOCKS["odour"]
    assert np.allclose(x_Vx8240[:, lo:hi], (q_MxW[ia_V] - q_MxW[ib_V]) ** 2, atol=1e-4), "odour block differs"
    lo, hi = ablation.BLOCKS["odour_raw"]
    x_Vx8240[:, lo:hi] = (raw_MxW[ia_V] - raw_MxW[ib_V]) ** 2
    q33_Mx33 = transform_33(v50[words].values)
    lo, hi = ablation.BLOCKS["odour_33"]
    x_Vx8240[:, lo:hi] = (q33_Mx33[ia_V] - q33_Mx33[ib_V]) ** 2

    in_file = defs.set_index("Mixture Label").iloc[:, 1:].notna().sum(axis=1)
    ident = np.concatenate([x_Vx8240[:, slice(*ablation.BLOCKS["identity_head"])],
                            x_Vx8240[:, slice(*ablation.BLOCKS["identity_tail"])]], axis=1)
    marked_V = ident[:, :204].sum(1)
    defined_V = np.array([in_file[a] for a, _ in keys_V])
    rep.line(f"check  identity columns mark {int((marked_V == defined_V).sum())} of {len(keys_V)} first mixtures completely "
             "(the V50 file has an index column, so the team's 2-column skip does not drop a molecule here, unlike T46)")
    seen = {c for p in td.pairs_O for c in p.a | p.b}
    v_mols = {int(c) for c in defs.iloc[:, 2:].stack().astype(int)}
    rep.line(f"V50     {len(keys_V)} pairs, {len(v50)} mixtures, {len(v_mols)} molecules, of which "
             f"{len(v_mols - seen)} not in any training pair")
    return x_Vx8240, {"a_q": q_MxW[ia_V], "b_q": q_MxW[ib_V], "a_raw": raw_MxW[ia_V], "b_raw": raw_MxW[ib_V]}


class _Linear:
    """Least squares saved as plain numbers, with the save_model / load_model / predict of XGBoost."""
    def __init__(self, model=None):
        self.coef, self.intercept = (model.coef_, model.intercept_) if model is not None else (None, None)

    def save_model(self, path):
        path.write_text(json.dumps({"coef": np.asarray(self.coef).tolist(), "intercept": float(self.intercept)}))

    def load_model(self, path):
        saved = json.loads(path.read_text())
        self.coef, self.intercept = np.array(saved["coef"]), saved["intercept"]

    def predict(self, x):
        return x @ self.coef + self.intercept


def train_or_load(rep: Report, td: ablation.TeamData, x_Vx8240, retrain: bool, keys=VARIANTS) -> tuple:
    """Per variant -> (pred_SxV, pred_SxT): ten seed models, trained as the team did and saved, or loaded."""
    import xgboost as xgb
    from sklearn.model_selection import KFold
    settings_S = ablation.saved_settings(locate("olfboost_repo", needs="output"))
    out_V, out_T = {}, {}
    for key in keys:
        variant, folder = ablation.BY_KEY[key], CHECKPOINTS / key
        linear = variant.model == "linear"
        cols = td.columns(variant.blocks)
        paths = [folder / f"seed_{seed}.json" for seed in ablation.SEEDS]
        trained = retrain or not all(p.exists() for p in paths)
        if trained:
            folder.mkdir(parents=True, exist_ok=True)
            x_RxF = td.x_Rx8240[:, cols]
            rows, y = ablation.training_rows(td, np.arange(len(td.pairs_O)), variant.augment, team_order=True)
            for seed, settings, path in zip(ablation.SEEDS, settings_S, paths):
                part = list(KFold(n_splits=10, shuffle=True, random_state=seed).split(rows))[-1][0]
                model = ablation.fit(x_RxF[rows[part]], y[part], variant, settings)
                (_Linear(model) if linear else model).save_model(path)
            del x_RxF
            (folder / "about.json").write_text(json.dumps({"variant": variant.label, "blocks": variant.blocks,
                                                           "columns": len(cols), "seeds": ablation.SEEDS,
                                                           "settings": settings_S}, indent=1))
        models = []
        for path in paths:
            m = _Linear() if linear else xgb.XGBRegressor()
            m.load_model(path)
            models.append(m)
        out_V[key] = np.array([m.predict(x_Vx8240[:, cols]) for m in models])
        out_T[key] = np.array([m.predict(td.x_Tx8240[:, cols]) for m in models])
        rep.line(f"{LABEL[key]:<28} {'trained and saved' if trained else 'loaded'}  {folder.relative_to(REPO)}/")
    return out_V, out_T


def check_against_ablation(rep: Report, td: ablation.TeamData, pred_T: dict) -> tuple:
    """T46 predictions of the saved models vs those of the recorded ablation run; returns that run's
    T46 change table rows {label: score dict} for the side-by-side view."""
    files = sorted((results_dir() / "eval_d2smell_ablation").glob("*_final.json")) or \
        sorted((REPO / "results/eval_d2smell_ablation").glob("*_final.json"))     # a dry run writes elsewhere
    run = json.loads(files[-1].read_text())
    saved = next(i["value"] for i in run["items"] if i["type"] == "value" and i["key"] == "t46_predictions")
    order_T = [td.keys_T.index((p.id_a, p.id_b)) for p in data.load_test()]
    for key in pred_T:
        ours, theirs = pred_T[key].mean(axis=0)[order_T], np.array(saved[ablation.BY_KEY[key].label])
        rep.line(f"check  T46 predictions, {LABEL[key]}: these models vs the ablation run {files[-1].name}: "
                 f"r {metrics.pearson(ours, theirs):.4f}, largest difference {np.abs(ours - theirs).max():.4f}")
    scores, title, seeds = {}, "", None
    for item in run["items"]:
        if item["type"] == "section":
            title = item["title"]
        elif item["type"] == "scores" and title.startswith("4. T46"):
            scores = {row["label"]: row for row in item["rows"]}
        elif item["type"] == "table" and item["columns"] == ["", "r", "RMSE"]:
            _, r, rmse = next(row for row in item["rows"] if row[0] == "single model: median")
            seeds = {"r": r, "rmse": rmse}
    return {**scores, ONE_SEED: seeds}


def change(s: dict, base: dict) -> list:
    signed = lambda x: "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"      # noqa: E731
    return [signed(s["rmse"] - base["rmse"]), signed(s["r"] - base["r"]) + "*" * (s.get("p_vs_ref", 1.0) < 0.05)]


def report_scores(rep: Report, rows: dict, seeds_SxV, y_V, t46: dict) -> None:
    full = rows[LABEL["full"]]
    scores = {label: metrics.score(p, y_V, reference=None if label == LABEL["full"] else full) for label, p in rows.items()}
    single = np.array([[metrics.rmse(p, y_V), metrics.pearson(p, y_V)] for p in seeds_SxV])
    scores[ONE_SEED] = {"rmse": float(np.median(single[:, 0])), "r": float(np.median(single[:, 1]))}
    t46_label = {**T46_LABEL, ONE_SEED: ONE_SEED}
    base, t46_base = scores[LABEL["full"]], t46[t46_label[LABEL["full"]]]
    rep.line("RMSE change: + is worse.  r change: - is worse.  * = r change with p < 0.05 (paired bootstrap).")
    table = [["baseline, absolute scores", f"{base['rmse']:.3f}", f"{base['r']:.3f}",
              f"{t46_base['rmse']:.3f}", f"{t46_base['r']:.3f}"]]
    table += [[label, *change(s, base), *change(t46[t46_label[label]], t46_base)]
              for label, s in scores.items() if label != LABEL["full"]]
    rep.table(["", "V50 RMSE", "V50 r", "T46 RMSE", "T46 r"], table)
    rep.line("T46 columns: the recorded ablation run (finding 2); its 'team's published' row is the team's T46 file.")
    rep.section("Absolute scores on V50 (RMSE / r, 95% CI of r)")
    rep.scores([(label, s) for label, s in scores.items() if label != ONE_SEED], label="")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", action="store_true", help="score on the V50 labels (do this once)")
    ap.add_argument("--retrain", action="store_true", help="train again even if checkpoints exist")
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS), help="train or load only these")
    args = ap.parse_args()
    rep = Report("eval_d2smell_v50", vars(args), tag="final" * args.final)
    source = locate("olfboost_repo", needs="output")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ablation.free_gpus(6000)[-1])

    rep.section("1. Data and checks")
    td = ablation.load_team_data(source)
    keys_V, team_V = v50_pairs(locate("pnas_repo"))
    x_Vx8240, prof_V = build_v50(rep, source, td, keys_V)

    rep.section("2. Models (10 seeds per variant, trained as the team did for T46)")
    pred_V, pred_T = train_or_load(rep, td, x_Vx8240, args.retrain, args.variants)
    t46 = check_against_ablation(rep, td, pred_T)
    rows = {LABEL[k]: pred_V[k].mean(axis=0) for k in args.variants}
    y_O = evaluate.labels(td.pairs_O)
    for label, distance, q in ((B1A, mixture.unit_euclidean_distance, "q"), (B1, mixture.cosine_distance, "raw")):
        a, c = evaluate.fit_line(distance(td.prof_O[f"a_{q}"], td.prof_O[f"b_{q}"]), y_O)
        rows[label] = a * distance(prof_V[f"a_{q}"], prof_V[f"b_{q}"]) + c
    rows[TEAM] = team_V

    rep.section("3. Our full model vs the team's published V50 predictions (no labels)")
    full_V = rows[LABEL["full"]]
    rep.line(f"r between the two sets of predictions {metrics.pearson(full_V, team_V):.3f}; largest difference "
             f"{np.abs(full_V - team_V).max():.3f}; mean ours {full_V.mean():.3f}, theirs {team_V.mean():.3f}")
    single = [metrics.pearson(p, team_V) for p in pred_V["full"]]
    rep.line(f"single seeds vs the team's file: r {min(single):.3f} to {max(single):.3f}")

    if args.final:
        assert list(args.variants) == list(VARIANTS), "score V50 once, with all variants"
        rep.section("4. V50: change from the baseline (our retrained full model)")
        earlier = rep.previous_runs(final=True)
        if earlier:
            rep.line(f"NOTE: --final was already run at {', '.join(earlier)}")
        y_of = {(p.id_a, p.id_b): p.y for p in data.load_validation()}
        y_V = np.array([y_of[tuple(sorted(k))] for k in keys_V])
    else:
        rep.section("4. DRY RUN: random numbers in place of the V50 labels (pass --final to score)")
        y_V = np.random.default_rng(0).uniform(0.2, 0.9, len(keys_V))
    report_scores(rep, rows, pred_V["full"], y_V, t46)
    rep.value("v50_predictions", {"keys": keys_V, **{label: np.asarray(p).tolist() for label, p in rows.items()}})
    rep.save()


if __name__ == "__main__":
    main()
