# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   R = training rows: the 780 original pairs first, then their 19,798 augmented copies
#   O = original training pairs (780)            T = T46 pairs (46)
#   X = mixtures in the team's profile file (20,740)
#   W = odour words of the odor-pair network (25 kept by the team, or all 33)
#   F = feature columns of one variant            S = saved models, one per seed (10)
# ---------------------------------------------------------------------------------------------
"""Ablation of team D2Smell's mixture model, shared by eval_d2smell_ablation.py. Not a script.

The team's model is XGBoost on one row per pair of mixtures. Its 8,182 columns are three groups:
    identity   0/1 per molecule of the team's list of 204, for mixture 1 and for mixture 2       408
    ratings    21 single-molecule ratings (measured or predicted), one slot per molecule      7,749
    odour      per odour word, squared difference between the two mixtures' profiles              25
A profile is the odor-pair network's output averaged over the molecule pairs of a mixture (the
team's saved file), with 8 of the 33 words dropped and each word quantile-transformed twice.
Training rows are the 780 pairs plus 19,798 augmented copies: one molecule of the other mixture
added to mixture 1 ("add"), or one shared molecule removed from mixture 1 ("remove").

How the baseline is kept faithful
  * The baseline matrix is built by the team's own functions (utils/utils.py and
    train/mixture_regressor_ensemble.py, imported, not copied), so it has the team's column order
    and its slips: only the first 369 identity columns get rating slots; slot k holds row k of a
    differently ordered ratings table (row k - 184 from slot 185 on; own molecule in 18 of 368);
    T46 rows do not mark the first molecule of a mixture. check_d2smell_features.py verifies these.
  * ``check_features`` runs the team's 10 saved models on that matrix for T46 and compares with the
    team's prediction file.
  * Settings come from the team's saved models (one set per seed). The two label shifts of the
    augmented rows are not stored in them, so the middle of the team's search ranges is used.

What differs from the team's training: no Optuna search (it scored each trial on leaderboard pairs
that are in the training rows), and in cross-validation one model per fold on all training rows of
that fold. T46 follows the team: one model per seed on the last 90% split of that seed, mean of 10.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mixdist import data, evaluate, mixture  # noqa: E402

SEEDS = (42, 3407, 5067, 1111, 2222, 3333, 6666, 7777, 8888, 9999)      # the team's scripts/generate_batch.sh
DROPPED_WORDS = (11, 18, 21, 22, 27, 28, 31, 32)                        # the 8 of 33 words the team drops
ADD_SHIFT, REMOVE_SHIFT = 0.0005, 0.05                  # middle of the team's ranges [1e-6, 1e-3] and [0.001, 0.1]
NONE, ADD, REMOVE = 0, 1, 2                             # action_R codes
TUNED = ("n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
         "colsample_bytree", "gamma", "reg_lambda", "reg_alpha")
TEAM_T46_FILE = "final_results/Test_set_Submission_form_ensemble.csv"

# Column ranges. The first four are the team's layout (in this order); the last two are added here.
BLOCKS = {
    "identity_head": (0, 369),        # mixture 1 (204) and the first 165 molecules of mixture 2
    "ratings": (369, 8118),           # 369 x 21 rating slots
    "identity_tail": (8118, 8157),    # the last 39 molecules of mixture 2 (the team's slicing slip)
    "odour": (8157, 8182),            # 25 squared differences, quantile-transformed profiles
    "odour_raw": (8182, 8207),        # the same 25 without the quantile transform
    "odour_33": (8207, 8240),         # all 33 words, quantile-transformed
}
TEAM_BLOCKS = ("identity_head", "ratings", "identity_tail", "odour")
IDENTITY = ("identity_head", "identity_tail")


@dataclass
class TeamData:
    pairs_O: list                     # data.Pair; a and b are the team's mixture definitions
    y_R: np.ndarray                   # label as stored (an augmented row carries its parent's label)
    parent_R: np.ndarray              # index into pairs_O; an original row is its own parent
    action_R: np.ndarray              # NONE, ADD or REMOVE
    team_order_R: np.ndarray          # row indices in the order the team's code trains on (its merge regroups the rows)
    x_Rx8240: np.ndarray              # all blocks, float32
    x_Tx8240: np.ndarray
    prof_O: dict                      # "a_raw", "b_raw", "a_q", "b_q": O x 25 profiles of the original pairs
    prof_T: dict
    keys_T: list                      # (Mixture_1, Mixture_2) of each T row, as strings
    notes: list                       # lines for the report: what was checked while building

    def columns(self, blocks) -> np.ndarray:
        return np.concatenate([np.arange(*BLOCKS[b]) for b in blocks])


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _team_code(source: Path):
    """The team's two modules, imported from its folder -> (utils.utils, train/mixture_regressor_ensemble)."""
    sys.dont_write_bytecode = True                           # leave the team's folder untouched
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    with _quiet():
        from utils import utils as team_utils
        spec = importlib.util.spec_from_file_location("d2smell_mixture_regressor",
                                                      source / "train/mixture_regressor_ensemble.py")
        team_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(team_train)
    return team_utils, team_train


def _expand_into(out_Nx8240: np.ndarray, rows_N: np.ndarray, team_utils, x_Nx433: np.ndarray, ratings, rating_ids,
                 chunk: int = 2000) -> None:
    """The team's expand_features, a chunk of rows at a time (it needs ~200 kB per row); row n of
    its output is written to row rows_N[n], columns 0 to 8182, of ``out_Nx8240``."""
    for i in range(0, len(x_Nx433), chunk):
        with _quiet():
            out_Nx8240[rows_N[i:i + chunk], :8182] = team_utils.expand_features(x_Nx433[i:i + chunk], ratings, rating_ids).values


def _augmentation_links(gt, mix: dict) -> tuple[np.ndarray, np.ndarray]:
    """Parent pair and action of every row, rebuilt from the team's augmentation rule and checked
    against the file -> (parent_R, action_R)."""
    n_orig = int(gt["augmentation_Action"].isna().sum())
    assert gt["augmentation_Action"].iloc[:n_orig].isna().all(), "original rows are expected first"
    parent, action = list(range(n_orig)), [NONE] * n_orig
    for i, (ds, m1, m2) in enumerate(gt[["Dataset", "Mixture 1", "Mixture 2"]].iloc[:n_orig].itertuples(index=False)):
        a, b = mix[(ds, m1)], mix[(ds, m2)]
        n_add_12, n_add_21, n_remove = len(b - a), len(a - b), len(a & b)
        parent += [i] * (n_add_12 + n_add_21 + n_remove)
        action += [ADD] * (n_add_12 + n_add_21) + [REMOVE] * n_remove
    parent_R, action_R = np.array(parent), np.array(action)
    assert len(parent_R) == len(gt), "augmented rows do not follow the team's augmentation rule"
    stored = gt["augmentation_Action"].fillna("").str.split("_").str[0].map({"": NONE, "add": ADD, "remove": REMOVE})
    assert (stored.values == action_R).all(), "actions in the file differ from the rule"
    assert np.allclose(gt["Experimental Values"].values, gt["Experimental Values"].values[parent_R])
    for r in np.flatnonzero(action_R == REMOVE)[::50]:       # "remove" changes mixture 1 only
        ds, new, other = gt["Dataset"][r], gt["Mixture 1"][r], gt["Mixture 2"][r]
        p1, p2 = mix[(ds, gt["Mixture 1"][parent_R[r]])], mix[(ds, gt["Mixture 2"][parent_R[r]])]
        assert other == gt["Mixture 2"][parent_R[r]] and len(p1 - mix[(ds, new)]) == 1 and (p1 - mix[(ds, new)]) <= p2
    return parent_R, action_R


def load_team_data(source: Path) -> TeamData:
    """Build every feature block for the training rows and the T46 rows. No labels of T46 are read."""
    import pandas as pd
    team_utils, team_train = _team_code(source)
    read = lambda path: pd.read_csv(source / path)           # noqa: E731
    gt = read("data/processed/gt_with_dataset_V2_augmented_dataset.csv")
    defs = read("data/processed/Mixure_Definitions_augmented_dataset.csv")
    prof = read("data/processed/predictions_separated_mean_33_Augmentation_Dataset.csv")
    form = read("data/raw/forms/Test_set_Submission_form.csv")
    defs_test = read("data/raw/mixtures/Mixure_Definitions_test_set.csv")
    ratings = read("output/percept_scaled.csv")
    all_cids = sorted(read("data/processed/CID.csv")["CID"].astype(int).tolist())
    rating_ids = list(range(1, ratings.shape[1]))
    notes = []

    # profiles: raw, the team's (8 words dropped, quantile twice), and all 33 words quantile twice
    words = [c for c in prof.columns if c.startswith("Prediction_")]
    kept = [c for c in words if int(c.split("_")[1]) not in DROPPED_WORDS]
    prof_q = team_utils.applying_2QuantileTransformer(prof[["Dataset", "Mixture Label"] + kept].copy())
    prof_q33 = team_utils.applying_2QuantileTransformer(prof.copy())
    raw_Xx25, q_Xx25, q_Xx33 = prof[kept].values, prof_q[kept].values, prof_q33[words].values
    once_Xx25 = team_utils.QuantileTransformer(output_distribution="normal", n_quantiles=500,
                                               random_state=0).fit_transform(raw_Xx25)
    notes.append(f"quantile transform applied twice (the team) vs once: largest difference {np.abs(q_Xx25 - once_Xx25).max():.3f}, "
                 f"mean {np.abs(q_Xx25 - once_Xx25).mean():.4f} (values span {q_Xx25.min():.1f} to {q_Xx25.max():.1f})")

    # the team's matrices
    with _quiet():
        all_df, x_Rx433 = team_train.prepare_training_data(gt, prof_q, defs, all_cids)
        x_Tx433 = team_train.prepare_test_data(form, prof_q, defs_test, all_cids)
    team_order_R = all_df.index.values                       # the team's merge regroups the rows; entry n = file row
    assert sorted(team_order_R) == list(range(len(gt))), "the team's merge dropped or repeated rows"
    x_Rx8240 = np.empty((len(gt), BLOCKS["odour_33"][1]), np.float32)        # stored in file order: pair i is row i
    x_Tx8240 = np.empty((len(form), BLOCKS["odour_33"][1]), np.float32)
    _expand_into(x_Rx8240, team_order_R, team_utils, x_Rx433, ratings, rating_ids)
    _expand_into(x_Tx8240, np.arange(len(form)), team_utils, x_Tx433, ratings, rating_ids)

    # our own odour blocks, from the same profiles
    row_of = {key: i for i, key in enumerate(zip(prof["Dataset"], prof["Mixture Label"]))}
    ia_R = np.array([row_of[k] for k in zip(gt["Dataset"], gt["Mixture 1"])])
    ib_R = np.array([row_of[k] for k in zip(gt["Dataset"], gt["Mixture 2"])])
    ia_T = np.array([row_of[("Test", m)] for m in form["Mixture_1"]])
    ib_T = np.array([row_of[("Test", m)] for m in form["Mixture_2"]])

    def squared(p_XxW, ia, ib):
        return ((p_XxW[ia] - p_XxW[ib]) ** 2).astype(np.float32)

    (lo, hi), (raw_lo, raw_hi), (w33_lo, w33_hi) = BLOCKS["odour"], BLOCKS["odour_raw"], BLOCKS["odour_33"]
    for x, ia, ib in ((x_Rx8240, ia_R, ib_R), (x_Tx8240, ia_T, ib_T)):
        assert np.allclose(x[:, lo:hi], squared(q_Xx25, ia, ib), atol=1e-5), "odour block differs from the team's"
        x[:, raw_lo:raw_hi] = squared(raw_Xx25, ia, ib)
        x[:, w33_lo:w33_hi] = squared(q_Xx33, ia, ib)
    notes.append("odour block recomputed here from the profile file equals the team's (training rows and T46)")

    # pairs, augmentation links
    cid_cols = [c for c in defs.columns if c.startswith("CID_")]
    mix = {(ds, label): frozenset(int(c) for c in cids if c == c and c != 0)
           for ds, label, *cids in defs[["Dataset", "Mixture Label"] + cid_cols].itertuples(index=False)}
    parent_R, action_R = _augmentation_links(gt, mix)
    n_orig = int((action_R == NONE).sum())
    pairs_O = [data.Pair(ds, str(m1), str(m2), mix[(ds, m1)], mix[(ds, m2)], float(y))
               for ds, m1, m2, y in gt[["Dataset", "Mixture 1", "Mixture 2", "Experimental Values"]].iloc[:n_orig].itertuples(index=False)]
    notes.append(f"augmented rows follow the team's rule: {int((action_R == ADD).sum())} 'add' (one molecule of the other "
                 f"mixture added to mixture 1), {int((action_R == REMOVE).sum())} 'remove' (one shared molecule removed from "
                 "mixture 1 only; mixture 2 keeps it)")
    rated = ratings["CID"].astype(int).tolist()
    notes.append(f"rating slots: slot j holds row j of the ratings table, which is in another order; own molecule for "
                 f"{sum(a == b for a, b in zip(all_cids, rated))} of the first {len(all_cids)} (all slots: check_d2smell_features.py)")

    sel = np.arange(n_orig)
    return TeamData(
        pairs_O, gt["Experimental Values"].values.astype(float), parent_R, action_R, team_order_R, x_Rx8240, x_Tx8240,
        {"a_raw": raw_Xx25[ia_R[sel]], "b_raw": raw_Xx25[ib_R[sel]], "a_q": q_Xx25[ia_R[sel]], "b_q": q_Xx25[ib_R[sel]]},
        {"a_raw": raw_Xx25[ia_T], "b_raw": raw_Xx25[ib_T], "a_q": q_Xx25[ia_T], "b_q": q_Xx25[ib_T]},
        [(str(a), str(b)) for a, b in zip(form["Mixture_1"], form["Mixture_2"])], notes)


def saved_settings(source: Path) -> list[dict]:
    """The tuned XGBoost settings of the team's saved models, in the order of SEEDS -> S dicts."""
    import joblib
    out = []
    for seed in SEEDS:
        params = joblib.load(source / f"output/mixture_model_{seed}.pkl")["model"][0].get_params()
        out.append({k: params[k] for k in TUNED})
    return out


def check_features(source: Path, td: TeamData) -> float:
    """Run the team's saved models on our T46 matrix -> largest difference from the team's prediction file."""
    import csv

    import joblib
    cols = td.columns(TEAM_BLOCKS)
    pred_SxT = []
    for seed in SEEDS:
        saved = joblib.load(source / f"output/mixture_model_{seed}.pkl")
        pred_SxT.append(sum(w * m.predict(td.x_Tx8240[:, cols]) for m, w in zip(saved["model"], saved["weight"])))
    with open(source / TEAM_T46_FILE, newline="", encoding="utf-8-sig") as f:
        team = {(r["Mixture_1"].strip(), r["Mixture_2"].strip()): float(r["Predicted_Experimental_Values"])
                for r in csv.DictReader(f)}
    return float(np.abs(np.mean(pred_SxT, axis=0) - np.array([team[k] for k in td.keys_T])).max())


# ---- variants -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    blocks: tuple = TEAM_BLOCKS
    augment: str = "team"             # "team"; "none" (780 rows); "unchanged" (augmented rows keep the parent's label)
    tuned: bool = True                # False: XGBoost's own default settings
    model: str = "xgboost"            # or "linear": least squares on the same columns and rows


VARIANTS = (
    Variant("full", "full model (the team's)"),
    # A. which columns
    Variant("identity", "A identity only", IDENTITY),
    Variant("ratings", "A ratings columns only", ("ratings",)),
    Variant("odour", "A odour differences only", ("odour",)),
    Variant("no_ratings", "A identity + odour (ratings columns removed)", ("identity_head", "identity_tail", "odour")),
    Variant("team_semantic", "A odour + ratings (the team's 'semantic' run)", ("ratings", "identity_tail", "odour")),
    # B. how the rows and the odour columns are made, everything else as the full model
    Variant("no_aug", "B no augmentation (780 rows)", augment="none"),
    Variant("aug_same_label", "B augmented rows keep the parent's label", augment="unchanged"),
    Variant("no_quantile", "B odour without quantile transform", ("identity_head", "ratings", "identity_tail", "odour_raw")),
    Variant("words_33", "B all 33 odour words", ("identity_head", "ratings", "identity_tail", "odour_33")),
    Variant("default_settings", "B XGBoost default settings (no tuning)", tuned=False),
    # C. the same steps for the model that sees only the odour differences
    Variant("odour_no_aug", "C odour only, no augmentation", ("odour",), augment="none"),
    Variant("odour_no_quantile", "C odour only, no quantile transform", ("odour_raw",)),
    Variant("odour_words_33", "C odour only, all 33 words", ("odour_33",)),
    Variant("odour_linear", "C odour only, least squares instead of XGBoost", ("odour",), model="linear"),
)
# The team's own "semantic" ablation (class23) is ratings + the last 64 columns, hence identity_tail there.
BY_KEY = {v.key: v for v in VARIANTS}


def training_rows(td: TeamData, train_O, augment: str, team_order: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose parent pair is in ``train_O`` and their labels -> (row indices, y); in file order,
    or in the order the team's code trains on (needed to repeat its 90% splits)."""
    in_train_O = np.zeros(len(td.pairs_O), bool)
    in_train_O[np.asarray(train_O)] = True
    keep_R = in_train_O[td.parent_R]
    if augment == "none":
        keep_R &= td.action_R == NONE
    rows = td.team_order_R[keep_R[td.team_order_R]] if team_order else np.flatnonzero(keep_R)
    y = td.y_R[rows].copy()
    if augment == "team":
        y[td.action_R[rows] == ADD] -= ADD_SHIFT
        y[td.action_R[rows] == REMOVE] += REMOVE_SHIFT
    return rows, y


def fit(x_NxF: np.ndarray, y_N: np.ndarray, variant: Variant, settings: dict):
    if variant.model == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression().fit(x_NxF, y_N)
    import xgboost as xgb
    model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="gpu_hist", predictor="gpu_predictor",
                             **(settings if variant.tuned else {}))
    return model.fit(x_NxF, y_N)


def cross_validate(td: TeamData, variant: Variant, settings_S: list, k: int, repeats: int) -> evaluate.CVResult:
    """Whole mixtures held out (mixdist.cv) over the 780 original pairs. A fold trains on the
    original pairs of its training side and on their augmented copies only; repeat i uses the
    saved settings of seed i."""
    x_RxF = td.x_Rx8240[:, td.columns(variant.blocks)]
    calls = itertools.count()

    def fit_predict(train_O, test_O):
        settings = settings_S[(next(calls) // k) % len(settings_S)]
        rows, y = training_rows(td, train_O, variant.augment)
        return fit(x_RxF[rows], y, variant, settings).predict(x_RxF[test_O])   # original pair i is row i

    return evaluate.cross_validate(td.pairs_O, fit_predict, k=k, repeats=repeats)


def predict_t46(td: TeamData, variant: Variant, settings_S: list) -> np.ndarray:
    """As the team: per seed one model on the last of that seed's ten 90% splits -> pred_SxT."""
    from sklearn.model_selection import KFold
    cols = td.columns(variant.blocks)
    x_RxF, x_TxF = td.x_Rx8240[:, cols], td.x_Tx8240[:, cols]
    rows, y = training_rows(td, np.arange(len(td.pairs_O)), variant.augment, team_order=True)
    pred = []
    for seed, settings in zip(SEEDS, settings_S):
        part = list(KFold(n_splits=10, shuffle=True, random_state=seed).split(rows))[-1][0]
        pred.append(fit(x_RxF[rows[part]], y[part], variant, settings).predict(x_TxF))
    return np.array(pred)


# ---- rows without a trained model: one number per pair, then the calibration line ------------------

def distances(td: TeamData) -> dict:
    """label -> (dist_O, dist_T): the network's profiles through B1 and B1a, and molecule overlap."""
    def overlap_distance(pairs):
        return np.array([1.0 - p.jaccard for p in pairs])
    o, t = td.prof_O, td.prof_T
    return {
        "network profile, cosine + line (B1)": (mixture.cosine_distance(o["a_raw"], o["b_raw"]),
                                               mixture.cosine_distance(t["a_raw"], t["b_raw"])),
        "network profile, quantile, length 1, Euclidean + line (B1a)": (
            mixture.unit_euclidean_distance(o["a_q"], o["b_q"]), mixture.unit_euclidean_distance(t["a_q"], t["b_q"])),
        "molecule overlap (1 - Jaccard) + line": (overlap_distance(td.pairs_O), None),   # T46: no shared molecules
    }


def constant_fit_predict(y_O: np.ndarray):
    """fit_predict that predicts the mean label of the training pairs."""
    return lambda train_O, test_O: np.full(len(test_O), y_O[train_O].mean())


# ---- running the trained variants on several GPUs --------------------------------------------------

def free_gpus(min_free_mib: int = 4000) -> list[str]:
    """GPUs to use, one worker each: CUDA_VISIBLE_DEVICES if set, else every GPU with enough free memory."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [i.strip() for i, free in (line.split(",") for line in out.strip().splitlines()) if int(free) >= min_free_mib]


_WORKER: dict = {}


def _start_worker(gpu_queue, source: str, k: int, repeats: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_queue.get()     # before xgboost touches a GPU
    source = Path(source)
    _WORKER.update(td=load_team_data(source), settings=saved_settings(source), k=k, repeats=repeats)


def _run_task(task: tuple[str, str]):
    key, protocol = task
    start, w = time.time(), _WORKER
    if protocol == "cv":
        out = cross_validate(w["td"], BY_KEY[key], w["settings"], w["k"], w["repeats"]).pred_N
    else:
        out = predict_t46(w["td"], BY_KEY[key], w["settings"])
    return key, protocol, out, time.time() - start, os.environ["CUDA_VISIBLE_DEVICES"]


def run_variants(source: Path, protocols: tuple, k: int, repeats: int, gpus: list[str], log=print) -> dict:
    """Every variant under every protocol ("cv", "t46"), one worker process per GPU ->
    {protocol: {variant key: pred_N or pred_SxT}}. Widest variants are started first."""
    width = lambda v: len(np.concatenate([np.arange(*BLOCKS[b]) for b in v.blocks]))      # noqa: E731
    tasks = [(v.key, p) for v in sorted(VARIANTS, key=width, reverse=True) for p in protocols]
    ctx = multiprocessing.get_context("spawn")
    gpu_queue = ctx.Queue()
    for g in gpus:
        gpu_queue.put(g)
    out = {p: {} for p in protocols}
    with ctx.Pool(len(gpus), _start_worker, (gpu_queue, str(source), k, repeats)) as pool:
        for n, (key, protocol, pred, seconds, gpu) in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            out[protocol][key] = pred
            log(f"  {n:>2}/{len(tasks)}  {protocol:<4}{key:<20}{seconds / 60:5.1f} min  GPU {gpu}")
    return out
