"""Retrain team D2Smell's model from scratch with the team's own code, then score it on T46.

Level-3 reproduction: evaluation/check_d2smell.py reruns the team's SAVED models; this script trains
them again. For each seed it runs the team's train/mixture_regressor_ensemble.py --seed <seed>
(XGBoost on the GPU, 20 Optuna trials, 10-fold CV models averaged), as the team's
scripts/generate_batch.sh does for its 10 seeds, then train/ensemble.py averages the seeds and
writes the T46 predictions. The team's saved models are removed from the run copy first, so every
prediction scored here comes from models trained in this run. The earlier steps (augmentation,
odor-pair features, percept model) are not rerun: their saved outputs in the repo are used.

Note: the team's Optuna objective is the Pearson r on the leaderboard pairs, which are also in its
training data (in-sample tuning). That is how the team trained, so it is reproduced as it is;
T46 labels are never used.

Env:     mist (numpy only) to start it; the team's code runs in the conda env d2smell (see check_d2smell.py)
Usage:   python training/train_d2smell.py                 all 10 seeds of the submission (~1 h per seed)
         python training/train_d2smell.py --seeds 42       one seed, for a quick run
         Uses the GPU with the most free memory; set CUDA_VISIBLE_DEVICES to choose one.
Writes:  checkpoints/d2smell/mixture_model_<seed>.pkl, results/train_d2smell/<date_time>[_<n>seeds].{txt,json}
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evaluation"))
from check_d2smell import OUTPUT, PAPER, REPO_FILE  # noqa: E402

from mixdist import teams  # noqa: E402
from mixdist.data import locate  # noqa: E402
from mixdist.report import Report  # noqa: E402

SUBMISSION_SEEDS = (42, 3407, 5067, 1111, 2222, 3333, 6666, 7777, 8888, 9999)   # scripts/generate_batch.sh
CHECKPOINTS = REPO / "checkpoints/d2smell"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SUBMISSION_SEEDS))
    args = ap.parse_args()
    all_seeds = sorted(args.seeds) == sorted(SUBMISSION_SEEDS)
    rep = Report("train_d2smell", vars(args), tag="" if all_seeds else f"{len(args.seeds)}seeds")

    source = locate("olfboost_repo", needs="output")
    work = teams.fresh_copy(source, teams.runs_dir() / "d2smell_train")
    (work / OUTPUT).unlink()                                 # predictions must come from this run
    for saved in (work / "output").glob("mixture_model_*.pkl"):
        saved.unlink()                                       # models must be retrained, not loaded
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or teams.free_gpu()
    extra_env = {"CUDA_VISIBLE_DEVICES": gpu} if gpu else None

    rep.section("Training runs, one per seed")
    rows = []
    for seed in args.seeds:
        rep.line(f"training seed {seed} ...")
        seconds = teams.run_in_env("d2smell", ["train/mixture_regressor_ensemble.py", "--seed", str(seed)], work, extra_env)
        (work / "run.log").rename(work / f"train_{seed}.log")
        rows.append([str(seed), f"{seconds / 60:.0f}", f"train_{seed}.log"])
    rep.table(["seed", "minutes", "log"], rows)
    seconds = teams.run_in_env("d2smell", ["train/ensemble.py"], work, extra_env)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for model in (work / "output").glob("mixture_model_*.pkl"):
        shutil.copy2(model, CHECKPOINTS / model.name)

    rep.section("Run")
    rep.line(f"team code   {source}")
    rep.line(f"trained     {len(args.seeds)} seed(s) from scratch: {', '.join(map(str, args.seeds))}"
             + ("" if all_seeds else f"  (the submission used all {len(SUBMISSION_SEEDS)})"))
    rep.line(f"ran in      {work}   (GPU {gpu if gpu else 'not set'})")
    rep.line(f"checkpoints {CHECKPOINTS}")
    teams.compare_t46(rep, "D2Smell", work / OUTPUT, source / REPO_FILE, PAPER)
    rep.line("'the team's files' = predictions of the team's own saved models: a difference shows how far")
    rep.line("retraining from scratch moves the result")
    rep.save()


if __name__ == "__main__":
    main()
