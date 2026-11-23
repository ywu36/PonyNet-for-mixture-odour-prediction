"""Retrain team PL21's model from scratch with the team's own code, then score it on T46.

Level-3 reproduction: evaluation/check_pl21.py reruns the team's SAVED models; this script trains
them again. The team's notebook (pnas_repo/Models/PL21/PL21Submit.ipynb) is run with
TRAINING = True: it refits the 21 single-molecule elastic nets (one per odour word, on Dragon
descriptors, 5-fold grid search each) and then the mixture model, and writes its T46
predictions. The team's saved models are removed from the run copy first, so every prediction
scored here comes from models trained in this run.

The mixture model is fitted on the team's 60-pair gt_SBs file, as in the submitted notebook;
nothing is tuned on T46.

Env:     mist (numpy only) to start it; the team's code runs in the conda env pl21 (see check_pl21.py)
Usage:   python training/train_pl21.py            (a few minutes on the CPU)
Writes:  checkpoints/pl21/model_single/ (the 21 retrained models),
         results/train_pl21/<date_time>.{txt,json}
"""
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evaluation"))
from check_pl21 import DATA_FILES, DATA_URL, OUTPUT, PAPER, notebook_as_script  # noqa: E402

from mixdist import teams  # noqa: E402
from mixdist.data import locate  # noqa: E402
from mixdist.report import Report  # noqa: E402

CHECKPOINTS = REPO / "checkpoints/pl21"


def main():
    rep = Report("train_pl21")
    source = locate("pnas_repo", needs="Models/PL21") / "Models/PL21"
    data_dir = teams.runs_dir() / "pl21_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        if not (data_dir / name).exists():
            urllib.request.urlretrieve(DATA_URL + name, data_dir / name)

    work = teams.fresh_copy(source, teams.runs_dir() / "pl21_train")
    (work / OUTPUT).unlink()                                 # predictions must come from this run
    shutil.rmtree(work / "model_single")                     # models must be retrained, not loaded
    (work / "model_single").mkdir()
    script = notebook_as_script(work, data_dir).replace("TRAINING = False", "TRAINING = True", 1)
    (work / "pl21_run.py").write_text(script)
    seconds = teams.run_in_env("pl21", ["pl21_run.py"], work)

    n_models = len(list((work / "model_single").iterdir()))
    if CHECKPOINTS.exists():
        shutil.rmtree(CHECKPOINTS)
    shutil.copytree(work / "model_single", CHECKPOINTS / "model_single")

    rep.section("Training run")
    rep.line(f"team code   {source}")
    rep.line(f"trained     {n_models} single-molecule models from scratch (saved models removed first)")
    rep.line(f"ran in      {work}   ({seconds:.0f} s, log in run.log)")
    rep.line(f"checkpoints {CHECKPOINTS / 'model_single'}")
    teams.compare_t46(rep, "PL21", work / OUTPUT, source / OUTPUT, PAPER)
    rep.line("'the team's files' = predictions of the team's own saved models: a difference shows how far")
    rep.line("retraining from scratch moves the result")
    rep.save()


if __name__ == "__main__":
    main()
