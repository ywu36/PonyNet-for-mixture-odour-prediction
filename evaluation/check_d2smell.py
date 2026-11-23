"""Check that team D2Smell's code and pre-trained models reproduce its published test-set (T46) result.

Testing only. The team's last pipeline step (olfboost_repo/train/ensemble.py; despite the folder
name it only predicts) loads the team's 10 saved XGBoost ensembles (output/mixture_model_<seed>.pkl),
averages their predictions for the 46 test pairs and writes output/Test_set_Submission_form_ensemble.csv.
The earlier steps (data augmentation, odor-pair features, single-molecule percept model) are not
rerun: their saved outputs in the repo are used as they are.

Env:     mist (numpy only) to start it; the team's code runs in the conda environment d2smell:
           conda create -n d2smell python=3.9.12 pip
           pip install numpy==1.24.3 pandas==2.1.4 scikit-learn==1.0.2 xgboost==1.5.0 scipy==1.10.1 joblib==1.4.0
         (the versions pinned in the team's conda-environment.yaml)
Usage:   python evaluation/check_d2smell.py
         Uses the GPU with the most free memory; set CUDA_VISIBLE_DEVICES to choose one.
Writes:  results/check_d2smell/<date_time>.{txt,json}
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import teams  # noqa: E402
from mixdist.data import locate  # noqa: E402
from mixdist.report import Report  # noqa: E402

PAPER = (0.081, 0.469)                                       # SI Table S4: median RMSE, median Pearson r
OUTPUT = "output/Test_set_Submission_form_ensemble.csv"
REPO_FILE = "final_results/Test_set_Submission_form_ensemble.csv"


def main():
    rep = Report("check_d2smell")
    source = locate("olfboost_repo", needs="output")
    work = teams.fresh_copy(source, teams.runs_dir() / "d2smell")
    (work / OUTPUT).unlink()                                 # so the file scored below can only come from this run
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or teams.free_gpu()
    seconds = teams.run_in_env("d2smell", ["train/ensemble.py"], work, {"CUDA_VISIBLE_DEVICES": gpu} if gpu else None)

    rep.section("Run")
    rep.line(f"team code   {source}")
    rep.line(f"models      {len(list((source / 'output').glob('mixture_model_*.pkl')))} saved models output/mixture_model_<seed>.pkl (not retrained)")
    rep.line(f"ran in      {work}   ({seconds:.0f} s, GPU {gpu if gpu else 'not set'}, log in run.log)")
    teams.compare_t46(rep, "D2Smell", work / OUTPUT, source / REPO_FILE, PAPER)
    rep.save()


if __name__ == "__main__":
    main()
