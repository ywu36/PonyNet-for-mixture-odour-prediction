"""Check that team PL21's code and pre-trained models reproduce its published test-set (T46) result.

Testing only. The team's notebook (pnas_repo/Models/PL21/PL21Submit.ipynb) is run as it is, with
TRAINING = False: it loads the team's 21 saved single-molecule models (model_single/, one elastic
net per odour word, on Dragon descriptors), predicts the 46 test pairs and writes
Test_set_Submission_form_PL21.csv. The two public tables the notebook reads from GitHub
(Dragon descriptors, single-molecule ratings) are downloaded once to ~/.cache/osmo-runs/pl21_data.

Env:     mist (numpy only) to start it; the team's code runs in the conda environment pl21:
           conda create -n pl21 python=3.9 pip
           pip install scikit-learn==1.5.0 numpy==1.26.4 pandas scipy matplotlib shap
         (scikit-learn 1.5.0 is the version the saved models were made with)
Usage:   python evaluation/check_pl21.py
Writes:  results/check_pl21/<date_time>.{txt,json}
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import teams  # noqa: E402
from mixdist.data import locate  # noqa: E402
from mixdist.report import Report  # noqa: E402

PAPER = (0.086, 0.346)                                       # SI Table S4: median RMSE, median Pearson r
DATA_URL = "https://raw.githubusercontent.com/jeriscience/OlfactionAD/main/data/"
DATA_FILES = ("dragon_all.csv", "percept_single.csv")
OUTPUT = "Test_set_Submission_form_PL21.csv"


def notebook_as_script(work: Path, data_dir: Path) -> str:
    """The notebook's code cell, unchanged except that the two tables are read from ``data_dir``."""
    code = "".join(json.loads((work / "PL21Submit.ipynb").read_text())["cells"][0]["source"])
    code = "\n".join(line for line in code.split("\n") if not line.startswith("%"))   # notebook magics
    assert "TRAINING = False" in code, "the notebook is expected to load the saved models, not train"
    for name in DATA_FILES:
        assert code.count(DATA_URL + name) == 1
        code = code.replace(DATA_URL + name, str(data_dir / name))
    return "display = print\n" + code


def main():
    rep = Report("check_pl21")
    source = locate("pnas_repo", needs="Models/PL21") / "Models/PL21"
    data_dir = teams.runs_dir() / "pl21_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        if not (data_dir / name).exists():
            urllib.request.urlretrieve(DATA_URL + name, data_dir / name)

    work = teams.fresh_copy(source, teams.runs_dir() / "pl21")
    (work / OUTPUT).unlink()                                 # so the file scored below can only come from this run
    (work / "pl21_run.py").write_text(notebook_as_script(work, data_dir))
    seconds = teams.run_in_env("pl21", ["pl21_run.py"], work)

    rep.section("Run")
    rep.line(f"team code   {source}")
    rep.line(f"models      {len(list((source / 'model_single').iterdir()))} saved models in model_single/ (not retrained)")
    rep.line(f"ran in      {work}   ({seconds:.0f} s, log in run.log)")
    teams.compare_t46(rep, "PL21", work / OUTPUT, source / OUTPUT, PAPER)
    rep.save()


if __name__ == "__main__":
    main()
