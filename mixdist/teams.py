"""Run a DREAM team's own code with its pre-trained models, and compare what it predicts with
the team's files and the paper. Used by evaluation/check_<team>.py. Nothing here trains a model.

A team's code runs unchanged, in its own conda environment, inside a copy of its folder under
~/.cache/osmo-runs/ (set OSMO_RUNS to move it), because the code writes its output next to itself.

Shapes: P pairs.
Standard library and numpy only.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import data, evaluate, metrics


def runs_dir() -> Path:
    return Path(os.environ.get("OSMO_RUNS", Path.home() / ".cache/osmo-runs"))


def env_python(env_name: str) -> Path:
    return Path(os.environ.get("CONDA_ENVS", Path.home() / "miniforge3/envs")) / env_name / "bin/python"


def fresh_copy(source: Path, work: Path) -> Path:
    """Replace ``work`` with a clean copy of the team's folder (without .git)."""
    if work.exists():
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, work, ignore=shutil.ignore_patterns(".git"))
    return work


def free_gpu() -> str | None:
    """Index of the GPU with the most free memory (the RTX 2080 is often in use), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return max((line.split(",") for line in out.strip().splitlines()), key=lambda x: int(x[1]))[0].strip()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def run_in_env(env_name: str, command: list[str], work: Path, extra_env: dict | None = None) -> float:
    """Run ``python <command>`` of a conda environment inside ``work``; return the seconds it took.
    Output goes to ``work/run.log``."""
    python = env_python(env_name)
    if not python.exists():
        raise SystemExit(f"conda environment '{env_name}' not found at {python}; "
                         "the commands to create it are at the top of the calling script")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore", **(extra_env or {})}
    t = time.time()
    with open(work / "run.log", "w") as log:
        code = subprocess.run([str(python), *command], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"{' '.join(command)} failed (exit {code}); see {work / 'run.log'}")
    return time.time() - t


def read_submission(path: Path, pairs) -> np.ndarray:
    """A challenge submission form (Mixture_1, Mixture_2, Predicted_Experimental_Values) -> pred_P
    in the order of ``pairs``."""
    with open(path, newline="") as f:
        rows = {(r["Mixture_1"].strip(), r["Mixture_2"].strip()): float(r["Predicted_Experimental_Values"])
                for r in csv.DictReader(f)}
    return np.array([rows[(p.id_a, p.id_b)] for p in pairs])


def compare_t46(rep, team: str, ours_csv: Path, repo_csv: Path, paper: tuple[float, float]) -> float:
    """Report our rerun of ``team`` on T46 next to the paper and the team's own files.

    ``paper`` = (median RMSE, median Pearson r) from SI Table S4. Returns the largest difference
    between our bootstrap medians and the paper's.
    """
    t46 = data.load_test()
    y_P = evaluate.labels(t46)
    ours_P = read_submission(ours_csv, t46)
    repo_P = read_submission(repo_csv, t46)
    official = data.load_baseline_predictions("T46")[team]
    official_P = np.array([official[(p.id_a, p.id_b)] for p in t46])

    rmse, r = metrics.bootstrap_medians(ours_P, y_P)
    worst = max(abs(rmse - paper[0]), abs(r - paper[1]))
    rep.section(f"{team} on T46 ({len(t46)} pairs): our rerun vs the paper (SI Table S4, bootstrap medians)")
    rep.table(["", "RMSE", "Pearson r"],
              [["our rerun", rmse, r], ["paper", paper[0], paper[1]]])
    rep.line()
    rep.line(f"largest difference from the paper: {worst:.3f}")
    rep.section("Our predictions vs the team's files, largest difference over the pairs")
    rep.table(["file", "max |ours - file|"],
              [["predictions saved in the team's repo", f"{np.abs(ours_P - repo_P).max():.1e}"],
               ["official submission (pnas_repo/Predictions)", f"{np.abs(ours_P - official_P).max():.1e}"]])
    rep.value("predictions", ours_P.tolist())
    rep.value("largest_difference", worst)
    return worst
