# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = molecules with SMILES (235)
#   W = odour words   (MIST 135, open-POM 138)
#   E = embedding size (MIST 512, open-POM 256)
# ---------------------------------------------------------------------------------------------
"""Per-molecule features from frozen pretrained models: one cached file per model.

Every model is stored the same way, as ``features/<model>.npz`` with at least these keys
(row i of every array is molecule ``cids[i]``):
    cids (M,)  smiles (M,)  words (W,)  probs (M x W)  emb (M x E)
plus any model-specific extras (e.g. MIST ``logits``, open-POM ``probs_per_model``).

The files are written by the embed scripts, each in its own conda environment:
    mist      python evaluation/embed_mist.py      (env: mist)
    openpom   python evaluation/embed_openpom.py   (env: openpom)
Reading them needs numpy only. ``ensure(model)`` runs the right embed script in its own
conda env when the file is missing, so experiments run end to end from one command.

Cache folder: OSMO_FEATURES, else ``features/`` in this repository (git-ignored).
"""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODELS = {
    "mist": ("MIST", "MIST odour model (SMILES transformer)", "embed_mist.py", "mist"),
    "openpom": ("open-POM", "open-POM 10-model ensemble (graph neural network)", "embed_openpom.py", "openpom"),
    "unimol2": ("Uni-Mol2", "Uni-Mol2 84M (3D molecular foundation model, frozen, embedding only)", "embed_unimol2.py", "unimol"),
}
REQUIRED_KEYS = ("cids", "smiles", "words", "probs", "emb")
REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FeatureSet:
    model: str
    cids_M: np.ndarray
    smiles_M: np.ndarray
    words_W: np.ndarray
    probs_MxW: np.ndarray                                    # odour-word probabilities in [0, 1]
    emb_MxE: np.ndarray
    extra: dict = field(default_factory=dict)                # other arrays stored in the file

    @property
    def label(self) -> str:
        return MODELS[self.model][0]

    def describe(self) -> str:
        return (f"{MODELS[self.model][1]}: {len(self.cids_M)} molecules, "
                f"{self.probs_MxW.shape[1]} odour words, {self.emb_MxE.shape[1]}-d embedding")


def features_dir() -> Path:
    env = os.environ.get("OSMO_FEATURES")
    d = Path(env) if env else Path(__file__).resolve().parents[1] / "features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path(model: str) -> Path:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {', '.join(MODELS)}")
    return features_dir() / f"{model}.npz"


def save(model: str, arrays: dict) -> Path:
    """Write ``arrays`` (must contain REQUIRED_KEYS, rows aligned with ``cids``) to the cache."""
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"missing keys {missing}")
    m = len(arrays["cids"])
    for k in ("smiles", "probs", "emb"):
        if len(arrays[k]) != m:
            raise ValueError(f"{k} has {len(arrays[k])} rows, cids has {m}")
    if arrays["probs"].shape[1] != len(arrays["words"]):
        raise ValueError("probs columns do not match words")
    f = path(model)
    np.savez_compressed(f, **arrays)
    return f


def load(model: str) -> FeatureSet:
    f = path(model)
    if not f.exists():
        _, _, script, env = MODELS[model]
        raise FileNotFoundError(f"{f} missing: run `python evaluation/{script}` in the {env} conda env")
    z = np.load(f, allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    return FeatureSet(model, arrays.pop("cids"), arrays.pop("smiles"), arrays.pop("words"),
                      arrays.pop("probs"), arrays.pop("emb"), arrays)


def computed_at(model: str) -> str:
    """When the cached file was written (file modification time)."""
    return f"{_dt.datetime.fromtimestamp(path(model).stat().st_mtime):%Y-%m-%d %H:%M}"


def _conda() -> str:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda") or str(Path.home() / "miniforge3/bin/conda")
    if not Path(exe).exists():
        raise FileNotFoundError("conda not found: set CONDA_EXE")
    return exe


def ensure(model: str, recompute: bool = False, log=print) -> FeatureSet:
    """Load the cached features, first running evaluation/<embed script> in the model's own conda env
    (via ``conda run``) if the file is missing or ``recompute`` is set."""
    f = path(model)
    if recompute or not f.exists():
        _, _, script, env = MODELS[model]
        log(f"{'recomputing' if f.exists() else 'no cached file'}: running evaluation/{script} in conda env {env}")
        cmd = [_conda(), "run", "--no-capture-output", "-n", env, "python", str(REPO / "evaluation" / script)]
        subprocess.run(cmd, check=True)
    return load(model)
