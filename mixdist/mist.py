# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = molecules with SMILES (235)   W = MIST odour words (135)   E = MIST embedding size (512)
# ---------------------------------------------------------------------------------------------
"""Run the frozen MIST odour model (dream2_data/mist_odour) on every molecule.

Needs the `mist` conda environment (torch, transformers, rdkit). Used by evaluation/embed_mist.py.
"""
from __future__ import annotations

import numpy as np

from . import data
from .device import pick_device


def kekulize(smiles: str) -> str:
    """MIST was trained on Kekulized SMILES (model card), so aromatic notation is rewritten."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def run_model(smiles_M: list[str], device: str | None = None, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """MIST on any SMILES (Kekulized first) -> (logits_MxW, emb_MxE, indices RDKit could not parse,
    left out of the arrays)."""
    from transformers import AutoModel

    device = device or pick_device()
    model = AutoModel.from_pretrained(str(data.locate("mist_odour")), trust_remote_code=True).to(device).eval()
    kek, bad = [], []
    for i, smi in enumerate(smiles_M):
        try:
            kek.append(kekulize(smi))
        except Exception:
            bad.append(i)
    logit_chunks, emb_chunks = [], []
    for i in range(0, len(kek), batch_size):
        batch = kek[i:i + batch_size]
        logit_chunks.append(model.predict(batch, return_dict=False).float().numpy())
        emb_chunks.append(model.embed(batch).float().numpy())
    return np.concatenate(logit_chunks), np.concatenate(emb_chunks), bad


def compute(device: str | None = None, batch_size: int = 64) -> dict:
    """135 odour-word logits/probabilities and the 512-d first-token embedding per molecule.

    Returns arrays in the standard feature layout (see mixdist.features) plus ``logits``.
    """
    from transformers import AutoModel

    device = device or pick_device()
    model = AutoModel.from_pretrained(str(data.locate("mist_odour")), trust_remote_code=True).to(device).eval()
    smiles = data.load_smiles()                              # {CID: SMILES}, M entries
    cids_M = sorted(smiles)
    kek_M = [kekulize(smiles[c]) for c in cids_M]
    logit_chunks, emb_chunks = [], []                        # lists of (batch x W), (batch x E)
    for i in range(0, len(kek_M), batch_size):
        batch = kek_M[i:i + batch_size]
        logit_chunks.append(model.predict(batch, return_dict=False).float().numpy())
        emb_chunks.append(model.embed(batch).float().numpy())
    logits_MxW = np.concatenate(logit_chunks)
    return {"cids": np.array(cids_M), "smiles": np.array(kek_M),
            "words": np.array([c["name"] for c in model.config.channels]),
            "probs": 1.0 / (1.0 + np.exp(-logits_MxW)),
            "emb": np.concatenate(emb_chunks),
            "logits": logits_MxW, "device": np.array(device)}
