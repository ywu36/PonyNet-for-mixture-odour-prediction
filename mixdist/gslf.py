# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  G = our molecules that have GS-LF labels (229)
#                                       L = GS-LF odour labels (138)
# ---------------------------------------------------------------------------------------------
"""Ground-truth odour labels for our molecules from GS-LF (GoodScents + Leffingwell, 4,983
molecules x 138 labels, curated by Lee et al. 2023), the dataset MIST and open-POM were trained on.

Molecules are matched to our CIDs by RDKit canonical SMILES without stereochemistry (GS-LF has no
CIDs). Needs rdkit (mist or openpom env).
"""
from __future__ import annotations

import csv

import numpy as np

from . import data

FILE = "Training_Dataset/curated_GS_LF_merged_4983.csv"


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None


def labels_for_our_molecules() -> tuple[np.ndarray, list[str], np.ndarray]:
    """-> (cids_G, labels_L, y_GxL) with y in {0, 1}, for our molecules found in GS-LF."""
    path = data.locate("pnas_repo", needs=FILE) / FILE
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    labels_L = [k for k in rows[0] if k not in ("nonStereoSMILES", "descriptors")]
    row_of = {_canonical(r["nonStereoSMILES"]): r for r in rows}
    cids_G, y = [], []
    for cid, smiles in sorted(data.load_smiles().items()):
        r = row_of.get(_canonical(smiles))
        if r is not None:
            cids_G.append(cid)
            y.append([int(float(r[w])) for w in labels_L])
    return np.array(cids_G), labels_L, np.array(y, dtype=np.int8)
