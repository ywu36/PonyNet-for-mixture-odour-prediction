# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = our molecules (235)   E = embedding size (768)
# ---------------------------------------------------------------------------------------------
"""Uni-Mol2 84M, frozen public pretrained weights: CLS embedding per molecule. Nothing trained.

Uni-Mol2 has no odour-word output (no olfaction checkpoint is published), so the file holds the
embedding only (words / probs empty); experiments use ``emb``. 3D conformers by unimol_tools/RDKit.
Embeddings are cached per canonical SMILES in features/unimol2_emb_cache.npz (reused if present).

Env:     unimol (unimol_tools, torch, rdkit)
Usage:   CUDA_VISIBLE_DEVICES=1 python evaluation/embed_unimol2.py   (11 GB card; batch 4 because
         Uni-Mol2's pairwise atom maps ran out of memory at 32 on the 8 GB card)
Writes:  features/unimol2.npz (cids, smiles, words [], probs (M x 0), emb (M x 768))
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "features/unimol2_emb_cache.npz"


def canon(s):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def main():
    ours = np.load(REPO / "features/openpom.npz", allow_pickle=False)
    cids_M, smiles_M = ours["cids"], ours["smiles"]
    keys = [canon(s) for s in smiles_M]
    have = {}
    if CACHE.exists():
        z = np.load(CACHE)
        have = dict(zip([str(s) for s in z["smiles"]], z["emb"]))
    todo = sorted(set(keys) - set(have))
    if todo:
        from unimol_tools import UniMolRepr
        model = UniMolRepr(data_type="molecule", remove_hs=False, model_name="unimolv2", model_size="84m", batch_size=4)
        out = model.get_repr(todo, return_atomic_reprs=False)
        have.update(zip(todo, np.asarray(out["cls_repr"] if isinstance(out, dict) else out, dtype=np.float32)))
    emb_MxE = np.stack([have[k] for k in keys]).astype(np.float32)
    sys.path.insert(0, str(REPO))
    from mixdist import features
    f = features.save("unimol2", {"cids": cids_M, "smiles": smiles_M, "words": np.array([], dtype=str),
                                  "probs": np.zeros((len(cids_M), 0)), "emb": emb_MxE})
    print(f"{emb_MxE.shape} ({len(todo)} newly embedded) -> {f}", flush=True)


if __name__ == "__main__":
    main()
