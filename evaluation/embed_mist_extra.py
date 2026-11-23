# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = extra molecules (GS-LF + GoodScents pair molecules, unique SMILES strings)
#   W = MIST odour words (135)      E = MIST embedding size (512)
# ---------------------------------------------------------------------------------------------
"""MIST features for the extra single molecules O1 trains on (not the 235 DREAM molecules).

Same molecules and keys as embed_openpom_extra.py (GS-LF + blend-pair molecules, keyed by the SMILES
string as written in those files); each is fed to MIST as RDKit canonical Kekule SMILES
(mixdist.mist.kekulize, PLAN.md rule 9), like the DREAM molecules in features/mist.npz.

Env:     mist (torch, transformers, rdkit; GPU)
Usage:   python evaluation/embed_mist_extra.py
Writes:  features/mist_extra.npz (smiles, probs, emb, in_gslf, in_pairs),
         results/embed_mist_extra/<date_time>.{txt,json}
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mixdist import data, features, mist  # noqa: E402
from mixdist.report import Report  # noqa: E402

GSLF = "Training_Dataset/curated_GS_LF_merged_4983.csv"
PAIR_MOLECULES = REPO / "external/odor_pair/molecules.csv"


def main():
    rep = Report("embed_mist_extra")
    gslf_smiles = [r["nonStereoSMILES"] for r in csv.DictReader(open(data.locate("pnas_repo", needs=GSLF) / GSLF))]
    pair_smiles = [r["smiles"] for r in csv.DictReader(open(PAIR_MOLECULES))]
    smiles_M = sorted(set(gslf_smiles) | set(pair_smiles))
    rep.line(f"GS-LF {len(gslf_smiles)}, pair molecules {len(pair_smiles)}, unique SMILES {len(smiles_M)}")
    logits_MxW, emb_MxE, bad = mist.run_model(smiles_M)
    if bad:
        rep.line(f"RDKit could not parse ({len(bad)}), left out: {[smiles_M[i] for i in bad]}")
    kept_M = [s for i, s in enumerate(smiles_M) if i not in set(bad)]
    out = features.features_dir() / "mist_extra.npz"
    np.savez_compressed(out, smiles=np.array(kept_M), probs=1.0 / (1.0 + np.exp(-logits_MxW)), emb=emb_MxE,
                        in_gslf=np.isin(kept_M, gslf_smiles), in_pairs=np.isin(kept_M, pair_smiles))
    rep.line(f"wrote {out}: {len(kept_M)} molecules")
    rep.save()


if __name__ == "__main__":
    main()
