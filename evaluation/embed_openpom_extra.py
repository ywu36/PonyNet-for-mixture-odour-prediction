# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = extra molecules (GS-LF + GoodScents pair molecules, unique SMILES strings)
#   W = open-POM odour words (138)      E = open-POM embedding size (256)
# ---------------------------------------------------------------------------------------------
"""Open-POM features for the extra single molecules O1 trains on (not the 235 DREAM molecules).

Molecules: the GS-LF odour-label set (pnas_repo/Training_Dataset/curated_GS_LF_merged_4983.csv,
column nonStereoSMILES) and the GoodScents blend-pair molecules (external/odor_pair/molecules.csv).
Same 10-model ensemble and averaging as embed_openpom.py; molecules that fail to featurize are
listed and left out. Rows are keyed by the SMILES string exactly as written in those files.

Env:     openpom (CPU; roughly 30 minutes for ~7k molecules)
Usage:   OSMO_DATA=~/.cache/osmo-data-mirror python evaluation/embed_openpom_extra.py
Writes:  features/openpom_extra.npz (smiles, probs, emb, in_gslf, in_pairs),
         results/embed_openpom_extra/<date_time>.{txt,json}
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mixdist import data, features, openpom  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402

GSLF = "Training_Dataset/curated_GS_LF_merged_4983.csv"
PAIR_MOLECULES = REPO / "external/odor_pair/molecules.csv"


def main():
    rep = Report("embed_openpom_extra")
    gslf_smiles = [r["nonStereoSMILES"] for r in csv.DictReader(open(data.locate("pnas_repo", needs=GSLF) / GSLF))]
    pair_smiles = [r["smiles"] for r in csv.DictReader(open(PAIR_MOLECULES))]
    smiles_M = sorted(set(gslf_smiles) | set(pair_smiles))
    rep.line(f"GS-LF {len(gslf_smiles)}, pair molecules {len(pair_smiles)}, unique SMILES {len(smiles_M)}")
    probs_RxMxW, emb_RxMxE, bad = openpom.run_ensemble(smiles_M, pick_device(), progress=rep.line)
    if bad:
        rep.line(f"failed to featurize ({len(bad)}), left out: {[smiles_M[i] for i in bad]}")
    kept_M = [s for i, s in enumerate(smiles_M) if i not in set(bad)]
    out = features.features_dir() / "openpom_extra.npz"
    np.savez_compressed(out, smiles=np.array(kept_M), words=np.array(openpom.WORDS_W),
                        probs=probs_RxMxW.mean(axis=0), emb=emb_RxMxE.mean(axis=0),
                        in_gslf=np.isin(kept_M, gslf_smiles), in_pairs=np.isin(kept_M, pair_smiles))
    rep.line(f"wrote {out}: {len(kept_M)} molecules")
    rep.save()


if __name__ == "__main__":
    main()
