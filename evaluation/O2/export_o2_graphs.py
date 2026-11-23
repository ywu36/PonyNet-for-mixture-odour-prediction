# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = molecules (GS-LF 4,983 then DREAM 235)   W = words (138)
#   V = atoms over all molecules   E = directed bonds over all molecules   F = atom / bond feature size
# ---------------------------------------------------------------------------------------------
"""Export molecule graphs with open-POM's own featurizer, for the PyTorch rebuild of open-POM (O2).

GS-LF molecules (open-POM's training file curated_GS_LF_merged_4983.csv, nonStereoSMILES) and the 235 DREAM
molecules (SMILES as embed_openpom.py uses them). Also the audit split of GS-LF (80 / 20, iterative
stratification order 2, as open-POM's splitter with its default seed None: deterministic). No mixture labels.

Env:     openpom
Usage:   python evaluation/O2/export_o2_graphs.py
Writes:  features/o2_graphs.npz, results/export_o2_graphs/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, features, openpom  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    from openpom.feat.graph_featurizer import GraphFeaturizer
    from skmultilearn.model_selection import IterativeStratification

    rep = Report("export_o2_graphs")
    gslf = pd.read_csv(openpom.repo() / "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv")
    Y_NxW = gslf[openpom.WORDS_W].to_numpy(np.float32)
    smiles_d = data.load_smiles(); cids = sorted(smiles_d)
    smiles_N = list(gslf.nonStereoSMILES) + [smiles_d[c] for c in cids]
    graphs = GraphFeaturizer().featurize(smiles_N)
    bad = [i for i, g in enumerate(graphs) if not hasattr(g, "node_features")]
    rep.line(f"molecules: GS-LF {len(gslf)}, DREAM {len(cids)}; failed to featurize: {len(bad)} {bad[:10]}")
    if bad:
        raise RuntimeError("featurization failed; handle before training")
    node_VxF = np.concatenate([g.node_features for g in graphs]).astype(np.float32)
    edge_ExF = np.concatenate([g.edge_features for g in graphs]).astype(np.float32)
    edge_2xE = np.concatenate([g.edge_index for g in graphs], axis=1).astype(np.int64)     # local atom indices
    node_off = np.cumsum([0] + [g.num_nodes for g in graphs]); edge_off = np.cumsum([0] + [g.num_edges for g in graphs])
    strat = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8])   # deterministic (no shuffle), as open-POM default seed None
    train_idx, test_idx = next(strat.split(np.zeros((len(gslf), 1)), Y_NxW))
    audit_test = np.zeros(len(gslf), bool); audit_test[test_idx] = True
    path = features.features_dir() / "o2_graphs.npz"
    np.savez_compressed(path, smiles=np.array(smiles_N), n_gslf=len(gslf), dream_cids=np.array(cids), Y=Y_NxW,
                        words=np.array(openpom.WORDS_W), node=node_VxF, edge=edge_ExF, edge_index=edge_2xE,
                        node_off=node_off, edge_off=edge_off, audit_test=audit_test)
    rep.line(f"atoms {len(node_VxF)} x {node_VxF.shape[1]}, directed bonds {edge_ExF.shape[0]} x {edge_ExF.shape[1]}")
    rep.line(f"audit split: train {len(train_idx)}, test {len(test_idx)}; words with 0 test positives: "
             f"{int((Y_NxW[test_idx].sum(0) == 0).sum())}")
    rep.line(f"wrote {path}")
    rep.save()


if __name__ == "__main__":
    main()
