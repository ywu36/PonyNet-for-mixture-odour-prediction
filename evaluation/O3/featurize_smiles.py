# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = molecules   V = atoms over all molecules   E = directed bonds   F = feature size
# ---------------------------------------------------------------------------------------------
"""Graphs for arbitrary SMILES with open-POM's own featurizer (same layout as features/o2_graphs.npz), for the
new-molecule path of predict_o5b.py.

Env:     openpom
Usage:   python -I evaluation/O3/featurize_smiles.py in.txt out.npz      (in.txt: one "<id> <SMILES>" per line)
Writes:  out.npz with ids, smiles, node, edge, edge_index, node_off, edge_off; exits 1 naming any SMILES that fail
"""
import sys

import numpy as np


def main():
    from openpom.feat.graph_featurizer import GraphFeaturizer
    rows = [ln.split(None, 1) for ln in open(sys.argv[1]) if ln.strip()]
    ids, smiles = [r[0] for r in rows], [r[1].strip() for r in rows]
    graphs = GraphFeaturizer().featurize(smiles)
    bad = [i for i, g in zip(ids, graphs) if not hasattr(g, "node_features")]
    if bad:
        sys.exit(f"failed to featurize: {bad}")
    np.savez_compressed(sys.argv[2], ids=np.array(ids), smiles=np.array(smiles),
                        node=np.concatenate([g.node_features for g in graphs]).astype(np.float32),
                        edge=np.concatenate([g.edge_features for g in graphs]).astype(np.float32),
                        edge_index=np.concatenate([g.edge_index for g in graphs], axis=1).astype(np.int64),
                        node_off=np.cumsum([0] + [g.num_nodes for g in graphs]), edge_off=np.cumsum([0] + [g.num_edges for g in graphs]))


if __name__ == "__main__":
    main()
