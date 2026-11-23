# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = molecules in the file (odor_pair, then Ma)   W = open-POM words (138)
#   Q = odor_pair pairs kept   T = Ma mixtures   V / E = atoms / directed bonds over all molecules   F = feature size
# ---------------------------------------------------------------------------------------------
"""Mixture training data for O2-J (joint training): graphs (open-POM's featurizer) and labels.

GoodScents blender pairs (external/odor_pair): each pair's blend groups mapped to open-POM words, exact name
(78 of 108 groups) plus a fixed synonym list (SYN below, chosen before any training); other groups dropped; pairs
with no mapped group dropped. Label = count per word (from "group:count"). Molecule-disjoint held-out split: 10% of
the odor_pair molecules (random, seed 0); a pair is "train" if both molecules are train molecules, "test" if both
are held-out, otherwise unused.
Ma 2021 (external/ma_2021): per mixture L = log((IAmix/IA) / (IBmix/IB)), means over judges (subject 47 removed,
IA > 0, IB > 0), as findings 4-Ma and eval_o12b.py. Molecule folds for unseen-molecule CV: 5 folds, random, seed 0.
No DREAM labels.

Env:     openpom
Usage:   python evaluation/O2/export_o2j_data.py
Writes:  features/o2j_data.npz, results/export_o2j_data/<date_time>.{txt,json}
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import features, openpom  # noqa: E402
from mixdist.report import Report  # noqa: E402

SYN = {"minty": "mint", "anise": "anisic", "licorice": "anisic", "medicinal,": "medicinal", "toasted": "roasted",
       "acidic": "sour", "acetic": "sour", "moldy": "musty", "fungal": "mushroom", "mentholic": "cooling"}


def main():
    from openpom.feat.graph_featurizer import GraphFeaturizer
    rep = Report("export_o2j_data", {"synonyms": SYN})
    W = {w: i for i, w in enumerate(openpom.WORDS_W)}
    op_mol = pd.read_csv(ROOT / "external/odor_pair/molecules.csv")
    pairs = pd.read_csv(ROOT / "external/odor_pair/pairs.csv")
    ma_mol = pd.read_csv(ROOT / "external/ma_2021/pyrfume_molecules.csv").sort_values("CID").reset_index(drop=True)
    smiles_N = list(op_mol.smiles) + list(ma_mol.IsomericSMILES)
    graphs = GraphFeaturizer().featurize(smiles_N)
    bad = [i for i, g in enumerate(graphs) if not hasattr(g, "node_features")]
    rep.line(f"molecules: odor_pair {len(op_mol)}, Ma {len(ma_mol)}; failed to featurize {len(bad)}")
    ok_N = np.array([hasattr(g, "node_features") for g in graphs])
    empty = lambda: type("G", (), {"node_features": np.zeros((0, 134), np.float32), "edge_features": np.zeros((0, 6), np.float32),  # noqa: E731
                                   "edge_index": np.zeros((2, 0), np.int64), "num_nodes": 0, "num_edges": 0})()
    graphs = [g if ok else empty() for g, ok in zip(graphs, ok_N)]

    row = {s: i for i, s in enumerate(op_mol.smiles)}
    a_Q, b_Q, lab = [], [], []
    used, dropped = Counter(), Counter()
    for sa, sb, grp in zip(pairs.smiles_a, pairs.smiles_b, pairs.blender_groups.astype(str)):
        y = np.zeros(len(W), np.float32)
        for t in grp.split(";"):
            name, _, cnt = t.partition(":"); name = name.strip(); name = SYN.get(name, name)
            if name in W:
                y[W[name]] += float(cnt or 1); used[name] += 1
            else:
                dropped[name] += 1
        ia, ib = row.get(sa), row.get(sb)
        if y.sum() > 0 and ia is not None and ib is not None and ok_N[ia] and ok_N[ib]:
            a_Q.append(ia); b_Q.append(ib); lab.append(y)
    a_Q, b_Q, Y_QxW = np.array(a_Q), np.array(b_Q), np.stack(lab)
    held = np.random.default_rng(0).random(len(op_mol)) < 0.10
    split_Q = np.where(~held[a_Q] & ~held[b_Q], 0, np.where(held[a_Q] & held[b_Q], 1, -1))   # 0 train, 1 test, -1 unused
    targets = sorted({W[n] for n in used})
    rep.line(f"pairs kept {len(a_Q)} of {len(pairs)}; target words {len(targets)}; train {int((split_Q == 0).sum())}, "
             f"held-out (both molecules unseen) {int((split_Q == 1).sum())}, unused {int((split_Q == -1).sum())}; "
             f"held-out molecules {int(held.sum())}")
    rep.line(f"groups dropped (no word): {dict(dropped.most_common())}")

    r = pd.read_csv(ROOT / "external/ma_2021/pyrfume_behavior.csv"); r = r[(r.Subject != 47) & (r.IA > 0) & (r.IB > 0)]
    g = r.groupby(["Stimulus A", "Stimulus B"])[["IA", "IAmix", "IB", "IBmix"]].mean().reset_index()
    g = g[(g.IAmix > 0) & (g.IBmix > 0)]
    L_T = np.log((g.IAmix / g.IA) / (g.IBmix / g.IB)).values
    ix = {c: len(op_mol) + i for i, c in enumerate(ma_mol.CID)}
    ma_a, ma_b = g["Stimulus A"].map(ix).values, g["Stimulus B"].map(ix).values
    ma_fold = np.random.default_rng(0).integers(0, 5, len(ma_mol))
    rep.line(f"Ma mixtures {len(L_T)} (L mean {L_T.mean():+.3f}, sd {L_T.std():.3f}); molecule folds sizes {np.bincount(ma_fold).tolist()}")

    node = np.concatenate([g_.node_features for g_ in graphs]).astype(np.float32)
    edge = np.concatenate([g_.edge_features for g_ in graphs]).astype(np.float32)
    eidx = np.concatenate([g_.edge_index for g_ in graphs], axis=1).astype(np.int64)
    node_off = np.cumsum([0] + [g_.num_nodes for g_ in graphs]); edge_off = np.cumsum([0] + [g_.num_edges for g_ in graphs])
    path = features.features_dir() / "o2j_data.npz"
    np.savez_compressed(path, smiles=np.array(smiles_N), ok=ok_N, n_odorpair=len(op_mol), node=node, edge=edge, edge_index=eidx,
                        node_off=node_off, edge_off=edge_off, pair_a=a_Q, pair_b=b_Q, pair_Y=Y_QxW, pair_split=split_Q,
                        targets=np.array(targets), ma_a=ma_a, ma_b=ma_b, ma_L=L_T, ma_cids=ma_mol.CID.values, ma_fold=ma_fold)
    rep.line(f"wrote {path}")
    rep.save()


if __name__ == "__main__":
    main()
