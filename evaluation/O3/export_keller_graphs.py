# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  K = Keller molecules (481)   V = atoms over all molecules
#   E = directed bonds over all molecules   F = atom / bond feature size   S = Keller stimuli (2 per molecule)
# ---------------------------------------------------------------------------------------------
"""Keller & Vosshall 2016 single molecules (pyrfume-data, external/keller_2016/): graphs with open-POM's own featurizer
(same layout as features/o2_graphs.npz) and per-stimulus intensity labels (measured, 55 subjects, 0-100 scale).

Labels per stimulus (molecule x dilution): mean "HOW STRONG IS THE SMELL?" over all subject ratings (numeric only),
number of ratings, log10 dilution. Also mean pleasantness (side). Raw values only, no imputation.

Env:     openpom
Usage:   python -I evaluation/O3/export_keller_graphs.py
Writes:  features/keller_graphs.npz, results/export_keller_graphs/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, features  # noqa: E402
from mixdist.report import Report  # noqa: E402

KD = ROOT / "external/keller_2016"


def main():
    from openpom.feat.graph_featurizer import GraphFeaturizer

    rep = Report("export_keller_graphs")
    mol = pd.read_csv(KD / "molecules.csv", index_col=0)
    st = pd.read_csv(KD / "stimuli.csv", index_col=0)
    b = pd.read_csv(KD / "behavior.csv", index_col=0, usecols=[0, 1, 2, 4, 5])
    b = b[b.MeasurementValue.isin(["HOW STRONG IS THE SMELL?", "HOW PLEASANT IS THE SMELL?"])].copy()
    b["v"] = pd.to_numeric(b.Value, errors="coerce")
    rep.line(f"ratings: {len(b)}; non-numeric (dropped): {int(b.v.isna().sum())}")
    g = b.dropna(subset=["v"]).groupby(["Stimulus", "MeasurementValue"]).v.agg(["mean", "count"]).unstack()
    g.columns = [f"{a}|{m}" for a, m in g.columns]
    st = st.set_index("Stimulus").join(g)
    cas2cid = dict(zip(mol.CAS, mol.CID.astype(int)))
    st["cid"] = [int(c) if str(c).isdigit() else cas2cid[c] for c in st.CIDs]
    nos = mol.CanonicalSMILES.isna(); rep.line(f"molecules without SMILES (dropped): {list(mol.OdorName[nos])}")
    mol = mol[~nos].reset_index(drop=True); st["logdil"] = np.log10(st.Concentration.astype(float))
    cids_K = mol.CID.astype(int).to_numpy()
    graphs = GraphFeaturizer().featurize(list(mol.CanonicalSMILES))
    ok = np.array([hasattr(x, "node_features") for x in graphs])
    rep.line(f"molecules {len(mol)}; failed to featurize {int((~ok).sum())}: {list(mol.OdorName[~ok])}")
    keep = np.where(ok)[0]; graphs = [graphs[i] for i in keep]; cids_K = cids_K[keep]
    smiles_K = mol.CanonicalSMILES.to_numpy()[keep]
    pos = {c: i for i, c in enumerate(cids_K)}
    st = st[st.cid.isin(pos)]
    mol_S = st.cid.map(pos).to_numpy()
    I_S = st["mean|HOW STRONG IS THE SMELL?"].to_numpy(np.float32)
    nI_S = st["count|HOW STRONG IS THE SMELL?"].to_numpy(np.int64)
    P_S = st["mean|HOW PLEASANT IS THE SMELL?"].to_numpy(np.float32)
    node_VxF = np.concatenate([x.node_features for x in graphs]).astype(np.float32)
    edge_ExF = np.concatenate([x.edge_features for x in graphs]).astype(np.float32)
    edge_2xE = np.concatenate([x.edge_index for x in graphs], axis=1).astype(np.int64)
    node_off = np.cumsum([0] + [x.num_nodes for x in graphs]); edge_off = np.cumsum([0] + [x.num_edges for x in graphs])
    dream = set(int(c) for c in data.load_smiles())
    in_dream_K = np.array([c in dream for c in cids_K])
    path = features.features_dir() / "keller_graphs.npz"
    np.savez_compressed(path, smiles=smiles_K, cids=cids_K, in_dream=in_dream_K, node=node_VxF, edge=edge_ExF,
                        edge_index=edge_2xE, node_off=node_off, edge_off=edge_off,
                        mol_S=mol_S, logdil_S=st.logdil.to_numpy(np.float32), I_S=I_S, nI_S=nI_S, P_S=P_S)
    rep.line(f"stimuli {len(mol_S)}; ratings per stimulus (intensity) min {nI_S.min()} median {int(np.median(nI_S))}")
    rep.line(f"dilutions: {st.Ratio.value_counts().to_dict()}")
    rep.line(f"intensity mean {np.nanmean(I_S):.1f}, sd {np.nanstd(I_S):.1f}; Keller molecules also in DREAM mixtures: "
             f"{int(in_dream_K.sum())} of {len(cids_K)}")
    rep.line(f"wrote {path}")
    rep.save()


if __name__ == "__main__":
    main()
