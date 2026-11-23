# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes
#   X = mixtures      W = odour words of the odor-pair network (33)      P = molecule pairs of one mixture
# ---------------------------------------------------------------------------------------------
"""Odour profiles of mixtures from team D2Smell's odor-pair network, for T46 (check) and V50.

The team made its profile file (data/processed/predictions_separated_mean_33_Augmentation_Dataset.csv)
with odor-pair-model/33label_mixture_prediction.ipynb: the saved network model_33.pt (a GIN graph
network on pairs of molecules, 33 odour words, arXiv 2312.16124) is run on every pair of molecules
inside a mixture (in the order of the definitions file; a single molecule is paired with itself),
the 33 sigmoid outputs are averaged over the pairs. This script repeats that notebook's pair loop. The
network classes (GCN, MixturePredictor) are read from the notebook and PairData from the team's
odor-pair-main.zip at run time (team_classes below), not copied here; the molecule graphs come from
ogb.utils.smiles2graph as in the team's copy of the odor-pair code. Its file has no V50 mixtures, so they are computed here.

Check: the 92 T46 ('Test') mixtures are recomputed and compared with the team's saved rows.
No labels are read.

Env:     odorpair (a clone of pommix, which has torch 2.3 and torch_geometric 2.5, plus pip ogb)
Usage:   python evaluation/D2Smell/odorpair_profiles.py
Writes:  features/odorpair_profiles.csv (Dataset, Mixture Label, Prediction_1..33; Test and V50),
         results/odorpair_profiles/<date_time>.{txt,json}
"""
import ast
import csv
import json
import sys
import zipfile
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mixdist.data import locate  # noqa: E402
from mixdist.report import Report  # noqa: E402

OUT = REPO / "features/odorpair_profiles.csv"
N_WORDS = 33
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PairData = None                                             # set by team_classes()


def _classes(source: str, names: set, space: dict, where: str) -> dict:
    """Execute only the named class definitions of ``source`` in ``space``."""
    body = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in body} != names:
        raise ImportError(f"{where}: expected classes {sorted(names)}")
    exec(compile(ast.Module(body=body, type_ignores=[]), where, "exec"), space)
    return space


def team_classes(net: Path) -> None:
    """Load the team's GCN and MixturePredictor (needed to unpickle model_33.pt, saved from a notebook, so
    looked up in __main__) and the odor-pair PairData, from the team's own files in your copy of its repo."""
    global PairData
    nb = net / "33label_mixture_prediction.ipynb"
    cell = "".join(json.load(open(nb))["cells"][0]["source"])
    main = sys.modules["__main__"].__dict__
    main.update({"torch": torch, "global_add_pool": global_add_pool, "global_mean_pool": global_mean_pool})
    _classes(cell, {"GCN", "MixturePredictor"}, main, str(nb))
    with zipfile.ZipFile(net / "odor-pair-main.zip") as z:
        src = z.read("odor-pair-main/pairing/data.py").decode()
    PairData = _classes(src, {"PairData"}, {"Data": Data, "__name__": "pairing.data"}, "pairing/data.py")["PairData"]


_graphs = {}


def graph(smiles: str) -> dict:
    if smiles not in _graphs:
        g = smiles2graph(smiles)
        _graphs[smiles] = {k: torch.tensor(g[k]) for k in ("edge_index", "edge_feat", "node_feat")}
    return _graphs[smiles]


def pair_data(s1: str, s2: str):
    g1, g2 = graph(s1), graph(s2)
    return PairData(x_s=g1["node_feat"].float(), edge_attr_s=g1["edge_feat"].float(), edge_index_s=g1["edge_index"],
                    smiles_s=s1, x_t=g2["node_feat"].float(), edge_attr_t=g2["edge_feat"].float(),
                    edge_index_t=g2["edge_index"], smiles_t=s2, y=torch.zeros(N_WORDS))


def profile(model, cids: list, smiles_of: dict) -> np.ndarray:
    """33 odour-word probabilities of one mixture: mean over its molecule pairs (the notebook's loop)."""
    smiles = [smiles_of[c] for c in cids if c in smiles_of]
    pairs = product(smiles, repeat=2) if len(smiles) == 1 else combinations(smiles, 2)
    probs_PxW = []
    for s1, s2 in pairs:
        for batch in DataLoader([pair_data(s1, s2)], batch_size=1, follow_batch=["x_s", "x_t"]):
            batch = batch.to(device)
            with torch.no_grad():
                probs_PxW.append(torch.sigmoid(model(**batch.to_dict())).cpu().numpy())
    return np.vstack(probs_PxW).mean(axis=0)


def read_mixtures(path: Path, label_col: int) -> dict:
    """{mixture label: [CIDs in file order]} from a definitions file whose CIDs follow ``label_col``."""
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))[1:]
    return {r[label_col].strip(): [int(float(c)) for c in r[label_col + 1:] if c.strip() and float(c) != 0] for r in rows}


def main():
    rep = Report("odorpair_profiles", {"device": str(device)})
    team = locate("olfboost_repo", needs="output")
    net = team / "odor-pair-model"
    team_classes(net)
    model = torch.load(net / "model_33.pt", map_location=device, weights_only=False)
    model.eval()
    smiles_of = {int(r["CID"]): r["Canonical SMILES"] for r in csv.DictReader(open(net / "CID_SMILES.csv"))}

    saved = {}                                               # the team's saved Test rows
    for r in csv.DictReader(open(team / "data/processed/predictions_separated_mean_33_Augmentation_Dataset.csv")):
        if r["Dataset"] == "Test":
            saved[r["Mixture Label"]] = np.array([float(r[f"Prediction_{i + 1}"]) for i in range(N_WORDS)])
    test_defs = {}
    for r in csv.DictReader(open(team / "data/processed/Mixure_Definitions_augmented_dataset.csv")):
        if r["Dataset"] == "Test":
            test_defs[r["Mixture Label"]] = [int(float(r[k])) for k in r if k.startswith("CID_") and r[k] not in ("", None)
                                             and float(r[k]) != 0]
    v50_defs = read_mixtures(locate("pnas_repo") / "Validation_Dataset/Mixure_Definitions_Validation_set.csv", label_col=1)

    rep.section("1. Check: the 92 T46 mixtures recomputed vs the team's saved profile file")
    test_XxW = np.array([profile(model, test_defs[m], smiles_of) for m in saved])
    saved_XxW = np.array(list(saved.values()))
    diff = np.abs(test_XxW - saved_XxW)
    rep.line(f"mixtures {len(saved)}; largest difference {diff.max():.2e}, mean {diff.mean():.2e} "
             f"(values span {saved_XxW.min():.1e} to {saved_XxW.max():.2f})")
    rep.line(f"correlation over all {diff.size} values: {np.corrcoef(test_XxW.ravel(), saved_XxW.ravel())[0, 1]:.6f}")
    rep.value("t46_check", {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())})

    rep.section("2. V50 mixtures")
    sizes = [len(c) for c in v50_defs.values()]
    missing = sorted({c for cids in v50_defs.values() for c in cids} - set(smiles_of))
    rep.line(f"mixtures {len(v50_defs)}, {min(sizes)}-{max(sizes)} molecules each; molecules without a SMILES in the "
             f"team's CID_SMILES.csv: {len(missing)}")
    v50_XxW = np.array([profile(model, cids, smiles_of) for cids in v50_defs.values()])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Mixture Label"] + [f"Prediction_{i + 1}" for i in range(N_WORDS)])
        for ds, labels, x_XxW in (("Test", list(saved), test_XxW), ("V50", list(v50_defs), v50_XxW)):
            for label, row in zip(labels, x_XxW):
                w.writerow([ds, label] + [f"{v:.9g}" for v in row])
    rep.line(f"wrote {OUT.relative_to(REPO)}: Test {len(saved)} (recomputed) + V50 {len(v50_defs)} mixtures")
    rep.save()


if __name__ == "__main__":
    main()
