# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   R = ensemble members (10)   M = molecules (235)
#   W = open-POM odour words (138, GS-LF labels)   E = open-POM embedding size (256)
#   C = molecules present in both our list and the paper's file
# ---------------------------------------------------------------------------------------------
"""Run the authors' pretrained open-POM ensemble (github.com/BioMachineLearning/openpom).

Open-POM is a graph neural network (MPNN) replication of the Principal Odor Map (Lee et al.
2023). The 10 models are built with the exact settings of the repo's predict_odors.py and
restored from models/ensemble_models/experiments_{1..10}/checkpoint2.pt (Git LFS files: the
real checkpoints are 25 MB each, a 133-byte file is only the LFS pointer). The ensemble output
is the mean over the 10 models, as in predict_odors.py.

Needs the `openpom` conda environment (DeepChem + DGL). Used by evaluation/embed_openpom.py.
"""
from __future__ import annotations

import csv

import numpy as np

from . import data

N_MODELS = 10

# label order of the released checkpoints (predict_odors.py)
WORDS_W = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic', 'apple', 'apricot',
    'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot', 'berry', 'bitter', 'black currant', 'brandy',
    'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy',
    'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 'coconut', 'coffee', 'cognac',
    'cooked', 'cooling', 'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal',
    'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic', 'gassy', 'geranium',
    'grape', 'grapefruit', 'grassy', 'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lily', 'malty',
    'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty',
    'natural', 'nutty', 'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone', 'peach',
    'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent', 'radish',
    'raspberry', 'ripe', 'roasted', 'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco',
    'tomato', 'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey',
    'woody']


def repo():
    return data.locate("openpom")


def build_model(i: int, device: str):
    """Ensemble member ``i`` (0-based), restored from its released checkpoint."""
    from openpom.feat.graph_featurizer import GraphConvConstants
    from openpom.models.mpnn_pom import MPNNPOMModel

    model_dir = repo() / f"models/ensemble_models/experiments_{i + 1}"
    ckpt = model_dir / "checkpoint2.pt"
    if ckpt.stat().st_size < 1_000_000:
        raise RuntimeError(f"{ckpt} is a Git LFS pointer, not the weights: fetch it with git lfs")
    model = MPNNPOMModel(
        n_tasks=len(WORDS_W), batch_size=128, class_imbalance_ratio=None, loss_aggr_type='sum',
        node_out_feats=100, edge_hidden_feats=75, edge_out_feats=100, num_step_message_passing=5,
        mpnn_residual=True, message_aggregator_type='sum', mode='classification',
        number_atom_features=GraphConvConstants.ATOM_FDIM, number_bond_features=GraphConvConstants.BOND_FDIM,
        n_classes=1, readout_type='set2set', num_step_set2set=3, num_layer_set2set=2,
        ffn_hidden_list=[392, 392], ffn_embeddings=256, ffn_activation='relu', ffn_dropout_p=0.12,
        ffn_dropout_at_input_no_act=False, weight_decay=1e-5, self_loop=False, optimizer_name='adam',
        log_frequency=32, model_dir=str(model_dir), device_name=device)
    model.restore(str(ckpt))
    return model


def run_ensemble(smiles_M: list[str], device: str = "cpu", progress=print) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """All 10 members on a list of SMILES -> (probs_RxMxW, emb_RxMxE, indices of SMILES that failed
    to featurize). Failed molecules are left out of the arrays, so rows follow the kept SMILES."""
    import deepchem as dc
    from openpom.feat.graph_featurizer import GraphFeaturizer

    graphs_M = GraphFeaturizer().featurize(smiles_M)
    bad = [i for i, g in enumerate(graphs_M) if not hasattr(g, "node_features")]
    dataset = dc.data.NumpyDataset([g for g in graphs_M if hasattr(g, "node_features")])
    n = len(dataset)
    probs_R, emb_R = [], []                                  # R entries of (M x W), (M x E)
    for i in range(N_MODELS):
        model = build_model(i, device)
        probs_R.append(np.asarray(model.predict(dataset), dtype=np.float32).reshape(n, -1))
        emb_R.append(np.asarray(model.predict_embedding(dataset), dtype=np.float32).reshape(n, -1))
        progress(f"ensemble member {i + 1}/{N_MODELS} done")
    return np.stack(probs_R), np.stack(emb_R), bad


def compute(device: str = "cpu", progress=print) -> dict:
    """Ensemble-mean odour probabilities and embeddings per molecule, plus each member's outputs.

    Returns arrays in the standard feature layout (see mixdist.features).
    """
    smiles = data.load_smiles()                              # {CID: SMILES}, M entries
    cids_M = sorted(smiles)
    smiles_M = [smiles[c] for c in cids_M]                   # as given; open-POM was trained on non-Kekule SMILES
    probs_RxMxW, emb_RxMxE, bad = run_ensemble(smiles_M, device, progress)
    if bad:
        raise ValueError(f"featurization failed for CIDs {[cids_M[i] for i in bad]}")
    return {"cids": np.array(cids_M), "smiles": np.array(smiles_M), "words": np.array(WORDS_W),
            "probs": probs_RxMxW.mean(axis=0), "emb": emb_RxMxE.mean(axis=0),
            "probs_per_model": probs_RxMxW, "emb_per_model": emb_RxMxE, "device": np.array(device)}


def compare_with_published(cids_M, probs_MxW) -> dict:
    """Our ensemble probabilities vs the paper's precomputed open-POM file
    (pnas_repo SOTA/data/raw/openpom_ensemble_predictions_results.csv), per molecule and word."""
    rel = "SOTA/data/raw/openpom_ensemble_predictions_results.csv"
    f = data.locate("pnas_repo", needs=rel) / rel
    with open(f, newline="") as fh:
        theirs = {int(r["CID"]): np.array([float(r[w]) for w in WORDS_W]) for r in csv.DictReader(fh)}
    row_of_cid = {int(c): i for i, c in enumerate(cids_M)}
    common_C = sorted(set(theirs) & set(row_of_cid))
    ours_CxW = probs_MxW[[row_of_cid[c] for c in common_C]]
    theirs_CxW = np.stack([theirs[c] for c in common_C])
    diff_CxW = np.abs(ours_CxW - theirs_CxW)
    worst = np.argsort(diff_CxW.max(axis=1))[::-1][:3]
    return {"file": str(f), "n_common": len(common_C),
            "n_ours_only": len(set(row_of_cid) - set(theirs)), "n_theirs_only": len(set(theirs) - set(row_of_cid)),
            "max_abs_diff": float(diff_CxW.max()), "mean_abs_diff": float(diff_CxW.mean()),
            "pearson_all_values": float(np.corrcoef(ours_CxW.ravel(), theirs_CxW.ravel())[0, 1]),
            "worst_cids": [(int(common_C[k]), float(diff_CxW[k].max())) for k in worst]}


def ensemble_spread(probs_per_model_RxMxW) -> float:
    """Mean |member - ensemble mean|: how much the 10 released models disagree with each other."""
    mean_MxW = probs_per_model_RxMxW.mean(axis=0)
    return float(np.abs(probs_per_model_RxMxW - mean_MxW).mean())
