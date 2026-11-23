"""Check that the released POMMix model runs here and reproduces its authors' saved results.

For each of the 5 released cross-validation checkpoints (pommix_repo/scripts/pommix/results/
random_cv/model), run inference on that fold's test split and compare with the predictions and
Pearson the authors saved next to the checkpoint. Then run a few training steps the way
run_pommix.py does (whole training split in one batch) to see whether training fits in memory.

Last, the same 5 released checkpoints on the DREAM test set T46, as released (no retraining, no
calibration line: POMMix is trained on the same 0-1 distance labels): each checkpoint and their
mean, RMSE and Pearson r. The POMMix paper reports no T46 number, so this is not a reproduction.
With --final the same is done on V50 (scored once). POMMix trains on its own 865 pairs (Snitz 1-3,
Ravia 1/2/4, Bushdid; not the DREAM split); the script counts how many held-out mixtures appear in them.

Env:     pommix (torch, torch_geometric, POMMix code from pommix_repo/src)
Usage:   python evaluation/check_pommix.py            checks and T46
         python evaluation/check_pommix.py --final    also V50 (do this once)
Writes:  results/check_pommix/<date_time>[_final].{txt,json}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data, metrics  # noqa: E402
from mixdist.data import locate  # noqa: E402
from mixdist.report import Report  # noqa: E402

POMMIX = locate("pommix_repo")
sys.path.append(str(POMMIX / "src"))

from ml_collections import ConfigDict  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from chemix import build_chemix, get_mixture_smiles  # noqa: E402
from chemix.train import LOSS_MAP  # noqa: E402
from dataloader import DatasetLoader, SplitLoader  # noqa: E402
from dataloader.representations.graph_utils import EDGE_DIM, NODE_DIM, from_smiles  # noqa: E402
from rdkit import Chem  # noqa: E402
from pom.gnn.graphnets import GraphNets  # noqa: E402

RESULTS = POMMIX / "scripts/pommix/results/random_cv/model"
N_TRAIN_STEPS = 5


def load_models(fold, hp_gnn, hp_mix, device):
    """GNN embedder + CheMix head of one released CV fold."""
    embedder = GraphNets(node_dim=NODE_DIM, edge_dim=EDGE_DIM, **hp_gnn)
    embedder.load_state_dict(torch.load(RESULTS / f"{fold}_gnn_embedder.pt", map_location="cpu", weights_only=True))
    chemix = build_chemix(config=hp_mix.chemix)
    chemix.load_state_dict(torch.load(RESULTS / f"{fold}_chemix.pt", map_location="cpu", weights_only=True))
    return embedder.to(device), chemix.to(device)


def peak_memory_mb(device) -> float:
    if device == "cpu":
        return float("nan")
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


def reset_memory(device) -> None:
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def check_inference(rep, dl, hp_gnn, hp_mix, devices, names):
    rep.section("Inference with the released checkpoints, each on its own held-out fold")
    rows, first = [], None
    for fold, train, _val, test in SplitLoader("random_cv").load_splits(dl.features, dl.labels):
        first = first or (fold, train)
        feats, labels = test
        graphs, idx = get_mixture_smiles(feats, from_smiles)
        y = np.asarray(labels, dtype=float).flatten()
        saved = pd.read_csv(RESULTS / f"{fold}_test_predictions.csv")["Predicted_Experimental_Values"].to_numpy()
        saved_r = float(pd.read_csv(RESULTS / f"{fold}_test_metrics.csv", index_col=0).loc["pearson"].iloc[0])
        for dev in devices:
            embedder, chemix = load_models(fold, hp_gnn, hp_mix, dev)
            embedder.eval(); chemix.eval()
            batch = Batch.from_data_list(graphs).to(dev)
            reset_memory(dev)
            t = time.time()
            with torch.no_grad():
                pred = chemix(embedder.graphs_to_mixtures(batch, idx, device=dev)).flatten().cpu().numpy()
            ms = (time.time() - t) * 1000
            rows.append([f"{fold} on {dev} ({names[dev]})", len(y), f"{metrics.pearson(pred, y):.4f}",
                         f"{saved_r:.4f}", f"{np.abs(pred - saved).max():.1e}", f"{ms:.0f}",
                         f"{peak_memory_mb(dev):.0f}"])
            del embedder, chemix, batch
    rep.table(["fold and device", "n", "Pearson", "authors' Pearson", "max |pred - saved|", "ms", "peak MB"], rows)
    return first


def check_training(rep, fold, train, hp_gnn, hp_mix, devices, names):
    feats, labels = train
    graphs, idx = get_mixture_smiles(feats, from_smiles)
    rep.section(f"{N_TRAIN_STEPS} training steps as in run_pommix.py: whole training split of {fold} "
                f"({len(labels)} pairs) in one batch")
    rows = []
    for dev in devices:
        embedder, chemix = load_models(fold, hp_gnn, hp_mix, dev)
        batch = Batch.from_data_list(graphs).to(dev)
        y = torch.tensor(labels, dtype=torch.float32).to(dev)
        opt = torch.optim.Adam([{"params": embedder.parameters(), "lr": 1e-5},
                                {"params": chemix.parameters(), "lr": 1e-4}])
        loss_fn = LOSS_MAP[hp_mix.loss_type]()
        reset_memory(dev)
        embedder.train(); chemix.train()
        losses, t = [], time.time()
        for _ in range(N_TRAIN_STEPS):
            opt.zero_grad()
            loss = loss_fn(chemix(embedder.graphs_to_mixtures(batch, idx, device=dev)), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        mem = peak_memory_mb(dev)
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30 if dev != "cpu" else float("nan")
        rows.append([f"{dev} ({names[dev]})", f"{total:.1f}", f"{(time.time() - t) / N_TRAIN_STEPS * 1000:.0f}",
                     f"{mem:.0f}", f"{losses[0]:.4f} -> {losses[-1]:.4f}"])
        del embedder, chemix, batch, y, opt
    rep.table(["device", "GPU GB", "ms per step", "peak MB", "loss"], rows)


def canonical(smiles) -> frozenset:
    return frozenset(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles)


def score_held_out(rep, name, test, hp_gnn, hp_mix, device):
    """The 5 released CV checkpoints and their mean on a held-out set, outputs used as they are."""
    smiles = data.load_smiles()
    mix_Px2 = np.empty((len(test), 2), dtype=object)                 # SMILES arrays, as mix_smiles
    for i, p in enumerate(test):
        mix_Px2[i, 0] = np.array([smiles[c] for c in sorted(p.a)])
        mix_Px2[i, 1] = np.array([smiles[c] for c in sorted(p.b)])
    graphs, idx = get_mixture_smiles(mix_Px2, from_smiles)
    y_P = np.array([p.y for p in test])

    seen = pd.read_csv(POMMIX / "datasets/mixtures/mixture_smi_definitions_clean.csv")
    seen = {canonical(r.dropna().to_numpy()) for _, r in seen.filter(like="smi_").iterrows()}
    held_mixtures = {canonical(m) for row in mix_Px2 for m in row}

    folds = sorted(f.name.split("_")[0] for f in RESULTS.glob("cv*_chemix.pt"))
    batch = Batch.from_data_list(graphs).to(device)
    preds = {}
    for fold in folds:
        embedder, chemix = load_models(fold, hp_gnn, hp_mix, device)
        embedder.eval(); chemix.eval()
        with torch.no_grad():
            preds[fold] = chemix(embedder.graphs_to_mixtures(batch, idx, device=device)).flatten().cpu().numpy()
    preds["mean of the 5"] = np.mean([preds[f] for f in folds], axis=0)

    rep.section(f"Released checkpoints on {name} ({len(test)} pairs), as released: no retraining, no calibration line")
    if name == "V50":
        earlier = rep.previous_runs(final=True)
        if earlier:
            rep.line(f"NOTE: --final was already run at {', '.join(earlier)}")
    rep.line(f"{name} mixtures also in POMMix's training data: {len(held_mixtures & seen)} of {len(held_mixtures)}")
    rep.line(f"{name} mean distance {y_P.mean():.3f}")
    rows = [[name, metrics.rmse(p, y_P), metrics.pearson(p, y_P), *metrics.bootstrap_medians(p, y_P), f"{p.mean():.3f}"]
            for name, p in preds.items()]
    rep.table(["checkpoint", "RMSE", "Pearson r", "RMSE (boot. median)", "r (boot. median)", "mean prediction"], rows)
    s = metrics.score(preds["mean of the 5"], y_P)
    rep.line(f"mean of the 5: r 95% CI [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]")
    rep.value(f"{name.lower()}_predictions", {k: v.tolist() for k, v in preds.items()})   # in the order of the loader


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", action="store_true", help="also score V50 (do this once)")
    args = ap.parse_args()
    rep = Report("check_pommix", vars(args), tag="final" * args.final)
    hp_gnn = ConfigDict(json.load(open(RESULTS / "hparams_graphnets.json")))
    hp_mix = ConfigDict(json.load(open(RESULTS / "hparams_chemix.json")))
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    names = {d: (torch.cuda.get_device_name(int(d[-1])) if d != "cpu" else "CPU") for d in devices}

    dl = DatasetLoader()
    dl.load_dataset("mixtures")
    dl.featurize("mix_smiles")
    rep.line(f"POMMix dataset: {len(dl.labels)} mixture pairs")

    fold, train = check_inference(rep, dl, hp_gnn, hp_mix, devices, names)
    check_training(rep, fold, train, hp_gnn, hp_mix, devices, names)
    n_par = sum(p.numel() for m in load_models(fold, hp_gnn, hp_mix, "cpu") for p in m.parameters())
    rep.line()
    rep.line(f"model size: {n_par / 1e6:.2f} M parameters (GNN embedder + CheMix)")
    score_held_out(rep, "T46", data.load_test(), hp_gnn, hp_mix, devices[0])
    if args.final:
        score_held_out(rep, "V50", data.load_validation(), hp_gnn, hp_mix, devices[0])
    rep.save()


if __name__ == "__main__":
    main()
