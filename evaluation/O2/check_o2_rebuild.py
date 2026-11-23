# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  R = ensemble members (10)   M = DREAM molecules (235)   W = words (138)
#   E = open-POM embedding (256)   H = encoder output (400)
# ---------------------------------------------------------------------------------------------
"""O2 audits 1 and 3 (design/o2_design.md): the PyTorch rebuild of open-POM (mixdist/o2.py) with the released
checkpoints reproduces open-POM run by its own code (features/openpom.npz, per member); the set layer at n = 1
equals the plain network.

Env:     mist (GPU)
Usage:   python evaluation/O2/check_o2_rebuild.py
Writes:  results/check_o2_rebuild/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import features, o2, openpom  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    dev = pick_device(); rep = Report("check_o2_rebuild", {"device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); op = features.load("openpom")
    ref = np.load(features.path("openpom"))
    assert np.array_equal(G.dream_cids.astype(int), np.asarray(op.cids_M, int)), "molecule order differs"
    ids = G.dream_rows()
    rows = []; dp, de, dpool = [], [], []
    for i in range(openpom.N_MODELS):
        net = o2.load_released(openpom.repo() / f"models/ensemble_models/experiments_{i + 1}/checkpoint2.pt", dev)
        with torch.no_grad():
            e_MxH = o2.encode_all(net, G, ids, dev)
            emb_MxE, logits_MxW = net.readout(e_MxH)
            pooled_MxH = net.pool(e_MxH, torch.eye(len(ids), device=dev))       # every molecule as a set of size 1
            _, logits1_MxW = net.readout(pooled_MxH)
        p = torch.sigmoid(logits_MxW).cpu().numpy(); e = emb_MxE.cpu().numpy()
        dp.append(np.abs(p - ref["probs_per_model"][i]).max()); de.append(np.abs(e - ref["emb_per_model"][i]).max())
        dpool.append(float((logits1_MxW - logits_MxW).abs().max()))
        rows.append([i + 1, f"{dp[-1]:.2e}", f"{de[-1]:.2e}", f"{dpool[-1]:.2e}"])
    rep.section("Released checkpoints in the rebuild vs open-POM's own code (235 DREAM molecules)")
    rep.table(["member", "max |prob diff|", "max |embedding diff|", "set layer n = 1: max |logit diff|"], rows)
    rep.line(f"all members: probs {max(dp):.2e}, embeddings {max(de):.2e} (embedding scale: max |e| "
             f"{np.abs(ref['emb_per_model']).max():.1f}); set layer {max(dpool):.2e}")
    rep.value("max_diff", {"probs": float(max(dp)), "emb": float(max(de)), "pool_n1": float(max(dpool))})
    rep.save()


if __name__ == "__main__":
    main()
