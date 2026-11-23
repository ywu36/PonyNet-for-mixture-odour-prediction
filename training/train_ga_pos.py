# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Counts + positive / sparse hidden layer (the user, 2026-11-19: "all positive + attention ... and the new feature").

Same as training/train_groupatoms.py --arch w64 (node-enriched atom input, GS-LF single-molecule words, o2.train_member),
but the hidden layer is receptor-like, as train_o3m.py's w64pos / w64sparse1:
  pos      v = softplus(.) >= 0, not centred (a mixture adds activations)
  sparse1  v = ReLU(.) >= 0 with an L1 activity penalty 1.0 (each unit fires for fewer molecules)
Read out later with the attention readout (evaluation/O3/eval_o3g.py --set pos). The checkpoint stores the full layout
("arch"), so it loads with O3M(**arch).

Env:     mist (GPU)
Usage:   python training/train_ga_pos.py --variant pos|sparse1 --seeds 0 1 2
Writes:  checkpoints/ga_w64_<variant>/member_<s>.pt, results/train_ga_pos/<date_time>_<variant>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o3m import O3M  # noqa: E402
import train_groupatoms  # noqa: E402

VARIANTS = {"pos": {"positive": True}, "sparse1": {"positive": True, "sparse_l1": 1.0}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args(); dev = pick_device()
    G, tr, te, names = train_groupatoms.graphs_with_groups()
    arch = {**train_groupatoms.train_o3m_w64(), **VARIANTS[a.variant], "atom_f": G.node.shape[1]}
    rep = Report("train_ga_pos", {**vars(a), "arch": arch, "device": dev, **o2.RECIPE}, tag=a.variant)
    ck = ROOT / f"checkpoints/ga_w64_{a.variant}"; ck.mkdir(parents=True, exist_ok=True); rows = []
    tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    for s in a.seeds:
        t0 = time.time()
        net, steps = o2.train_member(G, tr, s, dev, log=rep.line, make_net=lambda: O3M(**arch), beta_kl=arch.get("sparse_l1", 0.0))
        auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te]); auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
        with torch.no_grad():
            v = net.latent(o2.encode_all(net, G, te, dev))[0]
        active = float((v > 1e-3).float().sum(1).mean())
        torch.save({"model_state_dict": net.state_dict(), "arch": arch, "arch_name": "w64", "atom_f": arch["atom_f"], "groups": names,
                    "seed": s, "steps": steps, "single_auroc": auc_te, "train_auroc": auc_tr, "active_units": active}, ck / f"member_{s}.pt")
        rows.append([a.variant, s, f"{auc_te:.4f}", f"{auc_tr - auc_te:+.4f}", f"{active:.1f}", f"{time.time() - t0:.0f}"])
        rep.line(f"{a.variant} seed {s}: held-out AUROC {auc_te:.4f}, active units {active:.1f} of 64")
    rep.section(f"counts + {a.variant}: single molecules")
    rep.table(["variant", "seed", "held-out AUROC", "gap", "active units (of 64, held-out molecules)", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
