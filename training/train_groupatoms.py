# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  V = atoms   F = atom features (134 open-POM + 36 groups x 5 distance bins)
# ---------------------------------------------------------------------------------------------
"""Single-molecule networks WITH per-atom group-distance counts (mixdist/group_atoms.py): same data, same held-out
molecules (GS-LF audit split), same recipe and seeds as the networks trained without them, so the comparison is paired:
  --arch o2c : open-POM layout (O2-C, as checkpoints/o2c)            -> checkpoints/ga_o2c/member_<s>.pt
  --arch w64 : the minimal width-64 network (as checkpoints/o3m_w64)  -> checkpoints/ga_w64/member_<s>.pt
Groups kept: RDKit fr_* patterns present in >= 1% of the TRAINING molecules (36 of 85).

Env:     mist (GPU)
Usage:   python training/train_groupatoms.py --arch o2c --seeds 0 1 2
Writes:  checkpoints/ga_<arch>/member_<s>.pt, results/train_groupatoms/<date_time>_<arch>.{txt,json}
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
from mixdist import features, group_atoms, o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
import train_o3m  # noqa: E402

MIN_FRAC = 0.01


def graphs_with_groups():
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    names = group_atoms.attach(G, tr, MIN_FRAC)
    return G, tr, te, names


def make(arch, atom_f):
    if arch == "o2c":
        return o2.O2C(atom_f=atom_f)
    return train_o3m.O3M(**{**train_o3m_w64(), "atom_f": atom_f})


def train_o3m_w64():
    return {"d": 64, "h": 64, "steps": 3, "edge_h": 16}


def load(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    net = make(c["arch_name"], c["atom_f"]).float(); net.load_state_dict(c["model_state_dict"]); return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", choices=["o2c", "w64"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args(); dev = pick_device()
    rep = Report("train_groupatoms", {**vars(a), "device": dev, "min_frac": MIN_FRAC, **o2.RECIPE}, tag=a.arch)
    G, tr, te, names = graphs_with_groups(); atom_f = G.node.shape[1]
    rep.line(f"groups kept {len(names)}: {', '.join(n.replace('fr_', '') for n in names)}; atom features {atom_f}; "
             f"training molecules {len(tr)}, held-out {len(te)}")
    tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    ck = ROOT / f"checkpoints/ga_{a.arch}"; ck.mkdir(parents=True, exist_ok=True); rows = []
    for s in a.seeds:
        t0 = time.time()
        net, steps = o2.train_member(G, tr, s, dev, log=rep.line, make_net=lambda: make(a.arch, atom_f))
        auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te])
        auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
        torch.save({"model_state_dict": net.state_dict(), "arch_name": a.arch, "atom_f": atom_f, "groups": names, "seed": s,
                    "steps": steps, "single_auroc": auc_te, "train_auroc": auc_tr}, ck / f"member_{s}.pt")
        rows.append([s, f"{auc_te:.4f}", f"{auc_tr:.4f}", f"{auc_tr - auc_te:+.4f}", f"{time.time() - t0:.0f}"])
        rep.line(f"{a.arch} + groups seed {s}: held-out AUROC {auc_te:.4f}, training {auc_tr:.4f}")
    rep.section(f"{a.arch} with group-distance counts: held-out vs training-molecule AUROC")
    rep.table(["seed", "held-out AUROC", "training AUROC", "gap", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
