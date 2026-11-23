# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = held-out GS-LF molecules (1,008)   W = words (138)   R = members
# ---------------------------------------------------------------------------------------------
"""Train O2-N (design/o2n_design.md): O2 with the merge-layer normalization (centred unit directions; decoder sees
direction only), single molecules, open-POM's recipe, 80% of GS-LF. No mixture data. Scored by
evaluation/O2/eval_o2n.py.

Fit check: held-out 20% AUROC (mean over the 138 words) per member and for the ensemble; the released members
reached 0.873-0.881 each on their own 80 / 20 split (open-POM examples/ensemble_benchmark.ipynb).

Env:     mist (GPU)
Usage:   python training/train_o2n.py --seeds 0 1 2 3 4        (one process per GPU is fine)
         python training/train_o2n.py --report                 (ensemble AUROC of all saved members)
Writes:  checkpoints/o2n/member_<seed>.pt, results/train_o2n/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixdist import features, o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402

CKPT = ROOT / "checkpoints/o2n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=[]); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("train_o2n", {"seeds": a.seeds, "report": a.report, "device": dev, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); train_ids, test_ids = gslf[~G.audit_test], gslf[G.audit_test]
    rep.line(f"GS-LF train {len(train_ids)}, held out {len(test_ids)}; device {dev}")
    CKPT.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        t0 = time.time()
        net, steps = o2.train_member(G, train_ids, seed, dev, log=rep.line, make_net=o2.O2N)
        auc = o2.mean_auroc(o2.predict_probs(net, G, test_ids, dev), G.Y[test_ids])
        torch.save({"model_state_dict": net.state_dict(), "arch": "o2n", "seed": seed, "steps": steps, "held_out_auroc": auc}, CKPT / f"member_{seed}.pt")
        rep.line(f"seed {seed}: steps {steps}, held-out AUROC {auc:.4f}, {time.time() - t0:.0f} s")
    if a.report:
        files = sorted(CKPT.glob("member_*.pt")); probs_R, rows = [], []
        for f in files:
            c = torch.load(f, map_location="cpu", weights_only=False)
            net = o2.load_members([f], dev)[0]
            probs_R.append(o2.predict_probs(net, G, test_ids, dev)); rows.append([c["seed"], c["steps"], f"{c['held_out_auroc']:.4f}"])
        ens = o2.mean_auroc(np.mean(probs_R, 0), G.Y[test_ids])
        rep.section("Fit check: held-out 20% AUROC (mean over words)")
        rep.table(["seed", "steps", "held-out AUROC"], rows)
        single = [float(r[2]) for r in rows]
        rep.line(f"members {len(rows)}: single {min(single):.4f}-{max(single):.4f} (mean {np.mean(single):.4f}); ensemble {ens:.4f}")
        rep.line("references: released members 0.873-0.881 single; O2 0.875-0.883 single, 0.894 ensemble")
        rep.value("auroc", {"single": single, "ensemble": ens})
    rep.save()


if __name__ == "__main__":
    main()
