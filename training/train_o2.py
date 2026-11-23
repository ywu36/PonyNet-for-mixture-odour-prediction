# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = held-out GS-LF molecules (1,008)   W = words (138)   R = members
# ---------------------------------------------------------------------------------------------
"""Train O2 step 1 (design/o2_design.md): open-POM rebuilt in PyTorch (mixdist/o2.py), trained on single
molecules with open-POM's recipe, 80% of GS-LF (the audit split of evaluation/O2/export_o2_graphs.py).
No mixture data. Scoring on DREAM pairs is done by evaluation/O2/eval_o2.py.

Audit 2: held-out 20% AUROC (mean over the 138 words) per member and for the ensemble; the released members
reached 0.873-0.881 each on their own 80 / 20 split (open-POM examples/ensemble_benchmark.ipynb).

Env:     mist (GPU)
Usage:   python training/train_o2.py --seeds 0 1 2 3 4        (one process per GPU is fine)
         python training/train_o2.py --report                 (ensemble AUROC of all saved members)
Writes:  checkpoints/o2/member_<seed>.pt, results/train_o2/<date_time>.{txt,json}
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

CKPT = ROOT / "checkpoints/o2"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=[]); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("train_o2", {"seeds": a.seeds, "report": a.report, "device": dev, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); train_ids, test_ids = gslf[~G.audit_test], gslf[G.audit_test]
    rep.line(f"GS-LF train {len(train_ids)}, held out {len(test_ids)}; device {dev}")
    CKPT.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        t0 = time.time()
        net, steps = o2.train_member(G, train_ids, seed, dev, log=rep.line)
        auc = o2.mean_auroc(o2.predict_probs(net, G, test_ids, dev), G.Y[test_ids])
        torch.save({"model_state_dict": net.state_dict(), "seed": seed, "steps": steps, "held_out_auroc": auc}, CKPT / f"member_{seed}.pt")
        rep.line(f"seed {seed}: steps {steps}, held-out AUROC {auc:.4f}, {time.time() - t0:.0f} s")
    if a.report:
        files = sorted(CKPT.glob("member_*.pt")); probs_R, rows = [], []
        for f in files:
            c = torch.load(f, map_location="cpu", weights_only=False)
            net = o2.O2Net().to(dev); net.load_state_dict(c["model_state_dict"])
            probs_R.append(o2.predict_probs(net, G, test_ids, dev)); rows.append([c["seed"], c["steps"], f"{c['held_out_auroc']:.4f}"])
        ens = o2.mean_auroc(np.mean(probs_R, 0), G.Y[test_ids])
        rep.section("Audit 2: held-out 20% AUROC (mean over words)")
        rep.table(["seed", "steps", "held-out AUROC"], rows)
        single = [float(r[2]) for r in rows]
        rep.line(f"members {len(rows)}: single {min(single):.4f}-{max(single):.4f} (mean {np.mean(single):.4f}); ensemble {ens:.4f}")
        rep.line("released members on their own split: 0.873-0.881 single (ensemble_benchmark.ipynb)")
        rep.value("auroc", {"single": single, "ensemble": ens})
    rep.save()


if __name__ == "__main__":
    main()
