# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = pairs (development + T46 + V50)   M = DREAM molecules (235)   H = hidden (64)
# ---------------------------------------------------------------------------------------------
"""Held-out check requested by the user (2026-11-19, "validate it on testing and validation set"): the frozen width-64
network, mixture = plain mean of v, no group weights, 3 numbers (beta + line) fitted on ALL development pairs (the
lockbox stays out), then scored ONCE on T46 (test) and V50 (validation). Nothing is tuned on them.
Rows: O1.2b (27 numbers, reference), open-POM baseline (2), width 64 plain mean (3, mean over 3 members), each fitted on
the same development pairs. Paired bootstrap over pairs (10,000) for the r change from O1.2b; RMSE change. T46 first.

Env:     mist
Usage:   python evaluation/O3/eval_o3m_heldout.py [--folders o3m_w64 ...]
Writes:  results/eval_o3m_heldout/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import train_o3m  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folders", nargs="+", default=["o3m_w64"]); a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_o3m_heldout", {"folders": a.folders, "lam": LAM, "device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Pd, T46, V50 = dev_pairs(), data.load_test(), data.load_validation(); allP = Pd + T46 + V50
    y = evaluate.labels(allP); tr = np.arange(len(Pd)); sets = {"T46": np.arange(len(Pd), len(Pd) + len(T46)),
                                                              "V50": np.arange(len(Pd) + len(T46), len(allP))}
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(allP, op.cids_M)
    rep.line(f"fitted on {len(Pd)} development pairs (lockbox excluded); T46 {len(T46)} pairs, V50 {len(V50)} pairs; "
             f"pairs sharing no molecule: T46 {sum(not (set(p.a) & set(p.b)) for p in T46)}, V50 {sum(not (set(p.a) & set(p.b)) for p in V50)}")
    q_op = mixture.quantile_per_word(op.probs_MxW)
    pred = {"O1.2b (reference, 27 numbers)": O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(np.arange(len(allP))),
            "open-POM baseline (2 numbers)": O12b(q_op, X, A, B, False, False, lam=LAM).fit(tr, y[tr]).predict(np.arange(len(allP)))}
    for folder in a.folders:
        Vs = []
        for f in sorted((ROOT / f"checkpoints/{folder}").glob("member_*.pt")):
            net = train_o3m.load(f, dev)
            with torch.no_grad():
                Vs.append(net.latent(o2.encode_all(net, G, G.dream_rows(), dev))[0].double().cpu().numpy())
        pred[f"{folder}, plain mean of hidden vectors (3 numbers, {len(Vs)} members)"] = np.mean(
            [O12b(v, X, A, B, False, True, lam=LAM).fit(tr, y[tr]).predict(np.arange(len(allP))) for v in Vs], 0)
    ref = pred["O1.2b (reference, 27 numbers)"]
    for name, idx in sets.items():
        rep.section(f"{name} ({len(idx)} pairs), scored once; r change from O1.2b with paired 95% CI over pairs")
        rows = []
        for k, p_ in pred.items():
            s = metrics.score(p_[idx], y[idx], reference=None if p_ is ref else ref[idx])
            rows.append([k, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         "" if p_ is ref else f"{s['rmse'] - metrics.rmse(ref[idx], y[idx]):+.4f}",
                         "" if p_ is ref else f"{s['r'] - s['r_ref']:+.3f} [{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]"])
        rep.table(["model", "RMSE", "r [95% CI]", "RMSE change from O1.2b", "r change from O1.2b [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
