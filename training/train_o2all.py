# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = DREAM development pairs (514)   W = words (138)
# ---------------------------------------------------------------------------------------------
"""O2-N-Single+Scent+Ma+DREAM: all four datasets in one training (design/o2j_design.md, "four datasets").

Streams every step: single molecules (GS-LF, open-POM loss, update everything) + GoodScents pairs (learning to rank)
+ Ma (Siamese) + DREAM development pairs of one fold (distance, MSE). Mixture gradient dial alpha = 0: the encoder
learns from single molecules only; the three mixture streams update decoder, gain and heads (the user's frozen-encoder
idea). DREAM distance = direction only (no agreement term; the user, 2026-11-18).
5 folds x 1 repeat over the development pairs (same folds as train_o2d_dev.py), 3 members per fold; lockbox untouched.

Env:     mist (GPU)
Usage:   python training/train_o2all.py --fold 0 1 2
         (report: python training/train_o2d_dev.py --report, which scores every arm on the same folds)
Writes:  checkpoints/o2all_dev/a<alpha>/fold<k>_seed<s>.pt, results/train_o2all/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "training"))
from mixdist import cv, evaluate, features, o2, o2j  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o2d_dev import SEEDS, describe, dev_pairs, save_member  # noqa: E402
from train_o2j import word_auroc  # noqa: E402

CK = ROOT / "checkpoints/o2all_es"   # with validation lines (earlier runs without: checkpoints/o2all_dev)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--arch", choices=["o2n", "o2c"], default="o2c")
    ap.add_argument("--white", action="store_true", help="agreement control (white pull) in the decoder")
    ap.add_argument("--length", choices=["none", "rho"], default="none", help="DREAM distance: direction only, or + beta (log rho) (O1.2b form)")
    ap.add_argument("--tag", default=None, help="checkpoint sub-folder (default a<alpha>)")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("train_o2all", {**vars(a), "device": dev, "seeds": SEEDS, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    P = dev_pairs(); y = evaluate.labels(P); folds = list(cv.mixture_holdout_folds(P, k=5, repeats=1, seed=0))
    gslf = np.arange(G.n_gslf); single_ids, single_test = gslf[~G.audit_test], gslf[G.audit_test]
    D = o2j.DreamStream(G, P, [int(c) for c in G.dream_cids], dev); hq = np.where(S.pair_split == 1)[0]
    out = CK / f"{a.arch}_{a.tag or f'a{a.alpha:g}'}"; out.mkdir(parents=True, exist_ok=True)
    for k in a.fold:
        tr, te = np.array(folds[k].train), np.array(folds[k].test)
        for s in SEEDS:
            if (out / f"fold{k}_seed{s}.pt").exists():
                rep.line(f"fold {k} seed {s}: exists, skipped"); continue
            t0 = time.time(); ln = a.length == "rho"; itr, iva = o2j.inner_split(P, tr)
            tk = o2j.Tracker(G, D, y, itr, iva, te, single_test, dev, length=ln)
            m, steps = o2j.train(G, S, single_ids, s, dev, arch=a.arch, log=rep.line, mix_grad=a.alpha,
                                 dream=(D, y, itr), dream_length=ln, white=a.white, on_epoch=tk)
            gs, _ = word_auroc(o2j.pair_logits(m, S, hq, dev, True), S.pair_Y[hq], S.targets)
            save_member(out / f"fold{k}_seed{s}.pt", m, tk, te, {"arch": a.arch, "white": a.white, "length": a.length, "alpha": a.alpha,
                                                                 "fold": k, "seed": s, "scent_auroc": gs})
            rep.line(f"fold {k} seed {s}: " + describe(tk) + f"; GoodScents held-out AUROC (final) {gs:.4f}; {time.time() - t0:.0f} s")
    rep.save()


if __name__ == "__main__":
    main()
