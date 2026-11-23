# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Packaged averaged model (user, 2026-11-20: "do it"): mean rating of two O5-prior models, 3 members each:
  1. O5 (checkpoints/o3m_attpool, seeds 0-2) + prior + DREAM readout (69 numbers per member)
  2. O5 + counts + POM order 0.3 (checkpoints/ga_w64_att_rank03, seeds 0-2) + the same readout
Networks trained on GS-LF single molecules only, frozen. Readout (eval_atom_integrate.AtomReadout with prior): unit
merge; atom score = c x the network's own atom score (standardised) + q . atom vector (q pulled to 0); soft max over
atoms (tau); odd-one-out; line. Fitted on all 820 DREAM training pairs (as the held-out fit). One model for every pair
(no routing). Evidence: finding 7 section 8: CV all +0.025 vs O1.2b; held-out T46 0.481, V50 0.595. Lockbox not used.

Modes:
  fit      fit each member's readout, save checkpoints/final_o5_avg/readout.pt, then rebuild from the saved numbers and
           check T46 / V50 against the in-memory fit.
  predict  --pairs <csv> (columns a, b: CIDs separated by ';', DREAM molecules only) -> --out <csv> with column rating.

Env:     mist
Usage:   python evaluation/O3/final_o5_avg.py fit
         python evaluation/O3/final_o5_avg.py predict --pairs pairs.csv --out ratings.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from eval_atom_integrate import AtomReadout  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_variants as V5  # noqa: E402

OUT = ROOT / "checkpoints/final_o5_avg/readout.pt"
PARTS = ("O5", "O5 + counts + POM 0.3")
KEYS = ("q", "c", "t", "kappa", "lin", "bb")


def members(dev):
    V5.NETS = {k: V5.NETS[k] for k in PARTS}; BB = V5.backbones(dev)
    return {k: BB[k] for k in PARTS}


def rebuild(b, pairs, cids, saved):
    V, H, mol, sc, st = b; A, B = membership(pairs, cids); m = AtomReadout(V, H, mol, A, B, prior=sc)
    for k in KEYS:
        setattr(m, k, saved[k].double())
    with torch.no_grad():
        m.lw = m.mol_lw(m.q, m.c, m.t)
    return m


def predict(BB, saved, pairs, cids):
    idx = np.arange(len(pairs))
    parts = [np.mean([rebuild(b, pairs, cids, s).predict(idx) for b, s in zip(BB[k], saved[k])], 0) for k in PARTS]
    return np.mean(parts, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit", "predict"])
    ap.add_argument("--pairs"); ap.add_argument("--out"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; BB = members(pick_device()); cids = features.load("openpom").cids_M
    if a.mode == "fit":
        train = data.load_train(); y = evaluate.labels(train); tr = np.arange(len(train)); saved = {}
        for k in PARTS:
            saved[k] = []
            for V, H, mol, sc, st in BB[k]:
                A, B = membership(train, cids); m = AtomReadout(V, H, mol, A, B, prior=sc).fit(tr, y)
                saved[k].append({kk: getattr(m, kk).clone() for kk in KEYS})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": "average of O5 prior + DREAM and O5 + counts + POM 0.3 prior + DREAM", "parts": PARTS,
                    "fitted_on": f"{len(train)} DREAM training pairs", "members": saved}, OUT)
        print(f"saved {OUT.relative_to(ROOT)}")
        loaded = torch.load(OUT, weights_only=False)["members"]
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); te = np.arange(len(train), len(allp)); A, B = membership(allp, cids)
            p_mem = np.mean([np.mean([AtomReadout(V, H, mol, A, B, prior=sc).fit(tr, y).predict(te) for V, H, mol, sc, st in BB[k]], 0)
                             for k in PARTS], 0)
            p_pkg = predict(BB, loaded, list(hs.pairs), cids)
            print(f"{hs.name}: package r {metrics.pearson(p_pkg, hs.y_P):.3f} RMSE {metrics.rmse(p_pkg, hs.y_P):.4f}; "
                  f"max |package - fresh fit| {np.abs(p_pkg - p_mem).max():.2e}")
    else:
        rows = list(csv.DictReader(open(a.pairs)))
        pairs = [data.Pair("input", str(i), str(i), frozenset(int(c) for c in r["a"].split(";")),
                           frozenset(int(c) for c in r["b"].split(";")), float("nan")) for i, r in enumerate(rows)]
        bad = {c for p in pairs for c in p.a | p.b} - {int(c) for c in cids}
        if bad:
            sys.exit(f"molecules outside the 235 DREAM molecules (not supported): {sorted(bad)}")
        p = predict(BB, torch.load(OUT, weights_only=False)["members"], pairs, cids)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["a", "b", "rating"])
            for r, v in zip(rows, p):
                w.writerow([r["a"], r["b"], f"{v:.5f}"])
        print(f"wrote {len(p)} ratings to {a.out}")


if __name__ == "__main__":
    main()
