# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Packaged fallback model (user, 2026-11-20: "have the best ready"): O5, 10 members, 4 numbers fitted on DREAM.
Network: checkpoints/o3m_attpool/member_0..9 (frozen; GS-LF single molecules only). Readout per member: molecule
direction scaled to length 1; molecule weight ∝ exp(c x standardised learned strength + kappa x cos(molecule, mean of
the other molecules)); weighted mean; rating = slope x angle-distance + intercept; ratings averaged over members.
Fitted on all 820 DREAM training pairs (data.load_train(): 730 challenge + 90 Ravia 5/6 / Metamer), the same fit as
the held-out score (finding 7 section 6: T46 0.492, V50 0.606). Lockbox not scored here.

Modes:
  fit      fit the 4 numbers per member, save checkpoints/final_o5_4/readout.json, then check that predictions rebuilt
           from the saved numbers equal the in-memory model on T46 and V50 and print r / RMSE.
  predict  --pairs <csv> (columns a, b: CIDs separated by ';', DREAM molecules only) -> --out <csv> with column rating.

Env:     mist
Usage:   python evaluation/O3/final_o5_4.py fit
         python evaluation/O3/final_o5_4.py predict --pairs pairs.csv --out ratings.csv
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("O5_SEEDS", "0 1 2 3 4 5 6 7 8 9")
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from eval_attpool import backbones, LearnedStrength, SEEDS  # noqa: E402
import eval_context_dream as ecd  # noqa: E402

OUT = ROOT / "checkpoints/final_o5_4/readout.json"
F64 = torch.float64


def build(V, st, pairs, cids, saved=None):
    A, B = membership(pairs, cids); m = LearnedStrength(V, st, A, B, True)
    if saved is not None:
        m.c, m.kappa, m.lin = (torch.tensor(saved["c"], dtype=F64), torch.tensor(saved["kappa"], dtype=F64),
                               torch.tensor(saved["lin"], dtype=F64))
        m.lw = m.c * m.S
    return m


def predict(members, saved, pairs, cids):
    return np.mean([build(V, st, pairs, cids, s).predict(np.arange(len(pairs))) for (V, H, mol, st), s in zip(members, saved)], 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit", "predict"])
    ap.add_argument("--pairs"); ap.add_argument("--out"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; members = backbones(pick_device())["attpool"]; cids = features.load("openpom").cids_M
    if a.mode == "fit":
        train = data.load_train(); y = evaluate.labels(train); saved = []; fitted = []
        for (V, H, mol, st), s in zip(members, SEEDS):
            m = build(V, st, train, cids).fit(np.arange(len(train)), y); fitted.append(m)
            saved.append({"seed": s, "c": m.c.item(), "kappa": m.kappa.item(), "lin": m.lin.tolist(),
                          "strength_mean": float(np.mean(st)), "strength_sd": float(np.std(st))})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"model": "O5 own strength + odd-one-out + unit merge (4 numbers per member)",
                                   "network": "checkpoints/o3m_attpool", "fitted_on": f"{len(train)} DREAM training pairs",
                                   "members": saved}, indent=1) + "\n")
        print(f"saved {OUT.relative_to(ROOT)}")
        loaded = json.loads(OUT.read_text())["members"]
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); te = np.arange(len(train), len(allp))
            A, B = membership(allp, cids)
            p_mem = np.mean([LearnedStrength(V, st, A, B, True).fit(np.arange(len(train)), y).predict(te)
                             for V, H, mol, st in members], 0)                                    # same as eval_attpool
            p_pkg = predict(members, loaded, list(hs.pairs), cids)
            print(f"{hs.name}: package r {metrics.pearson(p_pkg, hs.y_P):.3f} RMSE {metrics.rmse(p_pkg, hs.y_P):.4f}; "
                  f"max |package - eval_attpool fit| {np.abs(p_pkg - p_mem).max():.2e}")
    else:
        rows = list(csv.DictReader(open(a.pairs)))
        pairs = [data.Pair("input", str(i), str(i), frozenset(int(c) for c in r["a"].split(";")),
                           frozenset(int(c) for c in r["b"].split(";")), float("nan")) for i, r in enumerate(rows)]
        bad = {c for p in pairs for c in p.a | p.b} - {int(c) for c in cids}
        if bad:
            sys.exit(f"molecules outside the 235 DREAM molecules (not supported): {sorted(bad)}")
        p = predict(members, json.loads(OUT.read_text())["members"], pairs, cids)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["a", "b", "rating"])
            for r, v in zip(rows, p):
                w.writerow([r["a"], r["b"], f"{v:.5f}"])
        print(f"wrote {len(p)} ratings to {a.out}")


if __name__ == "__main__":
    main()
