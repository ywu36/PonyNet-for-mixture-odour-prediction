# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   K = mixtures   P = development pairs (514)
# ---------------------------------------------------------------------------------------------
"""O2-C gate and inside rows (design/o2c_design.md), frozen members, 50 x 5 CV on the development pairs (rule 7).

Gate (fixed before running): O2C-out vs O2-out, paired; pass = r change CI includes 0 or is above, on all pairs and
on share no molecule. Only if it passes, the inside rows are run:
  O2C-in, direction-only distance       decoder gets the merged vector with its own length
  O2C-in + read-out length              + beta (h_A - h_B)^2, h = log length of quantile-mapped profile (O1.2b)
  O2C-in + agreement                    + beta (log rho_A - log rho_B)^2
References on the same folds: B1a++, O2-out, O2N-out.

Env:     mist (GPU)
Usage:   python evaluation/O2/eval_o2c.py
Writes:  results/eval_o2c/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p_ in ("", "evaluation/B1aplus", "evaluation/O2", "training"):
    sys.path.insert(0, str(ROOT / p_))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402


def main():
    dev = pick_device(); rep = Report("eval_o2c", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); S = size_features(P); q_rel = mixture.quantile_per_word(op.probs_MxW)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    cnets, cv_, _, cprobs = members(G, "checkpoints/o2c", dev)
    _, _, _, dprobs = members(G, "checkpoints/o2", dev)
    _, _, _, nprobs = members(G, "checkpoints/o2n", dev)
    rep.line(f"development pairs {len(P)}; O2-C members {len(cnets)}")
    mixes = sorted({m for p in P for m in (p.a, p.b)}, key=sorted); mi = {m: k for k, m in enumerate(mixes)}
    rw = {int(c): i for i, c in enumerate(op.cids_M)}
    M = np.zeros((len(mixes), len(op.cids_M)), np.float32)
    for k, m in enumerate(mixes):
        M[k, [rw[c] for c in m]] = 1
    pa = np.array([mi[p.a] for p in P]); pb = np.array([mi[p.b] for p in P]); cq = o2.QuantileMap(cprobs, dev); zero = np.zeros(len(P))
    ref = evaluate.cross_validate(P, lambda tr, te: B1aPlus(q_rel, X, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = {"O2-out (retrained O2)": evaluate.cross_validate(P, evaluate.linear_calibration(mixture.paper_distance(P, op.cids_M, dprobs), y, zero)),
           "O2N-out": evaluate.cross_validate(P, evaluate.linear_calibration(mixture.paper_distance(P, op.cids_M, nprobs), y, zero)),
           "O2C-out": evaluate.cross_validate(P, evaluate.linear_calibration(mixture.paper_distance(P, op.cids_M, cprobs), y, zero))}

    def score(name, subset, against):
        mk = None if subset == "all pairs" else res[name].no_overlap_N
        return evaluate.score_cv(P, res[name], reference=against, mask_N=mk)

    gate = {sub: score("O2C-out", sub, res["O2-out (retrained O2)"]) for sub in ("all pairs", "share no molecule")}
    passed = all(g["r_change_ci"][1] >= 0 for g in gate.values())
    rep.section("Gate: O2C-out vs O2-out (paired, same development folds)")
    for sub, g in gate.items():
        rep.line(f"{sub}: r change {g['r_change']:+.3f} [{g['r_change_ci'][0]:+.3f}, {g['r_change_ci'][1]:+.3f}], RMSE change {g['rmse_change']:+.4f}")
    rep.line(f"gate {'PASSED' if passed else 'NOT passed'} (rule: CI upper bound >= 0 on both subsets)")
    if passed:
        def head(beta, length):
            return lambda tr, te: o2.O2Head(cnets, cv_, X, M, pa, pb, cq, dev, False, beta, LAM, length=length).fit(tr, y[tr]).predict(te)
        for name, f in (("O2C-in, direction-only distance", head(False, "readout")), ("O2C-in + read-out length", head(True, "readout")),
                        ("O2C-in + agreement", head(True, "agreement"))):
            res[name] = evaluate.cross_validate(P, f)
    rep.section(f"Standard CV ({evaluate.CV_REPEATS} x 5) on development pairs: change from B1a++, and from O2C-out")
    rows = []
    for sub in ("all pairs", "share no molecule"):
        s0 = evaluate.score_cv(P, ref, mask_N=None if sub == "all pairs" else ref.no_overlap_N)
        rows.append(["B1a++ (baseline)", sub, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", "", ""])
        for name in res:
            s = score(name, sub, ref); s2 = score(name, sub, res["O2C-out"]) if name.startswith("O2C-in") else None
            rows.append([name, sub, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                         "" if s2 is None else f"{s2['r_change']:+.3f} [{s2['r_change_ci'][0]:+.3f}, {s2['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]", "r change from O2C-out"], rows)
    rep.save()


if __name__ == "__main__":
    main()
