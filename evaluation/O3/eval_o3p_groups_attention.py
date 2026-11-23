# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Hidden-layer attention (the other session's readout, imported unchanged: eval_context_dream.ContextReadoutQ,
direction only, exactly as eval_context_counts.py) on the node-enriched network with and without the POM-order loss:
  checkpoints/ga_w64        counts, word labels only
  checkpoints/ga_w64_rank   counts + POM-order loss (train_o3p.py --config rank --groups)
share_m ~ exp(q . s_m + kappa cos(v_m, mean of the others)); z = sum share_m v_m; d = angle; line. q 64 + kappa + line
= 67 numbers fitted on DREAM training folds; s = v / mean |v| over 2000 GS-LF training molecules (same rule as
eval_context_counts). Networks frozen; no DREAM in network training. 50 x 5, 4 subsets, RMSE and r; change from O1.2b,
and POM order minus without (same readout).

Env:     mist
Usage:   python evaluation/O3/eval_o3p_groups_attention.py
Writes:  results/eval_o3p_groups_attention/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402

SEEDS = (0, 1, 2)
NETS = {"counts": "ga_w64", "counts + POM order": "ga_w64_rank"}
REF = "O1.2b (reference, 27 numbers)"


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_groups_attention", {"nets": NETS, "seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS,
                                                                     "direction_only": True, "device": dev})
    G, tr, te, names = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows()
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); V = {}
    for k, folder in NETS.items():
        V[k] = []
        for s in SEEDS:
            n = train_groupatoms.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
                V[k].append(n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy() / scale)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share none": k_P == 0, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    res = {REF: evaluate.cross_validate(
        P, lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))}
    for k in NETS:
        res[f"{k}: attention (67)"] = evaluate.cross_validate(
            P, lambda a, b, Vs=V[k]: np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in Vs], 0))
        rep.line(f"done: {k}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if k == REF else res[REF])
            paired = ch(sc(r_, sub, res["counts: attention (67)"])) if k.startswith("counts + POM order") else ""
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if k == REF else ch(s_), paired])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "POM order minus without (same readout)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
