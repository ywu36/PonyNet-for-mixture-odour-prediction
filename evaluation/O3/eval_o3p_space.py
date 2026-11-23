# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
#   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Why does O1.2b's rule win share none and attention win share >= 3 on the SAME network? (user, 2026-11-19: "think and
see which is the reason"). O1.2b's rule differs from attention in three ways; change one at a time, same frozen network:
  1 O1.2b rule          word profiles (quantile per word), RDKit group weights, size term          27 numbers
  2 word profiles, no group weights (plain mean, size term)                                        3
  3 attention shares, mixture built and compared in word-profile space (diagnosis only: merge at
    word level, which O3 does not allow)                                                          67
  4 attention (hidden space, angle)                                                               67
  5 COMPATIBLE: attention, merge in the hidden layer, compared through the linear word head with
    every word put on the same scale (logit minus its mean, divided by its spread over 2000 GS-LF
    training molecules). Head is linear, so the hidden merge commutes with it; nothing decoded per
    molecule; 0 extra numbers                                                                     67
Networks: counts (ga_w64) and counts + POM order (ga_w64_rank), frozen, singles only. 50 x 5; 4 subsets; RMSE and r;
change from O1.2b.

Env:     mist
Usage:   python evaluation/O3/eval_o3p_space.py
Writes:  results/eval_o3p_space/<date_time>.{txt,json}
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
from eval_o3p_twoview import TwoView  # noqa: E402
import train_groupatoms  # noqa: E402

SEEDS = (0, 1, 2)
REF = "O1.2b (reference: released open-POM profiles, 27)"


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_space", {"seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G, tr, te, names = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows()
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    nets = {"counts": "ga_w64", "counts + POM order": "ga_w64_rank"}; V, L, Q = {}, {}, {}
    for k, folder in nets.items():
        V[k], L[k] = [], []
        ns = [train_groupatoms.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev) for s in SEEDS]
        for n in ns:
            with torch.no_grad():
                Ts = n.latent(o2.encode_all(n, G, tr_s, dev))[0].double().cpu().numpy()
                D = n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy()
            W = n.head.weight.detach().double().cpu().numpy(); b = n.head.bias.detach().double().cpu().numpy()
            lt, lD = Ts @ W.T + b, D @ W.T + b
            V[k].append(D / np.linalg.norm(Ts, axis=1).mean()); L[k].append((lD - lt.mean(0)) / lt.std(0).clip(1e-6))
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, G, rows_D, dev) for n in ns], 0).astype(float))
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share none": k_P == 0, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    rows_m = {REF: lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b)}
    for k in nets:
        rows_m[f"{k}: 1 O1.2b rule (27)"] = lambda a, b, k=k: O12b(Q[k], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b)
        rows_m[f"{k}: 2 word profiles, no group weights (3)"] = lambda a, b, k=k: O12b(Q[k], X, A, B, False, True, lam=LAM).fit(a, y[a]).predict(b)
        rows_m[f"{k}: 3 attention, word-profile space (67, diagnosis)"] = lambda a, b, k=k: np.mean(
            [TwoView(v, Q[k], A, B).fit(a, y[a]).predict(b) for v in V[k]], 0)
        rows_m[f"{k}: 4 attention, hidden angle (67)"] = lambda a, b, k=k: np.mean(
            [ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V[k]], 0)
        rows_m[f"{k}: 5 attention, hidden merge, equal-word yardstick (67)"] = lambda a, b, k=k: np.mean(
            [TwoView(v, l, A, B).fit(a, y[a]).predict(b) for v, l in zip(V[k], L[k])], 0)
    res = {}
    for n, f in rows_m.items():
        res[n] = evaluate.cross_validate(P, f); rep.line(f"done: {n}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for n, r_ in res.items():
            s_ = sc(r_, sub, None if n == REF else res[REF])
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if n == REF else ch(s_)])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b"], rows)
    rep.save()


if __name__ == "__main__":
    main()
