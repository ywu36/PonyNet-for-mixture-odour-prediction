# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""One network that keeps both: counts in the input, POM order on only the first 32 of 64 hidden units (the other 32
free for the counts' chemistry and strength) -- checkpoints/ga_w64_rank_half (train_o3p.py --config rank_half --groups).
Compared with plain O3 (o3m_w64), counts (ga_w64), counts + POM order on all units (ga_w64_rank); readouts attention
(67, direction only) and O1.2b's rule (27). One model for all pairs; subsets for reporting only.
New subset (2026-11-20): "share none, same size (gap <= 1)" -- T46 is all 10 vs 10 and V50 is 9-11 vs 9-11, while 105 of
the 186 development share-none pairs differ in size by >= 2 and the size gap alone correlates with the rating (r 0.23);
the same-size share-none pairs (81) are the closest development stand-in for the held-out sets.

Env:     mist
Usage:   python evaluation/O3/eval_o3p_half.py
Writes:  results/eval_o3p_half/<date_time>.{txt,json}
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
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
REF = "O1.2b (reference, 27)"
NETS = {"plain O3": "o3m_w64", "counts": "ga_w64", "counts + POM order (all units)": "ga_w64_rank",
        "counts + POM order (32 of 64 units)": "ga_w64_rank_half"}


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_half", {"nets": NETS, "seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    Gg, tr, te, names = train_groupatoms.graphs_with_groups(); G0 = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    rows_D = G0.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); V, Q, single = {}, {}, []
    for k, folder in NETS.items():
        g = G0 if folder == "o3m_w64" else Gg; ld = train_groupatoms.load if folder == "ga_w64" else train_o3m.load
        ns = [ld(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev) for s in SEEDS]; V[k] = []
        for n in ns:
            with torch.no_grad():
                Ts = n.latent(o2.encode_all(n, g, tr_s, dev))[0].double().cpu().numpy()
                D = n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy()
            V[k].append(D / np.linalg.norm(Ts, axis=1).mean())
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, g, rows_D, dev) for n in ns], 0).astype(float))
        single.append([k, folder, " ".join(f"{o2.mean_auroc(o2.predict_probs(n, g, te, dev), g.Y[te]):.4f}" for n in ns)])
    rep.section("Single molecules (official held-out AUROC per seed)")
    rep.table(["network", "folder", "AUROC"], single)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G0.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
    SUB = {"all pairs": None, "share none": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
           "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    res = {REF: evaluate.cross_validate(
        P, lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))}
    for k in NETS:
        res[f"{k}: attention (67)"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V[k]], 0))
        res[f"{k}: O1.2b rule (27)"] = evaluate.cross_validate(P, lambda a, b, k=k: O12b(Q[k], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))
        rep.line(f"done: {k}")

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
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", "" if n == REF else ch(s_)])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b"], rows)
    rep.save()


if __name__ == "__main__":
    main()
