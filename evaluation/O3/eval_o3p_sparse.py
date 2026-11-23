# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Receptor-like (positive + sparse) hidden layer on the counts network, with and without POM order (user, 2026-11-19:
"try all positive hidden layer ... with the best setting"). Networks (frozen, single molecules only):
  ga_w64            counts, centred hidden layer (current; reference for the paired change)
  ga_w64_sp03       counts, ReLU + activity penalty 0.3          ga_w64_sp03_rank   + POM order
  ga_w64_sp1        counts, ReLU + activity penalty 1.0          ga_w64_sp1_rank    + POM order
Readouts: plain mean (3), O1.2b rule on word profiles (27), attention (67, direction only, other session's code).
50 x 5; 4 subsets; RMSE and r; change from O1.2b and from the centred counts network with the same readout.

Env:     mist
Usage:   python evaluation/O3/eval_o3p_sparse.py
Writes:  results/eval_o3p_sparse/<date_time>.{txt,json}
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
NETS = {"counts, centred": "ga_w64", "counts, sparse 0.3": "ga_w64_sp03", "counts, sparse 0.3 + POM order": "ga_w64_sp03_rank",
        "counts, sparse 1.0": "ga_w64_sp1", "counts, sparse 1.0 + POM order": "ga_w64_sp1_rank"}


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_sparse", {"nets": NETS, "seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G, tr, te, names = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows()
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); V, Vraw, Q, single = {}, {}, {}, []
    for k, folder in NETS.items():
        ld = train_groupatoms.load if folder == "ga_w64" else train_o3m.load
        ns = [ld(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev) for s in SEEDS]; V[k], Vraw[k] = [], []
        for n in ns:
            with torch.no_grad():
                Ts = n.latent(o2.encode_all(n, G, tr_s, dev))[0].double().cpu().numpy()
                D = n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy()
            V[k].append(D / np.linalg.norm(Ts, axis=1).mean()); Vraw[k].append(D)
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, G, rows_D, dev) for n in ns], 0).astype(float))
        auc = [o2.mean_auroc(o2.predict_probs(n, G, te, dev), G.Y[te]) for n in ns]
        single.append([k, folder, " ".join(f"{x:.4f}" for x in auc), f"{np.mean([(v < -1e-6).mean() for v in Vraw[k]]):.2f}",
                       f"{np.mean([(np.abs(v) <= 1e-6).mean() for v in Vraw[k]]):.2f}", f"{np.mean([(v > 1e-6).sum(1).mean() for v in Vraw[k]]):.1f}"])
    rep.section("Single molecules (official held-out) and hidden layer on DREAM molecules")
    rep.table(["network", "folder", "held-out AUROC per seed", "values < 0", "exact zeros", "active units per molecule"], single)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share none": k_P == 0, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    res = {REF: evaluate.cross_validate(
        P, lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))}
    RO = ("plain mean (3)", "O1.2b rule (27)", "attention (67)")
    for k in NETS:
        res[f"{k}: {RO[0]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(a, y[a]).predict(b) for v in Vraw[k]], 0))
        res[f"{k}: {RO[1]}"] = evaluate.cross_validate(P, lambda a, b, k=k: O12b(Q[k], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))
        res[f"{k}: {RO[2]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V[k]], 0))
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
            s_ = sc(r_, sub, None if n == REF else res[REF]); base = None
            if n != REF and not n.startswith("counts, centred"):
                base = res["counts, centred: " + n.split(": ", 1)[1]]
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if n == REF else ch(s_), "" if base is None else ch(sc(r_, sub, base))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from counts, centred (same readout)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
