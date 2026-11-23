# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Combine the two networks at the prediction, not inside one space (user, 2026-11-19: "attention helps share >= 3,
counts + POM order helps share none, combine should beat all; strike balance").
Why not one network: POM order pulls every molecule toward one shared direction (mean cosine to the average direction
0.22 -> 0.39) -- good for placing different molecules (share none), bad for telling apart mixtures that share most of
their molecules (share >= 3). So keep both frozen networks and average their predicted distances:
  counts (ga_w64)             + attention (67 numbers, the other session's readout, unchanged)   strong on share >= 3
  counts + POM order (ga_w64_rank) + O1.2b's rule on its word profiles (27)                     strong on share none
Combination = plain average of the two predictions (no extra fitted number, no weight tuned on DREAM). Each part is
fitted on the same training fold; 50 x 5; 4 subsets; RMSE and r; change from O1.2b and from each part.

Env:     mist
Usage:   python evaluation/O3/eval_o3p_combine.py
Writes:  results/eval_o3p_combine/<date_time>.{txt,json}
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
REF = "O1.2b (reference, 27)"


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_combine", {"seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G, tr, te, names = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows()
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); V, Q = {}, {}
    for k, folder in (("counts", "ga_w64"), ("counts + POM order", "ga_w64_rank")):
        ns = [train_groupatoms.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev) for s in SEEDS]; V[k] = []
        for n in ns:
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
                V[k].append(n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy() / scale)
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, G, rows_D, dev) for n in ns], 0).astype(float))
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share none": k_P == 0, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}

    def att(k, a, b):
        return np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V[k]], 0)

    def rule(k, a, b):
        return O12b(Q[k], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b)
    parts = {"counts: attention (67)": lambda a, b: att("counts", a, b),
             "counts + POM order: O1.2b rule (27)": lambda a, b: rule("counts + POM order", a, b),
             "counts + POM order: attention (67)": lambda a, b: att("counts + POM order", a, b),
             "counts: O1.2b rule (27)": lambda a, b: rule("counts", a, b)}
    cache = {}

    def run(a, b):
        key = (np.asarray(a).tobytes(), np.asarray(b).tobytes())    # every fold computed once, reused by every row
        if key not in cache:
            cache[key] = {n: f(a, b) for n, f in parts.items()}
        return cache[key]
    res = {REF: evaluate.cross_validate(
        P, lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))}
    combos = {"average: counts attention + counts+POM O1.2b rule": ["counts: attention (67)", "counts + POM order: O1.2b rule (27)"],
              "average: counts attention + counts+POM attention": ["counts: attention (67)", "counts + POM order: attention (67)"],
              "average: all four": list(parts)}
    out = {n: [] for n in list(parts) + list(combos)}

    def collect(a, b):
        r_ = run(a, b)
        return r_
    # one pass over folds; each part and each average gets its own CVResult on identical folds
    for n in list(parts) + list(combos):
        members = [n] if n in parts else combos[n]
        res[n] = evaluate.cross_validate(P, lambda a, b, m=members: np.mean([collect(a, b)[x] for x in m], 0))
        rep.line(f"done: {n}")

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
            vs_att = ch(sc(r_, sub, res["counts: attention (67)"])) if n in combos else ""
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if n == REF else ch(s_), vs_att])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from counts: attention"], rows)
    rep.save()


if __name__ == "__main__":
    main()
