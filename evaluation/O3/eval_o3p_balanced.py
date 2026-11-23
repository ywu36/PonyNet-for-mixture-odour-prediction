# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
#   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Balanced word loss (user, 2026-11-20: remove the label-imbalance bias from the hidden layer). open-POM's loss weights
each word by log(1 + count / largest count), so common words shape the hidden layer (finding 6, section 5). The
balanced network (checkpoints/o3p_balanced, train_o3p.py --config balanced) counts every word equally and weights each
word's positive cases by (molecules without it) / (molecules with it); everything else as plain O3 (o3m_w64).
Readouts: attention (67, direction only) and O1.2b's rule (27). One model for all pairs; subsets for reporting only.
Change from O1.2b and from plain O3 with the same readout. Side notes: single-molecule AUROC (all / 8 most common /
rare words), and how strongly DREAM molecules point one way in the hidden layer (mean cosine to their average direction).

Env:     mist
Also scores any other plain-O3 variant against plain O3 (--net / --label), e.g. the word-profile order loss
(checkpoints/o3p_word_rank).

Usage:   python evaluation/O3/eval_o3p_balanced.py [--net o3p_balanced --label "plain O3, balanced word loss"]
Writes:  results/eval_o3p_balanced/<date_time>[_<net>].{txt,json}
"""
import argparse
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
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
REF = "O1.2b (reference, 27)"
NETS = {"plain O3": "o3m_w64", "plain O3, balanced word loss": "o3p_balanced"}
BASE_NET = "plain O3"


def mean_cos_to_average(D_MxH):
    u = D_MxH / np.linalg.norm(D_MxH, axis=1, keepdims=True); c = u.mean(0); c /= np.linalg.norm(c)
    return float((u @ c).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--net", default="o3p_balanced")
    ap.add_argument("--label", default="plain O3, balanced word loss"); a_ = ap.parse_args()
    NETS = {BASE_NET: "o3m_w64", a_.label: a_.net}
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_balanced", {"nets": NETS, "seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev},
                                         tag="" if a_.net == "o3p_balanced" else a_.net)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); V, Q, single = {}, {}, []
    f_W = G.Y[tr].mean(0); common = np.argsort(f_W)[-8:]; rare = np.where(f_W < 0.01)[0]
    for k, folder in NETS.items():
        ns = [train_o3m.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev) for s in SEEDS]; V[k], cs = [], []
        for n in ns:
            with torch.no_grad():
                Ts = n.latent(o2.encode_all(n, G, tr_s, dev))[0].double().cpu().numpy()
                D = n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy()
            V[k].append(D / np.linalg.norm(Ts, axis=1).mean()); cs.append(mean_cos_to_average(D))
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, G, rows_D, dev) for n in ns], 0).astype(float))
        pt = [o2.predict_probs(n, G, te, dev) for n in ns]
        single.append([k, folder, " ".join(f"{o2.mean_auroc(p, G.Y[te]):.4f}" for p in pt),
                       f"{np.mean([o2.mean_auroc(p[:, common], G.Y[te][:, common]) for p in pt]):.4f}",
                       f"{np.mean([o2.mean_auroc(p[:, rare], G.Y[te][:, rare]) for p in pt]):.4f}", f"{np.mean(cs):.3f}"])
    rep.section(f"Single molecules (official held-out; side note) and hidden-layer spread ({len(rare)} rare words = present in < 1% of training molecules)")
    rep.table(["network", "folder", "AUROC per seed", "AUROC, 8 most common words", "AUROC, rare words",
               "DREAM molecules: mean cosine to their average direction"], single)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
    SUB = {"all pairs": None, "share none": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
           "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    res = {REF: evaluate.cross_validate(
        P, lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))}
    RO = ("attention (67)", "O1.2b rule (27)")
    for k in NETS:
        res[f"{k}: {RO[0]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V[k]], 0))
        res[f"{k}: {RO[1]}"] = evaluate.cross_validate(P, lambda a, b, k=k: O12b(Q[k], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))
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
            if n != REF and not n.startswith(BASE_NET + ":"):
                base = res[f"{BASE_NET}: " + n.split(": ", 1)[1]]
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if n == REF else ch(s_), "" if base is None else ch(sc(r_, sub, base))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from plain O3 (same readout)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
