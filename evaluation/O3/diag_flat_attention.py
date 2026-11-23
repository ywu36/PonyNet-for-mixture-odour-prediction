# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = DREAM development pairs
#   G = RDKit groups
# ---------------------------------------------------------------------------------------------
"""Why is the hidden-layer attention flat while O1.2b's group weights are not? (user, 2026-11-20: "why does O1.2b
work, why B1a++, did our network design miss an important part, like normalization?")
C27/C28: normalisation changes nothing; setting 1 needs unequal molecule weights; attention stays ~uniform.
Suspect: the same penalty 1e-3 |.|^2 acts on very different scales. O1.2b's weight exp(theta . x) reads group
counts (0, 1, 2 ...), so a small theta gives a 7x range; attention exp(q . s) reads s = v / mean|v| (length ~1 spread
over 64 units), so the same range needs a much longer q, which the penalty forbids.
Training fits on all development pairs only (diagnostic, not a test). Plain O3 w64 (o3m_w64), 3 seeds.
  1. attention + kappa (angle only) at penalty 1e-3 (as used), 1e-4, 1e-5, 0: |q|, penalty / squared error,
     effective molecules / size and top share / uniform (mixtures >= 4), weight range across the 235 molecules
     (max/min of exp(q.s)), Spearman of q.s with O1.2b's log weight, training r all / setting 1.
  2. O1.2b (penalty 1e-3) the same numbers.
  3. Is O1.2b's log weight linearly in the hidden layer? ridge from v (64) to theta . x, 5-fold over molecules, R^2.

Env:     mist
Usage:   python evaluation/O3/diag_flat_attention.py
Writes:  results/diag_flat_attention/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
LAMS = (1e-3, 1e-4, 1e-5, 0.0)


def conc(lw_M, A, B):
    """effective molecules / size and top share / (1/n), mixtures with >= 4 molecules, both sides of every pair."""
    ne, top = [], []
    for Mx in (A, B):
        W = Mx * np.exp(lw_M - lw_M.max()); S = W / W.sum(1, keepdims=True); n = Mx.sum(1); big = n >= 4
        ne.append((1 / (S ** 2).sum(1))[big] / n[big]); top.append(S.max(1)[big] * n[big])
    return float(np.mean(np.concatenate(ne))), float(np.mean(np.concatenate(top)))


def r_(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("diag_flat_attention", {"seeds": SEEDS, "lams": LAMS, "device": dev})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    A, B = membership(P, op.cids_M); k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); s1 = k_P == 0
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    idx = np.arange(len(P))

    o12 = O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=1e-3).fit(idx, y)
    th = o12.theta.numpy(); lw_o = X @ th; p_o = o12.predict(idx); mse_o = float(np.mean((p_o - y) ** 2))
    ne, tp = conc(lw_o, A, B)
    rep.line(f"X (group counts): mean |x| per molecule {np.linalg.norm(X, axis=1).mean():.2f}; per-feature SD mean {X.std(0).mean():.3f}")
    rows = [["O1.2b group weights (penalty 1e-3)", f"|theta| {np.linalg.norm(th):.2f}", f"{1e-3 * (th ** 2).sum() / mse_o:.3f}",
             f"{ne:.3f}", f"{tp:.2f}", f"{np.exp(lw_o.max() - lw_o.min()):.1f}", "1", f"{r_(p_o, y):.3f}", f"{r_(p_o[s1], y[s1]):.3f}"]]

    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    Vs = []
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
            Vs.append(n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy() / scale)
    sd_dir = np.mean([np.linalg.svd(V - V.mean(0), compute_uv=False)[0] / np.sqrt(len(V)) for V in Vs])
    sd_rand = np.mean([(V @ (e / np.linalg.norm(e))).std() for V in Vs for e in np.random.default_rng(0).normal(size=(20, V.shape[1]))])
    rep.line(f"s = v / mean|v|: mean |s| {np.mean([np.linalg.norm(V, axis=1).mean() for V in Vs]):.2f}; SD of s along a random "
             f"direction {sd_rand:.3f}, along the widest direction {sd_dir:.3f}  (log-weight range ln 7 = 1.95 needs |q| ~ 2 / (4 SD))")

    for lam in LAMS:
        st = []
        for V in Vs:
            m = ecd.ContextReadoutQ(V, A, B, np.zeros(len(V)), True); m.lam = lam; m.fit(idx, y)
            q = m.q.numpy(); lw = V @ q; p = m.predict(idx); mse = float(np.mean((p - y) ** 2))
            ne, tp = conc(lw, A, B)   # self part only (kappa context left out of the concentration)
            st.append([np.linalg.norm(q), lam * ((q ** 2).sum() + m.kappa.item() ** 2) / mse, ne, tp, np.exp(lw.max() - lw.min()),
                       spearmanr(lw, lw_o)[0], r_(p, y), r_(p[s1], y[s1])])
        st = np.mean(st, 0)
        rows.append([f"attention + kappa, penalty {lam:g}", f"|q| {st[0]:.2f}", f"{st[1]:.3f}", f"{st[2]:.3f}", f"{st[3]:.2f}",
                     f"{st[4]:.1f}", f"{st[5]:+.2f}", f"{st[6]:.3f}", f"{st[7]:.3f}"])
        rep.line(f"done penalty {lam:g}")
    rep.section("1-2. Fitted on all development pairs (training fit; mean of 3 members for attention). Concentration over mixtures >= 4")
    rep.table(["readout", "weight-vector length", "penalty / squared error", "effective molecules / size", "top share x size",
               "max/min weight over 235 molecules", "Spearman log weight vs O1.2b", "training r all", "training r setting 1"], rows)

    # 3. is theta . x linear in v?
    r2 = []
    for V in Vs:
        f = np.random.default_rng(0).permutation(len(V)) % 5; pred = np.zeros(len(V))
        for k in range(5):
            tr_, te_ = f != k, f == k; Vt = np.c_[V[tr_], np.ones(tr_.sum())]
            w = np.linalg.solve(Vt.T @ Vt + 1e-2 * np.eye(Vt.shape[1]), Vt.T @ lw_o[tr_]); pred[te_] = np.c_[V[te_], np.ones(te_.sum())] @ w
        r2.append(1 - ((pred - lw_o) ** 2).sum() / ((lw_o - lw_o.mean()) ** 2).sum())
    rep.section("3. O1.2b's log weight (theta . x) predicted from the hidden vector by ridge, 5-fold over the 235 molecules")
    rep.line(f"R^2 per member: {', '.join(f'{v:.2f}' for v in r2)}")
    rep.save()


if __name__ == "__main__":
    main()
