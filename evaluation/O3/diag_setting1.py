# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = DREAM development pairs
#   T = held-out GS-LF molecules   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Why do the counts (and the Edge Transformer) fail on setting 1 (pairs sharing no molecule)? (user, 2026-11-19:
"is it no normalization of the merging, too many molecules tend to white? or failing to attend to the important one?")
No new fitting under CV; diagnostics only (development pairs; no held-out sets).

A. Where the error sits (CV predictions saved by eval_context_counts.py, averaged over repeats per pair), setting 1:
   r within mixture-size bins; correlation of the residual (label - prediction) with mean log size and with
   |log n_A - log n_B|. A whiteness / size problem shows as errors that grow with size or size mismatch.
B. The map itself, per frozen backbone:
   B1 smell map on unseen molecules: Spearman between hidden cosine distance and odour-label distance (1 - Jaccard)
      over pairs of GS-LF held-out molecules (50,000 sampled); and with open-POM's embedding distance on DREAM molecules.
   B2 whiteness: mean cosine of DREAM molecule vectors to their common mean; plain-mean mixture angle distance on
      setting-1 pairs: spread, and r with the label by size bin (no fitting at all).
C. The attention, fitted once on all development pairs (training fit, not a test): effective number of molecules
   (1 / sum share^2) relative to mixture size; largest share; and setting-1 training r when fitted on all pairs vs on
   setting-1 pairs only (does serving shared-molecule pairs pull the query away from what setting 1 needs?).

Env:     mist
Usage:   python evaluation/O3/diag_setting1.py
Writes:  results/diag_setting1/<date_time>.{txt,json}
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, o2  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_backbones  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
CV_PKL = ROOT / "results/eval_context_counts/2026-11-19_215950_cv.pkl"


def size_bins(nA, nB):
    m = (nA + nB) / 2
    return {"mean size <= 5": m <= 5, "mean size 6-15": (m > 5) & (m <= 15), "mean size > 15": m > 15}


def r_(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("diag_setting1", {"seeds": SEEDS, "cv_predictions": str(CV_PKL.name), "device": dev})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom")
    A, B = membership(P, op.cids_M); nA, nB = A.sum(1), B.sum(1)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); s1 = k_P == 0
    bins = size_bins(nA, nB)
    rep.line(f"development pairs {len(P)}; setting 1 {int(s1.sum())}; " +
             ", ".join(f"{b}: {int((m & s1).sum())}" for b, m in bins.items()) + " (setting-1 pairs per bin)")

    # ---- A. where the error sits
    d = pickle.load(open(CV_PKL, "rb")); res = d["res"]
    rows = []
    for name, cvr in res.items():
        pred = np.bincount(cvr.pair_N, cvr.pred_N, len(P)) / np.maximum(np.bincount(cvr.pair_N, minlength=len(P)), 1)
        res_P = y - pred; m = s1
        row = [name, f"{r_(pred[m], y[m]):.3f}"]
        row += [f"{r_(pred[m & bm], y[m & bm]):.3f}" for bm in bins.values()]
        row += [f"{r_(res_P[m], np.log((nA + nB)[m] / 2)):+.3f}", f"{r_(res_P[m], np.abs(np.log(nA[m] / nB[m]))):+.3f}",
                f"{res_P[m].std():.4f}"]
        rows.append(row)
    rep.section("A. Setting 1, CV predictions averaged per pair: r overall and by size bin; residual vs size")
    rep.table(["model", "r setting 1"] + [f"r {b}" for b in bins] +
              ["corr(residual, mean log size)", "corr(residual, |log size ratio|)", "residual SD"], rows)

    # ---- B. the map
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr, te, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    nets = {"message passing, no counts": ([train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G),
            "message passing + counts": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg),
            "edge transformer": ([train_backbones.load(ROOT / f"checkpoints/bb_et/member_{s}.pt", dev) for s in SEEDS], G),
            "message passing, POM order": ([train_o3m.load(ROOT / f"checkpoints/o3p_rank/member_{s}.pt", dev) for s in SEEDS], G)}
    rng = np.random.default_rng(1); i_T, j_T = rng.integers(0, len(te), 60000), rng.integers(0, len(te), 60000)
    keep = i_T != j_T; i_T, j_T = i_T[keep][:50000], j_T[keep][:50000]
    Yt = G.Y[te].astype(bool); inter = (Yt[i_T] & Yt[j_T]).sum(1); union = np.maximum((Yt[i_T] | Yt[j_T]).sum(1), 1)
    lab_dist = 1 - inter / union
    E = op.emb_MxE / np.linalg.norm(op.emb_MxE, axis=1, keepdims=True); iu = np.triu_indices(len(E), 1); pom_dist = 1 - (E @ E.T)[iu]
    unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: E731
    brow, crow, V_all = [], [], {}
    for k, (ns, g) in nets.items():
        sp_lab, sp_pom, white, Vs = [], [], [], []
        for n in ns:
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                Vt = n.latent(o2.encode_all(n, g, te, dev))[0].cpu().numpy()
                Vd = n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy() / scale
            Ut = unit(Vt); sp_lab.append(spearmanr(1 - (Ut[i_T] * Ut[j_T]).sum(1), lab_dist)[0])
            Ud = unit(Vd); sp_pom.append(spearmanr(1 - (Ud @ Ud.T)[iu], pom_dist)[0])
            white.append(float((Ud @ unit(Vd.mean(0, keepdims=True)).T).mean())); Vs.append(Vd)
        V_all[k] = Vs
        # plain-mean mixture angle distance on setting-1 pairs (no fitting)
        dist = []
        for Vd in Vs:
            zA, zB = unit(A @ Vd), unit(B @ Vd); dist.append(np.sqrt(np.maximum(2 - 2 * (zA * zB).sum(1), 0)))
        dist = np.mean(dist, 0)
        brow.append([k, f"{np.mean(sp_lab):.3f}", f"{np.mean(sp_pom):.3f}", f"{np.mean(white):.3f}",
                     f"{dist[s1].mean():.3f} / {dist[s1].std():.3f}", f"{-r_(dist[s1], y[s1]):.3f}"]
                    + [f"{-r_(dist[s1 & bm], y[s1 & bm]):.3f} (spread {dist[s1 & bm].std():.3f})" for bm in bins.values()])
    rep.section("B. The map (mean of 3 members). Higher Spearman = distances follow smell; whiteness = mean cos to the common mean")
    rep.table(["backbone", "Spearman vs label distance (held-out GS-LF)", "Spearman vs open-POM distance (DREAM)",
               "whiteness", "plain-mean distance setting 1: mean / SD", "plain-mean r setting 1 (no fit)"] +
              [f"plain-mean r, {b}" for b in bins], brow)

    # ---- C. the attention (training fits)
    idx_all, idx_s1 = np.arange(len(P)), np.where(s1)[0]
    for k, Vs in V_all.items():
        n_eff, top, r_all, r_s1 = [], [], [], []
        for Vd in Vs:
            m_all = ecd.ContextReadoutQ(Vd, A, B, np.zeros(len(Vd)), True).fit(idx_all, y)
            m_s1 = ecd.ContextReadoutQ(Vd, A, B, np.zeros(len(Vd)), True).fit(idx_s1, y[idx_s1])
            r_all.append(r_(m_all.predict(idx_s1), y[idx_s1])); r_s1.append(r_(m_s1.predict(idx_s1), y[idx_s1]))
            with torch.no_grad():
                for Mx, C in ((m_all.A, m_all.CA), (m_all.B, m_all.CB)):
                    lw = m_all.lw.unsqueeze(0) + m_all.kappa * C; lw = lw - lw.max(1, keepdim=True).values
                    W = Mx * torch.exp(lw); S_ = (W / W.sum(1, keepdim=True)).numpy(); n = Mx.sum(1).numpy()
                    big = n >= 4
                    n_eff.append(float(np.mean((1 / (S_ ** 2).sum(1))[big] / n[big]))); top.append(float(np.mean(S_.max(1)[big])))
        crow.append([k, f"{np.mean(n_eff):.3f}", f"{np.mean(top):.3f}", f"{np.mean(r_all):.3f}", f"{np.mean(r_s1):.3f}",
                     f"{np.mean(r_s1) - np.mean(r_all):+.3f}"])
    rep.section("C. Attention fitted on all development pairs (training fit, not a test); mixtures with >= 4 molecules")
    rep.table(["backbone", "effective molecules / size", "largest share", "setting-1 r, fitted on all pairs",
               "setting-1 r, fitted on setting 1 only", "difference"], crow)
    rep.save()


if __name__ == "__main__":
    main()
