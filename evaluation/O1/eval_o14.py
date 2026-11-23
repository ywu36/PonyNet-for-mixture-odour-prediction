# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   K = single molecules (extra + DREAM, ~7.6k)
#   W = open-POM words (138)   E = input size   G = RDKit groups (24)   P = pairs (820)   X = mixtures (800)
# ---------------------------------------------------------------------------------------------
"""O1.4: hidden cell layer between molecules and odour words (mixdist/o14.py), declared before running.

Step 1 (no pair labels): learn cells c(m) = softplus(Enc x(m)) (64 units, L1 1e-3) and a nonlinear read-out
Dec(c) ~ q(m) on single molecules: features/openpom_extra.npz (GS-LF + GoodScents, 7,335) + the 235 DREAM molecules;
10% held out. Two inputs x: POM words q (138) and POM's own hidden embedding (256, "connect to the existing hidden").
Reported: held-out R^2, fraction of active cells; interpolation by mixture size (theta = 0): cosine between the
direction of Dec(mean c) and of mean q, and length ratio |Dec(mean c)| / |mean q|; words most raised / lowered.
Step 2 (standard CV, change from B1a++ on the same folds; B1a++ and O1.2b reused from results/eval_o12c):
  B1a++ (baseline) | O1.2b | O1.4 words in -> cells -> words | O1.4 POM embedding in -> cells -> words |
  O1.4 words in -> distance in cell space
Pairs fit only theta (lam 1e-3), beta and the line, as O1.2b. Subsets: all pairs, share no molecule; share >= 1;
share 1-2 audit; plus content with size taken out (partial r). V50/T46 not scored.

Env:     mist
Usage:   python evaluation/O1/eval_o14.py --cells ; --variant q_words | emb_words | q_cells ; --report
Writes:  results/eval_o14/cells_<q|emb>.pt, results/eval_o12c/cv_<variant>.pkl, results/eval_o14/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import evaluate, features  # noqa: E402
from mixdist.o14 import O14, train_cells, word_profiles  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402
from audit_content_no_size import wls_resid  # noqa: E402

OUT = ROOT / "results/eval_o14"; CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "o12b": "O1.2b gain + length", "q_words": "O1.4 words -> cells -> words",
         "emb_words": "O1.4 POM embedding -> cells -> words", "q_cells": "O1.4 words -> cells (distance in cells)"}


def inputs():
    op = features.load("openpom"); z = np.load(ROOT / "features/openpom_extra.npz", allow_pickle=True)
    q_M, q_K = word_profiles(op.probs_MxW, z["probs"])
    e_all = np.vstack([z["emb"], op.emb_MxE]); mu, sd = e_all.mean(0), e_all.std(0) + 1e-9
    return {"q": (np.vstack([q_K, q_M]), q_M), "emb": ((e_all - mu) / sd, (op.emb_MxE - mu) / sd)}, np.vstack([q_K, q_M]), q_M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", action="store_true"); ap.add_argument("--variant")
    ap.add_argument("--report", action="store_true"); ap.add_argument("--sparse", type=int, default=0); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    X_in, qK, qM = inputs()
    if a.sparse:                                   # O1.5 fix (2026-11-18): selective cells, top-k of 64 per molecule
        rep = Report("eval_o14_cells_sparse", {"cells": 64, "topk": a.sparse})
        rows = []
        for tk in (None, a.sparse):
            net, r2, act, ov = train_cells(X_in["q"][0], qK, log=rep.line, topk=tk)
            if tk:
                torch.save(net, OUT / "cells_q_sparse.pt")
            rows.append(["q", "dense" if tk is None else f"top {tk} of 64", f"{r2:.3f}", f"{act:.2f}", f"{ov:.2f}"])
        rep.table(["input", "cells", "held-out R^2", "fraction of cells active (> 0.1)", "mean cell-pattern cosine between molecules"], rows)
        rep.save(); return
    if a.cells:
        rep = Report("eval_o14_cells", {"cells": 64, "l1": 1e-3})
        rows = []
        for nm in ("q", "emb"):
            net, r2, act, ov = train_cells(X_in[nm][0], qK, log=rep.line)
            torch.save(net, OUT / f"cells_{nm}.pt"); rows.append([nm, "dense", f"{r2:.3f}", f"{act:.2f}", f"{ov:.2f}"])
        rep.table(["input", "cells", "held-out R^2 (reconstruct POM words)", "fraction of cells active (> 0.1)", "mean cell-pattern cosine between molecules"], rows)
        rep.save(); return

    train, y, q, A, B, S, X, C = setup()
    assert np.allclose(q, qM), "quantile profiles differ from B1a's"
    if a.variant:
        nm, space = a.variant.split("_")
        net = torch.load(OUT / f"cells_{nm}.pt", weights_only=False)
        res = evaluate.cross_validate(train, lambda tr, te: O14(net, X_in[nm][1], q, X, A, B, space, LAM).fit(tr, y[tr]).predict(te))
        pickle.dump({"res": res}, open(CV / f"cv_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_o14", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    rep.section("1  Interpolation in the cell layer (theta = 0, 800 training mixtures)")
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted)
    op = features.load("openpom"); row = {int(c): i for i, c in enumerate(op.cids_M)}
    n_X = np.array([len(m) for m in mixes]); words = op.words_W
    for nm in ("q", "emb"):
        net = torch.load(OUT / f"cells_{nm}.pt", weights_only=False)
        with torch.no_grad():
            c_M = net.cells(torch.tensor(np.asarray(X_in[nm][1], float)))
            mq = np.array([q[[row[int(c)] for c in m]].mean(0) for m in mixes])
            dq = net.dec(torch.stack([c_M[[row[int(c)] for c in m]].mean(0) for m in mixes])).numpy()
        cos = (mq * dq).sum(1) / np.linalg.norm(mq, axis=1) / np.linalg.norm(dq, axis=1)
        lr_ = np.linalg.norm(dq, axis=1) / np.linalg.norm(mq, axis=1)
        rows = []
        for lo, hi in [(1, 1), (2, 4), (5, 10), (11, 20), (21, 43)]:
            m = (n_X >= lo) & (n_X <= hi)
            rows.append([f"{lo}-{hi}", int(m.sum()), f"{np.median(cos[m]):.3f}", f"{np.median(lr_[m]):.2f}"])
        rep.line(f"input {nm}:")
        rep.table(["mixture size", "mixtures", "cos(read-out of mixed cells, mean of words)", "length ratio"], rows)
        m = (n_X >= 2) & (n_X <= 10); diff = (dq / np.linalg.norm(dq, axis=1, keepdims=True) - mq / np.linalg.norm(mq, axis=1, keepdims=True))[m].mean(0)
        o = np.argsort(diff)
        rep.line(f"  2-10 molecules, read-out vs mean of words: raised {', '.join(words[i] for i in o[::-1][:8])}; lowered {', '.join(words[i] for i in o[:8])}")

    res = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb"))["res"] for k_ in NAMES}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nN = np.array([len(p.a & p.b) for p in train])[ref.pair_N]
    rep.section(f"2  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1), ("share 1-2 (audit)", (nN >= 1) & (nN <= 2))]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a2"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in list(NAMES)[1:]:
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask)
            rows.append([NAMES[k_], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]"], rows)
    rep.section("3  Content once size is taken out (partial r)")
    w_N = evaluate.cv_weights(ref); prow = []
    for subset, m in [("share no molecule", nN == 0), ("share 1 (audit)", nN == 1)]:
        Z = np.c_[np.ones(m.sum()), S[ref.pair_N[m]]]; w = w_N[m]
        for k_ in NAMES:
            r = res[k_]; f = replace(r, pred_N=r.pred_N.copy(), truth_N=r.truth_N.copy())
            f.pred_N[m] = wls_resid(Z, r.pred_N[m], w); f.truth_N[m] = wls_resid(Z, r.truth_N[m], w)
            s = evaluate.score_cv(train, f, mask_N=m); prow.append([NAMES[k_], subset, f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]"])
    rep.table(["model", "subset", "partial r (size removed) [95% CI]"], prow)
    rep.save()


if __name__ == "__main__":
    main()
