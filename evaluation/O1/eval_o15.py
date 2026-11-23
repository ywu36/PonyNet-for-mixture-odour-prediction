# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   H = cells (64)   G = RDKit groups (24)   P = pairs (820)
#   N = pooled out-of-fold predictions   I = (shared molecule, side) instances
# ---------------------------------------------------------------------------------------------
"""O1.5: per-cell divisive normalization in the hidden cell layer (mixdist/o14.py O15), declared before running.

Structure gives the cells (O1.4 words-input layer, frozen, learned from single molecules), normalization is the mixing
rule (a molecule is divided by the activity of the cells it drives), the metric compares (direction + length).
Rows, change from B1a++ on the same folds (B1a++, O1.2b reused from results/eval_o12c):
  B1a++ (baseline) | O1.2b | O1.5 per-cell normalization (cells) | control: same normalization over the 138 POM word
  probabilities instead of cells (does the learned cell layer matter?)
Subsets: all pairs, share no molecule; share >= 1; share 1-2 audit; partial r with size taken out.
Checks against what is known about shared molecules (b1a+ session, declared before running):
  (a) contrast: r(out-of-fold error, pair contrast) on pairs sharing >= 1, error = rating - prediction; contrast as
      b1a+ (|q_s - mean q of the mixture's unshared molecules|, z over instances, 0 if none, mean per pair).
      B1a++ leaves a negative r (standing-out shared molecule -> rated closer); a mechanism that captures it -> ~0.
  (b) b1a+'s grid lead: mean error by mixture size (5-9, 10-19, 20+) x shared count k (1, 2, 3-5, 6+).
  (c) full data: rho (two starts), beta (does the length term become unnecessary?); a shared molecule's relative
      weight in its mixture vs its contrast (Spearman); profile length by mixture size.
GPU (mixdist.device.pick_device).

Env:     mist
Usage:   python evaluation/O1/eval_o15.py --variant cells | words | sparse ; then --report
Writes:  results/eval_o12c/cv_o15_<variant>.pkl ; report results/eval_o15/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import evaluate, features  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o14 import O15  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402
from eval_o14 import inputs  # noqa: E402
from audit_content_no_size import wls_resid  # noqa: E402

CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "o12b": "O1.2b gain + length", "o15_cells": "O1.5 per-cell normalization",
         "o15_words": "control: normalization over POM words",
         "o15_sparse": "O1.5 fix: sparse cells (top 8 of 64)"}   # added 2026-11-18 after the dense run: dense cells overlap 0.88


def cells_for(variant):
    if variant == "words":
        return features.load("openpom").probs_MxW
    net = torch.load(ROOT / ("results/eval_o14/cells_q_sparse.pt" if variant == "sparse" else "results/eval_o14/cells_q.pt"), weights_only=False)
    X_in, _, _ = inputs()
    with torch.no_grad():
        return net.cells(torch.tensor(np.asarray(X_in["q"][1], float))).numpy()


def pair_contrast(train, q, cids):
    row = {int(c): i for i, c in enumerate(cids)}; inst = []          # (pair, contrast or nan)
    for p_, pr in enumerate(train):
        for mine in (pr.a, pr.b):
            un = [row[c] for c in mine - (pr.a & pr.b)]
            for s in pr.a & pr.b:
                inst.append((p_, np.linalg.norm(q[row[s]] - q[un].mean(0)) if un else np.nan))
    P_I = np.array([i for i, _ in inst]); v = np.array([c for _, c in inst])
    z = (v - np.nanmean(v)) / np.nanstd(v); z[np.isnan(z)] = 0.0
    out = np.full(len(train), np.nan)
    for p_ in np.unique(P_I):
        out[p_] = z[P_I == p_].mean()
    return out, P_I, z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup()
    dev = pick_device(); print("device", dev, flush=True)
    if a.variant:
        c = cells_for(a.variant)
        res = evaluate.cross_validate(train, lambda tr, te: O15(c, q, X, A, B, LAM, device=dev).fit(tr, y[tr]).predict(te))
        pickle.dump({"res": res}, open(CV / f"cv_o15_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_o15", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    allp = np.arange(len(train)); op = features.load("openpom")
    rep.section("0  Audit")
    c = cells_for("cells"); f0 = O15(c, q, X, A, B, LAM, device=dev)
    from mixdist.o12 import O12b
    th = torch.zeros(X.shape[1]); k = torch.tensor(0.7)
    d0 = f0.distance(th.to(dev), k.to(dev), torch.as_tensor(allp, device=dev), torch.tensor(0.0, device=dev)).cpu()
    rep.line(f"O1.5 at rho = 0 vs O1.2b distance: max |difference| {float((d0 - O12b(q, X, A, B).distance(th, k, allp)).abs().max()):.2e}")

    res = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb"))["res"] for k_ in NAMES}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nsh = np.array([len(p.a & p.b) for p in train]); nN = nsh[ref.pair_N]
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1), ("share 1-2 (audit)", (nN >= 1) & (nN <= 2))]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a2"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in list(NAMES)[1:]:
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask)
            rows.append([NAMES[k_], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]"], rows)
    w_N = evaluate.cv_weights(ref); prow = []
    for subset, m in [("share no molecule", nN == 0), ("share 1 (audit)", nN == 1)]:
        Z = np.c_[np.ones(m.sum()), S[ref.pair_N[m]]]; w = w_N[m]
        for k_ in NAMES:
            r = res[k_]; f = replace(r, pred_N=r.pred_N.copy(), truth_N=r.truth_N.copy())
            f.pred_N[m] = wls_resid(Z, r.pred_N[m], w); f.truth_N[m] = wls_resid(Z, r.truth_N[m], w)
            s = evaluate.score_cv(train, f, mask_N=m); prow.append([NAMES[k_], subset, f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]"])
    rep.table(["model", "subset", "partial r (size removed) [95% CI]"], prow)

    rep.section("2  Checks against the shared-molecule findings (out-of-fold error = rating - mean prediction per pair)")
    con_P, P_I, z_I = pair_contrast(train, q, op.cids_M)
    err = {}
    for k_, r in res.items():
        num = np.bincount(r.pair_N, weights=r.pred_N, minlength=len(train)); den = np.bincount(r.pair_N, minlength=len(train))
        err[k_] = y - num / np.maximum(den, 1)
    sh = nsh >= 1
    rep.table(["model", "r(error, contrast), share >= 1 (495 pairs)", "Spearman"],
              [[NAMES[k_], f"{np.corrcoef(err[k_][sh], con_P[sh])[0, 1]:+.3f}", f"{spearmanr(err[k_][sh], con_P[sh])[0]:+.3f}"] for k_ in NAMES])
    nbar = np.array([(len(p.a) + len(p.b)) / 2 for p in train]); grid = []
    for lo, hi in [(5, 9.5), (10, 19.5), (20, 99)]:
        for klo, khi in [(1, 1), (2, 2), (3, 5), (6, 99)]:
            m = (nbar >= lo) & (nbar < hi) & (nsh >= klo) & (nsh <= khi)
            if m.sum() >= 5:
                grid.append([f"{lo}-{hi if hi < 99 else '+'}", f"{klo}-{khi if khi < 99 else '+'}", int(m.sum())] + [f"{err[k_][m].mean():+.3f}" for k_ in NAMES])
    rep.table(["mean size", "shared k", "pairs"] + [f"mean error {NAMES[k_]}" for k_ in NAMES], grid)

    for cv_ in ("cells", "sparse"):
        full_data(rep, cv_, train, y, q, X, A, B, op, z_I, dev, allp)
    rep.save()


def full_data(rep, variant, train, y, q, X, A, B, op, z_I, dev, allp):
    c = cells_for(variant)
    rep.section(f"3  Full-data fit (O1.5, {variant} cells)")
    fits = {r0: O15(c, q, X, A, B, LAM, r0=r0, device=dev).fit(allp, y) for r0 in (0.1, 1.0)}
    f = fits[0.1]
    rep.line("rho (start r = 0.1 / 1.0): " + " / ".join(f"{fits[r0].rho:.4f}" for r0 in fits)
             + f";  beta {float(f.k ** 2):.3f} (O1.2b full data 1.091);  in-sample r {np.corrcoef(f.predict(allp), y)[0, 1]:.3f}")
    with torch.no_grad():
        idx = torch.as_tensor(allp, device=dev)
        eA = f.weights(f.theta, f.rho, f.A, idx).cpu().numpy(); eB = f.weights(f.theta, f.rho, f.B, idx).cpu().numpy()
        G = torch.exp(f.x @ f.theta).cpu().numpy()
    row = {int(cc): i for i, cc in enumerate(op.cids_M)}; rel = []
    for p_, pr in enumerate(train):
        for mine, e in ((pr.a, eA[p_]), (pr.b, eB[p_])):
            ii = [row[cc] for cc in mine]; base = e[ii] / G[ii]
            for s in pr.a & pr.b:
                rel.append(base[ii.index(row[s])] / base.mean())
    rel = np.array(rel)
    rep.line(f"shared molecule's suppression relative to its mixture's mean (gain removed) vs contrast, {len(rel)} instances: "
             f"Spearman {spearmanr(rel, z_I)[0]:+.3f}  (> 0 = standing-out shared molecules keep more weight)")
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); n_X = np.array([len(m) for m in mixes])
    supp = []
    for m in mixes:
        ii = [row[cc] for cc in m]; Sx = (G[ii, None] * c[ii]).sum(0)
        supp.append(np.mean((1 / (1 + f.rho * Sx)) @ (c[ii] / c[ii].sum(1, keepdims=True)).T))
    supp = np.array(supp)
    rep.line("mean suppression factor by mixture size: " + ", ".join(
        f"n {lo}-{hi} {np.median(supp[(n_X >= lo) & (n_X <= hi)]):.2f}" for lo, hi in [(1, 1), (2, 4), (5, 10), (11, 20), (21, 43)]))


if __name__ == "__main__":
    main()
