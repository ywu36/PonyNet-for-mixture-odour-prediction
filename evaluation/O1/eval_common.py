# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   W = open-POM words (138)   G = RDKit groups (24)
#   P = pairs (820)   X = mixtures (800)   N = pooled out-of-fold predictions
# ---------------------------------------------------------------------------------------------
"""Common features in the distance (Tversky 1977 contrast model), on top of B1a++, declared before running.

Why: every model of the B1 family compares mixtures only by their difference, so a shared molecule cancels (exactly
for equal sizes); b1a+ found a noticeable shared molecule pulls a pair closer (+0.17 [+0.01, +0.92]) while unshared
standouts add nothing. Tversky: similarity rises with salient common features. Head:
    prediction = a d + c + B1a++ size terms + g F,   F >= 0 a common-feature score (inputs only, theta-free profiles q)
  F_mol  = sum over shared molecules s of min(c_A(s), c_B(s)),  c_X(s) = |q_s - mean q of X's unshared molecules|
           (how much s stands out from its company on that side; 0 if that side has no unshared molecule)
  F_note = sum over words w of min(s_w(A), s_w(B)),  s_w(X) = max(0, z_w(X)), z = mixture mean profile standardized per
           word over the 800 training mixtures (a note noticeable in both mixtures, from any molecules)
Rows (change from B1a++ on the same folds; B1a++ reused from results/eval_o12c):
  B1a++ (baseline) | + F_mol | + F_note | + F_mol + F_note
Expected sign of g: negative (common features -> closer). F columns z-scored over pairs for the optimiser (scale only).
Subsets: all pairs, share no molecule (F_mol = 0 there; only F_note can act), share >= 1; share 1-2 audit; partial r.
Reported: g over the 250 CV fits (median [2.5, 97.5%]); r(OOF error, b1a+ contrast) on share >= 1 (was -0.113).
CPU (B1aPlus is a small L-BFGS model; 3 runs in parallel, single-threaded).

Env:     mist
Usage:   python evaluation/O1/eval_common.py --variant mol | note | both | k | mol_k | meansal_k ; then --report
Writes:  results/eval_o12c/cv_common_<variant>.pkl ; report results/eval_common/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import evaluate, features  # noqa: E402
from mixdist.b1a_plus import B1aPlus  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402
from eval_o15 import pair_contrast  # noqa: E402
from audit_content_no_size import wls_resid  # noqa: E402

CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "common_mol": "+ common shared molecule (F_mol)",
         "common_note": "+ common noticeable notes (F_note)", "common_both": "+ F_mol + F_note",
         # controls added 2026-11-18 after the first run (F_mol grows with the shared count k), declared before running:
         "common_k": "control: + k (number of shared molecules)", "common_mol_k": "+ F_mol + k",
         "common_meansal_k": "+ mean salience of shared molecules + k"}


def common_features(train, q, cids):
    row = {int(c): i for i, c in enumerate(cids)}
    F_mol = np.zeros(len(train))
    for p_, pr in enumerate(train):
        sh = pr.a & pr.b
        if not sh:
            continue
        side = []
        for mine in (pr.a, pr.b):
            un = [row[c] for c in mine - sh]
            side.append({s: (np.linalg.norm(q[row[s]] - q[un].mean(0)) if un else 0.0) for s in sh})
        F_mol[p_] = sum(min(side[0][s], side[1][s]) for s in sh)
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); ix = {m: i for i, m in enumerate(mixes)}
    P_XxW = np.array([q[[row[c] for c in m]].mean(0) for m in mixes])
    sal_XxW = np.maximum(0, (P_XxW - P_XxW.mean(0)) / P_XxW.std(0))
    F_note = np.array([np.minimum(sal_XxW[ix[p.a]], sal_XxW[ix[p.b]]).sum() for p in train])
    z = lambda v: (v - v.mean()) / v.std()
    k = np.array([len(p.a & p.b) for p in train], float)
    mean_sal = np.where(k > 0, F_mol / np.maximum(k, 1), 0.0)
    return {"mol": z(F_mol)[:, None], "note": z(F_note)[:, None], "both": np.c_[z(F_mol), z(F_note)],
            "k": z(k)[:, None], "mol_k": np.c_[z(F_mol), z(k)], "meansal_k": np.c_[z(mean_sal), z(k)]}, F_mol, F_note


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup(); op = features.load("openpom")
    F, F_mol, F_note = common_features(train, q, op.cids_M)
    if a.variant:
        Sx = np.c_[S, F[a.variant]]; store = []

        def fp(tr, te):
            f = B1aPlus(q, X, A, B, Sx, LAM).fit(tr, y[tr]); store.append(f.lin.numpy()[4:].copy())
            return f.predict(te)
        res = evaluate.cross_validate(train, fp)
        pickle.dump({"res": res, "g": store}, open(CV / f"cv_common_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_common", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    nsh = np.array([len(p.a & p.b) for p in train])
    rep.section("0  Features")
    rep.line(f"F_mol > 0 on {int((F_mol > 0).sum())} pairs (share >= 1: {int((nsh >= 1).sum())}); F_note > 0 on {int((F_note > 0).sum())} pairs")
    rep.line(f"r(F_note, B1a distance-free label y): share none {np.corrcoef(F_note[nsh == 0], y[nsh == 0])[0, 1]:+.3f}; "
             f"r(F_mol, y) share >= 1 {np.corrcoef(F_mol[nsh >= 1], y[nsh >= 1])[0, 1]:+.3f}; r(F_mol, F_note) share >= 1 {np.corrcoef(F_mol[nsh >= 1], F_note[nsh >= 1])[0, 1]:+.3f}")
    run = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nN = nsh[ref.pair_N]
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1), ("share 1-2 (audit)", (nN >= 1) & (nN <= 2))]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a2"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in list(NAMES)[1:]:
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask)
            rows.append([NAMES[k_], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE [CI] / r [CI]"], rows)
    w_N = evaluate.cv_weights(ref); prow = []
    for subset, m in [("share no molecule", nN == 0), ("share 1 (audit)", nN == 1)]:
        Z = np.c_[np.ones(m.sum()), S[ref.pair_N[m]]]; w = w_N[m]
        for k_ in NAMES:
            r = res[k_]; f = replace(r, pred_N=r.pred_N.copy(), truth_N=r.truth_N.copy())
            f.pred_N[m] = wls_resid(Z, r.pred_N[m], w); f.truth_N[m] = wls_resid(Z, r.truth_N[m], w)
            s = evaluate.score_cv(train, f, mask_N=m); prow.append([NAMES[k_], subset, f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]"])
    rep.table(["model", "subset", "partial r (size removed) [95% CI]"], prow)
    rep.section("2  Fitted common-feature weight g (per sd of F; < 0 = common features make mixtures closer), 250 CV fits")
    grow = []
    for k_ in list(NAMES)[1:]:
        G = np.array(run[k_]["g"]); labs = {"common_mol": ["F_mol"], "common_note": ["F_note"], "common_both": ["F_mol", "F_note"], "common_k": ["k"],
                "common_mol_k": ["F_mol", "k"], "common_meansal_k": ["mean salience", "k"]}[k_]
        for j, lb in enumerate(labs):
            grow.append([NAMES[k_], lb, f"{np.median(G[:, j]):+.4f} [{np.percentile(G[:, j], 2.5):+.4f}, {np.percentile(G[:, j], 97.5):+.4f}]"])
    rep.table(["model", "term", "g median [2.5, 97.5%]"], grow)
    con_P, _, _ = pair_contrast(train, q, op.cids_M); sh = nsh >= 1; crow = []
    for k_, r in res.items():
        num = np.bincount(r.pair_N, weights=r.pred_N, minlength=len(train)); den = np.bincount(r.pair_N, minlength=len(train))
        crow.append([NAMES[k_], f"{np.corrcoef((y - num / np.maximum(den, 1))[sh], con_P[sh])[0, 1]:+.3f}"])
    rep.table(["model", "r(OOF error, b1a+ contrast), share >= 1"], crow)
    rep.save()


if __name__ == "__main__":
    main()
