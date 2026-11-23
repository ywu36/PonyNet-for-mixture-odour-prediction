# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""Shared-molecule salience by chemical group (structure level), follow-up to eval_o12c.

Question: does a shared molecule stand out more (or less) in both mixtures depending on its chemical groups?
Model (mixdist/o12.py O12c mode "groups"): shared molecule's weight = G(m) exp(alpha + eta . x(m)), x = the same 24
RDKit groups as the gain theta; eta penalised by lam 1e-3 (as theta); alpha unpenalised. eta = 0 gives the constant
alpha model of eval_o12c (row "const"), which was alpha +0.17 [+0.01, +0.35] with no CV gain.
Rows, declared before running, change from B1a++ on the same folds (other rows reused from results/eval_o12c):
  B1a++ (baseline) | O1.2b | O1.2c const (alpha) | O1.2c groups (alpha + eta)
Subsets: all pairs, share no molecule, share >= 1; share 1-2 audit. Prior result on the same question at molecule level:
shared-molecule identity r +0.03 (b1a+ session). Reported per group: support (pairs whose shared molecules carry it,
distinct shared molecules), eta full data, eta over 250 CV fits (median [2.5, 97.5%]).

Env:     mist
Usage:   python evaluation/O1/eval_o12d.py --variant groups ; then --report
Writes:  results/eval_o12c/cv_groups.pkl ; report results/eval_o12d/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus")); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import data, evaluate, features, pushpull as pp  # noqa: E402
from mixdist.o12 import O12b, O12c  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402

CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "o12b": "O1.2b gain + length", "const": "O1.2c const (alpha)",
         "groups": "O1.2c groups (alpha + eta per group)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup()
    if a.variant == "groups":
        store = []

        def fp(tr, te):
            f = O12c(q, X, A, B, mode="groups", lam=LAM).fit(tr, y[tr]); store.append(f.s.numpy().copy())
            return f.predict(te)
        res = evaluate.cross_validate(train, fp)
        pickle.dump({"res": res, "params": store}, open(CV / "cv_groups.pkl", "wb")); print("saved groups"); return

    rep = Report("eval_o12d", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    allp = np.arange(len(train)); th = torch.zeros(X.shape[1]); k = torch.tensor(0.7)
    rep.section("0  Audit")
    rep.line(f"groups mode at alpha = eta = 0 vs O1.2b distance: max |difference| "
             f"{float((O12b(q, X, A, B).distance(th, k, allp) - O12c(q, X, A, B, mode='groups').distance(th, k, allp)).abs().max()):.2e}")
    run = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
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

    rep.section("2  Shared salience per group (support, full-data eta, eta over CV fits)")
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    _, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    Sh_PxM = A * B
    f = O12c(q, X, A, B, mode="groups", lam=LAM).fit(allp, y); eta_G = f.s.numpy()[1:]
    P = np.array(run["groups"]["params"])
    rep.line(f"alpha: full data {float(f.s[0]):+.3f}; CV fits {np.median(P[:, 0]):+.2f} [{np.percentile(P[:, 0], 2.5):+.2f}, {np.percentile(P[:, 0], 97.5):+.2f}]")
    prow = []
    for g in np.argsort(-eta_G):
        pairs_g = int(((Sh_PxM @ X[:, g]) > 0).sum()); mols_g = int(((Sh_PxM.sum(0) > 0) & (X[:, g] > 0)).sum())
        lo, md, hi = np.percentile(P[:, 1 + g], [2.5, 50, 97.5])
        prow.append([gn[g], pairs_g, mols_g, f"{eta_G[g]:+.3f}", f"{md:+.3f} [{lo:+.3f}, {hi:+.3f}]", "yes" if lo > 0 or hi < 0 else ""])
    rep.table(["group", "pairs with a shared carrier (of 495)", "distinct shared molecules", "eta full data",
               "eta CV fits median [2.5, 97.5%]", "CV range excludes 0"], prow)
    rep.line(f"distinct shared molecules overall: {int((Sh_PxM.sum(0) > 0).sum())}")
    rep.save()


if __name__ == "__main__":
    main()
