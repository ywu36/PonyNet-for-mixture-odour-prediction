# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""O1.2b standard CV (design/o1_2_design.md, last section; mixdist/o12.py class O12b).

Distance keeps B1a's direction u = p/|p| and adds the profile length h = log|p| (shrinks as mixtures grow, "white"):
    d = sqrt(|u(A) - u(B)|^2 + beta (h(A) - h(B))^2),  p = gain-weighted mean of quantile profiles, gain exp(theta . x)
Rows, all declared before running and all reported (change from B1a on the same folds):
  1 B1a (baseline)   2 O1.2b length only (theta = 0)   3 O1.2b theta + length
  4 O1.2 theta only (beta = 0; same as the previous run)   5 reference: B1a+ groups + size (covariates)
theta: 24 RDKit groups, penalty lam 1e-3; beta = b^2, b starts at 0.1. Subsets: all pairs, share no molecule;
share 1 as audit (chosen after seeing results).
Checks: beta from a second start (b = 2); r(residual, |log nA - log nB|) for full-data fits; r(h, log n) over
mixtures; Ma outside check (theta . (x(m1) - x(m2)) vs Ma's L, no refit). V50/T46 not scored.

Env:     mist
Usage:   python evaluation/O1/eval_o12b.py
Writes:  results/eval_o12b/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus")); sys.path.insert(0, str(ROOT / "findings"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.o12 import O12, O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from ma_check_dream_groups import groups_with_ma  # noqa: E402

LAM = 1e-3


def main():
    rep = Report("eval_o12b", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(train, op.cids_M); S = size_features(train)
    X_MxG, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    allp = np.arange(len(train))
    b1a_d = mixture.paper_distance(train, op.cids_M, op.probs_MxW)
    chk = O12b(q, X_MxG, A, B, False, False).distance(torch.zeros(len(gn)), torch.tensor(0.0), allp).numpy()
    rep.section("0  Audit")
    rep.line(f"O1.2b at theta = 0, beta = 0 vs B1a distance: max |difference| {np.abs(chk - b1a_d).max():.2e}")

    variants = {"O1.2b length only": lambda tr, te: O12b(q, X_MxG, A, B, False, True, lam=LAM).fit(tr, y[tr]).predict(te),
                "O1.2b theta + length": lambda tr, te: O12b(q, X_MxG, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
                "O1.2 theta only": lambda tr, te: O12(q, X_MxG, A, B, True, False, lam=LAM).fit(tr, y[tr]).predict(te),
                "reference: B1a+ groups + size": lambda tr, te: B1aPlus(q, X_MxG, A, B, S, LAM).fit(tr, y[tr]).predict(te)}
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(b1a_d, y, np.zeros(len(train))))
    res = {}
    for k_, f in variants.items():
        res[k_] = evaluate.cross_validate(train, f); print(f"  done {k_}", flush=True)
    share1 = np.array([len(p.a & p.b) == 1 for p in train])
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a on the same pairs")
    rows = []
    for subset in ("all pairs", "share no molecule", "share 1 (audit)"):
        def mask(r_):
            return None if subset == "all pairs" else (r_.no_overlap_N if subset == "share no molecule" else share1[r_.pair_N])
        s0 = evaluate.score_cv(train, ref, mask_N=mask(ref))
        rows.append(["B1a (baseline)", subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in variants:
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask(res[k_]))
            rows.append([k_, subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a: RMSE / r [95% CI]"], rows)

    rep.section("2  Full-data fits")
    fits = {"length only": O12b(q, X_MxG, A, B, False, True, lam=LAM).fit(allp, y),
            "theta + length": O12b(q, X_MxG, A, B, True, True, lam=LAM).fit(allp, y),
            "theta only": O12(q, X_MxG, A, B, True, False, lam=LAM).fit(allp, y)}
    b2 = {t: O12b(q, X_MxG, A, B, th, True, lam=LAM, k0=2.0).fit(allp, y).beta for t, th in [("length only", False), ("theta + length", True)]}
    rep.line("beta (start b = 0.1 / start b = 2): " + "; ".join(f"{t} {fits[t].beta:.4f} / {b2[t]:.4f}" for t in b2))
    mis = np.abs(np.log([len(p.a) for p in train]) - np.log([len(p.b) for p in train]))
    b1a_pred = evaluate.linear_calibration(b1a_d, y, np.zeros(len(train)))(allp, allp)
    rep.table(["full-data model", "r(residual, |log nA - log nB|)", "in-sample r"],
              [["B1a", f"{np.corrcoef(y - b1a_pred, mis)[0, 1]:+.3f}", f"{np.corrcoef(b1a_pred, y)[0, 1]:.3f}"]] +
              [[t, f"{np.corrcoef(y - f.predict(allp), mis)[0, 1]:+.3f}", f"{np.corrcoef(f.predict(allp), y)[0, 1]:.3f}"] for t, f in fits.items()])
    row = {int(c): i for i, c in enumerate(op.cids_M)}
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted)
    h = np.array([np.log(np.linalg.norm(q[[row[int(c)] for c in m]].mean(0))) for m in mixes]); n_ = np.log([len(m) for m in mixes])
    rep.line(f"over {len(mixes)} mixtures: r(h = log|p|, log n) {np.corrcoef(h, n_)[0, 1]:+.3f} (theta = 0 profiles)")

    mol = pd.read_csv(ROOT / "external/ma_2021/pyrfume_molecules.csv").sort_values("CID").reset_index(drop=True)
    _, Xm, _ = groups_with_ma({i: str(s) for i, s in zip(ids, op.smiles_M)}, {f"ma{c}": s for c, s in zip(mol.CID, mol.IsomericSMILES)})
    r = pd.read_csv(ROOT / "external/ma_2021/pyrfume_behavior.csv"); r = r[(r.Subject != 47) & (r.IA > 0) & (r.IB > 0)]
    g = r.groupby(["Stimulus A", "Stimulus B"])[["IA", "IAmix", "IB", "IBmix"]].mean().reset_index()
    L_P = np.log((g.IAmix / g.IA) / (g.IBmix / g.IB)).values
    ix = {c: i for i, c in enumerate(mol.CID)}; a_P, b_P = g["Stimulus A"].map(ix).values, g["Stimulus B"].map(ix).values
    rep.table(["full-data model", "Ma: r(theta . (x(m1) - x(m2)), L), 198 mixtures"],
              [[t, f"{np.corrcoef((Xm @ fits[t].theta.numpy())[a_P] - (Xm @ fits[t].theta.numpy())[b_P], L_P)[0, 1]:+.3f}"]
               for t in ("theta + length", "theta only")])
    rep.value("beta", {t: fits[t].beta for t in ("length only", "theta + length")})
    rep.save()


if __name__ == "__main__":
    main()
