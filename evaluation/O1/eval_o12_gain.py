# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = pairs (820)
#   N = pooled out-of-fold predictions
# ---------------------------------------------------------------------------------------------
"""O1.2 (revised, design/o1_2_design.md; mixdist/o12.py) standard CV.

Rows, all declared before running and all reported (every row = change from B1a on the same folds):
  1 B1a (baseline)                 2 O1.2 kappa only (theta = 0)        3 O1.2 theta only (kappa = 0)
  4 O1.2 theta + kappa             5 reference: B1a+ groups + size (size covariates, mixdist/b1a_plus.py)
theta: 24 RDKit groups (as finding 4), penalty lam 1e-3; kappa = k^2, k starts at 0.1 (kappa 0.01 ~ B1a).
Subsets: all pairs; pairs whose mixtures share no molecule (standard). Audit only (subset chosen after seeing
results): pairs sharing exactly 1 molecule.
Checks: (a) full-data fit: kappa, and with it the "whiteness" kappa / (kappa + |p(A)|) by mixture size; kappa from a
second start (k = 2) to confirm the optimum; (b) residual of rows 2 and 4 vs |log nA - log nB| (does kappa absorb
the size-mismatch effect?); (c) Ma outside check, no refit: r of theta . (x(m1) - x(m2)) with Ma's L.
V50/T46 not scored.

Env:     mist
Usage:   python evaluation/O1/eval_o12_gain.py
Writes:  results/eval_o12_gain/<date_time>.{txt,json}
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
from mixdist.o12 import O12  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from ma_check_dream_groups import groups_with_ma  # noqa: E402

LAM = 1e-3


def main():
    rep = Report("eval_o12_gain", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(train, op.cids_M); S = size_features(train)
    X_MxG, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    allp = np.arange(len(train))

    chk = O12(q, X_MxG, A, B, use_theta=False, use_kappa=False).distance(torch.zeros(len(gn)), torch.tensor(0.0), allp).numpy()
    rep.section("0  Audit")
    rep.line(f"O1.2 at theta = 0, kappa = 0 vs B1a distance: max |difference| "
             f"{np.abs(chk - mixture.paper_distance(train, op.cids_M, op.probs_MxW)).max():.2e}")

    def fp(use_theta, use_kappa):
        return lambda tr, te: O12(q, X_MxG, A, B, use_theta, use_kappa, lam=LAM).fit(tr, y[tr]).predict(te)
    variants = {"O1.2 kappa only": fp(False, True), "O1.2 theta only": fp(True, False), "O1.2 theta + kappa": fp(True, True),
                "reference: B1a+ groups + size": lambda tr, te: B1aPlus(q, X_MxG, A, B, S, LAM).fit(tr, y[tr]).predict(te)}
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(mixture.paper_distance(train, op.cids_M, op.probs_MxW), y, np.zeros(len(train))))
    res = {}
    for k_, f in variants.items():
        res[k_] = evaluate.cross_validate(train, f); print(f"  done {k_}", flush=True)
    share1 = np.array([len(p.a & p.b) == 1 for p in train])
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", ref.no_overlap_N), ("share 1 (audit)", share1[ref.pair_N])]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append(["B1a (baseline)", subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in variants:
            m = mask if mask is None else (ref.no_overlap_N if subset == "share no molecule" else share1[res[k_].pair_N])
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=m)
            rows.append([k_, subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a: RMSE / r [95% CI]"], rows)

    rep.section("2  Full-data fits: kappa, whiteness, size mismatch, Ma")
    fits = {"kappa only": O12(q, X_MxG, A, B, False, True, lam=LAM).fit(allp, y),
            "theta + kappa": O12(q, X_MxG, A, B, True, True, lam=LAM).fit(allp, y),
            "theta only": O12(q, X_MxG, A, B, True, False, lam=LAM).fit(allp, y)}
    k2 = {t: O12(q, X_MxG, A, B, th, True, lam=LAM, k0=2.0).fit(allp, y).kappa for t, th in [("kappa only", False), ("theta + kappa", True)]}
    rep.line("kappa (start k = 0.1 / start k = 2): " + "; ".join(f"{t} {fits[t].kappa:.3f} / {k2[t]:.3f}" for t in k2))
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted)
    row = {int(c): i for i, c in enumerate(op.cids_M)}
    n_X = np.array([len(m) for m in mixes])
    for t in ("kappa only", "theta + kappa"):
        f = fits[t]; w = np.exp((f.x @ f.theta).numpy())
        nrm = np.array([np.linalg.norm((w[[row[int(c)] for c in m]][:, None] * q[[row[int(c)] for c in m]]).sum(0) / w[[row[int(c)] for c in m]].sum()) for m in mixes])
        wh = f.kappa / (f.kappa + nrm)
        rep.line(f"{t}: whiteness kappa/(kappa+|p|) by size: " + ", ".join(
            f"n {lo}-{hi} {np.median(wh[(n_X >= lo) & (n_X <= hi)]):.2f}" for lo, hi in [(1, 1), (2, 4), (5, 10), (11, 20), (21, 43)]))
    mis = np.abs(S[:, 0] * 0 + np.log([len(p.a) for p in train]) - np.log([len(p.b) for p in train]))
    rr = []
    for t, f in [("B1a", None)] + list(fits.items()):
        pred = evaluate.linear_calibration(mixture.paper_distance(train, op.cids_M, op.probs_MxW), y, np.zeros(len(train)))(allp, allp) if f is None else f.predict(allp)
        rr.append([t, f"{np.corrcoef(y - pred, mis)[0, 1]:+.3f}"])
    rep.table(["full-data model", "r(residual, |log nA - log nB|)"], rr)

    mol = pd.read_csv(ROOT / "external/ma_2021/pyrfume_molecules.csv").sort_values("CID").reset_index(drop=True)
    _, Xm, _ = groups_with_ma({i: str(s) for i, s in zip(ids, op.smiles_M)}, {f"ma{c}": s for c, s in zip(mol.CID, mol.IsomericSMILES)})
    r = pd.read_csv(ROOT / "external/ma_2021/pyrfume_behavior.csv"); r = r[(r.Subject != 47) & (r.IA > 0) & (r.IB > 0)]
    g = r.groupby(["Stimulus A", "Stimulus B"])[["IA", "IAmix", "IB", "IBmix"]].mean().reset_index()
    L_P = np.log((g.IAmix / g.IA) / (g.IBmix / g.IB)).values
    ix = {c: i for i, c in enumerate(mol.CID)}; a_P, b_P = g["Stimulus A"].map(ix).values, g["Stimulus B"].map(ix).values
    rep.table(["full-data model", "Ma: r(theta . (x(m1) - x(m2)), L), 198 mixtures"],
              [[t, f"{np.corrcoef((Xm @ fits[t].theta.numpy())[a_P] - (Xm @ fits[t].theta.numpy())[b_P], L_P)[0, 1]:+.3f}"]
               for t in ("theta only", "theta + kappa")])
    rep.value("kappa", {t: fits[t].kappa for t in ("kappa only", "theta + kappa")})
    rep.save()


if __name__ == "__main__":
    main()
