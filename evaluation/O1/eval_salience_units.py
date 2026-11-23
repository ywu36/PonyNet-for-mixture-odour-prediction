# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   U = units (groups or classes)   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""Common features above the molecule: chemical groups and classes (follow-up to eval_salience.py), declared before running.

Molecule level (base): + F_mol + k on B1a++, share >= 1 +0.039, salient shared molecule pulls closer. Here the same
construction one level up. Unit u = one of the 24 RDKit groups (as B1a++) or a class (pushpull.CLASSES carried by >= 5 of
the 235 molecules). In mixture X, carriers C_X(u); u is common if both mixtures have a carrier (not necessarily the same
molecule). Salience of u in X: |mean q of C_X(u) - mean q of X minus C_X(u)| (0 if every molecule carries u).
F_unit = sum over common units of min(sal_A, sal_B); n_unit = number of common units.
Rows (all on top of B1a++, joint head, change from B1a++ on the same folds):
  + F_group + n_group                 (all molecules)
  + F_group + n_group, unshared only  (carriers among each mixture's unshared molecules: the same group from different
                                       molecules; on pairs sharing none this is all molecules)
  + F_class + n_class                 (all molecules)
  + F_mol + k + F_group + n_group, unshared only (does the group level add beyond the molecule level?)
Key subset: share no molecule (only unit-level terms can act there; V50/T46 are of this kind). Also all pairs,
share >= 1. Reported: weights over 250 CV fits; paired change vs base (+ F_mol + k). CPU.

Env:     mist
Usage:   python evaluation/O1/eval_salience_units.py --variant group | group_un | class | mol_group ; then --report
Writes:  results/eval_o12c/cv_unit_<variant>.pkl ; report results/eval_salience_units/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import evaluate, features  # noqa: E402
from mixdist.b1a_plus import B1aPlus  # noqa: E402
from mixdist.pushpull import CLASSES  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402
from eval_salience import features_all  # noqa: E402

CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "sal_min": "base: + F_mol + k (molecule level)",
         "unit_group": "+ F_group + n_group", "unit_group_un": "+ F_group + n_group, unshared molecules only",
         "unit_class": "+ F_class + n_class", "unit_mol_group": "+ F_mol + k + F_group + n_group (unshared only)"}
TERMS = {"sal_min": ["F_mol", "k"], "unit_group": ["F_group", "n_group"], "unit_group_un": ["F_group_un", "n_group_un"],
         "unit_class": ["F_class", "n_class"], "unit_mol_group": ["F_mol", "k", "F_group_un", "n_group_un"]}


def unit_terms(train, q, cids, U_MxU, unshared_only):
    row = {int(c): i for i, c in enumerate(cids)}
    F, n = np.zeros(len(train)), np.zeros(len(train))
    for p_, pr in enumerate(train):
        sh = pr.a & pr.b
        sides = [[row[c] for c in ((m - sh) if unshared_only else m)] for m in (pr.a, pr.b)]
        if not sides[0] or not sides[1]:
            continue
        for u in range(U_MxU.shape[1]):
            sal = []
            for ii in sides:
                car = [i for i in ii if U_MxU[i, u]]; rest = [i for i in ii if not U_MxU[i, u]]
                if not car:
                    break
                sal.append(np.linalg.norm(q[car].mean(0) - q[rest].mean(0)) if rest else 0.0)
            if len(sal) == 2:
                F[p_] += min(sal); n[p_] += 1
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    return z(F), z(n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup(); op = features.load("openpom")
    mols = [Chem.MolFromSmiles(str(s)) for s in op.smiles_M]
    K = np.array([[m.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in CLASSES.values()] for m in mols], float)
    keep = K.sum(0) >= 5; K = K[:, keep]; cnames = [c for c, k in zip(CLASSES, keep) if k]
    if a.variant:
        if a.variant == "group":
            T = np.c_[unit_terms(train, q, op.cids_M, X > 0, False)]
        elif a.variant == "group_un":
            T = np.c_[unit_terms(train, q, op.cids_M, X > 0, True)]
        elif a.variant == "class":
            T = np.c_[unit_terms(train, q, op.cids_M, K > 0, False)]
        else:
            T = np.c_[features_all(train, q, op.cids_M)["sal_min"], np.c_[unit_terms(train, q, op.cids_M, X > 0, True)]]
        T = T.T if T.shape[0] != len(train) else T
        Sx = np.c_[S, T]; store = []

        def fp(tr, te):
            f = B1aPlus(q, X, A, B, Sx, LAM).fit(tr, y[tr]); store.append(f.lin.numpy()[4:].copy())
            return f.predict(te)
        res = evaluate.cross_validate(train, fp)
        pickle.dump({"res": res, "g": store}, open(CV / f"cv_unit_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_salience_units", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "classes": cnames})
    rep.line(f"classes kept (>= 5 of 235 molecules): {', '.join(cnames)}; groups: {X.shape[1]}")
    run = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nN = np.array([len(p.a & p.b) for p in train])[ref.pair_N]
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1)]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a2"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in list(NAMES)[1:]:
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask)
            rows.append([NAMES[k_], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE [CI] / r [CI]"], rows)
    rep.section("2  Weights per sd, 250 CV fits (median [2.5, 97.5%]); < 0 = closer")
    grow = []
    for k_ in list(NAMES)[1:]:
        G = np.array(run[k_]["g"])
        for j, lb in enumerate(TERMS[k_]):
            grow.append([NAMES[k_], lb, f"{np.median(G[:, j]):+.4f} [{np.percentile(G[:, j], 2.5):+.4f}, {np.percentile(G[:, j], 97.5):+.4f}]"])
    rep.table(["model", "term", "weight"], grow)
    rep.save()


if __name__ == "__main__":
    main()
