# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = pairs (820)
#   N = pooled out-of-fold predictions
# ---------------------------------------------------------------------------------------------
"""O1.2c: a shared molecule's salience depends on its company (mixture size), as parameters inside the pooling.

Idea (b1a+ session, psychophysics: people pick out ~3-4 components; small mixtures are smelled analytically, large ones
as a whole): in a small mixture a shared molecule is noticed in both (pair closer than its share says), in a large
one it blurs (counts less). B1a++ cannot express this: a molecule's weight is fixed and size only shifts the output.
Model (mixdist/o12.py O12c): shared molecule's weight in mixture A = G(m) exp(alpha + gamma log(n_A / 10)).

Rows, declared before running, all reported, change from B1a++ (current best) on the same folds:
  1 B1a++ (baseline: groups + size covariates)   2 O1.2b (gain + length)
  3 O1.2c const (alpha only: one constant extra weight for shared molecules, as the earlier null anchor test)
  4 O1.2c size (alpha + gamma: weight depends on mixture size, can flip sign)
  6 (added before any result, user request) O1.2c tune-out: d = (1 - lmb) d(A, B) + lmb d(a, b), a, b = the parts
    that differ; lmb fitted (people compare only the differing parts). Row 3 is the weight form of the same idea
    (alpha -> -inf = fully tuned out).
  5 control for the other reading ("size term miscalibrated for overlapping pairs"): O1.2b + 2 head covariates
    o, o * (log nA + log nB - 2 log 10), o = 1 if the pair shares >= 1 molecule. Only shifts the output.
Reading: row 4 > row 3 and > row 5 supports "salience depends on company"; row 4 ~ row 5 cannot separate the two.
Subsets: all pairs; share no molecule (rows 3-4 can only act through refit theta there); share >= 1 (where the
mechanism acts). Audit (chosen after seeing b1a+'s grid): share 1-2. theta lam 1e-3; alpha, gamma unpenalised, start 0.
Reported: alpha, gamma per CV fit (median, 2.5-97.5%), full-data fit, size n* where the sign flips (gamma != 0).
V50/T46 not scored.

Env:     mist
Usage:   python evaluation/O1/eval_o12c.py --variant NAME  (b1a2, o12b, const, size, control, tuneout; theta for the audit) ; then --report
Writes:  results/eval_o12c/cv_<NAME>.pkl ; report results/eval_o12c/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.o12 import O12, O12b, O12c  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3
OUT = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "o12b": "O1.2b gain + length", "const": "O1.2c const (alpha)",
         "size": "O1.2c size (alpha + gamma)", "control": "control: O1.2b + overlap x size covariates",
         "tuneout": "O1.2c tune-out (blend d(A,B), d(a,b))"}


def setup():
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(train, op.cids_M); S = size_features(train)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    o_P = np.array([len(p.a & p.b) >= 1 for p in train], float)
    C = np.c_[o_P, o_P * (S[:, 0] - 2 * np.log(10))]
    return train, y, q, A, B, S, X, C


def make(name, y, q, A, B, S, X, C, store):
    if name == "b1a2":
        return lambda tr, te: B1aPlus(q, X, A, B, S, LAM).fit(tr, y[tr]).predict(te)
    if name == "o12b":
        return lambda tr, te: O12b(q, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)
    if name == "theta":                              # content-only model for audit_content_no_size.py
        return lambda tr, te: O12(q, X, A, B, True, False, lam=LAM).fit(tr, y[tr]).predict(te)
    if name == "control":
        return lambda tr, te: O12b(q, X, A, B, True, True, size_PxS=C, lam=LAM).fit(tr, y[tr]).predict(te)

    def fp(tr, te):
        f = O12c(q, X, A, B, mode=name, lam=LAM).fit(tr, y[tr]); store.append(f.s.numpy().copy())
        return f.predict(te)
    return fp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.variant:
        store = []
        res = evaluate.cross_validate(train, make(a.variant, y, q, A, B, S, X, C, store))
        pickle.dump({"res": res, "params": store}, open(OUT / f"cv_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_o12c", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    allp = np.arange(len(train))
    rep.section("0  Audit")
    th = torch.zeros(X.shape[1]); k = torch.tensor(0.7)
    d_b = O12b(q, X, A, B, True, True).distance(th, k, allp); d_c = O12c(q, X, A, B).distance(th, k, allp)
    rep.line(f"O1.2c at alpha = gamma = 0 vs O1.2b distance: max |difference| {float((d_b - d_c).abs().max()):.2e}")
    nsh = np.array([len(p.a & p.b) for p in train])
    rep.line(f"pairs: share 0 {int((nsh == 0).sum())}, share >= 1 {int((nsh >= 1).sum())}, share 1-2 {int(((nsh >= 1) & (nsh <= 2)).sum())}")

    run = {k_: pickle.load(open(OUT / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nN = nsh[ref.pair_N]
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1), ("share 1-2 (audit)", (nN >= 1) & (nN <= 2))]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([NAMES["b1a2"], subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k_ in NAMES:
            if k_ == "b1a2":
                continue
            s = evaluate.score_cv(train, res[k_], reference=ref, mask_N=mask)
            rows.append([NAMES[k_], subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]"], rows)

    rep.section("2  Shared-molecule weight parameters")
    prow = []
    for k_ in ("const", "size", "tuneout"):
        P = np.array(run[k_]["params"]); f = O12c(q, X, A, B, mode=k_, lam=LAM).fit(allp, y)
        if k_ == "tuneout":
            lm = 1 / (1 + np.exp(-P[:, 0]))
            prow.append([NAMES[k_], f"lambda {np.median(lm):.2f} [{np.percentile(lm, 2.5):.2f}, {np.percentile(lm, 97.5):.2f}]",
                         f"lambda {float(torch.sigmoid(f.s[0])):.3f}", "-", "-"]); continue
        al, ga = float(f.s[0]), float(f.s[1]) if k_ == "size" else 0.0
        flip = f"{10 * np.exp(-al / ga):.1f}" if k_ == "size" and abs(ga) > 1e-6 else "-"
        cvs = "; ".join(f"{nm} {np.median(P[:, j]):+.2f} [{np.percentile(P[:, j], 2.5):+.2f}, {np.percentile(P[:, j], 97.5):+.2f}]"
                        for j, nm in enumerate(["alpha", "gamma"][:1 + (k_ == "size")]))
        ws = "; ".join(f"n {n}: x{np.exp(al + ga * np.log(n / 10)):.2f}" for n in (5, 10, 20, 40))
        prow.append([NAMES[k_], cvs, f"alpha {al:+.3f} gamma {ga:+.3f}", flip, ws])
    rep.table(["model", "CV fits (250): median [2.5, 97.5%]", "full data", "n* sign flip", "full-data shared weight by size"], prow)
    rep.save()


if __name__ == "__main__":
    main()
