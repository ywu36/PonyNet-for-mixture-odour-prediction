# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   W = open-POM words (138)
#   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""Lead 2: let structure change the odor profile, not only the weight (mixdist/o12.py O12e).

    q'(m) = q(m) + (x(m) V) U, rank r; gain, pooling, direction + length distance as O1.2b.
Why: structure (24 RDKit groups) is the one lever that helps content in every subset (partial r about +0.09 with size
taken out, audit_content_no_size), but so far it only reweights molecules; the odor-word layer (open-POM) is the
weak part (content-only r 0.10).
Rows, declared before running, change from B1a++ on the same folds (B1a++ and O1.2b reused from results/eval_o12c):
  B1a++ (baseline) | O1.2b | O1.2e rank 2 | O1.2e rank 5        (theta, U, V lam 1e-3)
Subsets: all pairs, share no molecule; share >= 1; share 1-2 audit. Also: content once size is taken out (partial r,
as audit_content_no_size) on share no molecule and share 1 (audit). Full data: size of the shift |x V U| vs |q|,
and the words most moved (rank 2).

Env:     mist
Usage:   python evaluation/O1/eval_o12e.py --variant r2 | r5 ; then --report
Writes:  results/eval_o12c/cv_<r2|r5>.pkl ; report results/eval_o12e/<date_time>.{txt,json}
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
from mixdist.o12 import O12e  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import LAM, setup  # noqa: E402
from audit_content_no_size import wls_resid  # noqa: E402

CV = ROOT / "results/eval_o12c"
NAMES = {"b1a2": "B1a++ (baseline)", "o12b": "O1.2b gain + length", "r2": "O1.2e rank 2", "r5": "O1.2e rank 5",
         "r2l4": "O1.2e rank 2, lam 1e-4", "r2l5": "O1.2e rank 2, lam 1e-5"}
RANK = {"r2": 2, "r5": 5, "r2l4": 2, "r2l5": 2}
LAMS = {"r2l4": 1e-4, "r2l5": 1e-5}   # added after the first run: lam 1e-3 forced the shift to 0 (U = V = 0, test vacuous)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    train, y, q, A, B, S, X, C = setup()
    if a.variant:
        res = evaluate.cross_validate(train, lambda tr, te: O12e(q, X, A, B, RANK[a.variant], LAMS.get(a.variant, LAM)).fit(tr, y[tr]).predict(te))
        pickle.dump({"res": res}, open(CV / f"cv_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_o12e", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    res = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb"))["res"] for k_ in NAMES}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nN = np.array([len(p.a & p.b) for p in train])[ref.pair_N]
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

    rep.section("2  Content once size is taken out (partial r, as audit_content_no_size)")
    w_N = evaluate.cv_weights(ref); prow = []
    for subset, m in [("share no molecule", nN == 0), ("share 1 (audit)", nN == 1)]:
        Z = np.c_[np.ones(m.sum()), S[ref.pair_N[m]]]; w = w_N[m]
        for k_ in NAMES:
            r = res[k_]; fake = replace(r, pred_N=r.pred_N.copy(), truth_N=r.truth_N.copy())
            fake.pred_N[m] = wls_resid(Z, r.pred_N[m], w); fake.truth_N[m] = wls_resid(Z, r.truth_N[m], w)
            s = evaluate.score_cv(train, fake, mask_N=m)
            prow.append([NAMES[k_], subset, f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]"])
    rep.table(["model", "subset", "partial r (size removed) [95% CI]"], prow)

    rep.section("3  Full-data fit, rank 2: how far structure moves the profiles")
    f = O12e(q, X, A, B, 2, LAM).fit(np.arange(len(train)), y)
    shift_MxW = (f.x @ f.V @ f.U).numpy()
    rep.line(f"median |shift| / |q| over molecules: {np.median(np.linalg.norm(shift_MxW, axis=1) / np.linalg.norm(q, axis=1)):.3f}")
    words = features.load("openpom").words_W if hasattr(features.load("openpom"), "words_W") else None
    for j in range(2):
        u = f.U[j].numpy(); o = np.argsort(u)
        nm = (lambda i: words[i]) if words is not None else (lambda i: str(i))
        rep.line(f"axis {j + 1}: + " + ", ".join(nm(i) for i in o[::-1][:6]) + " | - " + ", ".join(nm(i) for i in o[:6]))
    rep.save()


if __name__ == "__main__":
    main()
