# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   W = words (138)   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""Clean rerun of "structure changes the odour profile" (O1.2e, rank 2), declared before running.

Why: in eval_o12e the declared lam 1e-3 forced the shift to 0; lam 1e-4 (picked after seeing that, from 2 tries) gave
share-none r +0.103 vs B1a++ but RMSE +0.0019 (predictions too spread). Here both choices move inside the training fold:
  penalty: inner CV (3 folds, whole mixtures held out, within the outer training pairs) over lam in
           {1e-3, 3e-4, 1e-4, 3e-5}, pick the lowest inner MSE (1e-3 = shift 0, i.e. O1.2b; selection may fall back)
  calibration: final line pred' = a pred + c fitted on the inner out-of-fold predictions of the picked lam
Rows, change from B1a++ on the same folds (B1a++, O1.2b reused from results/eval_o12c):
  B1a++ (baseline) | O1.2b | control: O1.2b + the same inner calibration | O1.2e nested lam + calibration
Subsets: all pairs, share no molecule; share >= 1; share 1-2 audit; partial r with size taken out. Also: which lam
each outer fit picked. Runs in shards of the 250 outer folds (same folds as cross_validate), then --assemble.
CPU (small L-BFGS fits, run in parallel single-threaded shards).

Env:     mist
Usage:   python evaluation/O1/eval_o12f.py --model e|b --shard i --of n ; then --assemble ; then --report
Writes:  results/eval_o12f/shard_<model>_<i>.pkl ; results/eval_o12c/cv_o12f_<model>.pkl ; results/eval_o12f/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from mixdist import cv, evaluate  # noqa: E402
from mixdist.o12 import O12b, O12e  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_o12c import setup  # noqa: E402
from audit_content_no_size import wls_resid  # noqa: E402

OUT = ROOT / "results/eval_o12f"; CV = ROOT / "results/eval_o12c"
GRID = {"e": [1e-3, 3e-4, 1e-4, 3e-5], "b": [1e-3]}
NAMES = {"b1a2": "B1a++ (baseline)", "o12b": "O1.2b gain + length", "o12f_b": "control: O1.2b + inner calibration",
         "o12f_e": "O1.2e structure shift, nested lam + calibration"}


def make(model, lam, q, X, A, B):
    return O12e(q, X, A, B, 2, lam) if model == "e" else O12b(q, X, A, B, True, True, lam=lam)


def nested(model, tr, te, train, y, q, X, A, B):
    sub = [train[i] for i in tr]
    inner = cv.mixture_holdout_folds(sub, k=3, repeats=1, seed=1)
    best = None
    for lam in GRID[model]:
        idx, pr = [], []
        for f in inner:
            itr, ite = tr[np.array(f.train)], tr[np.array(f.test)]
            pr.append(make(model, lam, q, X, A, B).fit(itr, y[itr]).predict(ite)); idx.append(ite)
        idx = np.concatenate(idx); pr = np.concatenate(pr); mse = np.mean((pr - y[idx]) ** 2)
        if best is None or mse < best[0]:
            best = (mse, lam, idx, pr)
    _, lam, idx, pr = best
    a, c = np.polyfit(pr, y[idx], 1)
    return a * make(model, lam, q, X, A, B).fit(tr, y[tr]).predict(te) + c, lam, a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--shard", type=int); ap.add_argument("--of", type=int)
    ap.add_argument("--assemble", action="store_true"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train, y, q, A, B, S, X, C = setup()
    folds = cv.mixture_holdout_folds(train, k=5, repeats=evaluate.CV_REPEATS, seed=0)
    if a.model and a.shard is not None:
        out = []
        for i in range(a.shard, len(folds), a.of):
            tr, te = np.array(folds[i].train), np.array(folds[i].test)
            pred, lam, slope = nested(a.model, tr, te, train, y, q, X, A, B)
            out.append((i, te, pred, lam, slope))
        pickle.dump(out, open(OUT / f"shard_{a.model}_{a.shard}.pkl", "wb")); print("saved shard", a.model, a.shard); return
    if a.assemble:
        for model in ("e", "b"):
            parts = sorted([x for f in OUT.glob(f"shard_{model}_*.pkl") for x in pickle.load(open(f, "rb"))], key=lambda t: t[0])
            assert [t[0] for t in parts] == list(range(len(folds))), f"missing folds for {model}"
            pair_N = np.concatenate([t[1] for t in parts]); pred_N = np.concatenate([t[2] for t in parts])
            res = evaluate.CVResult(pair_N, pred_N, y[pair_N], np.array([train[i].dataset for i in pair_N]),
                                    np.array([train[i].jaccard == 0 for i in pair_N]), 5, evaluate.CV_REPEATS, 0)
            pickle.dump({"res": res, "lams": [t[3] for t in parts], "slopes": [t[4] for t in parts]}, open(CV / f"cv_o12f_{model}.pkl", "wb"))
        print("assembled"); return

    rep = Report("eval_o12f", {"grid": GRID["e"], "repeats": evaluate.CV_REPEATS})
    run = {k_: pickle.load(open(CV / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
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
    lams, cnt = np.unique(run["o12f_e"]["lams"], return_counts=True)
    rep.line("O1.2e lam picked by the inner CV (of 250 outer fits): " + ", ".join(f"{l:g}: {c}" for l, c in zip(lams, cnt)))
    for k_ in ("o12f_b", "o12f_e"):
        sl = np.array(run[k_]["slopes"]); rep.line(f"{NAMES[k_]}: calibration slope median {np.median(sl):.2f} [{np.percentile(sl, 2.5):.2f}, {np.percentile(sl, 97.5):.2f}]")
    rep.save()


if __name__ == "__main__":
    main()
