# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   G = RDKit groups (24)   P = development pairs (514)
#   N = pooled out-of-fold predictions
# ---------------------------------------------------------------------------------------------
"""O1.2k: kernel set distance (user, 2026-11-18: "keep the functional modules uncompressed until the comparison").
Declared before running.

Why: every B1/O1.2 model averages a mixture into one profile before comparing (a compression); pooling inside a
network made it worse (O2C-in -0.116 on overlap pairs). Our data put identity at the molecule level (shared molecule
pulls, shared group/note does not) and strength at the group level (gain). Drug-side evidence: GEARS keeps per-unit
embeddings and sums them; Hu et al. 2020 show compressing to graph level only gives negative transfer.
Model (mixdist/o12.py O12k): each mixture stays a gain-weighted set of single-molecule profiles q; two sets are
compared by MMD with a Gaussian kernel (sigma learned, starts at the median molecule distance).

Rows (development pairs only, CV.md rule 7; standard CV 50 x 5 seed 0; lockbox, V50, T46 untouched):
  B1a++ (reference)  |  O1.2b  |
  set:   d = sqrt(MMD^2 + beta (h_A - h_B)^2)                      molecules kept separate, O1.2b's length term kept
  add:   d = sqrt(|u_A - u_B|^2 + beta dh^2 + gamma MMD^2)         O1.2b + set term (gamma = 0 is O1.2b)
  addI:  add with K = identity (control: only exact molecule identity counts, no similarity between molecules)
Lead = "add" or "set" r change vs O1.2b with CI > 0 on all pairs or share no molecule, RMSE not worse, and above addI
on share no molecule. Reported: gamma, sigma / median, beta over the CV fits; full-dev-data fit. CPU (small L-BFGS
on 235 molecules / ~410 pairs per fit; single-threaded parallel runs).

Env:     mist
Usage:   python evaluation/O1/eval_o12k.py --variant b1a2|o12b|set|add|addI ; then --report
Writes:  results/eval_o12k/cv_<variant>.pkl ; report results/eval_o12k/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import cv, data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.o12 import O12b, O12k  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3
OUT = ROOT / "results/eval_o12k"
NAMES = {"b1a2": "B1a++ (reference)", "o12b": "O1.2b", "set": "set distance + length (O1.2k set)",
         "add": "O1.2b + set term (O1.2k add)", "addI": "control: O1.2b + identity-kernel set term"}


def setup():
    train = data.load_train(); P = [train[i] for i in cv.lockbox_split(train).train]
    y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(P, op.cids_M); S = size_features(P)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    return P, y, q, A, B, S, X


def make(name, y, q, A, B, S, X, store):
    if name == "b1a2":
        return lambda tr, te: B1aPlus(q, X, A, B, S, LAM).fit(tr, y[tr]).predict(te)
    if name == "o12b":
        return lambda tr, te: O12b(q, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)
    mode = "set" if name == "set" else "add"

    def fp(tr, te):
        f = O12k(q, X, A, B, mode=mode, lam=LAM, identity=(name == "addI")).fit(tr, y[tr])
        store.append([f.gamma, f.sigma / np.exp(f.s0), f.beta]); return f.predict(te)
    return fp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    P, y, q, A, B, S, X = setup(); OUT.mkdir(parents=True, exist_ok=True)
    if a.variant:
        store = []
        res = evaluate.cross_validate(P, make(a.variant, y, q, A, B, S, X, store))
        pickle.dump({"res": res, "params": store}, open(OUT / f"cv_{a.variant}.pkl", "wb")); print("saved", a.variant); return

    rep = Report("eval_o12k", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "pairs": "development (CV.md rule 7)"})
    nsh = np.array([len(p.a & p.b) for p in P])
    rep.section("0  Audit")
    allp = np.arange(len(P)); import torch
    f0 = O12k(q, X, A, B, mode="add"); th = torch.zeros(X.shape[1]); k = torch.tensor(0.7)
    d_b = O12b(q, X, A, B, True, True).distance(th, k, allp); d_k = f0.distance(th, k, allp, torch.tensor(0.0), torch.tensor(f0.s0))
    rep.line(f"O1.2k add at gamma = 0 vs O1.2b: max |difference| {float((d_b - d_k).abs().max()):.2e}")
    rep.line(f"development pairs {len(P)}: share 0 {int((nsh == 0).sum())}, share >= 1 {int((nsh >= 1).sum())}; median molecule distance {np.exp(f0.s0):.2f}")
    run = {k_: pickle.load(open(OUT / f"cv_{k_}.pkl", "rb")) for k_ in NAMES}
    res = {k_: v["res"] for k_, v in run.items()}; ref = res["b1a2"]
    assert all(np.array_equal(r.pair_N, ref.pair_N) for r in res.values()), "folds differ"
    nN = nsh[ref.pair_N]
    fmt = lambda s: (f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / "
                     f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]")
    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5) on development pairs: change from B1a++ and from O1.2b (same pairs)")
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", nN == 0), ("share >= 1", nN >= 1)]:
        for k_ in NAMES:
            s1 = evaluate.score_cv(P, res[k_], reference=None if k_ == "b1a2" else ref, mask_N=mask)
            s2 = "" if k_ in ("b1a2", "o12b") else fmt(evaluate.score_cv(P, res[k_], reference=res["o12b"], mask_N=mask))
            rows.append([NAMES[k_], subset, s1["pairs"], f"{s1['rmse']:.4f}", f"{s1['r']:.3f} [{s1['r_ci'][0]:.3f}, {s1['r_ci'][1]:.3f}]",
                         "" if k_ == "b1a2" else fmt(s1), s2])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE [CI] / r [CI]", "change from O1.2b"], rows)
    rep.section("2  Parameters over the 250 CV fits (median [2.5, 97.5%]) and on all development pairs")
    prow = []
    for k_ in ("set", "add", "addI"):
        Pm = np.array(run[k_]["params"]); f = O12k(q, X, A, B, mode="set" if k_ == "set" else "add", lam=LAM, identity=(k_ == "addI")).fit(allp, y)
        cvs = "; ".join(f"{nm} {np.median(Pm[:, j]):.3g} [{np.percentile(Pm[:, j], 2.5):.3g}, {np.percentile(Pm[:, j], 97.5):.3g}]"
                        for j, nm in enumerate(["gamma", "sigma/median", "beta"]))
        prow.append([NAMES[k_], cvs, f"gamma {f.gamma:.3g}, sigma/median {f.sigma / np.exp(f.s0):.3g}, beta {f.beta:.3g}"])
    rep.table(["model", "CV fits", "all development pairs"], prow)
    rep.line("gamma is unused in 'set' (the set term replaces the direction); sigma unused in addI (K = identity).")
    rep.save()


if __name__ == "__main__":
    main()
