# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = DREAM pairs (820)   N = pooled test predictions
# ---------------------------------------------------------------------------------------------
"""O2-D (design/o2j_design.md, "DREAM in-fold"): O2-N trained from scratch on single molecules (anchor) + the DREAM
training pairs of one CV fold, end to end (merge -> decode -> direction + agreement length -> line, MSE).
Small scale: 5 folds x 1 repeat (whole mixtures held out, seed 0), 3 members per fold (seeds 0-2), ensemble mean.
--report scores the held-out predictions against B1a++ and frozen O2-N (O2N-in + agreement) on the same 5 folds.

Env:     mist (GPU)
Usage:   python training/train_o2d.py --fold 0 1        (one process per GPU is fine)
         python training/train_o2d.py --report
Writes:  checkpoints/o2d/fold<k>_seed<s>.pt (predictions inside), results/train_o2d/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import cv, data, evaluate, features, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402

SEEDS = (0, 1, 2)
CK = ROOT / "checkpoints/o2d"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, nargs="*", default=[]); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("train_o2d", {**vars(a), "device": dev, "seeds": SEEDS, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); train = data.load_train(); y = evaluate.labels(train)
    gslf = np.arange(G.n_gslf); single_ids = gslf[~G.audit_test]
    folds = list(cv.mixture_holdout_folds(train, k=5, repeats=1, seed=0))
    D = o2j.DreamStream(G, train, [int(c) for c in G.dream_cids], dev); CK.mkdir(parents=True, exist_ok=True)
    for k in a.fold:
        tr, te = np.array(folds[k].train), np.array(folds[k].test)
        for s in SEEDS:
            t0 = time.time()
            m = o2j.train_dream(G, single_ids, D, y, tr, s, dev, log=rep.line)
            p = o2j.dream_predict(m, G, D, te, dev); ptr = o2j.dream_predict(m, G, D, tr, dev)
            auc = o2.mean_auroc(o2.predict_probs(m.net, G, gslf[G.audit_test], dev), G.Y[gslf[G.audit_test]])
            torch.save({"model_state_dict": m.state_dict(), "fold": k, "seed": s, "test_idx": te, "pred": p, "single_auroc": auc}, CK / f"fold{k}_seed{s}.pt")
            rep.line(f"fold {k} seed {s}: train r {np.corrcoef(ptr, y[tr])[0, 1]:.3f}, held-out r {np.corrcoef(p, y[te])[0, 1]:.3f}, "
                     f"single-molecule AUROC {auc:.4f}, {time.time() - t0:.0f} s")
    if a.report:
        from mixdist.b1a_plus import B1aPlus, membership, size_features
        from eval_b1a_plus import piece_matrix
        saved = {}
        for f in CK.glob("fold*_seed*.pt"):
            c = torch.load(f, map_location="cpu", weights_only=False); saved.setdefault(c["fold"], []).append(c)
        rep.line(f"folds with predictions: {sorted(saved)}; members per fold {[len(saved[k]) for k in sorted(saved)]}")
        by_test = {tuple(folds[k].test): np.mean([c["pred"] for c in saved[k]], 0) for k in saved}
        o2d = evaluate.cross_validate(train, lambda tr, te: by_test[tuple(te)], repeats=1)
        op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
        X, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
        A, B = membership(train, op.cids_M); S = size_features(train); q = mixture.quantile_per_word(op.probs_MxW)
        ref = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, X, A, B, S, 1e-3).fit(tr, y[tr]).predict(te), repeats=1)
        rows = []
        for subset in ("all pairs", "share no molecule"):
            mk = lambda r_: None if subset == "all pairs" else r_.no_overlap_N  # noqa: E731
            s0 = evaluate.score_cv(train, ref, mask_N=mk(ref)); s1 = evaluate.score_cv(train, o2d, reference=ref, mask_N=mk(o2d))
            rows.append(["B1a++ (baseline)", subset, f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
            rows.append(["O2-N-Single+DREAM (in-fold)", subset, f"{s1['rmse']:.4f}", f"{s1['r']:.3f} [{s1['r_ci'][0]:.3f}, {s1['r_ci'][1]:.3f}]",
                         f"{s1['rmse_change']:+.4f} / {s1['r_change']:+.3f} [{s1['r_change_ci'][0]:+.3f}, {s1['r_change_ci'][1]:+.3f}]"])
        rep.section("DREAM held-out mixtures, 5 folds x 1 repeat (same folds for both)")
        rep.table(["model", "subset", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]"], rows)
        aucs = [c["single_auroc"] for k in saved for c in saved[k]]
        rep.line(f"single-molecule held-out AUROC of the O2-D members: {min(aucs):.4f}-{max(aucs):.4f} (O2-N 0.875-0.880)")
    rep.save()


if __name__ == "__main__":
    main()
