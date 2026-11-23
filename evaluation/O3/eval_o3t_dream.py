# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = middle layer (400)
# ---------------------------------------------------------------------------------------------
"""O3-T rung 4 (DREAM in training, per fold) on DREAM: 5 folds x 1 repeat (train_o2d_dev's folds, whole mixtures held
out), test predictions pooled over folds, mean over seeds. Same folds, paired:
  O1.2b (reference, 27 numbers fitted per fold)
  rung 3 (same losses, no DREAM in training), templates by cosine, theta fitted per fold (27)
  rung 4 anchored (DREAM as comparisons) and rung 4 mse (DREAM as values): templates with the trained theta and beta,
  line fitted on the fold's training pairs
Paired: rung 4 minus rung 3 (what DREAM training adds), anchored minus mse (comparisons vs values). 4 subsets.

Env:     mist
Usage:   python evaluation/O3/eval_o3t_dream.py
Writes:  results/eval_o3t_dream/<date_time>_rung4.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402

K, REPEATS, SEED = 5, 1, 0


def pooled(P, y, by_fold, folds):
    idx = np.concatenate([np.array(f.test) for f in folds]); pred = np.concatenate([by_fold[k] for k in range(len(folds))])
    return evaluate.CVResult(idx, pred, y[idx], np.array([P[i].dataset for i in idx]), np.array([P[i].jaccard == 0 for i in idx]), K, REPEATS, SEED)


def main():
    dev = pick_device(); rep = Report("eval_o3t_dream", {"k": K, "repeats": REPEATS, "seed": SEED, "lam": LAM}, tag="rung4")
    from mixdist import cv
    P = dev_pairs(); y = evaluate.labels(P); folds = list(cv.mixture_holdout_folds(P, k=K, repeats=REPEATS, seed=SEED))
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    res = {"O1.2b (reference)": evaluate.cross_validate(P, lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM)
                                                        .fit(tr, y[tr]).predict(te), k=K, repeats=REPEATS, seed=SEED)}
    _, mu3, _, _ = members(G, "checkpoints/o3t_r3", dev); V3 = [m_.double().cpu().numpy() for m_ in mu3]
    res["rung 3 (no DREAM in training), template, theta fitted (27)"] = evaluate.cross_validate(
        P, lambda tr, te: np.mean([O12b(v, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in V3], 0), k=K, repeats=REPEATS, seed=SEED)
    for dl in ("anchored", "mse"):
        cs = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted((ROOT / f"checkpoints/o3t_r4_{dl}").glob("fold*_seed*.pt"))]
        if not cs:
            rep.line(f"rung 4 {dl}: no checkpoints"); continue
        by_fold = {}
        for k in range(K):
            ck = [c for c in cs if c["fold"] == k]
            assert ck and all(np.array_equal(c["test_idx"], np.array(folds[k].test)) for c in ck), (dl, k)
            by_fold[k] = np.mean([c["test_pred"] for c in ck], 0)
        rep.line(f"rung 4 {dl}: {len(cs)} members; test r per member {', '.join(f'{c['test_r']:.3f}' for c in cs)}; "
                 f"training r {np.mean([c['train_r'] for c in cs]):.3f}; single AUROC {np.mean([c['single_auroc'] for c in cs]):.4f}")
        res[f"rung 4 {dl} (DREAM in training), template, trained theta + beta, line on training pairs"] = pooled(P, y, by_fold, folds)
    ref, r3 = res["O1.2b (reference)"], res["rung 3 (no DREAM in training), template, theta fitted (27)"]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs, {K} x {REPEATS} folds (whole mixtures held out; same folds for all rows)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", "" if r_ is ref else ch(s_)])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b"], rows)
    rows = []
    r4 = {k: v for k, v in res.items() if k.startswith("rung 4")}
    for k, r_ in r4.items():
        for sub in SUB:
            rows.append([k.split(" (")[0], "minus rung 3", sub, ch(sc(r_, sub, r3))])
    if len(r4) == 2:
        ka, km = list(r4)
        for sub in SUB:
            rows.append(["rung 4 anchored", "minus rung 4 mse", sub, ch(sc(r4[ka], sub, r4[km]))])
    rep.table(["model", "comparison (same folds)", "subset", "RMSE / r change [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
