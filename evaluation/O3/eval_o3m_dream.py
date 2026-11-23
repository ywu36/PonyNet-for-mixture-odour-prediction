# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Width-64 network with DREAM in training (train_o3m_mix.py --fold k --tag _dream): 5 folds x 1 (whole mixtures held
out, cv.mixture_holdout_folds seed 0), test predictions pooled over folds, mean over members. Same folds, paired:
  O1.2b (reference, 27 numbers fitted per fold)
  open-POM baseline (= B1a open-POM, 2 numbers)
  width 64 before any mixture training, plain mean of v (3 numbers fitted per fold)
  width 64 fine-tuned on Ma + GoodScents without DREAM (o3m_mix_ft_wsingle10), plain mean of v (3 numbers)
  width 64 with DREAM in training: real labels; shuffled labels (control) -- line fitted on the fold's training pairs
Paired: DREAM-trained minus no-DREAM fine-tune (what DREAM training adds); real minus shuffled. 4 subsets.

Env:     mist
Usage:   python evaluation/O3/eval_o3m_dream.py
Writes:  results/eval_o3m_dream/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import train_o3m  # noqa: E402

K, REPEATS, SEED = 5, 1, 0


def pooled(P, y, by_fold, folds):
    idx = np.concatenate([np.array(f.test) for f in folds]); pred = np.concatenate([by_fold[k] for k in range(len(folds))])
    return evaluate.CVResult(idx, pred, y[idx], np.array([P[i].dataset for i in idx]), np.array([P[i].jaccard == 0 for i in idx]), K, REPEATS, SEED)


def main():
    dev = pick_device(); rep = Report("eval_o3m_dream", {"k": K, "repeats": REPEATS, "seed": SEED, "lam": LAM, "device": dev})
    P = dev_pairs(); y = evaluate.labels(P); folds = list(cv.mixture_holdout_folds(P, k=K, repeats=REPEATS, seed=SEED))
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)  # API only
    A, B = membership(P, op.cids_M); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    q_op = mixture.quantile_per_word(op.probs_MxW); cvk = dict(k=K, repeats=REPEATS, seed=SEED)
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te), **cvk),
           "open-POM baseline (2 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, False, False, lam=LAM).fit(tr, y[tr]).predict(te), **cvk)}
    for name, folder in (("width 64, no mixture training", "o3m_w64"), ("width 64, Ma + GoodScents, no DREAM", "o3m_mix_ft_wsingle10")):
        with torch.no_grad():
            V = [train_o3m.load(f, dev).latent(o2.encode_all(train_o3m.load(f, dev), G, G.dream_rows(), dev))[0].double().cpu().numpy()
                 for f in sorted((ROOT / f"checkpoints/{folder}").glob("member_*.pt"))]
        res[f"{name} (3 numbers)"] = evaluate.cross_validate(
            P, lambda tr, te, V=V: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in V], 0), **cvk)
    for mode in ("ft", "shuffle"):
        cs = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted((ROOT / f"checkpoints/o3m_mix_{mode}_dream").glob("fold*_member*.pt"))]
        if not cs:
            rep.line(f"{mode}: no checkpoints"); continue
        by_fold = {}
        for k in range(K):
            ck = [c["after"] for c in cs if c["after"].get("fold") == k]
            assert ck and all(np.array_equal(c["test_idx"], np.array(folds[k].test)) for c in ck), (mode, k)
            by_fold[k] = np.mean([c["test_pred"] for c in ck], 0)
        rep.line(f"DREAM-trained {mode}: {len(cs)} members; single AUROC {np.mean([c['after']['single_auroc'] for c in cs]):.4f}; "
                 f"test r per member {', '.join(f'{c['after']['dream_test_r']:.3f}' for c in cs)}; training r {np.mean([c['after']['dream_train_r'] for c in cs]):.3f}")
        res[f"width 64, Ma + GoodScents + DREAM ({'real labels' if mode == 'ft' else 'shuffled labels, control'})"] = pooled(P, y, by_fold, folds)
    ref = res["O1.2b (reference, 27 numbers)"]; opb = res["open-POM baseline (2 numbers)"]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs, {K} x {REPEATS} folds (same folds for all rows): RMSE / r change [95% CI], positive r = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if r_ is ref else ch(s_), "" if r_ is opb else ch(sc(r_, sub, opb))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from open-POM baseline"], rows)
    nod = res["width 64, Ma + GoodScents, no DREAM (3 numbers)"]; rows = []
    real = res.get("width 64, Ma + GoodScents + DREAM (real labels)"); shuf = res.get("width 64, Ma + GoodScents + DREAM (shuffled labels, control)")
    for sub in SUB:
        if real is not None:
            rows.append(["DREAM-trained (real)", "minus no-DREAM fine-tune", sub, ch(sc(real, sub, nod))])
        if real is not None and shuf is not None:
            rows.append(["DREAM-trained (real)", "minus shuffled control", sub, ch(sc(real, sub, shuf))])
    rep.table(["model", "comparison (same folds)", "subset", "RMSE / r change [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
