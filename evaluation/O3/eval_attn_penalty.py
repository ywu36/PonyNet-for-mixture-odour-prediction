# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Attention with its penalty scaled to the hidden layer (clue C29; user go-ahead 2026-11-20, "ok you can try with this").
C29 (training fits): penalty 1e-3 on |q|^2, copied from O1.2b, keeps the attention flat (2.8x weight range vs O1.2b 7.1x);
1e-4 gives O1.2b's spread (7.7x). 1e-4 is fixed in advance by that match, not by score. No overlap term: pairs sharing
molecules must come through the blend itself.
Rows (fixed before running), frozen backbone, 3 seeds averaged, attention + kappa, angle only (67 numbers):
  O1.2b (reference, 27) | <backbone>, attention penalty 1e-3 (as before) | <backbone>, attention penalty 1e-4
--backbone plain (o3m_w64, default) | counts (ga_w64)
CV: development pairs 50 x 5, subsets all / setting 1 / setting 2 / share 1 / share 2 / share >= 3; change from
O1.2b and 1e-4 minus 1e-3 on the same folds.  --heldout: T46 then V50, fitted once on all training pairs.

Env:     mist
Usage:   python evaluation/O3/eval_attn_penalty.py [--backbone plain|counts] [--heldout]
Writes:  results/eval_attn_penalty/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
LAM_O12 = 1e-3
PENALTIES = (1e-3, 1e-4)
REF = "O1.2b (reference, 27)"


def vectors(kind, dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    if kind == "plain":
        nets, g = [train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G
    else:
        nets, g = [train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg
    Vs = []
    for n in nets:
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
            Vs.append(n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy() / scale)
    return Vs


def models(Vs, label, q_op, X, A, B):
    out = {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM_O12).fit(tr, y[tr]).predict(te)}
    for lam in PENALTIES:
        out[f"{label}, attention penalty {lam:g} (67)"] = (lambda l_: lambda tr, te, y: np.mean(
            [ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True, lam=l_).fit(tr, y[tr]).predict(te) for v in Vs], 0))(lam)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="plain", choices=["plain", "counts"])
    ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    label = {"plain": "plain O3", "counts": "counts"}[a.backbone]
    rep = Report("eval_attn_penalty", {"backbone": a.backbone, "penalties": PENALTIES, "seeds": SEEDS, "heldout": a.heldout,
                                       "repeats": evaluate.CV_REPEATS, "device": dev}, tag=a.backbone + ("_heldout" if a.heldout else ""))
    Vs = vectors(a.backbone, dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    hi, lo = [f"{label}, attention penalty {lam:g} (67)" for lam in PENALTIES]

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
        res = {}
        for nm, fp in models(Vs, label, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[hi])) if nm == lo else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "1e-4 minus 1e-3 (same folds)"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(Vs, label, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
