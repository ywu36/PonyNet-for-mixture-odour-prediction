# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   N = CV predictions
# ---------------------------------------------------------------------------------------------
"""Paper ablation (user, 2026-11-20: "start the ablation study"): one script, 10 members per network, the same folds,
CV on every subset + held-out. Each row adds one part; the last row of the chain is the reported model (finding 7
section 12, packaged as checkpoints/final_o5b by eval_o5_softmax / package_o5b).
Rows (fixed before running):
  A0  O1.2b (27)                                                   reference
  A1  plain O3, plain average of molecule vectors, equal weights   network only; line fitted (2)
  A2  plain O3, unit merge, equal weights                          + unit merge (2)
  A3  O5, unit merge, equal weights                                + atom attention inside the network (2)
  A4  O5 + odd-one-out                                             + context term (3)
  A5  O5 + own strength + odd-one-out                              + O5's atom scores as molecule weights (4)
  A6  O5 + DREAM correction toward own scorer                      reported model (69; eval_o5_softmax.PriorAtom "shrink")
  S1  plain O3 + DREAM atom soft max                               side route: scorer learned on DREAM from scratch (68)
Networks: checkpoints/o3m_w64 and checkpoints/o3m_attpool, seeds 0-9 (eval_o5_softmax.backbones; GS-LF singles only,
frozen). Readouts fitted per member, predictions averaged. Non-finite predictions are masked in every row; CV
predictions are pickled. No overlap term, no routing, no study offsets; lockbox not used.

Env:     mist
Usage:   python evaluation/O3/eval_ablation.py --heldout
         python evaluation/O3/eval_ablation.py --rows A0,A1,A2,A3,A4,A5     (CV part; pickles predictions)
         python evaluation/O3/eval_ablation.py --score <pkl> [<pkl> ...]    (merge CV parts and score)
Writes:  results/eval_ablation/<date_time>[_heldout|_cv_<rows>|_score].{txt,json,pkl}; held-out predictions: _heldout_pred.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report, results_dir  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_attpool import LearnedStrength  # noqa: E402
from eval_o5_softmax import backbones, PriorAtom, LAM  # noqa: E402
import eval_context_dream as ecd  # noqa: E402

NAMES = {"A0": "A0 O1.2b (27)", "A1": "A1 plain O3, plain average, equal weights (2)", "A2": "A2 plain O3, unit merge (2)",
         "A3": "A3 O5, unit merge (2)", "A4": "A4 O5 + odd-one-out (3)", "A5": "A5 O5 + own strength + odd-one-out (4)",
         "A6": "A6 O5 + DREAM correction toward own scorer (69) [reported]", "S1": "S1 plain O3 + DREAM atom soft max (68)"}
CHAIN = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]


class FixedAngle:
    """Equal weights, no fitted weighting: angle between the two mixture averages (unit or plain), then a fitted line."""

    def __init__(self, V, A, B, unit):
        U = V / np.linalg.norm(V, axis=1, keepdims=True) if unit else V
        za = A @ U / A.sum(1, keepdims=True); zb = B @ U / B.sum(1, keepdims=True)
        za /= np.linalg.norm(za, axis=1, keepdims=True); zb /= np.linalg.norm(zb, axis=1, keepdims=True)
        self.d = np.linalg.norm(za - zb, axis=1)

    def fit(self, idx, y):
        D = np.c_[self.d[idx], np.ones(len(idx))]; self.lin = np.linalg.lstsq(D, y, rcond=None)[0]; return self

    def predict(self, idx):
        return self.lin[0] * self.d[idx] + self.lin[1]


def models(BB, q_op, X, A, B, keys):
    def ens(k, make):
        return lambda tr, te, y: np.mean([make(m).fit(tr, y[tr]).predict(te) for m in BB[k]], 0)
    d = {"A0": lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
         "A1": ens("plain", lambda m: FixedAngle(m["V"], A, B, False)),
         "A2": ens("plain", lambda m: FixedAngle(m["V"], A, B, True)),
         "A3": ens("o5", lambda m: FixedAngle(m["V"], A, B, True)),
         "A4": ens("o5", lambda m: LearnedStrength(m["V"], m["strength"], A, B, False)),
         "A5": ens("o5", lambda m: LearnedStrength(m["V"], m["strength"], A, B, True)),
         "A6": ens("o5", lambda m: PriorAtom(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], "shrink")),
         "S1": ens("plain", lambda m: UnitAtom(m["V"], m["H"], m["mol"], A, B))}
    return {k: d[k] for k in keys}


def subsets(P):
    k = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
    return {"all pairs": None, "share none": k == 0, "share none, same size (gap <= 1)": (k == 0) & (gap <= 1),
            "share none, sizes differ (gap >= 2)": (k == 0) & (gap >= 2), "share >= 1": k >= 1, "share 1": k == 1,
            "share 2": k == 2, "share >= 3": k >= 3}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); ap.add_argument("--rows", default="")
    ap.add_argument("--score", nargs="*", default=None); a = ap.parse_args(); ecd.DIRECTION_ONLY = True
    tag = "heldout" if a.heldout else ("score" if a.score is not None else "cv_" + a.rows.replace(",", "-"))
    rep = Report("eval_ablation", {"heldout": a.heldout, "rows": a.rows, "score": a.score, "lam": LAM, "repeats": evaluate.CV_REPEATS}, tag=tag)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if a.score is not None:                                   # merge CV parts and score every subset
        res = {}
        for f in a.score:
            res.update(pickle.load(open(f, "rb"))); rep.line(f"predictions from {f}")
        P = dev_pairs(); SUB = subsets(P); keys = [k for k in CHAIN + ["S1"] if k in res]
        fin_N = np.all([np.isfinite(res[k].pred_N) for k in keys], 0)
        for k in keys:
            rep.line(f"non-finite predictions: {NAMES[k]}: {(~np.isfinite(res[k].pred_N)).sum()} of {len(fin_N)}")
        rep.line(f"masked in every row: {(~fin_N).sum()} of {len(fin_N)}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=fin_N if m is None else fin_N & m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r [95% CI]; changes are RMSE / r [95% CI]")
        rows = []
        for sub in SUB:
            for k in keys:
                s_ = sc(res[k], sub, None if k == "A0" else res["A0"])
                prev = CHAIN[CHAIN.index(k) - 1] if k in CHAIN and k not in ("A0", "A1") else ("A2" if k == "S1" else None)
                rows.append([NAMES[k], sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if k == "A0" else ch(s_), "" if prev is None else f"vs {prev}: " + ch(sc(res[k], sub, res[prev]))])
        rep.table(["row", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from previous step"], rows)
        rep.save(); return
    dev = pick_device(); _, BB = backbones(dev, rep); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    if a.heldout:
        train = data.load_train(); keys = CHAIN + ["S1"]; saved = {}
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {k: fp(tr, te, y_all) for k, fp in models(BB, q_op, X, A, B, keys).items()}
            y = hs.y_P; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for k in keys:
                p = preds[k]; ok = np.isfinite(p); s = metrics.score(p[ok], y[ok], ceiling=hs.ceiling); c1 = c2 = ""
                if k != "A0" and ok.all():
                    cr, ce = paired(p, preds["A0"], y)
                    c1 = f"{s['rmse'] - metrics.rmse(preds['A0'], y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / {s['r'] - metrics.pearson(preds['A0'], y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                prev = CHAIN[CHAIN.index(k) - 1] if k in CHAIN and k not in ("A0", "A1") else ("A2" if k == "S1" else None)
                if prev and ok.all():
                    cr, ce = paired(p, preds[prev], y)
                    c2 = f"vs {prev}: {s['r'] - metrics.pearson(preds[prev], y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([NAMES[k], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]",
                             f"{p[ok].mean() - y[ok].mean():+.3f}", c1 + ("" if ok.all() else f" ({(~ok).sum()} non-finite)"), c2])
            rep.table(["row", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE [CI] / r [CI]", "r change from previous step [CI]"], rows)
            rep.value(hs.name, {NAMES[k]: {"rmse": metrics.rmse(preds[k], y), "r": metrics.pearson(preds[k], y)} for k in keys})
            saved[hs.name] = {"pairs": list(hs.pairs), "y": y, "pred": preds}
        pk = results_dir() / "eval_ablation" / f"{rep.time:%Y-%m-%d_%H%M%S}_heldout_pred.pkl"; pk.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(saved, open(pk, "wb")); rep.line(f"held-out predictions saved to {pk}")
        rep.save(); return
    keys = [k.strip() for k in a.rows.split(",") if k.strip()]
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M); res = {}
    pk = results_dir() / "eval_ablation" / f"{rep.time:%Y-%m-%d_%H%M%S}_{tag}.pkl"; pk.parent.mkdir(parents=True, exist_ok=True)
    for k, fp in models(BB, q_op, X, A, B, keys).items():
        res[k] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {NAMES[k]}")
        pickle.dump(res, open(pk, "wb"))
    rep.line(f"predictions saved to {pk}"); rep.save()


if __name__ == "__main__":
    main()
