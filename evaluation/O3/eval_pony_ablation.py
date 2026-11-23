# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   T = atoms
# ---------------------------------------------------------------------------------------------
"""Removal ablation of the final model ("Pony" on the paper page = O6 without the size loss; user 2026-11-22 "can you
also do ablation study by remove things? another like table 6 but - not +").
Final model = eval_size_tune row "(b) + hidden length, learned (133)": O5 backbone (10 members, frozen), atom scores
h_a . (beta q0 + delta), molecule log-weight tau log mean_a exp(score / tau) + kappa cos(v_m, mean of the others),
unit merge, mixture vector exp(gamma . z) z, Euclidean distance, line; loss MSE + 1e-3 (beta^2 + kappa^2 + |delta|^2 +
|gamma|^2); L-BFGS 300 iterations from zero. Each row removes one part (its numbers held at 0, i.e. at their default):
  Pony (full)                      133 numbers   must reproduce eval_size_tune: T46 0.593, V50 0.622, CV 0.464 (audit)
  - odd one out                    kappa = 0
  - anchored correction            delta = 0 and tau = 1 (t = 0): strength = scale x own scorer, as in build-up step 3
  - strength from atom attention   beta = delta = 0, tau = 1: every molecule weighs the same apart from kappa
  - mixture length                 gamma = 0 (= model (b))
  - L2 penalty                     lambda = 0
  - odd one out and anchored correction        kappa = delta = 0, tau = 1   (row 6)
  - anchored correction and mixture length     delta = gamma = 0, tau = 1   (row 7 = Table 7 step 3)
Rows fixed before running. CV: standard 50 x 5 mixture holdout on the development pairs (same folds as eval_size_tune),
run in repeat chunks (--reps a:b) and joined with --merge; --heldout: T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_pony_ablation.py --heldout
         python evaluation/O3/eval_pony_ablation.py --reps 0:13   (also 13:25, 25:38, 38:50)
         python evaluation/O3/eval_pony_ablation.py --reps 0:13 --rows 0,1   (only these rows; pkl saved after each row)
         python evaluation/O3/eval_pony_ablation.py --merge results/eval_pony_ablation/*_reps*.pkl
Writes:  results/eval_pony_ablation/<date_time>.{txt,json} (+ .pkl)
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, evaluate, features, metrics, data  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402
import eval_size_tune as est  # noqa: E402

F64 = torch.float64
LAM = 1e-3
FULL = "Pony (full)"
ROWS = {FULL: set(), "- odd one out": {"kappa"}, "- anchored correction": {"delta", "tau"},
        "- strength from atom attention": {"beta", "delta", "tau"}, "- mixture length": {"gamma"}, "- L2 penalty": {"lam"},
        "- odd one out and anchored correction": {"kappa", "delta", "tau"},   # row 6 added 2026-11-22 on user request
        "- anchored correction and mixture length (= build-up step 3)": {"delta", "tau", "gamma"}}   # row 7: build-up step 3 refitted with beta inside the atom pooling (paper Table 7: T46 0.483 V50 0.603 CV 0.416); eval_ablation row A5 (0.492 / 0.606 / 0.423) puts beta outside the pooling
OUT = ROOT / "results/eval_pony_ablation"


class Ablate(est.SizeTune):
    def __init__(self, m, A, B, off):
        super().__init__(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], "hl64+plain", 0.0); self.off = off

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float)); off = self.off
        z = lambda *s_, on=True: torch.zeros(*s_, dtype=F64, requires_grad=on)
        beta, kappa, t = z((), on="beta" not in off), z((), on="kappa" not in off), z((), on="tau" not in off)
        delta, cvec = z(self.H.shape[1], on="delta" not in off), z(self.V.shape[1], on="gamma" not in off)
        b0 = torch.zeros((), dtype=F64); self.cvec = cvec; lam = 0.0 if "lam" in off else LAM
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t); d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        ps = [lin] + [p_ for p_ in (beta, kappa, t, delta, cvec) if p_.requires_grad]
        opt = torch.optim.LBFGS(ps, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.smax(self.score(beta, delta), t); d = self.distance(kappa, b0, idx)
            loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + lam * (beta ** 2 + kappa ** 2 + (delta ** 2).sum() + (cvec ** 2).sum())
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t).detach()
        self.beta, self.kappa, self.t, self.delta, self.b, self.lin, self.cvec = (beta.detach(), kappa.detach(), t.detach(),
                                                                                  delta.detach(), b0, lin.detach(), cvec.detach())
        return self


def models(BB, A, B):
    return {nm: (lambda tr, te, y, off=off: np.mean([Ablate(m, A, B, off).fit(tr, y[tr]).predict(te) for m in BB], 0))
            for nm, off in ROWS.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); ap.add_argument("--reps", default="")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--rows", default="", help="comma-separated row numbers (0 = full) to run; default all"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    tag = "heldout" if a.heldout else (f"reps{a.reps.replace(':', '-')}" if a.reps else "score")
    rep = Report("eval_pony_ablation", {"lam": LAM, "members": eo.SEEDS, "rows": list(ROWS), "device": dev}, tag=tag)
    OUT.mkdir(parents=True, exist_ok=True); stamp = time.strftime('%Y-%m-%d_%H%M%S')
    op = features.load("openpom")
    if a.merge is None:
        BB = eo.backbones(dev, rep)[1]["o5"]
    if a.heldout:
        train = data.load_train(); store = {}
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            keep = [list(ROWS)[int(i)] for i in a.rows.split(",")] if a.rows else list(ROWS)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, A, B).items() if nm in keep or nm == FULL}; y = hs.y_P; store[hs.name] = {"y": y, "pred": preds}
            rep.section(f"{hs.name}: {len(y)} pairs"); rows = []
            for nm, p in preds.items():
                s = metrics.score(p, y); c = ""
                if nm != FULL:
                    cr, _ = paired(p, preds[FULL], y); c = f"{s['r'] - metrics.pearson(preds[FULL], y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", c])
            rep.table(["model", "RMSE", "r [95% CI]", "r change from full [CI]"], rows)
        pickle.dump(store, open(OUT / f"{stamp}_heldout_pred.pkl", "wb")); rep.save(); return
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    if a.reps:
        ra, rb_ = map(int, a.reps.split(":")); folds = [f for f in cv.mixture_holdout_folds(P, k=5, repeats=rb_, seed=0) if f.repeat >= ra]
        res = {}; keep = [list(ROWS)[int(i)] for i in a.rows.split(",")] if a.rows else list(ROWS)
        tag_r = f"_rows{a.rows.replace(',', '-')}" if a.rows else ""
        for nm, fp in models(BB, A, B).items():
            if nm not in keep:
                continue
            t0 = time.time(); pr = [np.asarray(fp(np.array(f.train), np.array(f.test), y), float) for f in folds]
            res[nm] = (ra, rb_, np.concatenate([np.array(f.test) for f in folds]), np.concatenate(pr))
            rep.line(f"done: {nm}; repeats {ra}-{rb_ - 1}; {time.time() - t0:.0f} s; non-finite {int((~np.isfinite(res[nm][3])).sum())}")
            pickle.dump({"chunk": res}, open(OUT / f"{stamp}_reps{ra}-{rb_}{tag_r}.pkl", "wb"))   # saved after every row
        rep.save(); return
    chunks = {}
    for f in a.merge:
        for nm, c in pickle.load(open(f, "rb"))["chunk"].items():
            chunks.setdefault(nm, []).append(c)
    res = {}
    for nm, cs in chunks.items():
        cs.sort(key=lambda c: c[0]); assert [c[0] for c in cs] == [0] + [c[1] for c in cs[:-1]], f"chunks not contiguous: {nm}"
        pair_N = np.concatenate([c[2] for c in cs])
        res[nm] = evaluate.CVResult(pair_N, np.concatenate([c[3] for c in cs]), y[pair_N], np.array([P[i].dataset for i in pair_N]),
                                    np.array([P[i].jaccard == 0 for i in pair_N]), 5, cs[-1][1], 0)
        rep.line(f"joined {nm}: repeats 0-{cs[-1][1] - 1}, {len(cs)} chunks")
    rows = []
    for nm in ROWS:
        if nm in res:
            s_ = evaluate.score_cv(P, res[nm], reference=None if nm == FULL else res[FULL])
            rows.append([nm, f"{s_['rmse']:.4f}", f"{s_['r']:.3f}", "" if nm == FULL else
                         f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.section("CV on development pairs (50 x 5): RMSE / r; change from full [95% CI]")
    rep.table(["model", "RMSE", "r", "r change from full [CI]"], rows); rep.save()


if __name__ == "__main__":
    main()
