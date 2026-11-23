# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Receptor-level competition (user 2026-11-20: "each hidden unit mixes its molecules by a soft max instead of an
average, so the strongest response in each unit wins partly ... quick test"). First principle: hidden units = receptors;
mixing bends the interpolation. No earlier hidden-layer test (clues searched; finding 3's per-word dominance was in
open-POM word space).
Backbone: O5 (checkpoints/o3m_attpool), 10 members, frozen, GS-LF singles only; molecule vectors u_m = v_m / |v_m|.
Mixture, per hidden unit k:  z_k = sum_m w_m e_mk u_mk / sum_m w_m e_mk,  e_mk = exp(gamma |u_mk|)
  (gamma = 0: the usual unit merge; gamma > 0: the molecule with the strongest response in unit k dominates unit k;
  gamma < 0: strong responses suppressed). w_m = molecule weight (O5's own attention strength, + odd-one-out).
Then z / |z|, Euclidean distance, line. One shared gamma for all units: 1 DREAM number.
Rows (fixed before running), 10 members averaged:
  O1.2b (reference, 27)
  O5, own attention weights fixed, line (2)                | + per-unit competition gamma (3)
  O5, own attention x scale + odd-one-out, line (4) [C36]   | + per-unit competition gamma (5)
CV: development pairs 50 x 5, six subsets; change from O1.2b and from the same row without gamma (same folds;
predictions saved). --heldout: T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_unit_competition.py [--heldout]
Writes:  results/eval_unit_competition/<date_time>.{txt,json} (+ _cv.pkl)
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
from mixdist import data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_attpool import LearnedStrength  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402

F64 = torch.float64
LAM = 1e-3
REF = "O1.2b (reference, 27)"
R2, R3, R4, R5 = ("O5, own attention weights fixed (2)", "O5, own attention weights fixed + per-unit competition (3)",
                  "O5, own attention x scale + odd-one-out (4)", "O5, own attention x scale + odd-one-out + per-unit competition (5)")
BASE = {R3: R2, R5: R4}


class UnitCompetition(LearnedStrength):
    """Unit merge with per-hidden-unit competition exp(gamma |u_mk|); molecule weights fixed (free=False: raw O5
    strength, kappa 0) or scale x standardized strength + odd-one-out (free=True)."""

    def __init__(self, V, strength, A, B, free, use_gamma):
        super().__init__(V, strength, A, B, True); self.free, self.use_gamma = free, use_gamma
        if not free:
            self.S = torch.as_tensor(np.asarray(strength, float) - np.max(strength), dtype=F64)   # raw strength, c = 1
        self.absU = self.U.abs()

    def dist(self, kappa, idx, gamma=None):
        g = self.gamma if gamma is None else gamma; E = torch.exp(g * self.absU); EU = E * self.U

        def out(Mx, C):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(lw); z = (W @ EU) / (W @ E); return z / z.norm(dim=1, keepdim=True)
        return torch.sqrt(((out(self.A, self.CA) - out(self.B, self.CB)) ** 2).sum(1) + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        c = torch.tensor(0.0 if self.free else 1.0, dtype=F64, requires_grad=self.free)
        kappa = torch.zeros((), dtype=F64, requires_grad=self.free)
        gamma = torch.zeros((), dtype=F64, requires_grad=self.use_gamma)
        with torch.no_grad():
            self.lw = c * self.S; d = self.dist(kappa, idx, gamma)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        ps = [lin] + ([c, kappa] if self.free else []) + ([gamma] if self.use_gamma else [])
        opt = torch.optim.LBFGS(ps, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = c * self.S
            loss = ((lin[0] * self.dist(kappa, idx, gamma) + lin[1] - y) ** 2).mean() + self.lam * (kappa ** 2 + (c ** 2 if self.free else 0))
            loss.backward(); return loss
        opt.step(closure)
        self.c, self.kappa, self.gamma, self.lin = c.detach(), kappa.detach(), gamma.detach(), lin.detach()
        with torch.no_grad():
            self.lw = self.c * self.S
        return self

    def predict(self, idx):
        with torch.no_grad():
            return (self.lin[0] * self.dist(self.kappa, torch.as_tensor(np.asarray(idx))) + self.lin[1]).numpy()


def models(BB, q_op, X, A, B):
    def ens(free, g):
        return lambda tr, te, y: np.mean([UnitCompetition(m["V"], m["strength"], A, B, free, g).fit(tr, y[tr]).predict(te) for m in BB], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            R2: ens(False, False), R3: ens(False, True), R4: ens(True, False), R5: ens(True, True)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_unit_competition", {"lam": LAM, "members": len(eo.SEEDS), "heldout": a.heldout, "repeats": evaluate.CV_REPEATS,
                                           "device": dev}, tag="heldout" if a.heldout else "")
    _, BBall = eo.backbones(dev, rep); BB = BBall["o5"]; op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "same-size share none": (k_P == 0) & (gap <= 1),
               "setting 2 (share >= 1)": k_P >= 1, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        for nm, fr in ((R3, False), (R5, True)):
            fs = [UnitCompetition(m["V"], m["strength"], A, B, fr, True).fit(np.arange(len(P)), y) for m in BB]
            rep.line(f"{nm}, fitted on all development pairs: gamma " + ", ".join(f"{m.gamma.item():+.2f}" for m in fs))
        res = {}
        for nm, fp in models(BB, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y))
            rep.line(f"done: {nm}; non-finite predictions {int((~np.isfinite(res[nm].pred_N)).sum())}")
        pk = ROOT / "results/eval_unit_competition" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv.pkl"
        pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"CV predictions saved: {pk.name}")

        def sc(r_, sub, against):
            m = SUB[sub]; m = np.isfinite(r_.pred_N) if m is None else (m[r_.pair_N] & np.isfinite(r_.pred_N))
            if against is not None:
                m = m & np.isfinite(against.pred_N)
            return evaluate.score_cv(P, r_, reference=against, mask_N=m)
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[BASE[nm]])) if nm in BASE else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from same row without competition"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            fs = [UnitCompetition(m["V"], m["strength"], A, B, True, True).fit(tr, y_all[tr]) for m in BB]
            rep.line(f"{hs.name}: gamma (5-number row) " + ", ".join(f"{m.gamma.item():+.2f}" for m in fs))
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = c2 = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                if nm in BASE:
                    b = preds[BASE[nm]]; cr, _ = paired(p, b, y)
                    c2 = f"{s['rmse'] - metrics.rmse(b, y):+.4f} / {s['r'] - metrics.pearson(b, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c, c2])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]", "change from same row without competition"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
