# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Length-weighted merge, the face-recognition (MagFace / AdaFace) design: "how much a molecule counts" is the length
of its hidden vector, learned on single molecules; DREAM fits only an exponent. (User go-ahead 2026-11-20.)
C29: a 64-number query fitted on DREAM fits training pairs but not new pairs; C8: hidden length predicts who
dominates in Ma (AUROC 0.62, nothing fitted); C13: GoodScents disagrees.
    share_m in mixture K  ~  exp(tau * standardised log|v_m|) * exp(kappa * cos(v_m, mean of the other molecules in K))
    z(K) = sum share_m v_m;  d = angle distance |u_A - u_B|;  prediction = a d + b
Rows (fixed before running), frozen backbone (3 seeds averaged):
  O1.2b (reference, 27)
  <backbone>, attention + kappa (67)                     the current readout, penalty 1e-3
  <backbone>, plain mean + kappa (3)                     tau = 0 control
  <backbone>, length weight + kappa (4)                  tau fitted
log|v| standardised over the 235 molecules (SD 1), so tau is per SD and the penalty 1e-3 on tau^2 + kappa^2 does not
re-create C29's mis-scaling.
CV: development pairs 50 x 5, subsets all / setting 1 / setting 2 / share 1 / share 2 / share >= 3; change from O1.2b and
"length weight minus plain mean" on the same folds. --heldout: T46 then V50, fitted once on all training pairs.
Fitted tau (and kappa) on all development pairs printed per member.

Env:     mist
Usage:   python evaluation/O3/eval_len_weight.py [--backbone plain|counts] [--heldout]
Writes:  results/eval_len_weight/<date_time>.{txt,json}
"""
import argparse
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
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_attn_penalty import vectors  # noqa: E402
import eval_context_dream as ecd  # noqa: E402

F64 = torch.float64
LAM = 1e-3
REF = "O1.2b (reference, 27)"


class LengthReadout(ecd.ContextReadoutQ):
    """share ~ |v|^tau exp(kappa cos(v, others)); angle-only comparison; tau fixed 0 if not fit_tau."""

    def __init__(self, V, A, B, fit_tau):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=LAM); self.fit_tau = fit_tau
        self.logn = torch.log(self.V.norm(dim=1)); self.logn = (self.logn - self.logn.mean()) / self.logn.std()

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        tau = torch.zeros((), dtype=F64, requires_grad=self.fit_tau); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        b0 = torch.zeros((), dtype=F64)
        with torch.no_grad():
            self.lw = tau * self.logn; d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, kappa] + ([tau] if self.fit_tau else []), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = tau * self.logn; d = self.distance(kappa, b0, idx)
            loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + self.lam * (tau ** 2 + kappa ** 2); loss.backward(); return loss
        opt.step(closure)
        self.tau, self.kappa, self.b, self.lin = tau.detach(), kappa.detach(), b0, lin.detach()
        self.lw = (self.tau * self.logn).detach(); return self


def models(Vs, label, q_op, X, A, B):
    def ens(make):
        return lambda tr, te, y: np.mean([make(v).fit(tr, y[tr]).predict(te) for v in Vs], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            f"{label}, attention + kappa (67)": ens(lambda v: ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True, lam=LAM)),
            f"{label}, plain mean + kappa (3)": ens(lambda v: LengthReadout(v, A, B, False)),
            f"{label}, length weight + kappa (4)": ens(lambda v: LengthReadout(v, A, B, True))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="plain", choices=["plain", "counts"])
    ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    label = {"plain": "plain O3", "counts": "counts"}[a.backbone]
    rep = Report("eval_len_weight", {"backbone": a.backbone, "lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS,
                                     "device": dev}, tag=a.backbone + ("_heldout" if a.heldout else ""))
    Vs = vectors(a.backbone, dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    ln = [np.log(np.linalg.norm(v, axis=1)) for v in Vs]
    rep.line(f"log |v| over the 235 DREAM molecules, per member: SD {', '.join(f'{x.std():.3f}' for x in ln)}; "
             f"max/min |v| {', '.join(f'{np.exp(x.max() - x.min()):.1f}' for x in ln)}")
    k_pm, k_lw = f"{label}, plain mean + kappa (3)", f"{label}, length weight + kappa (4)"

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
        fits = [LengthReadout(v, A, B, True).fit(np.arange(len(P)), y) for v in Vs]
        rep.line("fitted on all development pairs: tau " + ", ".join(f"{m.tau.item():+.3f}" for m in fits)
                 + "; kappa " + ", ".join(f"{m.kappa.item():+.3f}" for m in fits)
                 + "; resulting max/min weight " + ", ".join(f"{np.exp(m.lw.max().item() - m.lw.min().item()):.1f}" for m in fits))
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
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[k_pm])) if nm == k_lw else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "length weight minus plain mean (same folds)"], rows)
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
