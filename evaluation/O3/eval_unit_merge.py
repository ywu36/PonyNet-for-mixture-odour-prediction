# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Direction = flavor, magnitude = how much the molecules agree; neither can take over (user, 2026-11-20).
Today's readout merges RAW hidden vectors, so a molecule's arbitrary (untrained) length leaks into the mixture's
direction and length; the hidden length term then did nothing (0.438 vs 0.438). Here every molecule is a unit direction:
    u_m = v_m / |v_m|;  shares w (attention q . s_m + kappa * cos(v_m, others), as before)
    z(K) = sum w_m u_m          |z| in (0, 1]: 1 = all molecules point the same way, small = "white" (bounded)
    d = sqrt( |z_A/|z_A| - z_B/|z_B||^2 + beta * ((log|z_A| - log|z_B|) / s)^2 ),  s = SD of log|z| over the fold's
        training mixtures at q = kappa = 0 (fixed scale, so the two terms start comparable); beta = b^2 learned
    prediction = a d + c
Rows (fixed before running), plain O3 w64 (o3m_w64), 3 seeds averaged:
  O1.2b (reference, 27)
  raw merge, angle only (67)          the current readout
  unit merge, angle only (67)         isolates "no molecule length in the merge"
  unit merge, angle + agreement (68)  + beta
Penalty 1e-3 on q and kappa (as before); beta unpenalised (b starts at 0.1).
CV: development pairs 50 x 5, six subsets, change from O1.2b, and "+ agreement minus unit angle only", "unit angle only
minus raw" on the same folds. --heldout: T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_unit_merge.py [--heldout]
Writes:  results/eval_unit_merge/<date_time>.{txt,json}
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
K_RAW, K_UNIT, K_AGREE = "raw merge, angle only (67)", "unit merge, angle only (67)", "unit merge, angle + agreement (68)"


class UnitMerge(ecd.ContextReadoutQ):
    """Attention + kappa shares over unit directions; optional agreement (log |z|) term."""

    def __init__(self, V, A, B, agree):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=LAM); self.agree = agree
        self.U = self.V / self.V.norm(dim=1, keepdim=True)

    def _out(self, kappa, Mx, C, idx):
        lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
        W = Mx[idx] * torch.exp(lw); z = (W @ self.U) / W.sum(1, keepdim=True); n = z.norm(dim=1, keepdim=True)
        return z / n, torch.log(n[:, 0])

    def dist(self, kappa, b, idx):
        uA, hA = self._out(kappa, self.A, self.CA, idx); uB, hB = self._out(kappa, self.B, self.CB, idx)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + (b ** 2 * ((hA - hB) / self.s) ** 2 if self.agree else 0) + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        b = torch.tensor(0.1, dtype=F64, requires_grad=self.agree)
        with torch.no_grad():
            self.lw = self.V @ q
            _, hA = self._out(kappa, self.A, self.CA, idx); _, hB = self._out(kappa, self.B, self.CB, idx)
            self.s = torch.cat([hA, hB]).std().clamp_min(1e-6)
            d = self.dist(kappa, b, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, q, kappa] + ([b] if self.agree else []), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.V @ q
            loss = ((lin[0] * self.dist(kappa, b, idx) + lin[1] - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = (self.V @ q).detach()
        self.q, self.kappa, self.bb, self.lin = q.detach(), kappa.detach(), b.detach(), lin.detach(); return self

    def predict(self, idx):
        idx_t = torch.as_tensor(np.asarray(idx))
        with torch.no_grad():
            return (self.lin[0] * self.dist(self.kappa, self.bb, idx_t) + self.lin[1]).numpy()


def models(Vs, q_op, X, A, B):
    def ens(make):
        return lambda tr, te, y: np.mean([make(v).fit(tr, y[tr]).predict(te) for v in Vs], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            K_RAW: ens(lambda v: ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True, lam=LAM)),
            K_UNIT: ens(lambda v: UnitMerge(v, A, B, False)),
            K_AGREE: ens(lambda v: UnitMerge(v, A, B, True))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_unit_merge", {"lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    Vs = vectors("plain", dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
        fits = [UnitMerge(v, A, B, True).fit(np.arange(len(P)), y) for v in Vs]
        rep.line("agreement model fitted on all development pairs: beta " + ", ".join(f"{m.bb.item() ** 2:.3f}" for m in fits)
                 + "; kappa " + ", ".join(f"{m.kappa.item():+.3f}" for m in fits))
        nA, nB = A.sum(1), B.sum(1)
        for m in fits[:1]:
            with torch.no_grad():
                _, hA = m._out(m.kappa, m.A, m.CA, torch.arange(len(P)))
            rep.line(f"  member 0: |z| of mixtures: single molecules 1.000; corr(log|z|, log n) {np.corrcoef(hA.numpy(), np.log(nA))[0, 1]:+.3f}")
        res = {}
        for nm, fp in models(Vs, q_op, X, A, B).items():
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
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[K_RAW])) if nm == K_UNIT else "",
                             ch(sc(r_, sub, res[K_UNIT])) if nm == K_AGREE else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "unit minus raw", "+ agreement minus unit"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(Vs, q_op, X, A, B).items()}
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
