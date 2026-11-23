# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Size-gap blend inside (b) (user 2026-11-20 "put it inside and see"; finding 6 sec. 23 item 5).
Distance per pair = (1 - g) z(d_unit) + g z(d_centred), g = exp(-|log n_a - log n_b| / 0.3) (fixed, 0 new numbers):
same-size pairs use the distance after removing the DREAM molecules' shared direction (centring), pairs of different
sizes keep the plain unit-merge distance. z = standardise each distance with its mean and SD over the training pairs at
the start of the fit (equal molecule weights; labels not used). Both distances use (b)'s molecule weights and odd-one-out.
Rows (fixed before running): O1.2b | (b) | (b) + blend | (b) + size loss | (b) + blend + size loss.
Held-out (T46, V50; fit on all 820 training pairs) and, with --dev, direct scores on the 514 development pairs by
shared-molecule count from a 5-fold split (fit on 4/5, score 1/5; mixture holdout, one repeat).

Env:     mist
Usage:   python evaluation/O3/eval_blend_b.py [--dev]
Writes:  results/eval_blend_b/<date_time>_{heldout,dev}.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from eval_size_robust import SizeRobust, centred_units  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402

F64 = torch.float64; LAM = eo.LAM; C_GAP = 0.3; REF = "O1.2b (reference, 27)"


class Blend(SizeRobust):
    def __init__(self, V, H, mol, A, B, s_atoms, q0, size_loss):
        super().__init__(V, H, mol, A, B, s_atoms, q0, False, False, size_loss)
        self.Uu = self.U; self.Uc = torch.as_tensor(centred_units(np.asarray(V)), dtype=F64)
        na, nb = self.A.sum(1), self.B.sum(1); self.g = torch.exp(-torch.abs(torch.log(na) - torch.log(nb)) / C_GAP)
        self.stats = None

    def _d(self, U, kappa, b, idx):
        self.U = U; d = eo.PriorAtom.distance(self, kappa, b, idx); self.U = self.Uu; return d

    def distance(self, kappa, b, idx):
        du, dc = self._d(self.Uu, kappa, b, idx), self._d(self.Uc, kappa, b, idx)
        if self.stats is None:      # first call = start of the fit (equal weights, training pairs)
            self.stats = (du.mean().detach(), du.std().detach(), dc.mean().detach(), dc.std().detach())
        mu, su, mc, scc = self.stats; g = self.g[torch.as_tensor(np.asarray(idx))]
        return (1 - g) * (du - mu) / su + g * (dc - mc) / scc


def heldout(O5, rep, q_op, X, op):
    train = data.load_train()
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
        y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
        preds = {REF: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y_all[tr]).predict(te)}
        for nm, mk in rows(A, B).items():
            P_ = np.array([mk(m).fit(tr, y_all[tr]).predict(te) for m in O5]); ok = np.isfinite(P_).all(1); preds[nm] = P_[ok].mean(0)
            if (~ok).any():
                rep.line(f"{hs.name} {nm}: {int((~ok).sum())} members dropped (non-finite)")
        y = hs.y_P; ref = preds[REF]; tab = []
        rep.section(f"{hs.name}: {len(y)} pairs")
        for nm, p in preds.items():
            s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
            if nm != REF:
                cr, _ = paired(p, ref, y); c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
            tab.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", c])
        rep.table(["model", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [CI]"], tab)
        rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})


def rows(A, B):
    return {"(b) (69)": lambda m: SizeRobust(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], False, False, False),
            "(b) + blend (69)": lambda m: Blend(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], False),
            "(b) + size loss (69)": lambda m: SizeRobust(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], False, False, True),
            "(b) + blend + size loss (69)": lambda m: Blend(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], True)}


def dev(O5, rep, q_op, X, op):
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    folds = cv.mixture_holdout_folds(P, k=5, repeats=1, seed=0)
    preds = {REF: np.full(len(P), np.nan)}; preds.update({nm: np.full(len(P), np.nan) for nm in rows(A, B)})
    for f in folds:
        tr, te = np.asarray(f.train), np.asarray(f.test)
        preds[REF][te] = O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)
        for nm, mk in rows(A, B).items():
            P_ = np.array([mk(m).fit(tr, y[tr]).predict(te) for m in O5]); ok = np.isfinite(P_).all(1); preds[nm][te] = P_[ok].mean(0)
    k = np.array([len(set(p.a) & set(p.b)) for p in P]); g = np.abs(A.sum(1) - B.sum(1))
    SUB = {"all": k >= 0, "share 0": k == 0, "share 0 same size": (k == 0) & (g <= 1), "share 0 sizes differ": (k == 0) & (g >= 2),
           "share 1-2": (k >= 1) & (k <= 2), "share >= 3": k >= 3}
    sc = {nm: np.isfinite(p) for nm, p in preds.items()}
    rep.section("Development pairs, one 5-fold mixture-holdout split (pairs with both mixtures held out); RMSE / r")
    tab = []
    for nm, p in preds.items():
        tab.append([nm] + [f"{metrics.rmse(p[m & sc[nm]], y[m & sc[nm]]):.4f} / {metrics.pearson(p[m & sc[nm]], y[m & sc[nm]]):+.3f} "
                           f"(n {int((m & sc[nm]).sum())})" for m in SUB.values()])
    rep.table(["model"] + list(SUB), tab)


def main():
    ecd.DIRECTION_ONLY = True; d = pick_device(); is_dev = "--dev" in sys.argv
    rep = Report("eval_blend_b", {"lam": LAM, "c_gap": C_GAP, "device": d}, tag="dev" if is_dev else "heldout")
    O5 = eo.backbones(d, rep)[1]["o5"]; op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    (dev if is_dev else heldout)(O5, rep, q_op, X, op); rep.save()


if __name__ == "__main__":
    main()
