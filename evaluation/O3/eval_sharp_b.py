# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Item 2 of "do all of them" (user 2026-11-20): sharper molecule weights inside (b).
(b) = O5 (o3m_attpool, 10 members) + atom query beta q0 + delta, soft max over atoms, penalty 1e-3 on beta, kappa, delta.
Finding 6 sec. 21: doubling O5's own attention sharpness with 0 DREAM numbers helped a little (T46 0.467 -> 0.484).
In (b) beta starts at 0 (equal weights) and its penalty pulls it toward 0 (flatter than O5's own attention).
Rows (fixed before running), same 69 numbers each, only the beta penalty changes:
  (b) as packaged              penalty lam beta^2
  (b), beta free               no penalty on beta
  (b), beta toward 1           penalty lam (beta - 1)^2, start 1 (O5's own sharpness)
  (b), beta toward 2           penalty lam (beta - 2)^2, start 2 (twice as sharp)
Held-out only (T46 then V50). Also printed: fitted beta, temperature, spread (SD) of the log molecule weights.

Env:     mist
Usage:   python evaluation/O3/eval_sharp_b.py
Writes:  results/eval_sharp_b/<date_time>_heldout.{txt,json}
"""
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
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402

F64 = torch.float64; LAM = eo.LAM; REF = "O1.2b (reference, 27)"


class SharpPrior(eo.PriorAtom):
    """(b) with beta penalised toward `target` (None = no beta penalty); beta starts at `target` (0 if None)."""

    def __init__(self, *a, target=0.0):
        super().__init__(*a, "shrink"); self.target = target

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        beta = torch.tensor(float(self.target or 0.0), dtype=F64, requires_grad=True)
        kappa = torch.zeros((), dtype=F64, requires_grad=True); t = torch.zeros((), dtype=F64, requires_grad=True)
        b0 = torch.zeros((), dtype=F64); delta = torch.zeros(self.H.shape[1], dtype=F64, requires_grad=True)
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t); d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, beta, kappa, t, delta], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.smax(self.score(beta, delta), t)
            pb = 0.0 if self.target is None else (beta - self.target) ** 2
            loss = ((lin[0] * self.distance(kappa, b0, idx) + lin[1] - y) ** 2).mean() + self.lam * (pb + kappa ** 2 + (delta ** 2).sum())
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t).detach()
        self.beta, self.kappa, self.t, self.delta, self.b, self.lin = beta.detach(), kappa.detach(), t.detach(), delta.detach(), b0, lin.detach()
        return self


VARIANTS = {"(b) as packaged": 0.0, "(b), beta free": None, "(b), beta toward 1": 1.0, "(b), beta toward 2": 2.0}


def main():
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_sharp_b", {"lam": LAM, "device": dev}, tag="heldout")
    O5 = eo.backbones(dev, rep)[1]["o5"]
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]; q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    own_sd = np.mean([m["strength"].std() / m["strength"].std() for m in O5])   # O5's own log weights in the same units = 1
    train = data.load_train()
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
        y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
        preds = {REF: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y_all[tr]).predict(te)}; info = {}
        for nm, tg in VARIANTS.items():
            fits = [SharpPrior(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], target=tg).fit(tr, y_all[tr]) for m in O5]
            P_ = np.array([f.predict(te) for f in fits]); ok = np.isfinite(P_).all(1); preds[nm] = P_[ok].mean(0)
            info[nm] = (np.mean([f.beta.item() for f in fits]), np.mean([np.exp(f.t.item()) for f in fits]),
                        np.mean([f.lw.std().item() for f in fits]), int((~ok).sum()))
        y = hs.y_P; ref = preds[REF]; rows = []
        rep.section(f"{hs.name}: {len(y)} pairs; O5's own log molecule weights SD {own_sd:.2f} in these units")
        for nm, p in preds.items():
            s = metrics.score(p, y, ceiling=hs.ceiling); c = ""; b, tau, sd, bad = info.get(nm, (np.nan,) * 3 + (0,))
            if nm != REF:
                cr, _ = paired(p, ref, y); c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
            rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", c, f"{b:.2f}", f"{tau:.2f}", f"{sd:.2f}", bad])
        rep.table(["model", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [CI]", "beta", "temperature", "SD log weight", "dropped"], rows)
        rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
