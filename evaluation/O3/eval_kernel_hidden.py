# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Learned-width set kernel in the hidden layer: the network decides how sharp "the same molecule" is, without being
told the overlap (user, 2026-11-20: "both"; the trade-off "telling molecules apart vs keeping similar smells close").
Finding 5l tried this on O1.2b's word profiles (width collapsed to identity, no gain); new here: plain O3 hidden space,
on top of the attention merge.
    shares w (attention + kappa, as before);  u = v / |v|;  k(a, b) = exp(-|u_a - u_b|^2 / (2 sigma^2))  (k(a, a) = 1)
    M(A, B) = sum_ab w_a w_b k(a, b) / sqrt(K(A, A) K(B, B))      (0..1; sigma -> 0 gives the share-weighted overlap)
    prediction = a * angle distance(z_A, z_B) + c * M(A, B) + b
Numbers: q (64) + kappa + sigma + a, b, c = 69. sigma starts at 0.5 x the median distance between different DREAM
molecules; penalty 1e-3 on q and kappa only (as before).
Rows (fixed before running), plain O3 w64 (o3m_w64), 3 seeds averaged:
  O1.2b (reference, 27) | attention + kappa (67) | attention + kappa + overlap J (68; explicit, as eval_overlap.py)
  | attention + kappa + learned-width kernel (69)
CV: development pairs 50 x 5, subsets all / setting 1 / setting 2 / share 1 / share 2 / share >= 3; change from O1.2b;
kernel minus attention and kernel minus overlap (same folds). Fitted sigma (on all development pairs) / median distance.
--heldout: T46 then V50 (the kernel term is not 0 there: similar molecules partly match).

Env:     mist
Usage:   python evaluation/O3/eval_kernel_hidden.py [--heldout]
Writes:  results/eval_kernel_hidden/<date_time>.{txt,json}
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
from eval_overlap import AttnOverlap, jaccard  # noqa: E402
import eval_context_dream as ecd  # noqa: E402

F64 = torch.float64
LAM = 1e-3
REF = "O1.2b (reference, 27)"


class AttnKernel(ecd.ContextReadoutQ):
    """Attention + kappa angle distance plus a learned-width set-kernel similarity term."""

    def __init__(self, V, A, B):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=LAM)
        U = self.V / self.V.norm(dim=1, keepdim=True); self.D2 = ((U[:, None] - U[None]) ** 2).sum(2)
        off = self.D2[~torch.eye(len(U), dtype=bool)]; self.med = off.sqrt().median().item()

    def _shares(self, kappa, Mx, C, idx):
        lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
        W = Mx[idx] * torch.exp(lw); return W / W.sum(1, keepdim=True)

    def kterm(self, kappa, ls, idx):
        K = torch.exp(-self.D2 / (2 * torch.exp(2 * ls)))
        WA, WB = self._shares(kappa, self.A, self.CA, idx), self._shares(kappa, self.B, self.CB, idx)
        kab = ((WA @ K) * WB).sum(1); kaa = ((WA @ K) * WA).sum(1); kbb = ((WB @ K) * WB).sum(1)
        return kab / torch.sqrt(kaa * kbb)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        ls = torch.tensor(np.log(0.5 * self.med), dtype=F64, requires_grad=True); b0 = torch.zeros((), dtype=F64)
        with torch.no_grad():
            self.lw = self.V @ q; d = self.distance(kappa, b0, idx); m = self.kterm(kappa, ls, idx)
            lin = torch.linalg.lstsq(torch.stack([d, m, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True); opt = torch.optim.LBFGS([lin, q, kappa, ls], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.V @ q
            pred = lin[0] * self.distance(kappa, b0, idx) + lin[1] * self.kterm(kappa, ls, idx) + lin[2]
            loss = ((pred - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = (self.V @ q).detach()
        self.q, self.kappa, self.ls, self.b, self.lin = q.detach(), kappa.detach(), ls.detach(), b0, lin.detach()
        return self

    @property
    def sigma_rel(self):
        return float(torch.exp(self.ls)) / self.med

    def predict(self, idx):
        idx_t = torch.as_tensor(np.asarray(idx))
        with torch.no_grad():
            return (self.lin[0] * self.distance(self.kappa, self.b, idx_t) + self.lin[1] * self.kterm(self.kappa, self.ls, idx_t)
                    + self.lin[2]).numpy()


def models(Vs, q_op, X, A, B, J):
    def ens(make):
        return lambda tr, te, y: np.mean([make(v).fit(tr, y[tr]).predict(te) for v in Vs], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            "plain O3, attention + kappa (67)": ens(lambda v: AttnOverlap(v, A, B, J, False)),
            "plain O3, attention + kappa + overlap (68)": ens(lambda v: AttnOverlap(v, A, B, J, True)),
            "plain O3, attention + kappa + learned-width kernel (69)": ens(lambda v: AttnKernel(v, A, B))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_kernel_hidden", {"lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    Vs = vectors("plain", dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    K_A, K_O, K_K = "plain O3, attention + kappa (67)", "plain O3, attention + kappa + overlap (68)", "plain O3, attention + kappa + learned-width kernel (69)"

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M); J = jaccard(P)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
        fits = [AttnKernel(v, A, B).fit(np.arange(len(P)), y) for v in Vs]
        rep.line("fitted on all development pairs: sigma / median distance between different molecules "
                 + ", ".join(f"{m.sigma_rel:.3f}" for m in fits) + "; kernel weight c " + ", ".join(f"{m.lin[1].item():+.3f}" for m in fits)
                 + "; kappa " + ", ".join(f"{m.kappa.item():+.3f}" for m in fits))
        for m in fits:   # how often does a pair of DIFFERENT molecules get a kernel value > 0.5 at the fitted width?
            K = torch.exp(-m.D2 / (2 * torch.exp(2 * m.ls))); off = K[~torch.eye(len(K), dtype=bool)]
            rep.line(f"  member: different-molecule kernel values: mean {off.mean().item():.3f}, share > 0.5 {(off > 0.5).double().mean().item():.4f}")
        res = {}
        for nm, fp in models(Vs, q_op, X, A, B, J).items():
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
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[K_A])) if nm == K_K else "",
                             ch(sc(r_, sub, res[K_O])) if nm == K_K else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "kernel minus attention", "kernel minus overlap"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M); J = jaccard(allp)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(Vs, q_op, X, A, B, J).items()}
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
