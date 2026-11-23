# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   T = atoms
# ---------------------------------------------------------------------------------------------
"""Make the size loss fit the reported model (b) (user 2026-11-21: "make it more compatible ... did you tune the balance
of the losses and did you use unified length for the comparison?").
Base (b) = O5 (o3m_attpool, 10 members, frozen) + unit merge (each molecule vector length 1, mixture vector rescaled to
length 1 before the distance) + atom soft max with the DREAM query shrunk toward O5's own scorer + odd-one-out + line
(69 numbers; eval_o5_softmax.PriorAtom). s = mean log mixture size of the pair (test-available).
Loss forms (penalty w x var(y) x corr(.,.)^2 added to the squared error of the DREAM fit):
  dist   corr(distance, s) on all training pairs       (the other session's form; w 1 = its "(b) + size loss")
  same   corr(distance, s) on same-size training pairs only (|na - nb| <= 1), where the label barely follows size
         (training r -0.00 on same-size share-none pairs), so pairs whose sizes differ keep their real size cue
  resid  corr(error, s) on all training pairs: error = prediction - label; the prediction may follow size only as
         much as the label does
  line   no penalty; one extra number: rating = a x distance + c x (s - mean s) + b   (70 numbers)
Weights w in 0.1, 0.3, 1, 3, 10 (fixed before running); resid also 30, 100 (added after a 1-member held-out smoke test
showed resid leaves r unchanged and its effect still growing at 10; the CV screen, not held-out, picks the weight). Chosen by CV on the 514 development pairs (10 x 5 screen,
all-pairs r, then RMSE); every row is still scored held-out (T46 then V50) so the choice is visible.
Modes: --heldout | --rows i,j (CV part, pickled) | --merge <pkls> (score CV parts).

Env:     mist
Usage:   python evaluation/O3/eval_size_tune.py [--heldout | --rows i,j [--repeats 10] | --merge pkl ...]
Writes:  results/eval_size_tune/<date_time>.{txt,json} (+ _cv_rows<..>.pkl, _heldout_pred.pkl)
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
from mixdist import cv, data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402

F64 = torch.float64
LAM = 1e-3
GS_INIT, GS_FREE = 0.0, True   # self-attention length: start value and whether the sharpness is fitted
WEIGHTS = (0.1, 0.3, 1.0, 3.0, 10.0)
REF, RB = "O1.2b (reference, 27)", "(b) (69)"
VARIANTS = {RB: ("none", 0.0)}
for form in ("dist", "same", "resid"):
    for w in WEIGHTS + ((30.0, 100.0) if form == "resid" else ()):
        VARIANTS[f"(b) + {form} w {w:g}"] = (form, w)
VARIANTS["(b) + size in line (70)"] = ("line", 0.0)
# step 2 (user 2026-11-21 "ok do it"): molecule size m = mean heavy atoms per molecule, averaged over the two mixtures
# (label r -0.31 on development pairs; O1.2b follows it, our models do not). Rows fixed before running.
for w in (1.0, 10.0, 100.0):
    VARIANTS[f"(b) + mol-error w {w:g}"] = ("mresid", w)
VARIANTS["(b) + molecule size in line (70)"] = ("mline", 0.0)
VARIANTS["(b) + mol-error w 10 + error w 10"] = ("both", 10.0)
HEAVY_M = None  # heavy atoms per DREAM molecule, set in main()
# step 3 (user 2026-11-21 "go deep with size loss w 0.3 ... anything we can further improve"). Rows fixed before running.
#   sharp    one extra number g (70): a mixture of n molecules uses weights exp((n / 10)^g x log-weight), so bigger
#            mixtures can lean on fewer molecules. (b) has no size-aware number; the penalty could only bend the atom query.
#   partial  penalty on the partial correlation of distance and mixture size given the label: only the size dependence
#            the label does not have is penalised (target r(distance, size) = r(distance, label) x r(label, size), not 0).
VARIANTS["(b) + sharp (70)"] = ("sharp", 0.0)
for w in (0.3, 1.0, 3.0):
    VARIANTS[f"(b) + sharp + dist w {w:g} (70)"] = ("sharp+dist", w)
for w in (1.0, 10.0):
    VARIANTS[f"(b) + partial w {w:g}"] = ("partial", w)
VARIANTS["(b) + sharp + partial w 10 (70)"] = ("sharp+partial", 10.0)
# step 4 (user 2026-11-21 "worth a try ... CV which is better? weight 1?"): confirm the CV-screen picks with 10 members.
#   mol-error w 3 fills the gap between 1 (CV pick) and 10 (T46 gain); combo = the two picks together
#   (distance vs mixture size w 0.3 + error vs molecule size w 1). Rows fixed before running.
VARIANTS["(b) + mol-error w 3"] = ("mresid", 3.0)
VARIANTS["(b) + dist w 0.3 + mol-error w 1"] = ("combo", 1.0)
#   msame = error vs molecule size among same-size pairs only (|na - nb| <= 1). Data look (development pairs, no fit):
#   label r with molecule size -0.31 same-size share none, -0.30 share 1-2, -0.16 share >= 3, but +0.17 where sizes differ
#   (molecule size and mixture size correlate -0.44), so one penalty over all pairs mixes two opposite relations.
for w in (1.0, 10.0):
    VARIANTS[f"(b) + mol-error same-size w {w:g}"] = ("msame", w)
# step 5 (user 2026-11-21 "same size still has a serious problem ... design a loss right to the point"). Diagnosis
# (scratch samesize_insample.py): fitted on all development pairs, (b) reaches only r 0.14 in-sample on the 81 same-size
# share-none pairs (O1.2b 0.28) because they are 16% of the pairs and carry little of the squared error; weighted x5 it
# reaches 0.26, x20 0.40. So: pair weights. Same-size share-none pairs (the kind T46 / V50 are) count w times in the
# squared error. The share-none flag weights training pairs only; it is never a model input. Rows fixed before running.
for w in (3.0, 5.0, 10.0, 20.0):
    VARIANTS[f"(b) + same-size share-none pairs x{w:g}"] = ("wsame", w)
VARIANTS["(b) + both losses + same-size share-none pairs x5"] = ("wcombo", 5.0)
# step 6: atom-count strength (70). (b) averages the atom scores of a molecule (log mean exp), so a molecule's weight
# cannot depend on how many atoms it has; O1.2b's group weights grow with molecule size (r +0.32). One number eta:
# molecule log-weight += eta x (log atoms - mean log atoms); eta 0 = mean over atoms (b), eta x temperature = sum over atoms.
VARIANTS["(b) + atom-count strength (70)"] = ("asum", 0.0)
VARIANTS["(b) + atom-count strength + both losses (70)"] = ("asum+combo", 1.0)
# step 7 (user 2026-11-21 "0.01 is also good, how to strengthen it further"). Standard CV: both losses +0.011, but
# sizes-differ pairs lose (-0.022) because there molecule size stands in for mixture size (label r +0.17, opposite sign).
# Data look (development labels, no fit): after removing mixture size and size gap from both, the label r with molecule
# size is -0.20 same-size share none, -0.25 share 1-2, -0.15 share >= 3, +0.02 sizes differ (raw: -0.31, -0.30, -0.16, +0.17).
# pcombo = both losses, with molecule size first cleaned of mixture size and size gap (least squares on the training pairs).
# Rows fixed before running: w 1 (same weight as before) and w 3; judged by the standard 50 x 5 CV only.
for w in (1.0, 3.0):
    VARIANTS[f"(b) + dist w 0.3 + clean mol-error w {w:g}"] = ("pcombo", w)
# step 8 (user 2026-11-21 "keep pushing the boundary and see how to improve and why"). Check (scratch hidden_size.py, no
# labels): O5's unit molecule vectors already carry molecule size (held-out linear R^2 of log heavy atoms 0.81), but the
# angle distance does not use it (molecule-pair cosine r with mean size -0.04). Unit merge throws the length away, so a
# pair-level effect ("mixtures of big molecules are more alike") has nowhere to live. Hidden length: mixture vector =
# g x unit direction, distance = |g_A z_A - g_B z_B|; g = 1 gives (b). No hand-made feature: g is read from the network.
#   hl1   g = exp(rho x weighted mean of log |V_m|), |V_m| = the network's own vector length; 1 number (70)
#   hl64  g = exp(c . z), c = 64 numbers with the same shrinkage as the atom-query correction (133)
# Each alone and with both losses (dist w 0.3 + mol-error w 1). Rows fixed before running; standard 50 x 5 CV only.
VARIANTS["(b) + hidden length, own (70)"] = ("hl1+plain", 0.0)
VARIANTS["(b) + hidden length, own + both losses (70)"] = ("hl1+combo", 1.0)
VARIANTS["(b) + hidden length, learned (133)"] = ("hl64+plain", 0.0)
VARIANTS["(b) + hidden length, learned + both losses (133)"] = ("hl64+combo", 1.0)
# step 9 (user 2026-11-21 "find the previous best solution for each problem and integrate"). Problem: learned length
# costs share-none pairs whose sizes differ (0.559 -> 0.468). Earlier solutions for that same trade: C50 (do not remove
# the size cue, the bias is in how unit merge averages) and step 7 (cleaned molecule-size loss halves the cost).
#   hm64  molecule-level length: g = exp(weighted mean over molecules of c . u_m). hl64 reads c . z with z = mean / |mean|,
#         and |mean| shrinks as mixtures grow, so hl64's length also moves with mixture size; hm64's cannot.
# Rows fixed before running; standard 50 x 5 CV only.
VARIANTS["(b) + hidden length, learned + clean losses (133)"] = ("hl64+pcombo", 1.0)
VARIANTS["(b) + molecule-level length + both losses (133)"] = ("hm64+combo", 1.0)
VARIANTS["(b) + molecule-level length + clean losses (133)"] = ("hm64+pcombo", 1.0)
# step 10 (user 2026-11-21 "no more possible way to improve it?"). Step 9 showed one trade: mixture-level length reads
# c . mean / |mean| (fixes same-size and share 1-2, costs sizes-differ), molecule-level reads c . mean (the reverse).
#   ha64   learned mix: c . mean / |mean|^alpha, one extra number alpha (starts at 1 = mixture-level; 0 = molecule-level) (134)
#   hl64 x10 / x0.1   same model as the lead with the shrinkage on the 64 length numbers 10 times stronger / weaker
# Rows fixed before running; standard 50 x 5 CV only.
VARIANTS["(b) + length, learned spread share + both losses (134)"] = ("ha64+combo", 1.0)
VARIANTS["(b) + hidden length, shrink x10 + both losses (133)"] = ("hl64s+combo", 1.0)
VARIANTS["(b) + hidden length, shrink x0.1 + both losses (133)"] = ("hl64w+combo", 1.0)
# ablation of the current best (user 2026-11-21 "do an ablation study"): 2 x 2 x 2 over learned length, mixture-size
# loss (w 0.3), molecule-size loss (w 1). Already in the standard CV: none = (b), both losses, length, length + both.
# Missing cells: each loss alone = rows "(b) + dist w 0.3" and "(b) + mol-error w 1" (run before only at 10 x 5), and
# length + each loss alone (new rows below).
VARIANTS["(b) + hidden length, learned + dist w 0.3 (133)"] = ("hl64+dist", 0.3)
VARIANTS["(b) + hidden length, learned + mol-error w 1 (133)"] = ("hl64+mresid", 1.0)
# Step 11 (2026-11-22): length by self-attention among the molecules of a mixture. Each molecule m reads its own context
#   h_m = sum_n w_n exp(10 g (u_m . u_n - 1)) u_n (n over the mixture), log length of the mixture = sum_m w_m c . h_m / |h_m|.
#   g = 0 is O6 exactly (every molecule sees the whole mixture); large g = each molecule sees only itself = molecule-level
#   length. One extra number g, shrunk like the others (134).
VARIANTS["(b) + self-attention length + dist w 0.3 (134)"] = ("hs64+dist", 0.3)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean(); return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


class SizeTune(eo.PriorAtom):
    def __init__(self, V, H, mol, A, B, s_atoms, q0, form, w):
        super().__init__(V, H, mol, A, B, s_atoms, q0, "shrink"); self.hl = 0
        self.hmol = form.startswith("hm"); self.halpha = form.startswith("ha"); self.clam = 1.0
        self.hself = form.startswith("hs"); self.gs = torch.zeros((), dtype=F64); self._tc = {}
        if form.startswith("hl") or self.hmol or self.halpha or self.hself:
            tag = form[2:form.index("+")]; self.clam = 10.0 if tag.endswith("s") else 0.1 if tag.endswith("w") else 1.0
            self.hl = int(tag.rstrip("sw")); form = form[form.index("+") + 1:]
            lv = torch.log(self.V.norm(dim=1)); self.logV = lv - lv.mean()
        self.rho = torch.zeros((), dtype=F64); self.cvec = torch.zeros(self.V.shape[1], dtype=F64); self.alpha = torch.ones((), dtype=F64)
        self.sharp = form.startswith("sharp"); self.asum = form.startswith("asum")
        form = (form[5:] or "plain") if self.asum else form
        self.form, self.w = (form[6:] or "plain") if self.sharp else form, w; self.gamma = torch.zeros((), dtype=F64)
        ln = torch.log(self.n.double()); self.logn_c = ln - ln.mean()
        na = self.A.sum(1).double(); nb = self.B.sum(1).double()
        self.S = (torch.log(na) + torch.log(nb)) / 2; self.same = (na - nb).abs() <= 1
        self.La, self.Lb = torch.log(na / 10), torch.log(nb / 10); self.gap = (torch.log(na) - torch.log(nb)).abs()
        self.same0 = self.same & ((self.A * self.B).sum(1) == 0)   # same size and nothing shared: used only to weight the fit
        if form in ("mresid", "mline", "both", "combo", "msame", "wcombo", "pcombo"):
            h = torch.as_tensor(HEAVY_M, dtype=F64); self.Hm = ((self.A.double() @ h) / na + (self.B.double() @ h) / nb) / 2

    def _trip(self, Mx, idx):   # all (molecule m, molecule n) pairs inside each mixture, cached per set of rows
        ii = torch.as_tensor(np.asarray(idx)); key = (id(Mx), hash(ii.numpy().tobytes()))
        if key not in self._tc:
            Mb = Mx[ii] > 0; pe, me = Mb.nonzero(as_tuple=True); cnt = Mb.sum(1); start = torch.cumsum(cnt, 0) - cnt
            r = cnt[pe]; et = torch.repeat_interleave(torch.arange(len(pe)), r)
            off = torch.arange(len(et)) - torch.repeat_interleave(torch.cumsum(r, 0) - r, r); nt = me[start[pe[et]] + off]
            self._tc[key] = (pe, me, et, nt, (self.U[me[et]] * self.U[nt]).sum(1) - 1)
        return self._tc[key]

    def distance(self, kappa, b, idx):
        if self.hl:   # hidden length: unit direction times a length read from the hidden layer
            def outl(Mx, C):
                lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
                W = Mx[idx] * torch.exp(lw); Wn = W / W.sum(1, keepdim=True); zm = Wn @ self.U; zn = zm.norm(dim=1, keepdim=True); z = zm / zn
                arg = zm if self.hmol else (zm / zn ** self.alpha if self.halpha else z)
                if self.hself:
                    pe, me, et, nt, sim = self._trip(Mx, idx); a = Wn[pe[et], nt] * torch.exp(10 * self.gs * sim)
                    h = torch.zeros(len(pe), self.U.shape[1], dtype=F64).index_add(0, et, a[:, None] * self.U[nt])
                    le = (h / h.norm(dim=1, keepdim=True)) @ self.cvec
                    return torch.exp(torch.zeros(len(Wn), dtype=F64).index_add(0, pe, Wn[pe, me] * le))[:, None] * z
                g = torch.exp(self.rho * (Wn @ self.logV)) if self.hl == 1 else torch.exp(arg @ self.cvec)
                return g[:, None] * z
            return torch.sqrt(((outl(self.A, self.CA) - outl(self.B, self.CB)) ** 2).sum(1) + 1e-12)
        if not self.sharp:
            return super().distance(kappa, b, idx)

        def out(Mx, C, L):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - (lw + (Mx[idx] - 1) * 1e30).max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(torch.exp(self.gamma * L[idx])[:, None] * lw.clamp_max(0)); z = (W @ self.U) / W.sum(1, keepdim=True)
            return z / z.norm(dim=1, keepdim=True)
        return torch.sqrt(((out(self.A, self.CA, self.La) - out(self.B, self.CB, self.Lb)) ** 2).sum(1) + 1e-12)

    def fit(self, idx, y, max_iter=300):
        if self.form == "none":
            return super().fit(idx, y, max_iter)
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float)); s = self.S[idx]
        self.s_mean = s.mean(); sc = s - self.s_mean; same = self.same[idx]; vy = y.var()
        if self.form in ("mresid", "mline", "both", "combo", "msame", "wcombo", "pcombo"):
            hm = self.Hm[idx]; self.h_mean = hm.mean(); hc = hm - self.h_mean
        if self.form == "pcombo":   # molecule size cleaned of mixture size and size gap
            Xs = torch.stack([torch.ones_like(s), s, self.gap[idx]], 1); hm = hm - Xs @ torch.linalg.lstsq(Xs, hm[:, None]).solution[:, 0]
        if self.form == "mline":
            sc = hc
        beta = torch.zeros((), dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        t = torch.zeros((), dtype=F64, requires_grad=True); b0 = torch.zeros((), dtype=F64)
        delta = torch.zeros(self.H.shape[1], dtype=F64, requires_grad=True)
        gamma = torch.zeros((), dtype=F64, requires_grad=self.sharp); self.gamma = gamma; r_sy = _corr(s, y)
        eta = torch.zeros((), dtype=F64, requires_grad=self.asum)
        rho = torch.zeros((), dtype=F64, requires_grad=self.hl == 1); cvec = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=self.hl == 64)
        alpha = torch.ones((), dtype=F64, requires_grad=self.halpha); self.rho, self.cvec, self.alpha = rho, cvec, alpha
        gs = torch.full((), GS_INIT, dtype=F64, requires_grad=self.hself and GS_FREE); self.gs = gs
        with torch.no_grad():
            self.lw = self.smax(self.score(beta, delta), t); d = self.distance(kappa, b0, idx)
            cols = [d, torch.ones_like(d)] + ([sc] if self.form in ("line", "mline") else [])
            lin = torch.linalg.lstsq(torch.stack(cols, 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, beta, kappa, t, delta] + ([gamma] if self.sharp else []) + ([eta] if self.asum else []) + ([rho] if self.hl == 1 else []) + ([cvec] if self.hl == 64 else []) + ([alpha] if self.halpha else []) + ([gs] if self.hself and GS_FREE else []), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.smax(self.score(beta, delta), t) + eta * self.logn_c; d = self.distance(kappa, b0, idx)
            pred = lin[0] * d + lin[1] + (lin[2] * sc if self.form in ("line", "mline") else 0)
            if self.form in ("wsame", "wcombo"):   # pair weights: same-size share-none pairs count w times
                pw = 1 + (self.w - 1) * self.same0[idx].double(); mse = (pw * (pred - y) ** 2).sum() / pw.sum()
            else:
                mse = ((pred - y) ** 2).mean()
            loss = mse + self.lam * (beta ** 2 + kappa ** 2 + (delta ** 2).sum() + gamma ** 2 + eta ** 2 + rho ** 2 + gs ** 2 + self.clam * (cvec ** 2).sum())
            if self.form == "partial":
                r_ds, r_dy = _corr(d, s), _corr(d, y)
                loss = loss + self.w * vy * (r_ds - r_dy * r_sy) ** 2 / ((1 - r_dy ** 2) * (1 - r_sy ** 2) + 1e-12)
            elif self.form == "dist":
                loss = loss + self.w * vy * _corr(d, s) ** 2
            elif self.form == "same":
                loss = loss + self.w * vy * _corr(d[same], s[same]) ** 2
            elif self.form == "resid":
                loss = loss + self.w * vy * _corr(pred - y, s) ** 2
            elif self.form == "mresid":
                loss = loss + self.w * vy * _corr(pred - y, hm) ** 2
            elif self.form == "msame":
                loss = loss + self.w * vy * _corr((pred - y)[same], hm[same]) ** 2
            elif self.form in ("combo", "pcombo"):
                loss = loss + vy * (0.3 * _corr(d, s) ** 2 + self.w * _corr(pred - y, hm) ** 2)
            elif self.form == "wcombo":
                loss = loss + vy * (0.3 * _corr(d, s) ** 2 + 1.0 * _corr(pred - y, hm) ** 2)
            elif self.form == "both":
                loss = loss + self.w * vy * (_corr(pred - y, hm) ** 2 + _corr(pred - y, s) ** 2)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = (self.smax(self.score(beta, delta), t) + eta * self.logn_c).detach()
        self.beta, self.kappa, self.t, self.delta, self.b, self.lin = beta.detach(), kappa.detach(), t.detach(), delta.detach(), b0, lin.detach()
        self.gamma = gamma.detach(); self.eta = eta.detach(); self.rho, self.cvec, self.alpha = rho.detach(), cvec.detach(), alpha.detach(); self.gs = gs.detach(); return self

    def predict(self, idx):
        if self.form not in ("line", "mline"):
            return super().predict(idx)
        with torch.no_grad():
            i = torch.as_tensor(np.asarray(idx)); d = self.distance(self.kappa, self.b, i)
            extra = self.S[i] - self.s_mean if self.form == "line" else self.Hm[i] - self.h_mean
            return (self.lin[0] * d + self.lin[1] + self.lin[2] * extra).numpy()


def models(BB, q_op, X, A, B):
    def ens(form, w):
        return lambda tr, te, y: np.mean([SizeTune(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], form, w).fit(tr, y[tr]).predict(te) for m in BB], 0)
    out = {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)}
    out.update({nm: ens(*v) for nm, v in VARIANTS.items()}); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rows", default=""); ap.add_argument("--merge", nargs="*", default=None); ap.add_argument("--only", default="")
    ap.add_argument("--reps", default="")   # "a:b" = repeats a..b-1 of the standard folds (seed 0); --merge joins the chunks
    a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_size_tune", {"lam": LAM, "weights": WEIGHTS, "members": eo.SEEDS, "heldout": a.heldout, "rows": a.rows,
                                    "repeats": a.repeats, "device": dev}, tag="heldout" if a.heldout else (f"rows{a.rows}" if a.rows else "score"))
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    from rdkit import Chem
    global HEAVY_M; HEAVY_M = np.array([Chem.MolFromSmiles(str(x)).GetNumHeavyAtoms() for x in op.smiles_M], float)
    BB = None if a.merge else eo.backbones(dev, rep)[1]["o5"]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    if a.heldout:
        train = data.load_train(); store = {}
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            mdl = models(BB, q_op, X, A, B)
            preds = {nm: fp(tr, te, y_all) for nm, fp in mdl.items() if not a.only or nm in (REF, RB) or any(o in nm for o in a.only.split(";"))}
            y = hs.y_P; ref = preds[REF]; rb = preds[RB]; rows = []; store[hs.name] = {"y": y, "pred": preds}
            rep.section(f"{hs.name}: {len(y)} pairs" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = c2 = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y); c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                if nm not in (REF, RB):
                    cr, _ = paired(p, rb, y); c2 = f"{s['rmse'] - metrics.rmse(rb, y):+.4f} / {s['r'] - metrics.pearson(rb, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c, c2])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]", "change from (b)"], rows)
        out = ROOT / "results/eval_size_tune" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_heldout_pred.pkl"
        out.parent.mkdir(parents=True, exist_ok=True); pickle.dump(store, open(out, "wb")); rep.line(f"predictions: {out.name}")
        rep.save(); return
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    if a.merge:
        res = {}; chunks = {}
        for f in a.merge:
            d = pickle.load(open(f, "rb")); res.update(d.get("res", {}))
            for nm, c in d.get("chunk", {}).items():
                chunks.setdefault(nm, []).append(c)
        for nm, cs in chunks.items():   # join repeat chunks into the standard result (same fold order as cross_validate)
            cs.sort(key=lambda c: c[0]); assert [c[0] for c in cs] == [0] + [c[1] for c in cs[:-1]], f"repeat chunks not contiguous: {nm}"
            pair_N = np.concatenate([c[2] for c in cs])
            res[nm] = evaluate.CVResult(pair_N, np.concatenate([c[3] for c in cs]), y[pair_N], np.array([P[i].dataset for i in pair_N]),
                                        np.array([P[i].jaccard == 0 for i in pair_N]), 5, cs[-1][1], 0)
            rep.line(f"joined {nm}: repeats 0-{cs[-1][1] - 1}, {len(cs)} chunks, non-finite {int((~np.isfinite(res[nm].pred_N)).sum())}")
        res = {nm: res[nm] for nm in [REF] + list(VARIANTS) if nm in res}
    else:
        mdl = models(BB, q_op, X, A, B); mdl = {k_: mdl[k_] for k_ in [list(mdl)[int(i)] for i in a.rows.split(",")]}
        res = {}
        if a.reps:   # one chunk of the standard 50 x 5 folds (a repeat's folds depend only on seed and repeat number)
            ra, rb_ = map(int, a.reps.split(":")); folds = [f for f in cv.mixture_holdout_folds(P, k=5, repeats=rb_, seed=0) if f.repeat >= ra]
            for nm, fp in mdl.items():
                t0 = time.time(); pr = [np.asarray(fp(np.array(f.train), np.array(f.test), y), float) for f in folds]
                res[nm] = (ra, rb_, np.concatenate([np.array(f.test) for f in folds]), np.concatenate(pr))
                rep.line(f"done: {nm}; repeats {ra}-{rb_ - 1}; {time.time() - t0:.0f} s; non-finite {int((~np.isfinite(res[nm][3])).sum())}")
            pk = ROOT / "results/eval_size_tune" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv50m{len(eo.SEEDS)}_rows{a.rows.replace(',', '-')}_reps{ra}-{rb_}.pkl"
            pickle.dump({"chunk": res}, open(pk, "wb")); rep.line(f"saved {pk.name}"); rep.save(); return
        for nm, fp in mdl.items():
            t0 = time.time(); res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y), repeats=a.repeats)
            rep.line(f"done: {nm}; {time.time() - t0:.0f} s; non-finite {int((~np.isfinite(res[nm].pred_N)).sum())}")
        pk = ROOT / "results/eval_size_tune" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv{a.repeats}{'' if len(eo.SEEDS) == 3 else f'm{len(eo.SEEDS)}'}_rows{a.rows.replace(',', '-')}.pkl"
        pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"saved {pk.name}"); rep.save(); return
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "same-size share none": (k_P == 0) & (gap <= 1),
           "share none, sizes differ": (k_P == 0) & (gap >= 2), "setting 2 (share >= 1)": k_P >= 1,
           "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    keep = [nm for nm in res if not a.only or nm in (REF, RB) or any(o in nm for o in a.only.split(";"))]
    res = {nm: res[nm] for nm in keep}
    rows = []
    for sub in SUB:
        for nm, r_ in res.items():
            m = np.isfinite(r_.pred_N) & np.isfinite(res[REF].pred_N)
            if SUB[sub] is not None:
                m &= SUB[sub][r_.pair_N]
            s_ = evaluate.score_cv(P, r_, reference=None if nm == REF else res[REF], mask_N=m)
            s2 = evaluate.score_cv(P, r_, reference=res[RB], mask_N=m) if nm not in (REF, RB) and RB in res else None
            rows.append([nm, sub, f"{s_['rmse']:.4f}", f"{s_['r']:.3f}",
                         "" if nm == REF else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]",
                         "" if s2 is None else f"{s2['rmse_change']:+.4f} / {s2['r_change']:+.3f} [{s2['r_change_ci'][0]:+.3f}, {s2['r_change_ci'][1]:+.3f}]"])
    rep.section("CV on development pairs: RMSE / r; change [95% CI]")
    rep.table(["model", "subset", "RMSE", "r", "change from O1.2b", "change from (b)"], rows); rep.save()


if __name__ == "__main__":
    main()
