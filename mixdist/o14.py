# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   K = single molecules for the cell layer (~7.5k)
#   W = open-POM words (138)   E = input size (138 or 256)   H = cell units   G = RDKit groups (24)   P = pairs
# ---------------------------------------------------------------------------------------------
"""O1.4: a hidden cell layer between molecules and odour words (POM = observation layer only).

Hypothesis (user, 2026-11-18): odour words are a read-out of a hidden cell layer. Mixture molecules combine in the cell
layer; the words are read from the combined activation through a nonlinear read-out, so a mixture can read out as
something no component smells of (interpolation), until large mixtures turn white.

    cells    c(m) = softplus(Enc x(m))                  H nonnegative units; x = POM words q(m) or POM embedding
    read-out q^(c) = Dec(c), a small MLP (nonlinear)     trained so q^(c(m)) ~ q(m) on ~7.5k single molecules
    mixture  c(A) = sum_m G(m) c(m) / sum_m G(m)         gain G = exp(theta . groups), one global divisor (as O1.2b)
    words    q^(A) = Dec(c(A))                          not the average of the components' words
    distance direction + length (O1.2b) on q^(A)  [or, variant, on c(A)]
The cell layer and read-out are learned from single molecules only (no pair labels) and then frozen; the pairs fit only
theta, beta and the line (same as O1.2b), so this adds no pair-fitted capacity (O1.0 learned its receptor layer
from the 820 pairs and did not help).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from mixdist.device import pick_device
from mixdist.o12 import K0, O12b

torch.set_default_dtype(torch.float64)


class CellLayer(torch.nn.Module):
    """topk = None: dense softplus cells (O1.4). topk = k: each molecule keeps only its k strongest cells (ReLU, rest 0),
    so molecules drive different subsets of cells (selective, as receptors are)."""

    def __init__(self, n_in, n_cells=64, n_words=138, n_hidden=128, topk=None):
        super().__init__()
        self.topk = topk
        self.enc = torch.nn.Linear(n_in, n_cells)
        self.dec = torch.nn.Sequential(torch.nn.Linear(n_cells, n_hidden), torch.nn.ReLU(), torch.nn.Linear(n_hidden, n_words))

    def cells(self, x):
        if getattr(self, "topk", None) is None:
            return torch.nn.functional.softplus(self.enc(x))
        a = torch.relu(self.enc(x))
        thr = a.topk(self.topk, dim=1).values[:, -1:]
        return a * (a >= thr)

    def forward(self, x):
        c = self.cells(x)
        return self.dec(c), c


def word_profiles(probs_DREAM_MxW, probs_extra_KxW):
    """Quantile profiles as B1a: transformer fitted on the 235 DREAM molecules, applied to the extra molecules too."""
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=100).fit(probs_DREAM_MxW)
    return qt.transform(probs_DREAM_MxW), qt.transform(probs_extra_KxW)


def train_cells(x_KxE, q_KxW, n_cells=64, l1=1e-3, steps=3000, lr=3e-3, holdout=0.1, seed=0, log=print, device=None, topk=None):
    """Fit the cell layer + read-out on single molecules (reconstruct q from x) on the GPU by default
    (pick_device). Returns the model (moved back to the CPU), held-out R^2, fraction of active cells."""
    device = device or pick_device(); log(f"  device {device}")
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    x = torch.tensor(np.asarray(x_KxE, float)).to(device); q = torch.tensor(np.asarray(q_KxW, float)).to(device)
    perm = torch.randperm(len(x), generator=g).to(device); n_te = int(holdout * len(x))
    te, tr = perm[:n_te], perm[n_te:]
    net = CellLayer(x.shape[1], n_cells, q.shape[1], topk=topk).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for s in range(steps):
        opt.zero_grad()
        qh, c = net(x[tr])
        loss = ((qh - q[tr]) ** 2).mean() + l1 * c.abs().mean()
        loss.backward(); opt.step()
        if s % 1000 == 0 or s == steps - 1:
            log(f"  cell layer step {s}: train MSE {float(((qh - q[tr]) ** 2).mean()):.4f}")
    with torch.no_grad():
        qh, c = net(x[te])
        r2 = 1 - float(((qh - q[te]) ** 2).sum() / ((q[te] - q[te].mean(0)) ** 2).sum())
        active = float((c > 0.1).double().mean())
        cn = c / c.norm(dim=1, keepdim=True).clamp_min(1e-9); ov = cn @ cn.T
        overlap = float((ov.sum() - ov.diagonal().sum()) / (len(c) * (len(c) - 1)))
    net = net.cpu()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, r2, active, overlap


class O14(O12b):
    """O1.2b with pooling in the frozen cell layer. space "words": distance on Dec(c(A)); "cells": on c(A)."""

    def __init__(self, net, x_MxE, q_MxW, x_MxG, mixA_PxM, mixB_PxM, space="words", lam=1e-3, k0=K0):
        super().__init__(q_MxW, x_MxG, mixA_PxM, mixB_PxM, True, True, None, lam, k0)
        self.net, self.space = net, space
        with torch.no_grad():
            self.c = net.cells(torch.tensor(np.asarray(x_MxE, float)))

    def mixture_words(self, theta, Mx, idx):
        w = torch.exp(self.x @ theta)
        cA = (Mx[idx] * w) @ self.c / (Mx[idx] @ w)[:, None]
        return cA if self.space == "cells" else self.net.dec(cA)

    def distance(self, theta, k, idx):
        beta = k ** 2

        def out(Mx):
            p = self.mixture_words(theta, Mx, idx)
            n = p.norm(dim=1, keepdim=True).clamp_min(1e-9)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)


class O15(O12b):
    """O1.5: per-cell divisive normalization (competition inside the hidden cell layer), read out in word space.

        S_j(A) = sum_m G(m) c_j(m)                          total drive of cell j in mixture A (c = frozen O1.4 cells)
        e_m(A) = G(m) sum_j c_j(m) / (1 + rho S_j(A)) / sum_j c_j(m)
                 = the molecule's gain divided by the activity of the cells it drives (overlap with its company)
        p(A)   = sum_m e_m q(m) / sum_m e_m ; direction + length distance as O1.2b
    rho = r^2 >= 0 learned (r starts at R0). rho = 0 gives O1.2b exactly. A molecule whose cells its company also drives
    is suppressed; a distinct one keeps its weight (b1a+ contrast effect). cells=None uses POM word probabilities as
    "cells" (control: normalization without a learned cell layer).
    """
    R0 = 0.1

    def __init__(self, c_MxH, q_MxW, x_MxG, mixA_PxM, mixB_PxM, lam=1e-3, k0=K0, r0=None, device="cpu"):
        super().__init__(q_MxW, x_MxG, mixA_PxM, mixB_PxM, True, True, None, lam, k0)
        self.dev = device
        for nm in ("q", "x", "A", "B"):
            setattr(self, nm, getattr(self, nm).to(device))
        self.c = torch.tensor(np.asarray(c_MxH, float), device=device)
        self.cn = self.c / self.c.sum(1, keepdim=True)
        self.r0 = self.R0 if r0 is None else r0

    def weights(self, theta, rho, Mx, idx):
        G = torch.exp(self.x @ theta)
        S = (Mx[idx] * G) @ self.c                                  # P x H
        supp = (1 / (1 + rho * S)) @ self.cn.T                      # P x M: mean suppression over each molecule's cells
        return Mx[idx] * G * supp

    def distance(self, theta, k, idx, r=None):
        r = self.r if r is None else r
        beta = k ** 2; rho = r ** 2

        def out(Mx):
            e = self.weights(theta, rho, Mx, idx)
            p = e @ self.q / e.sum(1, keepdim=True)
            n = p.norm(dim=1, keepdim=True)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)

    def _design(self, d, idx):
        return torch.stack([d, torch.ones_like(d)], 1)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev); y = torch.tensor(np.asarray(y, float), device=self.dev)
        theta = torch.zeros(self.x.shape[1], device=self.dev, requires_grad=True)
        k = torch.tensor(self.k0, device=self.dev, requires_grad=True)
        r = torch.tensor(self.r0, device=self.dev, requires_grad=True)
        with torch.no_grad():
            D = self._design(self.distance(theta, k, idx, r), idx)
            lin = torch.linalg.lstsq(D.cpu(), y.cpu()[:, None]).solution[:, 0].to(self.dev).clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, theta, k, r], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, k, idx, r), idx) @ lin
            loss = ((pred - y) ** 2).mean() + self.lam * (theta ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.k, self.lin, self.r = theta.detach(), k.detach(), lin.detach(), r.detach()
        return self

    @property
    def rho(self):
        return float(self.r ** 2)

    def predict(self, idx):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev)
        with torch.no_grad():
            return (self._design(self.distance(self.theta, self.k, idx), idx) @ self.lin).cpu().numpy()
