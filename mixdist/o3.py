# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules   W = words (138)   G = RDKit groups (24, DREAM's)
#   P = pairs   Q = GoodScents pairs   T = Ma mixtures   U = raw RDKit fr_ units
# ---------------------------------------------------------------------------------------------
"""O3 (design/o3_design.md): a lean O2 that avoids memorizing and focuses on the underlying interaction.

Frozen O2-C-Single members decode every molecule alone -> profile p(m) (mean of 10 members). A mixture is the
weighted mean of its molecules' profiles in O1.2b's form (direction + length distance), so O3.0 differs from O1.2b
only in where profiles come from:
    O3.0  w(m)   = exp(theta . x_m)
    O3.1  w(m|A) = exp(theta . x_m - (r . x_m) * mean_{m' in A, m' != m} (s . x_m'))      rank-1 company suppression
x = DREAM's 24 RDKit groups. s, r (24 + 24) are fitted on GoodScents blends only (network frozen) and kept fixed on
DREAM, where only theta, beta and the line are fitted (27 numbers, as O1.2b). No parameter belongs to a molecule.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from .o12 import O12b


# ------------------------------------------------------------------ groups for any molecule (DREAM's 24 columns)

def fr_units(smiles):
    """-> (F_MxU 0/1, names) RDKit fr_ functional-group indicators (as pushpull.units_rdkit_groups)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    fns = [(n.removeprefix("fr_"), f) for n, f in Descriptors.descList if n.startswith("fr_")]
    F = np.zeros((len(smiles), len(fns)))
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            F[i] = [f(m) > 0 for _, f in fns]
    return F, [n for n, _ in fns]


def group_matrix(smiles, group_names):
    """DREAM's merged group columns for any molecules: a merged name 'a (= b, c)' is carried if any of a, b, c is."""
    F, names = fr_units(smiles); col = {n: j for j, n in enumerate(names)}
    X = np.zeros((len(smiles), len(group_names)))
    for g, gname in enumerate(group_names):
        m = re.match(r"^(.*?)(?: \(= (.*)\))?$", gname)
        parts = [m.group(1)] + ([p.strip() for p in m.group(2).split(",")] if m.group(2) else [])
        X[:, g] = F[:, [col[p] for p in parts]].max(1)
    return X


# ------------------------------------------------------------------ DREAM: O1.2b form with context weights

class O3(O12b):
    """O1.2b (direction + length, group gain, beta, line) whose molecule weights may depend on the company (s, r fixed,
    from GoodScents). s = r = None -> O3.0 = O1.2b's form on the given profiles. use_theta False -> length only."""

    def __init__(self, q_MxW, x_MxG, mixA_PxM, mixB_PxM, use_theta=True, lam=1e-3, s_G=None, r_G=None):
        super().__init__(q_MxW, x_MxG, mixA_PxM, mixB_PxM, use_theta, True, None, lam)
        self.s = None if s_G is None else torch.tensor(np.asarray(s_G, float))
        self.r = None if r_G is None else torch.tensor(np.asarray(r_G, float))

    def log_w(self, theta, Mx):
        """-> log weight P x M of every molecule in its mixture (rows of Mx = mixtures)."""
        lw = (self.x @ theta).unsqueeze(0).expand(Mx.shape[0], -1)
        if self.s is None:
            return lw
        sx, rx = self.x @ self.s, self.x @ self.r                          # M
        n = Mx.sum(1, keepdim=True)
        comp = (Mx @ sx).unsqueeze(1) - sx.unsqueeze(0)                    # sum over the other members
        comp = torch.where(n > 1, comp / (n - 1).clamp_min(1), torch.zeros_like(comp))
        return lw - rx.unsqueeze(0) * comp

    def distance(self, theta, k, idx):
        beta = k ** 2

        def out(Mx):
            Mi = Mx[idx]; w = Mi * torch.exp(self.log_w(theta, Mi))
            p = w @ self.q / w.sum(1, keepdim=True); n = p.norm(dim=1, keepdim=True)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)


# ------------------------------------------------------------------ GoodScents: fit s, r (and its own theta) on blends

def pair_log_w(x1, x2, theta, s, r):
    """Two-molecule blend: log weights of molecule 1 and 2 (company = the other molecule)."""
    l1 = x1 @ theta - (x1 @ r) * (x2 @ s)
    l2 = x2 @ theta - (x2 @ r) * (x1 @ s)
    return l1, l2


def blend_logits(p1, p2, x1, x2, theta, s, r, tau):
    """Weighted mean of the two decoded profiles -> logits / tau (tau: one temperature, learned)."""
    l1, l2 = pair_log_w(x1, x2, theta, s, r)
    w = torch.sigmoid(l1 - l2).unsqueeze(1)
    p = (w * p1 + (1 - w) * p2).clamp(1e-6, 1 - 1e-6)
    return torch.logit(p) / tau


def rank_loss(logits_QxW, Y_QxW, targets):
    lt = logits_QxW[:, targets]; p = Y_QxW[:, targets]; p = p / p.sum(1, keepdim=True)
    return -(p * torch.log_softmax(lt, 1)).sum(1)


def fit_scent(P_MxW, X_MxG, a_Q, b_Q, Y_QxW, targets, suppression=True, lam=1e-3, max_iter=500):
    """Fit theta (24), tau and, if suppression, s, r (24 + 24) on GoodScents blends by L-BFGS on the mean rank loss
    + lam * |params|^2. -> dict of tensors."""
    P = torch.as_tensor(P_MxW, dtype=torch.float64); X = torch.as_tensor(X_MxG, dtype=torch.float64)
    Y = torch.as_tensor(Y_QxW, dtype=torch.float64); T = torch.as_tensor(targets)
    a, b = torch.as_tensor(a_Q), torch.as_tensor(b_Q); G = X.shape[1]
    th = torch.zeros(G, requires_grad=True); lt = torch.zeros((), requires_grad=True)
    s = torch.full((G,), 0.01 if suppression else 0.0, requires_grad=suppression)
    r = torch.full((G,), 0.01 if suppression else 0.0, requires_grad=suppression)
    params = [th, lt] + ([s, r] if suppression else [])
    opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lg = blend_logits(P[a], P[b], X[a], X[b], th, s, r, torch.exp(lt))
        loss = rank_loss(lg, Y, T).mean() + lam * sum((p_ ** 2).sum() for p_ in params)
        loss.backward()
        return loss
    opt.step(closure)
    return {"theta": th.detach(), "s": s.detach(), "r": r.detach(), "tau": torch.exp(lt).detach()}


@torch.no_grad()
def scent_loss(fit, P_MxW, X_MxG, a_Q, b_Q, Y_QxW, targets):
    """Per-pair rank loss (lower = better) of a fitted GoodScents model on pairs a_Q, b_Q."""
    P = torch.as_tensor(P_MxW, dtype=torch.float64); X = torch.as_tensor(X_MxG, dtype=torch.float64)
    lg = blend_logits(P[a_Q], P[b_Q], X[a_Q], X[b_Q], fit["theta"], fit["s"], fit["r"], fit["tau"])
    return rank_loss(lg, torch.as_tensor(Y_QxW, dtype=torch.float64), torch.as_tensor(targets)).numpy(), lg.numpy()


def ma_dominance(fit, X_MxG, a_T, b_T):
    """Predicted dominance of molecule 1 over 2 in Ma mixtures: log w1 - log w2 (no refit on Ma)."""
    X = torch.as_tensor(X_MxG, dtype=torch.float64)
    l1, l2 = pair_log_w(X[a_T], X[b_T], fit["theta"], fit["s"], fit["r"])
    return (l1 - l2).numpy()


# ------------------------------------------------------------------ O3.1j: one joint fit, shared mechanism

def context_log_w(x_MxG, theta, s, r, Mx):
    """log weight (rows of Mx = mixtures) x M: theta . x_m - (r . x_m) * mean over the other members of (s . x)."""
    lw = (x_MxG @ theta).unsqueeze(0).expand(Mx.shape[0], -1)
    if s is None:
        return lw
    sx, rx = x_MxG @ s, x_MxG @ r
    n = Mx.sum(1, keepdim=True)
    comp = (Mx @ sx).unsqueeze(1) - sx.unsqueeze(0)
    comp = torch.where(n > 1, comp / (n - 1).clamp_min(1), torch.zeros_like(comp))
    return lw - rx.unsqueeze(0) * comp


class O3J:
    """O3.1j ("ImageNet pattern"): the mixing mechanism -- group gain theta (24) and company suppression s, r (24 + 24)
    -- is shared and fitted in ONE fit on the fold's DREAM training pairs AND all GoodScents train pairs. Own heads:
    DREAM beta + line (3), GoodScents temperature tau (1). Loss = DREAM MSE / var(y_train) + GoodScents mean rank loss
    + lam * |theta, s, r|^2. Ma is not used (independent check). suppression=False -> shared theta only."""

    def __init__(self, q_MxW, x_MxG, mixA_PxM, mixB_PxM, scent, lam=1e-3, device="cpu", suppression=True, init=None):
        f = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=device)  # noqa: E731
        self.q, self.x, self.A, self.B = f(q_MxW), f(x_MxG), f(mixA_PxM), f(mixB_PxM)
        self.sP, self.sX, self.sY = f(scent["P"]), f(scent["X"]), f(scent["Y"])
        self.sa, self.sb = torch.as_tensor(scent["a"], device=device), torch.as_tensor(scent["b"], device=device)
        self.sT = torch.as_tensor(scent["targets"], device=device)
        self.lam, self.dev, self.sup, self.init = lam, device, suppression, init
        # gathered once (not at every optimiser step): profiles / groups of both molecules of every GoodScents pair
        self.sPa, self.sPb, self.sXa, self.sXb = self.sP[self.sa], self.sP[self.sb], self.sX[self.sa], self.sX[self.sb]

    def distance(self, theta, s, r, k, idx):
        def out(Mx):
            Mi = Mx[idx]; w = Mi * torch.exp(context_log_w(self.x, theta, s, r, Mi))
            p = w @ self.q / w.sum(1, keepdim=True); n = p.norm(dim=1, keepdim=True)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + k ** 2 * (hA - hB) ** 2 + 1e-12)

    def scent_loss(self, theta, s, r, lt):
        lg = blend_logits(self.sPa, self.sPb, self.sXa, self.sXb, theta,
                          s if s is not None else torch.zeros_like(theta), r if r is not None else torch.zeros_like(theta), torch.exp(lt))
        return rank_loss(lg, self.sY, self.sT).mean()

    def fit(self, idx, y, max_iter=200):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev); y = torch.as_tensor(np.asarray(y, float), device=self.dev)
        G, z = self.x.shape[1], lambda v: torch.as_tensor(v, dtype=torch.float64, device=self.dev)  # noqa: E731
        i = self.init or {}
        th = z(i.get("theta", np.zeros(G))).clone().requires_grad_(True)
        lt = z(np.log(i.get("tau", 1.0))).clone().requires_grad_(True)
        s = z(i.get("s", np.full(G, 0.01))).clone().requires_grad_(True) if self.sup else None
        r = z(i.get("r", np.full(G, 0.01))).clone().requires_grad_(True) if self.sup else None
        k = z(0.1).requires_grad_(True); var = y.var()
        with torch.no_grad():
            d = self.distance(th, s, r, k, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        params = [lin, th, lt, k] + ([s, r] if self.sup else [])
        opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            d = self.distance(th, s, r, k, idx)
            loss = ((lin[0] * d + lin[1] - y) ** 2).mean() / var + self.scent_loss(th, s, r, lt) \
                + self.lam * ((th ** 2).sum() + (0 if s is None else (s ** 2).sum() + (r ** 2).sum()))
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.s, self.r, self.k, self.lin, self.tau = th.detach(), None if s is None else s.detach(), \
            None if r is None else r.detach(), k.detach(), lin.detach(), torch.exp(lt).detach()
        return self

    def predict(self, idx):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev)
        with torch.no_grad():
            return (self.lin[0] * self.distance(self.theta, self.s, self.r, self.k, idx) + self.lin[1]).cpu().numpy()
