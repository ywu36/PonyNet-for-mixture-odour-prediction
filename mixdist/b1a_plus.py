# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   W = open-POM words (138)
#   K = pieces   P = pairs   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""B1a+ : B1a with sub-molecular dominance weights (and an optional size background).

B1a: per-molecule open-POM quantile profile q_i -> MEAN per mixture -> length 1 -> Euclidean -> line.
B1a+: the mean becomes a weighted mean whose weights are set by the molecule's pieces:
    w_i = exp(sum_k theta_k * x_ik)        x_ik = 1 if molecule i carries piece k
    mu_A = sum_{i in A} w_i q_i / sum_{i in A} w_i ;  d = || mu_A/|mu_A| - mu_B/|mu_B| ||
    y_hat = a * d + c  [+ b1 (log nA + log nB) + b2 |log nA - log nB|]  [+ one offset per experiment]
theta_k > 0: molecules carrying piece k dominate the mixture's smell; < 0: they are masked.
theta = 0 gives B1a exactly. Fitted by L-BFGS on squared error + lam * |theta|^2 (theta only).
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


class B1aPlus:
    def __init__(self, q_MxW, x_MxK, mixA_PxM, mixB_PxM, size_PxS=None, lam=1e-3, exp_PxE=None):
        self.q = torch.tensor(np.asarray(q_MxW, float)); self.x = torch.tensor(np.asarray(x_MxK, float))
        self.A = torch.tensor(mixA_PxM, dtype=torch.float64); self.B = torch.tensor(mixB_PxM, dtype=torch.float64)
        self.S = None if size_PxS is None else torch.tensor(np.asarray(size_PxS, float))
        self.lam = lam
        self.E = None if exp_PxE is None else torch.tensor(np.asarray(exp_PxE, float)[:, 1:])   # first experiment = reference

    def distance(self, theta, idx):
        w = torch.exp(self.x @ theta)
        def mu(Mx):
            m = (Mx[idx] * w) @ self.q / (Mx[idx] @ w)[:, None]
            return m / m.norm(dim=1, keepdim=True)
        return (mu(self.A) - mu(self.B)).norm(dim=1)

    def _design(self, d, idx):
        cols = [d, torch.ones_like(d)] + ([] if self.S is None else [self.S[idx, j] for j in range(self.S.shape[1])]) \
            + ([] if self.E is None else [self.E[idx, j] for j in range(self.E.shape[1])])
        return torch.stack(cols, 1)

    def fit(self, idx, y, w_pairs=None, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        wp = torch.ones(len(idx)) if w_pairs is None else torch.tensor(np.asarray(w_pairs, float))
        wp = wp / wp.sum()
        theta = torch.zeros(self.x.shape[1], requires_grad=True)
        with torch.no_grad():                                          # start: B1a line (+ size) at theta = 0
            D = self._design(self.distance(theta, idx), idx); sw = wp.sqrt()[:, None]
            lin0 = torch.linalg.lstsq(D * sw, (y * wp.sqrt())[:, None]).solution[:, 0]
        lin = lin0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([theta, lin], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, idx), idx) @ lin
            loss = (wp * (pred - y) ** 2).sum() + self.lam * (theta ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.lin = theta.detach(), lin.detach()
        return self

    def predict(self, idx):
        idx = torch.as_tensor(np.asarray(idx))
        with torch.no_grad():
            return (self._design(self.distance(self.theta, idx), idx) @ self.lin).numpy()


def membership(pairs, cids_M):
    row = {int(c): i for i, c in enumerate(cids_M)}
    A = np.zeros((len(pairs), len(cids_M))); B = np.zeros_like(A)
    for p_, pr in enumerate(pairs):
        A[p_, [row[c] for c in pr.a]] = 1; B[p_, [row[c] for c in pr.b]] = 1
    return A, B


def size_features(pairs):
    la = np.log([len(p.a) for p in pairs]); lb = np.log([len(p.b) for p in pairs])
    return np.c_[la + lb, np.abs(la - lb)]


def experiment_onehot(pairs):
    exps = sorted({p.dataset for p in pairs})
    return np.array([[p.dataset == e for e in exps] for p in pairs], float), exps
