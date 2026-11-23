# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   W = open-POM words (138)   P = pairs   G = RDKit groups (24)
# ---------------------------------------------------------------------------------------------
"""B1b: B1a++ plus the shared-molecule term (user-named 2026-11-18; spec settled with session pred-mixture-6c).

Observation (setting 2): a molecule present in both mixtures pulls the pair closer than its share when it stands out
from the rest of its own mixture, and pushes slightly when it blends in; unshared standouts and common notes from
different molecules do nothing. Per shared molecule s and side X in {A, B}:
    c_X(s) = || q_s - mean q over X's unshared molecules ||     (q = B1a quantile profiles, theta-free; 0 if none)
    F_mol  = sum over shared s of min(c_A(s), c_B(s))          (a sum: grows with the number shared)
    k      = number of shared molecules
Stage-wise, so pairs sharing nothing are exactly B1a++:
    pred_B1b = pred_B1a++ + [k > 0] * (g0 + g_k * z(k) + g_F * z(F_mol))
Stage 1: B1a++ (B1aPlus: groups theta, lam 1e-3, size line) on the training pairs. Stage 2: OLS of the training
setting-2 pairs' in-sample B1a++ residuals on [1, z(k), z(F)]; z-scaling on the training setting-2 pairs (only changes
how g is reported). All inputs are available for a new test pair.
"""
from __future__ import annotations

import numpy as np

from mixdist.b1a_plus import B1aPlus


def shared_features(q_MxW, A_PxM, B_PxM):
    """F_mol and k per pair (0 for pairs sharing nothing)."""
    Sh = A_PxM * B_PxM
    P = len(Sh); F = np.zeros(P); k = Sh.sum(1)
    for p in range(P):
        sh = np.where(Sh[p] > 0)[0]
        if not len(sh):
            continue
        cs = []
        for Mx in (A_PxM, B_PxM):
            un = np.where((Mx[p] > 0) & (Sh[p] == 0))[0]
            cs.append(np.linalg.norm(q_MxW[sh] - q_MxW[un].mean(0), axis=1) if len(un) else np.zeros(len(sh)))
        F[p] = np.minimum(cs[0], cs[1]).sum()
    return F, k


class B1b:
    def __init__(self, q_MxW, x_MxG, A_PxM, B_PxM, size_PxS, lam=1e-3, F_P=None, k_P=None):
        self.base = B1aPlus(q_MxW, x_MxG, A_PxM, B_PxM, size_PxS, lam)
        if F_P is None:
            F_P, k_P = shared_features(np.asarray(q_MxW, float), A_PxM, B_PxM)
        self.F, self.k = np.asarray(F_P, float), np.asarray(k_P, float)

    def _X(self, idx):
        return np.c_[np.ones(len(idx)), (self.k[idx] - self.mu[0]) / self.sd[0], (self.F[idx] - self.mu[1]) / self.sd[1]]

    def fit(self, idx, y, resid=None):
        """resid: optional stage-2 targets (e.g. inner out-of-fold B1a++ residuals); default in-sample residuals."""
        idx = np.asarray(idx)
        self.base.fit(idx, y)
        r = y - self.base.predict(idx) if resid is None else np.asarray(resid, float)
        s2 = self.k[idx] > 0
        self.mu = np.array([self.k[idx][s2].mean(), self.F[idx][s2].mean()])
        self.sd = np.array([self.k[idx][s2].std(), self.F[idx][s2].std()]) + 1e-12
        self.g = np.linalg.lstsq(self._X(idx[s2]), r[s2], rcond=None)[0]
        return self

    def correction(self, idx):
        idx = np.asarray(idx)
        return np.where(self.k[idx] > 0, self._X(idx) @ self.g, 0.0)

    def predict(self, idx):
        return self.base.predict(idx) + self.correction(idx)
