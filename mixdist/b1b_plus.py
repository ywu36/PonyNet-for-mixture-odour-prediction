# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235)   P = pairs
# ---------------------------------------------------------------------------------------------
"""B1b+ (user-named 2026-11-18): B1b (b1a+ session, mixdist/b1b.py, imported read-only) plus the related-component push.

B1b = B1a++ + pull of noticeable shared molecules (and push of bland ones), only on pairs sharing a molecule.
Push (finding 5, row 5h): noticeable components that are related but are different molecules push a pair apart
(same RDKit group carried by different unshared molecules: weight +0.032 [+0.010, +0.053]; noticeable notes +0.016).
Stage 3, on all pairs (the push also acts on pairs sharing nothing):
    pred_B1b+ = pred_B1b + h0 + h * z(F_push)
fitted by OLS on the training pairs' in-sample B1b residuals. F_push given per pair (inputs only).
"""
from __future__ import annotations

import numpy as np

from mixdist.b1b import B1b


class B1bPlus:
    def __init__(self, q_MxW, x_MxG, A_PxM, B_PxM, size_PxS, F_push_P, lam=1e-3):
        self.b1b = B1b(q_MxW, x_MxG, A_PxM, B_PxM, size_PxS, lam)
        self.Fp = np.asarray(F_push_P, float)

    def fit(self, idx, y):
        idx = np.asarray(idx)
        self.b1b.fit(idx, y)
        r = y - self.b1b.predict(idx)
        self.mu, self.sd = self.Fp[idx].mean(), self.Fp[idx].std() + 1e-12
        D = np.c_[np.ones(len(idx)), (self.Fp[idx] - self.mu) / self.sd]
        self.h = np.linalg.lstsq(D, r, rcond=None)[0]
        return self

    def predict(self, idx):
        idx = np.asarray(idx)
        return self.b1b.predict(idx) + self.h[0] + self.h[1] * (self.Fp[idx] - self.mu) / self.sd
