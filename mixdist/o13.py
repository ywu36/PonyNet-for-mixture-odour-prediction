# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules   W = open-POM words (138)   G = RDKit groups
#   P = pairs   E = extra metric parameters
# ---------------------------------------------------------------------------------------------
"""O1.3 metric variants on top of O1.2's gain-weighted profile (design/o1_2_design.md, "Next: the metric side").

Shared: G(m) = exp(theta . x(m)); p(A) = sum G q / sum G; u = p/|p| (direction); h = log|p| (length, specific vs white).
Metric (all fed to a line a d + c; theta penalised by lam, metric parameters e unpenalised):
  angle   d = sqrt(phi^2 + beta dh^2), phi = arccos(u_A . u_B), beta = e0^2                       (ArcFace: angle)
  pfe     d = |u_A - u_B|^2 / S + gamma log S, S = sigma_A^2 + sigma_B^2, sigma^2 = exp(tau - 2h)  (PFE: uncertainty)
          tau = e0, gamma = e1
  hyper   x = rho u, rho = tanh(s |p|), s = e0^2; d = Poincare distance                         (hyperbolic)
  white   w = common direction of the training mixtures (theta = 0 profiles, fixed per fit); a = u . w;
          r = unit residual of u after removing w; d = sqrt(|r_A - r_B|^2 + gamma da^2 + beta dh^2), beta = e0^2,
          gamma = e1^2                                                                           (factor out white)
  margin  d = sqrt(relu(phi - m)^2 + beta dh^2), beta = e0^2, m = e1^2                          (ArcFace margin as a
                                                                                                  perceptual threshold)
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
INIT = {"angle": [0.1], "pfe": [0.0, 0.0], "hyper": [0.3], "white": [0.1, 0.1], "margin": [0.1, 0.1]}
EPS = 1e-9


class O13:
    def __init__(self, q_MxW, x_MxG, mixA_PxM, mixB_PxM, metric, lam=1e-3):
        self.q = torch.tensor(np.asarray(q_MxW, float)); self.x = torch.tensor(np.asarray(x_MxG, float))
        self.A = torch.tensor(mixA_PxM, dtype=torch.float64); self.B = torch.tensor(mixB_PxM, dtype=torch.float64)
        self.metric, self.lam = metric, lam
        self.w = None

    def _p(self, theta, Mx, idx):
        g = torch.exp(self.x @ theta)
        return (Mx[idx] * g) @ self.q / (Mx[idx] @ g)[:, None]

    def _common_direction(self, idx):
        with torch.no_grad():
            z = torch.zeros(self.x.shape[1])
            U = torch.cat([self._p(z, self.A, idx), self._p(z, self.B, idx)])
            U = U / U.norm(dim=1, keepdim=True)
            w = U.mean(0)
            return w / w.norm()

    def distance(self, theta, e, idx):
        pA, pB = self._p(theta, self.A, idx), self._p(theta, self.B, idx)
        nA, nB = pA.norm(dim=1), pB.norm(dim=1)
        uA, uB = pA / nA[:, None], pB / nB[:, None]
        dh2 = (torch.log(nA) - torch.log(nB)) ** 2
        m = self.metric
        if m in ("angle", "margin"):
            phi = torch.arccos(torch.clamp((uA * uB).sum(1), -1 + 1e-7, 1 - 1e-7))
            if m == "margin":
                phi = torch.relu(phi - e[1] ** 2)
            return torch.sqrt(phi ** 2 + e[0] ** 2 * dh2 + EPS)
        if m == "pfe":
            S = torch.exp(e[0] - 2 * torch.log(nA)) + torch.exp(e[0] - 2 * torch.log(nB))
            return ((uA - uB) ** 2).sum(1) / S + e[1] * torch.log(S)
        if m == "hyper":
            rA = torch.clamp(torch.tanh(e[0] ** 2 * nA), max=1 - 1e-5); rB = torch.clamp(torch.tanh(e[0] ** 2 * nB), max=1 - 1e-5)
            xA, xB = rA[:, None] * uA, rB[:, None] * uB
            z = 2 * ((xA - xB) ** 2).sum(1) / ((1 - rA ** 2) * (1 - rB ** 2))
            return torch.log(1 + z + torch.sqrt(z ** 2 + 2 * z + EPS))
        if m == "white":
            w = self.w
            aA, aB = uA @ w, uB @ w
            rA = uA - aA[:, None] * w; rB = uB - aB[:, None] * w
            rA = rA / rA.norm(dim=1, keepdim=True); rB = rB / rB.norm(dim=1, keepdim=True)
            return torch.sqrt(((rA - rB) ** 2).sum(1) + e[1] ** 2 * (aA - aB) ** 2 + e[0] ** 2 * dh2 + EPS)
        raise ValueError(m)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        if self.metric == "white":
            self.w = self._common_direction(idx)
        theta = torch.zeros(self.x.shape[1], requires_grad=True)
        e = torch.tensor(INIT[self.metric], requires_grad=True)
        with torch.no_grad():
            d = self.distance(theta, e, idx)
            D = torch.stack([d, torch.ones_like(d)], 1)
            lin0 = torch.linalg.lstsq(D, y[:, None]).solution[:, 0]
        lin = lin0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([lin, theta, e], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            d = self.distance(theta, e, idx)
            pred = lin[0] * d + lin[1]
            loss = ((pred - y) ** 2).mean() + self.lam * (theta ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.e, self.lin = theta.detach(), e.detach(), lin.detach()
        return self

    def predict(self, idx):
        idx = torch.as_tensor(np.asarray(idx))
        with torch.no_grad():
            d = self.distance(self.theta, self.e, idx)
            return (self.lin[0] * d + self.lin[1]).numpy()
