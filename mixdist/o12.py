# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules   W = open-POM words (138)   G = RDKit groups
#   P = pairs   S = size covariates
# ---------------------------------------------------------------------------------------------
"""O1.2 (design/o1_2_design.md): gain-weighted pooling with divisive normalization.

    G(m) = exp(theta . x(m))                         gain per molecule from its RDKit groups
    p(A) = sum_m G(m) q(m) / sum_m G(m)              138 pooling units, one global divisor
    y(A) = p(A) / (kappa + |p(A)|)                   divisive normalization, semi-saturation kappa >= 0
    d(A, B) = |y(A) - y(B)| ;  prediction = a d + c  [+ size covariates, reference only]
theta = 0 and kappa = 0 give B1a exactly. kappa = k^2 with k learned (k starts at K0, kappa ~ 0).
Fitted by L-BFGS on squared error + lam |theta|^2 (theta only).
"""
from __future__ import annotations

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
K0 = 0.1                     # kappa starts at 0.01 (profile lengths are 2-13), i.e. ~B1a


class O12:
    def __init__(self, q_MxW, x_MxG, mixA_PxM, mixB_PxM, use_theta=True, use_kappa=True, size_PxS=None, lam=1e-3, k0=K0):
        self.q = torch.tensor(np.asarray(q_MxW, float)); self.x = torch.tensor(np.asarray(x_MxG, float))
        self.A = torch.tensor(mixA_PxM, dtype=torch.float64); self.B = torch.tensor(mixB_PxM, dtype=torch.float64)
        self.S = None if size_PxS is None else torch.tensor(np.asarray(size_PxS, float))
        self.use_theta, self.use_kappa, self.lam, self.k0 = use_theta, use_kappa, lam, k0

    def distance(self, theta, k, idx):
        w = torch.exp(self.x @ theta)
        kappa = k ** 2 if self.use_kappa else torch.zeros(())

        def out(Mx):
            p = (Mx[idx] * w) @ self.q / (Mx[idx] @ w)[:, None]
            return p / (kappa + p.norm(dim=1, keepdim=True))
        return (out(self.A) - out(self.B)).norm(dim=1)

    def _design(self, d, idx):
        cols = [d, torch.ones_like(d)] + ([] if self.S is None else [self.S[idx, j] for j in range(self.S.shape[1])])
        return torch.stack(cols, 1)

    def fit(self, idx, y, w_pairs=None, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        wp = torch.ones(len(idx)) if w_pairs is None else torch.tensor(np.asarray(w_pairs, float)); wp = wp / wp.sum()
        theta = torch.zeros(self.x.shape[1], requires_grad=self.use_theta)
        k = torch.tensor(self.k0 if self.use_kappa else 0.0, requires_grad=self.use_kappa)
        with torch.no_grad():
            D = self._design(self.distance(theta, k, idx), idx)
            lin0 = torch.linalg.lstsq(D * wp.sqrt()[:, None], (y * wp.sqrt())[:, None]).solution[:, 0]
        lin = lin0.clone().requires_grad_(True)
        params = [lin] + ([theta] if self.use_theta else []) + ([k] if self.use_kappa else [])
        opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, k, idx), idx) @ lin
            loss = (wp * (pred - y) ** 2).sum() + self.lam * (theta ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.k, self.lin = theta.detach(), k.detach(), lin.detach()
        return self

    @property
    def kappa(self):
        return float(self.k ** 2) if self.use_kappa else 0.0

    def predict(self, idx):
        idx = torch.as_tensor(np.asarray(idx))
        with torch.no_grad():
            return (self._design(self.distance(self.theta, self.k, idx), idx) @ self.lin).numpy()


class O12b(O12):
    """O1.2b: keep both B1a's direction and the profile length instead of trading them with kappa.

        u(A) = p(A) / |p(A)|  (direction, as B1a)      h(A) = log |p(A)|  (length: specific vs "white")
        d(A, B) = sqrt( |u(A) - u(B)|^2 + beta (h(A) - h(B))^2 ),   beta = b^2 >= 0 learned (b starts at k0)
    beta = 0 gives O1.2 theta only (and with theta = 0, B1a). The kappa slot of O12 holds b here.
    """

    def distance(self, theta, k, idx):
        w = torch.exp(self.x @ theta)
        beta = k ** 2 if self.use_kappa else torch.zeros(())

        def out(Mx):
            p = (Mx[idx] * w) @ self.q / (Mx[idx] @ w)[:, None]
            n = p.norm(dim=1, keepdim=True)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)

    @property
    def beta(self):
        return self.kappa


class O12c(O12b):
    """O1.2c: a shared molecule's weight depends on its company (mixture size), inside the pooling.

        For pair (A, B) and molecule m in both: weight in A = G(m) exp(alpha + gamma z_A), z_A = log(n_A / 10);
        unshared molecules keep G(m). p, u, h and d as O1.2b.
        mode "const": gamma fixed 0 (one constant extra weight, as the earlier anchor test).
        mode "size":  alpha and gamma learned (sign can flip with size: noticed in small mixtures, blurred in large).
        mode "groups": shared weight exp(alpha + eta . x(m)), one eta per RDKit group (does a shared molecule stand
                        out more for some chemical groups?); eta penalised by lam like theta.
        mode "tuneout": no reweighting; d = (1 - lmb) d(A, B) + lmb d(a, b), a = A minus shared, b = B minus shared
                        (people compare only the differing parts); lmb = sigmoid(s0), s0 starts at -4 (lmb 0.018);
                        d(a, b) = d(A, B) when a or b is empty.
    alpha = gamma = 0 gives O1.2b exactly. alpha, gamma unpenalised, start at 0.
    """
    Z0 = np.log(10.0)

    def __init__(self, q_MxW, x_MxG, mixA_PxM, mixB_PxM, mode="size", lam=1e-3, k0=K0):
        super().__init__(q_MxW, x_MxG, mixA_PxM, mixB_PxM, True, True, None, lam, k0)
        self.mode = mode
        self.shared = self.A * self.B
        self.zA = torch.log(self.A.sum(1)) - self.Z0; self.zB = torch.log(self.B.sum(1)) - self.Z0
        self.s = torch.zeros(1 + self.x.shape[1] if mode == "groups" else 2)

    def distance(self, theta, k, idx):
        if self.mode == "tuneout":
            return self._tuneout(theta, k, idx)
        w = torch.exp(self.x @ theta); beta = k ** 2
        gam = self.s[1] if self.mode == "size" else torch.zeros(())
        boost = (self.s[0] + self.x @ self.s[1:]) if self.mode == "groups" else None

        def out(Mx, z):
            if boost is not None:
                W = Mx[idx] * w * torch.exp(self.shared[idx] * boost)
            else:
                W = Mx[idx] * w * torch.exp(self.shared[idx] * (self.s[0] + gam * z[idx])[:, None])
            p = W @ self.q / W.sum(1, keepdim=True)
            n = p.norm(dim=1, keepdim=True)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A, self.zA); uB, hB = out(self.B, self.zB)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)

    def _tuneout(self, theta, k, idx):
        w = torch.exp(self.x @ theta); beta = k ** 2

        def d(MA, MB):
            def out(Mx):
                W = Mx * w; p = W @ self.q / W.sum(1, keepdim=True).clamp_min(1e-12)
                n = p.norm(dim=1, keepdim=True).clamp_min(1e-12)
                return p / n, torch.log(n[:, 0])
            uA, hA = out(MA); uB, hB = out(MB)
            return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)
        A, B, S = self.A[idx], self.B[idx], self.shared[idx]
        full = d(A, B)
        ok = ((A - S).sum(1) > 0) & ((B - S).sum(1) > 0)
        diff = torch.where(ok, d(torch.where(ok[:, None], A - S, A), torch.where(ok[:, None], B - S, B)), full)
        lmb = torch.sigmoid(self.s[0])
        return (1 - lmb) * full + lmb * diff

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        theta = torch.zeros(self.x.shape[1], requires_grad=True)
        k = torch.tensor(self.k0, requires_grad=True)
        ns = 1 + self.x.shape[1] if self.mode == "groups" else 2
        self.s = torch.zeros(ns); self.s[0] = -4.0 if self.mode == "tuneout" else 0.0; self.s.requires_grad_(True)
        with torch.no_grad():
            D = self._design(self.distance(theta, k, idx), idx)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, theta, k, self.s], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, k, idx), idx) @ lin
            loss = ((pred - y) ** 2).mean() + self.lam * (theta ** 2).sum()
            if self.mode == "groups":
                loss = loss + self.lam * (self.s[1:] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.k, self.lin, self.s = theta.detach(), k.detach(), lin.detach(), self.s.detach()
        return self


class O12e(O12b):
    """O1.2e: structure changes the odor profile, not only the weight (low rank).

        q'(m) = q(m) + (x(m) V) U        V: groups x rank, U: rank x words; x = the 24 RDKit groups
        then gain G(m) = exp(theta . x(m)), pooled p, direction u, length h and d exactly as O1.2b.
    U = 0 gives O1.2b. U starts at 0, V at small fixed random values (seed 0) so U gets a gradient.
    theta, U and V penalised by lam; beta = b^2 unpenalised (as O1.2b).
    """

    def __init__(self, q_MxW, x_MxG, mixA_PxM, mixB_PxM, rank=2, lam=1e-3, k0=K0):
        super().__init__(q_MxW, x_MxG, mixA_PxM, mixB_PxM, True, True, None, lam, k0)
        self.rank = rank
        self.U = torch.zeros(rank, self.q.shape[1]); self.V = torch.zeros(self.x.shape[1], rank)

    def _qs(self, U, V):
        return self.q + (self.x @ V) @ U

    def distance(self, theta, k, idx, U=None, V=None):
        U = self.U if U is None else U; V = self.V if V is None else V
        q = self._qs(U, V); w = torch.exp(self.x @ theta); beta = k ** 2

        def out(Mx):
            p = (Mx[idx] * w) @ q / (Mx[idx] @ w)[:, None]
            n = p.norm(dim=1, keepdim=True).clamp_min(1e-9)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)

    def fit(self, idx, y, max_iter=500):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        theta = torch.zeros(self.x.shape[1], requires_grad=True)
        k = torch.tensor(self.k0, requires_grad=True)
        g = torch.Generator().manual_seed(0)
        U = torch.zeros(self.rank, self.q.shape[1], requires_grad=True)
        V = (0.1 * torch.randn(self.x.shape[1], self.rank, generator=g)).requires_grad_(True)
        with torch.no_grad():
            D = self._design(self.distance(theta, k, idx, U, V), idx)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, theta, k, U, V], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, k, idx, U, V), idx) @ lin
            loss = ((pred - y) ** 2).mean() + self.lam * ((theta ** 2).sum() + (U ** 2).sum() + (V ** 2).sum())
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.k, self.lin, self.U, self.V = theta.detach(), k.detach(), lin.detach(), U.detach(), V.detach()
        return self


class O12k(O12b):
    """O1.2k: kernel set distance -- each mixture stays a set of molecules (no averaging before the comparison).

        w_A(m) = G(m) / sum_A G       gain-weighted membership (G = exp(theta . groups), as O1.2b)
        K(m, n) = exp(-|q_m - q_n|^2 / (2 sigma^2))     Gaussian kernel on single-molecule profiles, sigma learned
        MMD^2(A, B) = w_A K w_A + w_B K w_B - 2 w_A K w_B
    A shared molecule matches itself (k = 1, pulls closer); an unmatched standout matches nothing (pushes); a tight
    mixture has high self-similarity, a spread one low. mode "set": d = sqrt(MMD^2 + beta (h_A - h_B)^2) (O1.2b's
    length term kept, direction replaced by the set comparison). mode "add": d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2
    + gamma MMD^2), gamma = g^2 >= 0, so gamma = 0 is O1.2b exactly. sigma starts at the median molecule distance.
    """

    def __init__(self, q_MxW, x_MxG, mixA_PxM, mixB_PxM, mode="add", lam=1e-3, k0=K0, g0=0.5, identity=False):
        super().__init__(q_MxW, x_MxG, mixA_PxM, mixB_PxM, True, True, None, lam, k0)
        self.mode, self.g0, self.identity = mode, g0, identity     # identity: K = I (control: only exact molecule identity counts)
        self.D2 = torch.cdist(self.q, self.q) ** 2
        self.s0 = float(np.log(np.sqrt(np.median(self.D2[self.D2 > 0].numpy()))))

    def mmd2(self, theta, s, idx):
        G = torch.exp(self.x @ theta)
        K = torch.eye(len(self.q), dtype=self.q.dtype) if self.identity else torch.exp(-self.D2 / (2 * torch.exp(2 * s)))
        wA = self.A[idx] * G; wA = wA / wA.sum(1, keepdim=True)
        wB = self.B[idx] * G; wB = wB / wB.sum(1, keepdim=True)
        kAA = ((wA @ K) * wA).sum(1); kBB = ((wB @ K) * wB).sum(1); kAB = ((wA @ K) * wB).sum(1)
        return (kAA + kBB - 2 * kAB).clamp_min(0)

    def distance(self, theta, k, idx, g=None, s=None):
        g = self.g if g is None else g; s = self.s if s is None else s
        w = torch.exp(self.x @ theta); beta = k ** 2

        def out(Mx):
            p = (Mx[idx] * w) @ self.q / (Mx[idx] @ w)[:, None]
            n = p.norm(dim=1, keepdim=True)
            return p / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        m2 = self.mmd2(theta, s, idx)
        if self.mode == "set":
            return torch.sqrt(m2 + beta * (hA - hB) ** 2 + 1e-12)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + g ** 2 * m2 + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        theta = torch.zeros(self.x.shape[1], requires_grad=True); k = torch.tensor(self.k0, requires_grad=True)
        g = torch.tensor(self.g0, requires_grad=True); s = torch.tensor(self.s0, requires_grad=True)
        with torch.no_grad():
            D = self._design(self.distance(theta, k, idx, g, s), idx)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, theta, k, g, s], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, k, idx, g, s), idx) @ lin
            loss = ((pred - y) ** 2).mean() + self.lam * (theta ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.k, self.lin, self.g, self.s = theta.detach(), k.detach(), lin.detach(), g.detach(), s.detach()
        return self

    @property
    def gamma(self):
        return float(self.g ** 2)

    @property
    def sigma(self):
        return float(torch.exp(self.s))
