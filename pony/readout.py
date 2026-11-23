"""Stage 2: the mixture readout, fitted on rated mixture pairs while the network stays frozen.

For a mixture A of molecules m, with atoms j and frozen hidden vectors (Section 2.4 of the paper):

    s_j       = H_j . (beta u0 + delta)                                  atom score; u0 = the network's own scorer
    l(m)      = tau log mean_j exp(s_j / tau)                            strength: soft maximum over the atoms
    w_A(m)    = exp( l(m) + kappa cos(v(m), mean of the other molecules' v in A) )   odd one out
    pi_A(m)   = w_A(m) / sum_m' w_A(m')                                  share within the mixture
    z(A)      = sum_m pi_A(m) v(m) / |v(m)|,   z^(A) = z(A) / |z(A)|      additive mix of unit directions
    g(A)      = exp(gamma . z^(A))                                        learned mixture length
    d(A, B)   = | g(A) z^(A) - g(B) z^(B) |,   y^(A, B) = a d(A, B) + c

133 numbers: a, c, beta, kappa, tau (= exp t), delta (64), gamma (64). Loss: mean squared error + 1e-3 (beta^2 + kappa^2 +
|delta|^2 + |gamma|^2), minimised by L-BFGS (strong Wolfe line search, 300 iterations) from beta = kappa = t = 0,
delta = gamma = 0 and a, c by least squares. Everything runs in float64.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

F64 = torch.float64
LAM = 1e-3
MAX_ITER = 300


@dataclass
class MoleculeTable:
    """Frozen per-molecule quantities of one network copy, in the units the readout was fitted in.
    V: molecules x D hidden vectors (divided by the copy's scale); H: atoms x D atom hidden vectors (same scale);
    mol: atom -> molecule row; u0: the copy's own atom scorer expressed in H (D)."""
    V: torch.Tensor
    H: torch.Tensor
    mol: torch.Tensor
    u0: torch.Tensor
    n_atoms: torch.Tensor = field(init=False)
    U: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.V, self.H, self.u0 = (torch.as_tensor(x, dtype=F64) for x in (self.V, self.H, self.u0))
        self.mol = torch.as_tensor(self.mol, dtype=torch.long)
        self.n_atoms = torch.bincount(self.mol, minlength=len(self.V)).double()
        self.U = self.V / self.V.norm(dim=1, keepdim=True)

    def __len__(self):
        return len(self.V)


@dataclass
class ReadoutParams:
    a: float = 0.0
    c: float = 0.0
    beta: float = 0.0
    kappa: float = 0.0
    t: float = 0.0                       # tau = exp(t)
    delta: np.ndarray | None = None      # D
    gamma: np.ndarray | None = None      # D

    def tensors(self, D: int):
        z = np.zeros(D)
        return (torch.tensor([self.a, self.c], dtype=F64), torch.tensor(self.beta, dtype=F64), torch.tensor(self.kappa, dtype=F64),
                torch.tensor(self.t, dtype=F64), torch.as_tensor(z if self.delta is None else self.delta, dtype=F64),
                torch.as_tensor(z if self.gamma is None else self.gamma, dtype=F64))

    def to_dict(self):
        return {"a": self.a, "c": self.c, "beta": self.beta, "kappa": self.kappa, "t": self.t,
                "delta": np.asarray(self.delta, float).tolist(), "gamma": np.asarray(self.gamma, float).tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["a"], d["c"], d["beta"], d["kappa"], d["t"], np.asarray(d["delta"], float), np.asarray(d["gamma"], float))


def strengths(tab: MoleculeTable, beta, delta, t):
    """l(m) = tau log mean_j exp(s_j / tau), s_j = H_j . (beta u0 + delta), for every molecule in the table."""
    tau = torch.exp(t).clamp_min(1e-4); z = (tab.H @ (beta * tab.u0 + delta)) / tau
    M = len(tab)
    mx = torch.full((M,), -1e30, dtype=F64).scatter_reduce(0, tab.mol, z, reduce="amax", include_self=True)
    se = torch.zeros(M, dtype=F64).index_add(0, tab.mol, torch.exp(z - mx[tab.mol]))
    return tau * (mx + torch.log(se / tab.n_atoms))


def context(tab: MoleculeTable, Mx: torch.Tensor, chunk: int = 256):
    """cos(v(m), mean of the other molecules' v) for each molecule of each mixture (rows of Mx); 0 for a lone molecule
    and for molecules not in the mixture."""
    out = []
    for k in range(0, len(Mx), chunk):
        X = Mx[k:k + chunk]; n = X.sum(1, keepdim=True); tot = X @ tab.V
        others = (tot.unsqueeze(1) - tab.V.unsqueeze(0)) / (n - 1).clamp_min(1).unsqueeze(2)
        c = torch.nn.functional.cosine_similarity(others, tab.V.unsqueeze(0).expand_as(others), dim=2)
        out.append(torch.where((X > 0) & (n > 1), c, torch.zeros_like(c)))
    return torch.cat(out) if out else torch.zeros_like(Mx)


def mixture_points(tab, Mx, C, l_M, kappa, gamma):
    """Each mixture as a point g(A) z^(A); also returns the shares pi (rows of Mx) and the lengths g."""
    lw = l_M.unsqueeze(0) + kappa * C
    lw = lw - torch.where(Mx > 0, lw, torch.full_like(lw, -torch.inf)).max(1, keepdim=True).values
    W = Mx * torch.exp(lw); pi = W / W.sum(1, keepdim=True)
    zm = pi @ tab.U; z = zm / zm.norm(dim=1, keepdim=True); g = torch.exp(z @ gamma)
    return g.unsqueeze(1) * z, pi, g


def distance(tab, A, B, CA, CB, l_M, kappa, gamma):
    pa, _, _ = mixture_points(tab, A, CA, l_M, kappa, gamma); pb, _, _ = mixture_points(tab, B, CB, l_M, kappa, gamma)
    return torch.sqrt(((pa - pb) ** 2).sum(1) + 1e-12)


def points(tab: MoleculeTable, Mx, p: ReadoutParams):
    """Each mixture (rows of the membership matrix Mx) as its point g(A) z(A) in the hidden layer (mixtures x 64)."""
    Mx = torch.as_tensor(Mx, dtype=F64)
    _, beta, kappa, t, delta, gamma = p.tensors(tab.V.shape[1])
    with torch.no_grad():
        return mixture_points(tab, Mx, context(tab, Mx), strengths(tab, beta, delta, t), kappa, gamma)[0].numpy()


def predict(tab: MoleculeTable, A, B, p: ReadoutParams):
    """Predicted ratings for the pairs (rows of the membership matrices A, B over the table's molecules)."""
    A, B = torch.as_tensor(A, dtype=F64), torch.as_tensor(B, dtype=F64)
    lin, beta, kappa, t, delta, gamma = p.tensors(tab.V.shape[1])
    with torch.no_grad():
        l_M = strengths(tab, beta, delta, t)
        d = distance(tab, A, B, context(tab, A), context(tab, B), l_M, kappa, gamma)
        return (lin[0] * d + lin[1]).numpy()


def fit(tab: MoleculeTable, A, B, y, lam: float = LAM, max_iter: int = MAX_ITER) -> ReadoutParams:
    """Fit the 133 readout numbers on rated pairs (rows of A, B) with ratings y."""
    A, B = torch.as_tensor(A, dtype=F64), torch.as_tensor(B, dtype=F64); y = torch.as_tensor(np.asarray(y, float))
    CA, CB = context(tab, A), context(tab, B); D = tab.V.shape[1]
    zero = lambda *s: torch.zeros(*s, dtype=F64, requires_grad=True)  # noqa: E731
    beta, kappa, t, delta, gamma = zero(()), zero(()), zero(()), zero(D), zero(D)
    with torch.no_grad():
        d = distance(tab, A, B, CA, CB, strengths(tab, beta, delta, t), kappa, gamma)
        lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
    lin.requires_grad_(True)
    opt = torch.optim.LBFGS([lin, beta, kappa, t, delta, gamma], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        d = distance(tab, A, B, CA, CB, strengths(tab, beta, delta, t), kappa, gamma)
        loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + lam * (beta ** 2 + kappa ** 2 + (delta ** 2).sum() + (gamma ** 2).sum())
        loss.backward(); return loss
    opt.step(closure)
    return ReadoutParams(lin[0].item(), lin[1].item(), beta.item(), kappa.item(), t.item(),
                         delta.detach().numpy().copy(), gamma.detach().numpy().copy())
