# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   T = molecules in the table     W = odour words (= receptors in O1.0), 138
#   N = pairs in a batch           K = molecules in one mixture (padded)
# ---------------------------------------------------------------------------------------------
"""O1.0: competition on open-POM's frozen odour words (design/o1_design.html, section O1.0).

    e(m, w)   = activation of word-receptor w by molecule m relative to its resting level: open-POM's
                probability for word w after B1a's per-word quantile step (standard normal over the
                molecule table; frozen, no labels used). Negative = below rest. Same input as B1a.
    s_w       = exp(rho_w), binding strength of word-receptor w, the same for every molecule (learned)
    r_w(A)    = s_w * sum_{m in A} e(m, w) / (1 + n_A * s_w)       competition for the receptor's sites
    u(A)      = r(A) / |r(A)|                                        (length-1 step, as in B1a; optional)
    d(A, B)   = alpha * sqrt(sum_w v_w^2 (u_w(A) - u_w(B))^2) + beta,  v_w = exp(nu_w) (learned)

O1.1 (occupancy=True): binding depends on the molecule. q(m, w) in [0, 1] is molecule m's rank for
word w over the molecule table (uniform quantile of p), so only molecules that carry a word occupy
its receptor; the signed activation e stays in the numerator (divisive normalisation by occupancy):
    r_w(A)    = sum_{m in A} e(m, w) / (1 + s_w * sum_{m in A} q(m, w)),   s_w = relu(rho_w) >= 0
At s_w = 0 this is the sum of e, and after the length-1 step exactly B1a. Run without v_w.
In O1.0 every molecule binds equally, so r_w(A) = g(w, n_A) * sum e: a per-word gain set by the
mixture size, not competition between molecules (T46 mixtures all have 10 molecules).

O1.2 (dominance=True): within a mixture, the molecules strongest on a word dominate that word
(humans pick out a few key odorants; a single molecule is a 1-molecule mixture, same rule):
    r_w(A)    = sum_{m in A} a(m, w) e(m, w),   a(., w) = softmax over m in A of tau_w * e(m, w)
tau_w = relu(rho_w) >= 0. tau_w = 0 is the plain mean (B1a); large tau_w approaches the max. No v_w.

Switches: competition=False uses the plain mean instead of r (B1a's mixing); weights=False keeps
v_w = 1; unit=False skips the length-1 step. With the length-1 step, one s shared by every word
only rescales r(A) and cancels, so at the start (all s_w = 1, v_w = 1) the model is B1a exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixture import quantile_per_word
from .o1 import pad


def efficacy_table(probs_TxW: np.ndarray) -> np.ndarray:
    """B1a's per-word quantile step (mixdist.mixture.quantile_per_word). A [0, 1] (uniform) version
    was tried in a smoke test and made the starting point much worse than B1a (1 x 5-fold CV r 0.34
    vs 0.48), which would confound the test of the mixing rule."""
    return quantile_per_word(probs_TxW)


def occupancy_table(probs_TxW: np.ndarray) -> np.ndarray:
    """q(m, w) in [0, 1]: rank of molecule m for word w over the molecule table (no labels used)."""
    from sklearn.preprocessing import QuantileTransformer
    return QuantileTransformer(output_distribution="uniform", n_quantiles=100).fit_transform(probs_TxW)


class O10(nn.Module):
    def __init__(self, n_words: int, competition: bool = True, weights: bool = True, unit: bool = True,
                 occupancy: bool = False, dominance: bool = False):
        super().__init__()
        self.competition, self.weights, self.unit, self.occupancy = competition, weights, unit, occupancy
        self.dominance = dominance
        rho0 = 1e-3 if (occupancy or dominance) else 0.0                    # O1.1: s_w = relu(rho) ~ 0 (B1a); O1.0: s_w = exp(0) = 1
        self.rho = nn.Parameter(torch.full((n_words,), rho0), requires_grad=competition)
        self.nu = nn.Parameter(torch.zeros(n_words), requires_grad=weights)        # v_w = exp(nu_w) = 1
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def binding_strength(self):
        return torch.relu(self.rho) if (self.occupancy or self.dominance) else self.rho.exp()

    def receptors(self, e_TxW, idx_NxK, mask_NxK, q_TxW=None):
        m = mask_NxK.unsqueeze(-1).float()
        sum_NxW = (e_TxW[idx_NxK] * m).sum(1)
        n_Nx1 = m.sum(1)
        if self.dominance:
            e_NxKxW = e_TxW[idx_NxK]
            logit = (self.binding_strength() * e_NxKxW).masked_fill(~mask_NxK.unsqueeze(-1), float("-inf"))
            return (torch.softmax(logit, dim=1) * e_NxKxW).sum(1)
        if self.occupancy:
            occ_NxW = (q_TxW[idx_NxK] * m).sum(1)
            return sum_NxW / (1 + self.binding_strength() * occ_NxW)
        if not self.competition:
            return sum_NxW / n_Nx1                                   # plain mean
        s_W = self.rho.exp()
        return s_W * sum_NxW / (1 + n_Nx1 * s_W)

    def raw_distance(self, rA_NxW, rB_NxW):
        if self.unit:
            rA_NxW, rB_NxW = F.normalize(rA_NxW, dim=1), F.normalize(rB_NxW, dim=1)
        diff = (rA_NxW - rB_NxW) * self.nu.exp()
        return (diff.pow(2).sum(1) + 1e-12).sqrt()

    def forward(self, rA_NxW, rB_NxW):
        return self.alpha * self.raw_distance(rA_NxW, rB_NxW) + self.beta


@dataclass
class Settings10:
    steps: int = 1000
    lr: float = 0.03
    log_every: int = 100
    competition: bool = True
    weights: bool = True
    unit: bool = True
    occupancy: bool = False         # O1.1
    dominance: bool = False         # O1.2


class Trainer10:
    """Full-batch training of O1.0 on DREAM pairs (squared error). alpha, beta start at the
    least-squares line on the starting distances, so step 0 is B1a with its fitted line."""

    def __init__(self, cids_T, probs_TxW, pairs, s: Settings10, device: str = "cpu", progress=None):
        self.s, self.dev, self.progress, self.tag = s, device, progress, ""
        self.row_of_cid = {int(c): i for i, c in enumerate(cids_T)}
        self.e = torch.tensor(efficacy_table(probs_TxW), dtype=torch.float32, device=device)
        self.q = torch.tensor(occupancy_table(probs_TxW), dtype=torch.float32, device=device)
        self.y = torch.tensor([p.y for p in pairs], dtype=torch.float32, device=device)
        self.A, self.B = self._mix(pairs)

    def _mix(self, pairs):
        A = pad([[self.row_of_cid[c] for c in sorted(p.a)] for p in pairs], self.dev)
        B = pad([[self.row_of_cid[c] for c in sorted(p.b)] for p in pairs], self.dev)
        return A, B

    def new_model(self):
        s = self.s
        return O10(self.e.shape[1], s.competition, s.weights, s.unit, s.occupancy, s.dominance).to(self.dev)

    def _r(self, model, mix):
        return model.receptors(self.e, *mix, q_TxW=self.q)

    def _raw(self, model, A, B):
        return model.raw_distance(self._r(model, A), self._r(model, B))

    def fit(self, train_idx):
        s, model, log = self.s, self.new_model(), []
        tr = torch.as_tensor(np.asarray(train_idx), device=self.dev)
        A, B, y = (self.A[0][tr], self.A[1][tr]), (self.B[0][tr], self.B[1][tr]), self.y[tr]
        with torch.no_grad():
            a, c = np.polyfit(self._raw(model, A, B).cpu().numpy(), y.cpu().numpy(), 1)
            model.alpha.fill_(float(a)); model.beta.fill_(float(c))
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=s.lr)
        for step in range(0, s.steps + 1):
            pred = model.alpha * self._raw(model, A, B) + model.beta
            loss = F.mse_loss(pred, y)
            if step % s.log_every == 0 or step == s.steps:
                p, yv = pred.detach().cpu().numpy(), y.cpu().numpy()
                sw, vw = model.binding_strength().detach().cpu().numpy(), model.nu.detach().exp().cpu().numpy()
                entry = {"step": step, "train_rmse": float(np.sqrt(((p - yv) ** 2).mean())),
                         "train_r": float(np.corrcoef(p, yv)[0, 1]),
                         "s_min": float(sw.min()), "s_med": float(np.median(sw)), "s_max": float(sw.max()),
                         "v_min": float(vw.min()), "v_med": float(np.median(vw)), "v_max": float(vw.max())}
                log.append(entry)
                if self.progress:
                    self.progress(f"[{self.tag}] " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()))
            if step == s.steps:
                break
            opt.zero_grad(); loss.backward(); opt.step()
        return model, log

    @torch.no_grad()
    def predict(self, model, idx):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev)
        A, B = (self.A[0][idx], self.A[1][idx]), (self.B[0][idx], self.B[1][idx])
        return model(self._r(model, A), self._r(model, B)).cpu().numpy()

    @torch.no_grad()
    def predict_pairs(self, model, pairs) -> np.ndarray:
        A, B = self._mix(pairs)
        return model(self._r(model, A), self._r(model, B)).cpu().numpy()
