# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   Q = GoodScents pairs   T = Ma mixtures
#   P = DREAM development pairs   M = DREAM molecules (235)   H = middle layer (400)   W = words (138)   G = groups (24)
# ---------------------------------------------------------------------------------------------
"""O3-T: losses on BOTH the latent template and the decoded profiles, all real data (the user, 2026-11-19), as set-based
face recognition (OGCTL, arXiv 1903.04752: the summed template is judged by an angular loss, never decoded).

Latent v = O2-C middle layer (centred, 400). Template of a set: z = sum_m w_m v_m / sum_m w_m, w = exp(theta . x).
Angle head (latent word decoder, linear-angular): score_w(z) = s cos(z, c_w) + b_w, c_w in R^400, s learned.

Losses, added one rung at a time (--rung):
  decoded (all rungs)  D1 singles words through O2-C's decoder (open-POM loss)
                       D2 GoodScents: decode each molecule alone, w-weighted average, rank loss over 79 notes / tau
                       D3 Ma sign: P(A dominates) = sigmoid(theta . (x_A - x_B)), cross-entropy vs [L > 0] (no free scale)
  rung 1  + L1 angle loss, singles: the angle head on v, open-POM loss
          + L2 angle loss, GoodScents: the angle head on the blend template z, rank loss (same head as singles)
  rung 2  + L3 center loss: mean_m 1 - cos(v_m, z) on GoodScents blends
  rung 3  + L4 Ma set vs single: softplus(-sign(L) (cos(z, v_A) - cos(z, v_B)) / 0.1) (fixed temperature, no free scale)
  --scent-rel (with any rung): GoodScents "which ingredient does the blend lean to", the Ma idea on GoodScents.
            Label from real notes only: sign(cos(Y_blend, Y_A) - cos(Y_blend, Y_B)) on the 79 notes, Y_A = A's own GS-LF
            words; pairs where A or B has no GS-LF words or the gap is < 0.05 are skipped (no label made up).
            D6 = cross-entropy of sigmoid(theta . (x_A - x_B)) vs the label (as D3);
            L6 = softplus(-sign (cos(z, v_A) - cos(z, v_B)) / 0.1) on the blend template z (as L4).
  --no-d2: drop D2 from training and from early stopping (the user's rule, 2026-11-19: no merging at the word level;
            merge in the hidden layer only). D2 is still computed for the report. Needs rung >= 1 (L2 carries GoodScents).
  rung 4  + L5 DREAM in-fold training pairs, as COMPARISONS (face verification), not values: template distance
            d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2) (u = z / |z|, h = log |z|).
            --dream-loss anchored (default): two training pairs sharing a mixture, (A, B) and (A, C); people rated one
              farther; A's template must be farther from that one: softplus(-sign(y_AB - y_AC) (d_AB - d_AC) / 0.05),
              pairs with |y_AB - y_AC| < 0.02 skipped. Both ratings share mixture A, so they come from the same
              experiment; no study label is used.
            --dream-loss global: the same over random pairs of training pairs (any experiments).
            --dream-loss mse: a d + c fitted to the values (the earlier value loss), MSE / var(y).
            Trained per fold of 5 x 1 (train_o2d_dev's folds); test predictions = the fold's line (fitted on its
            training pairs) on d.
Mixture-side numbers: theta 24 (shared by every loss) + tau 1 (D2) + beta (rung 4; + a, c for mse; the line for RMSE is
fitted on training pairs at the end) = 26-28.
Guards: balanced GoodScents exposure; early stopping on the sum of held-out singles (10% of training singles: D1 + L1),
held-out GoodScents molecules (D2 + L2 on their pairs) and, rung 4, 1 - r on an inner mixture-disjoint 10% of the fold's
training pairs; Ma fold-0 mixtures never trained on; single-molecule
BatchNorm statistics for all set passes.

Env:     mist (GPU)
Usage:   python training/train_o3t.py --rung 1 --seeds 0 1 2
         python training/train_o3t.py --rung 4 --fold 0 1 2 3 4 --seeds 0 1
Writes:  checkpoints/o3t_r<rung>/member_<seed>.pt (rungs 1-3), checkpoints/o3t_r4/fold<k>_seed<s>.pt,
         results/train_o3t/<date_time>_r<rung>[_fold<k>].{txt,json}
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, evaluate, features, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o2j import word_auroc  # noqa: E402
from train_o3s_scent import MA_HOLDOUT_FOLD, middle, split_molecules, split_singles  # noqa: E402

F32 = torch.float32
cos = nn.functional.cosine_similarity


class AngleHead(nn.Module):
    """score_w(z) = s cos(z, c_w) + b_w: magnitude-invariant, so a template of 2 or 40 molecules is read the same way."""

    def __init__(self, dim=400, n_words=138):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_words, dim) / math.sqrt(dim)); self.b = nn.Parameter(torch.zeros(n_words))
        self.log_s = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, z):
        return torch.exp(self.log_s) * nn.functional.normalize(z, dim=1) @ nn.functional.normalize(self.c, dim=1).T + self.b


def template(v, member, w):
    """member: K x n 0/1; w: n -> K x H."""
    W = member * w.unsqueeze(0)
    return (W @ v) / W.sum(1, keepdim=True)


def scent_rel_labels(G, S, gap=0.05, measure="cosine"):
    """Per GoodScents pair: 1 (blend leans to A), 0 (to B), nan (no label). From real notes only.
    measure "cosine": cos(Y_blend, Y_A) - cos(Y_blend, Y_B), gap 0.05 (slightly favours the ingredient with fewer notes,
    rank correlation -0.12 with the note-count difference); "excess": shared notes above chance,
    (|Y_blend & Y_A| - n_blend n_A / 79) - (same for B), |difference| >= 1 note (count-neutral: correlation -0.005)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def can(x):
        m_ = Chem.MolFromSmiles(str(x)); return Chem.MolToSmiles(m_) if m_ is not None else None
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    gs = {c: i for i, c in ((i, can(x)) for i, x in enumerate(G.smiles_N[:G.n_gslf])) if c is not None}
    row_N = np.array([gs.get(can(x), -1) for x in z["smiles"][:S.n_odorpair]])
    ra, rb = row_N[S.pair_a], row_N[S.pair_b]; ok = (ra >= 0) & (rb >= 0)
    Ys = G.Y[:, S.targets].astype(float); YQ = S.pair_Y[:, S.targets].astype(float)

    def c_(u, v):
        return (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-9)
    d = np.full(len(ra), np.nan)
    if measure == "excess":
        nT = len(S.targets); A_, B_, Q_ = Ys[ra[ok]], Ys[rb[ok]], YQ[ok]; nq = Q_.sum(1)
        d[ok] = ((Q_ * A_).sum(1) - nq * A_.sum(1) / nT) - ((Q_ * B_).sum(1) - nq * B_.sum(1) / nT); gap = 1.0
    else:
        d[ok] = c_(YQ[ok], Ys[ra[ok]]) - c_(YQ[ok], Ys[rb[ok]])
    lab = np.where(np.abs(d) >= gap, (d > 0).astype(float), np.nan); lab[~ok] = np.nan
    return lab


class Data:
    def __init__(self, dev, scent_rel=False):
        self.G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); self.S = o2j.Store(features.features_dir() / "o2j_data.npz")
        z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
        op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
        Xd, self.gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
        self.XN = torch.as_tensor(o3.group_matrix(z["smiles"], self.gn), dtype=F32, device=dev)
        self.XM = torch.as_tensor(Xd, dtype=F32, device=dev)
        gslf = np.arange(self.G.n_gslf); self.single_tr, self.single_te = gslf[~self.G.audit_test], gslf[self.G.audit_test]
        self.P = dev_pairs(); self.y = evaluate.labels(self.P); A, B = membership(self.P, op.cids_M)
        self.A = torch.as_tensor(A, dtype=F32, device=dev); self.B = torch.as_tensor(B, dtype=F32, device=dev)
        self.folds = list(cv.mixture_holdout_folds(self.P, k=5, repeats=1, seed=0))
        self.Y = torch.as_tensor(self.G.Y, device=dev); self.PY = torch.as_tensor(self.S.pair_Y, dtype=F32, device=dev)
        self.T = torch.as_tensor(self.S.targets, device=dev); self.ma_y = torch.as_tensor(self.S.ma_L > 0, dtype=F32, device=dev)
        self.te_q = np.where(self.S.pair_split == 1)[0]; self.ma_tr, self.ma_te = self.S.ma_split(MA_HOLDOUT_FOLD)
        self.rel = None
        if scent_rel:
            self.rel = torch.as_tensor(scent_rel_labels(self.G, self.S), dtype=F32, device=dev)


class O3T:
    def __init__(self, D, rung, dev):
        self.D, self.rung, self.dev = D, rung, dev
        self.net = o2.O2C().float().to(dev); self.head = AngleHead().float().to(dev)
        self.theta = torch.zeros(len(D.gn), dtype=F32, device=dev, requires_grad=True)
        self.log_tau = torch.zeros((), dtype=F32, device=dev, requires_grad=True)
        self.dream = torch.tensor([0.5, 1.0, 0.0], dtype=F32, device=dev, requires_grad=True)   # b (beta = b^2), a, c

    def params(self):
        return list(self.net.parameters()) + list(self.head.parameters()) + [self.theta, self.log_tau, self.dream]

    def blends(self, q):
        """GoodScents pairs -> dict of losses / logits (per pair)."""
        D, S = self.D, self.D.S; a, b = S.pair_a[q], S.pair_b[q]
        mols, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
        v = middle(self.net, S, mols, self.dev); ia, ib = inv[:len(q)], inv[len(q):]
        w = torch.sigmoid((D.XN[mols[ia]] - D.XN[mols[ib]]) @ self.theta).unsqueeze(1)
        with o2j.single_stats(self.net):
            p = torch.sigmoid(self.net.readout(v)[1])
        lg_dec = torch.logit((w * p[ia] + (1 - w) * p[ib]).clamp(1e-6, 1 - 1e-6))
        out = {"D2": o3.rank_loss(lg_dec / torch.exp(self.log_tau), D.PY[q], D.T), "lg_dec": lg_dec}
        if self.rung >= 1:
            z = w * v[ia] + (1 - w) * v[ib]; lg_ang = self.head(z)
            out["L2"] = o3.rank_loss(lg_ang, D.PY[q], D.T); out["lg_ang"] = lg_ang
            if self.rung >= 2:
                out["L3"] = 0.5 * ((1 - cos(v[ia], z, dim=1)) + (1 - cos(v[ib], z, dim=1)))
        if D.rel is not None:
            lab = D.rel[torch.as_tensor(q, device=self.dev)]; k = ~torch.isnan(lab)
            if k.any():
                zr = w * v[ia] + (1 - w) * v[ib]; d = (cos(zr, v[ia], dim=1) - cos(zr, v[ib], dim=1))[k]; s_ = 2 * lab[k] - 1
                out["D6"] = nn.functional.binary_cross_entropy_with_logits(((D.XN[mols[ia]] - D.XN[mols[ib]]) @ self.theta)[k], lab[k], reduction="none")
                out["L6"] = nn.functional.softplus(-s_ * d / 0.1); out["x_rel_d"] = d; out["x_rel_y"] = lab[k]
                out["x_rel_t"] = ((D.XN[mols[ia]] - D.XN[mols[ib]]) @ self.theta)[k]
        return out

    def ma_sets(self, t):
        D, S = self.D, self.D.S; mols, inv = np.unique(np.r_[S.ma_a[t], S.ma_b[t]], return_inverse=True)
        v = middle(self.net, S, mols, self.dev); ia, ib = inv[:len(t)], inv[len(t):]
        wa, wb = torch.exp(D.XN[mols[ia]] @ self.theta).unsqueeze(1), torch.exp(D.XN[mols[ib]] @ self.theta).unsqueeze(1)
        z = (wa * v[ia] + wb * v[ib]) / (wa + wb)
        d = cos(z, v[ia], dim=1) - cos(z, v[ib], dim=1)
        return nn.functional.softplus(-(2 * D.ma_y[t] - 1) * d / 0.1)

    def dream_pred(self, idx):
        D = self.D; v = middle(self.net, D.G, D.G.dream_rows(), self.dev); w = torch.exp(D.XM @ self.theta)
        zA, zB = template(v, D.A[idx], w), template(v, D.B[idx], w)
        du = ((nn.functional.normalize(zA, dim=1) - nn.functional.normalize(zB, dim=1)) ** 2).sum(1)
        dh = (torch.log(zA.norm(dim=1)) - torch.log(zB.norm(dim=1))) ** 2
        b, a, c = self.dream
        return a * torch.sqrt(du + b ** 2 * dh + 1e-12) + c


def dream_dist(m, idx):
    D = m.D; v = middle(m.net, D.G, D.G.dream_rows(), m.dev); w = torch.exp(D.XM @ m.theta)
    zA, zB = template(v, D.A[idx], w), template(v, D.B[idx], w)
    du = ((nn.functional.normalize(zA, dim=1) - nn.functional.normalize(zB, dim=1)) ** 2).sum(1)
    dh = (torch.log(zA.norm(dim=1)) - torch.log(zB.norm(dim=1))) ** 2
    return torch.sqrt(du + m.dream[0] ** 2 * dh + 1e-12)


def anchor_pairs(P, idx):
    """Training pairs grouped by a shared mixture: list of arrays (pair indices with that mixture), each of length >= 2."""
    by = {}
    for i in idx:
        by.setdefault(P[i].a, []).append(i); by.setdefault(P[i].b, []).append(i)
    return [np.array(v) for v in by.values() if len(v) >= 2]


def dream_rank_loss(m, y_t, groups, rng, n, mode):
    if mode == "anchored":
        g = [groups[j] for j in rng.integers(0, len(groups), n)]; ij = np.array([rng.choice(x, 2, replace=False) for x in g])
    else:
        allp = np.concatenate(groups); ij = rng.choice(allp, (n, 2))
    i, j = ij[:, 0], ij[:, 1]; dy = y_t[i] - y_t[j]; keep = dy.abs() >= 0.02
    if not keep.any():
        return torch.zeros((), device=y_t.device)
    d = dream_dist(m, np.r_[i, j]); di, dj = d[:len(i)], d[len(i):]
    return nn.functional.softplus(-torch.sign(dy[keep]) * (di - dj)[keep] / 0.05).mean()


def train(D, rung, seed, dev, fold=None, recipe=o2.RECIPE, pair_batch=128, ma_batch=32, dream_batch=64, log=print, dream_loss="anchored", no_d2=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = O3T(D, rung, dev); S = D.S
    counts = D.G.Y[D.single_tr].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=F32, device=dev)
    tr_q, val_q = split_molecules(S, seed); val_q = val_q[np.random.default_rng(seed).permutation(len(val_q))[:2000]]
    single_tr, s_va = split_singles(D.single_tr, seed)
    by_mol = {}
    for q in tr_q:
        by_mol.setdefault(int(S.pair_a[q]), []).append(q); by_mol.setdefault(int(S.pair_b[q]), []).append(q)
    mol_list = np.array(sorted(by_mol))
    if fold is not None:
        ftr = np.array(D.folds[fold].train); sub = [D.P[i] for i in ftr]
        inner = cv.mixture_holdout_folds(sub, k=10, repeats=1, seed=seed)[0]
        d_tr, d_va = ftr[np.array(inner.train)], ftr[np.array(inner.test)]
        y_t = torch.as_tensor(D.y, dtype=F32, device=dev); var_y = float(D.y[d_tr].var())
        groups = anchor_pairs(D.P, d_tr) if dream_loss != "mse" else None
        log(f"  fold {fold}: DREAM training pairs {len(d_tr)}, inner validation {len(d_va)} (mixture-disjoint), test {len(D.folds[fold].test)}; "
            f"loss {dream_loss}" + (f", anchor mixtures with >= 2 training pairs {len(groups)}" if groups is not None else ""))
    opt = torch.optim.Adam(m.params(), lr=recipe["lr"])
    bs = recipe["batch"]; nb = math.ceil(len(single_tr) / bs); step = 0; best = (np.inf, -1, None)
    for epoch in range(recipe["epochs"]):
        m.net.train(); perm = rng.permutation(single_tr)
        perm = np.concatenate([perm, rng.choice(single_tr, nb * bs - len(perm), replace=False)]); tot = {}
        for k in range(nb):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]; x, e, s_, d_, g = D.G.batch(b, dev)
            v = m.net.latent(m.net.encode(x, e, s_, d_, g, int(g.max()) + 1))[0]
            L = {"D1": o2.pom_loss(m.net.readout(v)[1], D.Y[b], class_w)}
            if rung >= 1:
                L["L1"] = o2.pom_loss(m.head(v), D.Y[b], class_w)
            q = np.array([rng.choice(by_mol[int(mm)]) for mm in rng.choice(mol_list, pair_batch, replace=False)])
            for kk, vv in m.blends(q).items():
                if not kk.startswith(("lg", "x_")) and not (no_d2 and kk == "D2"):
                    L[kk] = vv.mean()
            t = rng.choice(D.ma_tr, ma_batch, replace=False)
            L["D3"] = nn.functional.binary_cross_entropy_with_logits((D.XN[S.ma_a[t]] - D.XN[S.ma_b[t]]) @ m.theta, D.ma_y[t])
            if rung >= 3:
                L["L4"] = m.ma_sets(t).mean()
            if fold is not None:
                if dream_loss == "mse":
                    dq = rng.choice(d_tr, dream_batch, replace=False); L["L5"] = ((m.dream_pred(dq) - y_t[dq]) ** 2).mean() / var_y
                else:
                    L["L5"] = dream_rank_loss(m, y_t, groups, rng, dream_batch, dream_loss)
            loss = sum(L.values()) + o2.regularization(m.net, recipe["weight_decay"]) + 1e-3 * (m.theta ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            for kk, vv in L.items():
                tot[kk] = tot.get(kk, 0.0) + vv.item() / nb
        m.net.eval()
        with torch.no_grad():
            vs = 0.0
            for i in range(0, len(s_va), 256):
                bb = s_va[i:i + 256]; vv_ = middle(m.net, D.G, bb, dev)
                with o2j.single_stats(m.net):
                    vs += o2.pom_loss(m.net.readout(vv_)[1], D.Y[bb], class_w).item() * len(bb) / len(s_va)
                if rung >= 1:
                    vs += o2.pom_loss(m.head(vv_), D.Y[bb], class_w).item() * len(bb) / len(s_va)
            vl = 0.0
            for i in range(0, len(val_q), 500):
                o = m.blends(val_q[i:i + 500]); vl += ((0 if no_d2 else o["D2"].sum()) + (o["L2"].sum() if "L2" in o else 0)).item() / len(val_q)
            vd = (1.0 - float(np.corrcoef(dream_dist(m, d_va).cpu().numpy(), D.y[d_va])[0, 1])) if fold is not None else 0.0   # 1 - r, scale-free
        crit = vs + vl + vd
        if crit < best[0]:
            best = (crit, epoch, ({k_: x_.detach().clone() for k_, x_ in m.net.state_dict().items()},
                                  {k_: x_.detach().clone() for k_, x_ in m.head.state_dict().items()},
                                  m.theta.detach().clone(), m.log_tau.detach().clone(), m.dream.detach().clone()))
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            log(f"  rung {rung} seed {seed}{'' if fold is None else f' fold {fold}'} epoch {epoch + 1}: "
                + ", ".join(f"{kk} {vv:.4f}" for kk, vv in tot.items()) + f"; validation singles {vs:.4f}, GoodScents {vl:.4f}"
                + ("" if fold is None else f", DREAM inner 1 - r {vd:.4f}") + f" (best epoch {best[1] + 1})")
    sd, hd, th, lt, dr = best[2]; m.net.load_state_dict(sd); m.head.load_state_dict(hd)
    with torch.no_grad():
        m.theta.copy_(th); m.log_tau.copy_(lt); m.dream.copy_(dr)
    m.net.eval()
    return m, best[1] + 1, tr_q


@torch.no_grad()
def checks(m, tr_q, seed):
    D, S = m.D, m.D.S; out = {}
    out["single_auroc"] = o2.mean_auroc(o2.predict_probs(m.net, D.G, D.single_te, m.dev), D.G.Y[D.single_te])
    if m.rung >= 1:
        v = middle(m.net, D.G, D.single_te, m.dev); out["single_auroc_angle"] = o2.mean_auroc(torch.sigmoid(m.head(v)).cpu().numpy(), D.G.Y[D.single_te])
    for name, qs in (("heldout", D.te_q), ("train", tr_q[np.random.default_rng(seed).permutation(len(tr_q))[:2000]])):
        acc = {}
        for i in range(0, len(qs), 500):
            o = m.blends(qs[i:i + 500])
            for kk in o:
                acc.setdefault(kk, []).append(o[kk])
        if "x_rel_d" in acc and name == "heldout":
            yy = torch.cat(acc["x_rel_y"]).cpu().numpy(); out["scent_rel_heldout_n"] = float(len(yy))
            out["scent_rel_auroc_sets"] = roc_auc_score(yy, torch.cat(acc["x_rel_d"]).cpu().numpy())
            out["scent_rel_auroc_theta"] = roc_auc_score(yy, torch.cat(acc["x_rel_t"]).cpu().numpy())
        for kk, vv in acc.items():
            vv = torch.cat(vv)
            if kk.startswith("x_"):
                continue
            if kk.startswith("lg"):
                if name == "heldout":
                    out[f"scent_auroc_{kk[3:]}"] = word_auroc(vv.cpu().numpy(), S.pair_Y[qs], S.targets)[0]
            else:
                out[f"scent_{kk}_{name}"] = vv.mean().item()
    d = ((D.XN[S.ma_a[D.ma_te]] - D.XN[S.ma_b[D.ma_te]]) @ m.theta).cpu().numpy()
    out["ma_heldout_auroc"] = roc_auc_score(S.ma_L[D.ma_te] > 0, d) if d.std() > 0 else float("nan")
    if m.rung >= 3:
        out["ma_heldout_auroc_sets"] = roc_auc_score(S.ma_L[D.ma_te] > 0, -m.ma_sets(D.ma_te).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rung", type=int, choices=[0, 1, 2, 3, 4], required=True)
    ap.add_argument("--dream-loss", choices=["anchored", "global", "mse"], default="anchored")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--fold", type=int, nargs="*", default=[])
    ap.add_argument("--no-d2", action="store_true", help="no word-level merge in training (drop D2)")
    ap.add_argument("--scent-rel", action="store_true", help="add D6 + L6 (GoodScents: which ingredient the blend leans to)")
    a = ap.parse_args()
    if a.no_d2 and a.rung < 1:
        raise SystemExit("--no-d2 needs rung >= 1 (L2 is then the only GoodScents loss)")
    if (a.rung == 4) != bool(a.fold):
        raise SystemExit("rung 4 (DREAM in training) needs --fold; rungs 0-3 take no --fold")
    dev = pick_device(); tag = f"r{a.rung}" + ("_rel" if a.scent_rel else "") + ("_nod2" if a.no_d2 else "") + (f"_{a.dream_loss}_fold{'-'.join(map(str, a.fold))}" if a.fold else "")
    rep = Report("train_o3t", {**vars(a), "device": dev, **o2.RECIPE}, tag=tag)
    D = Data(dev, a.scent_rel); ck = ROOT / (f"checkpoints/o3t_r{a.rung}" + ("_rel" if a.scent_rel else "") + ("_nod2" if a.no_d2 else "") + (f"_{a.dream_loss}" if a.rung == 4 else "")); ck.mkdir(parents=True, exist_ok=True)
    rep.line(f"rung {a.rung}; mixture-side numbers: theta {len(D.gn)} + tau 1" + (" + beta, a, c" if a.rung == 4 else "")
             + f"; Ma held-out fold {MA_HOLDOUT_FOLD} ({len(D.ma_te)} mixtures); GoodScents held-out pairs {len(D.te_q)}")
    rows = []
    for fold in (a.fold or [None]):
        for s in a.seeds:
            path = ck / (f"member_{s}.pt" if fold is None else f"fold{fold}_seed{s}.pt")
            if path.exists():
                rep.line(f"{path.name}: exists, not retrained"); continue
            t0 = time.time(); m, ep, tr_q = train(D, a.rung, s, dev, fold, log=rep.line, dream_loss=a.dream_loss, no_d2=a.no_d2); c = checks(m, tr_q, s)
            extra = {}
            if fold is not None:
                with torch.no_grad():
                    te, ftr = np.array(D.folds[fold].test), np.array(D.folds[fold].train)
                    d_te, d_tr_ = dream_dist(m, te).cpu().numpy(), dream_dist(m, ftr).cpu().numpy()
                    a_, c_ = evaluate.fit_line(d_tr_, D.y[ftr])
                    extra = {"fold": fold, "test_idx": te, "test_dist": d_te, "test_pred": a_ * d_te + c_,
                             "train_r": float(np.corrcoef(d_tr_, D.y[ftr])[0, 1]), "test_r": float(np.corrcoef(d_te, D.y[te])[0, 1])}
                    rep.line(f"fold {fold} seed {s}: DREAM template r training pairs {extra['train_r']:.3f}, test pairs {extra['test_r']:.3f}")
            torch.save({"model_state_dict": m.net.state_dict(), "head_state_dict": m.head.state_dict(), "arch": "o2c", "seed": s, "rung": a.rung,
                        "theta": m.theta.detach().cpu().numpy(), "tau": float(torch.exp(m.log_tau.detach())), "dream_head": m.dream.detach().cpu().numpy(), "groups": D.gn,
                        "best_epoch": ep, "scent_auroc": c.get("scent_auroc_dec"), "scent_loss": c.get("scent_D2_heldout"), **c, **extra}, path)
            rows.append([path.name, ep] + [f"{c[k]:.4f}" if k in c else "" for k in
                         ("single_auroc", "single_auroc_angle", "scent_auroc_dec", "scent_auroc_ang", "scent_D2_heldout", "scent_D2_train",
                          "scent_L2_heldout", "scent_L3_heldout", "ma_heldout_auroc", "ma_heldout_auroc_sets", "scent_rel_auroc_theta", "scent_rel_auroc_sets")])
            rep.line(f"{path.name}: {time.time() - t0:.0f} s; " + ", ".join(f"{k} {v:.4f}" for k, v in c.items()))
    rep.section(f"rung {a.rung}: molecule-level checks (data never trained on)")
    rep.table(["member", "best epoch", "single AUROC (decoder)", "single AUROC (angle head)", "GoodScents held-out word AUROC (decoded)",
               "GoodScents held-out word AUROC (angle head on template)", "GoodScents held-out D2 loss", "D2 training loss",
               "L2 held-out", "center held-out", "Ma held-out AUROC (theta)", "Ma held-out AUROC (set vs single)",
               "GoodScents lean-to held-out AUROC (theta)", "GoodScents lean-to held-out AUROC (set vs single)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
