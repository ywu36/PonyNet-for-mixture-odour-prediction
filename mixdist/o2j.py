# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   Q = GoodScents pairs   T = Ma mixtures
#   W = words (138)   H = encoder output (400)   D = merge-space size (O2-N 400, VAE 32)
# ---------------------------------------------------------------------------------------------
"""O2-J (design/o2j_design.md): joint training of a backbone (O2-N or O2-VAE) on single molecules + GoodScents
blender pairs (learning to rank) + Ma dominance (Siamese), with a gain head G(m) = exp(linear(e(m)))."""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from . import o2
from .evaluate import fit_line


class Store(o2.Graphs):
    """features/o2j_data.npz (evaluation/O2/export_o2j_data.py): graphs of GoodScents + Ma molecules, labels."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        for k in ("node", "edge", "edge_index", "node_off", "edge_off", "pair_a", "pair_b", "pair_Y", "pair_split",
                  "targets", "ma_a", "ma_b", "ma_L", "ma_fold", "n_odorpair"):
            setattr(self, k, z[k])
        self.n_odorpair = int(self.n_odorpair)

    def ma_split(self, holdout_fold=None):
        """Ma mixture indices (train, test): test = either molecule in the held-out molecule fold."""
        if holdout_fold is None:
            return np.arange(len(self.ma_L)), np.array([], int)
        f = self.ma_fold
        fa, fb = f[self.ma_a - self.n_odorpair], f[self.ma_b - self.n_odorpair]
        test = (fa == holdout_fold) | (fb == holdout_fold)
        return np.where(~test)[0], np.where(test)[0]


@contextmanager
def single_stats(m):
    """Mixture passes use the BatchNorm statistics learned from single molecules (running mean / variance) and do not
    update them. Fix 2026-11-18: before, mixtures were normalised by their own batch during training but by the running
    statistics when scored, so the scored network was not the trained one (one member: inner-train r 0.97 with batch
    statistics vs 0.61 as scored, predictions 0.12 too low)."""
    bns = [b for b in m.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    was = [b.training for b in bns]
    for b in bns:
        b.eval()
    try:
        yield
    finally:
        for b, w in zip(bns, was):
            b.train(w)


def grad_dial(t, alpha):
    """Forward unchanged; backward passes alpha x the gradient (0 = detached, 1 = unchanged)."""
    return t if alpha == 1.0 else alpha * t + (1 - alpha) * t.detach()


class Joint(nn.Module):
    """white=True adds the agreement control (O2-N only): a merged mixture decodes its direction as usual, then its profile
    is pulled toward a learned "white" profile by lam(rho) = sigmoid(c) (1 - rho^gamma); one molecule has rho = 1, lam = 0,
    so single molecules are unchanged."""

    def __init__(self, arch="o2n", latent=32, white=False):
        super().__init__()
        self.arch, self.white = arch, white
        if white:
            self.white_logit = nn.Parameter(torch.zeros(o2.N_WORDS if hasattr(o2, "N_WORDS") else 138))
            self.white_c = nn.Parameter(torch.tensor(0.0)); self.white_g = nn.Parameter(torch.tensor(0.0))
        self.net = {"o2n": o2.O2N, "o2c": o2.O2C}[arch]() if arch in ("o2n", "o2c") else o2.O2VAE(latent)
        self.gain = nn.Linear(2 * (o2.NODE_OUT + o2.EDGE_OUT), 1)
        nn.init.zeros_(self.gain.weight); nn.init.zeros_(self.gain.bias)
        self.ma_line = nn.Parameter(torch.tensor([1.0, 0.0]))

    def molecules(self, store, ids, device, enc_grad=1.0):
        """-> (merge-space v B x D, gain logit B) for molecules ``ids`` of ``store``; enc_grad = gradient dial into the encoder."""
        x, e, s, d, g = store.batch(ids, device)
        enc = grad_dial(self.net.encode(x, e, s, d, g, len(ids)), enc_grad)
        v, _ = self.net.latent(enc)
        return v, self.gain(enc)[:, 0]

    def pull(self, logits, rho):
        """Agreement control (white=True): profile -> (1 - lam) profile + lam white, lam = sigmoid(c)(1 - rho^gamma)."""
        if not self.white:
            return logits
        lam = (torch.sigmoid(self.white_c) * (1 - rho.clamp(1e-6, 1) ** nn.functional.softplus(self.white_g + 0.5413))).unsqueeze(1)
        p = (1 - lam) * torch.sigmoid(logits) + lam * torch.sigmoid(self.white_logit).unsqueeze(0)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))

    def decode_merged(self, z):
        """Merged vector z (K x D) -> logits: O2-N decodes the direction; the agreement rho = |z| enters only via pull().
        O2-C (and the VAE) decode z as it is, length included."""
        if self.arch != "o2n":
            return self.net.readout(z)[1]
        rho = z.norm(dim=1)
        return self.pull(self.net.readout(self.net.scale * z / rho.clamp_min(1e-8).unsqueeze(1))[1], rho)

    def merged_logits(self, v1, v2, g1, g2):
        """Two-molecule merge z = (G1 v1 + G2 v2) / (G1 + G2), decoded -> (z, logits)."""
        w = torch.sigmoid(g1 - g2).unsqueeze(1)
        z = w * v1 + (1 - w) * v2
        return z, self.decode_merged(z)

    def similarity(self, z, v):
        return nn.functional.cosine_similarity(z, v, dim=1) if self.arch in ("o2n", "o2c") else -(z - v).norm(dim=1)


def rank_loss(logits_QxW, Y_QxW, targets):
    """Learning to rank over the target words T: -sum_w p_w log softmax_T(logits)_w, p = label counts normalised.
    Words outside T get no gradient; no word is forced to 0."""
    lt = logits_QxW[:, targets]; p = Y_QxW[:, targets]; p = p / p.sum(1, keepdim=True)
    return -(p * torch.log_softmax(lt, 1)).sum(1).mean()


def train(G: o2.Graphs, S: Store, single_ids, seed, device, arch="o2n", latent=32, beta_kl=1e-2, ma_holdout=None,
          weights=(1.0, 1.0, 1.0), pair_batch=128, ma_batch=32, log=print, recipe=o2.RECIPE,
          mix_grad=1.0, dream=None, dream_batch=64, dream_length=True, white=False, streams=("scent", "ma"), on_epoch=None):
    """Streams: single molecules (update everything) + GoodScents pairs + Ma (+ DREAM if ``dream`` = (DreamStream, y_P,
    train_idx)). mix_grad = fraction of every mixture stream's gradient that reaches the encoder (0 = the encoder learns
    from single molecules only; mixture streams update decoder, gain and heads). white: agreement control (Joint).
    streams: which of "scent", "ma" run."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Joint(arch, latent, white).float().to(device)
    if dream is not None:
        D, y_P, dtr = dream
        m.dream_line = nn.Parameter(torch.tensor([0.5, float(np.mean(y_P[dtr])) - 0.5, 0.3], device=device))
        m.dream_offset = nn.Parameter(torch.zeros(G.Y.shape[1], device=device))
        yP = torch.as_tensor(y_P, dtype=torch.float32, device=device); varD = float(np.var(y_P[dtr]))
    Y = torch.as_tensor(G.Y, device=device)
    counts = G.Y[single_ids].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=device)
    pairs = np.where(S.pair_split == 0)[0]; PY = torch.as_tensor(S.pair_Y, device=device); T = torch.as_tensor(S.targets, device=device)
    ma_tr, _ = S.ma_split(ma_holdout); L = torch.as_tensor(S.ma_L, dtype=torch.float32, device=device); varL = float(S.ma_L[ma_tr].var())
    opt = torch.optim.Adam(m.parameters(), lr=recipe["lr"], betas=(0.9, 0.999), eps=1e-8)
    bs = recipe["batch"]; n_batches = math.ceil(len(single_ids) / bs); step = 0
    for epoch in range(recipe["epochs"]):
        m.train(); perm = rng.permutation(single_ids)
        perm = np.concatenate([perm, rng.choice(single_ids, n_batches * bs - len(perm), replace=False)])
        tot = np.zeros(5)
        for k in range(n_batches):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            _, logits = m.net(G.batch(b, device))
            l_single = o2.pom_loss(logits, Y[b], class_w)
            if arch == "vae":
                l_single = l_single + beta_kl * min(1.0, (step + 1) / (10 * n_batches)) * m.net.last_kl
            l_pair = l_ma = torch.zeros((), device=device)
            with single_stats(m):   # mixtures normalised with single-molecule statistics
                if "scent" in streams:
                    q = rng.choice(pairs, pair_batch, replace=False)
                    mols, inv = np.unique(np.concatenate([S.pair_a[q], S.pair_b[q]]), return_inverse=True)
                    v, g = m.molecules(S, mols, device, mix_grad); ia, ib = inv[:pair_batch], inv[pair_batch:]
                    _, pl = m.merged_logits(v[ia], v[ib], g[ia], g[ib])
                    l_pair = rank_loss(pl, PY[q], T)
                if "ma" in streams:
                    t = rng.choice(ma_tr, min(ma_batch, len(ma_tr)), replace=False)
                    mols, inv = np.unique(np.concatenate([S.ma_a[t], S.ma_b[t]]), return_inverse=True)
                    v, g = m.molecules(S, mols, device, mix_grad); ia, ib = inv[:len(t)], inv[len(t):]
                    z, _ = m.merged_logits(v[ia], v[ib], g[ia], g[ib])
                    ds = m.similarity(z, v[ia]) - m.similarity(z, v[ib])
                    l_ma = ((m.ma_line[0] * ds + m.ma_line[1] - L[t]) ** 2).mean() / varL
            loss = weights[0] * l_single + weights[1] * l_pair + weights[2] * l_ma + o2.regularization(m, recipe["weight_decay"])
            l_dream = torch.zeros((), device=device)
            if dream is not None:
                qd = torch.as_tensor(rng.choice(dtr, dream_batch, replace=False), device=device)
                with single_stats(m):
                    l_dream = ((dream_forward(m, G, D, qd, device, mix_grad, dream_length) - yP[qd]) ** 2).mean() / varD
                loss = loss + l_dream
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            tot += [loss.item(), l_single.item(), l_pair.item(), l_ma.item(), l_dream.item()]
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            t_ = tot / n_batches
            log(f"  seed {seed} epoch {epoch + 1}/{recipe['epochs']} loss {t_[0]:.3f} (single {t_[1]:.3f}, pairs {t_[2]:.3f}, Ma {t_[3]:.3f}, DREAM {t_[4]:.3f})")
        if on_epoch is not None:
            on_epoch(epoch + 1, m)
    return m.eval(), step


@torch.no_grad()
def ma_predict(m: Joint, S: Store, idx, device):
    mols, inv = np.unique(np.concatenate([S.ma_a[idx], S.ma_b[idx]]), return_inverse=True)
    v, g = m.molecules(S, mols, device); ia, ib = inv[:len(idx)], inv[len(idx):]
    z, _ = m.merged_logits(v[ia], v[ib], g[ia], g[ib])
    ds = m.similarity(z, v[ia]) - m.similarity(z, v[ib])
    return (m.ma_line[0] * ds + m.ma_line[1]).cpu().numpy(), ds.cpu().numpy()


@torch.no_grad()
def pair_logits(m: Joint, S: Store, idx, device, merged=True, batch=512):
    """GoodScents pairs ``idx`` -> logits (merged decode) or mean of the two single-molecule probabilities' logits."""
    out = []
    for k in range(0, len(idx), batch):
        q = idx[k:k + batch]
        mols, inv = np.unique(np.concatenate([S.pair_a[q], S.pair_b[q]]), return_inverse=True)
        v, g = m.molecules(S, mols, device); ia, ib = inv[:len(q)], inv[len(q):]
        if merged:
            out.append(m.merged_logits(v[ia], v[ib], g[ia], g[ib])[1])
        else:
            x = m.net.scale * v if m.arch == "o2n" else v
            p = torch.sigmoid(m.net.readout(x)[1]); pm = (p[ia] + p[ib]) / 2
            out.append(torch.logit(pm.clamp(1e-6, 1 - 1e-6)))
    return torch.cat(out).cpu().numpy()


def as_joint(net, arch, device):
    """Wrap an already trained backbone (no joint training; gain 0, Ma line 1, 0) for the same checks."""
    m = Joint(arch, getattr(getattr(net, "mu", None), "out_features", 32)).float().to(device)
    m.net = net
    return m.eval()


def load(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    m = Joint(c["arch"], c.get("latent", 32), c.get("white", False)).float().to(device)
    if "dream_line" in c["model_state_dict"]:
        m.dream_line = nn.Parameter(torch.zeros(3)); m.dream_offset = nn.Parameter(torch.zeros(c["model_state_dict"]["dream_offset"].shape))
    m.load_state_dict(c["model_state_dict"]); m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval(), c


# ------------------------------------------------------------------ O2-D: DREAM pairs as a stream (inside CV folds)

class DreamStream:
    """DREAM training pairs of one CV fold as a training stream. Every step encodes all 235 DREAM molecules, merges
    each mixture (gain-weighted, O2-N normalization), decodes; distance = sqrt(|uA - uB|^2 + b^2 (log rhoA - log rhoB)^2),
    u = unit(logits - offset), prediction a d + c (a, c, b, offset learned in Joint.dream_*)."""

    def __init__(self, G: o2.Graphs, pairs, cids, device):
        row = {int(c): i for i, c in enumerate(cids)}
        mixes = sorted({x for p in pairs for x in (p.a, p.b)}, key=sorted); mi = {x: k for k, x in enumerate(mixes)}
        M = np.zeros((len(mixes), len(cids)), np.float32)
        for k, x in enumerate(mixes):
            M[k, [row[c] for c in x]] = 1
        self.M = torch.as_tensor(M, device=device); self.rows = G.dream_rows()
        self.pa = torch.as_tensor([mi[p.a] for p in pairs], device=device); self.pb = torch.as_tensor([mi[p.b] for p in pairs], device=device)


def dream_forward(m: Joint, G: o2.Graphs, D: DreamStream, idx, device, enc_grad=1.0, length=True):
    """enc_grad = alpha: fraction of the DREAM gradient that reaches the encoder (0 = encoder learns from single
    molecules only; DREAM then updates the decoder, gain and head)."""
    x, e, s, d, g = G.batch(D.rows, device)
    enc = m.net.encode(x, e, s, d, g, len(D.rows))
    enc = grad_dial(enc, enc_grad)
    v, _ = m.net.latent(enc); G_ = torch.exp(m.gain(enc)[:, 0])
    used = torch.unique(torch.cat([D.pa[idx], D.pb[idx]]))
    W = D.M[used] * G_.unsqueeze(0); z = (W @ v) / W.sum(1, keepdim=True); rho = z.norm(dim=1)
    if m.arch == "o2c":                       # agreement relative to the molecules' own lengths
        rho = rho / ((W @ v.norm(dim=1, keepdim=True)) / W.sum(1, keepdim=True))[:, 0].clamp_min(1e-8)
    logits = m.decode_merged(z)
    u = logits - m.dream_offset; u = u / u.norm(dim=1, keepdim=True)
    pos = torch.full((len(D.M),), -1, dtype=torch.long, device=device); pos[used] = torch.arange(len(used), device=device)
    ia, ib = pos[D.pa[idx]], pos[D.pb[idx]]; a, c, b = m.dream_line
    h = torch.log(rho.clamp_min(1e-8))
    dist = torch.sqrt(((u[ia] - u[ib]) ** 2).sum(1) + (b ** 2 * (h[ia] - h[ib]) ** 2 if length else 0) + 1e-12)
    return a * dist + c


def train_dream(G: o2.Graphs, single_ids, D: DreamStream, y_P, train_idx, seed, device, dream_batch=64, log=print, recipe=o2.RECIPE, enc_grad=1.0, on_epoch=None, arch="o2n"):
    """O2-D: O2-N trained from scratch on single molecules (anchor) + one fold's DREAM training pairs (MSE / var)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Joint(arch).float().to(device)
    m.dream_line = nn.Parameter(torch.tensor([0.5, float(np.mean(y_P[train_idx])) - 0.5, 0.3], device=device))
    m.dream_offset = nn.Parameter(torch.zeros(G.Y.shape[1], device=device))
    Y = torch.as_tensor(G.Y, device=device); yP = torch.as_tensor(y_P, dtype=torch.float32, device=device); var = float(np.var(y_P[train_idx]))
    counts = G.Y[single_ids].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=recipe["lr"], betas=(0.9, 0.999), eps=1e-8)
    bs = recipe["batch"]; n_batches = math.ceil(len(single_ids) / bs); step = 0
    for epoch in range(recipe["epochs"]):
        m.train(); perm = rng.permutation(single_ids)
        perm = np.concatenate([perm, rng.choice(single_ids, n_batches * bs - len(perm), replace=False)])
        tot = np.zeros(3)
        for k in range(n_batches):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            _, logits = m.net(G.batch(b, device)); l_single = o2.pom_loss(logits, Y[b], class_w)
            q = torch.as_tensor(rng.choice(train_idx, dream_batch, replace=False), device=device)
            with single_stats(m):   # mixtures normalised with single-molecule statistics
                l_dream = ((dream_forward(m, G, D, q, device, enc_grad) - yP[q]) ** 2).mean() / var
            loss = l_single + l_dream + o2.regularization(m, recipe["weight_decay"])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += [loss.item(), l_single.item(), l_dream.item()]
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            t_ = tot / n_batches; log(f"  seed {seed} epoch {epoch + 1} loss {t_[0]:.3f} (single {t_[1]:.3f}, DREAM {t_[2]:.3f})")
        if on_epoch is not None:
            on_epoch(epoch + 1, m)
    return m.eval()


@torch.no_grad()
def dream_predict(m, G, D, idx, device, length=True):
    return dream_forward(m, G, D, torch.as_tensor(np.asarray(idx), device=device), device, length=length).cpu().numpy()


# ------------------------------------------------------------------ validation lines (every epoch)

def inner_split(P, tr, seed=1):
    """Inner validation inside one fold's training pairs (CV.md rule 6): whole mixtures held out, 1/5, seed 1.
    -> (inner_train, inner_val) as indices into P; pairs with one inner-val mixture are used for neither."""
    from . import cv
    f = cv.mixture_holdout_folds([P[i] for i in tr], k=5, repeats=1, seed=seed)[0]
    return np.asarray(tr)[list(f.train)], np.asarray(tr)[list(f.test)]


class Tracker:
    """Called after every epoch: records the validation line and keeps the held-out predictions of the epoch with the
    lowest inner-validation loss (early stopping without stopping). The held-out r per epoch is recorded as a
    diagnostic only; no choice uses it."""

    def __init__(self, G, D, y_P, inner_tr, inner_val, test_idx, single_test, device, length=True):
        self.G, self.D, self.y, self.dev, self.length = G, D, np.asarray(y_P), device, length
        self.itr, self.iva, self.te, self.st = inner_tr, inner_val, test_idx, single_test
        self.var = float(np.var(self.y[inner_tr])); self.curve, self.best = [], (np.inf, 0, None, None)

    def __call__(self, epoch, m):
        m.eval()
        pr = {k: dream_predict(m, self.G, self.D, idx, self.dev, length=self.length) for k, idx in (("tr", self.itr), ("va", self.iva), ("te", self.te))}
        y = {k: self.y[idx] for k, idx in (("tr", self.itr), ("va", self.iva), ("te", self.te))}
        # line refitted on the inner-training pairs as scored (eval mode), as the frozen and "out" rows get; removes the
        # offset dropout leaves between training and scoring (about -0.04 on one member)
        a, c = fit_line(pr["tr"], y["tr"]); pr = {k: a * v + c for k, v in pr.items()}
        auc = o2.mean_auroc(o2.predict_probs(m.net, self.G, self.st, self.dev), self.G.Y[self.st])
        loss = float(np.mean((pr["va"] - y["va"]) ** 2) / self.var)
        row = {"epoch": epoch, "train_r": float(np.corrcoef(pr["tr"], y["tr"])[0, 1]), "val_r": float(np.corrcoef(pr["va"], y["va"])[0, 1]),
               "val_loss": loss, "test_r_diagnostic": float(np.corrcoef(pr["te"], y["te"])[0, 1]), "single_auroc": auc}
        po = o2.predict_probs(m.net, self.G, self.G.dream_rows(), self.dev)   # decoded DREAM molecules, for "out"
        self.curve.append(row)
        if loss < self.best[0]:
            self.best = (loss, epoch, pr["te"], po)
        self.final, self.final_probs = pr["te"], po
        m.train()
