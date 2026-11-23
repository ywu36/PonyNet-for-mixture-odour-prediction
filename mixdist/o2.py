# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   N = molecules in the graph file   B = molecules in a batch   V = atoms   E = directed bonds
#   W = words (138)   H = encoder output size (400)   K = mixtures   F = feature size
# ---------------------------------------------------------------------------------------------
"""O2 (design/o2_design.md): open-POM rebuilt in plain PyTorch, with the mixture pooled inside.

Layer names match open-POM's checkpoints (mpnn, project_edge_feats, readout_set2set, ffn), so the released
weights load unchanged. Encoder = message passing (NNConv + GRU, 5 steps) + radius-0 fold + set2set -> e (400);
read-out = ffn 400 -> 392 -> 392 -> 256 -> 138 with BatchNorm, ReLU, dropout 0.12 (open-POM's defaults).
Set layer: z(A) = sum_m G(m) e(m) / sum_m G(m). Training recipe: train_member().
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

NODE_OUT, EDGE_HIDDEN, EDGE_OUT, STEPS = 100, 75, 100, 5
S2S_ITERS, S2S_LAYERS = 3, 2
FFN_HIDDEN, FFN_EMB, DROPOUT = [392, 392], 256, 0.12
RECIPE = {"batch": 128, "epochs": 62, "lr": 1e-3, "decay": 0.5, "decay_steps": 640, "weight_decay": 1e-5}


class Graphs:
    """The exported graph file (evaluation/O2/export_o2_graphs.py), with batching."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.smiles_N = z["smiles"]; self.n_gslf = int(z["n_gslf"]); self.dream_cids = z["dream_cids"]
        self.Y = z["Y"]; self.words = z["words"]; self.audit_test = z["audit_test"]
        self.node, self.edge, self.edge_index = z["node"], z["edge"], z["edge_index"]
        self.node_off, self.edge_off = z["node_off"], z["edge_off"]

    def dream_rows(self):
        """Rows of the 235 DREAM molecules, in dream_cids order."""
        return np.arange(self.n_gslf, len(self.smiles_N))

    def batch(self, ids, device):
        """Molecules ``ids`` -> (node_VxF, edge_ExF, src_E, dst_E, graph_V) on ``device`` (global atom indices)."""
        nodes, edges, src, dst, gid, base = [], [], [], [], [], 0
        for b, i in enumerate(ids):
            n0, n1, e0, e1 = self.node_off[i], self.node_off[i + 1], self.edge_off[i], self.edge_off[i + 1]
            nodes.append(self.node[n0:n1]); edges.append(self.edge[e0:e1])
            src.append(self.edge_index[0, e0:e1] + base); dst.append(self.edge_index[1, e0:e1] + base)
            gid.append(np.full(n1 - n0, b)); base += n1 - n0
        t = lambda a, dt: torch.as_tensor(np.concatenate(a), dtype=dt, device=device)  # noqa: E731
        return t(nodes, torch.float32), t(edges, torch.float32), t(src, torch.long), t(dst, torch.long), t(gid, torch.long)


class NNConv(nn.Module):
    """DGL NNConv, sum aggregation, identity residual, bias: h'_j = sum_{i->j} h_i W(e_ij) + h_j + b."""

    def __init__(self, n_in, n_out, edge_in, edge_hidden):
        super().__init__()
        self.edge_func = nn.Sequential(nn.Linear(edge_in, edge_hidden), nn.ReLU(), nn.Linear(edge_hidden, n_in * n_out))
        self.bias = nn.Parameter(torch.zeros(n_out)); self.n_in, self.n_out = n_in, n_out

    def forward(self, h_VxF, W_ExFxF, src_E, dst_E):
        m_ExF = torch.bmm(h_VxF[src_E].unsqueeze(1), W_ExFxF).squeeze(1)
        return torch.zeros_like(h_VxF).index_add_(0, dst_E, m_ExF) + h_VxF + self.bias


class MPNN(nn.Module):
    def __init__(self, atom_f, bond_f):
        super().__init__()
        self.project_node_feats = nn.Sequential(nn.Linear(atom_f, NODE_OUT), nn.ReLU())
        self.gnn_layer = NNConv(NODE_OUT, NODE_OUT, bond_f, EDGE_HIDDEN)
        self.gru = nn.GRU(NODE_OUT, NODE_OUT)

    def forward(self, x_VxF, e_ExF, src_E, dst_E):
        h = self.project_node_feats(x_VxF); hidden = h.unsqueeze(0)
        W = self.gnn_layer.edge_func(e_ExF).view(-1, NODE_OUT, NODE_OUT)       # same every step (DGL recomputes it)
        for _ in range(STEPS):
            h = torch.relu(self.gnn_layer(h, W, src_E, dst_E))
            h, hidden = self.gru(h.unsqueeze(0), hidden); h = h.squeeze(0)
        return h


class Set2Set(nn.Module):
    """DGL Set2Set (atoms -> molecule): LSTM query, softmax attention over each molecule's atoms."""

    def __init__(self, dim, n_iters, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(2 * dim, dim, n_layers); self.dim, self.n_iters, self.n_layers = dim, n_iters, n_layers

    def forward(self, feat_VxF, graph_V, n_graphs):
        hc = (feat_VxF.new_zeros(self.n_layers, n_graphs, self.dim), feat_VxF.new_zeros(self.n_layers, n_graphs, self.dim))
        q_star = feat_VxF.new_zeros(n_graphs, 2 * self.dim)
        for _ in range(self.n_iters):
            q, hc = self.lstm(q_star.unsqueeze(0), hc); q = q.squeeze(0)
            e_V = (feat_VxF * q[graph_V]).sum(-1)
            emax = feat_VxF.new_full((n_graphs,), -math.inf).scatter_reduce(0, graph_V, e_V, "amax", include_self=True)
            ex = torch.exp(e_V - emax[graph_V])
            alpha = ex / feat_VxF.new_zeros(n_graphs).index_add_(0, graph_V, ex)[graph_V]
            readout = feat_VxF.new_zeros(n_graphs, self.dim).index_add_(0, graph_V, feat_VxF * alpha.unsqueeze(1))
            q_star = torch.cat([q, readout], -1)
        return q_star


class FFN(nn.Module):
    """open-POM CustomPositionwiseFeedForward with batch_norm (its default): returns (embedding 256, logits 138)."""

    def __init__(self, d_in, n_out):
        super().__init__()
        dims = [d_in] + FFN_HIDDEN + [FFN_EMB]
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])] + [nn.Linear(FFN_EMB, n_out)])
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(d) for d in dims[1:]])
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        for i in range(len(FFN_HIDDEN)):
            x = self.drop(torch.relu(self.batchnorms[i](self.linears[i](x))))
        emb = self.linears[len(FFN_HIDDEN)](x)
        x = self.drop(torch.relu(self.batchnorms[len(FFN_HIDDEN)](emb)))
        return emb, self.linears[-1](x)


class Encoder(nn.Module):
    """open-POM's encoder: message passing + radius-0 fold + set2set -> e (400). Layer names as the checkpoints."""

    def __init__(self, atom_f=134, bond_f=6):
        super().__init__()
        self.mpnn = MPNN(atom_f, bond_f)
        self.project_edge_feats = nn.Sequential(nn.Linear(bond_f, EDGE_OUT), nn.ReLU())
        self.readout_set2set = Set2Set(NODE_OUT + EDGE_OUT, S2S_ITERS, S2S_LAYERS)

    def encode(self, x_VxF, e_ExF, src_E, dst_E, graph_V, n_graphs):
        """Per-molecule encoder output e (n_graphs x 400)."""
        h = self.mpnn(x_VxF, e_ExF, src_E, dst_E)
        msg = torch.cat([h[src_E], self.project_edge_feats(e_ExF)], 1)          # radius-0 fold of atoms and bonds
        node = h.new_zeros(len(h), msg.shape[1]).index_add_(0, dst_E, msg)
        return self.readout_set2set(node, graph_V, n_graphs)

    @staticmethod
    def pool(e_BxH, member_KxB, gain_B=None):
        """Set layer: z(A) = sum_m G(m) e(m) / sum_m G(m); member_KxB = 0/1 membership of each mixture."""
        w = member_KxB if gain_B is None else member_KxB * gain_B.unsqueeze(0)
        return (w @ e_BxH) / w.sum(1, keepdim=True)


class O2Net(Encoder):
    """Encoder (mpnn + fold + set2set) -> set layer -> read-out (ffn)."""

    def __init__(self, atom_f=134, bond_f=6, n_words=138):
        super().__init__(atom_f, bond_f)
        self.ffn = FFN(2 * (NODE_OUT + EDGE_OUT), n_words)

    def readout(self, z_KxH):
        """Read-out of (pooled) encoder vectors -> (embedding 256, logits 138)."""
        return self.ffn(z_KxH)

    def latent(self, e_BxH):
        """The vector that is merged for a mixture (O2: e itself; no precision)."""
        return e_BxH, None

    def forward(self, batch):
        x, e, s, d, g = batch
        emb, logits = self.readout(self.encode(x, e, s, d, g, int(g.max()) + 1))
        return emb, logits


class O2VAE(Encoder):
    """O2-VAE (design/o2_vae_design.md): e -> 392 -> 392 -> (mu, log var) in D dims -> z -> 392 -> 256 -> 138.
    Training samples z = mu + sigma eps and stores the batch KL in last_kl; evaluation uses z = mu."""

    def __init__(self, latent=32, atom_f=134, bond_f=6, n_words=138):
        super().__init__(atom_f, bond_f)
        d_in = 2 * (NODE_OUT + EDGE_OUT)
        self.enc = nn.ModuleList([nn.Linear(d_in, FFN_HIDDEN[0]), nn.Linear(FFN_HIDDEN[0], FFN_HIDDEN[1])])
        self.enc_bn = nn.ModuleList([nn.BatchNorm1d(FFN_HIDDEN[0]), nn.BatchNorm1d(FFN_HIDDEN[1])])
        self.mu = nn.Linear(FFN_HIDDEN[1], latent); self.logvar = nn.Linear(FFN_HIDDEN[1], latent)
        self.dec1 = nn.Linear(latent, FFN_HIDDEN[1]); self.dec_bn1 = nn.BatchNorm1d(FFN_HIDDEN[1])
        self.dec2 = nn.Linear(FFN_HIDDEN[1], FFN_EMB); self.dec_bn2 = nn.BatchNorm1d(FFN_EMB)
        self.out = nn.Linear(FFN_EMB, n_words); self.drop = nn.Dropout(DROPOUT); self.last_kl = 0.0

    def latent(self, e_BxH):
        """-> (mu, precision exp(-log var)), each B x D."""
        x = e_BxH
        for lin, bn in zip(self.enc, self.enc_bn):
            x = self.drop(torch.relu(bn(lin(x))))
        lv = self.logvar(x).clamp(-10, 10)
        return self.mu(x), torch.exp(-lv)

    def readout(self, z_KxD):
        x = self.drop(torch.relu(self.dec_bn1(self.dec1(z_KxD))))
        emb = self.dec2(x)
        return emb, self.out(self.drop(torch.relu(self.dec_bn2(emb))))

    def forward(self, batch):
        x, e, s, d, g = batch
        mu, prec = self.latent(self.encode(x, e, s, d, g, int(g.max()) + 1))
        var = 1 / prec
        self.last_kl = 0.5 * (mu ** 2 + var + torch.log(prec) - 1).sum(1).mean()
        z = mu + var.sqrt() * torch.randn_like(mu) if self.training else mu
        return self.readout(z)


class O2N(O2Net):
    """O2-N (design/o2n_design.md): encoder output centred and scaled per unit (BatchNorm, no affine), made a unit
    vector, merged as a gain-weighted mean; the decoder gets only the merged direction (x sqrt(400)); the merged
    norm rho = agreement (1 for one molecule) is returned for the distance."""

    def __init__(self, atom_f=134, bond_f=6, n_words=138):
        super().__init__(atom_f, bond_f, n_words)
        d = 2 * (NODE_OUT + EDGE_OUT)
        self.centre = nn.BatchNorm1d(d, affine=False); self.scale = math.sqrt(d)

    def latent(self, e_BxH):
        v = self.centre(e_BxH)
        return v / v.norm(dim=1, keepdim=True).clamp_min(1e-8), None

    def merge(self, v_BxH, member_KxB, gain_B=None):
        """-> (decoder input K x 400, agreement rho K)."""
        z = self.pool(v_BxH, member_KxB, gain_B); rho = z.norm(dim=1)
        return self.scale * z / rho.clamp_min(1e-8).unsqueeze(1), rho

    def forward(self, batch):
        x, e, s, d, g = batch
        v, _ = self.latent(self.encode(x, e, s, d, g, int(g.max()) + 1))
        return self.readout(self.scale * v)


class O2C(O2Net):
    """O2-C (centred only; user 2026-11-18): encoder output centred and scaled per unit (BatchNorm, no affine), NOT made a
    unit vector; merged as a gain-weighted mean that keeps its length into the decoder. Direction and length are split
    only in the distance. agreement rho = |z| / (weighted mean of |v|) (1 for one molecule, < 1 when they differ).
    Note: for a network trained on single molecules only, O2-C equals O2 up to parametrisation (the per-unit affine map
    is absorbed by the decoder's first linear layer and commutes with the mean)."""

    def __init__(self, atom_f=134, bond_f=6, n_words=138):
        super().__init__(atom_f, bond_f, n_words)
        self.centre = nn.BatchNorm1d(2 * (NODE_OUT + EDGE_OUT), affine=False); self.scale = 1.0

    def latent(self, e_BxH):
        return self.centre(e_BxH), None

    def merge(self, v_BxH, member_KxB, gain_B=None):
        """-> (decoder input K x 400 = merged vector with its own length, agreement rho K)."""
        z = self.pool(v_BxH, member_KxB, gain_B)
        return z, z.norm(dim=1) / self.pool(v_BxH.norm(dim=1, keepdim=True), member_KxB, gain_B)[:, 0].clamp_min(1e-8)

    def forward(self, batch):
        x, e, s, d, g = batch
        v, _ = self.latent(self.encode(x, e, s, d, g, int(g.max()) + 1))
        return self.readout(v)


def load_released(path, device="cpu") -> O2Net:
    net = O2Net().float()                          # float32 even if another module set torch's default to float64
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def latent_all(net, G: Graphs, ids, device):
    """Merged-space vectors of molecules ``ids`` -> (vector, precision or None)."""
    mu, prec = net.latent(encode_all(net, G, ids, device))
    return mu, prec


@torch.no_grad()
def encode_all(net, G: Graphs, ids, device, batch=256):
    """Encoder outputs for molecules ``ids`` -> e (len(ids) x 400), eval mode."""
    net.eval(); out = []
    for k in range(0, len(ids), batch):
        b = ids[k:k + batch]; x, e, s, d, g = G.batch(b, device)
        out.append(net.encode(x, e, s, d, g, len(b)))
    return torch.cat(out)


def pom_loss(logits_BxW, y_BxW, class_w_W):
    """open-POM's CustomMultiLabelLoss as run by DeepChem: cross-entropy over [1 - z, z] on the logits z, weighted
    per word by log(1 + count / max count), summed over words, mean over the batch."""
    two = torch.stack([1 - logits_BxW, logits_BxW], 1)                      # B x 2 x W
    ce = nn.functional.cross_entropy(two, y_BxW.long(), reduction="none")     # B x W
    return (ce * class_w_W).sum(1).mean()


def regularization(net, wd):
    l1 = sum(p.abs().sum() for n, p in net.named_parameters() if "bias" not in n)
    l2 = sum(p.norm(2) for n, p in net.named_parameters() if "bias" not in n)
    return wd * (l1 + l2)


def train_member(G: Graphs, train_ids, seed, device, log=print, recipe=RECIPE, make_net=O2Net, beta_kl=0.0, kl_warmup_epochs=10):
    """One ensemble member with open-POM's recipe (single molecules; DeepChem pads the last batch to full size).
    make_net: O2Net (deterministic) or e.g. lambda: O2VAE(32); beta_kl > 0 adds beta_kl x KL, warmed up linearly."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = make_net().float().to(device)
    Y = torch.as_tensor(G.Y, device=device)
    counts = G.Y[train_ids].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=recipe["lr"], betas=(0.9, 0.999), eps=1e-8)
    bs, step = recipe["batch"], 0
    n_batches = math.ceil(len(train_ids) / bs)
    for epoch in range(recipe["epochs"]):
        net.train(); perm = rng.permutation(train_ids)
        perm = np.concatenate([perm, rng.choice(train_ids, n_batches * bs - len(perm), replace=False)])
        tot = 0.0
        for k in range(n_batches):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            _, logits = net(G.batch(b, device))
            loss = pom_loss(logits, Y[b], class_w) + regularization(net, recipe["weight_decay"])
            if beta_kl > 0:
                loss = loss + beta_kl * min(1.0, (step + 1) / (kl_warmup_epochs * n_batches)) * net.last_kl
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            log(f"  seed {seed} epoch {epoch + 1}/{recipe['epochs']} step {step} loss {tot / n_batches:.4f}")
    return net.eval(), step


@torch.no_grad()
def predict_probs(net, G: Graphs, ids, device, batch=256):
    net.eval(); out = []
    for k in range(0, len(ids), batch):
        _, logits = net(G.batch(ids[k:k + batch], device)); out.append(torch.sigmoid(logits))
    return torch.cat(out).cpu().numpy()


def mean_auroc(probs_NxW, y_NxW):
    """Mean AUROC over words; words whose labels are all 0 or all 1 in y are skipped."""
    from sklearn.metrics import roc_auc_score
    ok = [w for w in range(y_NxW.shape[1]) if 0 < y_NxW[:, w].sum() < len(y_NxW)]
    return float(np.mean([roc_auc_score(y_NxW[:, w], probs_NxW[:, w]) for w in ok]))


# ------------------------------------------------------------------ step-1 evaluation head (DREAM pairs)

class QuantileMap:
    """sklearn QuantileTransformer(n_quantiles=100, output_distribution='normal').transform, rebuilt in torch so a
    profile's gradient passes through; fitted on the molecules' single-molecule profiles (as B1a)."""
    BOUND = 1e-7

    def __init__(self, probs_MxW, device):
        from sklearn.preprocessing import QuantileTransformer
        self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=100).fit(probs_MxW)
        self.xp_WxQ = torch.as_tensor(self.qt.quantiles_.T.copy(), dtype=torch.float64, device=device).contiguous()
        self.fp_Q = torch.as_tensor(self.qt.references_, dtype=torch.float64, device=device)

    @staticmethod
    def _interp(x_WxK, xp_WxQ, fp_Q):
        idx = torch.searchsorted(xp_WxQ, x_WxK.contiguous(), right=True).clamp(1, xp_WxQ.shape[1] - 1)
        x0, x1 = xp_WxQ.gather(1, idx - 1), xp_WxQ.gather(1, idx)
        y0, y1 = fp_Q[idx - 1], fp_Q[idx]
        t = torch.where(x1 > x0, (x_WxK - x0) / (x1 - x0).clamp_min(1e-300), torch.zeros_like(x_WxK))
        y = y0 + t.clamp(0, 1) * (y1 - y0)
        y = torch.where(x_WxK < xp_WxQ[:, :1], fp_Q[0].expand_as(y), y)
        return torch.where(x_WxK >= xp_WxQ[:, -1:], fp_Q[-1].expand_as(y), y)

    def __call__(self, x_KxW):
        x = x_KxW.double().T
        u = 0.5 * (self._interp(x, self.xp_WxQ, self.fp_Q) - self._interp(-x, -self.xp_WxQ.flip(1), -self.fp_Q.flip(0)))
        u = torch.where(x - self.BOUND < self.xp_WxQ[:, :1], torch.zeros_like(u), u)
        u = torch.where(x + self.BOUND > self.xp_WxQ[:, -1:], torch.ones_like(u), u)
        eps = float(np.spacing(1))
        lo, hi = float(torch.special.ndtri(torch.tensor(self.BOUND - eps, dtype=torch.float64))), \
            float(torch.special.ndtri(torch.tensor(1 - (self.BOUND - eps), dtype=torch.float64)))
        return torch.special.ndtri(u.clamp(1e-300, 1 - 1e-16)).clamp(lo, hi).T


class O2Head:
    """O2 step-1 head on DREAM pairs: pool encoder vectors inside (gain G), read out through the frozen members,
    mean over members, quantile map, then direction (+ length) distance and a line, as O1.2b.

        G(m) = exp(theta . x(m)),   d = sqrt(|uA - uB|^2 + beta (hA - hB)^2),   prediction a d + c
    use_theta False -> G = 1; use_beta False -> beta = 0 (B1a's distance). theta penalised by lam (as O1.2b).
    A fit whose distances or line are not finite falls back to the line-only fit; O2Head.fallbacks counts them.
    """
    fallbacks = 0

    def __init__(self, nets, e_RxMxH, x_MxG, mix_KxM, pa_P, pb_P, qmap, device, use_theta=True, use_beta=True, lam=1e-3, k0=0.1,
                 prec_RxMxH=None, length="readout"):
        """e_RxMxH: the vectors merged (O2: encoder output; O2-VAE: mu). prec_RxMxH (O2-VAE): precision per dimension,
        given -> precision-weighted merge sum G P mu / sum G P; None -> weighted mean."""
        self.nets, self.e, self.dev, self.qmap, self.prec = nets, e_RxMxH, device, qmap, prec_RxMxH
        self.length = length                       # "readout": log |q(r(A))| (O1.2b); "agreement": log rho(A) (O2-N)
        self.rho = None
        self.x = torch.as_tensor(np.asarray(x_MxG, float), device=device)
        self.mix = torch.as_tensor(np.asarray(mix_KxM, np.float32), device=device)
        self.pa, self.pb = torch.as_tensor(pa_P, device=device), torch.as_tensor(pb_P, device=device)
        self.use_theta, self.use_beta, self.lam, self.k0 = use_theta, use_beta, lam, k0

    def profiles(self, theta):
        """Read-out profile r(A) of every mixture (K x W): pooled per member, sigmoid, mean over members."""
        g = torch.exp(self.x @ theta).float()
        out, rho = 0, 0
        for r, (net, e) in enumerate(zip(self.nets, self.e)):
            if hasattr(net, "merge"):
                z, rr = net.merge(e, self.mix, g); rho = rho + rr / len(self.nets)
            elif self.prec is None:
                z = net.pool(e, self.mix, g)
            else:
                w = self.mix * g.unsqueeze(0); P = self.prec[r]
                z = (w @ (P * e)) / (w @ P)
            _, logits = net.readout(z)
            out = out + torch.sigmoid(logits)
        self.rho = rho
        return out / len(self.nets)

    def distance(self, theta, b, idx):
        y = self.qmap(self.profiles(theta)); n = y.norm(dim=1, keepdim=True)
        u = y / n
        h = torch.log(self.rho.double().clamp_min(1e-8)) if self.length == "agreement" else torch.log(n[:, 0])
        ia, ib = self.pa[idx], self.pb[idx]
        beta = b ** 2 if self.use_beta else torch.zeros((), dtype=torch.float64, device=self.dev)
        return torch.sqrt(((u[ia] - u[ib]) ** 2).sum(1) + beta * (h[ia] - h[ib]) ** 2 + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev); y = torch.as_tensor(np.asarray(y, float), device=self.dev)
        theta = torch.zeros(self.x.shape[1], dtype=torch.float64, device=self.dev, requires_grad=self.use_theta)
        b = torch.tensor(self.k0 if self.use_beta else 0.0, dtype=torch.float64, device=self.dev, requires_grad=self.use_beta)
        with torch.no_grad():
            d = self.distance(theta, b, idx)
            D = torch.stack([d, torch.ones_like(d)], 1)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        params = [lin] + ([theta] if self.use_theta else []) + ([b] if self.use_beta else [])
        if len(params) > 1:
            opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                d = self.distance(theta, b, idx)
                loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + self.lam * (theta ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                ok = bool(torch.isfinite(self.distance(theta, b, idx)).all()) and bool(torch.isfinite(lin).all())
            if not ok:                                   # optimiser blew up: fall back to the line-only fit (counted)
                O2Head.fallbacks += 1
                theta = torch.zeros_like(theta); b = torch.zeros_like(b)
                with torch.no_grad():
                    d = self.distance(theta, b, idx)
                    lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0]
        self.theta, self.b, self.lin = theta.detach(), b.detach(), lin.detach()
        return self

    @property
    def beta(self):
        return float(self.b ** 2) if self.use_beta else 0.0

    def predict(self, idx):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev)
        with torch.no_grad():
            d = self.distance(self.theta, self.b, idx)
            return (self.lin[0] * d + self.lin[1]).cpu().numpy()


def load_members(paths, device):
    """Saved members (O2 or O2-VAE: a checkpoint with 'latent' is a VAE) -> frozen nets in eval mode."""
    nets = []
    for p in paths:
        c = torch.load(p, map_location="cpu", weights_only=False)
        arch = c.get("arch") or ("vae" if c.get("latent") else "o2")
        net = {"vae": lambda: O2VAE(c["latent"]), "o2n": O2N, "o2c": O2C, "o2": O2Net}[arch]().float().to(device)
        net.load_state_dict(c["model_state_dict"])
        for q in net.parameters():
            q.requires_grad_(False)
        nets.append(net.eval())
    return nets
