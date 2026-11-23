# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  V = atoms   E = directed bonds   B = molecules in a batch   F = atom width (d)
#   H = hidden layer (h)   W = words (138)
# ---------------------------------------------------------------------------------------------
"""O3-M: a minimal backbone, trained on single molecules only (the user, 2026-11-19: "learn a minimal network backbone
first"; no group weights: the network must carry a molecule's strength itself, latently).

  atoms (134 features) -> linear + ReLU -> d
  message passing: NNConv (bond-conditioned: 6 bond features -> 16 -> d x d) + GRU, `steps` steps, width d (open-POM: 75, 100, 5)
  molecule = MEAN over atoms (no set2set)  -> linear d -> h  -> centring (BatchNorm, no scale/shift)  = v (hidden layer)
  word head = dropout + ONE linear layer h -> 138                                                   (no deep decoder)

Why linear head: head(sum_m a_m v_m) = sum_m a_m head(v_m) (up to the bias), so a mixture's hidden vector can be summed
and read with the same head; the add-up structure is exact. A molecule's pull in a plain sum is its vector length |v|.
Training: open-POM's recipe and loss (62 epochs, lr 1e-3 halved every 640 steps, batch 128, L1 + L2 1e-5), GS-LF
training molecules; official held-out molecules (audit_test) for AUROC; training-molecule AUROC (1000) for the gap.

Env:     mist (GPU)
Usage:   python training/train_o3m.py --width 32 64 128 --seeds 0 1 2
         python training/train_o3m.py --config pomMP --seeds 0 1 2     (steps towards open-POM: see CONFIGS)
         python training/train_o3m.py --config o4 --seeds 0 1 2 3 4 5 6 7 8 9   (O4 -> checkpoints/o4)
Writes:  checkpoints/o3m_<w<d> | config>/member_<s>.pt, results/train_o3m/<date_time>.{txt,json}
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixdist import features, o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402


class CosHead(nn.Module):
    """Cosine word head (NormFace style): logit_k = s cos(v, w_k) + b_k, s learned (starts at 10). The word read-out
    depends only on v's direction, so the hidden space is shaped for cosine comparison and v's length is left free."""

    def __init__(self, h, n_words, s0=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_words, h) / math.sqrt(h)); self.bias = nn.Parameter(torch.zeros(n_words))
        self.log_s = nn.Parameter(torch.tensor(math.log(s0)))

    def forward(self, v_BxH):
        return self.log_s.exp() * nn.functional.normalize(v_BxH, dim=1) @ nn.functional.normalize(self.weight, dim=1).T + self.bias


class O3M(nn.Module):
    """pool: "mean" over atoms, or "set2set" (open-POM: atoms + bond messages folded per atom, set2set -> 4d).
    ffn: hidden widths between the pooled vector and v (open-POM: [392, 392], each Linear + BatchNorm + ReLU + dropout
    0.12); then one Linear to h, centring -> v; head = dropout + Linear(h, 138). open-POM = d 100, steps 5, edge_h 75,
    set2set, ffn [392, 392], h 256, except its last step is BatchNorm + ReLU + Linear instead of centring + Linear."""

    def __init__(self, d=64, h=None, steps=3, edge_h=16, dropout=0.1, pool="mean", ffn=(), positive=False, sparse_l1=0.0, cos_head=False, atom_f=134, bond_f=6, n_words=138):
        super().__init__()
        h = h or d; self.d, self.h, self.steps, self.pool, self.positive, self.sparse_l1 = d, h, steps, pool, positive, sparse_l1
        self.last_kl = 0.0
        self.atom = nn.Sequential(nn.Linear(atom_f, d), nn.ReLU())
        self.conv = o2.NNConv(d, d, bond_f, edge_h); self.gru = nn.GRU(d, d)
        if pool in ("attn", "attn2"):   # attention pooling over atoms (2026-11-20): atom score g . x_a, soft max within the molecule
            self.att = nn.Linear(d, 1)
        if pool == "attn2":             # O5-2h: second atom head for strength; v = sqrt(h) exp(strength) x unit direction
            self.att2 = nn.Linear(d, 1)
        if pool == "set2set":
            self.edge_proj = nn.Sequential(nn.Linear(bond_f, d), nn.ReLU()); self.s2s = o2.Set2Set(2 * d, 3, 2); out = 4 * d
        else:
            out = d
        layers = []
        for w in ffn:
            layers += [nn.Linear(out, w), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(0.12)]; out = w
        self.ffn = nn.Sequential(*layers)
        self.lin = nn.Linear(out, h); self.centre = nn.BatchNorm1d(h, affine=False)
        self.drop = nn.Dropout(dropout); self.head = CosHead(h, n_words) if cos_head else nn.Linear(h, n_words)

    def encode(self, x_VxF, e_ExF, src_E, dst_E, graph_V, n_graphs):
        x = self.atom(x_VxF); hidden = x.unsqueeze(0)
        W = self.conv.edge_func(e_ExF).view(-1, self.d, self.d)
        for _ in range(self.steps):
            x = torch.relu(self.conv(x, W, src_E, dst_E))
            x, hidden = self.gru(x.unsqueeze(0), hidden); x = x.squeeze(0)
        if self.pool == "set2set":
            msg = torch.cat([x[src_E], self.edge_proj(e_ExF)], 1)
            pooled = self.s2s(x.new_zeros(len(x), msg.shape[1]).index_add_(0, dst_E, msg), graph_V, n_graphs)
        elif self.pool in ("attn", "attn2"):
            w = self.atom_weights(x, graph_V, n_graphs)
            pooled = x.new_zeros(n_graphs, self.d).index_add_(0, graph_V, w.unsqueeze(1) * x)
            if self.pool == "attn2":    # strength rides along as one extra column (latent splits it off)
                return torch.cat([self.lin(self.ffn(pooled)), self.strength(x, graph_V, n_graphs).unsqueeze(1)], 1)
        else:
            n = x.new_zeros(n_graphs).index_add_(0, graph_V, torch.ones_like(graph_V, dtype=x.dtype))
            pooled = x.new_zeros(n_graphs, self.d).index_add_(0, graph_V, x) / n.unsqueeze(1)
        return self.lin(self.ffn(pooled))

    def atom_scores(self, x_VxF):
        return self.att(x_VxF).squeeze(1)

    def strength(self, x_VxF, graph_V, n_graphs, head=2):
        """log mean exp of a head's atom scores within each molecule (soft max, size-neutral)."""
        s = (self.att2 if head == 2 else self.att)(x_VxF).squeeze(1)
        mx = s.new_full((n_graphs,), -1e30).scatter_reduce(0, graph_V, s, reduce="amax", include_self=True)
        se = s.new_zeros(n_graphs).index_add_(0, graph_V, torch.exp(s - mx[graph_V]))
        n = s.new_zeros(n_graphs).index_add_(0, graph_V, torch.ones_like(s))
        return mx + torch.log(se / n)

    def atom_weights(self, x_VxF, graph_V, n_graphs):
        """Soft max of the atom scores within each molecule (weights sum to 1 per molecule)."""
        s = self.atom_scores(x_VxF)
        mx = s.new_full((n_graphs,), -1e30).scatter_reduce(0, graph_V, s, reduce="amax", include_self=True)
        e = torch.exp(s - mx[graph_V]); den = s.new_zeros(n_graphs).index_add_(0, graph_V, e)
        return e / den[graph_V]

    def latent(self, e_BxH):
        """positive: receptor-like activations softplus(.) >= 0, not centred (a mixture adds activations).
        sparse_l1 > 0: ReLU (exact zeros) and the mean activation is returned as last_kl, so train_member adds
        sparse_l1 x mean activation to the loss (warmed up over 10 epochs): each unit fires for fewer molecules."""
        if self.pool == "attn2":        # direction from head 1 (centred, length 1), length from head 2's strength
            d_ = self.centre(e_BxH[:, :-1]); st = e_BxH[:, -1].clamp(-5, 5)
            return (self.h ** 0.5) * torch.exp(st).unsqueeze(1) * d_ / d_.norm(dim=1, keepdim=True).clamp_min(1e-6), None
        if self.sparse_l1 > 0:
            v = torch.relu(e_BxH); self.last_kl = v.mean(); return v, None
        return (nn.functional.softplus(e_BxH) if self.positive else self.centre(e_BxH)), None

    def readout(self, v_BxH):
        return v_BxH, self.head(self.drop(v_BxH))

    def forward(self, batch):
        x, e, s, d, g = batch
        return self.readout(self.latent(self.encode(x, e, s, d, g, int(g.max()) + 1))[0])


# steps from the minimal backbone towards open-POM (one piece added at a time); all end in centring + ONE linear head
CONFIGS = {"pomMP": {"d": 100, "h": 100, "steps": 5, "edge_h": 75},                                   # open-POM message passing, mean
           "pomMP_s2s": {"d": 100, "h": 400, "steps": 5, "edge_h": 75, "pool": "set2set"},             # + set2set
           "pomMP_ffn": {"d": 100, "h": 256, "steps": 5, "edge_h": 75, "ffn": [392, 392]},             # + FFN, mean pooling
           "w64pos": {"d": 64, "h": 64, "steps": 3, "edge_h": 16, "positive": True},                    # width 64, v >= 0
           "w64sparse03": {"d": 64, "h": 64, "steps": 3, "edge_h": 16, "positive": True, "sparse_l1": 0.3},
           "w64sparse1": {"d": 64, "h": 64, "steps": 3, "edge_h": 16, "positive": True, "sparse_l1": 1.0},
           "pom_lin": {"d": 100, "h": 256, "steps": 5, "edge_h": 75, "pool": "set2set", "ffn": [392, 392]},  # full open-POM, linear last
           # O4 (user, 2026-11-20): O3's additive structure (mean over atoms, centred 64-number hidden layer, ONE linear head)
           # with a larger encoder (open-POM's message passing + its two 392-wide layers) and 10 ensemble members
           "o4": {"d": 100, "h": 64, "steps": 5, "edge_h": 75, "ffn": [392, 392]},
           # O5 = plain O3 with attention pooling over atoms (user, 2026-11-20): which atoms dominate a molecule is learned from
           # single molecules (word loss only), so the DREAM side need not fit a 64-number attention vector
           "attpool": {"d": 64, "h": 64, "steps": 3, "edge_h": 16, "pool": "attn"},
           # O5-2h (user, 2026-11-20: "double attention"): head 1 pools atoms into the smell direction, head 2 gives the
           # molecule's strength = length of v; the linear word head reads both, so both heads learn from single molecules
           "attpool2": {"d": 64, "h": 64, "steps": 3, "edge_h": 16, "pool": "attn2"},
           # O5-big (user, 2026-11-20: "for O5 did you use the big encoder?"): O5's attention pooling on O4's encoder
           "o4att": {"d": 100, "h": 64, "steps": 5, "edge_h": 75, "ffn": [392, 392], "pool": "attn"}}


def load(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    net = O3M(**c["arch"]).float(); net.load_state_dict(c["model_state_dict"]); return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--width", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--config", choices=list(CONFIGS), help="a named step towards open-POM (overrides --width/--steps)")
    a = ap.parse_args(); dev = pick_device()
    rep = Report("train_o3m", {**vars(a), "device": dev, **o2.RECIPE}, tag=a.config or "w" + "-".join(map(str, a.width)))
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    rep.line(f"GS-LF training molecules {len(tr)}, held-out (official) {len(te)}; reference O2-C-Single (400 hidden, "
             f"5 steps, set2set, deep decoder, 2.09 M parameters): held-out single AUROC 0.875-0.883 per member")
    rows = []
    runs = [(a.config, CONFIGS[a.config])] if a.config else [(f"w{d}", {"d": d, "h": d, "steps": a.steps, "edge_h": 16}) for d in a.width]
    for name, arch in runs:
        ck = ROOT / ("checkpoints/o4" if name == "o4" else f"checkpoints/o3m_{name}"); ck.mkdir(parents=True, exist_ok=True)
        for s in a.seeds:
            t0 = time.time(); d = name
            net, step = o2.train_member(G, tr, s, dev, log=rep.line, make_net=lambda: O3M(**arch), beta_kl=arch.get("sparse_l1", 0.0))
            n_par = sum(p.numel() for p in net.parameters())
            auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te])
            auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
            torch.save({"model_state_dict": net.state_dict(), "arch": arch, "seed": s, "single_auroc": auc_te,
                        "train_auroc": auc_tr, "n_params": n_par}, ck / f"member_{s}.pt")
            rows.append([name, s, n_par, f"{auc_te:.4f}", f"{auc_tr:.4f}", f"{auc_tr - auc_te:+.4f}", f"{time.time() - t0:.0f}"])
            with torch.no_grad():
                vv = net.latent(o2.encode_all(net, G, te, dev))[0]
            zero = float((vv <= 1e-6).float().mean()); per_mol = float((vv > 1e-6).float().sum(1).mean())
            rep.line(f"{name} seed {s}: params {n_par}, held-out AUROC {auc_te:.4f}, training {auc_tr:.4f}; "
                     f"units at zero {zero:.3f}, active units per molecule {per_mol:.1f} of {vv.shape[1]}")
    rep.section("Single molecules: held-out (official) vs training-molecule AUROC")
    rep.table(["backbone", "seed", "parameters", "held-out AUROC", "training AUROC", "gap (training - held-out)", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
