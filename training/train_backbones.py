# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  V = atoms in a batch   E = directed bonds   B = molecules   N = atoms per
#   molecule (padded)   D = width (d)   H = attention heads   C = channels per head   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Backbones that see groups and rings by themselves (no extra atom features), tried one at a time in the order of the
literature check (findings/literature_2026-11-19.md; user, 2026-11-19: "try a few of them in our backbone, in order").

  et   Edge Transformer (Bergen et al. 2021; Mueller et al., NeurIPS 2024, 2401.10119): a state X_ij for every atom
       pair; each layer updates (i, j) through every third atom l (AlphaFold-style triangle):
         a_ilj = softmax_l( Q X_il . K X_lj / sqrt(C) ),   X_ij += O sum_l a_ilj (V1 X_il * V2 X_lj)
       then a feed-forward block. Start: X_ij = A h_i + B h_j + bond(i, j) (or a learned "no bond"), + "self" on i = j.
       Provably as strong as 3-WL (counts rings) without positional encodings. Atom state = X_ii.
Everything after the backbone is as in O3-M width 64 (training/train_o3m.py): mean over atoms -> linear d -> h ->
centring (BatchNorm, no scale/shift) = v; ONE linear head h -> 138; open-POM's recipe and loss; GS-LF training
molecules; official held-out molecules for AUROC; training-molecule AUROC (1000) for the memorization gap.

Env:     mist (GPU)
Usage:   python training/train_backbones.py --arch et --seeds 0 1 2
Writes:  checkpoints/bb_<arch>/member_<s>.pt, results/train_backbones/<date_time>_<arch>.{txt,json}
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


class TriangleLayer(nn.Module):
    """Pre-norm triangle attention + feed-forward on pair states X (B x N x N x D)."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.c = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v1, self.v2, self.o = (nn.Linear(d, d) for _ in range(5))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, X_BxNxNxD, mask_BxN):
        B, N = X_BxNxNxD.shape[:2]; y = self.ln1(X_BxNxNxD)
        sh = lambda t: t.view(B, N, N, self.h, self.c)  # noqa: E731
        q, k, v1, v2 = sh(self.q(y)), sh(self.k(y)), sh(self.v1(y)), sh(self.v2(y))
        logit = torch.einsum("bilhc,bljhc->bhilj", q, k) / math.sqrt(self.c)              # via third atom l
        logit = logit.masked_fill(~mask_BxN[:, None, None, :, None], float("-inf"))
        a = torch.softmax(logit, dim=3)
        out = torch.einsum("bhilj,bilhc,bljhc->bijhc", a, v1, v2).reshape(B, N, N, -1)
        X = X_BxNxNxD + self.drop(self.o(out))
        return X + self.drop(self.ff(self.ln2(X)))


class EdgeTransformer(nn.Module):
    """Same interface as train_o3m.O3M (encode / latent / readout / forward)."""

    def __init__(self, d=64, h=None, layers=3, heads=4, dropout=0.1, atom_f=134, bond_f=6, n_words=138, budget=1.5e6):
        super().__init__()
        h = h or d; self.d, self.h, self.budget = d, h, budget
        self.positive, self.sparse_l1, self.last_kl = False, 0.0, 0.0
        self.atom = nn.Sequential(nn.Linear(atom_f, d), nn.ReLU())
        self.a_i, self.a_j = nn.Linear(d, d), nn.Linear(d, d, bias=False)
        self.bond = nn.Linear(bond_f, d, bias=False)
        self.no_bond, self.self_pair = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList(TriangleLayer(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.lin = nn.Linear(d, h); self.centre = nn.BatchNorm1d(h, affine=False)
        self.drop = nn.Dropout(dropout); self.head = nn.Linear(h, n_words)

    def _chunks(self, sizes):
        """Molecules sorted by size, grouped so that (molecules x N^3) stays under the budget."""
        order = np.argsort(sizes, kind="stable"); out, cur = [], []
        for g in order:
            if cur and (len(cur) + 1) * int(sizes[g]) ** 3 > self.budget:
                out.append(cur); cur = []
            cur.append(g)
        return out + ([cur] if cur else [])

    def encode(self, x_VxF, e_ExF, src_E, dst_E, graph_V, n_graphs):
        hv = self.atom(x_VxF); dev = hv.device
        sizes = torch.bincount(graph_V, minlength=n_graphs); start = torch.cumsum(sizes, 0) - sizes
        local_V = torch.arange(len(graph_V), device=dev) - start[graph_V]; graph_E = graph_V[src_E]
        bond_E = self.bond(e_ExF); sizes_np = sizes.cpu().numpy()
        pooled = hv.new_zeros(n_graphs, self.d)
        for gi in self._chunks(sizes_np):
            gi_t = torch.as_tensor(gi, device=dev); b, N = len(gi), int(sizes_np[gi].max())
            pos = torch.full((n_graphs,), -1, device=dev, dtype=torch.long); pos[gi_t] = torch.arange(b, device=dev)
            sel_V = pos[graph_V] >= 0; sel_E = pos[graph_E] >= 0
            H = hv.new_zeros(b, N, self.d); mask = torch.zeros(b, N, dtype=torch.bool, device=dev)
            H[pos[graph_V[sel_V]], local_V[sel_V]] = hv[sel_V]; mask[pos[graph_V[sel_V]], local_V[sel_V]] = True
            X = self.a_i(H)[:, :, None, :] + self.a_j(H)[:, None, :, :] + self.no_bond
            ge, ls, ld = pos[graph_E[sel_E]], local_V[src_E[sel_E]], local_V[dst_E[sel_E]]
            X = X.index_put((ge, ls, ld), X[ge, ls, ld] - self.no_bond + bond_E[sel_E])
            ar = torch.arange(N, device=dev); X[:, ar, ar] = X[:, ar, ar] - self.no_bond + self.self_pair
            for layer in self.layers:
                X = layer(X, mask)
            diag = self.ln(X[:, ar, ar]) * mask[..., None]
            pooled = pooled.index_copy(0, gi_t, diag.sum(1) / mask.sum(1, keepdim=True))
        return self.lin(pooled)

    def latent(self, e_BxH):
        return self.centre(e_BxH), None

    def readout(self, v_BxH):
        return v_BxH, self.head(self.drop(v_BxH))

    def forward(self, batch):
        x, e, s, d, g = batch
        return self.readout(self.latent(self.encode(x, e, s, d, g, int(g.max()) + 1))[0])


ARCHS = {"et": (EdgeTransformer, {"d": 64, "layers": 3, "heads": 4})}


def load(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    cls, _ = ARCHS[c["arch_name"]]; net = cls(**c["arch"]).float(); net.load_state_dict(c["model_state_dict"])
    return net.to(device).eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args(); dev = pick_device()
    cls, arch = ARCHS[a.arch]
    rep = Report("train_backbones", {**vars(a), "arch": arch, "device": dev, **o2.RECIPE}, tag=a.arch)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    rep.line(f"GS-LF training molecules {len(tr)}, held-out (official) {len(te)}; device {dev}; reference message passing "
             "width 64 (117k): held-out 0.862 / 0.860 / 0.858, 3-member 0.870, gap 0.025")
    ck = ROOT / f"checkpoints/bb_{a.arch}"; ck.mkdir(parents=True, exist_ok=True); rows = []
    for s in a.seeds:
        t0 = time.time()
        net, _ = o2.train_member(G, tr, s, dev, log=rep.line, make_net=lambda: cls(**arch))
        n_par = sum(p.numel() for p in net.parameters())
        auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te])
        auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
        torch.save({"model_state_dict": net.state_dict(), "arch_name": a.arch, "arch": arch, "seed": s,
                    "single_auroc": auc_te, "train_auroc": auc_tr, "n_params": n_par}, ck / f"member_{s}.pt")
        rows.append([a.arch, s, n_par, f"{auc_te:.4f}", f"{auc_tr:.4f}", f"{auc_tr - auc_te:+.4f}", f"{time.time() - t0:.0f}"])
        rep.line(f"{a.arch} seed {s}: params {n_par}, held-out AUROC {auc_te:.4f}, training {auc_tr:.4f}, {time.time() - t0:.0f} s")
    rep.section("Single molecules: held-out (official) vs training-molecule AUROC")
    rep.table(["backbone", "seed", "parameters", "held-out AUROC", "training AUROC", "gap (training - held-out)", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
