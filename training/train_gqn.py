# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   H = hidden layer (64)   K = knots per unit (101)
#   N = reference molecules (GS-LF training set)
# ---------------------------------------------------------------------------------------------
"""Global quantile-normalisation layer on the hidden layer (the user, 2026-11-19: "a global quantile-normalisation layer,
zero learned numbers").

O1.2b's input maps every word to a standard normal by its rank over a whole population of molecules (quantile per word).
The count network (ga_w64) already has a global z-score on its 64 hidden units (BatchNorm, no learned scale / shift):
mean and spread only. This layer replaces that BatchNorm with the full rank version:
    per unit j: knots x_j1 < ... < x_jK = the unit's quantiles at p_k = (k - 0.5) / K over the GS-LF training molecules;
    v_j = piecewise-linear map x_jk -> Phi^-1(p_k)  (linear beyond the end knots, so the gradient never vanishes).
The knots are buffers, not learned: rebuilt at the start of every epoch from all GS-LF training molecules (no grad, as
BatchNorm's running statistics), saved with the network. Everything else = training/train_groupatoms.py --arch w64
(node-enriched atom input, GS-LF single-molecule words, o2.RECIPE, same seeds).

Env:     mist (GPU)
Variants (first run, "epoch": single AUROC 0.78, knots stale within an epoch):
  epoch     knots refreshed once per epoch, layer replaces BatchNorm          -> checkpoints/ga_w64_gqn
  steps     + refreshed every 5 steps                                        -> checkpoints/ga_w64_gqn_steps
  bn        BatchNorm kept, quantile layer after it (knots once per epoch)   -> checkpoints/ga_w64_gqn_bn
  bn_steps  both                                                             -> checkpoints/ga_w64_gqn_bn_steps
--pom: + POM-rank (as checkpoints/ga_w64_rank) -> checkpoints/ga_w64_rank_gqn[_<variant>]
Usage:   python training/train_gqn.py --variant epoch|steps|bn|bn_steps [--pom] --seeds 0 1 2
Writes:  checkpoints/ga_w64_gqn[_<variant>]/member_<s>.pt, results/train_gqn/<date_time>_<variant>.{txt,json}
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
for p in (ROOT, ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o3m import O3M  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3p  # noqa: E402

K = 101


class GlobalQuantileNorm(nn.Module):
    """Per unit: rank over a fixed reference population -> standard normal. Zero learned numbers."""

    def __init__(self, h, k=K):
        super().__init__()
        p = (torch.arange(k, dtype=torch.float32) + 0.5) / k
        self.register_buffer("z_K", torch.special.ndtri(p)); self.register_buffer("p_K", p)
        self.register_buffer("knots_HxK", self.z_K.repeat(h, 1).clone())          # identity until the first refresh

    @torch.no_grad()
    def refresh(self, x_NxH):
        q = torch.quantile(x_NxH.float().T, self.p_K.to(x_NxH.device), dim=1).T.contiguous()       # H x K
        q = torch.cummax(q, dim=1).values + 1e-6 * torch.arange(q.shape[1], device=q.device)        # strictly increasing
        self.knots_HxK.copy_(q)

    def forward(self, x_BxH):
        x = x_BxH.T.contiguous(); kn = self.knots_HxK
        i = torch.searchsorted(kn, x).clamp(1, kn.shape[1] - 1)
        x0, x1 = kn.gather(1, i - 1), kn.gather(1, i); z0, z1 = self.z_K[i - 1], self.z_K[i]
        return (z0 + (x - x0) * (z1 - z0) / (x1 - x0)).T


class O3MQ(O3M):
    """O3M whose hidden layer is normalised by GlobalQuantileNorm instead of BatchNorm (bn_first: after BatchNorm)."""

    def __init__(self, **arch):
        self.bn_first = bool(arch.get("bn_first", False))
        arch = {k: v for k, v in arch.items() if k not in ("gqn", "bn_first", "refresh_every")}
        super().__init__(**arch); self.gqn = GlobalQuantileNorm(self.h)

    def pre(self, e_BxH):
        return self.centre(e_BxH) if self.bn_first else e_BxH

    def latent(self, e_BxH):
        return self.gqn(self.pre(e_BxH)), None


@torch.no_grad()
def refresh(net, G, ref_ids, dev, batch=512):
    was = net.training; net.eval()
    e = torch.cat([net.encode(*G.batch(ref_ids[k:k + batch], dev), len(ref_ids[k:k + batch])) for k in range(0, len(ref_ids), batch)])
    net.gqn.refresh(net.pre(e)); net.train(was)


def train(G, tr, seed, dev, arch, log, recipe=o2.RECIPE, w_rank=0.0, Pemb=None):
    """o2.train_member's loop (same recipe, padding, decay) + knot refresh at the start of every epoch and, with
    arch["refresh_every"] = R, also every R steps."""
    every = arch.get("refresh_every", 0)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = O3MQ(**arch).float().to(dev); Y = torch.as_tensor(G.Y, device=dev)
    Pt = torch.as_tensor(Pemb, device=dev) if w_rank > 0 else None
    counts = G.Y[tr].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=recipe["lr"], betas=(0.9, 0.999), eps=1e-8)
    bs, step, nb = recipe["batch"], 0, math.ceil(len(tr) / recipe["batch"])
    for epoch in range(recipe["epochs"]):
        refresh(net, G, tr, dev); net.train()
        perm = rng.permutation(tr); perm = np.concatenate([perm, rng.choice(tr, nb * bs - len(perm), replace=False)]); tot = 0.0
        for k in range(nb):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            if every and k and k % every == 0:
                refresh(net, G, tr, dev)
            b = perm[k * bs:(k + 1) * bs]
            v = net.latent(net.encode(*G.batch(b, dev), len(b)))[0]; logits = net.head(net.drop(v))
            loss = o2.pom_loss(logits, Y[b], class_w) + o2.regularization(net, recipe["weight_decay"])
            if w_rank > 0:                                  # POM-rank (train_o3p "rank": keep open-POM's distance order)
                loss = loss + w_rank * train_o3p.order_loss(v, Pt[b])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            log(f"  seed {seed} epoch {epoch + 1}/{recipe['epochs']} step {step} loss {tot / nb:.4f}")
    refresh(net, G, tr, dev)
    return net.eval(), step


def load(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    net = O3MQ(**c["arch"]).float(); net.load_state_dict(c["model_state_dict"]); return net.to(device).eval()


VARIANTS = {"epoch": {}, "steps": {"refresh_every": 5}, "bn": {"bn_first": True}, "bn_steps": {"bn_first": True, "refresh_every": 5}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variant", choices=list(VARIANTS), default="epoch")
    ap.add_argument("--pom", action="store_true", help="add POM-rank (train_o3p rank recipe, weight 1.0)"); a = ap.parse_args(); dev = pick_device()
    G, tr, te, names = train_groupatoms.graphs_with_groups()
    arch = {**train_groupatoms.train_o3m_w64(), "atom_f": G.node.shape[1], "gqn": True, **VARIANTS[a.variant]}
    w_rank = train_o3p.CONFIGS["rank"][1] if a.pom else 0.0; Pemb = train_o3p.pom_targets(G) if a.pom else None
    tag = a.variant + ("_rank" if a.pom else "")
    rep = Report("train_gqn", {**vars(a), "arch": arch, "w_rank": w_rank, "knots": K, "device": dev, **o2.RECIPE}, tag=tag)
    ck = ROOT / ("checkpoints/ga_w64" + ("_rank" if a.pom else "") + "_gqn" + ("" if a.variant == "epoch" else f"_{a.variant}"))
    ck.mkdir(parents=True, exist_ok=True); rows = []
    tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    for s in a.seeds:
        t0 = time.time(); net, steps = train(G, tr, s, dev, arch, rep.line, w_rank=w_rank, Pemb=Pemb)
        auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te]); auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
        with torch.no_grad():
            v = net.latent(o2.encode_all(net, G, te, dev))[0]
        torch.save({"model_state_dict": net.state_dict(), "arch": arch, "arch_name": "w64", "atom_f": arch["atom_f"], "groups": names,
                    "seed": s, "steps": steps, "single_auroc": auc_te, "train_auroc": auc_tr}, ck / f"member_{s}.pt")
        rows.append([s, f"{auc_te:.4f}", f"{auc_tr - auc_te:+.4f}", f"{v.mean().item():+.3f}", f"{v.std().item():.3f}", f"{time.time() - t0:.0f}"])
        rep.line(f"gqn seed {s}: held-out AUROC {auc_te:.4f}")
    rep.section("Count network + global quantile normalisation: single molecules (reference ga_w64 with BatchNorm: 0.846-0.850)")
    rep.table(["seed", "held-out AUROC", "gap (training - held-out)", "hidden mean (held-out)", "hidden sd (held-out)", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
