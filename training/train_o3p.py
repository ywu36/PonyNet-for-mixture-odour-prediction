# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   H = hidden layer (64)   E = open-POM map (256)
#   S = molecules in one mixture   W = words (138)
# ---------------------------------------------------------------------------------------------
"""O3-P: the width-64 backbone (train_o3m.O3M, centred v), trained on single molecules, with the user's additions
(2026-11-19):
  cosine word head  (NormFace style): words read from v's direction only, so the hidden space is built for cosine
                    comparison, which is how mixtures are compared.
  POM-order loss    the hidden layer keeps open-POM's map distances in ORDER, not in value ("preserve the pom's
                    distance in general but not strict"): for an anchor molecule i and two others j, k in the batch,
                    if open-POM puts j clearly closer to i than k (gap > 0.05 in cosine distance), the hidden layer
                    should too: softplus(-sign(Dp_ik - Dp_ij) (Dh_ik - Dh_ij) / 0.05). Training molecules only.
  salience loss     inside a real DREAM development mixture (composition only, no similarity labels; lockbox never
                    used), a molecule that stands out from the mixture's mean on open-POM's map should stand out in the
                    hidden layer too: s_i = 1 - cos(x_i, mean_S x), same rank form within the mixture (gap > 0.02).
open-POM's map = its 256-number embedding (10-model ensemble). It is used only as a training target; the network
needs no open-POM at test time. The mixture side is unchanged (fitted later, frozen network, <= 27 numbers).

Env:     mist (GPU)
Usage:   python training/train_o3p.py --config cos --seeds 0 1 2
Writes:  checkpoints/o3p_<config>/member_<s>.pt, results/train_o3p/<date_time>.{txt,json}
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import features, o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o3m import O3M  # noqa: E402

BASE = {"d": 64, "h": 64, "steps": 3, "edge_h": 16}
CONFIGS = {"cos": ({**BASE, "cos_head": True}, 0.0, 0.0),           # (arch, weight of POM-order loss, weight of salience loss)
           "rank": (BASE, 1.0, 0.0),
           "cos_rank": ({**BASE, "cos_head": True}, 1.0, 0.0),
           "cos_rank_sal": ({**BASE, "cos_head": True}, 1.0, 1.0),
           "rank_sal": (BASE, 1.0, 1.0),
           "sp03": ({**BASE, "positive": True, "sparse_l1": 0.3}, 0.0, 0.0),            # receptor-like: ReLU + L1 activity
           "sp03_rank": ({**BASE, "positive": True, "sparse_l1": 0.3}, 1.0, 0.0),
           "sp1": ({**BASE, "positive": True, "sparse_l1": 1.0}, 0.0, 0.0),
           "sp1_rank": ({**BASE, "positive": True, "sparse_l1": 1.0}, 1.0, 0.0),
           "rank03": (BASE, 0.3, 0.0),        # softer POM-order pull (user, 2026-11-20: try a smaller weight)
           "rank_half": (BASE, 1.0, 0.0), "rank_centred": (BASE, 1.0, 0.0), "rank_half_centred": (BASE, 1.0, 0.0),
           "balanced": (BASE, 0.0, 0.0),      # plain O3 with the balanced word loss (user, 2026-11-20)
           "word_rank": (BASE, 0.0, 0.0),
           # O5 variants (user, 2026-11-20: "O5 prior + DREAM but with count and POM"): attention pooling over atoms
           # (as train_o3m --config attpool) + counts (--groups) and/or POM order 0.3
           "att": ({**BASE, "pool": "attn"}, 0.0, 0.0),
           "att_rank03": ({**BASE, "pool": "attn"}, 0.3, 0.0),
           # O5-big + POM 0.3: O4's encoder with attention pooling
           "o4att_rank03": ({"d": 100, "h": 64, "steps": 5, "edge_h": 75, "ffn": [392, 392], "pool": "attn"}, 0.3, 0.0)}     # plain O3 + order loss on the word profiles (see EXTRA)                                    # linear head (the cosine head cost single AUROC)
# extra options per config: units = POM order acts only on the first `units` hidden units (the rest stay free for the
# counts' chemistry and strength); centre = target order from open-POM vectors minus their average over GS-LF training
# molecules (copies which molecules are near which, not POM's collapse onto one shared direction)
# balanced = word loss without open-POM's frequency weights (which favour common words 15x over the median word): every
# word counts equally and its positive cases are weighted by (molecules without the word) / (molecules with it), the
# standard per-label positive weight of molecular multi-task networks; total rescaled to the same expected size as
# open-POM's loss so the L1 + L2 penalty keeps its relative strength
# count dropout (user 2026-11-20, finding 6 sec. 22: counts memorise single molecules, train-held-out AUROC gap 0.048 vs
# 0.025): with --groups, each group-distance count feature of each atom is zeroed with probability p during training
# (inverse-scaled), so the network cannot learn molecules by their exact group pattern; open-POM's 134 atom features untouched
CONFIGS.update({"att_cdrop05": ({**BASE, "pool": "attn"}, 0.0, 0.0), "att_cdrop08": ({**BASE, "pool": "attn"}, 0.0, 0.0)})
N_OPENPOM_ATOM_F = 134
EXTRA = {"att_cdrop05": {"count_drop": 0.5}, "att_cdrop08": {"count_drop": 0.8},
         "rank_half": {"units": 32}, "rank_centred": {"centre": True}, "rank_half_centred": {"units": 32, "centre": True},
         "balanced": {"balanced": True}, "word_rank": {"w_word": 1.0}}
# word_rank (user, 2026-11-20): the order loss acts on the 138 word predictions put on O1.2b's per-word scale, which is
# what O1.2b's rule reads, instead of on the hidden layer. Per-word scale during training = each word's logit centred and
# scaled to unit spread over the batch (a smooth stand-in for O1.2b's per-word quantile, which cannot be trained through);
# target = open-POM's 138 word predictions for the same GS-LF training molecules, same scaling on their logits; same loose
# rank form and gap as the hidden-layer POM order. Main loss unchanged (open-POM's word loss). No DREAM data.
TAU, GAP_PAIR, GAP_SAL, N_MIX = 0.05, 0.05, 0.02, 8


def cos_dist(x_NxD):
    u = torch.nn.functional.normalize(x_NxD, dim=1); return 1 - u @ u.T


def order_loss(v_BxH, p_BxE):
    """Anchor i, others j, k: hidden distances keep open-POM's order where open-POM's gap > GAP_PAIR."""
    Dh, Dp = cos_dist(v_BxH), cos_dist(p_BxE)
    dp = Dp[:, None, :] - Dp[:, :, None]; dh = Dh[:, None, :] - Dh[:, :, None]      # [i, j, k] = D_ik - D_ij
    m = dp.abs() > GAP_PAIR
    return torch.nn.functional.softplus(-torch.sign(dp[m]) * dh[m] / TAU).mean()


def salience(x_SxD):
    return 1 - torch.nn.functional.cosine_similarity(x_SxD, x_SxD.mean(0, keepdim=True), dim=1)


def salience_loss(v_NxH, p_NxE, mixes):
    """mixes: list of index arrays into v / p (one per mixture)."""
    tot, n = 0.0, 0
    for S in mixes:
        sh, sp = salience(v_NxH[S]), salience(p_NxE[S])
        dp = sp[None, :] - sp[:, None]; dh = sh[None, :] - sh[:, None]; m = dp.abs() > GAP_SAL
        if m.any():
            tot = tot + torch.nn.functional.softplus(-torch.sign(dp[m]) * dh[m] / TAU).mean(); n += 1
    return tot / max(n, 1)


def pom_targets(G):
    """open-POM 256 map for every graph row: GS-LF rows by SMILES (openpom_extra), DREAM rows by CID (openpom)."""
    fd = features.features_dir(); ex = np.load(fd / "openpom_extra.npz", allow_pickle=True); op = features.load("openpom")
    ex_smi, ex_emb = ex["smiles"], ex["emb"]                     # read once (npz re-reads an array on every access)
    by_smi = {s: i for i, s in enumerate(ex_smi)}; by_cid = {int(c): i for i, c in enumerate(op.cids_M)}
    gs = ex_emb[[by_smi[s] for s in G.smiles_N[:G.n_gslf]]]
    dr = np.asarray(op.emb_MxE)[[by_cid[int(c)] for c in G.dream_cids]]
    return np.concatenate([gs, dr]).astype(np.float32)


def dream_mixtures(G):
    row = {int(c): G.n_gslf + i for i, c in enumerate(G.dream_cids)}
    mixes = {m for p in dev_pairs() for m in (p.a, p.b) if len(m) >= 3}
    return [np.array(sorted(row[c] for c in m)) for m in sorted(mixes, key=lambda m: sorted(m))]


def word_scale(z_BxW):
    """Each word centred and scaled to unit spread over the batch (smooth stand-in for O1.2b's per-word quantile)."""
    return (z_BxW - z_BxW.mean(0, keepdim=True)) / (z_BxW.std(0, keepdim=True) + 1e-4)


def pom_word_logits(G):
    """open-POM's 138 word predictions as logits for GS-LF rows (by SMILES); DREAM rows zero (not used)."""
    ex = np.load(features.features_dir() / "openpom_extra.npz", allow_pickle=True)
    smi, pr = ex["smiles"], np.clip(ex["probs"].astype(np.float64), 1e-6, 1 - 1e-6); by = {s: i for i, s in enumerate(smi)}
    z = np.log(pr / (1 - pr))[[by[s] for s in G.smiles_N[:G.n_gslf]]]
    return np.concatenate([z, np.zeros((len(G.smiles_N) - G.n_gslf, z.shape[1]))]).astype(np.float32)


def balanced_loss(logits_BxW, y_BxW, pos_w_W, scale):
    """Same per-element cross-entropy as open-POM's loss, weighted pos_w_W on positive cases and 1 on negative cases,
    summed over words, mean over the batch, times scale."""
    two = torch.stack([1 - logits_BxW, logits_BxW], 1)
    ce = torch.nn.functional.cross_entropy(two, y_BxW.long(), reduction="none")
    w = torch.where(y_BxW > 0.5, pos_w_W.expand_as(ce), torch.ones_like(ce))
    return scale * (ce * w).sum(1).mean()


def train(G, tr, seed, dev, arch, w_rank, w_sal, Pemb, mixes, log, recipe=o2.RECIPE, units=None, balanced=False, w_word=0.0, Pw=None,
          count_drop=0.0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = O3M(**arch).float().to(dev); Y = torch.as_tensor(G.Y, device=dev); Pt = torch.as_tensor(Pemb, device=dev)
    Pwt = torch.as_tensor(Pw, device=dev) if w_word > 0 else None
    counts = G.Y[tr].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=dev)
    c1 = np.maximum(counts, 1); f = c1 / len(tr); pos_w = torch.as_tensor((len(tr) - c1) / c1, dtype=torch.float32, device=dev)
    scale = float(class_w.sum().item() / (2 * (1 - f)).sum())     # expected loss weight per molecule matches open-POM's
    if balanced:
        log(f"  balanced word loss: positive weight min {pos_w.min().item():.1f} median {pos_w.median().item():.1f} "
            f"max {pos_w.max().item():.1f}; scale {scale:.4f}")
    opt = torch.optim.Adam(net.parameters(), lr=recipe["lr"]); bs, step = recipe["batch"], 0; nb = math.ceil(len(tr) / bs)
    for epoch in range(recipe["epochs"]):
        net.train(); perm = rng.permutation(tr); perm = np.concatenate([perm, rng.choice(tr, nb * bs - len(perm), replace=False)])
        tot = np.zeros(3)
        for k in range(nb):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]; ids, local = b, []
            if w_sal > 0:
                pick = [mixes[i] for i in rng.choice(len(mixes), N_MIX, replace=False)]
                extra = np.unique(np.concatenate(pick)); ids = np.concatenate([b, extra])
                pos = {r: bs + i for i, r in enumerate(extra)}; local = [np.array([pos[r] for r in S]) for S in pick]
            x, e, s, d_, g = G.batch(ids, dev)
            if count_drop > 0:                         # dropout on the group-distance count features only
                assert x.shape[1] > N_OPENPOM_ATOM_F, "count_drop needs --groups"
                keep = (torch.rand(x.shape[0], x.shape[1] - N_OPENPOM_ATOM_F, device=dev) > count_drop).float() / (1 - count_drop)
                x = torch.cat([x[:, :N_OPENPOM_ATOM_F], x[:, N_OPENPOM_ATOM_F:] * keep], 1)
            v = net.latent(net.encode(x, e, s, d_, g, len(ids)))[0]; logits = net.head(net.drop(v[:bs]))
            lw = (balanced_loss(logits, Y[b], pos_w, scale) if balanced else o2.pom_loss(logits, Y[b], class_w)) + o2.regularization(net, recipe["weight_decay"])
            lr_ = w_rank * order_loss(v[:bs, :units], Pt[b]) if w_rank > 0 else torch.zeros((), device=dev)   # weighted here
            if w_word > 0:                             # order loss on the word profiles (word_rank); logged in the same slot
                lr_ = lr_ + w_word * order_loss(word_scale(logits), word_scale(Pwt[b]))
            ls = salience_loss(v, Pt[ids], local) if w_sal > 0 else torch.zeros((), device=dev)
            loss = lw + lr_ + w_sal * ls
            if net.sparse_l1 > 0:                      # same activity penalty and 10-epoch warm-up as train_o3m (o2.train_member)
                loss = loss + net.sparse_l1 * min(1.0, (step + 1) / (10 * nb)) * net.last_kl
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += [lw.item(), lr_.item(), float(ls.item() if torch.is_tensor(ls) else ls)]
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            log(f"  seed {seed} epoch {epoch + 1}: word loss {tot[0] / nb:.4f}, order loss (hidden or words) {tot[1] / nb:.4f}, salience loss {tot[2] / nb:.4f}")
    return net.eval()


@torch.no_grad()
def agreement(net, G, te, Pemb, mixes, dev):
    """Spearman between hidden and open-POM cosine distances over held-out molecule pairs; mean within-mixture
    Spearman of salience over DREAM development mixtures (composition only)."""
    v = net.latent(o2.encode_all(net, G, te, dev))[0]; iu = np.triu_indices(len(te), 1)
    r_pair = spearmanr(cos_dist(v).cpu().numpy()[iu], cos_dist(torch.as_tensor(Pemb[te])).numpy()[iu])[0]
    rows = np.unique(np.concatenate(mixes)); vd = net.latent(o2.encode_all(net, G, rows, dev))[0].cpu(); pos = {r: i for i, r in enumerate(rows)}
    rs = [spearmanr(salience(vd[[pos[r] for r in S]]).numpy(), salience(torch.as_tensor(Pemb[S])).numpy())[0] for S in mixes]
    return float(r_pair), float(np.nanmean(rs))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--groups", action="store_true", help="node-enriched input (train_groupatoms: atom group-distance counts) -> checkpoints/ga_w64_<config>")
    a = ap.parse_args(); dev = pick_device()
    arch, w_rank, w_sal = CONFIGS[a.config]
    rep = Report("train_o3p", {**vars(a), "arch": arch, "w_rank": w_rank, "w_sal": w_sal, "tau": TAU, "gap_pair": GAP_PAIR,
                               "gap_sal": GAP_SAL, "n_mix": N_MIX, "device": dev, **o2.RECIPE}, tag=("ga_" if a.groups else "") + a.config)
    if a.groups:
        import train_groupatoms
        assert not arch.get("cos_head"), "cosine head not supported with counts"   # load with train_o3m.load (reads arch)
        G, tr, te, names = train_groupatoms.graphs_with_groups(); arch = {**arch, "atom_f": G.node.shape[1]}
    else:
        G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    Pemb = pom_targets(G); ex = EXTRA.get(a.config, {})
    if ex.get("centre"):
        Pemb = Pemb - Pemb[tr].mean(0, keepdims=True)
    mixes = dream_mixtures(G); tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    rep.line(f"GS-LF training molecules {len(tr)}, held-out {len(te)}; DREAM development mixtures with >= 3 molecules {len(mixes)}")
    ck = ROOT / (f"checkpoints/ga_w64_{a.config}" if a.groups else f"checkpoints/o3p_{a.config}"); ck.mkdir(parents=True, exist_ok=True); rows = []
    for s in a.seeds:
        t0 = time.time(); net = train(G, tr, s, dev, arch, w_rank, w_sal, Pemb, mixes, rep.line, units=ex.get("units"), balanced=ex.get("balanced", False),
                                                                     w_word=ex.get("w_word", 0.0), Pw=pom_word_logits(G) if ex.get("w_word") else None,
                                                                     count_drop=ex.get("count_drop", 0.0))
        auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te]); auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
        r_pair, r_sal = agreement(net, G, te, Pemb, mixes, dev)
        torch.save({"model_state_dict": net.state_dict(), "arch": arch, "seed": s, "single_auroc": auc_te, "train_auroc": auc_tr,
                    "pom_order_r": r_pair, "salience_r": r_sal, "w_rank": w_rank, "w_sal": w_sal, "extra": ex,
                    **({"arch_name": "w64", "atom_f": arch["atom_f"], "groups": names} if a.groups else {})}, ck / f"member_{s}.pt")
        rows.append([a.config, s, f"{auc_te:.4f}", f"{auc_tr - auc_te:+.4f}", f"{r_pair:.3f}", f"{r_sal:.3f}", f"{time.time() - t0:.0f}"])
        rep.line(f"{a.config} seed {s}: held-out AUROC {auc_te:.4f}, gap {auc_tr - auc_te:+.4f}, POM-order r {r_pair:.3f}, salience r {r_sal:.3f}")
    rep.section("Single molecules; agreement with open-POM's map")
    rep.table(["config", "seed", "held-out AUROC", "gap (training - held-out)", "POM distance Spearman (held-out pairs)",
               "salience Spearman (DREAM dev mixtures)", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
