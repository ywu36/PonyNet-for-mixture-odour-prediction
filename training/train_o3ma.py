# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   H = hidden layer (64)   T = Ma mixtures
# ---------------------------------------------------------------------------------------------
"""O3-Ma: train the network jointly on single molecules and Ma (the user, 2026-11-19: "jointly training with Ma").

Base recipe = training/train_groupatoms.py --arch w64 (checkpoints/ga_w64): width-64 network, node-enriched atom input
(36 RDKit group-distance counts), GS-LF single-molecule word loss. Same seeds, same optimiser schedule (o2.RECIPE).
Added: Ma direction loss in the hidden layer, order-free plain mean, no weights anywhere:
    z = (v_A + v_B) / 2;  m = cos(z, v_A) - cos(z, v_B);  loss_Ma = softplus(-y m / 0.1), y = +1 if A dominates (L > 0)
With a plain mean, m > 0 needs |v_A| > |v_B| (or v_B pointing away): the network must carry strength as vector length.
Ma mixtures with a molecule in Ma molecule fold 0 are held out (unseen-molecule check); 32 training mixtures per step.
--shuffle: Ma signs permuted across the training mixtures (control). DREAM is never used for training.

Env:     mist (GPU)
Usage:   python training/train_o3ma.py --w-ma 1 [--shuffle] [--pom] --seeds 0 1 2
Writes:  checkpoints/ga_w64[_rank]_ma<w>[_shuf]/member_<s>.pt, results/train_o3ma/<date_time>_<tag>.{txt,json}
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "training", ROOT / "evaluation/O3"):
    sys.path.insert(0, str(p))
from mixdist import features, o2, o2j  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o3m import O3M  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3p  # noqa: E402
from eval_o3s_attn_ma import attach_named  # noqa: E402

MA_BATCH, MA_T, MA_FOLD = 32, 0.1, 0


def ma_margin(va, vb):
    z = 0.5 * (va + vb); c = torch.nn.functional.cosine_similarity
    return c(z, va, dim=1) - c(z, vb, dim=1)


def train(G, S, tr, t_tr, ysign, seed, dev, arch, w_rank, w_ma, Pemb, log, recipe=o2.RECIPE):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = O3M(**arch).float().to(dev); Y = torch.as_tensor(G.Y, device=dev); Pt = torch.as_tensor(Pemb, device=dev)
    ys = torch.as_tensor(ysign, dtype=torch.float32, device=dev)
    counts = G.Y[tr].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=recipe["lr"]); bs, step = recipe["batch"], 0; nb = math.ceil(len(tr) / bs)
    for epoch in range(recipe["epochs"]):
        net.train(); perm = rng.permutation(tr); perm = np.concatenate([perm, rng.choice(tr, nb * bs - len(perm), replace=False)])
        tot = np.zeros(3)
        for k in range(nb):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            v = net.latent(net.encode(*G.batch(b, dev), len(b)))[0]; logits = net.head(net.drop(v))
            lw = o2.pom_loss(logits, Y[b], class_w) + o2.regularization(net, recipe["weight_decay"])
            lr_ = train_o3p.order_loss(v, Pt[b]) if w_rank > 0 else torch.zeros((), device=dev)
            lm = torch.zeros((), device=dev)
            if w_ma > 0:
                t = rng.choice(t_tr, MA_BATCH, replace=False)
                mols, inv = np.unique(np.concatenate([S.ma_a[t], S.ma_b[t]]), return_inverse=True)
                vm = net.latent(net.encode(*S.batch(mols, dev), len(mols)))[0]
                m = ma_margin(vm[inv[:MA_BATCH]], vm[inv[MA_BATCH:]])
                lm = torch.nn.functional.softplus(-ys[t] * m / MA_T).mean()
            loss = lw + w_rank * lr_ + w_ma * lm
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += [lw.item(), lr_.item(), lm.item()]
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            log(f"  seed {seed} epoch {epoch + 1}: word loss {tot[0] / nb:.4f}, POM-order loss {tot[1] / nb:.4f}, Ma loss {tot[2] / nb:.4f}")
    return net.eval()


@torch.no_grad()
def ma_auc(net, S, idx, dev):
    mols, inv = np.unique(np.concatenate([S.ma_a[idx], S.ma_b[idx]]), return_inverse=True)
    vm = net.latent(o2.encode_all(net, S, mols, dev))[0]
    m = ma_margin(vm[inv[:len(idx)]], vm[inv[len(idx):]]).cpu().numpy()
    return float(roc_auc_score((S.ma_L[idx] > 0).astype(int), m))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--w-ma", type=float, required=True); ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--pom", action="store_true", help="add the POM-order loss (train_o3p rank recipe)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args(); dev = pick_device()
    tag = ("rank_" if a.pom else "") + f"ma{a.w_ma:g}" + ("_shuf" if a.shuffle else "")
    arch, w_rank, _ = train_o3p.CONFIGS["rank"]; w_rank = w_rank if a.pom else 0.0
    rep = Report("train_o3ma", {**vars(a), "arch": arch, "w_rank": w_rank, "ma_batch": MA_BATCH, "ma_T": MA_T, "ma_holdout_fold": MA_FOLD,
                                "device": dev, **o2.RECIPE}, tag=tag)
    G, tr, te, names = train_groupatoms.graphs_with_groups(); arch = {**arch, "atom_f": G.node.shape[1]}
    S = o2j.Store(features.features_dir() / "o2j_data.npz"); z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    attach_named(S, z["smiles"], names, np.unique(np.r_[S.ma_a, S.ma_b])); assert S.node.shape[1] == G.node.shape[1]
    t_tr, t_te = S.ma_split(MA_FOLD); ysign = np.where(S.ma_L > 0, 1.0, -1.0)
    if a.shuffle:
        ysign = ysign.copy(); ysign[t_tr] = ysign[np.random.default_rng(0).permutation(t_tr)]
    Pemb = train_o3p.pom_targets(G) if a.pom else np.zeros((len(G.smiles_N), 1), np.float32)
    tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    rep.line(f"GS-LF training molecules {len(tr)}, held-out {len(te)}; Ma training mixtures {len(t_tr)}, held-out (fold {MA_FOLD} "
             f"molecules) {len(t_te)}; Ma signs shuffled: {a.shuffle}; POM-order: {a.pom}")
    ck = ROOT / f"checkpoints/ga_w64_{tag}"; ck.mkdir(parents=True, exist_ok=True); rows = []
    for s in a.seeds:
        t0 = time.time(); net = train(G, S, tr, t_tr, ysign, s, dev, arch, w_rank, a.w_ma, Pemb, rep.line)
        auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te]); auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
        ma_tr, ma_te = ma_auc(net, S, t_tr, dev), ma_auc(net, S, t_te, dev)
        torch.save({"model_state_dict": net.state_dict(), "arch": arch, "seed": s, "single_auroc": auc_te, "train_auroc": auc_tr,
                    "ma_auc_train": ma_tr, "ma_auc_heldout": ma_te, "w_ma": a.w_ma, "w_rank": w_rank, "shuffle": a.shuffle,
                    "arch_name": "w64", "atom_f": arch["atom_f"], "groups": names}, ck / f"member_{s}.pt")
        rows.append([tag, s, f"{auc_te:.4f}", f"{auc_tr - auc_te:+.4f}", f"{ma_tr:.3f}", f"{ma_te:.3f}", f"{time.time() - t0:.0f}"])
        rep.line(f"{tag} seed {s}: single held-out AUROC {auc_te:.4f}, Ma AUROC training {ma_tr:.3f}, held-out molecules {ma_te:.3f}")
    rep.section("Single molecules; Ma direction with a plain mean (training mixtures / mixtures with unseen fold-0 molecules)")
    rep.table(["config", "seed", "single held-out AUROC", "gap", "Ma AUROC training", "Ma AUROC unseen molecules", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
