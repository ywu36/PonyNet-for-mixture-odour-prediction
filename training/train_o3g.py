# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   H = hidden layer (64)   E = open-POM map (256)
#   Q = GoodScents pairs   W = words (138)
# ---------------------------------------------------------------------------------------------
"""O3-G: does GoodScents help the best backbone? (the user, 2026-11-19: "see if GoodScents helps")

Backbone recipe = train_o3p.py --config rank --groups (checkpoints/ga_w64_rank): width-64 network, node-enriched atom
input (36 RDKit group-distance counts), GS-LF single-molecule word loss + POM-order loss (weight 1). Same seeds.
Added here: GoodScents word ranking, the O2-J rank_loss (mixdist/o2j.py) with an ORDER-FREE plain A + B:
    z = (v_A + v_B) / 2 in the hidden layer, words = the same linear word head;
    loss = - sum_w p_w log softmax_T(logits)_w over the 79 target words, p = recommended group counts (normalised):
    the recommended words should rank above all the others. No learned per-molecule weight (no lean-to question).
GoodScents pairs: official training split (both molecules training molecules of the odor_pair split); pairs with a
GS-LF held-out molecule (official single-molecule test set) or a DREAM molecule are dropped, so neither evaluation sees
the pair loss on its molecules. --shuffle: the group-word labels are permuted across pairs (control: structure kept,
labels random). DREAM is never used for training.

Env:     mist (GPU)
Usage:   python training/train_o3g.py --w-scent 0.3 [--shuffle] --seeds 0 1 2
Writes:  checkpoints/ga_w64_rank_scent<w>[_shuf]/member_<s>.pt, results/train_o3g/<date_time>_<tag>.{txt,json}
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

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

PAIR_BATCH = 128


def can(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m is not None else None


def scent_pairs(G, S, smiles_N, te):
    """Training pairs whose molecules are neither GS-LF held-out nor DREAM molecules; held-out pairs (split 1) likewise."""
    RDLogger.DisableLog("rdApp.*")
    banned = {can(x) for x in G.smiles_N[te]} | {can(x) for x in G.smiles_N[G.n_gslf:]}
    bad_N = np.array([can(x) in banned for x in smiles_N[:S.n_odorpair]])
    ok = ~bad_N[S.pair_a] & ~bad_N[S.pair_b]
    return np.where((S.pair_split == 0) & ok)[0], np.where((S.pair_split == 1) & ok)[0], int(bad_N.sum())


def train(G, S, tr, q_tr, PY, T, seed, dev, arch, w_rank, w_scent, Pemb, log, recipe=o2.RECIPE):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = O3M(**arch).float().to(dev); Y = torch.as_tensor(G.Y, device=dev); Pt = torch.as_tensor(Pemb, device=dev)
    PYt = torch.as_tensor(PY, device=dev); Tt = torch.as_tensor(T, device=dev)
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
            ls = torch.zeros((), device=dev)
            if w_scent > 0:
                q = rng.choice(q_tr, PAIR_BATCH, replace=False)
                mols, inv = np.unique(np.concatenate([S.pair_a[q], S.pair_b[q]]), return_inverse=True)
                vm = net.latent(net.encode(*S.batch(mols, dev), len(mols)))[0]
                z = 0.5 * (vm[inv[:PAIR_BATCH]] + vm[inv[PAIR_BATCH:]])          # order-free A + B
                ls = o2j.rank_loss(net.head(net.drop(z)), PYt[q], Tt)
            loss = lw + w_rank * lr_ + w_scent * ls
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; tot += [lw.item(), lr_.item(), ls.item()]
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            log(f"  seed {seed} epoch {epoch + 1}: word loss {tot[0] / nb:.4f}, POM-order loss {tot[1] / nb:.4f}, GoodScents rank loss {tot[2] / nb:.4f}")
    return net.eval()


@torch.no_grad()
def scent_rank_auc(net, S, q, PY, T, dev):
    """Held-out GoodScents pairs: mean over pairs of the AUROC of recommended vs other target words (plain A + B)."""
    from sklearn.metrics import roc_auc_score
    mols, inv = np.unique(np.concatenate([S.pair_a[q], S.pair_b[q]]), return_inverse=True)
    vm = net.latent(o2.encode_all(net, S, mols, dev))[0]
    z = 0.5 * (vm[inv[:len(q)]] + vm[inv[len(q):]]); lg = net.head(z).cpu().numpy()[:, T]; yy = PY[q][:, T] > 0
    return float(np.mean([roc_auc_score(yy[i], lg[i]) for i in range(len(q)) if 0 < yy[i].sum() < len(T)]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--w-scent", type=float, required=True); ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args(); dev = pick_device()
    tag = f"scent{a.w_scent:g}" + ("_shuf" if a.shuffle else "")
    arch, w_rank, _ = train_o3p.CONFIGS["rank"]
    rep = Report("train_o3g", {**vars(a), "arch": arch, "w_rank": w_rank, "pair_batch": PAIR_BATCH, "device": dev, **o2.RECIPE}, tag=tag)
    G, tr, te, names = train_groupatoms.graphs_with_groups(); arch = {**arch, "atom_f": G.node.shape[1]}
    S = o2j.Store(features.features_dir() / "o2j_data.npz"); z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    smiles_N = z["smiles"]; ok_N = z["ok"].astype(bool)
    q_tr, q_te, n_bad = scent_pairs(G, S, smiles_N, te)
    need = np.unique(np.r_[S.pair_a[q_tr], S.pair_b[q_tr], S.pair_a[q_te], S.pair_b[q_te]]); assert ok_N[need].all()
    attach_named(S, smiles_N, names, need); assert S.node.shape[1] == G.node.shape[1]
    PY = S.pair_Y.astype(np.float32); T = S.targets
    if a.shuffle:
        PY = PY.copy(); PY[q_tr] = PY[np.random.default_rng(0).permutation(q_tr)]
    Pemb = train_o3p.pom_targets(G); tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    rep.line(f"GS-LF training molecules {len(tr)}, held-out {len(te)}; GoodScents pairs train {len(q_tr)}, held-out {len(q_te)} "
             f"(molecules dropped as GS-LF held-out or DREAM: {n_bad}); target words {len(T)}; labels shuffled: {a.shuffle}")
    ck = ROOT / f"checkpoints/ga_w64_rank_{tag}"; ck.mkdir(parents=True, exist_ok=True); rows = []
    for s in a.seeds:
        t0 = time.time(); net = train(G, S, tr, q_tr, PY, T, s, dev, arch, w_rank, a.w_scent, Pemb, rep.line)
        auc_te = o2.mean_auroc(o2.predict_probs(net, G, te, dev), G.Y[te]); auc_tr = o2.mean_auroc(o2.predict_probs(net, G, tr_sample, dev), G.Y[tr_sample])
        r_pair, _ = train_o3p.agreement(net, G, te, Pemb, train_o3p.dream_mixtures(G), dev)
        sc = scent_rank_auc(net, S, q_te, S.pair_Y, T, dev)
        torch.save({"model_state_dict": net.state_dict(), "arch": arch, "seed": s, "single_auroc": auc_te, "train_auroc": auc_tr,
                    "pom_order_r": r_pair, "scent_rank_auc": sc, "w_rank": w_rank, "w_scent": a.w_scent, "shuffle": a.shuffle,
                    "arch_name": "w64", "atom_f": arch["atom_f"], "groups": names}, ck / f"member_{s}.pt")
        rows.append([tag, s, f"{auc_te:.4f}", f"{auc_tr - auc_te:+.4f}", f"{r_pair:.3f}", f"{sc:.3f}", f"{time.time() - t0:.0f}"])
        rep.line(f"{tag} seed {s}: held-out AUROC {auc_te:.4f}, POM-order r {r_pair:.3f}, GoodScents held-out rank AUROC {sc:.3f}")
    rep.section("Single molecules; open-POM order; GoodScents held-out word ranking (plain A + B)")
    rep.table(["config", "seed", "held-out AUROC", "gap (training - held-out)", "POM distance Spearman", "GoodScents rank AUROC (held-out pairs)", "seconds"], rows)
    rep.save()


if __name__ == "__main__":
    main()
