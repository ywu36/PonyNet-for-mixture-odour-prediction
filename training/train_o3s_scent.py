# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   Q = GoodScents pairs   T = Ma mixtures
#   H = middle layer (400)   W = words (138)   G = RDKit groups (24)
# ---------------------------------------------------------------------------------------------
"""O3-S loss exploration (findings/o3_exploration_2026-11-19.md, part A, redo with all known flaws fixed).
One script, one set of guards, arms differ only in the stated switch.

Per GoodScents blend (A, B): v = O2-C middle layer (centred, 400); w_A = sigmoid(theta . (x_A - x_B)), w_B = 1 - w_A.
  --merge out     blend profile = w_A p_A + w_B p_B, p = each molecule decoded ALONE (O1.2b's place of merging)
  --merge middle  blend profile = decode(z), z = w_A v_A + w_B v_B
  --target rank   learning-to-rank over the 79 notes on logit(profile) / tau (tau learned)
  --target cos    1 - cos(profile[notes], blend labels[notes])  (direction, as DREAM's distance; no tau)
  --center c      + c * mean_m (1 - cos(v_m, z))  (center loss, members near the merged center)
  --ma sign       Ma: P(A dominates) = w_A = sigmoid(theta . (x_A - x_B)), cross-entropy against [L > 0]. Sign only, no
                  line and no temperature: theta's scale is shared with GoodScents' blend weight, so nothing can trade
                  scale with theta (fix of the O3-E flaw: there a * theta was free and only theta was penalised).
                  Ma mixtures with a molecule in Ma molecule fold 0 are never trained on (held-out Ma check).
Every step: singles (GS-LF train, open-POM word loss). Mixture-side numbers: theta 24 (+ tau 1 for rank).
Guards (all arms): balanced exposure (molecules drawn uniformly, then one of their pairs); early stopping on the sum of
held-out singles (10% of training singles, word loss) and held-out GoodScents training molecules (10%, pairs with both
held out, target loss); BatchNorm in single-molecule statistics for pair passes.

Reported per member (all arms, same tests): single held-out AUROC; GoodScents official held-out pairs (both molecules
unseen): word AUROC (comparable across targets), target loss, memorization gap (held-out minus training target loss);
Ma held-out (fold-0 mixtures): AUROC of theta . (x_A - x_B) vs [L > 0] (arms without Ma: theta from GoodScents alone).

Env:     mist (GPU)
Usage:   python training/train_o3s_scent.py --merge out --target rank --center 0 --ma none --seeds 0 1 2
Writes:  checkpoints/o3s_<merge>_<target>_c<center>[_masign]/member_<seed>.pt,
         results/train_o3s_scent/<date_time>_<arm>.{txt,json}
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus")); sys.path.insert(0, str(ROOT / "training"))
from mixdist import features, o2, o2j, o3  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o2j import word_auroc  # noqa: E402

F32 = torch.float32
MA_HOLDOUT_FOLD = 0


def dream_groups():
    from eval_b1a_plus import piece_matrix
    from mixdist import pushpull as pp
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    return piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)[1]


def arm_name(a):
    return f"o3s_{a.merge}_{a.target}_c{a.center:g}" + ("_masign" if a.ma == "sign" else "")


def middle(net, S, mols, device):
    x, e, s, d, g = S.batch(mols, device)
    with o2j.single_stats(net):
        return net.latent(net.encode(x, e, s, d, g, int(g.max()) + 1))[0]


def pair_forward(net, S, X, a_Q, b_Q, theta, merge, device):
    """-> (blend logits Q x W, center term Q)."""
    mols, inv = np.unique(np.concatenate([a_Q, b_Q]), return_inverse=True)
    v = middle(net, S, mols, device); ia, ib = inv[:len(a_Q)], inv[len(a_Q):]
    w = torch.sigmoid((X[mols[ia]] - X[mols[ib]]) @ theta).unsqueeze(1)
    z = w * v[ia] + (1 - w) * v[ib]
    with o2j.single_stats(net):
        if merge == "middle":
            lg = net.readout(z)[1]
        else:
            p = torch.sigmoid(net.readout(v)[1]); lg = torch.logit((w * p[ia] + (1 - w) * p[ib]).clamp(1e-6, 1 - 1e-6))
    cos = torch.nn.functional.cosine_similarity
    return lg, 0.5 * ((1 - cos(v[ia], z, dim=1)) + (1 - cos(v[ib], z, dim=1)))


def target_loss(lg, Y, T, target, log_tau):
    if target == "rank":
        return o3.rank_loss(lg / torch.exp(log_tau), Y, T)
    return 1 - torch.nn.functional.cosine_similarity(torch.sigmoid(lg)[:, T], Y[:, T], dim=1)


def split_molecules(S, seed, frac=0.1):
    tr = np.where(S.pair_split == 0)[0]; mols = np.unique(np.r_[S.pair_a[tr], S.pair_b[tr]])
    held = np.random.default_rng(seed).choice(mols, int(frac * len(mols)), replace=False)
    ina, inb = np.isin(S.pair_a[tr], held), np.isin(S.pair_b[tr], held)
    return tr[~ina & ~inb], tr[ina & inb]


def split_singles(single_ids, seed, frac=0.1):
    """10% of the training singles held out for early stopping (fix 2026-11-19: stopping on GoodScents alone stopped at
    epochs 12-24 and left single-molecule words undertrained, AUROC 0.76-0.82)."""
    perm = np.random.default_rng(2000 + seed).permutation(single_ids); n = int(frac * len(perm))
    return perm[n:], perm[:n]


def train(G, S, X, single_ids, a, seed, device, recipe=o2.RECIPE, pair_batch=128, ma_batch=32, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = o2.O2C().float().to(device)
    theta = torch.zeros(X.shape[1], dtype=F32, device=device, requires_grad=True)
    log_tau = torch.zeros((), dtype=F32, device=device, requires_grad=(a.target == "rank"))
    Y = torch.as_tensor(G.Y, device=device)
    counts = G.Y[single_ids].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=F32, device=device)
    PY = torch.as_tensor(S.pair_Y, dtype=F32, device=device); T = torch.as_tensor(S.targets, device=device)
    tr_q, val_q = split_molecules(S, seed); val_q = val_q[np.random.default_rng(seed).permutation(len(val_q))[:2000]]
    single_ids, s_va = split_singles(single_ids, seed)
    by_mol = {}
    for q in tr_q:
        by_mol.setdefault(int(S.pair_a[q]), []).append(q); by_mol.setdefault(int(S.pair_b[q]), []).append(q)
    mol_list = np.array(sorted(by_mol))
    ma_tr, _ = S.ma_split(MA_HOLDOUT_FOLD); ma_y = torch.as_tensor(S.ma_L > 0, dtype=F32, device=device)
    params = list(net.parameters()) + [theta] + ([log_tau] if a.target == "rank" else [])
    opt = torch.optim.Adam(params, lr=recipe["lr"])
    bs = recipe["batch"]; nb = math.ceil(len(single_ids) / bs); step = 0; best = (np.inf, -1, None)
    for epoch in range(recipe["epochs"]):
        net.train(); perm = rng.permutation(single_ids)
        perm = np.concatenate([perm, rng.choice(single_ids, nb * bs - len(perm), replace=False)]); tot = np.zeros(4)
        for k in range(nb):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            _, lg = net(G.batch(b, device)); l_single = o2.pom_loss(lg, Y[b], class_w)
            q = np.array([rng.choice(by_mol[int(m)]) for m in rng.choice(mol_list, pair_batch, replace=False)])
            lgq, cen = pair_forward(net, S, X, S.pair_a[q], S.pair_b[q], theta, a.merge, device)
            l_t = target_loss(lgq, PY[q], T, a.target, log_tau).mean(); l_c = cen.mean()
            l_ma = torch.zeros((), device=device)
            if a.ma == "sign":
                t = rng.choice(ma_tr, ma_batch, replace=False)
                l_ma = torch.nn.functional.binary_cross_entropy_with_logits((X[S.ma_a[t]] - X[S.ma_b[t]]) @ theta, ma_y[t])
            loss = l_single + l_t + a.center * l_c + l_ma + o2.regularization(net, recipe["weight_decay"]) + 1e-3 * (theta ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            tot += [l_single.item(), l_t.item(), l_c.item(), l_ma.item()]
        net.eval()
        with torch.no_grad():
            vs = np.mean([o2.pom_loss(net(G.batch(s_va[i:i + 256], device))[1], Y[s_va[i:i + 256]], class_w).item() for i in range(0, len(s_va), 256)])
            vl = np.mean([target_loss(pair_forward(net, S, X, S.pair_a[val_q[i:i + 500]], S.pair_b[val_q[i:i + 500]], theta, a.merge, device)[0],
                                      PY[val_q[i:i + 500]], T, a.target, log_tau).mean().item() for i in range(0, len(val_q), 500)])
        if vs + vl < best[0]:
            best = (vs + vl, epoch, ({k_: v.detach().clone() for k_, v in net.state_dict().items()}, theta.detach().clone(), log_tau.detach().clone()))
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            t_ = tot / nb
            log(f"  {arm_name(a)} seed {seed} epoch {epoch + 1}: single {t_[0]:.3f}, GoodScents target {t_[1]:.4f}, center {t_[2]:.4f}, "
                f"Ma {t_[3]:.4f}; validation singles {vs:.4f} + GoodScents molecules {vl:.4f} (best {best[0]:.4f} at epoch {best[1] + 1})")
    sd, th, lt = best[2]; net.load_state_dict(sd); net.eval()
    return net, th, lt, best[1] + 1, tr_q


@torch.no_grad()
def scent_eval(net, S, X, q_all, theta, log_tau, a, device):
    PY = torch.as_tensor(S.pair_Y, dtype=F32, device=device); T = torch.as_tensor(S.targets, device=device)
    ls, lgs, cs = [], [], []
    for i in range(0, len(q_all), 500):
        q = q_all[i:i + 500]; lg, c = pair_forward(net, S, X, S.pair_a[q], S.pair_b[q], theta, a.merge, device)
        ls.append(target_loss(lg, PY[q], T, a.target, log_tau)); lgs.append(lg); cs.append(c)
    lg = torch.cat(lgs).cpu().numpy()
    return torch.cat(ls).mean().item(), word_auroc(lg, S.pair_Y[q_all], S.targets)[0], torch.cat(cs).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", choices=["out", "middle"], required=True); ap.add_argument("--target", choices=["rank", "cos"], required=True)
    ap.add_argument("--center", type=float, default=0.0); ap.add_argument("--ma", choices=["none", "sign"], default="none")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); a = ap.parse_args()
    dev = pick_device(); arm = arm_name(a); rep = Report("train_o3s_scent", {**vars(a), "arm": arm, "device": dev, **o2.RECIPE}, tag=arm)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False); gn = dream_groups()
    X = torch.as_tensor(o3.group_matrix(z["smiles"], gn), dtype=F32, device=dev)
    gslf = np.arange(G.n_gslf); single_ids, single_te = gslf[~G.audit_test], gslf[G.audit_test]
    te_q = np.where(S.pair_split == 1)[0]; _, ma_te = S.ma_split(MA_HOLDOUT_FOLD)
    ck = ROOT / "checkpoints" / arm; ck.mkdir(parents=True, exist_ok=True)
    rep.line(f"arm {arm}; singles train {len(single_ids)}, held-out {len(single_te)}; GoodScents held-out pairs {len(te_q)} (both molecules unseen); "
             f"Ma held-out (fold {MA_HOLDOUT_FOLD}) mixtures {len(ma_te)}; mixture-side numbers theta {len(gn)}" + (" + tau 1" if a.target == "rank" else ""))
    rows = []
    for s in a.seeds:
        if (ck / f"member_{s}.pt").exists():
            rep.line(f"seed {s}: checkpoint exists in {ck.name}, not retrained"); continue
        t0 = time.time()
        net, th, lt, ep, tr_q = train(G, S, X, single_ids, a, s, dev, log=rep.line)
        auc = o2.mean_auroc(o2.predict_probs(net, G, single_te, dev), G.Y[single_te])
        gl, ga, gc = scent_eval(net, S, X, te_q, th, lt, a, dev)
        sub = tr_q[np.random.default_rng(s).permutation(len(tr_q))[:2000]]; tl, _, tc = scent_eval(net, S, X, sub, th, lt, a, dev)
        dma = ((X[S.ma_a[ma_te]] - X[S.ma_b[ma_te]]) @ th).cpu().numpy(); yma = S.ma_L[ma_te] > 0
        ma_auc = roc_auc_score(yma, dma) if dma.std() > 0 else float("nan")
        torch.save({"model_state_dict": net.state_dict(), "arch": "o2c", "seed": s, "arm": arm, "args": vars(a), "theta": th.cpu().numpy(),
                    "tau": float(torch.exp(lt)), "groups": gn, "best_epoch": ep, "single_auroc": auc, "scent_loss": gl, "scent_auroc": ga,
                    "scent_train_loss": tl, "ma_heldout_auroc": ma_auc}, ck / f"member_{s}.pt")
        rows.append([s, ep, f"{auc:.4f}", f"{ga:.4f}", f"{gl:.4f}", f"{tl:.4f}", f"{gl - tl:+.4f}", f"{tc:.4f} / {gc:.4f}", f"{ma_auc:.3f}",
                     f"{float(th.abs().max()):.3f}"])
        rep.line(f"seed {s}: done in {time.time() - t0:.0f} s; " + ", ".join(map(str, rows[-1][1:])))
    rep.section(f"{arm}: molecule-level checks (data never used in training)")
    rep.table(["seed", "best epoch", "single held-out AUROC (O2-C-Single 0.875-0.883)", "GoodScents held-out word AUROC",
               "GoodScents held-out target loss", "training-pair target loss", "memorization gap", "center term train / held-out",
               f"Ma held-out AUROC (fold {MA_HOLDOUT_FOLD}, {len(ma_te)} mixtures)", "max |theta|"], rows)
    rep.save()


if __name__ == "__main__":
    main()
