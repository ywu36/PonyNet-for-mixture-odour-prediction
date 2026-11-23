# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  B = molecules in a batch   Q = GoodScents pairs   T = Ma mixtures   W = words (138)
#   G = RDKit groups (24, DREAM's)
# ---------------------------------------------------------------------------------------------
"""O3-E (design/o3_design.md, "O3-E"): the molecule encoder (O2-C network, structure -> 138-word profile) is trained
jointly on all real molecule-level information; the mixture side stays O1.2b's (averaging after decoding, group
loudness theta, direction + length distance). User rules: mixture-side parameters ~ O1.2b's; capacity only in the
encoder; decoders simple; no memorizing.

Streams every step (each loss weight 1):
  singles     GS-LF training molecules, open-POM word loss (as O2)                             -> network
  GoodScents  128 train blends: p(blend) = w p(m1) + (1 - w) p(m2), w = sigmoid(theta . (x1 - x2)),
              p = the network's decoded word probabilities of each molecule ALONE; learning-to-rank over the 79
              target notes on logit(p) / tau                                                 -> network, theta, tau
  Ma          32 mixtures: L_hat = a * theta . (x1 - x2) + c, (L_hat - L)^2 / var(L)           -> theta, a, c
Mixture-side parameters: theta 24 (+ tau 1 for GoodScents, a, c for Ma). No DREAM in training; DREAM later fits
O1.2b's 27 numbers (or only beta + line with theta from here). GoodScents molecules pass the network one by one (no
merged vectors), so BatchNorm sees single molecules only, as in single-molecule training.

Env:     mist (GPU)
Usage:   python training/train_o3e.py --seeds 0 1 2 3 4 [--no-ma]
         python training/train_o3e.py --report
Writes:  checkpoints/o3e[_noma]/member_<seed>.pt, results/train_o3e/<date_time>.{txt,json}
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus")); sys.path.insert(0, str(ROOT / "training"))
from mixdist import features, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from train_o2j import word_auroc  # noqa: E402


def dream_groups():
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    return piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)[1]


def batch_drop(G, ids, device, p_drop=0.0, rng=None, frac=0.25):
    """G.batch with MolCLR-style subgraph removal: each molecule, with probability p_drop, gets a connected subgraph of
    ~frac of its atoms (breadth-first from a random atom) set to zero features; bonds kept (as MSN-DDI's MSAN-SD)."""
    x, e, s, d, g = G.batch(ids, device)
    if p_drop <= 0:
        return x, e, s, d, g
    keep = np.ones(len(x), dtype=np.float32); base = 0
    for i in ids:
        n0, n1, e0, e1 = G.node_off[i], G.node_off[i + 1], G.edge_off[i], G.edge_off[i + 1]; n = n1 - n0
        if n > 2 and rng.random() < p_drop:
            src, dst = G.edge_index[0, e0:e1], G.edge_index[1, e0:e1]; k = max(1, int(round(frac * n)))
            seen, queue = [], [int(rng.integers(n))]
            while queue and len(seen) < k:
                a_ = queue.pop(0)
                if a_ in seen:
                    continue
                seen.append(a_); queue += [int(t) for t in dst[src == a_] if t not in seen]
            keep[base + np.array(seen)] = 0.0
        base += n
    return x * torch.as_tensor(keep, device=device).unsqueeze(1), e, s, d, g


def blend_logits(net, S, X, a, b, theta, tau, device, p_drop=0.0, rng=None):
    """Decode every molecule of the pairs ALONE, average the two word profiles with group loudness."""
    mols, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    _, lg = net(batch_drop(S, mols, device, p_drop, rng)); p = torch.sigmoid(lg)
    ia, ib = inv[:len(a)], inv[len(a):]
    w = torch.sigmoid((X[mols[ia]] - X[mols[ib]]) @ theta).unsqueeze(1)
    pb = (w * p[ia] + (1 - w) * p[ib]).clamp(1e-6, 1 - 1e-6)
    return torch.logit(pb) / tau


MA_HOLDOUT_FOLD = 0      # Ma mixtures with a molecule in this fold are never trained on (as O3-S)


def cap_pairs(S, pairs, cap, rng):
    """De-duplicate GoodScents: a fixed random subset where every molecule is in at most `cap` pairs."""
    cnt, keep = {}, []
    for q in rng.permutation(pairs):
        a, b = int(S.pair_a[q]), int(S.pair_b[q])
        if cnt.get(a, 0) < cap and cnt.get(b, 0) < cap:
            cnt[a] = cnt.get(a, 0) + 1; cnt[b] = cnt.get(b, 0) + 1; keep.append(q)
    return np.array(sorted(keep))


@torch.no_grad()
def train_pair_loss(net, S, X, pairs, theta, tau, device, n=3000, seed=0):
    """Eval-mode rank loss on (a sample of) the training pairs: with the held-out loss gives the memorization gap."""
    q_ = np.random.default_rng(seed).choice(pairs, min(n, len(pairs)), replace=False); out = []
    for k in range(0, len(q_), 256):
        q = q_[k:k + 256]; out.append(blend_logits(net, S, X, S.pair_a[q], S.pair_b[q], theta, tau, device))
    return o3.rank_loss(torch.cat(out), torch.as_tensor(S.pair_Y[q_], dtype=torch.float32, device=device),
                        torch.as_tensor(S.targets, device=device)).mean().item()


@torch.no_grad()
def ma_heldout_auroc(S, X, theta):
    """AUROC of theta . (x_A - x_B) vs [L > 0] on Ma mixtures with a molecule in the held-out fold."""
    from sklearn.metrics import roc_auc_score
    te = S.ma_split(MA_HOLDOUT_FOLD)[1]; d = ((X[S.ma_a[te]] - X[S.ma_b[te]]) @ theta).cpu().numpy()
    return float(roc_auc_score(S.ma_L[te] > 0, d)) if d.std() > 0 else float("nan"), len(te)


def inner_split(S, single_ids, rng, frac=0.15):
    """Molecule-disjoint inner validation: frac of the GoodScents train molecules and 10% of the single molecules are held
    out; training pairs = both molecules inner-train, validation pairs = both molecules inner-val (at most 3000)."""
    tr_q = np.where(S.pair_split == 0)[0]; mols = np.unique(np.concatenate([S.pair_a[tr_q], S.pair_b[tr_q]]))
    val_m = set(rng.choice(mols, int(frac * len(mols)), replace=False).tolist())
    ina, inb = np.isin(S.pair_a[tr_q], list(val_m)), np.isin(S.pair_b[tr_q], list(val_m))
    q_tr, q_va = tr_q[~ina & ~inb], tr_q[ina & inb]
    q_va = rng.choice(q_va, min(3000, len(q_va)), replace=False)
    s_perm = rng.permutation(single_ids); n_va = len(single_ids) // 10
    return q_tr, q_va, s_perm[n_va:], s_perm[:n_va]


def train(G, S, X, single_ids, seed, device, use_ma=True, recipe=o2.RECIPE, pair_batch=128, ma_batch=32, log=print,
          p_drop=0.0, wd_mult=1.0, early_stop=False, cap=0, ma_sign=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    q_va = s_va = None
    if early_stop:
        pairs_tr, q_va, single_ids, s_va = inner_split(S, single_ids, np.random.default_rng(1000 + seed))
        log(f"  seed {seed}: inner split: GoodScents train pairs {len(pairs_tr)}, val pairs {len(q_va)} (molecules unseen); singles train {len(single_ids)}, val {len(s_va)}")
    net = o2.O2C().float().to(device)
    theta = torch.zeros(X.shape[1], device=device, dtype=torch.float32, requires_grad=True)
    log_tau = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
    ma_line = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32, requires_grad=True)
    Y = torch.as_tensor(G.Y, device=device)
    counts = G.Y[single_ids].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=device)
    pairs = pairs_tr if early_stop else np.where(S.pair_split == 0)[0]; PY = torch.as_tensor(S.pair_Y, dtype=torch.float32, device=device)
    best = (np.inf, -1, None)
    if cap:
        pairs = cap_pairs(S, pairs, cap, np.random.default_rng(2000 + seed))
        log(f"  seed {seed}: de-duplicated GoodScents: at most {cap} pairs per molecule -> {len(pairs)} pairs")
    ma_tr = S.ma_split(MA_HOLDOUT_FOLD)[0] if ma_sign else np.arange(len(S.ma_L))
    ma_y = torch.as_tensor(S.ma_L > 0, dtype=torch.float32, device=device)
    T = torch.as_tensor(S.targets, device=device); L = torch.as_tensor(S.ma_L, dtype=torch.float32, device=device); varL = float(S.ma_L.var())
    opt = torch.optim.Adam(list(net.parameters()) + [theta, log_tau, ma_line], lr=recipe["lr"])
    bs = recipe["batch"]; n_batches = math.ceil(len(single_ids) / bs); step = 0
    for epoch in range(recipe["epochs"]):
        net.train(); perm = rng.permutation(single_ids)
        perm = np.concatenate([perm, rng.choice(single_ids, n_batches * bs - len(perm), replace=False)]); tot = np.zeros(4)
        for k in range(n_batches):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            _, lg = net(batch_drop(G, b, device, p_drop, rng)); l_single = o2.pom_loss(lg, Y[b], class_w)
            q = rng.choice(pairs, pair_batch, replace=False)
            l_scent = o3.rank_loss(blend_logits(net, S, X, S.pair_a[q], S.pair_b[q], theta, torch.exp(log_tau), device, p_drop, rng), PY[q], T).mean()
            l_ma = torch.zeros((), device=device)
            if use_ma:
                t = rng.choice(ma_tr, ma_batch, replace=False)
                dlt = (X[S.ma_a[t]] - X[S.ma_b[t]]) @ theta
                if ma_sign:     # P(A dominates) = sigmoid(theta . (x_A - x_B)); sign only, no free line (O3-S fix)
                    l_ma = torch.nn.functional.binary_cross_entropy_with_logits(dlt, ma_y[t])
                else:
                    l_ma = ((ma_line[0] * dlt + ma_line[1] - L[t]) ** 2).mean() / varL
            loss = l_single + l_scent + l_ma + o2.regularization(net, wd_mult * recipe["weight_decay"]) + 1e-3 * (theta ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            tot += [loss.item(), l_single.item(), l_scent.item(), l_ma.item()]
        vl = ""
        if early_stop:
            net.eval()
            with torch.no_grad():
                v_s = np.mean([o2.pom_loss(net(G.batch(s_va[i:i + 256], device))[1], Y[s_va[i:i + 256]], class_w).item() for i in range(0, len(s_va), 256)])
                v_g = np.mean([o3.rank_loss(blend_logits(net, S, X, S.pair_a[q_va[i:i + 256]], S.pair_b[q_va[i:i + 256]], theta, torch.exp(log_tau), device),
                                            PY[q_va[i:i + 256]], T).mean().item() for i in range(0, len(q_va), 256)])
            if v_s + v_g < best[0]:
                best = (v_s + v_g, epoch + 1, ({k: v.detach().clone() for k, v in net.state_dict().items()}, theta.detach().clone(), log_tau.detach().clone(),
                                               ma_line.detach().clone(), v_s, v_g, tot[2] / n_batches))
            vl = f"; inner-val (unseen molecules) single {v_s:.3f}, GoodScents {v_g:.3f}"
        if (epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1:
            t_ = tot / n_batches; log(f"  seed {seed} epoch {epoch + 1} loss {t_[0]:.3f} (single {t_[1]:.3f}, GoodScents {t_[2]:.3f}, Ma {t_[3]:.3f}){vl}")
    if early_stop:
        sd, th, lt, ml, v_s, v_g, tr_g = best[2]; net.load_state_dict(sd)
        log(f"  seed {seed}: early stop at epoch {best[1]}: inner-val single {v_s:.3f}, GoodScents {v_g:.3f}; GoodScents train loss that epoch {tr_g:.3f} "
            f"(memorization gap {v_g - tr_g:+.3f})")
        net.eval(); net.train_pairs = pairs
        return net, th, torch.exp(lt), ml
    net.eval(); net.train_pairs = pairs
    return net, theta.detach(), torch.exp(log_tau).detach(), ma_line.detach()


@torch.no_grad()
def scent_heldout(net, S, X, theta, tau, device):
    te = np.where(S.pair_split == 1)[0]; out = []
    for k in range(0, len(te), 256):
        q = te[k:k + 256]; out.append(blend_logits(net, S, X, S.pair_a[q], S.pair_b[q], theta, tau, device))
    lg = torch.cat(out)
    loss = o3.rank_loss(lg, torch.as_tensor(S.pair_Y[te], dtype=torch.float32, device=device), torch.as_tensor(S.targets, device=device)).mean().item()
    auc, _ = word_auroc(lg.cpu().numpy(), S.pair_Y[te], S.targets)
    return loss, auc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--no-ma", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--drop", type=float, default=0.0, help="M1: probability of subgraph removal per molecule (MolCLR / MSN-DDI)")
    ap.add_argument("--wd", type=float, default=1.0, help="M2: weight-decay multiplier")
    ap.add_argument("--es", action="store_true", help="M2: early stopping on molecule-disjoint inner validation")
    ap.add_argument("--tag", default="", help="checkpoint folder suffix")
    ap.add_argument("--cap", type=int, default=0, help="de-duplicate GoodScents: at most this many pairs per molecule (fixed subset)")
    ap.add_argument("--ma-sign", action="store_true", help="Ma by sign only (BCE on theta . dx), Ma fold 0 held out")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("train_o3e" + (f"_{a.tag}" if a.tag else "") + ("_noma" if a.no_ma else ""), {**vars(a), "device": dev, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False); gn = dream_groups()
    X = torch.as_tensor(o3.group_matrix(z["smiles"], gn), dtype=torch.float32, device=dev)
    gslf = np.arange(G.n_gslf); single_ids, single_test = gslf[~G.audit_test], gslf[G.audit_test]
    ck = ROOT / (("checkpoints/o3e_noma" if a.no_ma else "checkpoints/o3e") + (f"_{a.tag}" if a.tag else "")); ck.mkdir(parents=True, exist_ok=True)
    rep.line(f"singles train {len(single_ids)}, held out {len(single_test)}; GoodScents train pairs {int((S.pair_split == 0).sum())}, "
             f"held-out (both molecules unseen) {int((S.pair_split == 1).sum())}; Ma mixtures {len(S.ma_L)} ({'used' if not a.no_ma else 'not used'}); "
             f"groups {len(gn)}; mixture-side parameters: theta {len(gn)} + tau 1 (+ Ma line 2)")
    for s in a.seeds:
        if (ck / f"member_{s}.pt").exists():
            rep.line(f"seed {s}: exists, skipped"); continue
        t0 = time.time()
        net, theta, tau, ma_line = train(G, S, X, single_ids, s, dev, not a.no_ma, log=rep.line, p_drop=a.drop, wd_mult=a.wd, early_stop=a.es,
                                         cap=a.cap, ma_sign=a.ma_sign)
        auc = o2.mean_auroc(o2.predict_probs(net, G, single_test, dev), G.Y[single_test])
        gl, ga = scent_heldout(net, S, X, theta, tau, dev)
        tl = train_pair_loss(net, S, X, net.train_pairs, theta, tau, dev); mauc, n_ma = ma_heldout_auroc(S, X, theta)
        torch.save({"train_pair_loss": tl, "ma_heldout_auroc": mauc, "model_state_dict": net.state_dict(), "arch": "o2c", "seed": s, "theta": theta.cpu().numpy(), "tau": float(tau),
                    "ma_line": ma_line.cpu().numpy(), "groups": gn, "single_auroc": auc, "scent_loss": gl, "scent_auroc": ga}, ck / f"member_{s}.pt")
        rep.line(f"seed {s}: single held-out AUROC {auc:.4f} (O2-C-Single 0.875-0.883); GoodScents held-out (unseen molecules) "
                 f"rank loss {gl:.4f} (frozen O2-C + theta: 2.141), word AUROC {ga:.4f}; training-pair loss {tl:.4f} (memorization gap {gl - tl:+.4f}); "
                 f"Ma held-out AUROC {mauc:.3f} ({n_ma} mixtures, a molecule unseen); max |theta| {theta.abs().max():.3f}; {time.time() - t0:.0f} s")
    if a.report:
        cs = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(ck.glob("member_*.pt"))]
        rep.table(["seed", "single held-out AUROC", "GoodScents held-out loss", "GoodScents held-out AUROC", "memorization gap", "Ma held-out AUROC"],
                  [[c["seed"], f"{c['single_auroc']:.4f}", f"{c['scent_loss']:.4f}", f"{c['scent_auroc']:.4f}",
                    f"{c['scent_loss'] - c['train_pair_loss']:+.4f}" if "train_pair_loss" in c else "", f"{c.get('ma_heldout_auroc', float('nan')):.3f}"] for c in cs])
        th = np.array([c["theta"] for c in cs]); order = np.argsort(-np.abs(th.mean(0)))
        rep.table(["group", "theta mean", "theta sd over members"], [[gn[g], f"{th.mean(0)[g]:+.3f}", f"{th.std(0)[g]:.3f}"] for g in order])
    rep.save()


if __name__ == "__main__":
    main()
