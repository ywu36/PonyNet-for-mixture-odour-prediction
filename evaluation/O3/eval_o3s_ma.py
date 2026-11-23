# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  T = Ma mixtures (198)   H = middle layer (400)   G = RDKit groups (24)
#   K = network members
# ---------------------------------------------------------------------------------------------
"""O3-S step 1: Ma as set vs single, direction only (the user, 2026-11-19: "for Ma don't use the actual value but
only the direction; the final direction after adding in the middle layer should be closer to one than the other").

Middle layer v(m) = O2-C's centred encoder output (400). Mixture z = (w_A v_A + w_B v_B) / (w_A + w_B),
w = exp(theta . x). Score  d = cos(z, v_A) - cos(z, v_B)  (> 0: the mixture points to A). Target: sign(L) only,
L = log((I_A,mix / I_A) / (I_B,mix / I_B)) (Ma 2021). Loss softplus(-sign(L) d / tau). No L values are used in training.
Identity: d = (w_A |v_A| - w_B |v_B|) (1 - cos(v_A, v_B)) / ((w_A + w_B) |z|), so the sign is set by
"loudness x middle-layer length" w |v|: Ma trains theta and the lengths of v (how far each molecule sits from the centre).

Rows (unseen-molecule CV over Ma's 5 molecule folds: a test mixture has a held-out molecule; its training mixtures
have none):
  R0  frozen O2-C-Single (10 members), theta = 0: does the middle-layer length alone predict who dominates? (0 numbers)
  R1  frozen, theta + tau fitted on the training mixtures' signs (25 numbers)
  R2  encoder trained jointly: singles (GS-LF train, word loss) + Ma signs (theta, tau and the encoder), 2 seeds / fold
  Rg  groups only, no network: ridge of sign(L) on x_A - x_B (24 numbers), for scale
Metrics on test mixtures (a mixture tested in two folds: predictions averaged): AUROC of d vs sign(L), accuracy,
95% CI by resampling mixtures; also on |L| above its median (clearer cases). R2 also reports single held-out AUROC.

Env:     mist (GPU)
Saved encoders: checkpoints/o3s_ma_set_folds/fold<f>_seed<s>.pt (one per fold and seed, for the Ma check) and
checkpoints/o3s_ma_set/member_<s>.pt (trained on all Ma mixtures, for DREAM / GoodScents tests; keys as O3-E).

Usage:   python evaluation/O3/eval_o3s_ma.py [--seeds 0 1] [--full-seeds 0 1 2]
Writes:  results/eval_o3s_ma/<date_time>_set_vs_single.{txt,json}
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import features, o2, o2j, o3  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from train_o3e import dream_groups  # noqa: E402

F32 = torch.float32


def middle(net, S, ids, device):
    x, e, s, d, g = S.batch(ids, device)
    return net.latent(net.encode(x, e, s, d, g, int(g.max()) + 1))[0]


def score(vA, vB, xA, xB, theta):
    wA, wB = torch.exp(xA @ theta).unsqueeze(1), torch.exp(xB @ theta).unsqueeze(1)
    z = (wA * vA + wB * vB) / (wA + wB)
    cos = torch.nn.functional.cosine_similarity
    return cos(z, vA, dim=1) - cos(z, vB, dim=1)


def sign_loss(d, s, log_tau):
    return torch.nn.functional.softplus(-s * d / torch.exp(log_tau)).mean()


def fit_theta(vA, vB, xA, xB, s, lam=1e-3, steps=400):
    theta = torch.zeros(xA.shape[1], dtype=F32, device=vA.device, requires_grad=True)
    log_tau = torch.tensor(math.log(0.1), dtype=F32, device=vA.device, requires_grad=True)
    opt = torch.optim.Adam([theta, log_tau], lr=0.05)
    for _ in range(steps):
        loss = sign_loss(score(vA, vB, xA, xB, theta), s, log_tau) + lam * (theta ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return theta.detach()


def train_joint(G, S, X, single_ids, ma_tr, seed, device, recipe=o2.RECIPE, ma_batch=32):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = o2.O2C().float().to(device)
    theta = torch.zeros(X.shape[1], dtype=F32, device=device, requires_grad=True)
    log_tau = torch.tensor(math.log(0.1), dtype=F32, device=device, requires_grad=True)
    Y = torch.as_tensor(G.Y, device=device)
    counts = G.Y[single_ids].sum(0); class_w = torch.as_tensor(np.log1p(counts / counts.max()), dtype=F32, device=device)
    sgn = torch.as_tensor(np.sign(S.ma_L), dtype=F32, device=device)
    opt = torch.optim.Adam(list(net.parameters()) + [theta, log_tau], lr=recipe["lr"])
    bs = recipe["batch"]; nb = math.ceil(len(single_ids) / bs); step = 0
    for _ in range(recipe["epochs"]):
        net.train(); perm = rng.permutation(single_ids)
        perm = np.concatenate([perm, rng.choice(single_ids, nb * bs - len(perm), replace=False)])
        for k in range(nb):
            for gr in opt.param_groups:
                gr["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            _, lg = net(G.batch(b, device)); loss = o2.pom_loss(lg, Y[b], class_w)
            t = rng.choice(ma_tr, min(ma_batch, len(ma_tr)), replace=False)
            mols, inv = np.unique(np.r_[S.ma_a[t], S.ma_b[t]], return_inverse=True)
            with o2j.single_stats(net):
                v = middle(net, S, mols, device)
            ia, ib = inv[:len(t)], inv[len(t):]
            d = score(v[ia], v[ib], X[mols[ia]], X[mols[ib]], theta)
            loss = loss + sign_loss(d, sgn[t], log_tau) + o2.regularization(net, recipe["weight_decay"]) + 1e-3 * (theta ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
    net.eval()
    return net, theta.detach()


@torch.no_grad()
def member_scores(net, S, X, theta, device):
    mols = np.unique(np.r_[S.ma_a, S.ma_b]); v = torch.zeros(int(mols.max()) + 1, 400, dtype=F32, device=device)
    v[mols] = middle(net, S, mols, device)
    return v


def metrics(d_T, L_T, n_boot=2000, seed=0):
    y = (L_T > 0).astype(int); rng = np.random.default_rng(seed)
    auc = roc_auc_score(y, d_T); acc = float(((d_T > 0) == y).mean())
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], d_T[i]))
    big = np.abs(L_T) > np.median(np.abs(L_T))
    return auc, np.percentile(bs, [2.5, 97.5]), acc, roc_auc_score(y[big], d_T[big])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--full-seeds", type=int, nargs="*", default=[0, 1, 2]); a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_o3s_ma", {"seeds": a.seeds, "full_seeds": a.full_seeds, "device": dev, **o2.RECIPE}, tag="set_vs_single")
    ckf, ckm = ROOT / "checkpoints/o3s_ma_set_folds", ROOT / "checkpoints/o3s_ma_set"; ckf.mkdir(parents=True, exist_ok=True); ckm.mkdir(parents=True, exist_ok=True)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False); gn = dream_groups()
    X = torch.as_tensor(o3.group_matrix(z["smiles"], gn), dtype=F32, device=dev)
    gslf = np.arange(G.n_gslf); single_ids, single_te = gslf[~G.audit_test], gslf[G.audit_test]
    L = S.ma_L; sgn = torch.as_tensor(np.sign(L), dtype=F32, device=dev); nT = len(L)
    folds = [S.ma_split(f) for f in range(5)]
    rep.line(f"Ma mixtures {nT} (A dominates {(L > 0).sum()}, B {(L < 0).sum()}), molecules {len(np.unique(np.r_[S.ma_a, S.ma_b]))}; "
             f"test mixtures per fold {[len(te) for _, te in folds]}; training uses sign(L) only")
    pred = {k: np.zeros(nT) for k in ("R0", "R1", "R2", "Rg")}; cnt = np.zeros(nT)
    for _, te in folds:
        cnt[te] += 1
    # frozen members
    nets = o2.load_members(sorted((ROOT / "checkpoints/o2c").glob("member_*.pt")), dev)
    V = [member_scores(n, S, X, None, dev) for n in nets]
    XA, XB = X[S.ma_a], X[S.ma_b]; zero = torch.zeros(X.shape[1], dtype=F32, device=dev)
    for f, (tr, te) in enumerate(folds):
        d0 = np.mean([score(v[S.ma_a[te]], v[S.ma_b[te]], XA[te], XB[te], zero).cpu().numpy() for v in V], 0)
        d1 = []
        for v in V:
            th = fit_theta(v[S.ma_a[tr]], v[S.ma_b[tr]], XA[tr], XB[tr], sgn[tr])
            d1.append(score(v[S.ma_a[te]], v[S.ma_b[te]], XA[te], XB[te], th).detach().cpu().numpy())
        dx = (XA - XB).cpu().numpy(); s_ = np.sign(L)
        w = np.linalg.solve(dx[tr].T @ dx[tr] + 1.0 * np.eye(dx.shape[1]), dx[tr].T @ s_[tr])
        pred["R0"][te] += d0; pred["R1"][te] += np.mean(d1, 0); pred["Rg"][te] += dx[te] @ w
        rep.line(f"fold {f}: frozen rows done")
    # jointly trained encoder
    sauc = []
    for f, (tr, te) in enumerate(folds):
        d2 = []
        for s in a.seeds:
            net, th = train_joint(G, S, X, single_ids, tr, 1000 * f + s, dev)
            with torch.no_grad():
                v = member_scores(net, S, X, th, dev)
                d2.append(score(v[S.ma_a[te]], v[S.ma_b[te]], XA[te], XB[te], th).cpu().numpy())
            sauc.append(o2.mean_auroc(o2.predict_probs(net, G, single_te, dev), G.Y[single_te]))
            torch.save({"model_state_dict": net.state_dict(), "arch": "o2c", "seed": s, "fold": f, "theta": th.cpu().numpy(), "groups": gn,
                        "single_auroc": sauc[-1]}, ckf / f"fold{f}_seed{s}.pt")
            rep.line(f"fold {f} seed {s}: trained; single held-out AUROC {sauc[-1]:.4f}; theta |mean| {th.abs().mean():.3f}")
        pred["R2"][te] += np.mean(d2, 0)
    rep.section("Who dominates the mixture, unseen-molecule folds (sign of L only)")
    names = {"Rg": "groups only, ridge on x_A - x_B (24 numbers)", "R0": "frozen O2-C middle layer, theta = 0 (0 numbers)",
             "R1": "frozen O2-C middle layer, theta fitted (25 numbers)", "R2": "encoder trained on singles + Ma signs (theta 24 + tau)"}
    rows = []
    for k, nm in names.items():
        auc, ci, acc, auc_big = metrics(pred[k] / cnt, L)
        rows.append([nm, f"{auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", f"{acc:.3f}", f"{auc_big:.3f}"])
    rep.table(["model", "AUROC [95% CI]", "accuracy", f"AUROC, |L| > median ({int((np.abs(L) > np.median(np.abs(L))).sum())} mixtures)"], rows)
    rep.line(f"R2 single held-out AUROC mean {np.mean(sauc):.4f} (O2-C-Single 0.875-0.883)")
    rep.section("Members trained on all Ma mixtures (for DREAM / GoodScents tests; no Ma check possible on these)")
    for s in a.full_seeds:
        net, th = train_joint(G, S, X, single_ids, np.arange(nT), 500 + s, dev)
        auc = o2.mean_auroc(o2.predict_probs(net, G, single_te, dev), G.Y[single_te])
        torch.save({"model_state_dict": net.state_dict(), "arch": "o2c", "seed": s, "arm": "o3s_ma_set", "theta": th.cpu().numpy(), "groups": gn,
                    "single_auroc": auc}, ckm / f"member_{s}.pt")
        rep.line(f"full member {s}: single held-out AUROC {auc:.4f}; max |theta| {float(th.abs().max()):.3f}")
    rep.save()


if __name__ == "__main__":
    main()
