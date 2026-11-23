# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = GoodScents + Ma molecules   W = words (138)   G = groups (24)
#   Q = GoodScents pairs   T = Ma mixtures   M = DREAM molecules (235)   P = development pairs (514)
# ---------------------------------------------------------------------------------------------
"""Verification requested by the user via the "set aggregation" session (2026-11-19): GoodScents used ONLY through the
group loudness theta (24) + temperature tau, encoder frozen (O2-C-Single, 10 members, profiles = mean of members'
decoded probabilities). Blend = w p_A + (1 - w) p_B, w = sigmoid(theta . (x_A - x_B)) (decoded alone, then averaged).

1. theta + tau fitted on GoodScents train pairs (L-BFGS, lam 1e-3) vs baseline theta = 0 (tau fitted): rank loss and word
   AUROC on the official held-out pairs (both molecules unseen); paired change with 95% CI by resampling held-out
   molecules (a pair counts c_A x c_B times). Also Ma fold-0 AUROC of theta . (x_A - x_B) (no Ma in this fit).
2. Same + Ma sign in the fit: binary cross-entropy of theta . (x_A - x_B) vs [L > 0] on Ma mixtures without a fold-0
   molecule; GoodScents held-out as in 1, Ma fold-0 AUROC.
3. DREAM transfer, 3 numbers: profiles = frozen O2-C-Single quantile profiles (O3.0's), theta frozen from 1 or 2, DREAM
   fits beta + line only; vs O1.2b, paired, development pairs 50 x 5, 4 subsets. O3.0 (theta fitted on DREAM, 27) shown.

Env:     mist (CPU is enough)
Usage:   python evaluation/O3/verify_theta_scent.py
Writes:  results/verify_theta_scent/<date_time>_theta_only.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from eval_o3e import O3Fixed  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o2j import word_auroc  # noqa: E402

F64 = torch.float64


def fit(P, X, a, b, Y, T, ma=None, use_theta=True, lam=1e-3, max_iter=500):
    """L-BFGS on mean GoodScents rank loss (+ mean Ma sign cross-entropy) + lam |theta, log tau|^2."""
    P, X, Y = (torch.as_tensor(v, dtype=F64) for v in (P, X, Y)); T = torch.as_tensor(T)
    Pa, Pb, dX = P[a], P[b], X[a] - X[b]
    th = torch.zeros(X.shape[1], dtype=F64, requires_grad=use_theta); lt = torch.zeros((), dtype=F64, requires_grad=True)
    params = [th, lt] if use_theta else [lt]
    opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w = torch.sigmoid(dX @ th).unsqueeze(1); lg = torch.logit((w * Pa + (1 - w) * Pb).clamp(1e-6, 1 - 1e-6)) / torch.exp(lt)
        loss = o3.rank_loss(lg, Y, T).mean() + lam * ((th ** 2).sum() + lt ** 2)
        if ma is not None:
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(torch.as_tensor(ma[0], dtype=F64) @ th, torch.as_tensor(ma[1], dtype=F64))
        loss.backward()
        return loss
    opt.step(closure)
    return th.detach(), torch.exp(lt).detach()


@torch.no_grad()
def heldout(P, X, a, b, Y, T, th, tau):
    P, X, Y = (torch.as_tensor(v, dtype=F64) for v in (P, X, Y))
    w = torch.sigmoid((X[a] - X[b]) @ th).unsqueeze(1); lg = torch.logit((w * P[a] + (1 - w) * P[b]).clamp(1e-6, 1 - 1e-6)) / tau
    return o3.rank_loss(lg, Y, torch.as_tensor(T)).numpy(), lg.numpy()


def mol_boot(diff_Q, a, b, n=2000, seed=0):
    mols = np.unique(np.r_[a, b]); pos = {m: i for i, m in enumerate(mols)}; ia = np.array([pos[m] for m in a]); ib = np.array([pos[m] for m in b])
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        c = np.bincount(rng.integers(0, len(mols), len(mols)), minlength=len(mols)).astype(float); w = c[ia] * c[ib]
        if w.sum() > 0:
            out.append((w * diff_Q).sum() / w.sum())
    return np.percentile(out, [2.5, 97.5])


def main():
    dev = pick_device(); rep = Report("verify_theta_scent", {"lam": 1e-3, "repeats": evaluate.CV_REPEATS, "device": dev}, tag="theta_only")
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M)
    nets, _, _, cprobs = members(G, "checkpoints/o2c", dev)
    good = np.where(z["ok"].astype(bool))[0]; PN = np.full((len(z["smiles"]), 138), 0.5)
    PN[good] = np.mean([o2.predict_probs(n, S, good, dev) for n in nets], 0); XN = o3.group_matrix(z["smiles"], gn)
    tr_q, te_q = np.where(S.pair_split == 0)[0], np.where(S.pair_split == 1)[0]
    ma_tr, ma_te = S.ma_split(0); dXma = XN[S.ma_a] - XN[S.ma_b]; yma = (S.ma_L > 0).astype(float)
    rep.line(f"frozen O2-C-Single, {len(nets)} members (profiles averaged); GoodScents train pairs {len(tr_q)}, held-out {len(te_q)} "
             f"(both molecules unseen, {len(np.unique(np.r_[S.pair_a[te_q], S.pair_b[te_q]]))} molecules); Ma train {len(ma_tr)}, held-out fold 0 {len(ma_te)}")
    args = (PN, XN, S.pair_a[tr_q], S.pair_b[tr_q], S.pair_Y[tr_q], S.targets)
    fits = {"theta = 0 (tau only)": fit(*args, use_theta=False), "1. theta + tau (GoodScents)": fit(*args),
            "2. theta + tau (GoodScents + Ma sign)": fit(*args, ma=(dXma[ma_tr], yma[ma_tr]))}
    hold = {k: heldout(PN, XN, S.pair_a[te_q], S.pair_b[te_q], S.pair_Y[te_q], S.targets, *v) for k, v in fits.items()}
    base = hold["theta = 0 (tau only)"][0]; rows = []
    for k, (l_, lg) in hold.items():
        th = fits[k][0].numpy(); d = dXma[ma_te] @ th
        ma_auc = roc_auc_score(yma[ma_te], d) if np.std(d) > 0 else float("nan")
        diff = l_ - base; ci = mol_boot(diff, S.pair_a[te_q], S.pair_b[te_q]) if k != "theta = 0 (tau only)" else None
        rows.append([k, f"{l_.mean():.4f}", "" if ci is None else f"{diff.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]",
                     f"{word_auroc(lg, S.pair_Y[te_q], S.targets)[0]:.4f}", f"{ma_auc:.3f}", f"{float(fits[k][1]):.3f}", f"{np.abs(th).max():.3f}"])
    rep.section("GoodScents held-out pairs (both molecules unseen) and Ma held-out (fold 0), frozen encoder")
    rep.table(["fit", "held-out rank loss", "change vs theta = 0 [95% CI, molecules resampled]", "held-out word AUROC",
               f"Ma fold-0 AUROC ({len(ma_te)} mixtures)", "tau", "max |theta|"], rows)
    th1, th2 = fits["1. theta + tau (GoodScents)"][0].numpy(), fits["2. theta + tau (GoodScents + Ma sign)"][0].numpy()
    order = np.argsort(-np.abs(th1))
    rep.table(["group", "theta (1)", "theta (2)"], [[gn[g], f"{th1[g]:+.3f}", f"{th2[g]:+.3f}"] for g in order])

    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    q_c = mixture.quantile_per_word(cprobs)
    res = {"O1.2b (reference)": evaluate.cross_validate(P, lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), Xd, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)),
           "O3.0 (frozen O2-C-Single, theta fitted on DREAM, 27)": evaluate.cross_validate(P, lambda tr, te: o3.O3(q_c, Xd, A, B, True, LAM).fit(tr, y[tr]).predict(te)),
           "theta = 0, beta + line (3)": evaluate.cross_validate(P, lambda tr, te: O3Fixed(q_c, Xd, A, B, np.zeros(len(gn))).fit(tr, y[tr]).predict(te)),
           "3a. theta from 1 (GoodScents), beta + line (3)": evaluate.cross_validate(P, lambda tr, te: O3Fixed(q_c, Xd, A, B, th1).fit(tr, y[tr]).predict(te)),
           "3b. theta from 2 (GoodScents + Ma sign), beta + line (3)": evaluate.cross_validate(P, lambda tr, te: O3Fixed(q_c, Xd, A, B, th2).fit(tr, y[tr]).predict(te))}
    ref, zero = res["O1.2b (reference)"], res["theta = 0, beta + line (3)"]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"3. DREAM development pairs ({evaluate.CV_REPEATS} x 5): change from O1.2b (paired); and minus theta = 0 (same 3-number form)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if r_ is ref else ch(s_), ch(sc(r_, sub, zero)) if k.startswith("3") else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "minus theta = 0 (3 numbers)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
