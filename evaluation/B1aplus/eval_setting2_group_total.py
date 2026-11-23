# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = pairs sharing 1-2 molecules (179)   M = molecules (235)
#   G = RDKit groups (24)   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""Pairs sharing 1-2 molecules: group-level pull and push, groups counted over ALL molecules ("total").

s_A(g) = share of mixture A's molecules carrying group g (shared or not). Per pair and group:
    pull_g = min(s_A(g), s_B(g))      how much of g both mixtures have
    push_g = |s_A(g) - s_B(g)|        how unevenly g is spread
    z      = mean over A, B of (shared molecules / mixture size)   control: how much is shared
Errors from standard CV on all 820 pairs (mean held-out prediction per pair):
    e0 = y - prediction of B1a + size (equal weights)      e1 = y - prediction of B1a++ (B1a + size + groups)
Per group: e ~ 1 + z + pull_g + push_g, both slopes per sd, 95% CI and p by resampling mixtures (2000),
BH q per feature over groups with support >= 20 (pairs with pull_g > 0 for pull; push_g > 0 for push).
Confirmation predictions fixed before running, with theta from SETTING 1 ONLY (never saw these pairs):
  dominant groups pull more when common:       Spearman rho(theta_g, -pull slope on e0) > 0
  dominant groups push more when one-sided:    Spearman rho(theta_g, +push slope on e0) > 0
Audit subset (pairs sharing 1-2 molecules was picked after seeing setting-2 results).

Env:     mist
Usage:   python evaluation/B1aplus/eval_setting2_group_total.py [low|all]   (all = all 495 setting-2 pairs; added 2026-11-18 to fill the 2 x 2)
Writes:  results/eval_setting2_group_total/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
sys.path.insert(0, str(ROOT / "findings"))
from pair_signal_by_sharing import bh  # noqa: E402
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_group_pull import mean_pred  # noqa: E402

LAM = 1e-3
N_BOOT = 2000
MIN_SUPPORT = 20


def coefs(X, e, W):
    out = []
    for w in W:
        Xw = X * w[:, None]
        try:
            out.append(np.linalg.solve(X.T @ Xw, Xw.T @ e)[2:4])
        except np.linalg.LinAlgError:
            out.append([np.nan, np.nan])
    return np.array(out)


def pval(bb):
    bb = bb[np.isfinite(bb)]
    return max(min(1, 2 * min(np.mean(bb <= 0), np.mean(bb >= 0))), 1 / N_BOOT)


def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else "low"
    rep = Report("eval_setting2_group_total", {"subset": subset, "lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "min_support": MIN_SUPPORT})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    row = {int(c): i for i, c in enumerate(op.cids_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    X_MxG, names = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    short = [n.split(" (= ")[0] for n in names]

    r0 = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, np.zeros((len(ids), 0)), A, B, S, 0.0).fit(tr, y[tr]).predict(te))
    r1 = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, X_MxG, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    p0, p1 = mean_pred(r0, len(train)), mean_pred(r1, len(train))
    s1 = np.array([i for i, p in enumerate(train) if p.jaccard == 0])
    theta1 = B1aPlus(q, X_MxG, A, B, S, LAM).fit(s1, y[s1]).theta.numpy()
    print("  done CV", flush=True)

    idx = np.array([i for i, p in enumerate(train) if (1 <= len(p.a & p.b) <= 2 if subset == "low" else p.jaccard > 0)])
    P = [train[i] for i in idx]
    sA = np.stack([X_MxG[[row[c] for c in p.a]].mean(0) for p in P]); sB = np.stack([X_MxG[[row[c] for c in p.b]].mean(0) for p in P])
    pull, push = np.minimum(sA, sB), np.abs(sA - sB)
    z = np.array([0.5 * (len(p.a & p.b) / len(p.a) + len(p.a & p.b) / len(p.b)) for p in P])
    e0, e1 = y[idx] - p0[idx], y[idx] - p1[idx]
    mixes = sorted({x for p in P for x in (p.a, p.b)}, key=sorted); im = {x: i for i, x in enumerate(mixes)}
    ia = np.array([im[p.a] for p in P]); ib = np.array([im[p.b] for p in P])
    c = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    W = np.vstack([np.ones(len(P)), (c[:, ia] * c[:, ib]).astype(float)])
    keep = np.where(((pull > 0).sum(0) >= MIN_SUPPORT) & ((push > 0).sum(0) >= MIN_SUPPORT))[0]
    rep.section(f"0. {len(P)} pairs (subset {subset}); groups with pull and push support >= {MIN_SUPPORT}: {len(keep)} of {len(names)}")
    rep.line(f"B1a + size on these pairs: error sd {e0.std():.3f}; B1a++ error sd {e1.std():.3f}; rating sd {y[idx].std():.3f}")

    out = {}
    for tag, e in [("B1a + size", e0), ("B1a++", e1)]:
        C = np.array([coefs(np.c_[np.ones(len(P)), z, pull[:, g] / pull[:, g].std(), push[:, g] / push[:, g].std()], e, W) for g in keep])  # G x (1+B) x 2
        est, boot = C[:, 0, :], C[:, 1:, :]
        lo, hi = np.nanpercentile(boot, 2.5, axis=1), np.nanpercentile(boot, 97.5, axis=1)
        pv = np.array([[pval(boot[g, :, k]) for k in range(2)] for g in range(len(keep))])
        qv = np.c_[bh(list(pv[:, 0])), bh(list(pv[:, 1]))]
        rp = spearmanr(theta1[keep], -est[:, 0]).statistic; rs = spearmanr(theta1[keep], est[:, 1]).statistic
        rpb = np.array([spearmanr(theta1[keep], -boot[:, b, 0]).statistic for b in range(N_BOOT)])
        rsb = np.array([spearmanr(theta1[keep], boot[:, b, 1]).statistic for b in range(N_BOOT)])
        out[tag] = dict(est=est, lo=lo, hi=hi, q=qv, rho_pull=(rp, *np.nanpercentile(rpb, [2.5, 97.5])), rho_push=(rs, *np.nanpercentile(rsb, [2.5, 97.5])))
    t0 = out["B1a + size"]
    o = np.argsort(-theta1[keep])
    rep.section("1. Per group, on the B1a + size error (raw group effect; slope per sd; < 0 = rated closer)")
    rep.table(["group", "pairs both / uneven", "setting-1 theta", "pull: common g [95% CI]", "q", "push: uneven g [95% CI]", "q"],
              [[short[keep[j]], f"{int((pull[:, keep[j]] > 0).sum())} / {int((push[:, keep[j]] > 0).sum())}", f"{theta1[keep[j]]:+.2f}",
                f"{t0['est'][j, 0]:+.4f} [{t0['lo'][j, 0]:+.4f}, {t0['hi'][j, 0]:+.4f}]", f"{t0['q'][j, 0]:.2f}",
                f"{t0['est'][j, 1]:+.4f} [{t0['lo'][j, 1]:+.4f}, {t0['hi'][j, 1]:+.4f}]", f"{t0['q'][j, 1]:.2f}"] for j in o])
    rep.line("sorted by setting-1 theta. Pull slope < 0: more common g -> closer than predicted; push slope > 0: more uneven g -> farther.")
    rep.section("2. Confirmation: does setting-1 dominance predict the group effects here?")
    rep.table(["error of", "rho(theta, pull) [95% CI]", "rho(theta, push) [95% CI]", "groups", "q < 0.1 (pull / push)"],
              [[k, f"{v['rho_pull'][0]:+.2f} [{v['rho_pull'][1]:+.2f}, {v['rho_pull'][2]:+.2f}]", f"{v['rho_push'][0]:+.2f} [{v['rho_push'][1]:+.2f}, {v['rho_push'][2]:+.2f}]",
                len(keep), f"{int((v['q'][:, 0] < 0.1).sum())} / {int((v['q'][:, 1] < 0.1).sum())}"] for k, v in out.items()])
    rep.line("Prediction fixed before running: both rho > 0 on the B1a + size error. On the B1a++ error: what B1a++ still misses.")
    rep.value("result", {k: {"groups": [short[g] for g in keep], "pull": v["est"][:, 0].tolist(), "push": v["est"][:, 1].tolist(),
                             "rho_pull": [float(x) for x in v["rho_pull"]], "rho_push": [float(x) for x in v["rho_push"]]} for k, v in out.items()})
    rep.save()


if __name__ == "__main__":
    main()
