# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = setting-2 pairs (495)   M = molecules (235)
#   G = RDKit groups (24)   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""Setting 2: group-level pull/push of SHARED molecules, and a confirmation test of setting-1 dominance.

For a setting-2 pair (A, B) and group g:
    z_g = mean over A, B of (number of shared molecules carrying g) / (mixture size)
        = share of each mixture made of shared molecules that carry g
    z   = mean over A, B of (number of shared molecules) / (mixture size)   (control: how much is shared)
Errors from standard CV on all 820 pairs (mean held-out prediction per pair):
    e0 = y - prediction of B1a + size   (equal weights: no group dominance)
    e1 = y - prediction of B1a++        (B1a + size + groups, lam 1e-3)
Per group: slope of e on z_g with z as control (per sd of z_g), 95% CI and p by resampling mixtures (2000),
BH q over groups with support (pairs with z_g > 0) >= 20. Slope < 0: pairs whose shared molecules carry g
are rated closer than predicted (g pulls when shared); > 0 push.
Confirmation test, fixed before running: if group g dominates (theta_g > 0), a shared molecule carrying g
dominates both mixtures, so the pair is rated closer than the equal-weight model predicts. Prediction:
Spearman rho(theta_g from SETTING 1 ONLY, -slope_g on e0) > 0. CI from the same mixture resamples
(theta held fixed). With e1, any remaining pull is what B1a++ misses.

Env:     mist
Usage:   python evaluation/B1aplus/eval_setting2_group_pull.py [all|low]   (low = pairs sharing 1-2 molecules; added 2026-11-18 to fill the 2 x 2 with eval_setting2_group_total.py)
Writes:  results/eval_setting2_group_pull/<date_time>.{txt,json}
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

LAM = 1e-3
N_BOOT = 2000
MIN_SUPPORT = 20


def mean_pred(res, n):
    w = evaluate.cv_weights(res)
    return np.bincount(res.pair_N, weights=w * res.pred_N, minlength=n) / np.bincount(res.pair_N, weights=w, minlength=n)


def slopes(Zg, z, e, W):
    """Weighted LS of e on [1, z, Zg] per weight row -> coefficient of Zg, shape (rows,)."""
    X = np.c_[np.ones(len(e)), z, Zg]
    out = []
    for w in W:
        Xw = X * w[:, None]
        try:
            out.append(np.linalg.solve(X.T @ Xw, Xw.T @ e)[2])
        except np.linalg.LinAlgError:
            out.append(np.nan)
    return np.array(out)


def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else "all"
    rep = Report("eval_setting2_group_pull", {"subset": subset, "lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "min_support": MIN_SUPPORT})
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

    idx = np.array([i for i, p in enumerate(train) if (p.jaccard > 0 if subset == "all" else 1 <= len(p.a & p.b) <= 2)])
    P = [train[i] for i in idx]
    Zg = np.zeros((len(P), len(names))); z = np.zeros(len(P))
    for k, p in enumerate(P):
        sh = [row[c] for c in p.a & p.b]
        cnt = X_MxG[sh].sum(0)
        Zg[k] = 0.5 * (cnt / len(p.a) + cnt / len(p.b)); z[k] = 0.5 * (len(sh) / len(p.a) + len(sh) / len(p.b))
    e0, e1 = y[idx] - p0[idx], y[idx] - p1[idx]
    mixes = sorted({x for p in P for x in (p.a, p.b)}, key=sorted); im = {x: i for i, x in enumerate(mixes)}
    ia = np.array([im[p.a] for p in P]); ib = np.array([im[p.b] for p in P])
    c = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    W = np.vstack([np.ones(len(P)), (c[:, ia] * c[:, ib]).astype(float)])

    sup = (Zg > 0).sum(0)
    keep = np.where(sup >= MIN_SUPPORT)[0]
    rep.section(f"0. {len(P)} setting-2 pairs (subset {subset}); groups carried by a shared molecule in >= {MIN_SUPPORT} pairs: {len(keep)} of {len(names)}")
    rep.line(f"shared share z: median {np.median(z):.2f}; B1a + size error sd {e0.std():.3f}, B1a++ error sd {e1.std():.3f}")

    out = {}
    for tag, e in [("B1a + size (raw group pull)", e0), ("B1a++ (pull B1a++ misses)", e1)]:
        b_GxB = np.array([slopes(Zg[:, g] / Zg[:, g].std(), z, e, W) for g in keep])     # per sd of z_g
        est, boot = b_GxB[:, 0], b_GxB[:, 1:]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=1)
        pv = np.array([max(min(1, 2 * min(np.mean(bb[np.isfinite(bb)] <= 0), np.mean(bb[np.isfinite(bb)] >= 0))), 1 / N_BOOT) for bb in boot])
        qv = bh(list(pv))
        rho = spearmanr(theta1[keep], -est).statistic
        rho_b = np.array([spearmanr(theta1[keep], -boot[:, b]).statistic for b in range(N_BOOT)])
        rlo, rhi = np.nanpercentile(rho_b, [2.5, 97.5])
        out[tag] = dict(est=est, lo=lo, hi=hi, p=pv, q=qv, rho=rho, rho_ci=(rlo, rhi))
    rep.section("1. Pull (-) / push (+) per sd of z_g when shared molecules carry group g (control: how much is shared)")
    o = np.argsort(-theta1[keep])
    t0, t1 = out["B1a + size (raw group pull)"], out["B1a++ (pull B1a++ misses)"]
    rep.table(["group", "pairs (support)", "setting-1 theta", "raw pull: slope on B1a+size error [95% CI]", "q", "left for B1a++ [95% CI]", "q"],
              [[short[keep[j]], int(sup[keep[j]]), f"{theta1[keep[j]]:+.2f}", f"{t0['est'][j]:+.4f} [{t0['lo'][j]:+.4f}, {t0['hi'][j]:+.4f}]", f"{t0['q'][j]:.2f}",
                f"{t1['est'][j]:+.4f} [{t1['lo'][j]:+.4f}, {t1['hi'][j]:+.4f}]", f"{t1['q'][j]:.2f}"] for j in o])
    rep.line("sorted by setting-1 theta (most dominant first). Slope < 0: rated closer than predicted (pull).")
    rep.section("2. Confirmation test: does setting-1 dominance predict which groups pull when shared?")
    rep.table(["error of", "Spearman rho(theta setting 1, pull) [95% CI]", "groups"],
              [[k, f"{v['rho']:+.2f} [{v['rho_ci'][0]:+.2f}, {v['rho_ci'][1]:+.2f}]", len(keep)] for k, v in out.items()])
    rep.line("Prediction fixed before running: rho > 0 on the B1a + size error (dominant groups pull when shared).")
    rep.value("result", {k: {"groups": [short[g] for g in keep], "slope": v["est"].tolist(), "lo": v["lo"].tolist(), "hi": v["hi"].tolist(),
                             "q": [float(x) for x in v["q"]], "rho": float(v["rho"]), "rho_ci": [float(x) for x in v["rho_ci"]]} for k, v in out.items()})
    rep.value("theta_setting1", {"groups": short, "theta": theta1.tolist()})
    rep.save()


if __name__ == "__main__":
    main()
