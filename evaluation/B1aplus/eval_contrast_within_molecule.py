"""Is the shared-contrast lead a fixed property of the molecule (plain dominance) or does it depend on the company?

Instances: (pair p, shared molecule i). contrast_pi = mean over the two sides of || q_i - mean q of that side's
UNSHARED molecules || (as eval_b1a_plus_shared_contrast.py), z-scored. err_p = target - B1a++ CV prediction (mean over
the 50 repeats; results/eval_setting2_shared_fraction/b1a_pp_cv_pred.npz).
WITHIN-molecule test: subtract each molecule's own mean contrast and mean error (molecules shared in >= 3 pairs with
contrast), then correlate. A fixed per-molecule dominance (however strong) is removed by the demeaning, so it predicts
r_within = 0. The company effect predicts r_within < 0 (same molecule, more standing out -> rated closer than predicted).
Also BETWEEN molecules (molecule means), for comparison. Subsets: all setting-2 pairs; pairs sharing 1-2 molecules
(one or two shared molecules, so the pair's error is attributable). 95% CI: resample mixtures (instance weight
count(A) x count(B)), 2000 draws; plus a permutation null shuffling contrast within molecule (2000).
Prediction fixed before running: r_within < 0. Description, no model fitted.

Env:     mist
Usage:   python evaluation/B1aplus/eval_contrast_within_molecule.py
Writes:  results/eval_contrast_within_molecule/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, evaluate, features, mixture  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.report import Report  # noqa: E402

N_BOOT = 2000; MIN_PAIRS = 3


def wcorr(a, b, w):
    w = w / w.sum(); a = a - (w * a).sum(); b = b - (w * b).sum()
    return (w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum())


def main():
    rep = Report("eval_contrast_within_molecule", {"n_boot": N_BOOT, "min_pairs": MIN_PAIRS})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); Sh = A * B
    pred = np.load(ROOT / "results/eval_setting2_shared_fraction/b1a_pp_cv_pred.npz")["pred_P"]; err = y - pred
    inst = []
    for p in range(P):
        for i in np.where(Sh[p] > 0)[0]:
            cs = []
            for Mx in (A, B):
                un = np.where((Mx[p] > 0) & (Sh[p] == 0))[0]
                if len(un):
                    cs.append(np.linalg.norm(q[i] - q[un].mean(0)))
            if cs:
                inst.append((p, i, np.mean(cs)))
    pI = np.array([t[0] for t in inst]); mI = np.array([t[1] for t in inst]); c = np.array([t[2] for t in inst])
    c = (c - c.mean()) / c.std(); e = err[pI]; k = Sh.sum(1)[pI]
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: j for j, m in enumerate(mixes)}
    ja = np.array([im[train[p].a] for p in pI]); jb = np.array([im[train[p].b] for p in pI])
    rng = np.random.default_rng(0)
    cnt = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)

    rows = []
    for name, sub in [("all setting-2 pairs", np.ones(len(c), bool)), ("pairs sharing 1-2 molecules", k <= 2)]:
        mols, nper = np.unique(mI[sub], return_counts=True)
        keep = sub & np.isin(mI, mols[nper >= MIN_PAIRS])
        idx = np.where(keep)[0]
        def within(w):
            cc, ee = c[idx].copy(), e[idx].copy()
            for m in np.unique(mI[idx]):
                s = mI[idx] == m; ws = w[s]
                if ws.sum() == 0:
                    continue
                cc[s] -= (ws * cc[s]).sum() / ws.sum(); ee[s] -= (ws * ee[s]).sum() / ws.sum()
            return wcorr(cc, ee, w + 1e-12)
        def between(w):
            ms = np.unique(mI[idx]); mc, me, mw = [], [], []
            for m in ms:
                s = mI[idx] == m; ws = w[s]
                if ws.sum() > 0:
                    mc.append((ws * c[idx][s]).sum() / ws.sum()); me.append((ws * e[idx][s]).sum() / ws.sum()); mw.append(ws.sum())
            return wcorr(np.array(mc), np.array(me), np.array(mw))
        w1 = np.ones(len(idx))
        r_w, r_b = within(w1), between(w1)
        bw = np.array([within((cnt[b, ja[idx]] * cnt[b, jb[idx]]).astype(float)) for b in range(N_BOOT)])
        bb = np.array([between((cnt[b, ja[idx]] * cnt[b, jb[idx]]).astype(float)) for b in range(N_BOOT)])
        # permutation: shuffle contrast within molecule
        perm = []
        for _ in range(N_BOOT):
            cp = c.copy()
            for m in np.unique(mI[idx]):
                s = idx[mI[idx] == m]; cp[s] = cp[rng.permutation(s)]
            cc, ee = cp[idx].copy(), e[idx].copy()
            for m in np.unique(mI[idx]):
                s = mI[idx] == m; cc[s] -= cc[s].mean(); ee[s] -= ee[s].mean()
            perm.append(wcorr(cc, ee, w1))
        perm = np.array(perm); p_perm = float((perm <= r_w).mean())
        sd_within = float(np.sqrt(np.mean([np.var(c[idx][mI[idx] == m]) for m in np.unique(mI[idx])])))
        rows.append([name, len(idx), len(np.unique(mI[idx])), f"{sd_within:.2f}",
                     f"{r_w:+.3f} [{np.nanpercentile(bw, 2.5):+.3f}, {np.nanpercentile(bw, 97.5):+.3f}]", f"{p_perm:.3f}",
                     f"{r_b:+.3f} [{np.nanpercentile(bb, 2.5):+.3f}, {np.nanpercentile(bb, 97.5):+.3f}]"])
    rep.section("Contrast vs B1a++ error: within the same molecule vs between molecules")
    rep.table(["subset", "instances", "molecules", "within-molecule sd of contrast (z)", "r within molecule [CI]", "perm. p (one-sided, < 0)", "r between molecules [CI]"], rows)
    rep.line("r < 0: more standing out -> rated closer than B1a++ predicts. Fixed dominance predicts r_within = 0.")
    rep.value("rows", rows)
    rep.save()


if __name__ == "__main__":
    main()
