"""How strongly does the shared part dominate, depending on what FRACTION of each mixture it is?

One shared molecule in 2-molecule mixtures is half of each; in 10-molecule mixtures it is a tenth.
Per pair (A, B), k = molecules in both:
    f       = (k/|A| + k/|B|) / 2            count share of the shared part
    f_w     = same with B1a++ weights         (w_i = exp(theta . groups_i), theta fitted on all 820 pairs;
                                              description only: in-sample weights)
    err     = target - B1a++ prediction       (standard CV, mean over 50 repeats per pair)
Logic: B1a++ already mixes the shared part in by its weight share. If the shared part pulls the two
mixtures together MORE than its share, err < 0 (rated closer than predicted); if less (it is tuned out
as a common background), err > 0. Whether this depends on f is the question.
Fixed before running (description, no claim threshold; leads only):
  1. by f bin: pairs, mean size, mean target, mean B1a++ prediction, mean err [95% CI resampling
     mixtures], RMSE / r [CI].
  2. grid: mixture size (mean of |A|, |B|) x k: pairs, mean f, mean target, mean err.
  3. Spearman rho(f, err) over setting-2 pairs [CI]; and rho(f_w - f, err): does the extra weight
     B1a++ gives the shared part go the right way?
Env:     mist
Usage:   python evaluation/B1aplus/eval_setting2_shared_fraction.py
Writes:  results/eval_setting2_shared_fraction/<date_time>.{txt,json} (+ b1a_pp_cv_pred.npz)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3
N_BOOT = 2000
F_BINS = [(0, 0, "0 (setting 1)"), (1e-9, 0.10, "(0, 0.10]"), (0.10, 0.20, "(0.10, 0.20]"), (0.20, 0.35, "(0.20, 0.35]"),
          (0.35, 0.50, "(0.35, 0.50]"), (0.50, 0.75, "(0.50, 0.75]"), (0.75, 1.0, "(0.75, 1)")]
N_BINS = [(1, 4.5, "1-4"), (4.5, 9.5, "5-9"), (9.5, 19.5, "10-19"), (19.5, 99, "20+")]
K_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 99, "6+")]


def main():
    rep = Report("eval_setting2_shared_fraction", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    res = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)
    pred = np.bincount(res.pair_N, res.pred_N, P) / np.bincount(res.pair_N, minlength=P)
    err = y - pred
    out_dir = ROOT / "results/eval_setting2_shared_fraction"; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "b1a_pp_cv_pred.npz", pred_P=pred, pair_N=res.pair_N, pred_N=res.pred_N)

    th = B1aPlus(q, Xg, A, B, S, LAM).fit(np.arange(P), y).theta.detach().numpy()
    w = np.exp(Xg @ th)
    Sh = A * B
    nA, nB = A.sum(1), B.sum(1); k = Sh.sum(1); nbar = (nA + nB) / 2
    f = (k / nA + k / nB) / 2
    f_w = ((Sh * w).sum(1) / (A * w).sum(1) + (Sh * w).sum(1) / (B * w).sum(1)) / 2

    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)

    def boot_mean(v, m):
        wb = (cnt[:, ja[m]] * cnt[:, jb[m]]).astype(float)
        s = wb.sum(1); ok = s > 0
        b = (wb[ok] * v[m]).sum(1) / s[ok]
        return v[m].mean(), np.percentile(b, [2.5, 97.5])

    rep.section("1. By shared fraction f (count share of the shared part, mean over the two mixtures)")
    rows = []
    for lo, hi, lab in F_BINS:
        m = (f == 0) if hi == 0 else (f > lo) & (f <= hi)
        if m.sum() == 0:
            continue
        me, ci = boot_mean(err, m)
        mN = m[res.pair_N]
        s = evaluate.score_cv(train, res, mask_N=mN, n_boot=N_BOOT)
        rows.append([lab, int(m.sum()), f"{nbar[m].mean():.1f}", f"{k[m].mean():.1f}", f"{f_w[m].mean():.2f}", f"{y[m].mean():.3f}", f"{pred[m].mean():.3f}",
                     f"{me:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", f"{s['rmse']:.3f} / {s['r']:.2f} [{s['r_ci'][0]:+.2f}, {s['r_ci'][1]:+.2f}]"])
    rep.table(["f bin", "pairs", "mean size", "mean k", "mean f_w", "mean target", "mean B1a++ pred", "mean err = target - pred [CI]", "B1a++ RMSE / r [CI]"], rows)
    rep.line("err < 0: rated closer than B1a++ predicts (shared part pulls more than its weight share); > 0: less.")

    rep.section("2. Grid: mixture size (mean of the two) x molecules shared k   [pairs | mean f | mean target | mean err]")
    rows = []
    for nlo, nhi, nlab in N_BINS:
        r_ = [nlab]
        for klo, khi, _ in K_BINS:
            m = (nbar >= nlo) & (nbar < nhi) & (k >= klo) & (k <= khi)
            r_.append("-" if m.sum() == 0 else f"{int(m.sum())} | {f[m].mean():.2f} | {y[m].mean():.3f} | {err[m].mean():+.3f}")
        m0 = (nbar >= nlo) & (nbar < nhi) & (k == 0)
        r_.append("-" if m0.sum() == 0 else f"{int(m0.sum())} | {y[m0].mean():.3f} | {err[m0].mean():+.3f}")
        rows.append(r_)
    rep.table(["size"] + [f"k = {lab}" for _, _, lab in K_BINS] + ["k = 0: pairs | target | err"], rows)

    rep.section("3. Does the error depend on the shared fraction? (setting-2 pairs)")
    s2 = k > 0
    idx2 = np.where(s2)[0]
    def rho_ci(a, b):
        r0 = spearmanr(a[idx2], b[idx2]).statistic
        bs = []
        for c in cnt[:500]:
            wpair = c[ja[idx2]] * c[jb[idx2]]
            rr = np.repeat(idx2, wpair)
            if len(rr) > 10:
                bs.append(spearmanr(a[rr], b[rr]).statistic)
        return r0, np.percentile(bs, [2.5, 97.5])
    for lab, a, b in [("rho(f, target)", f, y), ("rho(f, B1a++ pred)", f, pred), ("rho(f, err)", f, err), ("rho(f_w - f, err)", f_w - f, err)]:
        r0, ci = rho_ci(a, b)
        rep.line(f"{lab:22s} {r0:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]   ({s2.sum()} pairs; CI resampling mixtures, 500 draws)")
    rep.line(f"B1a++ weights: shared part's weight share f_w vs count share f over setting 2: mean f_w - f = {(f_w - f)[s2].mean():+.3f}")
    rep.value("bins", rows)
    rep.save()


if __name__ == "__main__":
    main()
