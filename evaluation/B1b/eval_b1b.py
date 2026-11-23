"""B1b vs B1a++: standard CV, same folds (spec: mixdist/b1b.py; settled with session pred-mixture-6c 2026-11-18).

Fixed before running:
  1. Change from B1a++ on the same folds (B1a++ CV predictions recomputed here). Primary subset setting 2 (share >= 1):
     B1b becomes the reference only if the r-change CI > 0 and the within-study r-change CI > 0. Audit: share 1-2,
     share >= 3, share 0 (must be 0 by construction: max |B1b - B1a++| reported), all pairs.
  2. g0, g_k, g_F on all 820 pairs, 95% CI from 50 mixture-resample refits; and the same g's fitted on inner
     out-of-fold B1a++ residuals (3-fold inner split over mixtures of the 820) - should match the in-sample ones.
  3. Pull/push per shared molecule (description): its mean contribution g_F * min-contrast / sd(F) over the pairs it
     shares, with its mean B1a++ error there (< 0 = rated closer than B1a++ predicts).
Term form: base F_mol (sum over shared molecules of min-side company contrast). Argument "variant" reserved for the
form chosen by the second session's variant test, if one beats the base.

Env:     mist
Usage:   python evaluation/B1b/eval_b1b.py
Writes:  results/eval_b1b/<date_time>.{txt,json}
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.b1b import B1b, shared_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402

LAM = 1e-3; N_BOOT = 2000; N_BOOT_G = 50


def main():
    rep = Report("eval_b1b", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_g": N_BOOT_G})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = np.asarray(mixture.quantile_per_word(op.probs_MxW), float)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    F, k = shared_features(q, A, B)
    rep.line(f"setting-2 pairs {int((k > 0).sum())}; F_mol mean {F[k > 0].mean():.2f} sd {F[k > 0].std():.2f}; corr(F, k) over them {np.corrcoef(F[k > 0], k[k > 0])[0, 1]:+.2f}")
    make = lambda: B1b(q, Xg, A, B, S, LAM, F, k)  # noqa: E731

    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: make().fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)
    kN = k[base.pair_N]
    s0m = kN == 0
    rep.line(f"share-0 check: max |B1b - B1a++| over share-0 test predictions = {np.abs(res.pred_N[s0m] - base.pred_N[s0m]).max():.2e}")
    rows, ok = [], False
    for subset, mask in [("setting 2 (primary)", kN > 0), ("  share 1-2", (kN >= 1) & (kN <= 2)), ("  share >= 3", kN >= 3),
                         ("share 0 (= B1a++ by construction)", s0m), ("all pairs", np.ones(len(kN), bool))]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        if subset.startswith("share 0"):
            wc = "-"
        else:
            d0, (lo, hi), _ = within_change(train, base, res, mask); wc = f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            if subset.startswith("setting 2"):
                ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]", wc])
    rep.section("1. B1b vs B1a++ (same folds)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "B1b RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 2): {'MET -> B1b becomes the reference' if ok else 'NOT met -> B1b stays a candidate; B1a++ remains the reference'}")

    allp = np.arange(P)
    full = make().fit(allp, y)
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: j for j, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    rng = np.random.default_rng(0)
    cnt = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_G)
    gb = []
    for b in range(N_BOOT_G):
        rr = np.repeat(allp, cnt[b, ja] * cnt[b, jb])
        gb.append(make().fit(rr, y[rr]).g)
    gb = np.array(gb)
    # inner out-of-fold residuals: 3 folds over mixtures
    fold_m = rng.integers(0, 3, len(mixes)); oof = np.full(P, np.nan)
    for f in range(3):
        te = np.where((fold_m[ja] == f) | (fold_m[jb] == f))[0]
        tr = np.where((fold_m[ja] != f) & (fold_m[jb] != f))[0]
        oof[te] = B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te) if len(te) else oof[te]
    ok_ = ~np.isnan(oof)
    g_oof = make().fit(allp[ok_], y[ok_], resid=(y - oof)[ok_]).g
    rep.section("2. Stage-2 numbers (all 820 pairs; per sd of the setting-2 training pairs)")
    for j, lab in enumerate(["g0 (shared pairs, intercept)", "g_k (number shared)", "g_F (standing-out shared molecules)"]):
        rep.line(f"{lab:38s} in-sample {full.g[j]:+.4f} [{np.percentile(gb[:, j], 2.5):+.4f}, {np.percentile(gb[:, j], 97.5):+.4f}]   inner-OOF residuals {g_oof[j]:+.4f}")
    rep.line(f"inner-OOF check covers {ok_.sum()} pairs (pairs with mixtures in two inner folds are left out)")

    names = {}
    try:
        for r in csv.DictReader(open(ROOT / "external/pnas_repo/SOTA/data/raw/openpom_ensemble_predictions_results.csv")):
            names[int(r["CID"])] = r["IUPAC Name"]
    except Exception:
        pass
    Sh = A * B; err = y - np.bincount(base.pair_N, base.pred_N, P) / np.bincount(base.pair_N, minlength=P)
    per = {}
    for p in np.where(k > 0)[0]:
        sh = np.where(Sh[p] > 0)[0]; cs = []
        for Mx in (A, B):
            un = np.where((Mx[p] > 0) & (Sh[p] == 0))[0]
            cs.append(np.linalg.norm(q[sh] - q[un].mean(0), axis=1) if len(un) else np.zeros(len(sh)))
        for i, c in zip(sh, np.minimum(cs[0], cs[1])):
            per.setdefault(i, []).append((full.g[2] * c / full.sd[1], err[p]))
    st = sorted([(i, np.mean([a for a, _ in v]), np.mean([b for _, b in v]), len(v)) for i, v in per.items() if len(v) >= 3], key=lambda t: t[1])
    words = [str(w) for w in op.words_W]
    fmt = lambda t: [names.get(int(op.cids_M[t[0]]), str(op.cids_M[t[0]]))[:38], t[3], f"{t[1]:+.4f}", f"{t[2]:+.3f}", ", ".join(words[j] for j in np.argsort(-q[t[0]])[:3])]  # noqa: E731
    rep.section("3. Per shared molecule (in >= 3 pairs): contribution g_F * contrast / sd(F); < 0 = pulls the pair closer")
    rep.table(["molecule", "pairs", "mean contribution", "mean B1a++ error", "top words"], [fmt(t) for t in st[:8]] + [["...", "", "", "", ""]] + [fmt(t) for t in st[-8:]])
    rep.line("The contribution is relative: the intercept g0 and g_k shift all shared pairs; molecules differ only through g_F.")
    rep.value("result", {"claim_met": bool(ok), "g": full.g.tolist(), "g_ci": [list(np.percentile(gb[:, j], [2.5, 97.5])) for j in range(3)], "g_oof": g_oof.tolist()})
    rep.save()


if __name__ == "__main__":
    main()
