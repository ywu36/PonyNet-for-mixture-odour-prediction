"""Can the GROUPS a pair has in common (or not) explain what B1a++ misses? Lasso on group multi-hot pair features.

Groups: the 24 RDKit fr_ groups of B1a++ (carried by >= 5 of the 235 molecules, identical columns merged).
Per pair (A, B), with frac_X(g) = share of mixture X's molecules carrying group g:
    S_g  shared part:   (number of SHARED molecules carrying g) / mean size        "common molecule with g"
    C_g  common group:  min(fracA'(g), fracB'(g)) over the UNSHARED molecules       "different molecules, same group"
    D_g  group gap:     |fracA'(g) - fracB'(g)| over the UNSHARED molecules         "one side has g, the other not"
(A', B' = unshared molecules; for setting-1 pairs A' = A, B' = B and S_g = 0.) 72 features, standardised.
Model, inside each standard-CV training fold: fit B1a++; residual = target - B1a++ fit on the training pairs;
LassoCV (5 inner folds over pairs) on the residual; test prediction = B1a++ + lasso. Nothing fitted on test pairs.
Fixed before running: change from B1a++ on the same folds; primary subset setting 2 (share >= 1); claim only if
r-change CI > 0 and within-study CI > 0. Setting 1, share 1-2 and all pairs reported. Lasso on all 820 pairs:
selected features and signs; selection frequency over 50 mixture-resample refits (stability).
Sign reading: coefficient < 0 = the feature makes pairs closer than B1a++ predicts (pull); > 0 = push apart.

Env:     mist
Usage:   python evaluation/B1aplus/eval_group_lasso_residual.py
Writes:  results/eval_group_lasso_residual/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LassoCV

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402

LAM = 1e-3; N_BOOT = 2000; N_STAB = 50


def pair_features(A, B, Xg):
    Sh = A * B; UA = A - Sh; UB = B - Sh
    nbar = (A.sum(1) + B.sum(1)) / 2
    S = (Sh @ Xg) / nbar[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.nan_to_num((UA @ Xg) / UA.sum(1, keepdims=True)); fb = np.nan_to_num((UB @ Xg) / UB.sum(1, keepdims=True))
    return np.hstack([S, np.minimum(fa, fb), np.abs(fa - fb)])


class Stacked:
    def __init__(self, q, Xg, A, B, S, F):
        self.b = B1aPlus(q, Xg, A, B, S, LAM); self.F = F

    def fit(self, idx, y):
        self.b.fit(idx, y)
        r = y - self.b.predict(idx)
        F = self.F[idx]; self.mu = F.mean(0); self.sd = F.std(0) + 1e-9
        self.l = LassoCV(cv=5, alphas=50, max_iter=20000, random_state=0).fit((F - self.mu) / self.sd, r)
        return self

    def predict(self, idx):
        return self.b.predict(idx) + self.l.predict((self.F[idx] - self.mu) / self.sd)


def main():
    rep = Report("eval_group_lasso_residual", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_stab": N_STAB})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, names = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    short = [n.split(" (= ")[0] for n in names]
    F = pair_features(A, B, Xg)
    fn = [f"S:{g}" for g in short] + [f"C:{g}" for g in short] + [f"D:{g}" for g in short]

    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: Stacked(q, Xg, A, B, S, F).fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)
    s1 = base.no_overlap_N; k = (A * B).sum(1); kN = k[base.pair_N]
    rows, ok = [], False
    for subset, mask in [("setting 2 (primary)", ~s1), ("  share 1-2", (kN >= 1) & (kN <= 2)), ("setting 1", s1), ("all pairs", np.ones(len(s1), bool))]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        d0, (lo, hi), _ = within_change(train, base, res, mask)
        if subset.startswith("setting 2"):
            ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"])
    rep.section("1. B1a++ + group lasso on its residual vs B1a++ (same folds)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 2): {'MET' if ok else 'NOT met'}")

    allp = np.arange(P)
    full = Stacked(q, Xg, A, B, S, F).fit(allp, y)
    coef = full.l.coef_
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    rng = np.random.default_rng(0); sel = np.zeros(len(fn)); sgn = np.zeros(len(fn))
    r0 = y - full.b.predict(allp); Fz = (F - full.mu) / full.sd
    for _ in range(N_STAB):
        c = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)))
        rr = np.repeat(allp, c[ja] * c[jb])
        cf = LassoCV(cv=5, alphas=50, max_iter=20000, random_state=0).fit(Fz[rr], r0[rr]).coef_
        sel += cf != 0; sgn += np.sign(cf)
    rep.section(f"2. Lasso on all 820 pairs: {int((coef != 0).sum())} of {len(fn)} features selected (alpha {full.l.alpha_:.4f})")
    order = np.argsort(-np.abs(coef))
    rows = [[fn[j], f"{coef[j]:+.4f}", f"{sel[j] / N_STAB:.0%}", f"{sgn[j] / max(sel[j], 1):+.2f}", int((F[:, j] > 0).sum())] for j in order if coef[j] != 0]
    rep.table(["feature", "coef (per sd)", f"selected in {N_STAB} resamples", "sign agreement", "pairs with feature > 0"], rows or [["none", "", "", "", ""]])
    stable = [[fn[j], f"{sel[j] / N_STAB:.0%}", f"{sgn[j] / max(sel[j], 1):+.2f}"] for j in np.argsort(-sel) if sel[j] / N_STAB >= 0.5]
    rep.line("features selected in >= 50% of resamples: " + (", ".join(f"{a} ({b}, sign {c})" for a, b, c in stable) if stable else "none"))
    rep.line("S = shared molecule carries group; C = different molecules on both sides carry group; D = group on one side only.")
    rep.value("result", {"claim_met": bool(ok), "selected": {fn[j]: float(coef[j]) for j in range(len(fn)) if coef[j] != 0}})
    rep.save()


if __name__ == "__main__":
    main()
