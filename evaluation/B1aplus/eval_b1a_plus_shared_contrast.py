"""Does a shared molecule pull harder when it STANDS OUT from the rest of its mixture?

Known (pred-mixture-6c, O12c): a shared molecule weighs about x1.19 its share (alpha > 0 in every CV fold), not
size-dependent, not tuned out. Hypothesis: the extra pull comes from shared molecules that smell unlike their
company: noticed as a separate note in both mixtures. Blending shared molecules add only their share.
Contrast of shared molecule i in mixture A: || q_i - mean q over A's UNSHARED molecules || (open-POM quantile
profiles, 138 words); standardised over all (pair, side, shared molecule) instances; 0 if A has no unshared
molecule. Model on B1a++ (lam 1e-3 on theta only):
    log w_i = theta . groups_i + [i shared] * (alpha + beta * contrast_i)
Prediction fixed before running: beta > 0. Standard CV, change from B1a++ on the same folds; primary subset
setting 2; claim only if r-change CI > 0 and within-study CI > 0. Setting 1, share 1-2 and all pairs reported.
alpha, beta on all 820 pairs, 95% CI from 50 mixture-resample refits. Description: the shared molecules with
the highest and lowest mean contrast and their mean B1a++ error (target - CV prediction) in the pairs they share.

Env:     mist
Control variant (argument "ctrl", added after seeing beta > 0): the same model plus the shared molecule's
ABSOLUTE salience ||q_i|| (distance from the average molecule's smell, independent of its company), z-scored
over the same instances:  [i shared] * (alpha + beta * contrast_i + beta_s * salience_i).
If beta keeps its size and CI > 0, "stands out from its company" is not just "strong smell in general".
Usage:   python evaluation/B1aplus/eval_b1a_plus_shared_contrast.py [ctrl]
Writes:  results/eval_b1a_plus_shared_contrast[_ctrl]/<date_time>.{txt,json}
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402
from eval_b1a_plus_shared_size import SharedSize  # noqa: E402

LAM = 1e-3; N_BOOT = 2000; N_BOOT_X = 50


class SharedTerms(SharedSize):
    """[i shared] * (alpha + sum_j beta_j * Z_j[i]) with Z_j per side (P x M); all extra numbers unpenalised."""

    def __init__(self, q, x, A, B, S, lam, ZsA, ZsB, Sh):
        super().__init__(q, x, A, B, S, lam, ZsA[0], ZsB[0], Sh, False)
        self.ZA = [__import__("torch").tensor(z) for z in ZsA]; self.ZB = [__import__("torch").tensor(z) for z in ZsB]
        self.J = len(ZsA)

    def ab(self, theta):
        return [theta[self.K + j] for j in range(self.J + 1)]

    def distance(self, theta, idx):
        import torch
        lw = self.x @ theta[:self.K]; c = self.ab(theta)
        sh = self.Sh[idx]
        def mu(Mx, Z):
            extra = c[0] + sum(c[j + 1] * Z[j][idx] for j in range(self.J))
            w = Mx[idx] * torch.exp(lw[None, :] + sh * extra)
            m = w @ self.q / w.sum(1)[:, None]
            return m / m.norm(dim=1, keepdim=True)
        return (mu(self.A, self.ZA) - mu(self.B, self.ZB)).norm(dim=1)

    def fit(self, idx, y, w_pairs=None, max_iter=300):
        import torch
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        wp = torch.ones(len(idx)) if w_pairs is None else torch.tensor(np.asarray(w_pairs, float)); wp = wp / wp.sum()
        theta = torch.zeros(self.K + self.J + 1, requires_grad=True)
        with torch.no_grad():
            D = self._design(self.distance(theta, idx), idx); sw = wp.sqrt()[:, None]
            lin0 = torch.linalg.lstsq(D * sw, (y * wp.sqrt())[:, None]).solution[:, 0]
        lin = lin0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([theta, lin], max_iter=max_iter, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, idx), idx) @ lin
            loss = (wp * (pred - y) ** 2).sum() + self.lam * (theta[:self.K] ** 2).sum()
            loss.backward(); return loss
        opt.step(closure)
        self.theta, self.lin = theta.detach(), lin.detach()
        return self


def main():
    ctrl = len(sys.argv) > 1 and sys.argv[1] == "ctrl"
    rep = Report("eval_b1a_plus_shared_contrast" + ("_ctrl" if ctrl else ""), {"ctrl": ctrl, "lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_x": N_BOOT_X})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    Sh = A * B

    def contrast(Mx):
        C = np.zeros(Mx.shape)
        for p in range(P):
            sh = np.where(Sh[p] > 0)[0]; un = np.where((Mx[p] > 0) & (Sh[p] == 0))[0]
            if len(sh) and len(un):
                C[p, sh] = np.linalg.norm(q[sh] - q[un].mean(0), axis=1)
        return C
    CA, CB = contrast(A), contrast(B)
    inst = np.r_[CA[(Sh > 0) & (CA > 0)], CB[(Sh > 0) & (CB > 0)]]
    mu, sd = inst.mean(), inst.std()
    ZA = np.where((Sh > 0) & (CA > 0), (CA - mu) / sd, 0.0); ZB = np.where((Sh > 0) & (CB > 0), (CB - mu) / sd, 0.0)
    rep.line(f"contrast instances: {len(inst)} (shared molecule x side); raw mean {mu:.2f}, sd {sd:.2f}")
    if ctrl:
        sal = np.linalg.norm(q, axis=1)
        mA = (Sh > 0) & (CA > 0); mB = (Sh > 0) & (CB > 0)
        si = np.r_[np.broadcast_to(sal, A.shape)[mA], np.broadcast_to(sal, A.shape)[mB]]
        SA = np.where(mA, (sal[None, :] - si.mean()) / si.std(), 0.0); SB = np.where(mB, (sal[None, :] - si.mean()) / si.std(), 0.0)
        rep.line(f"correlation of contrast and absolute salience over shared instances: {np.corrcoef(np.r_[ZA[mA], ZB[mB]], np.r_[SA[mA], SB[mB]])[0, 1]:+.2f}")
        make = lambda: SharedTerms(q, Xg, A, B, S, LAM, [ZA, SA], [ZB, SB], Sh)  # noqa: E731
    else:
        # SharedSize adds sh * (alpha - beta * L): pass L = -z so the term is alpha + beta * contrast_z
        make = lambda: SharedSize(q, Xg, A, B, S, LAM, -ZA, -ZB, Sh, False)  # noqa: E731

    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: make().fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)
    s1 = base.no_overlap_N; k = Sh.sum(1); kN = k[base.pair_N]
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
    rep.section("1. + shared alpha + beta * contrast vs B1a++ (same folds)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 2): {'MET' if ok else 'NOT met'}")

    allp = np.arange(P)
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_X)
    ab = lambda m: [float(v) for v in m.ab(m.theta)]  # noqa: E731
    fit0 = ab(make().fit(allp, y)); a0, b0 = fit0[0], fit0[1]
    abb = np.array([ab(make().fit(allp, y, w_pairs=(cnt[i, ja] * cnt[i, jb]).astype(float) + 1e-12)) for i in range(N_BOOT_X)])
    rep.section("2. Fitted numbers (all 820 pairs)")
    rep.line(f"alpha = {a0:+.3f} [{np.percentile(abb[:, 0], 2.5):+.3f}, {np.percentile(abb[:, 0], 97.5):+.3f}] (shared weight x{np.exp(a0):.2f} at average contrast)")
    rep.line(f"beta  = {b0:+.3f} [{np.percentile(abb[:, 1], 2.5):+.3f}, {np.percentile(abb[:, 1], 97.5):+.3f}] per sd of contrast (prediction: > 0)")
    if ctrl:
        rep.line(f"beta_s = {fit0[2]:+.3f} [{np.percentile(abb[:, 2], 2.5):+.3f}, {np.percentile(abb[:, 2], 97.5):+.3f}] per sd of absolute salience (control)")

    pred = np.bincount(base.pair_N, base.pred_N, P) / np.bincount(base.pair_N, minlength=P); err = y - pred
    names = {}
    try:
        for r in csv.DictReader(open(ROOT / "external/pnas_repo/SOTA/data/raw/openpom_ensemble_predictions_results.csv")):
            names[int(r["CID"])] = r["IUPAC Name"]
    except Exception:
        pass
    words = [str(w) for w in op.words_W]
    per = {}
    for p in range(P):
        for i in np.where(Sh[p] > 0)[0]:
            z = [v for v in (ZA[p, i], ZB[p, i]) if v != 0]
            if z:
                per.setdefault(i, []).append((np.mean(z), err[p]))
    st = [(i, np.mean([a for a, _ in v]), np.mean([b for _, b in v]), len(v)) for i, v in per.items() if len(v) >= 3]
    st.sort(key=lambda t: -t[1])
    rep.section("3. Shared molecules (in >= 3 pairs) by mean contrast; err < 0 = rated closer than B1a++ predicts")
    fmt = lambda t: [names.get(int(op.cids_M[t[0]]), str(op.cids_M[t[0]]))[:38], t[3], f"{t[1]:+.2f}", f"{t[2]:+.3f}", ", ".join(words[j] for j in np.argsort(-q[t[0]])[:3])]  # noqa: E731
    rep.table(["molecule", "pairs", "mean contrast z", "mean err", "top words"], [fmt(t) for t in st[:8]] + [["...", "", "", "", ""]] + [fmt(t) for t in st[-8:]])
    from scipy.stats import spearmanr
    rep.line(f"over {len(st)} molecules: Spearman rho(mean contrast, mean err) = {spearmanr([t[1] for t in st], [t[2] for t in st]).statistic:+.2f} "
             "(prediction: < 0; description, molecules share pairs)")
    rep.value("result", {"alpha": a0, "beta": b0, "alpha_ci": list(np.percentile(abb[:, 0], [2.5, 97.5])), "beta_ci": list(np.percentile(abb[:, 1], [2.5, 97.5])), "claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
