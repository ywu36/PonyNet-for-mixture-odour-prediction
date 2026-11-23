"""Remove each study's label bias in TRAINING only; predict every new pair with one average correction.

The DREAM labels merge triangle-test fractions and similarity ratings onto one 0-1 scale without correction
(the DREAM 2 paper: "overlooks paradigm-specific biases"). Between-study level/scale shifts can swamp the small
within-study differences the content weights (theta) must learn from.
Variants (argument), all on B1a++ (= B1a + size + groups, lam 1e-3 on theta):
  z    labels z-scored within each study, using the training fold's mean and sd of that study; B1a++ fitted
       on z; prediction mapped back with the pair-weighted average of the training studies' means and sds.
  z2   as z, but the theta penalty is rescaled to z units: lam_z = lam / (mean over training pairs of their
       study's sd^2), so theta is regularised as strongly as in B1a++ (z showed theta sd 0.10 -> 0.88 without it).
  ls   per-study level c_s and scale g_s learned jointly:  y = g_s * (B1a++ line) + c_s,
       log g_s and c_s centred (pair-weighted mean 0 over training pairs) so the shared line is the average
       study; a tiny ridge 1e-3 on log g_s and c_s. Prediction: g = 1, c = 0 (the average study).
No study label is used at prediction. Fixed before running: standard CV, change from B1a++ on the same folds;
primary subset setting 1; claim only if r-change CI > 0 and within-study r-change CI > 0 (within-study r
ignores level shifts, so it shows whether theta itself improved). Setting 2 and all pairs reported.
Also on all 820 pairs: correlation of theta with B1a++'s theta, size coefficients, per-study level and scale (ls).

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_debias.py {z|z2|ls}
Writes:  results/eval_b1a_plus_debias_<variant>/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402

LAM = 1e-3
LAM_STUDY = 1e-3
N_BOOT = 2000


class ZScored:
    """B1a++ fitted on within-study z-scores (training-fold statistics); predictions mapped back."""

    def __init__(self, q, x, A, B, S, lam, study_P, scale_lam=False):
        self.m = B1aPlus(q, x, A, B, S, lam); self.study = study_P; self.lam = lam; self.scale_lam = scale_lam

    def fit(self, idx, y):
        idx = np.asarray(idx); st = self.study[idx]
        mu = {s: y[st == s].mean() for s in np.unique(st)}
        sd = {s: max(y[st == s].std(), 1e-3) for s in np.unique(st)}
        z = np.array([(v - mu[s]) / sd[s] for v, s in zip(y, st)])
        self.mu0 = float(np.mean([mu[s] for s in st])); self.sd0 = float(np.mean([sd[s] for s in st]))
        if self.scale_lam:
            self.m.lam = self.lam / float(np.mean([sd[s] ** 2 for s in st]))
        self.m.fit(idx, z)
        self.mu, self.sd = mu, sd
        return self

    def predict(self, idx):
        return self.m.predict(idx) * self.sd0 + self.mu0


class LevelScale(B1aPlus):
    def __init__(self, q, x, A, B, S, lam, study_P):
        super().__init__(q, x, A, B, S, lam)
        self.names = sorted(set(study_P)); self.sidx = torch.tensor([self.names.index(s) for s in study_P])

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        si = self.sidx[idx]; G = len(self.names)
        cnt = torch.bincount(si, minlength=G).double(); wst = cnt / cnt.sum()
        theta = torch.zeros(self.x.shape[1], requires_grad=True)
        with torch.no_grad():
            D = self._design(self.distance(theta, idx), idx)
            lin0 = torch.linalg.lstsq(D, y[:, None]).solution[:, 0]
        lin = lin0.clone().requires_grad_(True)
        lg = torch.zeros(G, requires_grad=True); c = torch.zeros(G, requires_grad=True)
        opt = torch.optim.LBFGS([theta, lin, lg, c], max_iter=max_iter, line_search_fn="strong_wolfe")

        def centred():
            return lg - (wst * lg).sum(), c - (wst * c).sum()

        def closure():
            opt.zero_grad()
            lgc, cc = centred()
            core = self._design(self.distance(theta, idx), idx) @ lin
            pred = torch.exp(lgc[si]) * core + cc[si]
            loss = ((pred - y) ** 2).mean() + self.lam * (theta ** 2).sum() + LAM_STUDY * ((lgc ** 2).sum() + (cc ** 2).sum())
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.lin = theta.detach(), lin.detach()
        with torch.no_grad():
            lgc, cc = centred()
        self.scale = {n: float(torch.exp(lgc[i])) for i, n in enumerate(self.names) if cnt[i] > 0}
        self.level = {n: float(cc[i]) for i, n in enumerate(self.names) if cnt[i] > 0}
        return self


def main():
    variant = sys.argv[1]
    assert variant in ("z", "z2", "ls")
    rep = Report(f"eval_b1a_plus_debias_{variant}", {"variant": variant, "lam": LAM, "lam_study": LAM_STUDY, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, names = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    study = np.array([p.dataset for p in train])
    zs = variant in ("z", "z2")
    make = (lambda: ZScored(q, Xg, A, B, S, LAM, study, variant == "z2")) if zs else (lambda: LevelScale(q, Xg, A, B, S, LAM, study))

    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: make().fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)

    s1 = base.no_overlap_N
    rows, ok = [], False
    for subset, mask in [("setting 1 (primary)", s1), ("setting 2", ~s1), ("all pairs", np.ones(len(s1), bool))]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        d0, (lo, hi), _ = within_change(train, base, res, mask)
        if subset.startswith("setting 1"):
            ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"])
    rep.section(f"1. Debiased training ({variant}) vs B1a++ (same folds; no study label at prediction)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 1): {'MET' if ok else 'NOT met'}")

    allp = np.arange(P)
    b = B1aPlus(q, Xg, A, B, S, LAM).fit(allp, y)
    n = make().fit(allp, y)
    nm = n.m if zs else n
    th0, th1 = b.theta.numpy(), nm.theta.numpy()
    rep.section("2. What changed (all 820 pairs)")
    rep.line(f"theta: r with B1a++ theta {np.corrcoef(th0, th1)[0, 1]:+.2f}; sd {th0.std():.3f} -> {th1.std():.3f}")
    short = [s.split(' (= ')[0] for s in names]
    top = lambda th: ", ".join(f"{short[i]} {th[i]:+.2f}" for i in np.argsort(-th)[:4]) + " ... " + ", ".join(f"{short[i]} {th[i]:+.2f}" for i in np.argsort(th)[:2])  # noqa: E731
    rep.line(f"B1a++ theta: {top(th0)}"); rep.line(f"new   theta: {top(th1)}")
    rep.line(f"line [distance, intercept, log nA+log nB, |log nA-log nB|]: B1a++ {np.round(b.lin.numpy(), 3).tolist()}  new {np.round(nm.lin.numpy(), 3).tolist()}"
             + ("  (new is in z units)" if zs else ""))
    if zs:
        rep.line(f"theta penalty used: {n.m.lam:.4f}")
        rep.table(["study", "pairs", "mean", "sd"], [[s, int((study == s).sum()), f"{n.mu[s]:.3f}", f"{n.sd[s]:.3f}"] for s in sorted(n.mu)])
    else:
        rep.table(["study", "pairs", "level c_s", "scale g_s"], [[s, int((study == s).sum()), f"{n.level[s]:+.3f}", f"{n.scale[s]:.2f}"] for s in sorted(n.level)])
    rep.value("result", {"variant": variant, "claim_met": bool(ok), "theta_r": float(np.corrcoef(th0, th1)[0, 1])})
    rep.save()


if __name__ == "__main__":
    main()
