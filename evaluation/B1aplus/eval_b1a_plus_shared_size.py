"""Does the shared part's dominance depend on mixture size? B1a++ with a size-dependent weight for shared molecules.

Lead (seen first in eval_setting2_shared_fraction.py, grid): at a similar shared fraction, the shared molecules
pull the pair closer than B1a++ predicts in 5-9-molecule mixtures, and less than predicted in 10-19-molecule
ones. Psychophysics reading: few components -> each is noticed (analytic), so a shared one is noticed in both;
many components -> one overall impression, a minor shared part goes unnoticed.
Model (on B1a++ = B1a + size + groups, lam 1e-3 on theta only):
    log w_i = theta . groups_i + [i in both A and B] * (alpha - beta * (log n - c))
n = size of the mixture the molecule sits in (each side its own), c = mean log n over setting-2 mixtures.
alpha: shared boost at the typical size; beta > 0: the boost shrinks (turns into a discount) as mixtures grow.
Variants (argument):
    free  alpha, beta unconstrained
    mono  beta >= 0 ENFORCED (beta = softplus(raw)): dominance of a shared molecule can only fall with size.
Pairs sharing no molecule only change through the refit theta.
Fixed before running: standard CV, change from B1a++ on the same folds; primary subset setting 2 (share >= 1);
claim only if r-change CI > 0 and within-study CI > 0. Audit: share 1-2, share >= 3, setting 1, all pairs.
alpha, beta on all 820 pairs with 95% CI from 50 mixture-resample refits; implied shared-weight multiplier
exp(alpha - beta (log n - c)) at n = 6 and n = 15.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_shared_size.py {free|mono}
Writes:  results/eval_b1a_plus_shared_size/<variant>_<date_time>.{txt,json}
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
N_BOOT = 2000
N_BOOT_X = 50
RAW0 = -5.0                    # mono: softplus(-5) = 0.007, i.e. start at beta ~ 0


class SharedSize(B1aPlus):
    def __init__(self, q, x, A, B, S, lam, LA_PxM, LB_PxM, ShA_PxM, mono):
        super().__init__(q, x, A, B, S, lam)
        self.K = self.x.shape[1]
        self.LA = torch.tensor(LA_PxM); self.LB = torch.tensor(LB_PxM); self.Sh = torch.tensor(ShA_PxM, dtype=torch.float64)
        self.mono = mono

    def ab(self, theta):
        beta = torch.nn.functional.softplus(theta[self.K + 1]) if self.mono else theta[self.K + 1]
        return theta[self.K], beta

    def distance(self, theta, idx):
        lw = self.x @ theta[:self.K]; alpha, beta = self.ab(theta)
        sh = self.Sh[idx]
        def mu(Mx, L):
            w = Mx[idx] * torch.exp(lw[None, :] + sh * (alpha - beta * L[idx]))
            m = w @ self.q / w.sum(1)[:, None]
            return m / m.norm(dim=1, keepdim=True)
        return (mu(self.A, self.LA) - mu(self.B, self.LB)).norm(dim=1)

    def fit(self, idx, y, w_pairs=None, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        wp = torch.ones(len(idx)) if w_pairs is None else torch.tensor(np.asarray(w_pairs, float))
        wp = wp / wp.sum()
        theta = torch.zeros(self.K + 2)
        if self.mono:
            theta[self.K + 1] = RAW0
        theta.requires_grad_(True)
        with torch.no_grad():
            D = self._design(self.distance(theta, idx), idx); sw = wp.sqrt()[:, None]
            lin0 = torch.linalg.lstsq(D * sw, (y * wp.sqrt())[:, None]).solution[:, 0]
        lin = lin0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([theta, lin], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, idx), idx) @ lin
            loss = (wp * (pred - y) ** 2).sum() + self.lam * (theta[:self.K] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.lin = theta.detach(), lin.detach()
        return self


def main():
    variant = sys.argv[1]
    assert variant in ("free", "mono")
    rep = Report("eval_b1a_plus_shared_size", {"variant": variant, "lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_x": N_BOOT_X})
    train = data.load_train(); y = evaluate.labels(train); P = len(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    Sh = A * B
    k = Sh.sum(1)
    la = np.log(A.sum(1)); lb = np.log(B.sum(1))
    c = float(np.r_[la[k > 0], lb[k > 0]].mean())
    LA = np.repeat((la - c)[:, None], A.shape[1], 1); LB = np.repeat((lb - c)[:, None], A.shape[1], 1)
    rep.line(f"variant {variant}; c = mean log n over setting-2 mixtures = {c:.2f} (n = {np.exp(c):.1f})")
    make = lambda: SharedSize(q, Xg, A, B, S, LAM, LA, LB, Sh, variant == "mono")  # noqa: E731

    base = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, Xg, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = evaluate.cross_validate(train, lambda tr, te: make().fit(tr, y[tr]).predict(te))
    print("  done CV", flush=True)

    s1 = base.no_overlap_N
    kN = k[base.pair_N]
    rows, ok = [], False
    for subset, mask in [("setting 2 (primary)", ~s1), ("  share 1-2", (kN >= 1) & (kN <= 2)), ("  share >= 3", kN >= 3),
                         ("setting 1", s1), ("all pairs", np.ones(len(s1), bool))]:
        s0 = evaluate.score_cv(train, base, mask_N=mask, n_boot=N_BOOT)
        s = evaluate.score_cv(train, res, reference=base, mask_N=mask, n_boot=N_BOOT)
        d0, (lo, hi), _ = within_change(train, base, res, mask)
        if subset.startswith("setting 2"):
            ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"])
    rep.section(f"1. + shared x size ({variant}) vs B1a++ (same folds)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 2): {'MET' if ok else 'NOT met'}")

    allp = np.arange(P)
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_X)

    def ab(m):
        a, b = m.ab(m.theta); return float(a), float(b)
    a0, b0 = ab(make().fit(allp, y))
    abb = np.array([ab(make().fit(allp, y, w_pairs=(cnt[i, ja] * cnt[i, jb]).astype(float) + 1e-12)) for i in range(N_BOOT_X)])
    mult = lambda a, b, n: np.exp(a - b * (np.log(n) - c))  # noqa: E731
    rep.section("2. Fitted shared-molecule terms (all 820 pairs)")
    rep.line(f"alpha = {a0:+.3f} [{np.percentile(abb[:, 0], 2.5):+.3f}, {np.percentile(abb[:, 0], 97.5):+.3f}]")
    rep.line(f"beta  = {b0:+.3f} [{np.percentile(abb[:, 1], 2.5):+.3f}, {np.percentile(abb[:, 1], 97.5):+.3f}]   (> 0: shared boost shrinks with size)")
    for n in (6, 15):
        mb = mult(abb[:, 0], abb[:, 1], n)
        rep.line(f"shared molecule weight x{mult(a0, b0, n):.2f} [x{np.percentile(mb, 2.5):.2f}, x{np.percentile(mb, 97.5):.2f}] in a {n}-molecule mixture")
    rep.value("result", {"variant": variant, "alpha": a0, "beta": b0, "alpha_ci": list(np.percentile(abb[:, 0], [2.5, 97.5])),
                         "beta_ci": list(np.percentile(abb[:, 1], [2.5, 97.5])), "claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
