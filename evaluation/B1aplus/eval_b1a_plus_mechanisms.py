"""Four receptor/judgment mechanisms, each ONE number on top of B1a++ (= B1a + size + groups, lam 1e-3).

  A  antagonism grows with mixture size (sulfur): log w_i += kappa * sulfur_i * (log n_mix - mean log n)
     sulfur_i = molecule carries any S atom (class "sulfur"). kappa < 0: sulfur dominates less in big mixtures.
  B  per-channel soft competition: word w of a mixture = sum_i a_iw q_iw with a_iw = softmax over member
     molecules of (log w_i + tau * q_iw). tau = 0: B1a++ (weighted mean); tau -> inf: max per word.
  C  structural overlap as a receptor-input proxy: extra line term = fragment-overlap distance of the two
     mixtures (Morgan radius 1 fragment sets, 1 - Tanimoto). Radius 1 was the best of 3 radii seen earlier
     (setting-1 r 0.249), so this is a seen lead.
  D  session anchor: before length 1, subtract lam_a * (mean B1a profile of all mixtures in the same set/study).
     Uses inputs only (which mixtures are in the set), never ratings; at test the set is the test file.
     The within-study check guards against D acting as a study offset.
The extra number is unpenalised (A, B, D) or a free line coefficient (C); theta keeps lam 1e-3.
Fixed before running: standard CV, change from B1a++ on the same folds; primary subset setting 1 (pairs
sharing no molecule); claim only if r-change CI > 0 and within-study CI > 0. Setting 2 and all pairs
reported. Extra number on all 820 pairs with 95% CI from 50 mixture-resample refits.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_mechanisms.py {A|B|C|D}
Writes:  results/eval_b1a_plus_mechanisms/<date_time>.{txt,json}
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
from eval_b1a_plus_group_graph import B1aPlusGraph  # noqa: E402
from eval_morgan_b1a_frag import frag_distance  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402

LAM = 1e-3
N_BOOT = 2000
N_BOOT_X = 50


class _Extra(B1aPlusGraph):
    """B1aPlusGraph's fit (theta + one unpenalised extra number at theta[K]); subclasses override distance."""

    def __init__(self, q, x, A, B, S, lam):
        P, M = A.shape
        super().__init__(q, x, A, B, S, lam, np.zeros((P, M)), np.zeros((P, M)))


class SoftChannel(_Extra):
    def __init__(self, q, x, A, B, S, lam):
        super().__init__(q, x, A, B, S, lam)
        K = int(max(A.sum(1).max(), B.sum(1).max()))
        def pad(Mx):
            idx = np.zeros((len(Mx), K), int); msk = np.zeros((len(Mx), K), bool)
            for p in range(len(Mx)):
                m = np.where(Mx[p] > 0)[0]; idx[p, :len(m)] = m; msk[p, :len(m)] = True
            return torch.tensor(idx), torch.tensor(msk)
        self.iA, self.mA = pad(A); self.iB, self.mB = pad(B)

    def distance(self, theta, idx):
        lw = self.x @ theta[:self.K]; tau = theta[self.K]
        def mu(I, Mk):
            ii, mk = I[idx], Mk[idx]
            qg = self.q[ii]                                             # P x K x W
            logit = lw[ii][:, :, None] + tau * qg
            logit = torch.where(mk[:, :, None], logit, torch.tensor(-1e30))
            a = torch.softmax(logit, dim=1)
            m = (a * qg).sum(1)
            return m / m.norm(dim=1, keepdim=True)
        return (mu(self.iA, self.mA) - mu(self.iB, self.mB)).norm(dim=1)


class Anchor(_Extra):
    def __init__(self, q, x, A, B, S, lam, anchor_PxW):
        super().__init__(q, x, A, B, S, lam)
        self.anc = torch.tensor(anchor_PxW)

    def distance(self, theta, idx):
        w = torch.exp(self.x @ theta[:self.K]); la = theta[self.K]
        def mu(Mx):
            m = (Mx[idx] * w) @ self.q / (Mx[idx] @ w)[:, None] - la * self.anc[idx]
            return m / m.norm(dim=1, keepdim=True)
        return (mu(self.A) - mu(self.B)).norm(dim=1)


def main():
    mech = sys.argv[1]
    rep = Report("eval_b1a_plus_mechanisms", {"mechanism": mech, "lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_boot_x": N_BOOT_X})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    smiles = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups(smiles)), ids)

    if mech == "A":
        Xc, nc = piece_matrix(pp.units_smarts(smiles, pp.CLASSES), ids)
        sul = Xc[:, nc.index("sulfur")]
        la = np.log([len(p.a) for p in train]); lb = np.log([len(p.b) for p in train]); mlog = np.r_[la, lb].mean()
        NA = A * sul[None, :] * (la - mlog)[:, None]; NB = B * sul[None, :] * (lb - mlog)[:, None]
        make = lambda: B1aPlusGraph(q, Xg, A, B, S, LAM, NA, NB)                       # noqa: E731
        label, xname = "+ sulfur x mixture size", "kappa"
        rep.line(f"sulfur molecules: {int(sul.sum())}; log n centred at {mlog:.2f} (n = {np.exp(mlog):.1f})")
    elif mech == "B":
        make = lambda: SoftChannel(q, Xg, A, B, S, LAM)                                  # noqa: E731
        label, xname = "+ per-channel soft competition", "tau"
    elif mech == "C":
        fd = frag_distance(train, op.cids_M, radius=1)
        S2 = np.c_[S, fd]
        make = lambda: B1aPlus(q, Xg, A, B, S2, LAM)                                     # noqa: E731
        label, xname = "+ structural overlap (frag r1)", "line coefficient"
        rep.line(f"fragment-overlap distance: mean {fd.mean():.3f}, sd {fd.std():.3f}; corr with B1a distance "
                 f"{np.corrcoef(fd, mixture.paper_distance(train, op.cids_M, op.probs_MxW))[0, 1]:+.2f}")
    elif mech == "D":
        mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted)
        row = {int(c): i for i, c in enumerate(op.cids_M)}
        prof = {m: q[[row[c] for c in m]].mean(0) for m in mixes}
        by_set = {}
        for p in train:
            by_set.setdefault(p.dataset, set()).update([p.a, p.b])
        anc = {s: np.mean([prof[m] for m in ms], 0) for s, ms in by_set.items()}
        anchor = np.stack([anc[p.dataset] for p in train])
        make = lambda: Anchor(q, Xg, A, B, S, LAM, anchor)                               # noqa: E731
        label, xname = "+ session anchor", "lam_anchor"
        rep.line(f"sets: {len(anc)}; anchor length (mean profile of a set): " + ", ".join(f"{np.linalg.norm(v):.2f}" for v in anc.values()))
    else:
        raise SystemExit("mechanism must be A, B, C or D")

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
    rep.section(f"{mech}. {label} vs B1a++ (same folds)")
    rep.table(["subset", "pairs", "B1a++ RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (setting 1): {'MET' if ok else 'NOT met'}")

    allp = np.arange(len(train))
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ja = np.array([im[p.a] for p in train]); jb = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_X)

    def extra(m):
        return float(m.lin[-1]) if mech == "C" else float(m.theta[-1])
    x0 = extra(make().fit(allp, y))
    xb = np.array([extra(make().fit(allp, y, w_pairs=(cnt[b, ja] * cnt[b, jb]).astype(float) + 1e-12)) for b in range(N_BOOT_X)])
    rep.line(f"{xname} (all 820 pairs) = {x0:+.4f} [{np.percentile(xb, 2.5):+.4f}, {np.percentile(xb, 97.5):+.4f}] ({N_BOOT_X} refits)")
    rep.value("result", {"mechanism": mech, "x": x0, "x_ci": [float(np.percentile(xb, 2.5)), float(np.percentile(xb, 97.5))], "claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
