# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden vector size   G = RDKit groups (24)
#   T = Ma mixtures (198)   P = DREAM development pairs (514)   K = members
# ---------------------------------------------------------------------------------------------
"""O3-W redo with Ma only (the user, 2026-11-19): GoodScents lean-to label was invalid (catalogue lines, never a blend;
see findings C17 and memory goodscents-pair-usage), so every joint merge fit is redone WITHOUT GoodScents. GoodScents
is not used here at all: as a symmetric A + B -> group-word map it has no valid signal for merge weights (which
molecule counts more in A + B is exactly the lean-to question).

Merge only, frozen backbone, hidden-layer template (no decoding):
  importance w = exp(group weights . x),  z(A + B) = (w_A v_A + w_B v_B) / (w_A + w_B)
  Ma direction regulariser (the user's form): s = cos(z, v_A) - cos(z, v_B); push s positive when A dominates
  (L > 0), negative otherwise: softplus(-y s / T), y = +1 / -1, T one temperature.
  DREAM: O1.2b distance on the template, line.  loss = DREAM MSE + lam |group weights|^2 + c var(y_train) Ma loss.
c = 0 is DREAM only (27 numbers). DREAM side stays 27 numbers; T belongs to the Ma loss only.
Backbones: o2c400 (O2-C-Single middle layer, 10 members, the template that tied O1.2b) and o3m_w64sparse1 (O3s,
3 members). Checks: Ma 5 unseen-molecule folds AUROC of s with weights from Ma alone (is the Ma rule learnable on this
backbone?); DREAM 50 x 5 change from O1.2b and from c = 0 on the same backbone (paired, 4 subsets).

Env:     mist
Usage:   python evaluation/O3/eval_o3w_ma.py [--spaces o2c400 o3m_w64sparse1] [--c 1 10] [--jobs 8] [--repeats 50]
Writes:  results/eval_o3w_ma/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from eval_readout_latent import backbone, vectors  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402

NAMES = {"o2c400": "O2-C-Single", "o3m_w64sparse1": "O3s"}


def ma_stats(v, x, a, b, y):
    """Per Ma mixture: |v_a|^2, |v_b|^2, v_a.v_b, group flags, sign (+1 A dominates)."""
    v, x = torch.as_tensor(v), torch.as_tensor(x); va, vb = v[a], v[b]
    return ((va * va).sum(1), (vb * vb).sum(1), (va * vb).sum(1), x[a], x[b], torch.as_tensor(2.0 * y - 1))


def ma_score(s, gw):
    """s = cos(z, v_A) - cos(z, v_B), z = importance-weighted mean of v_A, v_B (exact, from the stats)."""
    aa, bb, ab, xa, xb, _ = s
    al = torch.sigmoid(xa @ gw - xb @ gw)
    zz = (al ** 2 * aa + 2 * al * (1 - al) * ab + (1 - al) ** 2 * bb).clamp_min(1e-12).sqrt()
    return (al * aa + (1 - al) * ab) / (zz * aa.clamp_min(1e-12).sqrt()) - (al * ab + (1 - al) * bb) / (zz * bb.clamp_min(1e-12).sqrt())


def ma_loss(s, gw, lt):
    return torch.nn.functional.softplus(-s[5] * ma_score(s, gw) / torch.exp(lt)).mean()


def fit_ma_only(s, n_groups, lam=1e-3):
    gw = torch.zeros(n_groups, requires_grad=True); lt = torch.tensor(float(np.log(0.1)), requires_grad=True)
    opt = torch.optim.LBFGS([gw, lt], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); loss = lam * (gw ** 2).sum() + ma_loss(s, gw, lt); loss.backward(); return loss
    opt.step(closure)
    return gw.detach()


class JointMa(O12b):
    """Hidden template in O1.2b's form; the same group weights also answer Ma's direction (c > 0)."""

    def __init__(self, v, x, A, B, s_ma, c, lam=LAM):
        super().__init__(v, x, A, B, True, True, lam=lam); self.s_ma, self.c = s_ma, c

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float)); vy = float(y.var())
        theta = torch.zeros(self.x.shape[1], requires_grad=True); k = torch.tensor(self.k0, requires_grad=True)
        lt = torch.tensor(float(np.log(0.1)), requires_grad=True)
        with torch.no_grad():
            D = self._design(self.distance(theta, k, idx), idx)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone().requires_grad_(True)
        opt = torch.optim.LBFGS([lin, theta, k, lt], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, k, idx), idx) @ lin
            loss = ((pred - y) ** 2).mean() + self.lam * (theta ** 2).sum()
            if self.c > 0:
                loss = loss + self.c * vy * ma_loss(self.s_ma, theta, lt)
            loss.backward(); return loss
        opt.step(closure)
        self.theta, self.k, self.lin = theta.detach(), k.detach(), lin.detach()
        return self


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spaces", nargs="+", default=["o2c400", "o3m_w64sparse1"])
    ap.add_argument("--c", type=float, nargs="+", default=[1.0, 10.0]); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=evaluate.CV_REPEATS); a_ = ap.parse_args()
    dev = pick_device()
    rep = Report("eval_o3w_ma", {"spaces": a_.spaces, "c": a_.c, "lam": LAM, "repeats": a_.repeats, "encoder_device": dev, "fit_device": "cpu"},
                 tag="smoke" if a_.repeats < 5 else "")
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    XN = o3.group_matrix(z["smiles"], gn).astype(float)
    L = S.ma_L; y_ma = (L > 0).astype(float); ma_folds = [S.ma_split(f) for f in range(5)]
    used = np.unique(np.r_[S.ma_a, S.ma_b])
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    rep.line(f"Ma mixtures {len(L)} (A dominates {y_ma.mean():.3f}); DREAM development pairs {len(P)}; GoodScents not used; "
             f"mixture-side numbers 27 (+1 Ma temperature)")

    from joblib import Parallel, delayed
    pool = Parallel(n_jobs=a_.jobs, backend="loky")

    def one(v, s, c, tr, te):
        torch.set_num_threads(1)
        return JointMa(v, Xd, A, B, s, c).fit(tr, y[tr]).predict(te)

    res = {"O1.2b (reference, word level)": evaluate.cross_validate(
        P, lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), Xd, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te), repeats=a_.repeats)}
    ma_rows, pairs_ = [], []
    for sp in a_.spaces:
        nets, fn = backbone(sp, dev)
        VN = [v.numpy() for v in vectors(nets, fn, S, used, len(z["smiles"]), dev)]
        rows_M = G.dream_rows(); VM = [v[rows_M].numpy() for v in vectors(nets, fn, G, rows_M, len(G.smiles_N), dev)]
        del nets; torch.cuda.empty_cache()
        nm = NAMES.get(sp, sp); D = VM[0].shape[1]
        rep.line(f"{nm} ({sp}): {len(VM)} members, vector size {D}")
        # Ma learnable on this backbone? 5 unseen-molecule folds, weights from Ma alone
        pred = {"0 (plain mean)": np.zeros(len(L)), "from Ma (training folds)": np.zeros(len(L))}; cnt = np.zeros(len(L))
        for tr, te in ma_folds:
            cnt[te] += 1
            for v in VN:
                s_tr = ma_stats(v, XN, S.ma_a[tr], S.ma_b[tr], y_ma[tr]); s_te = ma_stats(v, XN, S.ma_a[te], S.ma_b[te], y_ma[te])
                pred["0 (plain mean)"][te] += ma_score(s_te, torch.zeros(len(gn))).numpy() / len(VN)
                pred["from Ma (training folds)"][te] += ma_score(s_te, fit_ma_only(s_tr, len(gn))).numpy() / len(VN)
        rng = np.random.default_rng(0)
        for k, p_ in pred.items():
            p_ = p_ / cnt; bs = [roc_auc_score(y_ma[i], p_[i]) for i in (rng.integers(0, len(L), len(L)) for _ in range(2000)) if 0 < y_ma[i].mean() < 1]
            ma_rows.append([nm, k, f"{roc_auc_score(y_ma, p_):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"])
        s_all = [ma_stats(v, XN, S.ma_a, S.ma_b, y_ma) for v in VN]
        base = f"{nm} template, DREAM only (c = 0)"
        for c in [0.0] + list(a_.c):
            k = base if c == 0 else f"{nm} template, DREAM + Ma direction (c = {c:g})"
            f = lambda tr, te, c=c: np.mean(pool(delayed(one)(v, s, c, tr, te) for v, s in zip(VM, s_all)), 0)  # noqa: E731
            t0 = time.time(); res[k] = evaluate.cross_validate(P, f, repeats=a_.repeats); rep.line(f"{k}: done ({(time.time() - t0) / 60:.1f} min)")
            if c > 0:
                pairs_.append((k, base))
    ref = res["O1.2b (reference, word level)"]

    def sc_(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
    rep.section("Ma: which molecule dominates, 5 unseen-molecule folds (198 mixtures), AUROC of s [95% CI, mixtures resampled]")
    rep.table(["backbone", "group weights", "AUROC"], ma_rows)
    rep.section(f"DREAM development pairs ({a_.repeats} x 5), frozen hidden template: change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc_(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)
    rep.section("Paired steps (same folds): + Ma direction minus DREAM only, same backbone")
    rows = []
    for hi, lo in pairs_:
        for sub in SUB:
            s_ = sc_(res[hi], sub, res[lo])
            rows.append([hi, sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "RMSE change", "r change [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
