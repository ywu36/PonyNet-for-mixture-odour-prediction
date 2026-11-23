# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = GoodScents + Ma molecules   M = DREAM molecules (235)   H = middle layer (400)
#   G = RDKit groups (24)   Q = GoodScents pairs   T = Ma mixtures   P = DREAM development pairs (514)   K = members
# ---------------------------------------------------------------------------------------------
"""O3-W joint (the user, 2026-11-19: "fit on goodscents and ma and pull towards dream?").

eval_o3w.py tested only one direction: group weights fitted on DREAM, pulled toward fixed GoodScents + Ma weights.
Here the SAME 24 group weights are fitted on all three at once, inside each DREAM training fold:
  loss = DREAM MSE (training pairs) + lam |group weights|^2 + c * var(y_train) * (GoodScents lean loss + Ma sign loss)
c = 0 is the plain template (27 fitted on DREAM). Larger c = GoodScents + Ma lead, DREAM pulls.
Merge only: frozen O2-C-Single (10 members), hidden template z = sum w_m v_m / sum w_m, w_m = exp(group weights . x_m),
compared by O1.2b's distance (no decoding). DREAM side: 24 group weights + length weight + slope + offset = 27;
the 2 temperatures belong to the GoodScents / Ma losses only.
GoodScents: all training lean pairs; Ma: all 198 mixtures (no DREAM labels involved).
Checks: DREAM 50 x 5 change from O1.2b and from c = 0 (paired, 4 subsets); GoodScents held-out lean AUROC of the
weights fitted with all 514 DREAM pairs (to see the GoodScents rule is kept).

Env:     mist
Usage:   python evaluation/O3/eval_o3w_joint.py [--c 0.3 1 3] [--jobs 10] [--smoke]
Writes:  results/eval_o3w_joint/<date_time>.{txt,json}
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
from eval_o3w import score  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o3t import scent_rel_labels  # noqa: E402


class Joint(O12b):
    """Template (O1.2b form on hidden vectors) whose group weights also answer GoodScents lean and Ma sign."""

    def __init__(self, v, x, A, B, aux, c, lam=LAM, dream=True, ma=True):
        super().__init__(v, x, A, B, True, True, lam=lam)
        self.c, self.dream, self.ma = c, dream, ma; a = {k: torch.as_tensor(np.asarray(t)) for k, t in aux.items()}
        # the two-molecule score needs only |v_a|^2, |v_b|^2, v_a.v_b per pair: precompute (exact, much faster)
        self.sets = []
        for ia, ib, yy in ((a["qa"], a["qb"], a["qy"]), (a["ta"], a["tb"], a["ty"])):
            va, vb = a["v"][ia], a["v"][ib]
            self.sets.append(((va * va).sum(1), (vb * vb).sum(1), (va * vb).sum(1), a["x"][ia], a["x"][ib], 2 * yy - 1))

    @staticmethod
    def fast_score(s, theta):
        aa, bb, ab, xa, xb, _ = s
        al = torch.sigmoid(xa @ theta - xb @ theta)                    # w_a / (w_a + w_b)
        zz = (al ** 2 * aa + 2 * al * (1 - al) * ab + (1 - al) ** 2 * bb).clamp_min(1e-12).sqrt()
        return (al * aa + (1 - al) * ab) / (zz * aa.sqrt()) - (al * ab + (1 - al) * bb) / (zz * bb.sqrt())

    def aux_loss(self, theta, lt):
        return sum(torch.nn.functional.softplus(-s[5] * self.fast_score(s, theta) / torch.exp(lt[j])).mean()
                   for j, s in enumerate(self.sets[:2 if self.ma else 1]))

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float)); vy = float(y.var())
        theta = torch.zeros(self.x.shape[1], requires_grad=True); k = torch.tensor(self.k0, requires_grad=True)
        lt = torch.full((2,), float(np.log(0.1)), requires_grad=True)
        with torch.no_grad():
            D = self._design(self.distance(theta, k, idx), idx)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone().requires_grad_(True)
        opt = torch.optim.LBFGS([lin, theta, k, lt], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.distance(theta, k, idx), idx) @ lin
            if not self.dream:                     # sanity: GoodScents / Ma only, as eval_o3w.fit_weights
                loss = self.lam * (theta ** 2).sum() + self.aux_loss(theta, lt) + 0 * pred.sum()
                loss.backward(); return loss
            loss = ((pred - y) ** 2).mean() + self.lam * (theta ** 2).sum()
            if self.c > 0:
                loss = loss + self.c * vy * self.aux_loss(theta, lt)
            loss.backward(); return loss
        opt.step(closure)
        self.theta, self.k, self.lin = theta.detach(), k.detach(), lin.detach()
        return self


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--c", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--jobs", type=int, default=10); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()
    REP = 1 if a_.smoke else evaluate.CV_REPEATS
    dev = pick_device()
    rep = Report("eval_o3w_joint", {"c": a_.c, "lam": LAM, "repeats": REP, "goodscents_pairs": "all train",
                                    "encoder_device": dev, "fit_device": "cpu"}, tag="smoke" if a_.smoke else "")
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    lab = scent_rel_labels(G, S)
    q_all = np.where((S.pair_split == 0) & ~np.isnan(lab))[0]
    q_tr = q_all
    q_te = np.where((S.pair_split == 1) & ~np.isnan(lab))[0]
    used = np.unique(np.r_[S.pair_a[q_tr], S.pair_b[q_tr], S.pair_a[q_te], S.pair_b[q_te], S.ma_a, S.ma_b])
    pos = {u: i for i, u in enumerate(used)}; re_ = np.vectorize(pos.get)
    XN = o3.group_matrix(z["smiles"][used], gn).astype(float)
    nets = o2.load_members(sorted((ROOT / "checkpoints/o2c").glob("member_*.pt")), dev)
    VN, VM = [], []
    for n in nets:
        v = np.zeros((len(used), 400))
        for i in range(0, len(used), 512):
            u = used[i:i + 512]; x, e, s, d, g = S.batch(u, dev)
            with torch.no_grad():
                v[i:i + len(u)] = n.latent(n.encode(x, e, s, d, g, int(g.max()) + 1))[0].double().cpu().numpy()
        VN.append(v); VM.append(n.latent(o2.encode_all(n, G, G.dream_rows(), dev))[0].double().cpu().numpy())
    del nets; torch.cuda.empty_cache()
    L = S.ma_L
    aux0 = dict(x=XN, qa=re_(S.pair_a[q_tr]), qb=re_(S.pair_b[q_tr]), qy=lab[q_tr], ta=re_(S.ma_a), tb=re_(S.ma_b), ty=(L > 0).astype(float))
    auxK = [dict(aux0, v=v) for v in VN]
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    rep.line(f"members {len(VM)}; molecules in GoodScents/Ma losses {len(used)}; GoodScents lean pairs: all {len(q_all)} train, "
             f"held-out {len(q_te)}; Ma mixtures {len(L)}; DREAM pairs {len(P)}; mixture-side numbers 27 (+2 temperatures of the extra losses)")

    from joblib import Parallel, delayed
    pool = Parallel(n_jobs=a_.jobs, backend="loky")

    def one(v, aux, c, tr, te):
        torch.set_num_threads(1)
        return Joint(v, Xd, A, B, aux, c).fit(tr, y[tr]).predict(te)

    t0 = time.time(); one(VM[0], auxK[0], max(a_.c), np.arange(400), np.arange(400, 514))
    rep.line(f"one joint fit (c = {max(a_.c)}): {time.time() - t0:.1f} s")

    def ens(c):
        return lambda tr, te: np.mean(pool(delayed(one)(v, ax, c, tr, te) for v, ax in zip(VM, auxK)), 0)
    name = lambda c: "template, 27 fitted on DREAM only (c = 0)" if c == 0 else f"template, joint DREAM + GoodScents + Ma (c = {c:g})"  # noqa: E731
    rows_def = {"O1.2b (reference, word level)": lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), Xd, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)}
    for c in [0.0] + list(a_.c):
        rows_def[name(c)] = ens(c)
    res = {}
    for k, f in rows_def.items():
        t0 = time.time(); res[k] = evaluate.cross_validate(P, f, repeats=REP); rep.line(f"{k}: done ({(time.time() - t0) / 60:.1f} min)")
    ref, base = res["O1.2b (reference, word level)"], res[name(0.0)]

    def sc_(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
    rep.section(f"DREAM development pairs ({REP} x 5), frozen hidden template: change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc_(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)
    rep.section("Paired steps (same folds): joint minus DREAM only (c = 0)")
    rows = []
    for c in a_.c:
        for sub in SUB:
            s_ = sc_(res[name(c)], sub, base)
            rows.append([f"c = {c:g}", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["joint weight", "subset", "RMSE change", "r change [95% CI]"], rows)

    # GoodScents rule kept? weights fitted with all 514 DREAM pairs + GoodScents + Ma, held-out lean pairs
    qa_te, qb_te = torch.as_tensor(re_(S.pair_a[q_te])), torch.as_tensor(re_(S.pair_b[q_te]))
    rows, gws = [], {}
    for c in [0.0] + list(a_.c):
        fits = pool(delayed(lambda v, ax: Joint(v, Xd, A, B, ax, c).fit(np.arange(len(P)), y).theta.numpy())(v, ax) for v, ax in zip(VM, auxK))
        d = np.mean([score(torch.as_tensor(v), torch.as_tensor(XN), qa_te, qb_te, torch.as_tensor(g)).numpy() for v, g in zip(VN, fits)], 0)
        gws[c] = np.mean(fits, 0); rows.append([name(c), f"{roc_auc_score(lab[q_te], d):.3f}"])
    rep.section("GoodScents lean, held-out pairs (both molecules unseen), AUROC; group weights fitted with all DREAM pairs "
                "(earlier: no weights 0.497, GoodScents only 0.588, GoodScents + Ma 0.479)")
    rep.table(["model", "AUROC"], rows)
    rep.section("Group weights (mean over members), fitted with all DREAM pairs")
    order = np.argsort(-np.abs(gws[max(a_.c)]))
    rep.table(["group"] + [f"c = {c:g}" for c in gws], [[gn[g]] + [f"{gws[c][g]:+.3f}" for c in gws] for g in order])
    rep.save()


if __name__ == "__main__":
    main()
