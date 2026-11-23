# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   G = RDKit groups (24, as in B1a+ groups)   L = GS-LF odour labels (138)   S = GS-LF molecules
#   M = DREAM molecules (235)   P = training pairs (820)   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""Group-level graph inside a mixture: do groups collaborate (allies dominate) or counteract?

The graph (fixed before running, learned outside DREAM):
  nodes  the 24 RDKit groups of B1a+ groups
  edges  sim(g, h), two versions:
         semantic   cosine of the groups' smell vectors v_g (ridge, GS-LF labels ~ group indicators,
                    penalty 300 = the inverse map's choice, 4,761 GS-LF molecules not in DREAM)
         structural Jaccard of the groups' carrier sets on the same GS-LF molecules (how often the
                    two groups sit on one molecule)
         sim(g, g) = 1 (the same group on another molecule is the closest ally)
In mixture A, molecule i, group g it carries: support(g) = sum_h sim(g, h) * share of the OTHER
molecules of A that carry h. N_i = mean of support over i's groups (0 if none, or if i is alone).
Model (B1a+ groups + size, one extra number gamma):
    log w_i = sum_g theta_g x_ig + gamma * N_i
gamma > 0: groups with similar allies in the mixture count more (collaboration);
gamma < 0: they crowd each other out and distinct groups stand out (counteraction).
theta penalised (lam 1e-3, as before); gamma unpenalised.
Test, same rule as eval_b1a_plus_setting1_claim.py: setting 1, change from B1a+ groups + size on the
same folds, claim only if r-change CI > 0 and within-study CI > 0; gamma on all 820 pairs with 95% CI
(100 mixture-resample refits) reported either way. The similarity matrices are saved for the Ma check.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_group_graph.py
Writes:  results/eval_b1a_plus_group_graph/<date_time>.{txt,json} (+ group_sim.json)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, gslf, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_b1a_plus_setting1_claim import within_study_r  # noqa: E402

LAM = 1e-3
RIDGE_V = 300.0
N_BOOT = 2000
N_BOOT_GAMMA = 100
MIN_STUDY_PAIRS = 10


class B1aPlusGraph(B1aPlus):
    """B1a+ with a per-pair, per-molecule log-weight term gamma * N (N_A, N_B: P x M)."""

    def __init__(self, q, x, A, B, S, lam, NA_PxM, NB_PxM):
        super().__init__(q, x, A, B, S, lam)
        self.NA = torch.tensor(NA_PxM); self.NB = torch.tensor(NB_PxM); self.K = self.x.shape[1]

    def distance(self, theta, idx):
        lw = self.x @ theta[:self.K]
        def mu(Mx, N):
            w = Mx[idx] * torch.exp(lw[None, :] + theta[self.K] * N[idx])
            m = w @ self.q / w.sum(1)[:, None]
            return m / m.norm(dim=1, keepdim=True)
        return (mu(self.A, self.NA) - mu(self.B, self.NB)).norm(dim=1)

    def fit(self, idx, y, w_pairs=None, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.tensor(np.asarray(y, float))
        wp = torch.ones(len(idx)) if w_pairs is None else torch.tensor(np.asarray(w_pairs, float))
        wp = wp / wp.sum()
        theta = torch.zeros(self.K + 1, requires_grad=True)
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


def gslf_groups(names):
    """GS-LF molecules not in DREAM -> (X_SxG for the 24 DREAM groups, Y_SxL labels, label words)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    fn = {n.removeprefix("fr_"): f for n, f in Descriptors.descList if n.startswith("fr_")}
    rep_name = [n.split(" (= ")[0] for n in names]                # merged units: the first (representative) group
    path = gslf.data.locate("pnas_repo", needs=gslf.FILE) / gslf.FILE
    rows = list(csv.DictReader(open(path, newline="")))
    words = [k for k in rows[0] if k not in ("nonStereoSMILES", "descriptors")]
    op = features.load("openpom")
    dream = {gslf._canonical(str(s)) for s in op.smiles_M}
    X, Y = [], []
    for r in rows:
        c = gslf._canonical(r["nonStereoSMILES"])
        if not c or c in dream:
            continue
        m = Chem.MolFromSmiles(c)
        X.append([fn[n](m) > 0 for n in rep_name]); Y.append([float(r[w]) for w in words])
    return np.array(X, float), np.array(Y, float), words


def support(sim_GxG, x_MxG, M_PxM):
    """N_PxM: for each pair-side mixture and member molecule, mean over its groups of
    sum_h sim(g, h) * share of the OTHER members carrying h."""
    cnt_PxG = M_PxM @ x_MxG
    n_P = M_PxM.sum(1)
    N = np.zeros_like(M_PxM)
    ng_M = x_MxG.sum(1)
    for p in range(len(M_PxM)):
        mem = np.where(M_PxM[p] > 0)[0]
        if len(mem) < 2:
            continue
        other_MxG = (cnt_PxG[p][None, :] - x_MxG[mem]) / (n_P[p] - 1)
        s = (x_MxG[mem] * (other_MxG @ sim_GxG.T)).sum(1)
        N[p, mem] = np.where(ng_M[mem] > 0, s / np.maximum(ng_M[mem], 1), 0.0)
    return N


def main():
    rep = Report("eval_b1a_plus_group_graph", {"lam": LAM, "ridge_v": RIDGE_V, "repeats": evaluate.CV_REPEATS,
                                               "n_boot": N_BOOT, "n_boot_gamma": N_BOOT_GAMMA})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    X_MxG, names = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    Xs_SxG, Ys_SxL, words = gslf_groups(names)
    V_GxL = Ridge(alpha=RIDGE_V).fit(Xs_SxG, Ys_SxL).coef_.T
    Vn = V_GxL / np.linalg.norm(V_GxL, axis=1, keepdims=True)
    sims = {"semantic": Vn @ Vn.T}
    inter = Xs_SxG.T @ Xs_SxG; tot = Xs_SxG.sum(0)
    sims["structural"] = inter / np.maximum(tot[:, None] + tot[None, :] - inter, 1)
    for k in sims:
        np.fill_diagonal(sims[k], 1.0)
    short = [n.split(" (= ")[0] for n in names]
    rep.section(f"0. The graph: 24 group nodes; edges from {len(Xs_SxG)} GS-LF molecules not in DREAM")
    rep.table(["group", "GS-LF carriers", "smell vector: top 3 words", "most similar groups, semantic (cos)", "most similar groups, structural (Jaccard)"],
              [[short[g], int(Xs_SxG[:, g].sum()), ", ".join(words[j] for j in np.argsort(-V_GxL[g])[:3]),
                ", ".join(f"{short[h]} {sims['semantic'][g, h]:+.2f}" for h in np.argsort(-sims["semantic"][g])[1:4]),
                ", ".join(f"{short[h]} {sims['structural'][g, h]:.2f}" for h in np.argsort(-sims["structural"][g])[1:4])] for g in range(len(names))])
    off = ~np.eye(len(names), dtype=bool)
    rep.line(f"off-diagonal edges: semantic mean {sims['semantic'][off].mean():+.2f} (range {sims['semantic'][off].min():+.2f} to {sims['semantic'][off].max():+.2f}); "
             f"structural mean {sims['structural'][off].mean():.2f}; correlation between the two {np.corrcoef(sims['semantic'][off], sims['structural'][off])[0, 1]:+.2f}")
    sim_path = rep.dir / "group_sim.json" if hasattr(rep, "dir") else ROOT / "results/eval_b1a_plus_group_graph/group_sim.json"
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"groups": names, "representative": short, "semantic": sims["semantic"].tolist(), "structural": sims["structural"].tolist(),
               "definition": "semantic = cosine of ridge(300) smell vectors on GS-LF non-DREAM; structural = Jaccard of GS-LF carriers; diag 1"},
              open(sim_path, "w"))

    N = {k: (support(s, X_MxG, A), support(s, X_MxG, B)) for k, s in sims.items()}
    for k, (na, nb) in N.items():
        v = np.concatenate([na[A > 0], nb[B > 0]])
        rep.line(f"support N ({k}) over molecule-in-mixture slots: mean {v.mean():+.3f}, sd {v.std():.3f}")

    def fp_base(tr, te):
        return B1aPlus(q, X_MxG, A, B, S, LAM).fit(tr, y[tr]).predict(te)

    def fp_graph(k):
        def f(tr, te):
            return B1aPlusGraph(q, X_MxG, A, B, S, LAM, *N[k]).fit(tr, y[tr]).predict(te)
        return f

    base = evaluate.cross_validate(train, fp_base); print("  done base", flush=True)
    res = {k: evaluate.cross_validate(train, fp_graph(k)) for k in sims}; print("  done graph", flush=True)
    s1 = base.no_overlap_N
    mixes = sorted({x for p in train for x in (p.a, p.b)}, key=sorted); idx = {x: i for i, x in enumerate(mixes)}
    allp = np.arange(len(train))
    ja = np.array([idx[p.a] for p in train]); jb = np.array([idx[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT_GAMMA)

    rep.section("1. Setting 1: + gamma (group graph) vs B1a+ groups + size, same folds")
    s0 = evaluate.score_cv(train, base, mask_N=s1, n_boot=N_BOOT)
    rows = [["B1a+ groups + size", s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", "", "", "", ""]]
    out = {}
    w = evaluate.cv_weights(base)[s1]; pairN = base.pair_N[s1]; study = base.study_N[s1]
    n_by = {st: len(np.unique(pairN[study == st])) for st in np.unique(study)}
    studies = [st for st, n in n_by.items() if n >= MIN_STUDY_PAIRS]
    keep = np.isin(study, studies)
    ia = np.array([idx[train[i].a] for i in pairN[keep]]); ib = np.array([idx[train[i].b] for i in pairN[keep]])
    for k in sims:
        s = evaluate.score_cv(train, res[k], reference=base, mask_N=s1, n_boot=N_BOOT)
        sa = evaluate.score_cv(train, res[k], reference=base, n_boot=N_BOOT)
        pb, pt, t = base.pred_N[s1][keep], res[k].pred_N[s1][keep], base.truth_N[s1][keep]
        rng = np.random.default_rng(0); d = []
        for _ in range(N_BOOT // 100):
            c = rng.multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=100)
            wb = w[keep] * c[:, ia] * c[:, ib]
            d.append(within_study_r(pt, t, wb, study[keep], studies) - within_study_r(pb, t, wb, study[keep], studies))
        d = np.concatenate(d); d = d[np.isfinite(d)]
        wd = within_study_r(pt, t, w[keep][None], study[keep], studies)[0] - within_study_r(pb, t, w[keep][None], study[keep], studies)[0]
        lo, hi = np.percentile(d, [2.5, 97.5])
        g_all = float(B1aPlusGraph(q, X_MxG, A, B, S, LAM, *N[k]).fit(allp, y).theta[-1])
        g_b = np.array([float(B1aPlusGraph(q, X_MxG, A, B, S, LAM, *N[k]).fit(allp, y, w_pairs=(cnt[b, ja] * cnt[b, jb]).astype(float) + 1e-12).theta[-1])
                        for b in range(N_BOOT_GAMMA)])
        glo, ghi = np.percentile(g_b, [2.5, 97.5])
        ok = s["r_change_ci"][0] > 0 and lo > 0
        rows.append([f"+ gamma, {k} graph", s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                     f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                     f"{wd:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{g_all:+.2f} [{glo:+.2f}, {ghi:+.2f}]", "MET" if ok else "not met"])
        out[k] = {"r_change": s["r_change"], "r_change_ci": list(s["r_change_ci"]), "rmse_change": s["rmse_change"],
                  "within": wd, "within_ci": [lo, hi], "gamma": g_all, "gamma_ci": [glo, ghi], "all_pairs_r_change": sa["r_change"],
                  "all_pairs_r_change_ci": list(sa["r_change_ci"]), "met": bool(ok)}
        print(f"  done {k}", flush=True)
    rep.table(["model", "pairs", "RMSE", "r [95% CI]", "change: RMSE [CI] / r [CI]", "within-study r change [CI]", "gamma, all 820 pairs [CI]", "rule"], rows)
    rep.line("gamma > 0 collaboration (allies dominate), < 0 counteraction. All pairs: " +
             "; ".join(f"{k} r change {v['all_pairs_r_change']:+.3f} [{v['all_pairs_r_change_ci'][0]:+.3f}, {v['all_pairs_r_change_ci'][1]:+.3f}]" for k, v in out.items()))
    rep.value("result", out)
    rep.save()


if __name__ == "__main__":
    main()
