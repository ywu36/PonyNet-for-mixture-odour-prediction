# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Unified model: one deterministic model for every pair, no routing (user, 2026-11-20). From clue C23: setting 2 is
mostly exact molecule overlap; the counts read it the long way round and cost setting 1. Here overlap enters directly:
    prediction = line(mixture distance) + c * J,   J = |A & B| / |A | B|  (Jaccard of the two molecule sets)
J is computed from the input compositions and is exactly 0 for every pair sharing no molecule (setting 1, T46, V50).
Rows (fixed before running), each with and without the overlap term (+1 number):
  O1.2b (27 -> 28; overlap added in a second least-squares step on the fold's training predictions)
  plain O3 w64 (o3m_w64), attention + kappa (67 -> 68; overlap fitted jointly)
  counts (ga_w64), attention + kappa
  counts + POM order 0.3 (ga_w64_rank03), attention + kappa
Development pairs, 50 x 5; subsets all / setting 1 / setting 2 / share 1 / share 2 / share >= 3; change from O1.2b
and "+ overlap minus without" on the same folds.
--heldout: T46 then V50 (fitted once on all training pairs), the same rows.

Env:     mist
Usage:   python evaluation/O3/eval_overlap.py [--heldout]
Writes:  results/eval_overlap/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
SEEDS = (0, 1, 2)
REF = "O1.2b (reference, 27)"


def jaccard(pairs):
    return np.array([len(set(p.a) & set(p.b)) / len(set(p.a) | set(p.b)) for p in pairs])


class AttnOverlap(ecd.ContextReadoutQ):
    """Attention + kappa (angle only) with the overlap term fitted jointly: pred = a d + b + c J."""

    def __init__(self, V, A, B, J, use_j):
        super().__init__(V, A, B, np.zeros(len(V)), True); self.J = torch.as_tensor(J, dtype=F64); self.use_j = use_j

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        b0 = torch.zeros((), dtype=F64)
        with torch.no_grad():
            self.lw = self.V @ q; d = self.distance(kappa, b0, idx)
            D = torch.stack([d, torch.ones_like(d)] + ([self.J[idx]] if self.use_j else []), 1)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True); opt = torch.optim.LBFGS([lin, q, kappa], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.V @ q; d = self.distance(kappa, b0, idx)
            pred = lin[0] * d + lin[1] + (lin[2] * self.J[idx] if self.use_j else 0)
            loss = ((pred - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.V @ q
        self.q, self.kappa, self.b, self.lin = q.detach(), kappa.detach(), b0, lin.detach(); self.lw = self.lw.detach()
        return self

    def predict(self, idx):
        idx_t = torch.as_tensor(np.asarray(idx))
        with torch.no_grad():
            d = self.distance(self.kappa, self.b, idx_t)
            return (self.lin[0] * d + self.lin[1] + (self.lin[2] * self.J[idx_t] if self.use_j else 0)).numpy()


class O12bOverlap:
    """O1.2b, then the overlap term in a second least-squares step: y ~ O1.2b(train) + J."""

    def __init__(self, q, X, A, B, J, use_j):
        self.m = O12b(q, X, A, B, True, True, lam=LAM); self.J, self.use_j = J, use_j

    def fit(self, tr, y):
        self.m.fit(tr, y)
        if self.use_j:
            D = np.c_[self.m.predict(tr), self.J[tr], np.ones(len(tr))]; self.w = np.linalg.lstsq(D, y, rcond=None)[0]
        return self

    def predict(self, te):
        p = self.m.predict(te)
        return np.c_[p, self.J[te], np.ones(len(te))] @ self.w if self.use_j else p


def backbones(dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    nets = {"plain O3": ([train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G),
            "counts": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg),
            "counts + POM order 0.3": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64_rank03/member_{s}.pt", dev) for s in SEEDS], Gg)}
    V = {}
    for k, (ns, g) in nets.items():
        V[k] = []
        for n in ns:
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                V[k].append(n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy() / scale)
    return G, V


def models(V, q_op, X, A, B, J):
    """name -> fit_predict(tr, te, y)."""
    out = {REF: lambda tr, te, y: O12bOverlap(q_op, X, A, B, J, False).fit(tr, y[tr]).predict(te),
           "O1.2b + overlap (28)": lambda tr, te, y: O12bOverlap(q_op, X, A, B, J, True).fit(tr, y[tr]).predict(te)}
    for k, Vs in V.items():
        for use_j in (False, True):
            nm = f"{k}, attention + kappa" + (" + overlap (68)" if use_j else " (67)")
            out[nm] = (lambda Vs_, uj: lambda tr, te, y: np.mean(
                [AttnOverlap(v, A, B, J, uj).fit(tr, y[tr]).predict(te) for v in Vs_], 0))(Vs, use_j)
    return out


def partner(nm):
    return nm.replace(" + overlap (68)", " (67)").replace("O1.2b + overlap (28)", REF) if "overlap" in nm else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_overlap", {"lam": LAM, "seeds": SEEDS, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    G, V = backbones(dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M); J = jaccard(P)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
        res = {}
        for nm, fp in models(V, q_op, X, A, B, J).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF]); pt = partner(nm)
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[pt])) if pt and pt != REF or nm == "O1.2b + overlap (28)" else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "+ overlap minus without (same folds)"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M); J = jaccard(allp)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(V, q_op, X, A, B, J).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (overlap 0 on all: {bool((J[te] == 0).all())})"
                        + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
                if nm != REF:
                    cr, ce = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
