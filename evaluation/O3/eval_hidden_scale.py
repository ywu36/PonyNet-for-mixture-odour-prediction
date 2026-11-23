# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   T = DREAM atoms   N = GS-LF training molecules
#                                       D = hidden size (64)
# ---------------------------------------------------------------------------------------------
"""Hidden-unit scaling from single molecules (2026-11-20). Question: does O1.2b win CV share none / share 1 because its
per-word scale (quantile -> normal per word) puts every word on the same footing, while our angle in the hidden layer
is dominated by a few high-variance hidden directions? Test: rescale the hidden layer with statistics of the GS-LF
training molecules only (no DREAM numbers added), then the same readouts:
  standardize: each hidden unit divided by its sd over training molecules;
  whiten:      Sigma^(-1/2) of the training molecules' hidden vectors (ZCA, ridge 1e-3 x mean variance).
The map is linear, so applied to atom vectors H and molecule vectors V alike (V = mean or weighted mean of H).
Rows (fixed before running), 3 seeds averaged:
  O1.2b (27); current best = plain O3 unit merge + atom soft max (68) raw / standardize / whiten;
  O5 own atom scores (4) raw / standardize / whiten.
CV: 514 development pairs, 50 x 5, all subsets; --heldout: fit on 820 training pairs, T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_hidden_scale.py [--heldout]
Writes:  results/eval_hidden_scale/<date_time>[_heldout].{txt,json}
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
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_unit_merge import LAM  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_attpool import backbones, LearnedStrength, REF, CUR, SEEDS  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

O5 = "O5: own atom scores + odd-one-out (4)"
FOLDER = {"plain": "o3m_w64", "attpool": "o3m_attpool"}


@torch.no_grad()
def maps(dev, rep):
    """Per backbone and seed: D x D maps (standardize, whiten) from all GS-LF training molecules' hidden vectors."""
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups(); out = {}
    for k, folder in FOLDER.items():
        out[k] = []
        for s in SEEDS:
            n = train_o3m.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
            L_NxD = n.latent(o2.encode_all(n, G, tr, dev))[0].double().cpu().numpy()
            L_NxD = L_NxD / np.linalg.norm(L_NxD, axis=1).mean()          # same scalar scale as backbones() (all molecules)
            C = np.cov(L_NxD.T); ev, U = np.linalg.eigh(C); ev_r = ev + 1e-3 * ev.mean()
            S = np.diag(1 / np.sqrt(np.diag(C))); Wz = U @ np.diag(ev_r ** -0.5) @ U.T
            top = ev[::-1] / ev.sum()
            rep.line(f"{k} seed {s}: {len(L_NxD)} training molecules; mean |centre| {np.linalg.norm(L_NxD.mean(0)):.3f}; "
                     f"unit sd range {np.sqrt(np.diag(C)).min():.3f}-{np.sqrt(np.diag(C)).max():.3f}; "
                     f"variance share of top 1/5/10 directions {top[0]:.2f}/{top[:5].sum():.2f}/{top[:10].sum():.2f}")
            out[k].append({"standardize": S, "whiten": Wz})
    return out


def models(BB, MP, q_op, X, A, B):
    def ens(k, kind, make):
        def f(tr, te, y):
            ps = []
            for (V, H, mol, st), m in zip(BB[k], MP[k]):
                T = np.eye(V.shape[1]) if kind == "raw" else m[kind]
                ps.append(make(V @ T, H @ T, mol, st).fit(tr, y[tr]).predict(te))
            return np.mean(ps, 0)
        return f
    d = {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)}
    for kind in ("raw", "standardize", "whiten"):
        d[CUR if kind == "raw" else f"current best, hidden {kind}d (68)"] = ens("plain", kind, lambda V, H, mol, st: UnitAtom(V, H, mol, A, B))
    for kind in ("raw", "standardize", "whiten"):
        d[O5 if kind == "raw" else f"O5, hidden {kind}d (4)"] = ens("attpool", kind, lambda V, H, mol, st: LearnedStrength(V, st, A, B, True))
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_hidden_scale", {"seeds": SEEDS, "lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    BB = backbones(dev); MP = maps(dev, rep); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
               "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "setting 2 (share >= 1)": k_P >= 1,
               "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        res = {}
        for nm, fp in models(BB, MP, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF]); base = CUR if nm.startswith("current best") else O5
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), "" if nm in (REF, CUR, O5) else ch(sc(r_, sub, res[base]))])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from same model unscaled"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, MP, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
                if nm != REF:
                    cr, ce = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE [CI] / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
