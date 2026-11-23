# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   T = atoms
# ---------------------------------------------------------------------------------------------
"""Backbones TRAINED with unit length on (cosine word head: words read from the direction only), read out with unit
merge and unit merge + atom soft max (user, 2026-11-20: "have we trained with unit length on? try variations but keep
the + atom in"). Readout as eval_unit_atom.py; backbones frozen, GS-LF singles only, 3 seeds each:
  plain O3 (o3m_w64, linear head; reference) | o3p_cos | o3p_cos_rank (+ POM order 1.0) | o3p_cos_rank_sal (+ salience)
Rows (fixed before running): O1.2b; per backbone unit merge (67) and unit merge + atom soft max (68).
CV: development pairs 50 x 5, six subsets (CV predictions saved). --heldout: T46 then V50.

Env:     mist
Usage:   python evaluation/O3/eval_unit_atom_trained.py [--heldout]
Writes:  results/eval_unit_atom_trained/<date_time>.{txt,json} (+ _cv.pkl)
"""
import argparse
import pickle
import sys
import time
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
from eval_atom_attn import atom_vectors  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_unit_merge import UnitMerge  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

LAM = 1e-3
SEEDS = (0, 1, 2)
REF = "O1.2b (reference, 27)"
NETS = {"plain O3": "o3m_w64", "cos head": "o3p_cos", "cos head + POM order": "o3p_cos_rank",
        "cos head + POM order + salience": "o3p_cos_rank_sal"}


def backbones(dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = {}
    for k, folder in NETS.items():
        out[k] = []
        for s in SEEDS:
            n = train_o3m.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
            H, mol = atom_vectors(n, G, rows_D, dev); H = H / scale
            V = np.zeros((len(rows_D), H.shape[1])); np.add.at(V, mol, H); V /= np.bincount(mol)[:, None]
            out[k].append((V, H, mol))
    return G, out


def models(BB, q_op, X, A, B):
    def ens(BBk, make):
        return lambda tr, te, y: np.mean([make(V, H, mol).fit(tr, y[tr]).predict(te) for V, H, mol in BBk], 0)
    out = {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)}
    for k, BBk in BB.items():
        out[f"{k}, unit merge (67)"] = ens(BBk, lambda V, H, mol: UnitMerge(V, A, B, False))
        out[f"{k}, unit merge + atom soft max (68)"] = ens(BBk, lambda V, H, mol: UnitAtom(V, H, mol, A, B))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_unit_atom_trained", {"lam": LAM, "seeds": SEEDS, "nets": NETS, "heldout": a.heldout,
                                            "repeats": evaluate.CV_REPEATS, "device": dev}, tag="heldout" if a.heldout else "")
    G, BB = backbones(dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
        res = {}
        for nm, fp in models(BB, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y))
            rep.line(f"done: {nm}; non-finite predictions {int((~np.isfinite(res[nm].pred_N)).sum())}")
        pk = ROOT / "results/eval_unit_atom_trained" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_cv.pkl"
        pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"CV predictions saved: {pk.name}")

        def sc(r_, sub, against):
            m = SUB[sub]; m = np.isfinite(r_.pred_N) if m is None else (m[r_.pair_N] & np.isfinite(r_.pred_N))
            if against is not None:
                m = m & np.isfinite(against.pred_N)
            return evaluate.score_cv(P, r_, reference=against, mask_N=m)
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF]); pl = "plain O3, " + nm.split(", ", 1)[1] if nm != REF else nm
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[pl])) if nm != REF and not nm.startswith("plain O3") else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from plain O3, same readout"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
                if nm != REF:
                    cr, _ = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
