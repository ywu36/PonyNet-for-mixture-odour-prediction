# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""O5 prior + DREAM correction (69) on O5 networks with counts and/or POM order (user, 2026-11-20: "use O5 prior +
DREAM but with count and POM, try different variations"). Counts won share 1-2 on CV but lost new pairs (finding 6
section 15); does atom attention + the O5-prior readout keep the share 1-2 gain without the share-none loss?
Networks (single molecules only, frozen; train_o3p.py, 3 seeds):
  O5                         checkpoints/o3m_attpool        (train_o3m --config attpool)
  O5 + counts                checkpoints/ga_w64_att         (train_o3p --config att --groups)
  O5 + counts + POM 0.3      checkpoints/ga_w64_att_rank03  (train_o3p --config att_rank03 --groups)
  O5 + POM 0.3               checkpoints/o3p_att_rank03     (train_o3p --config att_rank03)
Readout for each (eval_atom_integrate.AtomReadout with prior): unit merge; atom score = c x the network's own learned
atom score (standardised) + q . atom vector, q pulled to 0 (penalty 1e-3); soft max over atoms (tau); odd-one-out; line.
Rows fixed before running: O1.2b (27); plain O3 + atom soft max (68, current best); the 4 networks above with the
prior readout (69 each) and with its own strength only (4 each; added before running at the fork's request).
Non-finite predictions masked in every row; CV predictions pickled after each model.
CV: 514 development pairs, 50 x 5, all subsets incl. share 1 and share 2; --heldout: fit on 820 training pairs, T46, V50.

Env:     mist
Usage:   python evaluation/O3/eval_o5_variants.py [--heldout] [--from-pkl <file>]
Writes:  results/eval_o5_variants/<date_time>[_heldout].{txt,json} (+ _cv.pkl)
"""
import argparse
import pickle
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
from mixdist.report import Report, results_dir  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_unit_merge import LAM  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_atom_attn import atom_vectors  # noqa: E402
from eval_attpool import attpool_atoms, LearnedStrength, REF, CUR, SEEDS  # noqa: E402
from eval_atom_integrate import AtomReadout, logmeanexp  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

NETS = {"O5": ("o3m_attpool", False), "O5 + counts": ("ga_w64_att", True),
        "O5 + counts + POM 0.3": ("ga_w64_att_rank03", True), "O5 + POM 0.3": ("o3p_att_rank03", False)}


def backbones(dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = {"plain": []}
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        H, mol = atom_vectors(n, G, rows_D, dev); H = H / scale
        V = np.zeros((len(rows_D), H.shape[1])); np.add.at(V, mol, H); V /= np.bincount(mol)[:, None]
        out["plain"].append((V, H, mol))
    for k, (folder, groups) in NETS.items():
        g = Gg if groups else G; out[k] = []
        for s in SEEDS:
            n = train_o3m.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                H, sc, mol, V = attpool_atoms(n, g, rows_D, dev)
            st = logmeanexp(torch.as_tensor(sc), torch.as_tensor(mol), len(V)).numpy()
            out[k].append((V / scale, H / scale, mol, sc, st))
    return out


def models(BB, q_op, X, A, B):
    def ens(k, make):
        return lambda tr, te, y: np.mean([make(*b).fit(tr, y[tr]).predict(te) for b in BB[k]], 0)
    d = {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
         CUR: ens("plain", lambda V, H, mol: UnitAtom(V, H, mol, A, B))}
    for k in NETS:
        d[f"{k}: own strength (4)"] = ens(k, lambda V, H, mol, sc, st: LearnedStrength(V, st, A, B, True))
        d[f"{k}: prior + DREAM (69)"] = ens(k, lambda V, H, mol, sc, st: AtomReadout(V, H, mol, A, B, prior=sc))
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); ap.add_argument("--from-pkl", default="")
    a = ap.parse_args(); ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_o5_variants", {"seeds": SEEDS, "lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev,
                                      "nets": NETS}, tag="heldout" if a.heldout else "")
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]; q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    BB = None if a.from_pkl else backbones(dev)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
               "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        if a.from_pkl:
            res = pickle.load(open(a.from_pkl, "rb")); rep.line(f"predictions from {a.from_pkl}")
        else:
            res = {}; pk = results_dir() / "eval_o5_variants" / f"{rep.time:%Y-%m-%d_%H%M%S}_cv.pkl"; pk.parent.mkdir(parents=True, exist_ok=True)
            for nm, fp in models(BB, q_op, X, A, B).items():
                res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")
                pickle.dump(res, open(pk, "wb"))
            rep.line(f"predictions saved to {pk}")
        fin_N = np.all([np.isfinite(r_.pred_N) for r_ in res.values()], 0)
        for nm, r_ in res.items():
            rep.line(f"non-finite predictions: {nm}: {(~np.isfinite(r_.pred_N)).sum()} of {len(r_.pred_N)}")
        rep.line(f"masked in every row: {(~fin_N).sum()} of {len(fin_N)}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=fin_N if m is None else fin_N & m[r_.pair_N])
        o5 = "O5: prior + DREAM (69)"
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), "" if nm in (REF, CUR, o5) else ch(sc(r_, sub, res[o5]))])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from O5 prior + DREAM"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                ok = np.isfinite(p); s = metrics.score(p[ok], y[ok], ceiling=hs.ceiling); c = ""
                if nm != REF and ok.all():
                    cr, ce = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p[ok].mean() - y[ok].mean():+.3f}",
                             c + ("" if ok.all() else f" ({(~ok).sum()} non-finite)")])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE [CI] / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
