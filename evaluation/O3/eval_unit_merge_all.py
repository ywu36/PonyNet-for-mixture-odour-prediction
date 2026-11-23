# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = pairs   M = DREAM molecules (235)   H = hidden layer   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Unit merge on every network we have (user, 2026-11-20: "a no-brainer post-training trick ... use it on all our
settings ... CV, held-out, everything"). Unit merge (other session, eval_unit_merge.UnitMerge): each molecule's hidden
vector scaled to length 1 before the attention-weighted mean, so the mixture's length = how much its molecules agree.
Per network (frozen, single molecules only, 3 members averaged), three attention readouts:
  raw merge, angle only (67)            the current readout
  unit merge, angle only (67)
  unit merge, angle + agreement (68)    + weight x (log length difference / its spread)^2
References: O1.2b (27); O1.2b with unit merge (27): each molecule's word profile scaled to length 1 before the
group-weighted mean (its length term then measures agreement).
Networks: plain O3 (o3m_w64), counts (ga_w64), counts + POM order 0.3 (ga_w64_rank03), counts + POM order 1.0
(ga_w64_rank), POM order without counts (o3p_rank), balanced word loss (o3p_balanced), O4 (o4, members 0-2);
cosine word head (o3p_cos: words read from the hidden vector's direction only, i.e. unit vectors learned in training).
CV: 514 development pairs, 50 x 5, subsets for reporting only (all; setting 1 = share none, same size, sizes differ;
setting 2 = share >= 1, share 1-2, share >= 3). --heldout: fit on all 820 training pairs, score T46 then V50.
No overlap term, no routing; lockbox not used.

Env:     mist
Usage:   python evaluation/O3/eval_unit_merge_all.py [--nets plain counts ...] [--heldout]
Writes:  results/eval_unit_merge_all/<date_time>[_<tag>].{txt,json}
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
from eval_unit_merge import UnitMerge, LAM  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
REF, REF_U = "O1.2b (reference, 27)", "O1.2b, unit merge (27)"
NETS = {"plain": ("plain O3", "o3m_w64", False), "counts": ("counts", "ga_w64", True),
        "counts_pom03": ("counts + POM order 0.3", "ga_w64_rank03", True), "counts_pom1": ("counts + POM order 1.0", "ga_w64_rank", True),
        "pom": ("POM order (no counts)", "o3p_rank", False), "balanced": ("balanced word loss", "o3p_balanced", False),
        "o4": ("O4", "o4", False),
        "cos": ("cosine word head (unit hidden vector learned in training)", "o3p_cos", False)}   # added 2026-11-20
RO = ("raw merge, angle only (67)", "unit merge, angle only (67)", "unit merge, angle + agreement (68)")


def vectors(keys, dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = {}
    for k in keys:
        label, folder, groups = NETS[k]; g = Gg if groups else G
        ld = train_groupatoms.load if folder == "ga_w64" else train_o3m.load
        Vs = []
        for s in SEEDS:
            n = ld(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                Vs.append(n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy() / scale)
        out[label] = Vs
    return out


def models(V, q_op, X, A, B):
    q_unit = q_op / np.linalg.norm(q_op, axis=1, keepdims=True)
    m = {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
         REF_U: lambda tr, te, y: O12b(q_unit, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)}
    for label, Vs in V.items():
        def ens(make, Vs=Vs):
            return lambda tr, te, y: np.mean([make(v).fit(tr, y[tr]).predict(te) for v in Vs], 0)
        m[f"{label}: {RO[0]}"] = ens(lambda v: ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True, lam=LAM))
        m[f"{label}: {RO[1]}"] = ens(lambda v: UnitMerge(v, A, B, False))
        m[f"{label}: {RO[2]}"] = ens(lambda v: UnitMerge(v, A, B, True))
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nets", nargs="+", default=list(NETS), choices=list(NETS))
    ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    tag = ("heldout_" if a.heldout else "") + "-".join(a.nets)
    rep = Report("eval_unit_merge_all", {"nets": a.nets, "seeds": SEEDS, "lam": LAM, "heldout": a.heldout,
                                         "repeats": evaluate.CV_REPEATS, "device": dev}, tag=tag)
    V = vectors(a.nets, dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
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
        for nm, fp in models(V, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                base = res[REF] if nm == REF_U else (res[nm.replace(RO[1], RO[0]).replace(RO[2], RO[0])] if (RO[1] in nm or RO[2] in nm) else None)
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), "" if base is None else ch(sc(r_, sub, base))])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "unit minus raw (same network)"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(V, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
                if nm != REF:
                    cr, ce = paired(p, ref, y)
                    c = (f"{s['rmse'] - metrics.rmse(ref, y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / "
                         f"{s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]")
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE [CI] / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
