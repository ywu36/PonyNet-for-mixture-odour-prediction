# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = DREAM development pairs
# ---------------------------------------------------------------------------------------------
"""Group-distance counts + hidden-layer attention: no group weights anywhere (user go-ahead, 2026-11-19).

Backbones (frozen, trained on GS-LF single molecules only): width 64 without counts (checkpoints/o3m_w64) and with
the group-distance counts in the atom input (checkpoints/ga_w64), seeds 0-2.
Mixing (fitted on DREAM training folds, angle only, as eval_context_dream.py --direction-only):
    share_m ~ exp(q . s_m + kappa cos(v_m, mean of the other molecules' v));  z = sum share_m v_m;  d = |u_A - u_B|
    q (64) + kappa + line (2) = 67 numbers.   s = v / mean |v| over GS-LF training molecules (same rule for both).
Third backbone (added before the run): Edge Transformer (checkpoints/bb_et, training/train_backbones.py --arch et),
no counts, same atom input; paired against message passing without counts.
Also: O1.2b's rule (27) on each backbone's quantile word profiles, for reference.
Paired, development pairs, 50 x 5; subsets: all, setting 1 (share none), setting 2 (share >= 1), share 1, share 2,
share >= 3; change from O1.2b and with minus without counts (same mixing rule).

Env:     mist
Usage:   python evaluation/O3/eval_context_counts.py [--set counts|pomorder|pomweight]   (pomorder: o3m_w64 vs o3p_rank;
         pomweight: ga_w64 vs ga_w64_rank vs ga_w64_rank03, paired against ga_w64)
Writes:  results/eval_context_counts/<date_time>.{txt,json,pkl}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report, results_dir  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_backbones  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
REF = "O1.2b (reference, 27 numbers)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", choices=["counts", "pomorder", "pomweight"], default="counts"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_context_counts", {"seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS,
                                                               "direction_only": True, "set": a.set, "device": dev},
                                      tag="" if a.set == "counts" else a.set)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr, te, names = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    nets = {"no counts": ([train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G),
            "with counts": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg),
            "edge transformer": ([train_backbones.load(ROOT / f"checkpoints/bb_et/member_{s}.pt", dev) for s in SEEDS], G)}
    if a.set == "pomweight":  # user, 2026-11-20: POM-order weight 1.0 vs 0.3 on the counts backbone
        nets = {"with counts": nets["with counts"],
                "counts + POM order 1.0": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64_rank/member_{s}.pt", dev) for s in SEEDS], Gg),
                "counts + POM order 0.3": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64_rank03/member_{s}.pt", dev) for s in SEEDS], Gg)}
    if a.set == "pomorder":   # user, 2026-11-19: attention on the POM-order backbone (no counts; training/train_o3p.py --config rank)
        nets = {"no counts": nets["no counts"],
                "POM order": ([train_o3m.load(ROOT / f"checkpoints/o3p_rank/member_{s}.pt", dev) for s in SEEDS], G)}
    V, Q = {}, {}
    for k, (ns, g) in nets.items():
        V[k] = []
        for n in ns:
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                V[k].append(n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy() / scale)
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, g, rows_D, dev) for n in ns], 0).astype(float))

    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
           "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
    rep.line(f"development pairs {len(P)}; " + ", ".join(f"{s}: {len(P) if m is None else int(m.sum())}" for s, m in SUB.items()))

    res = {REF: evaluate.cross_validate(
        P, lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_))}
    for k in nets:
        res[f"O1.2b rule, {k} (27)"] = evaluate.cross_validate(
            P, lambda tr_, te_, q=Q[k]: O12b(q, X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_))
        rep.line(f"done: O1.2b rule, {k}")
        res[f"attention + kappa, {k} (67)"] = evaluate.cross_validate(
            P, lambda tr_, te_, Vs=V[k]: np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(tr_, y[tr_]).predict(te_)
                                                  for v in Vs], 0))
        rep.line(f"done: attention + kappa, {k}")
    (results_dir() / "eval_context_counts").mkdir(parents=True, exist_ok=True)
    with open(results_dir() / "eval_context_counts" / f"{rep.time:%Y-%m-%d_%H%M%S}_{a.set}_cv.pkl", "wb") as fh:
        pickle.dump({"res": res, "k_P": k_P}, fh)

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    out = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if k == REF else res[REF])
            base = (k.replace("counts + POM order 1.0", "with counts").replace("counts + POM order 0.3", "with counts") if a.set == "pomweight"
                    else k.replace("with counts", "no counts").replace("edge transformer", "no counts").replace("POM order", "no counts"))
            paired = ch(sc(r_, sub, res[base])) if base != k else ""
            out.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                        "" if k == REF else ch(s_), paired])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "minus message passing without counts (same rule)"], out)
    for k in nets:
        kap = [ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(np.arange(len(P)), y).kappa.item() for v in V[k]]
        rep.line(f"kappa, {k}, fitted on all development pairs, per member: {', '.join(f'{x:+.3f}' for x in kap)}")
    rep.save()


if __name__ == "__main__":
    main()
