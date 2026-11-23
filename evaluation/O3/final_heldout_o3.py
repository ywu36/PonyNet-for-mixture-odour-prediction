# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   P = pairs (training + held-out)   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Final held-out test of the O3 backbones with group-distance counts (user request, 2026-11-19: "try on the val and
testing dataset and see if it really works"). Same protocol as evaluation/O1/final_heldout.py: every model fitted
once on all training pairs, scored on T46 (first) and V50. Both sets contain only pairs sharing no molecule (setting 1).

Rows fixed before running (no other rows, no retries):
  B1a open-POM (open-POM baseline) | O1.2b (reference, 27)
  O3 w64, O1.2b's rule (27)             | O3 w64 + counts, O1.2b's rule (27)
  O3 w64, attention + kappa (67)        | O3 w64 + counts, attention + kappa (67)   (angle only, as eval_context_counts.py)
Backbones frozen, trained on GS-LF single molecules only (checkpoints/o3m_w64, checkpoints/ga_w64; seeds 0-2).
--all (user, 2026-11-20, "test all of them"): adds POM order without counts (o3p_rank), counts + POM order 1.0
(ga_w64_rank) and 0.3 (ga_w64_rank03), Edge Transformer (bb_et); each with O1.2b's rule and attention + kappa.
Metrics: RMSE and r with 95% bootstrap CI over pairs; paired bootstrap change from O1.2b (RMSE / r); bias.

Env:     mist
Usage:   python evaluation/O3/final_heldout_o3.py [--all]
Writes:  results/final_heldout_o3/<date_time>.{txt,json}
"""
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
import eval_context_dream as ecd  # noqa: E402
import train_backbones  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

LAM = 1e-3
SEEDS = (0, 1, 2)
ALL = False
REF = "O1.2b (reference, 27)"


def main():
    global ALL
    ALL = "--all" in sys.argv; ecd.DIRECTION_ONLY = True
    O4 = "--o4" in sys.argv   # user, 2026-11-20: O4 (larger encoder, O3's additive structure); rows fixed before running
    dev = pick_device(); rep = Report("final_heldout_o3", {"lam": LAM, "seeds": SEEDS, "all": ALL, "o4": O4, "device": dev},
                                      tag="all" if ALL else ("o4" if O4 else ""))
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr_m, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr_m, 2000, replace=False)
    nets = {"O3 w64": ([train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G),
            "O3 w64 + counts": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg)}
    if ALL:   # user, 2026-11-20: "test all of them" (rows fixed before running)
        nets.update({
            "POM order (no counts)": ([train_o3m.load(ROOT / f"checkpoints/o3p_rank/member_{s}.pt", dev) for s in SEEDS], G),
            "counts + POM order 1.0": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64_rank/member_{s}.pt", dev) for s in SEEDS], Gg),
            "counts + POM order 0.3": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64_rank03/member_{s}.pt", dev) for s in SEEDS], Gg),
            "Edge Transformer": ([train_backbones.load(ROOT / f"checkpoints/bb_et/member_{s}.pt", dev) for s in SEEDS], G)})
    if O4:
        nets.update({"O4 (3 members)": ([train_o3m.load(ROOT / f"checkpoints/o4/member_{s}.pt", dev) for s in SEEDS], G),
                     "O4 (10 members)": ([train_o3m.load(ROOT / f"checkpoints/o4/member_{s}.pt", dev) for s in range(10)], G)})
    V, Q = {}, {}
    for k, (ns, g) in nets.items():
        V[k] = []
        for n in ns:
            with torch.no_grad():
                scale = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                V[k].append(n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy() / scale)
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, g, rows_D, dev) for n in ns], 0).astype(float))

    train = data.load_train(); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
        y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
        lin = lambda d: evaluate.linear_calibration(d, y_all, np.zeros(len(allp)))(tr, te)  # noqa: E731
        preds = {"B1a open-POM (open-POM baseline)": lin(mixture.paper_distance(allp, op.cids_M, op.probs_MxW)),
                 REF: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y_all[tr]).predict(te)}
        for k in nets:
            preds[f"{k}, O1.2b's rule (27)"] = O12b(Q[k], X, A, B, True, True, lam=LAM).fit(tr, y_all[tr]).predict(te)
        for k in nets:
            preds[f"{k}, attention + kappa (67)"] = np.mean(
                [ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(tr, y_all[tr]).predict(te) for v in V[k]], 0)
        y = hs.y_P; ref = preds[REF]
        rep.section(f"{hs.name}: {len(y)} pairs, all sharing no molecule (setting 1); mean label {y.mean():.3f}"
                    + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
        rows = []
        for nm, p in preds.items():
            s = metrics.score(p, y, ceiling=hs.ceiling); ch = ""
            if nm != REF:
                cr, ce = paired(p, ref, y)
                ch = (f"{s['rmse'] - metrics.rmse(ref, y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / "
                      f"{s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]")
            rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", ch])
        rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE [CI] / r [CI]"], rows)
        rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
