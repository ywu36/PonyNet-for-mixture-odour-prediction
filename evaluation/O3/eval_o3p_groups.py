# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Does the POM-order loss add anything on top of the group-distance counts? Paired, same seeds (0-2), same folds:
  checkpoints/ga_w64        node-enriched width 64, word labels only            (train_groupatoms.py --arch w64)
  checkpoints/ga_w64_rank   the same + POM-order loss during singles training    (train_o3p.py --config rank --groups)
Both frozen; no DREAM in network training. DREAM development pairs, 50 x 5, 4 subsets, RMSE and r:
  a. plain mean of hidden vectors (3 numbers: size weight + line), mean over members
  b. O1.2b's form on each network's word profiles (3-member mean probabilities, quantile per word; 27 numbers)
Change from O1.2b (released open-POM profiles), and POM order minus no POM order (same form).
Also: single-molecule AUROC and agreement with open-POM's map (held-out pairs), per network.

Env:     mist
Usage:   python evaluation/O3/eval_o3p_groups.py
Writes:  results/eval_o3p_groups/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3p  # noqa: E402

SEEDS = (0, 1, 2)
NETS = {"counts": "ga_w64", "counts + POM order": "ga_w64_rank"}


def main():
    dev = pick_device(); rep = Report("eval_o3p_groups", {"nets": NETS, "seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G, tr, te, names = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows()
    Pemb = train_o3p.pom_targets(G); mixes = train_o3p.dream_mixtures(G)
    probs_D, V_D, single = {}, {}, []
    for k, folder in NETS.items():
        ns = [train_groupatoms.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev) for s in SEEDS]
        probs_D[k] = np.mean([o2.predict_probs(n, G, rows_D, dev) for n in ns], 0).astype(float)
        with torch.no_grad():
            V_D[k] = [n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy() for n in ns]
        auc = [o2.mean_auroc(o2.predict_probs(n, G, te, dev), G.Y[te]) for n in ns]
        agr = [train_o3p.agreement(n, G, te, Pemb, mixes, dev)[0] for n in ns]
        single.append([k, folder, " ".join(f"{x:.4f}" for x in auc), f"{np.mean(agr):.3f}"])
    rep.section("Single molecules (official held-out) and agreement with open-POM's map")
    rep.table(["network", "folder", "held-out AUROC per seed", "POM distance Spearman (held-out pairs, mean)"], single)

    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share none": k_P == 0, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    ref_name = "O1.2b (reference, released open-POM profiles, 27)"
    res = {ref_name: evaluate.cross_validate(
        P, lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))}
    for k in NETS:
        res[f"{k}: plain mean (3)"] = evaluate.cross_validate(
            P, lambda a, b, Vs=V_D[k]: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(a, y[a]).predict(b) for v in Vs], 0))
        q = mixture.quantile_per_word(probs_D[k])
        res[f"{k}: O1.2b form (27)"] = evaluate.cross_validate(
            P, lambda a, b, q=q: O12b(q, X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))
        rep.line(f"DREAM done: {k}")
    ref = res[ref_name]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            paired = ch(sc(r_, sub, res[k.replace("counts + POM order", "counts")])) if k.startswith("counts + POM order") else ""
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if r_ is ref else ch(s_), paired])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "POM order minus without (same form)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
