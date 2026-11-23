# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   W = words (138)
#   T = held-out single molecules (1008)
# ---------------------------------------------------------------------------------------------
"""Do per-atom group-distance counts help? Paired, same seeds (0, 1, 2), same held-out molecules, same pairs:
  open-POM layout:  checkpoints/o2c (members 0-2, no groups)   vs  checkpoints/ga_o2c (with groups)
  width 64:         checkpoints/o3m_w64 (no groups)            vs  checkpoints/ga_w64 (with groups)

1. Single molecules (GS-LF official held-out): AUROC per member and of the 3-member mean; with minus without, 95% CI by
   resampling held-out molecules.
2. DREAM development pairs, 50 x 5, 4 subsets, RMSE and r:
   a. O1.2b's form (27 numbers: its group weights, length weight, line) on each network's word profiles
      (3-member mean probabilities, quantile per word) -- does the new single-molecule network help O1.2b?
   b. hidden mean (3 numbers: length weight + line, no group weights), mean over members.
   Change from O1.2b (released open-POM profiles), and with minus without groups (same form).

Env:     mist
Usage:   python evaluation/O3/eval_groupatoms.py
Writes:  results/eval_groupatoms/<date_time>.{txt,json}
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
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)


def auroc_by_word(p_TxW, y_TxW):
    from sklearn.metrics import roc_auc_score
    ok = [w for w in range(y_TxW.shape[1]) if 0 < y_TxW[:, w].sum() < len(y_TxW)]
    return np.mean([roc_auc_score(y_TxW[:, w], p_TxW[:, w]) for w in ok])


def main():
    dev = pick_device(); rep = Report("eval_groupatoms", {"seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); Gg, tr, te, names = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 1000, replace=False)
    rep.line(f"groups {len(names)}; atom features {Gg.node.shape[1]}; held-out single molecules {len(te)}; DREAM molecules {len(rows_D)}")
    nets = {"open-POM layout, no groups": ([o2.load_members([ROOT / f"checkpoints/o2c/member_{s}.pt"], dev)[0] for s in SEEDS], G),
            "open-POM layout, with groups": ([train_groupatoms.load(ROOT / f"checkpoints/ga_o2c/member_{s}.pt", dev) for s in SEEDS], Gg),
            "width 64, no groups": ([train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G),
            "width 64, with groups": ([train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg)}
    pairs = [("open-POM layout, with groups", "open-POM layout, no groups"), ("width 64, with groups", "width 64, no groups")]

    probs_te, probs_D, V_D, single = {}, {}, {}, []
    for k, (ns, g) in nets.items():
        pt = [o2.predict_probs(n, g, te, dev) for n in ns]; probs_te[k] = np.mean(pt, 0)
        probs_D[k] = np.mean([o2.predict_probs(n, g, rows_D, dev) for n in ns], 0).astype(float)
        with torch.no_grad():
            V_D[k] = [n.latent(o2.encode_all(n, g, rows_D, dev))[0].double().cpu().numpy() for n in ns]
        a_te = [auroc_by_word(p, g.Y[te]) for p in pt]; a_tr = [auroc_by_word(o2.predict_probs(n, g, tr_s, dev), g.Y[tr_s]) for n in ns]
        single.append([k, sum(p.numel() for p in ns[0].parameters()), " ".join(f"{x:.4f}" for x in a_te),
                       f"{auroc_by_word(probs_te[k], g.Y[te]):.4f}", f"{np.mean(a_tr) - np.mean(a_te):+.4f}"])
    rep.section("1. Single molecules, official held-out (1008): AUROC (mean over 138 words; 0.5 chance)")
    rep.table(["network", "parameters", "held-out AUROC per seed (0, 1, 2)", "3-member mean held-out AUROC", "training minus held-out"], single)
    rng = np.random.default_rng(0); Y = G.Y[te]; rows = []
    for a_, b_ in pairs:
        d0 = auroc_by_word(probs_te[a_], Y) - auroc_by_word(probs_te[b_], Y); boot = []
        for _ in range(300):
            i = rng.integers(0, len(te), len(te)); boot.append(auroc_by_word(probs_te[a_][i], Y[i]) - auroc_by_word(probs_te[b_][i], Y[i]))
        rows.append([a_, "minus", b_, f"{d0:+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]"])
    rep.table(["network", "", "same layout, no groups", "3-member mean AUROC change [95% CI, molecules resampled]"], rows)

    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
           "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
    res = {"O1.2b (reference: released open-POM profiles, 27 numbers)": evaluate.cross_validate(
        P, lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_))}
    for k in nets:
        q = mixture.quantile_per_word(probs_D[k])
        res[f"O1.2b form on {k} (27)"] = evaluate.cross_validate(P, lambda tr_, te_, q=q: O12b(q, X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_))
        res[f"hidden mean, {k} (3)"] = evaluate.cross_validate(
            P, lambda tr_, te_, Vs=V_D[k]: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(tr_, y[tr_]).predict(te_) for v in Vs], 0))
        rep.line(f"DREAM done: {k}")
    ref = res["O1.2b (reference: released open-POM profiles, 27 numbers)"]
    import pickle  # CV predictions kept so later subsets need no rerun
    with open(ROOT / f"results/eval_groupatoms/cv_predictions_{rep.stamp if hasattr(rep, 'stamp') else 'latest'}.pkl", "wb") as fh:
        pickle.dump({"res": res, "k_P": k_P}, fh)

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"2. DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref); paired = ""
            if "with groups" in k:
                paired = ch(sc(r_, sub, res[k.replace("with groups", "no groups")]))
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if r_ is ref else ch(s_), paired])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "with minus without groups (same form)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
