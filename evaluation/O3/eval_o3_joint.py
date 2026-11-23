# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   N = GoodScents + Ma molecules   W = words (138)
#   G = RDKit groups (24)   P = development pairs (514)   Q = GoodScents pairs   T = Ma mixtures
# ---------------------------------------------------------------------------------------------
"""O3.1j (design/o3_design.md, "joint fit"): the mixing mechanism is shared across datasets and fitted in one fit
(the user: use the data well; the "ImageNet moment" = learn the mechanism from the big real dataset, transfer it with
few parameters). Per DREAM CV fold: theta (24) + s, r (24 + 24) fitted jointly on the fold's DREAM training pairs AND all
GoodScents train pairs; own heads DREAM beta + line, GoodScents temperature. Ma not used (independent check).
Rows (change from O1.2b, paired, development pairs, 50 x 5): O3.0, O3.1 (s, r fitted on GoodScents alone, fixed),
O3.1j theta shared only, O3.1j theta + s, r shared. Paired: O3.1j vs O3.1, vs O3.0. Shuffle control (5 permutations of
the DREAM molecules' profile + groups, 10 x 5). Checks of a joint fit on all development pairs: GoodScents held-out
pairs (both molecules unseen) and Ma (no refit). Init: the GoodScents-only fit (features/o3_scent_fit.npz, step 1).

Env:     mist (GPU)
Usage:   python evaluation/O3/eval_o3_joint.py
Writes:  results/eval_o3_joint/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o2j import word_auroc  # noqa: E402

N_SHUFFLE, SHUFFLE_REPEATS = 5, 10


def main():
    dev = pick_device(); rep = Report("eval_o3_joint", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "shuffles": N_SHUFFLE,
                                                        "shuffle_repeats": SHUFFLE_REPEATS, "device": dev})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    cnets, _, _, cprobs = members(G, "checkpoints/o2c", dev)
    q_op, q_c = mixture.quantile_per_word(op.probs_MxW), mixture.quantile_per_word(cprobs)
    S = o2j.Store(features.features_dir() / "o2j_data.npz"); z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    smiles, good = z["smiles"], np.where(z["ok"].astype(bool))[0]
    P_N = np.full((len(smiles), 138), 0.5); P_N[good] = np.mean([o2.predict_probs(n, S, good, dev) for n in cnets], 0)
    X_N = o3.group_matrix(smiles, gn)
    tr_q, te_q = np.where(S.pair_split == 0)[0], np.where(S.pair_split == 1)[0]
    scent = {"P": P_N, "X": X_N, "a": S.pair_a[tr_q], "b": S.pair_b[tr_q], "Y": S.pair_Y[tr_q], "targets": S.targets}
    fz = np.load(features.features_dir() / "o3_scent_fit.npz", allow_pickle=False)
    assert list(fz["groups"]) == list(gn)
    init = {k: fz[k] for k in ("theta", "s", "r")}; init["tau"] = float(fz["tau"])
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": lambda n: k_P[n] == 0, "share 1-2 molecules": lambda n: (k_P[n] >= 1) & (k_P[n] <= 2),
           "share >= 3 molecules": lambda n: k_P[n] >= 3}
    rep.line(f"development pairs {len(P)}; GoodScents train pairs {len(tr_q)} (held-out {len(te_q)}, both molecules unseen); "
             f"shared parameters: theta {len(gn)} (+ s, r {2 * len(gn)}); DREAM own: beta + line; GoodScents own: temperature")

    def joint(q, x, sup):
        return lambda tr, te: o3.O3J(q, x, A, B, scent, LAM, dev, sup, init if sup else {"theta": init["theta"], "tau": init["tau"]}).fit(tr, y[tr]).predict(te)

    rows_def = {"O1.2b (reference)": lambda tr, te: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
                "O3.0": lambda tr, te: o3.O3(q_c, X, A, B, True, LAM).fit(tr, y[tr]).predict(te),
                "O3.1 (s, r from GoodScents alone, fixed)": lambda tr, te: o3.O3(q_c, X, A, B, True, LAM, fz["s"], fz["r"]).fit(tr, y[tr]).predict(te),
                "O3.1j theta shared (DREAM + GoodScents)": joint(q_c, X, False),
                "O3.1j theta + s, r shared (DREAM + GoodScents)": joint(q_c, X, True)}
    res = {}
    for k, f in rows_def.items():
        res[k] = evaluate.cross_validate(P, f); rep.line(f"{k}: done")
    ref = res["O1.2b (reference)"]

    def mask(sub, r_):
        f = SUB[sub]
        return None if f is None else f(r_.pair_N)

    rep.section(f"DREAM development pairs, standard CV ({evaluate.CV_REPEATS} x 5): change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for name, r_ in res.items():
            s_ = evaluate.score_cv(P, r_, reference=None if r_ is ref else ref, mask_N=mask(sub, r_))
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([name, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)

    rep.section("Paired steps (same folds): first minus second")
    rows = []
    for hi, lo in (("O3.1j theta + s, r shared (DREAM + GoodScents)", "O3.1 (s, r from GoodScents alone, fixed)"),
                   ("O3.1j theta + s, r shared (DREAM + GoodScents)", "O3.0"),
                   ("O3.1j theta shared (DREAM + GoodScents)", "O3.0"),
                   ("O3.1j theta + s, r shared (DREAM + GoodScents)", "O3.1j theta shared (DREAM + GoodScents)")):
        for sub in SUB:
            s_ = evaluate.score_cv(P, res[hi], reference=res[lo], mask_N=mask(sub, res[hi]))
            rows.append([f"{hi}  minus  {lo}", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["comparison", "subset", "RMSE change", "r change [95% CI]"], rows)

    rep.section("Structure-shuffle control: O3.1j (theta + s, r) minus O3.0, DREAM molecules' profile + groups permuted")
    rows = []
    for kk in range(N_SHUFFLE):
        perm = np.random.default_rng(100 + kk).permutation(len(ids)); qs, xs = q_c[perm], X[perm]
        r_hi = evaluate.cross_validate(P, joint(qs, xs, True), repeats=SHUFFLE_REPEATS)
        r_lo = evaluate.cross_validate(P, lambda tr, te: o3.O3(qs, xs, A, B, True, LAM).fit(tr, y[tr]).predict(te), repeats=SHUFFLE_REPEATS)
        for sub in SUB:
            s_ = evaluate.score_cv(P, r_hi, reference=r_lo, mask_N=mask(sub, r_hi))
            rows.append([f"shuffle {kk + 1}", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["structures", "subset", "RMSE change", "r change [95% CI]"], rows)

    rep.section("Joint fit on all development pairs + GoodScents train: checks on data never used in the fit")
    allp = np.arange(len(P)); rows = []
    for name, sup in (("theta shared", False), ("theta + s, r shared", True)):
        m = o3.O3J(q_c, X, A, B, scent, LAM, dev, sup, init if sup else {"theta": init["theta"], "tau": init["tau"]}).fit(allp, y)
        fit = {"theta": m.theta.cpu(), "s": (m.s if sup else torch.zeros_like(m.theta)).cpu(), "r": (m.r if sup else torch.zeros_like(m.theta)).cpu(), "tau": m.tau.cpu()}
        l_, lg = o3.scent_loss(fit, P_N, X_N, S.pair_a[te_q], S.pair_b[te_q], S.pair_Y[te_q], S.targets)
        auc, _ = word_auroc(lg, S.pair_Y[te_q], S.targets)
        d = o3.ma_dominance(fit, X_N, S.ma_a, S.ma_b)
        rows.append([name, f"{l_.mean():.4f}", f"{auc:.4f}", f"{np.corrcoef(d, S.ma_L)[0, 1]:+.3f}" if np.std(d) > 0 else "n/a", f"{float(m.k) ** 2:.3f}"])
    rep.table(["joint fit", "GoodScents held-out rank loss", "GoodScents held-out word AUROC", "Ma r (dominance vs L, no refit)", "beta"], rows)
    rep.save()


if __name__ == "__main__":
    main()
