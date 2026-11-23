# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   N = GoodScents + Ma molecules   W = words (138)
#   G = RDKit groups (24)   P = development pairs (514)   Q = GoodScents pairs   T = Ma mixtures
# ---------------------------------------------------------------------------------------------
"""O3 ladder, one step per run (design/o3_design.md). Development pairs only (CV.md rule 7); V50 / T46 not scored.

--step 0  O3.0 additive core = O1.2b's form on profiles decoded by the frozen O2-C-Single members (mean of 10):
          rows O1.2b (reference), O3.0 without group gain (length only), O3.0; structure-shuffle control of the group gain.
--step 1  O3.1 = O3.0 + rank-1 company suppression; s, r fitted on GoodScents train pairs (network frozen), fixed on
          DREAM (theta, beta, line refitted). Checks first: GoodScents held-out pairs (both molecules unseen) and Ma
          (no refit, r with L = log((IAmix/IA)/(IBmix/IB))), then DREAM CV vs O3.0 and the shuffle control.
Acceptance (fixed in the design): paired r change CI > 0 vs the previous step (all pairs or share none); the gain
disappears under structure shuffling; GoodScents / Ma unseen-molecule checks improve; single molecules unchanged
(network frozen).
Shuffle control: the 235 DREAM molecules' profiles and groups are permuted together (structure -> wrong molecule,
mixture compositions kept), 5 permutations, 10 x 5 folds; the step's paired gain is recomputed under each.

Env:     mist (GPU for decoding only)
Usage:   python evaluation/O3/eval_o3.py --step 0
Writes:  results/eval_o3/<date_time>.{txt,json}, features/o3_scent_fit.npz (step 1)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o2j import word_auroc  # noqa: E402

N_SHUFFLE, SHUFFLE_REPEATS = 5, 10


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, choices=[0, 1], required=True)
    a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_o3", {"step": a.step, "lam": LAM, "repeats": evaluate.CV_REPEATS,
                                                  "shuffles": N_SHUFFLE, "shuffle_repeats": SHUFFLE_REPEATS, "device": dev})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    cnets, _, _, cprobs = members(G, "checkpoints/o2c", dev)
    q_op, q_c = mixture.quantile_per_word(op.probs_MxW), mixture.quantile_per_word(cprobs)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": lambda n: k_P[n] == 0, "share 1-2 molecules": lambda n: (k_P[n] >= 1) & (k_P[n] <= 2),
           "share >= 3 molecules": lambda n: k_P[n] >= 3}
    rep.line(f"development pairs {len(P)}; groups {len(gn)}; O2-C members {len(cnets)}; pairs by shared molecules "
             f"0 {int((k_P == 0).sum())}, 1-2 {int(((k_P >= 1) & (k_P <= 2)).sum())}, >= 3 {int((k_P >= 3).sum())}")

    sr = None
    if a.step == 1:
        sr = scent_step(rep, dev, cnets, gn)

    def model(q, x, use_theta=True, s=None, r=None):
        return lambda tr, te: o3.O3(q, x, A, B, use_theta, LAM, s, r).fit(tr, y[tr]).predict(te)

    rows_def = {"O1.2b (reference)": lambda tr, te: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
                "O3.0 without group gain (length only, O2-C profiles)": model(q_c, X, False),
                "O3.0 (O1.2b form, O2-C profiles)": model(q_c, X)}
    if sr is not None:
        rows_def["O3.1 (O3.0 + rank-1 company suppression from GoodScents)"] = model(q_c, X, True, sr["s"], sr["r"])
    res = {k: evaluate.cross_validate(P, f) for k, f in rows_def.items()}
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

    names = list(res)
    step_pair = (names[2], names[1], "group gain (O3.0 minus O3.0 without group gain)") if a.step == 0 else \
                (names[3], names[2], "company suppression (O3.1 minus O3.0)")
    rep.section("Step gain (paired, same folds) and the structure-shuffle control")
    rows = []
    for sub in SUB:
        s_ = evaluate.score_cv(P, res[step_pair[0]], reference=res[step_pair[1]], mask_N=mask(sub, res[step_pair[0]]))
        rows.append([step_pair[2], "real structures", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    for k in range(N_SHUFFLE):
        perm = np.random.default_rng(100 + k).permutation(len(ids))      # structure (profile + groups) -> another molecule
        qs, xs = q_c[perm], X[perm]
        lo = model(qs, xs, False) if a.step == 0 else model(qs, xs)
        hi = model(qs, xs) if a.step == 0 else model(qs, xs, True, sr["s"], sr["r"])
        r_hi = evaluate.cross_validate(P, hi, repeats=SHUFFLE_REPEATS); r_lo = evaluate.cross_validate(P, lo, repeats=SHUFFLE_REPEATS)
        for sub in SUB:
            s_ = evaluate.score_cv(P, r_hi, reference=r_lo, mask_N=mask(sub, r_hi))
            rows.append([step_pair[2], f"shuffle {k + 1} ({SHUFFLE_REPEATS} x 5)", sub, f"{s_['rmse_change']:+.4f}",
                         f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["step", "structures", "subset", "RMSE change", "r change [95% CI]"], rows)
    rep.save()


def scent_step(rep, dev, cnets, gn):
    """Fit s, r on GoodScents train pairs with the frozen O2-C members; check held-out pairs and Ma. -> fit dict."""
    S = o2j.Store(features.features_dir() / "o2j_data.npz"); z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    smiles, ok = z["smiles"], z["ok"].astype(bool); good = np.where(ok)[0]
    P_N = np.full((len(smiles), 138), np.nan)
    P_N[good] = np.mean([o2.predict_probs(n, S, good, dev) for n in cnets], 0)
    X_N = o3.group_matrix(smiles, gn)
    tr, te = np.where(S.pair_split == 0)[0], np.where(S.pair_split == 1)[0]
    rep.section("GoodScents (frozen O2-C profiles; s, r fitted on train pairs only)")
    rep.line(f"pairs train {len(tr)}, held-out (both molecules unseen) {len(te)}; target words {len(S.targets)}")
    fits = {"equal weights (tau only)": o3.fit_scent(P_N, X_N * 0, S.pair_a[tr], S.pair_b[tr], S.pair_Y[tr], S.targets, False),
            "group gain theta": o3.fit_scent(P_N, X_N, S.pair_a[tr], S.pair_b[tr], S.pair_Y[tr], S.targets, False),
            "group gain + rank-1 suppression s, r": o3.fit_scent(P_N, X_N, S.pair_a[tr], S.pair_b[tr], S.pair_Y[tr], S.targets, True)}
    losses, rows = {}, []
    for name, f in fits.items():
        Xf = X_N * 0 if name.startswith("equal") else X_N
        l_, lg = o3.scent_loss(f, P_N, Xf, S.pair_a[te], S.pair_b[te], S.pair_Y[te], S.targets); losses[name] = l_
        auc, nw = word_auroc(lg, S.pair_Y[te], S.targets)
        rows.append([name, f"{l_.mean():.4f}", f"{auc:.4f} ({nw} words)"])
    rep.table(["model", "held-out rank loss (lower = better)", "held-out mean word AUROC"], rows)
    names = list(fits)
    for hi, lo in ((names[2], names[1]), (names[1], names[0])):
        m, l, h = boot_ci(losses[lo] - losses[hi])
        rep.line(f"held-out loss reduction, {hi} vs {lo}: {m:+.4f} [{l:+.4f}, {h:+.4f}] (bootstrap over pairs; > 0 = better)")
    gs_train = set(S.pair_a[tr]) | set(S.pair_b[tr]); can = {}
    from rdkit import Chem
    for i in range(len(smiles)):
        m_ = Chem.MolFromSmiles(str(smiles[i])); can[i] = Chem.MolToSmiles(m_) if m_ is not None else str(smiles[i])
    seen = {can[i] for i in gs_train}
    unseen_T = np.array([can[a_] not in seen and can[b_] not in seen for a_, b_ in zip(S.ma_a, S.ma_b)])
    rep.section("Ma 2021 (no refit): r between predicted dominance log w1 - log w2 and L")
    rows = []
    for name in names[1:]:
        d = o3.ma_dominance(fits[name], X_N, S.ma_a, S.ma_b)
        for sub, sel in (("all mixtures", np.ones(len(d), bool)), ("both molecules unseen in GoodScents train", unseen_T)):
            if sel.sum() > 3 and np.std(d[sel]) > 0:
                rng = np.random.default_rng(0); bs = []
                for _ in range(2000):
                    i = rng.integers(0, sel.sum(), sel.sum()); bs.append(np.corrcoef(d[sel][i], S.ma_L[sel][i])[0, 1])
                rows.append([name, sub, int(sel.sum()), f"{np.corrcoef(d[sel], S.ma_L[sel])[0, 1]:+.3f} [{np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}]"])
            else:
                rows.append([name, sub, int(sel.sum()), "n/a"])
    rep.table(["model", "mixtures", "n", "r [95% CI]"], rows)
    f = fits[names[2]]
    order = np.argsort(-np.abs(f["s"].numpy()))
    rep.section("Fitted s (suppresses) and r (susceptible) per group, by |s|")
    rep.table(["group", "theta (GoodScents)", "s", "r"], [[gn[g], f"{f['theta'][g]:+.3f}", f"{f['s'][g]:+.3f}", f"{f['r'][g]:+.3f}"] for g in order])
    np.savez(features.features_dir() / "o3_scent_fit.npz", groups=np.array(gn), **{k: v.numpy() for k, v in f.items()})
    return {k: v.numpy() for k, v in f.items()}


if __name__ == "__main__":
    main()
