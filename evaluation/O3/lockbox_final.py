# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   P = the 820 DREAM training pairs   N = lockbox pairs
# ---------------------------------------------------------------------------------------------
"""DREAM lockbox, scored ONCE (evaluation/CV.md rule 7; user go-ahead 2026-11-20, "both").
Every model fitted on the 514 development pairs only, scored on the 129 lockbox pairs (both mixtures never seen in
development; 24 share no molecule, 105 share >= 1). The 177 pairs with one lockbox mixture are used for neither.
Model list fixed 2026-11-20 before running (before the option-1 CV finished; no model added or dropped after):
  1 B1a++ open-POM (rule-7 reference)              2 O1.2b (27)                 3 O1.2b + overlap (28)
  4 plain O3, attention + kappa (67)               5 plain O3, attention + kappa + overlap (68)
  6 plain O3, attention + kappa + learned-width kernel (69)                     7 counts, attention + kappa + overlap (68)
Backbones frozen (GS-LF singles only), 3 seeds averaged. Subsets all / share none / setting 2 / share 1 / share 2 /
share >= 3. RMSE and r with 95% CI by resampling mixtures (as CV); paired change from O1.2b and from B1a++.
Note: rule 7 also asked that V50 / T46 wait for the lockbox; they were scored earlier at the user's request.

Env:     mist
Usage:   python evaluation/O3/lockbox_final.py
Writes:  results/lockbox_final/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_attn_penalty import vectors  # noqa: E402
from eval_kernel_hidden import AttnKernel  # noqa: E402
from eval_overlap import AttnOverlap, O12bOverlap, jaccard  # noqa: E402
import eval_context_dream as ecd  # noqa: E402

LAM = 1e-3
REF, B1PP = "O1.2b (27)", "B1a++ open-POM (rule-7 reference)"


def main():
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("lockbox_final", {"lam": LAM, "device": dev, "split": "cv.lockbox_split (k 4, seed 20261118)"})
    P = data.load_train(); y = evaluate.labels(P); split = cv.lockbox_split(P)
    tr, te = np.array(split.train), np.array(split.test)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]; q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); J = jaccard(P); S = size_features(P)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    rep.line(f"training pairs {len(P)}: development {len(tr)}, lockbox {len(te)} (share none {int((k_P[te] == 0).sum())}, "
             f"share >= 1 {int((k_P[te] >= 1).sum())}), neither {len(split.half)}")
    V_plain, V_counts = vectors("plain", dev), vectors("counts", dev)

    def ens(Vs, make):
        return np.mean([make(v).fit(tr, y[tr]).predict(te) for v in Vs], 0)
    preds = {B1PP: B1aPlus(q_op, X, A, B, S, LAM).fit(tr, y[tr]).predict(te),
             REF: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
             "O1.2b + overlap (28)": O12bOverlap(q_op, X, A, B, J, True).fit(tr, y[tr]).predict(te),
             "plain O3, attention + kappa (67)": ens(V_plain, lambda v: AttnOverlap(v, A, B, J, False)),
             "plain O3, attention + kappa + overlap (68)": ens(V_plain, lambda v: AttnOverlap(v, A, B, J, True)),
             "plain O3, attention + kappa + learned-width kernel (69)": ens(V_plain, lambda v: AttnKernel(v, A, B)),
             "counts, attention + kappa + overlap (68)": ens(V_counts, lambda v: AttnOverlap(v, A, B, J, True))}

    def as_res(p):
        return evaluate.CVResult(te, np.asarray(p, float), y[te], np.array([P[i].dataset for i in te]),
                                 np.array([P[i].jaccard == 0 for i in te]), 1, 1, cv.LOCKBOX_SEED)
    res = {nm: as_res(p) for nm, p in preds.items()}
    kt = k_P[te]
    SUB = {"all pairs": None, "share none": kt == 0, "setting 2 (share >= 1)": kt >= 1, "share 1": kt == 1, "share 2": kt == 2,
           "share >= 3": kt >= 3}

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rows, vals = [], {}
    for sub, m in SUB.items():
        for nm, r_ in res.items():
            s_ = evaluate.score_cv(P, r_, mask_N=m)
            c_o = "" if nm == REF else ch(evaluate.score_cv(P, r_, reference=res[REF], mask_N=m))
            c_b = "" if nm == B1PP else ch(evaluate.score_cv(P, r_, reference=res[B1PP], mask_N=m))
            rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", c_o, c_b])
            vals.setdefault(sub, {})[nm] = {"rmse": s_["rmse"], "r": s_["r"]}
    rep.section("DREAM lockbox (129 pairs, scored once): RMSE / r [95% CI, mixture resampling]; change = RMSE / r [CI]")
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from B1a++"], rows)
    rep.value("lockbox", vals); rep.save()


if __name__ == "__main__":
    main()
