# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   W = words (138)
#   E = open-POM map (256)
# ---------------------------------------------------------------------------------------------
"""Which part of O1.2b's rule helps on pairs that share no molecule and have the same size (gap <= 1 molecule: the
kind of pair T46 / V50 contain)? (user, 2026-11-20: "is that because of the normalization or something? check clearly")
O1.2b = open-POM word profiles, each word put on the same scale (quantile per word over the 235 DREAM molecules),
mixture = group-weighted mean (24 RDKit group weights), distance = direction difference + size term (length), line.
Take one part away at a time, same folds:
  full O1.2b; no group weights; no size term; neither (= open-POM baseline); raw probabilities instead of the per-word
  scale (with / without group weights); open-POM's 256 map instead of words (with group weights).
All fitted on all development pairs per fold (one model for all pairs); subsets for reporting only. 50 x 5.

Env:     mist
Usage:   python evaluation/O3/eval_o12b_parts_samesize.py
Writes:  results/eval_o12b_parts_samesize/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402

REF = "O1.2b full (words on same scale + group weights + size term, 27)"


def main():
    rep = Report("eval_o12b_parts_samesize", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
    SUB = {"all pairs": None, "share none": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
           "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "share >= 1": k_P >= 1}
    q = mixture.quantile_per_word(op.probs_MxW); raw = np.asarray(op.probs_MxW, float); emb = np.asarray(op.emb_MxE, float)
    rows_m = {REF: (q, True, True),
              "no group weights (3)": (q, False, True),
              "no size term (26)": (q, True, False),
              "neither (= open-POM baseline, 2)": (q, False, False),
              "raw probabilities, no per-word scale (27)": (raw, True, True),
              "raw probabilities, no group weights (3)": (raw, False, True),
              "open-POM 256 map instead of words (27)": (emb, True, True)}
    res = {}
    for n, (M, g, ln) in rows_m.items():
        res[n] = evaluate.cross_validate(P, lambda a, b, M=M, g=g, ln=ln: O12b(M, X, A, B, g, ln, lam=LAM).fit(a, y[a]).predict(b))
        rep.line(f"done: {n}")
    ref = res[REF]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change from full O1.2b [95% CI]")
    rows = []
    for sub in SUB:
        for n, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", "" if r_ is ref else ch(s_)])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from full O1.2b"], rows)
    rep.save()


if __name__ == "__main__":
    main()
