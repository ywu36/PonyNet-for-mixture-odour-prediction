# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   W = words   P = pairs
# ---------------------------------------------------------------------------------------------
"""Is the overlap term a general fix? (user, 2026-11-20: "change the backbone to POM and MIST and see if it is a
general solution"). Same term as eval_overlap.py: + c * J, J = Jaccard of the two molecule sets (0 for setting 1).
Rows (fixed before running), each without / with overlap:
  B1a open-POM (open-POM baseline: line on the profile distance, 2 -> 3 numbers)
  B1a MIST (MIST baseline, 2 -> 3)
  B1a++ open-POM, B1a++ MIST (B1a + group weights + mixture size; user: "do all of them", 2026-11-20)
  O1.2b on open-POM word profiles (27 -> 28)       O1.2b on MIST word profiles (27 -> 28)
Overlap enters in a second least-squares step on the fold's training predictions (as O1.2b + overlap there).
Part 1: development pairs, 50 x 5, subsets all / setting 1 / setting 2 / share 1 / share 2 / share >= 3.
Part 2: T46 then V50 (all setting 1), fitted once on all training pairs.

Env:     mist
Usage:   python evaluation/O3/eval_overlap_general.py
Writes:  results/eval_overlap_general/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from eval_overlap import jaccard  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402

REF = "O1.2b open-POM (reference, 27)"


def with_overlap(base_fp, J, use_j):
    """base_fp(tr, te_or_tr, y) -> predictions; overlap added by least squares on training predictions."""
    def fp(tr, te, y):
        if not use_j:
            return base_fp(tr, te, y)
        p_tr, p_te = base_fp(tr, tr, y), base_fp(tr, te, y)
        w = np.linalg.lstsq(np.c_[p_tr, J[tr], np.ones(len(tr))], y[tr], rcond=None)[0]
        return np.c_[p_te, J[te], np.ones(len(te))] @ w
    return fp


def models(pairs, fs, J):
    out = {}
    for name, f in fs.items():
        A, B = membership(pairs, f.cids_M); q = mixture.quantile_per_word(f.probs_MxW)
        ids = [str(int(c)) for c in f.cids_M]
        X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, f.smiles_M)})), ids)
        d = mixture.paper_distance(pairs, f.cids_M, f.probs_MxW)
        b1a = (lambda d_: lambda tr, te, y: evaluate.linear_calibration(d_, y, np.zeros(len(d_)))(tr, te))(d)
        o12 = (lambda q_, X_, A_, B_: lambda tr, te, y: O12b(q_, X_, A_, B_, True, True, lam=LAM).fit(tr, y[tr]).predict(te))(q, X, A, B)
        S = size_features(pairs)
        b1pp = (lambda q_, X_, A_, B_, S_: lambda tr, te, y: B1aPlus(q_, X_, A_, B_, S_, LAM).fit(tr, y[tr]).predict(te))(q, X, A, B, S)
        for use_j in (False, True):
            out[f"B1a {name}" + (" + overlap (3)" if use_j else " (2)")] = with_overlap(b1a, J, use_j)
            out[f"B1a++ {name}" + (" + overlap (+1)" if use_j else "")] = with_overlap(b1pp, J, use_j)
            nm = (REF if name == "open-POM" else f"O1.2b {name} (27)") if not use_j else f"O1.2b {name} + overlap (28)"
            out[nm] = with_overlap(o12, J, use_j)
    return out


def partner(nm):
    if "overlap" not in nm:
        return None
    return REF if nm == "O1.2b open-POM + overlap (28)" else (nm.replace(" + overlap (3)", " (2)").replace(" + overlap (28)", " (27)")
                                                              .replace(" + overlap (+1)", ""))


def main():
    rep = Report("eval_overlap_general", {"lam": LAM, "repeats": evaluate.CV_REPEATS})
    fs = {"open-POM": features.load("openpom"), "MIST": features.load("mist")}

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    P = dev_pairs(); y = evaluate.labels(P); J = jaccard(P); k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
           "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
    res = {nm: evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)) for nm, fp in models(P, fs, J).items()}

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
    rep.section(f"1. DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
    rows = []
    for sub in SUB:
        for nm, r_ in res.items():
            s_ = sc(r_, sub, None if nm == REF else res[REF]); pt = partner(nm)
            rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if nm == REF else ch(s_), ch(sc(r_, sub, res[pt])) if pt else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b open-POM", "+ overlap minus without"], rows)

    train = data.load_train()
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
        y_all = np.r_[evaluate.labels(train), hs.y_P]; Ja = jaccard(allp)
        preds = {nm: fp(tr, te, y_all) for nm, fp in models(allp, fs, Ja).items()}
        yy = hs.y_P; ref = preds[REF]; rows = []
        rep.section(f"2. {hs.name}: {len(yy)} pairs (overlap 0 on all: {bool((Ja[te] == 0).all())})"
                    + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
        for nm, p in preds.items():
            s = metrics.score(p, yy, ceiling=hs.ceiling); c = ""
            if nm != REF:
                cr, _ = paired(p, ref, yy)
                c = f"{s['rmse'] - metrics.rmse(ref, yy):+.4f} / {s['r'] - metrics.pearson(ref, yy):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
            rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - yy.mean():+.3f}", c])
        rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b open-POM: RMSE / r [CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
