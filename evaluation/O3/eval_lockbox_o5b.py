# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Item 4 of "do all of them" (user 2026-11-20): lockbox score of (b). Second use of the DREAM lockbox (first: finding 5,
"Learned-width hidden kernel and the DREAM lockbox"), authorised by the user. CV.md rule 7: list fixed beforehand,
fitted on the 514 development pairs only, scored once on the 129 lockbox pairs; no candidate changed after seeing it.
Candidates (fixed before running; count-dropout networks of item 1 deliberately not added, their results came later):
  O1.2b (reference, 27)
  plain O3: unit merge + atom soft max fitted on DREAM (68)         eval_o5_softmax CUR
  O5: own strength + odd-one-out (4)                                eval_o5_softmax O4
  (b) O5 + atom query shrunk toward own attention (69)              eval_o5_softmax OB (packaged model)
  (b) + size loss (69)                                              eval_size_robust SizeRobust(size_loss=True)
Subsets: all; share 0; share 0 same size; share 1-2; share >= 3 (shared-molecule count between the two mixtures).

Env:     mist
Usage:   python evaluation/O3/eval_lockbox_o5b.py
Writes:  results/eval_lockbox_o5b/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, data, evaluate, features, metrics, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from eval_size_robust import SizeRobust  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402


def main():
    ecd.DIRECTION_ONLY = True; dev = pick_device(); rep = Report("eval_lockbox_o5b", {"lam": eo.LAM, "device": dev})
    _, BB = eo.backbones(dev, rep); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    train = data.load_train(); y = evaluate.labels(train); A, B = membership(train, op.cids_M)
    f = cv.lockbox_split(train); tr, te = np.asarray(f.train), np.asarray(f.test)
    rep.line(f"development pairs {len(tr)} (fit), lockbox pairs {len(te)} (score once)")
    M = eo.models(BB, q_op, X, A, B)
    sz = lambda tr_, te_, y_: np.mean([SizeRobust(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], False, False, True)  # noqa: E731
                                       .fit(tr_, y_[tr_]).predict(te_) for m in BB["o5"]], 0)
    cands = {eo.REF: M[eo.REF], eo.CUR: M[eo.CUR], eo.O4: M[eo.O4], "(b) " + eo.OB: M[eo.OB], "(b) + size loss (69)": sz}
    preds = {nm: fn(tr, te, y) for nm, fn in cands.items()}
    yt = y[te]; k = np.array([len(set(train[i].a) & set(train[i].b)) for i in te])
    gap = np.abs(A[te].sum(1) - B[te].sum(1))
    SUB = {"all": k >= 0, "share 0": k == 0, "share 0 same size": (k == 0) & (gap <= 1), "share 1-2": (k >= 1) & (k <= 2), "share >= 3": k >= 3}
    ref = preds[eo.REF]; rows = []
    for nm, p in preds.items():
        s = metrics.score(p, yt); c = ""
        if nm != eo.REF:
            cr, _ = paired(p, ref, yt); c = f"{s['rmse'] - metrics.rmse(ref, yt):+.4f} / {s['r'] - metrics.pearson(ref, yt):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
        sub = [f"{metrics.rmse(p[m], yt[m]):.4f} / {metrics.pearson(p[m], yt[m]):+.3f}" for nm2, m in SUB.items() if nm2 != "all"]
        rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", c] + sub)
    rep.section("Lockbox (129 pairs), fitted on the 514 development pairs; RMSE / r")
    rep.line("n per subset: " + ", ".join(f"{s} {int(m.sum())}" for s, m in SUB.items()))
    rep.table(["model", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [CI]"] + [s for s in SUB if s != "all"], rows)
    rep.value("lockbox", {nm: {s: {"rmse": metrics.rmse(p[m], yt[m]), "r": metrics.pearson(p[m], yt[m])} for s, m in SUB.items()} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
