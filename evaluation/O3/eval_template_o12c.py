# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   H = middle layer (400)   P = development pairs (514)
#   G = RDKit groups (24)   K = network members
# ---------------------------------------------------------------------------------------------
"""O1.2c's shared-molecule ideas on the HIDDEN-LAYER TEMPLATE (no word-level merge, no decoding; the user 2026-11-19:
"don't do word level merge ... I like the without decoding ... test o1.2c and see if previous hidden level help").

Template (as eval_template_match.py, set-based face recognition): z(A) = sum_m w_m v_m / sum_m w_m, v = O2-C middle
layer (centred, 400), w = exp(theta . x); compared by d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2), u = z/|z|, h = log|z|,
then a line. Never decoded. O1.2c's rules act on w of molecules that are in BOTH mixtures (mixdist/o12.py, O12c), the
same formulas with the template in place of the decoded profile:
  const    shared weight x exp(alpha)                                 (+1 number)
  size     shared weight x exp(alpha + gamma log(n / 10))             (+2)
  tuneout  d = (1 - lmb) d(A, B) + lmb d(A minus shared, B minus shared)  (+1)
  groups   shared weight x exp(alpha + eta . x)                        (+25: over the budget, shown for reference only)
Rows: change from O1.2b (reference, word level, as the only comparison point), paired, 50 x 5, 4 subsets; and each
O1.2c rule minus the plain template (same folds). Prediction = mean over members of each member's line.

Env:     mist
Usage:   python evaluation/O3/eval_template_o12c.py [--folder checkpoints/o2c]
Writes:  results/eval_template_o12c/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b, O12c  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folder", default="checkpoints/o2c"); a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_template_o12c", {"folder": a.folder, "lam": LAM, "repeats": evaluate.CV_REPEATS}, tag=Path(a.folder).name)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M)
    nets = o2.load_members(sorted((ROOT / a.folder).glob("member_*.pt")), dev)
    V = [n.latent(o2.encode_all(n, G, G.dream_rows(), dev))[0].double().cpu().numpy() for n in nets]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    rep.line(f"members {len(nets)} ({a.folder}); middle layer {V[0].shape[1]}; development pairs {len(P)}; "
             f"pairs sharing 0 / 1-2 / >= 3 molecules: {(k_P == 0).sum()} / {((k_P >= 1) & (k_P <= 2)).sum()} / {(k_P >= 3).sum()}")

    from joblib import Parallel, delayed

    def one(make, v, tr, te):
        import torch
        torch.set_num_threads(1)
        return make(v).fit(tr, y[tr]).predict(te)

    pool = Parallel(n_jobs=len(V), backend="loky")

    def ens(make):
        return lambda tr, te: np.mean(pool(delayed(one)(make, v, tr, te) for v in V), 0)
    base = "template, O1.2b form (27 numbers)"
    rows_def = {"O1.2b (reference, word level)": lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
                base: ens(lambda v: O12b(v, X, A, B, True, True, lam=LAM)),
                "template + O1.2c const (28)": ens(lambda v: O12c(v, X, A, B, "const", LAM)),
                "template + O1.2c size (29)": ens(lambda v: O12c(v, X, A, B, "size", LAM)),
                "template + O1.2c tune-out (28)": ens(lambda v: O12c(v, X, A, B, "tuneout", LAM)),
                "template + O1.2c groups (52, over budget)": ens(lambda v: O12c(v, X, A, B, "groups", LAM))}
    res = {}
    for k, f in rows_def.items():
        res[k] = evaluate.cross_validate(P, f); rep.line(f"{k}: done")
    ref = res["O1.2b (reference, word level)"]

    def score(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): hidden-layer templates, change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = score(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)

    rep.section("Paired steps (same folds): O1.2c rule on the template minus the plain template")
    rows = []
    for k in [k for k in res if k.startswith("template +")]:
        for sub in SUB:
            s_ = score(res[k], sub, res[base])
            rows.append([k, sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["rule", "subset", "RMSE change", "r change [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
