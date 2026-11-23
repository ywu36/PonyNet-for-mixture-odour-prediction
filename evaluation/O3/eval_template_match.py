# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   H = middle layer (400)   P = development pairs (514)
#   G = RDKit groups (24)   K = network members
# ---------------------------------------------------------------------------------------------
"""Template matching in the middle layer, as set-based face recognition (Wu & Kakadiaris 2019, OGCTL, arXiv 1903.04752:
per-part vectors are summed into one template and two templates are compared by cosine only; the summed template is
never passed through a decoder, so the number of parts does not matter). Frozen networks, no training.

For each member: mixture template z(A) = sum_m w_m v_m / sum_m w_m, w = exp(theta . x), v = middle layer (O2-C:
centred, 400). d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2), u = z / |z|, h = log |z| (O1.2b's form on templates
instead of decoded profiles). Prediction = mean over members of each member's line.
Rows: change from O1.2b, paired, development pairs, 50 x 5, 4 subsets:
  template, direction only, theta = 0       (2 numbers: line)
  template, direction + length, theta = 0   (3 numbers)
  template, direction + length, theta       (27 numbers, = O1.2b's count)

Env:     mist
Usage:   python evaluation/O3/eval_template_match.py [--folder checkpoints/o2c]
Writes:  results/eval_template_match/<date_time>.{txt,json}
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
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folder", default="checkpoints/o2c"); a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_template_match", {"folder": a.folder, "lam": LAM, "repeats": evaluate.CV_REPEATS}, tag=Path(a.folder).name)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M)
    nets = o2.load_members(sorted((ROOT / a.folder).glob("member_*.pt")), dev)
    V = [n.latent(o2.encode_all(n, G, G.dream_rows(), dev))[0].double().cpu().numpy() for n in nets]
    rep.line(f"members {len(nets)} ({a.folder}); middle layer {V[0].shape[1]}; development pairs {len(P)}")
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}

    def ens(use_theta, use_len):
        return lambda tr, te: np.mean([O12b(v, X, A, B, use_theta, use_len, lam=LAM).fit(tr, y[tr]).predict(te) for v in V], 0)
    rows_def = {"O1.2b (reference)": lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
                "template, direction only, theta = 0 (2 numbers)": ens(False, False),
                "template, direction + length, theta = 0 (3 numbers)": ens(False, True),
                "template, direction + length, theta (27 numbers)": ens(True, True)}
    res = {}
    for k, f in rows_def.items():
        res[k] = evaluate.cross_validate(P, f); rep.line(f"{k}: done")
    ref = res["O1.2b (reference)"]
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): middle-layer templates compared by cosine, change from O1.2b (paired)")
    rows = []
    for sub, m in SUB.items():
        for k, r_ in res.items():
            s_ = evaluate.score_cv(P, r_, reference=None if r_ is ref else ref, mask_N=None if m is None else m[r_.pair_N])
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
