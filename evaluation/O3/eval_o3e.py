# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   W = words (138)   G = RDKit groups (24)
#   P = development pairs (514)
# ---------------------------------------------------------------------------------------------
"""O3-E on DREAM (design/o3_design.md): profiles from the encoder trained jointly on singles + GoodScents (+ Ma)
(training/train_o3e.py; no DREAM in training), read out with O1.2b's form.
Rows (change from O1.2b, paired, development pairs, 50 x 5, 4 subsets):
  O3.0                                   frozen O2-C-Single profiles, O1.2b form (27 numbers per fold)
  O3-E[-noMa], theta fitted on DREAM      27 numbers per fold
  O3-E[-noMa], theta from joint training  theta fixed (GoodScents + Ma), DREAM fits beta + line only (3 numbers)
Paired steps: O3-E minus O3.0 (what joint training of the encoder adds). The GoodScents blend weight
sigmoid(theta . (x1 - x2)) equals O1.2b's exp(theta . x) normalized over two molecules, so theta transfers as is.

Env:     mist (GPU)
Usage:   python evaluation/O3/eval_o3e.py
Writes:  results/eval_o3e/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402


class O3Fixed(o3.O3):
    """O1.2b form with the group gain theta fixed from elsewhere; fits beta + line only."""

    def __init__(self, q, x, A, B, theta_G, lam=LAM):
        super().__init__(q, x, A, B, False, lam); self.tf = torch.as_tensor(np.asarray(theta_G, float))

    def log_w(self, theta, Mx):
        return (self.x @ self.tf).unsqueeze(0).expand(Mx.shape[0], -1)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="*", default=["o3e:O3-E", "o3e_noma:O3-E-noMa"],
                                                    help="checkpoint folder:label pairs (under checkpoints/)")
    ap.add_argument("--name", default="eval_o3e"); a = ap.parse_args()
    dev = pick_device(); rep = Report(a.name, {"lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev, "arms": a.arms})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    q_op = mixture.quantile_per_word(op.probs_MxW)
    rows_def = {"O1.2b (reference)": lambda tr, te: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)}
    _, _, _, cprobs = members(G, "checkpoints/o2c", dev); q_c = mixture.quantile_per_word(cprobs)
    rows_def["O3.0 (frozen O2-C-Single profiles)"] = lambda tr, te: o3.O3(q_c, X, A, B, True, LAM).fit(tr, y[tr]).predict(te)
    for folder, tag in [("checkpoints/" + x.split(":")[0], x.split(":")[1]) for x in a.arms]:
        files = sorted((ROOT / folder).glob("member_*.pt"))
        cks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        assert all(list(c["groups"]) == list(gn) for c in cks)
        _, _, _, probs = members(G, folder, dev); q = mixture.quantile_per_word(probs); th = np.mean([c["theta"] for c in cks], 0)
        rep.line(f"{tag}: {len(cks)} members; single held-out AUROC {np.mean([c['single_auroc'] for c in cks]):.4f}; GoodScents held-out loss "
                 f"{np.mean([c['scent_loss'] for c in cks]):.4f} (sd {np.std([c['scent_loss'] for c in cks]):.4f}); theta |mean| {np.abs(th).mean():.3f}")
        rows_def[f"{tag}, theta fitted on DREAM (27 numbers)"] = lambda tr, te, q=q: o3.O3(q, X, A, B, True, LAM).fit(tr, y[tr]).predict(te)
        rows_def[f"{tag}, theta from joint training (3 numbers)"] = lambda tr, te, q=q, th=th: O3Fixed(q, X, A, B, th).fit(tr, y[tr]).predict(te)
    res = {}
    for k, f in rows_def.items():
        res[k] = evaluate.cross_validate(P, f); rep.line(f"{k}: done")
    ref = res["O1.2b (reference)"]

    def score(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = score(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)

    rep.section("Paired steps (same folds): first minus second")
    rows = []
    base = [k for k in res if k.startswith(a.arms[0].split(":")[1] + ",")]
    pairs_cmp = [(hi, "O3.0 (frozen O2-C-Single profiles)") for hi in res if hi not in ("O1.2b (reference)", "O3.0 (frozen O2-C-Single profiles)")]
    pairs_cmp += [(hi, base[i % 2]) for hi in res if hi not in base and hi not in ("O1.2b (reference)", "O3.0 (frozen O2-C-Single profiles)")
                  for i in [0 if "27 numbers" in hi else 1]]
    for hi, lo in pairs_cmp:
        for sub in SUB:
            s_ = score(res[hi], sub, res[lo])
            rows.append([f"{hi}  minus  {lo}", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["comparison", "subset", "RMSE change", "r change [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
