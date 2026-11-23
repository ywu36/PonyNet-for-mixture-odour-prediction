# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   H = middle layer (400)   P = development pairs (514)
#   G = RDKit groups (24)   K = network members
# ---------------------------------------------------------------------------------------------
"""One evaluation for every O3 network (findings/o3_exploration_2026-11-19.md). Same tests for all; the loss question
(A) and the aggregation question (B) are reported in separate sections.

Section 1  molecule-level checks, from the checkpoints (computed in training on data never trained on): single held-out
           AUROC, GoodScents held-out word AUROC / target loss, Ma held-out AUROC (fold 0) where recorded.
Section 2  A. losses: DREAM with ONE fixed aggregation, O1.2b's (merge after decoding, quantile profiles):
           theta fitted on DREAM (27 numbers) and theta from training (3 numbers). Paired vs O1.2b, vs O3.0 (frozen
           O2-C-Single) and vs the base arm (first in --arms).
Section 3  B. aggregation: the same networks, middle-layer templates compared by cosine (no decoding; OGCTL,
           arXiv 1903.04752), theta fitted (27 numbers), mean of members' predictions. Paired: template minus
           after-decoding, same network, same folds.
DREAM development pairs, 50 x 5, 4 subsets; changes as RMSE / r [95% CI by resampling mixtures].

Env:     mist
Usage:   python evaluation/O3/eval_o3_suite.py --arms o3s_out_rank_c0 o3s_out_rank_c0_masign ... --tag losses_v1
Writes:  results/eval_o3_suite/<date_time>_<tag>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from eval_o3e import O3Fixed  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    return f"{v.mean():.4f} ({v.std():.4f})" if len(v) else "n/a"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--no-template", action="store_true"); a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_o3_suite", {**vars(a), "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev}, tag=a.tag)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}

    def cv(f):
        return evaluate.cross_validate(P, f)

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"

    res = {"O1.2b (reference)": cv(lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te))}
    _, _, _, cprobs = members(G, "checkpoints/o2c", dev); q_c = mixture.quantile_per_word(cprobs)
    res["O3.0 (frozen O2-C-Single), after decoding, 27"] = cv(lambda tr, te: o3.O3(q_c, X, A, B, True, LAM).fit(tr, y[tr]).predict(te))
    rep.line("references done")
    mol_rows, tmpl = [], {}
    for arm in a.arms:
        files = sorted((ROOT / "checkpoints" / arm).glob("member_*.pt"))
        cks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        assert cks and all(list(c["groups"]) == list(gn) for c in cks), arm
        mol_rows.append([arm, len(cks), ms([c.get("single_auroc") for c in cks]), ms([c.get("scent_auroc") for c in cks]),
                         ms([c.get("scent_loss") for c in cks]), ms([c.get("ma_heldout_auroc") for c in cks]),
                         f"{np.mean([np.abs(c['theta']).max() for c in cks]):.3f}"])
        nets, mu, _, probs = members(G, f"checkpoints/{arm}", dev); q = mixture.quantile_per_word(probs)
        th = np.mean([c["theta"] for c in cks], 0)
        res[f"{arm}: after decoding, theta fitted (27)"] = cv(lambda tr, te, q=q: o3.O3(q, X, A, B, True, LAM).fit(tr, y[tr]).predict(te))
        res[f"{arm}: after decoding, theta from training (3)"] = cv(lambda tr, te, q=q, th=th: O3Fixed(q, X, A, B, th).fit(tr, y[tr]).predict(te))
        if not a.no_template:
            V = [m_.double().cpu().numpy() for m_ in mu]
            tmpl[arm] = cv(lambda tr, te, V=V: np.mean([O12b(v, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in V], 0))
        rep.line(f"{arm}: {len(cks)} members done")
    if not a.no_template:
        _, mu_c, _, _ = members(G, "checkpoints/o2c", dev); Vc = [m_.double().cpu().numpy() for m_ in mu_c]
        tmpl["O3.0 (frozen O2-C-Single)"] = cv(lambda tr, te: np.mean([O12b(v, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in Vc], 0))

    rep.section("1  Molecule-level checks (mean (sd) over members; data never used in training)")
    rep.table(["arm", "members", "single held-out AUROC (O2-C-Single 0.875-0.883)", "GoodScents held-out word AUROC",
               "GoodScents held-out target loss", "Ma held-out AUROC (fold 0)", "max |theta|"], mol_rows)

    ref, o30 = res["O1.2b (reference)"], res["O3.0 (frozen O2-C-Single), after decoding, 27"]
    rep.section(f"2  A. Losses, one aggregation (after decoding, O1.2b's form): DREAM development pairs ({evaluate.CV_REPEATS} x 5)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", "" if r_ is ref else ch(s_)])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)
    base = a.arms[0]; rows = []
    for k, r_ in res.items():
        if r_ is ref or r_ is o30:
            continue
        kind = k.split(": ", 1)[1]
        for sub in SUB:
            rows.append([k, "minus O3.0", sub, ch(sc(r_, sub, o30))])
            if not k.startswith(base + ":"):
                rows.append([k, f"minus {base} ({kind})", sub, ch(sc(r_, sub, res[f'{base}: {kind}']))])
    rep.table(["model", "comparison (same folds)", "subset", "RMSE / r change [95% CI]"], rows)

    if not a.no_template:
        rep.section("3  B. Aggregation: middle-layer templates by cosine (no decoding), theta fitted (27): DREAM development pairs")
        rows = []
        for arm, r_ in tmpl.items():
            after = o30 if arm.startswith("O3.0") else res[f"{arm}: after decoding, theta fitted (27)"]
            for sub in SUB:
                s_ = sc(r_, sub, ref)
                rows.append([arm, sub, f"{s_['rmse']:.4f}", f"{s_['r']:.3f}", ch(s_), ch(sc(r_, sub, after))])
        rep.table(["network", "subset", "RMSE", "r", "template: change from O1.2b", "template minus after decoding (same network)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
