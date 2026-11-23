# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   W = words (138)   P = DREAM development pairs (514)
# ---------------------------------------------------------------------------------------------
"""Per-unit normalisation of the hidden layer (the user, 2026-11-19: "we may need a normalization layer").

why_setting1.py: on pairs sharing no molecule the SAME count network reads 0.349 through its words + O1.2b rule but
0.192 through its hidden layer + attention; length weight, attention focus and whitening alone do not explain it.
The word route has a normalisation the hidden route lacks: quantile per word = each column mapped to a standard normal
by rank over the 235 DREAM molecules (centres and scales every dimension; no labels). Test the same on hidden units,
and separate the other differences (word head, group weights):
  hidden raw, attention angle only (67)                            current O3s readout (baseline)
  hidden, quantile per unit, plain mean + length (3)               normalisation alone
  hidden, quantile per unit, attention angle only (67) / + length (68)
  words (quantile), plain mean + length (3)                        count network only: word head without group weights
  words (quantile), O1.2b rule (27)                                count network only: + group weights
Networks: count network (ga_w64) and plain network (o3m_w64), 3 members, predictions averaged. Paired 50 x 5.

Env:     mist
Usage:   python evaluation/O3/eval_unit_norm.py [--repeats 50]
Writes:  results/eval_unit_norm/<date_time>.{txt,json}
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report, results_dir  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from why_setting1 import Attn  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=evaluate.CV_REPEATS); a_ = ap.parse_args()
    dev = pick_device(); rep = Report("eval_unit_norm", {"repeats": a_.repeats, "seeds": SEEDS, "lam": LAM, "device": dev},
                                      tag="smoke" if a_.repeats < 5 else "")
    Gg, tr, _, _ = train_groupatoms.graphs_with_groups(); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    V, Q = {}, None
    for name, nets, g in (("count network", [train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg),
                          ("plain network", [train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G)):
        V[name] = []
        for n in nets:
            with torch.no_grad():
                sc = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                V[name].append(n.latent(o2.encode_all(n, g, g.dream_rows(), dev))[0].double().cpu().numpy() / sc)
        if name == "count network":
            Q = mixture.quantile_per_word(np.mean([o2.predict_probs(n, g, g.dream_rows(), dev) for n in nets], 0).astype(float))
    VQ = {k: [mixture.quantile_per_word(v) for v in vs] for k, vs in V.items()}
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
           "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}

    bad = {}

    def ens(Vs, direction_only, fit_q, key=None):
        def f(tr_, te_):
            out = []
            for v in Vs:                                   # a fit that diverges (non-finite) falls back to the training mean
                p_ = Attn(v, A, B, direction_only, fit_q).fit(tr_, y[tr_]).predict(te_)
                if not np.all(np.isfinite(p_)):
                    bad[key] = bad.get(key, 0) + 1; p_ = np.full(len(te_), y[tr_].mean())
                out.append(p_)
            return np.mean(out, 0)
        return f
    REF = "O1.2b (reference, 27)"; BASE = "count network, hidden raw, attention angle only (67)"
    arms = {REF: lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_),
            "count network, words (quantile), O1.2b rule (27)": lambda tr_, te_: O12b(Q, X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_),
            "count network, words (quantile), plain mean + length (3)": ens([Q], False, False, "count network, words (quantile), plain mean + length (3)")}
    for net in V:
        arms[f"{net}, hidden raw, attention angle only (67)"] = ens(V[net], True, True, f"{net}, hidden raw, attention angle only (67)")
        arms[f"{net}, hidden quantile per unit, plain mean + length (3)"] = ens(VQ[net], False, False, f"{net}, hidden quantile per unit, plain mean + length (3)")
        arms[f"{net}, hidden quantile per unit, attention angle only (67)"] = ens(VQ[net], True, True, f"{net}, hidden quantile per unit, attention angle only (67)")
        arms[f"{net}, hidden quantile per unit, attention + length (68)"] = ens(VQ[net], False, True, f"{net}, hidden quantile per unit, attention + length (68)")
    res = {}
    for k, f in arms.items():
        res[k] = evaluate.cross_validate(P, f, repeats=a_.repeats); rep.line(f"done: {k}")
    (results_dir() / "eval_unit_norm").mkdir(parents=True, exist_ok=True)
    with open(results_dir() / "eval_unit_norm" / f"{rep.time:%Y-%m-%d_%H%M%S}_cv.pkl", "wb") as fh:
        pickle.dump({"res": res, "k_P": k_P}, fh)
    rep.line(f"fits that diverged (fell back to the training mean), per arm: {bad if bad else 'none'}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({a_.repeats} x 5): RMSE / r; changes paired on the same folds")
    out = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None); base = k.split(", ")[0] + ", hidden raw, attention angle only (67)"
            out.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                        "" if k == REF else ch(sc(r_, sub, res[REF])), ch(sc(r_, sub, res[base])) if base in res and base != k else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from hidden raw attention (same network)"], out)
    rep.save()


if __name__ == "__main__":
    main()
