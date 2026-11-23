"""B1a+ with one offset per experiment (nuisance, allowed by evaluation/CV.md): what do size and
sub-molecular dominance add once experiment differences are removed?

Trained on all 820 training pairs (the previous run showed training on all pairs scores setting 1
better than training on setting 1 alone); scored on all pairs, setting 1 (share no molecule) and
setting 2 (share >= 1). Variants fixed before running, all reported:
  experiment offsets only
  size + experiment            (no B1a distance)
  B1a + experiment
  B1a + experiment + size
  B1a+ classes + experiment + size   theta for the 9 SMARTS classes, lam 1e-3
  B1a+ groups + experiment + size    theta for the 24 RDKit groups (>= 5 of the 235 molecules), lam 1e-3
Standard CV (50 x 5, whole mixtures held out); change from the standard B1a on the same pairs.
Attribution: theta for classes and for groups (each "+ experiment + size", lam 1e-3) on all 820
pairs, 95% CI by resampling mixtures (200). V50/T46 not scored.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_exp.py
Writes:  results/eval_b1a_plus_exp/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_b1a_plus import piece_matrix  # noqa: E402
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, experiment_onehot, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402

LAM, N_BOOT = 1e-3, 200


def main():
    rep = Report("eval_b1a_plus_exp", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    smiles = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    E, exps = experiment_onehot(train)
    d = mixture.paper_distance(train, op.cids_M, op.probs_MxW)
    Xc, nc = piece_matrix(pp.units_smarts(smiles, pp.CLASSES), ids)
    Xg, ng = piece_matrix(pp.merge_identical(pp.units_rdkit_groups(smiles)), ids)
    rep.line(f"experiments: " + ", ".join(f"{e} {int(E[:, k].sum())} pairs" for k, e in enumerate(exps)))
    s1 = np.array([p.jaccard == 0 for p in train])

    def ols(F):
        D = np.c_[F, E]
        return lambda tr, te: D[te] @ np.linalg.lstsq(D[tr], y[tr], rcond=None)[0]

    def plus(X):
        return lambda tr, te: B1aPlus(q, X, A, B, S, LAM, E).fit(tr, y[tr]).predict(te)

    variants = {"experiment offsets only": ols(np.zeros((len(train), 0))), "size + experiment": ols(S),
                "B1a + experiment": ols(d[:, None]), "B1a + experiment + size": ols(np.c_[d, S]),
                "B1a+ classes + experiment + size": plus(Xc), "B1a+ groups + experiment + size": plus(Xg)}
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(d, y, np.zeros(len(train))))
    res = {}
    for k, f in variants.items():
        res[k] = evaluate.cross_validate(train, f); print(f"  done {k}", flush=True)
    rows = []
    for subset, mask in [("all pairs", None), ("setting 1", s1[ref.pair_N]), ("setting 2", ~s1[ref.pair_N])]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([subset, "B1a (standard)", s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k in variants:
            s = evaluate.score_cv(train, res[k], reference=ref, mask_N=mask)
            rows.append([subset, k, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.section(f"1. Standard CV ({evaluate.CV_REPEATS} x 5), trained on all pairs, change from standard B1a")
    rep.table(["subset", "model", "pairs", "RMSE", "r [95% CI]", "change from B1a: RMSE / r [95% CI]"], rows)

    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: j for j, m in enumerate(mixes)}
    ia = np.array([im[p.a] for p in train]); ib = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    allidx = np.arange(len(train))
    out = {}
    for title, X, names in [("classes", Xc, nc), ("groups", Xg, ng)]:
        full = B1aPlus(q, X, A, B, S, LAM, E).fit(allidx, y)
        th = np.array([B1aPlus(q, X, A, B, S, LAM, E).fit(allidx, y, w_pairs=(cnt[b, ia] * cnt[b, ib]).astype(float) + 1e-12).theta.numpy()
                       for b in range(N_BOOT)])
        lo, hi = np.percentile(th, [2.5, 97.5], axis=0)
        mix_with = np.stack([A, B]).max(0) @ X > 0
        rep.section(f"2. theta per {title[:-2] if title == 'classes' else 'group'} (+ experiment + size, lam 1e-3, all 820 pairs); > 0 dominate, < 0 masked")
        o = np.argsort(-full.theta.numpy())
        rep.table(["piece", "molecules (of 235)", "pairs with it on a side", "theta [95% CI]", "weight factor"],
                  [[names[k][:44], int(X[:, k].sum()), int(mix_with[:, k].sum()), f"{full.theta[k]:+.2f} [{lo[k]:+.2f}, {hi[k]:+.2f}]",
                    f"{np.exp(full.theta[k].item()):.2f}"] for k in o])
        out[title] = {"names": names, "theta": full.theta.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    rep.value("theta", out)
    rep.save()


if __name__ == "__main__":
    main()
