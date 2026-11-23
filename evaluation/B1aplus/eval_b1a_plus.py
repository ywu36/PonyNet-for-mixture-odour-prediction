"""B1a+ (mixdist/b1a_plus.py): does sub-molecular dominance weighting help over B1a open-POM?

All variants fixed before running, all reported (no selection among them):
  B1a                         the frozen reference (line on the B1a distance)
  B1a + size                  + log nA + log nB, |log nA - log nB|
  B1a+ classes, lam L         theta for the 9 SMARTS classes (pushpull.CLASSES), no size
  B1a+ classes + size, lam L  theta for the 9 classes + size
  B1a+ groups + size, lam L   theta for RDKit fr_ groups carried by >= 5 of the 235 molecules (merged) + size
  L in {1e-4, 1e-3, 1e-2} (penalty on theta; mean squared error of y is ~0.016)
Standard CV (evaluation/CV.md: 50 x 5, whole mixtures held out); change from B1a on the same folds,
all pairs and pairs sharing no molecule. Then attribution: "classes + size, lam 1e-3" refitted on all
820 pairs, theta with 95% CI by resampling mixtures (200), next to singleton dominance D (Step 1).
V50/T46 not scored.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus.py
Writes:  results/eval_b1a_plus/<date_time>.{txt,json}
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402

LAMS = [1e-4, 1e-3, 1e-2]
N_BOOT = 200


def piece_matrix(units, ids):
    X = np.zeros((len(ids), len(units.names)))
    for r, i in enumerate(ids):
        X[r, list(units.carriers.get(i, ()))] = 1
    keep = X.sum(0) >= 5
    return X[:, keep], [n for n, k in zip(units.names, keep) if k]


def main():
    rep = Report("eval_b1a_plus", {"lams": LAMS, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    smiles = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    Xc, nc = piece_matrix(pp.units_smarts(smiles, pp.CLASSES), ids)
    Xg, ng = piece_matrix(pp.merge_identical(pp.units_rdkit_groups(smiles)), ids)
    rep.line(f"pieces: {len(nc)} classes, {len(ng)} RDKit groups (>= 5 of the 235 molecules)")
    chk = B1aPlus(q, Xc, A, B).distance(__import__("torch").zeros(len(nc)), np.arange(len(train))).numpy()
    rep.line(f"check: B1a+ at theta = 0 vs B1a distance, max |difference| {np.abs(chk - mixture.paper_distance(train, op.cids_M, op.probs_MxW)).max():.2e}")

    def fp(X, size, lam):
        def fit_predict(tr, te):
            m = B1aPlus(q, X, A, B, S if size else None, lam).fit(tr, y[tr])
            return m.predict(te)
        return fit_predict

    variants = {"B1a + size": fp(np.zeros((len(ids), 0)), True, 0.0)}
    for lam in LAMS:
        variants[f"B1a+ classes, lam {lam:g}"] = fp(Xc, False, lam)
        variants[f"B1a+ classes + size, lam {lam:g}"] = fp(Xc, True, lam)
        variants[f"B1a+ groups + size, lam {lam:g}"] = fp(Xg, True, lam)
    rep.section(f"1. Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a on the same pairs")
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(mixture.paper_distance(train, op.cids_M, op.probs_MxW), y, np.zeros(len(train))))
    res = {}
    for k, f in variants.items():
        res[k] = evaluate.cross_validate(train, f)
        print(f"  done {k}", flush=True)
    rows = []
    for subset, mask in [("all pairs", None), ("share no molecule", ref.no_overlap_N)]:
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append(["B1a (reference)", subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k in variants:
            s = evaluate.score_cv(train, res[k], reference=ref, mask_N=mask)
            rows.append([k, subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a: RMSE / r [95% CI]"], rows)

    rep.section("2. Attribution: theta per class (B1a+ classes + size, lam 1e-3, all 820 pairs)")
    full = B1aPlus(q, Xc, A, B, S, 1e-3).fit(np.arange(len(train)), y)
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ia = np.array([im[p.a] for p in train]); ib = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    th = np.array([B1aPlus(q, Xc, A, B, S, 1e-3).fit(np.arange(len(train)), y, w_pairs=(cnt[b, ia] * cnt[b, ib]).astype(float) + 1e-12).theta.numpy()
                   for b in range(N_BOOT)])
    lo, hi = np.percentile(th, [2.5, 97.5], axis=0)
    D = {}
    f = sorted(glob.glob(str(ROOT / "results/singleton_piece_dominance/*.json")))
    if f:
        items = json.load(open(f[-1]))["items"]
        for r in next(x["value"] for x in items if x.get("key") == "pieces")["chemical classes"]:
            D[r["piece"]] = r["pom"][0]
    mix_with = (np.stack([A, B]).max(0) @ Xc > 0)
    rep.table(["class", "molecules", "pairs with it on a side", "theta [95% CI]", "weight factor exp(theta)", "singleton dominance D (Step 1)"],
              [[n, int(Xc[:, k].sum()), int(mix_with[:, k].sum()), f"{full.theta[k]:+.2f} [{lo[k]:+.2f}, {hi[k]:+.2f}]",
                f"{np.exp(full.theta[k].item()):.2f}", f"{D.get(n, float('nan')):+.3f}"] for k, n in enumerate(nc)])
    rep.line("theta > 0: molecules of this class count more in their mixture's smell than B1a's equal weight (dominate); < 0: masked.")
    rep.value("theta", {"names": nc, "theta": full.theta.tolist(), "lo": lo.tolist(), "hi": hi.tolist()})
    rep.save()


if __name__ == "__main__":
    main()
