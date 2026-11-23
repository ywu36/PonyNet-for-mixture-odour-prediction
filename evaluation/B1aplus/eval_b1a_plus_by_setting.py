"""B1a+ fitted separately per setting (setting 1: pairs sharing no molecule; setting 2: pairs sharing >= 1).

Inside the standard CV (evaluation/CV.md: 50 x 5, whole mixtures held out) every model below is fitted
only on the training pairs of one setting and scored on the held-out pairs of that setting.
Variants fixed before running, all reported:
  size only           line on log nA + log nB, |log nA - log nB| (no B1a)
  B1a line            B1a distance, line fitted on this setting only
  B1a + size
  B1a+ classes + size theta for the 9 SMARTS classes, lam 1e-3
  B1a+ groups + size  theta for the 24 RDKit groups (>= 5 of the 235 molecules), lam 1e-3
Change from the standard B1a (line fitted on all pairs) on the same pairs. Then theta per class,
"classes + size, lam 1e-3" refitted on all pairs of each setting, 95% CI by resampling mixtures (200),
side by side with the all-pairs theta (results/eval_b1a_plus) and singleton dominance D (Step 1).
V50/T46 not scored.

Env:     mist
Usage:   python evaluation/B1aplus/eval_b1a_plus_by_setting.py
Writes:  results/eval_b1a_plus_by_setting/<date_time>.{txt,json}
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_b1a_plus import piece_matrix  # noqa: E402
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402

LAM, N_BOOT = 1e-3, 200


def latest_value(name, key):
    f = sorted(glob.glob(str(ROOT / f"results/{name}/*.json")))
    if not f:
        return None
    return next((x["value"] for x in json.load(open(f[-1]))["items"] if x.get("key") == key), None)


def main():
    rep = Report("eval_b1a_plus_by_setting", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    smiles = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    d = mixture.paper_distance(train, op.cids_M, op.probs_MxW)
    Xc, nc = piece_matrix(pp.units_smarts(smiles, pp.CLASSES), ids)
    Xg, ng = piece_matrix(pp.merge_identical(pp.units_rdkit_groups(smiles)), ids)
    s1 = np.array([p.jaccard == 0 for p in train])
    settings = {"setting 1 (share no molecule)": s1, "setting 2 (share >= 1 molecule)": ~s1}
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(d, y, np.zeros(len(train))))

    def ols(F, sel):
        D = np.c_[F, np.ones(len(train))]
        def fp(tr, te):
            t = tr[sel[tr]]
            return D[te] @ np.linalg.lstsq(D[t], y[t], rcond=None)[0]
        return fp

    def plus(X, sel):
        def fp(tr, te):
            return B1aPlus(q, X, A, B, S, LAM).fit(tr[sel[tr]], y[tr[sel[tr]]]).predict(te)
        return fp

    rows, thetas = [], {}
    for sname, sel in settings.items():
        variants = {"size only": ols(S, sel), "B1a line (this setting)": ols(d[:, None], sel), "B1a + size": ols(np.c_[d, S], sel),
                    "B1a+ classes + size": plus(Xc, sel), "B1a+ groups + size": plus(Xg, sel)}
        mask = sel[ref.pair_N]
        s0 = evaluate.score_cv(train, ref, mask_N=mask)
        rows.append([sname, "B1a (standard, all pairs)", s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for k, f in variants.items():
            res = evaluate.cross_validate(train, f)
            s = evaluate.score_cv(train, res, reference=ref, mask_N=mask)
            rows.append([sname, k, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
            print(f"  done {sname} {k}", flush=True)
        idx = np.where(sel)[0]
        full = B1aPlus(q, Xc, A, B, S, LAM).fit(idx, y[idx])
        mixes = sorted({m for i in idx for m in (train[i].a, train[i].b)}, key=sorted); im = {m: j for j, m in enumerate(mixes)}
        ia = np.array([im[train[i].a] for i in idx]); ib = np.array([im[train[i].b] for i in idx])
        cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
        th = np.array([B1aPlus(q, Xc, A, B, S, LAM).fit(idx, y[idx], w_pairs=(cnt[b, ia] * cnt[b, ib]).astype(float) + 1e-12).theta.numpy()
                       for b in range(N_BOOT)])
        lo, hi = np.percentile(th, [2.5, 97.5], axis=0)
        thetas[sname] = (full.theta.numpy(), lo, hi)
    rep.section(f"1. Standard CV ({evaluate.CV_REPEATS} x 5), each setting fitted and scored on its own pairs")
    rep.table(["setting", "model", "pairs", "RMSE", "r [95% CI]", "change from standard B1a: RMSE / r [95% CI]"], rows)

    rep.section("2. theta per class (classes + size, lam 1e-3), fitted per setting; > 0 dominate, < 0 masked")
    allp = latest_value("eval_b1a_plus", "theta")
    D = {}
    sp = latest_value("singleton_piece_dominance", "pieces")
    if sp:
        D = {r["piece"]: r["pom"][0] for r in sp["chemical classes"]}
    fmt = lambda t, k: f"{t[0][k]:+.2f} [{t[1][k]:+.2f}, {t[2][k]:+.2f}]"
    keys = list(thetas)
    rep.table(["class", "molecules", "setting 1 theta [95% CI]", "setting 2 theta [95% CI]", "all pairs theta", "single-molecule D"],
              [[n, int(Xc[:, k].sum()), fmt(thetas[keys[0]], k), fmt(thetas[keys[1]], k),
                f"{allp['theta'][allp['names'].index(n)]:+.2f}" if allp else "", f"{D.get(n, float('nan')):+.3f}"] for k, n in enumerate(nc)])
    rep.value("theta", {k: {"names": nc, "theta": v[0].tolist(), "lo": v[1].tolist(), "hi": v[2].tolist()} for k, v in thetas.items()})
    rep.save()


if __name__ == "__main__":
    main()
