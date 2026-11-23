"""Pairs sharing exactly 1 molecule: does the SAME shared molecule give B1a++ the same error across pairs?

Audit (the share-1 subset was chosen after seeing results). B1a++ = B1a + size + groups (B1a+ RDKit groups
+ size, lam 1e-3); its held-out error per pair e = y - mean CV prediction (standard CV, all 820 pairs).
For each share-1 pair p with shared molecule s: predict e_p by the mean error of the OTHER share-1 pairs
with the same shared molecule, excluding pairs that share a mixture with p (so a mixture rated oddly cannot
create the match). Pairs with no such partner are left out.
  r(e, leave-one-out molecule mean) and the slope; 95% CI by resampling mixtures (2000);
  null: shuffle shared-molecule labels among the pairs (2000), same exclusions.
Consistent (r > 0 beyond the null): the shared molecule's own identity carries a pull that groups and
open-POM miss. Not consistent: the leftover variation looks like noise.

Env:     mist
Usage:   python evaluation/B1aplus/eval_setting2_share1_repeat.py
Writes:  results/eval_setting2_share1_repeat/<date_time>.{txt,json}
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3
N_BOOT = 2000
N_PERM = 2000


def loo_means(e, mol, mixA, mixB):
    """-> leave-one-out mean error of other pairs with the same shared molecule and no shared mixture (nan if none)."""
    out = np.full(len(e), np.nan)
    for i in range(len(e)):
        j = [k for k in range(len(e)) if k != i and mol[k] == mol[i]
             and not ({mixA[k], mixB[k]} & {mixA[i], mixB[i]})]
        if j:
            out[i] = e[j].mean()
    return out


def wr(x, y, w):
    ok = np.isfinite(x)
    x, y, w = x[ok], y[ok], w[ok]
    mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
    return (w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())


def main():
    rep = Report("eval_setting2_share1_repeat", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT, "n_perm": N_PERM})
    train = data.load_train()
    y = evaluate.labels(train)
    op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M)
    S = size_features(train)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    cur = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q, X, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    w = evaluate.cv_weights(cur)
    pred = np.bincount(cur.pair_N, weights=w * cur.pred_N, minlength=len(train)) / np.bincount(cur.pair_N, weights=w, minlength=len(train))
    print("  done B1a++ CV", flush=True)

    idx = [i for i, p in enumerate(train) if len(p.a & p.b) == 1]
    P = [train[i] for i in idx]
    e = y[idx] - pred[idx]
    mol = [next(iter(p.a & p.b)) for p in P]
    mixA = [p.a for p in P]; mixB = [p.b for p in P]
    m = loo_means(e, mol, mixA, mixB)
    ok = np.isfinite(m)
    cnt = Counter(mol)
    rep.section(f"0. {len(P)} pairs share exactly 1 molecule; {len(cnt)} distinct shared molecules")
    rep.line(f"pairs per shared molecule: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(cnt.values()).items())) + " (pairs: molecules)")
    rep.line(f"pairs with >= 1 partner (same shared molecule, no shared mixture): {int(ok.sum())}, from {len({mol[i] for i in np.where(ok)[0]})} molecules")
    rep.line(f"B1a++ on these pairs: error mean {e.mean():+.3f}, sd {e.std():.3f}")

    mixes = sorted({x for p in P for x in (p.a, p.b)}, key=sorted); im = {x: i for i, x in enumerate(mixes)}
    ia = np.array([im[p.a] for p in P]); ib = np.array([im[p.b] for p in P])
    one = np.ones(len(P))
    r0 = wr(m, e, one)
    xs = m[ok]; es = e[ok]
    slope0 = np.polyfit(xs, es, 1)[0]
    c = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    rb = np.array([wr(m, e, (c[b, ia] * c[b, ib]).astype(float)) for b in range(N_BOOT)]); rb = rb[np.isfinite(rb)]
    rng = np.random.default_rng(1); rp = []
    for _ in range(N_PERM):
        mp = list(rng.permutation(mol))
        mm = loo_means(e, mp, mixA, mixB)
        if np.isfinite(mm).sum() >= 10:
            rp.append(wr(mm, e, one))
    rp = np.array(rp)
    rep.section("1. Does the same shared molecule give the same error?")
    rep.table(["quantity", "value"],
              [["r(error, mean error of other pairs with the same shared molecule)", f"{r0:+.3f} [{np.percentile(rb, 2.5):+.3f}, {np.percentile(rb, 97.5):+.3f}]"],
               ["slope (error per unit of partner mean error)", f"{slope0:+.3f}"],
               ["null (shuffled molecule labels): median r [95% range]", f"{np.median(rp):+.3f} [{np.percentile(rp, 2.5):+.3f}, {np.percentile(rp, 97.5):+.3f}]"],
               ["permutation p (one-sided, r >= observed)", f"{(np.sum(rp >= r0) + 1) / (len(rp) + 1):.3f}"]])
    o = sorted([(k, v) for k, v in cnt.items() if v >= 3], key=lambda t: -t[1])
    names = {}
    try:
        import csv
        for r in csv.DictReader(open(ROOT / "external/pnas_repo/SOTA/data/raw/openpom_ensemble_predictions_results.csv")):
            names[int(r["CID"])] = r["IUPAC Name"]
    except Exception:
        pass
    rep.table(["shared molecule", "pairs", "mean B1a++ error (rating - prediction)", "errors"],
              [[names.get(int(k), str(k))[:40], v, f"{e[[i for i in range(len(P)) if mol[i] == k]].mean():+.3f}",
                " ".join(f"{x:+.2f}" for x in e[[i for i in range(len(P)) if mol[i] == k]])] for k, v in o])
    rep.line("shared molecules in >= 3 pairs. Error < 0: rated closer than B1a++ predicts (pulls more).")
    rep.value("result", {"r": float(r0), "ci": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                         "null_median": float(np.median(rp)), "perm_p": float((np.sum(rp >= r0) + 1) / (len(rp) + 1)), "n_pairs_used": int(ok.sum())})
    rep.save()


if __name__ == "__main__":
    main()
