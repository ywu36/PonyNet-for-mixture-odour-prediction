# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   P = training pairs   M = molecules in the open-POM table   U = units   F = pair features
# ---------------------------------------------------------------------------------------------
"""Finding 3, I2/I3: B1a + discrete pull/push terms on recurring units.

    prediction = B1a  +  ridge over pair indicators of each unit u:
        both[u]    = 1 if both mixtures contain u          (pull if its weight < 0)
        one[u]     = 1 if exactly one mixture contains u    (push if its weight > 0)
        gap[u]     = |level(A) - level(B)|, level = how many of the mixture's molecules carry u,
                     in steps 0 / 1 / 2-3 / 4+ -> 0..3     (frequency within the mixture)
Unit sets (fixed before running):
    groups  the 24 RDKit functional groups in >= 10 training molecules (as E2)
    words   open-POM's 138 odour words; a molecule carries word w if its quantile score > 1
    shared  number of molecules the two mixtures share, and number they do not, both in steps
            0 / 1 / 2-3 / 4+ (one-hot); no per-molecule weights (a molecule recurs too rarely)
Checks (as E2): paired bootstrap vs B1a on the same folds; shuffled units (molecule rows of the
unit table permuted, --perms times; not for "shared"); L1 stability over the outer folds; every
feature's support = number of training pairs where it is non-zero (shown with every claim).

Shared by the eval_units_<set>.py scripts in this folder. Not a script.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_b1a_group_correction import ALPHAS, GroupCorrection, changes, group_table  # noqa: E402
from mixdist import data, evaluate, features, metrics, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402

STEPS = np.array([0, 1, 2, 2, 3])                            # molecules carrying u: 0,1,2,3,4+ -> level


def level(n):
    return STEPS[np.minimum(n, 4)]


def unit_features(pairs, row_of, has_MxU, names_U):
    """-> X_PxF, feature names (both / one / gap per unit)."""
    def count(mix):
        return has_MxU[[row_of[c] for c in mix]].sum(0)
    cA = np.stack([count(p.a) for p in pairs]); cB = np.stack([count(p.b) for p in pairs])
    both = ((cA > 0) & (cB > 0)).astype(float)
    one = ((cA > 0) ^ (cB > 0)).astype(float)
    gap = np.abs(level(cA) - level(cB)).astype(float)
    names = [f"both {u}" for u in names_U] + [f"one {u}" for u in names_U] + [f"gap {u}" for u in names_U]
    return np.hstack([both, one, gap]), names


def shared_features(pairs):
    def onehot(n):
        v = np.zeros(4); v[level(n)] = 1; return v
    rows = [np.concatenate([onehot(len(p.a & p.b)), onehot(len(p.a ^ p.b))]) for p in pairs]
    names = [f"shared {s}" for s in ("0", "1", "2-3", "4+")] + [f"not shared {s}" for s in ("0", "1", "2-3", "4+")]
    return np.array(rows), names


def run(script, unit_sets, doc):
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--perms", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.perms, args.repeats = 3, 2
    rep = Report(script, {"units": unit_sets, **vars(args)}, tag="quick" * args.quick)

    rep.section("1. Units")
    train = data.load_train()
    y_P = evaluate.labels(train)
    fs = features.load("openpom")
    row_of = {int(c): i for i, c in enumerate(fs.cids_M)}
    mol_tr = sorted(set().union(*[m for p in train for m in (p.a, p.b)]))
    train_rows = np.array([row_of[c] for c in mol_tr])
    tables = {}
    if "groups" in unit_sets:
        tables["groups"] = group_table(fs.smiles_M, train_rows)
    if "words" in unit_sets:
        tables["words"] = ([str(w) for w in fs.words_W], mixture.quantile_per_word(fs.probs_MxW) > 1)

    def build(tabs):
        X, names = [], []
        for key, (n_U, has) in tabs.items():
            x, nm = unit_features(train, row_of, has, n_U); X.append(x); names += nm
        if "shared" in unit_sets:
            x, nm = shared_features(train); X.append(x); names += nm
        return np.hstack(X), names

    X_PxF, fnames = build(tables)
    support_F = (X_PxF != 0).sum(0)
    keep = support_F >= 10                                   # a feature non-zero in < 10 pairs cannot carry a claim
    X_PxF, fnames, support_F = X_PxF[:, keep], [f for f, k in zip(fnames, keep) if k], support_F[keep]
    for key, (n_U, has) in tables.items():
        rep.line(f"{key}: {len(n_U)} units; molecules carrying a unit (training): median {int(np.median(has[train_rows].sum(0)))}")
    rep.line(f"{X_PxF.shape[1]} pair features kept (non-zero in >= 10 training pairs; {int((~keep).sum())} dropped)")

    rep.section(f"2. CV ({args.repeats} x 5 folds): B1a + unit terms vs B1a on the same folds")
    dist_P = mixture.paper_distance(train, fs.cids_M, fs.probs_MxW)
    t0 = time.time()
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(dist_P, y_P, np.zeros(len(train))), k=5, repeats=args.repeats)
    gc = GroupCorrection(train, dist_P, y_P, X_PxF)
    res = evaluate.cross_validate(train, lambda tr, te: gc(tr, te, l1=True), k=5, repeats=args.repeats)
    rows = changes(res, ref)
    masks = [np.ones(len(res.pred_N), bool), res.no_overlap_N]
    rep.table(["subset", "n", "B1a RMSE / r", "change RMSE", "change r", "95% CI of r change", "p"],
              [[n, s["n"], f"{metrics.rmse(ref.pred_N[m], ref.truth_N[m]):.3f} / {s['r_ref']:.3f}",
                f"{s['rmse'] - metrics.rmse(ref.pred_N[m], ref.truth_N[m]):+.3f}", f"{s['r'] - s['r_ref']:+.3f}",
                f"[{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]", f"{s['p_vs_ref']:.3f}"] for (n, s), m in zip(rows, masks)])
    gain = {n: s["r"] - s["r_ref"] for n, s in rows}
    rep.line(f"{time.time() - t0:.0f} s")

    if tables and args.perms:
        rep.section(f"3. Chance level: unit tables shuffled among training molecules, {args.perms} times")
        rng = np.random.default_rng(0)
        null = {n: [] for n in gain}
        for _ in range(args.perms):
            shuffled = {}
            for key, (n_U, has) in tables.items():
                h = has.copy(); h[train_rows] = has[rng.permutation(train_rows)]; shuffled[key] = (n_U, h)
            Xs, _ = build(shuffled)
            gp = GroupCorrection(train, dist_P, y_P, Xs[:, keep]); gp.inner = gc.inner
            r = evaluate.cross_validate(train, gp, k=5, repeats=args.repeats)
            for n, m in [("all pairs", np.ones(len(r.pred_N), bool)), ("share no molecule", r.no_overlap_N)]:
                null[n].append(metrics.pearson(r.pred_N[m], r.truth_N[m]) - metrics.pearson(ref.pred_N[m], ref.truth_N[m]))
        for n in null:
            nb = np.array(null[n])
            rep.line(f"{n}: real r change {gain[n]:+.3f}; shuffled 95th pct {np.percentile(nb, 95):+.3f}, median {np.median(nb):+.3f}; "
                     f"fraction of shuffles >= real {np.mean(nb >= gain[n]):.2f}")
        rep.value("null", null)

    rep.section("4. Stability (L1 in every outer fold) and support; pull = 'both' < 0, push = 'one'/'gap' > 0")
    C = np.array(gc.coefs)
    pos, neg = (C > 0).mean(0), (C < 0).mean(0)
    o = np.argsort(-np.maximum(pos, neg))
    rep.table(["feature", "support (pairs)", "selected +", "selected -", "reading"],
              [[fnames[f], int(support_F[f]), f"{pos[f]:.0%}", f"{neg[f]:.0%}",
                ("pull" if neg[f] > pos[f] else "push") + (" (stable)" if max(pos[f], neg[f]) >= 0.8 else "")]
               for f in o[:20]])
    stable = [fnames[f] for f in range(len(fnames)) if max(pos[f], neg[f]) >= 0.8]
    rep.line("stable (>= 80% of folds, one sign): " + (", ".join(stable) or "none"))
    rep.value("gain", gain)
    rep.value("features", {fnames[f]: {"support": int(support_F[f]), "pos": float(pos[f]), "neg": float(neg[f])} for f in range(len(fnames))})
    rep.save()
