# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   P = training pairs in the group   W = odour words (138)   G = RDKit functional groups
#   F = Morgan fragments (radius 0-2) in >= 10 training molecules   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""Same checks on the two kinds of training pair: sharing no molecule, sharing >= 1.

1. Substructure: do the two mixtures share substructures (Morgan fragments, functional groups)
   even when they share no molecule? Does that sharing follow the human rating y?
2. Labels: how spread out is y (one mean rating per pair; no repeat ratings in the training data,
   so a noise ceiling cannot be computed here).
3. Odour words: per word w, r between |u_w(A) - u_w(B)| and y (u = B1a's mixture profile:
   quantile per word, mean, length 1), and r with B1a's out-of-fold residual (beyond B1a).
4. Substructures: the same per functional group and per Morgan fragment, with the share of the
   mixture's molecules carrying it.
CIs and p-values by resampling mixtures (pairs sharing a mixture are not independent);
Benjamini-Hochberg q over the units of each table. Standard CV for the residual (evaluation/CV.md).

Env:     mist
Usage:   python findings/pair_signal_by_sharing.py
Writes:  results/pair_signal_by_sharing/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation/Substructure"))
from eval_b1a_group_correction import group_table  # noqa: E402
from mixdist import data, evaluate, features, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402

N_BOOT = 1000
MIN_MOLECULES = 10


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def wcorr(X_PxF, y_P, w_P):
    """Weighted Pearson of every column with y -> r_F."""
    w = w_P / w_P.sum()
    xc = X_PxF - w @ X_PxF; yc = y_P - w @ y_P
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ (xc * yc[:, None])) / np.sqrt((w @ xc ** 2) * (w @ yc ** 2))


def corr_table(X_PxF, y_P, w_BxP):
    """Point r, 95% CI and two-sided bootstrap p per column."""
    r = wcorr(X_PxF, y_P, np.ones(len(y_P)))
    rb = np.array([wcorr(X_PxF, y_P, w) for w in w_BxP])
    lo, hi = np.nanpercentile(rb, [2.5, 97.5], axis=0)
    p = np.minimum(1, 2 * np.minimum(np.nanmean(rb <= 0, 0), np.nanmean(rb >= 0, 0)))
    return r, lo, hi, np.maximum(p, 1 / len(w_BxP))


def morgan_table(smiles_M, train_rows):
    """Presence of Morgan fragments (radius 0-2) per molecule, kept if in >= MIN_MOLECULES training
    molecules -> names_F, has_MxF."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    frags, name = [], {}
    for s in smiles_M:
        m = Chem.MolFromSmiles(s); info = {}
        fp = AllChem.GetMorganFingerprint(m, 2, bitInfo=info)
        frags.append(set(fp.GetNonzeroElements()))
        for k, envs in info.items():
            if k in name:
                continue
            a, rad = envs[0]
            if rad == 0:
                name[k] = m.GetAtomWithIdx(a).GetSymbol() + ("(arom)" if m.GetAtomWithIdx(a).GetIsAromatic() else "")
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(m, rad, a)
                name[k] = Chem.MolToSmiles(Chem.PathToSubmol(m, env)) if env else str(k)
    ids = sorted(set().union(*[frags[i] for i in train_rows]))
    has = np.array([[k in f for k in ids] for f in frags])
    keep = has[train_rows].sum(0) >= MIN_MOLECULES
    return [name[k] for k, kk in zip(ids, keep) if kk], has[:, keep]


def main():
    rep = Report("pair_signal_by_sharing", {"n_boot": N_BOOT, "min_molecules": MIN_MOLECULES})
    train = data.load_train()
    y_P = evaluate.labels(train)
    fs = features.load("openpom")
    words_W = [str(w) for w in fs.words_W]
    row_of = {int(c): i for i, c in enumerate(fs.cids_M)}
    mol_tr = sorted(set().union(*[m for p in train for m in (p.a, p.b)]))
    train_rows = np.array([row_of[c] for c in mol_tr])

    d_P = mixture.paper_distance(train, fs.cids_M, fs.probs_MxW)
    res = evaluate.cross_validate(train, evaluate.linear_calibration(d_P, y_P, np.zeros(len(train))))
    n = np.bincount(res.pair_N, minlength=len(train))
    eps_P = y_P - np.bincount(res.pair_N, weights=res.pred_N, minlength=len(train)) / n

    a_PxW, b_PxW = mixture.mean_profiles(train, fs.cids_M, mixture.quantile_per_word(fs.probs_MxW))
    a_PxW /= np.linalg.norm(a_PxW, axis=1, keepdims=True); b_PxW /= np.linalg.norm(b_PxW, axis=1, keepdims=True)
    dW_PxW = np.abs(a_PxW - b_PxW)

    gnames, has_MxG = group_table(fs.smiles_M, train_rows)
    fnames, has_MxF = morgan_table(fs.smiles_M, train_rows)

    def frac(has):
        A = np.stack([has[[row_of[c] for c in p.a]].mean(0) for p in train])
        B = np.stack([has[[row_of[c] for c in p.b]].mean(0) for p in train])
        return A, B
    gA, gB = frac(has_MxG); fA, fB = frac(has_MxF)
    def tanimoto(A, B):
        a, b = A > 0, B > 0
        return (a & b).sum(1) / np.maximum((a | b).sum(1), 1)
    subsim_P = tanimoto(fA, fB)                              # share of fragments present in both mixtures
    groupsim_P = tanimoto(gA, gB)

    mixes = sorted({x for p in train for x in (p.a, p.b)}, key=sorted)
    idx = {x: i for i, x in enumerate(mixes)}
    ia = np.array([idx[p.a] for p in train]); ib = np.array([idx[p.b] for p in train])
    c = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    w_BxP = (c[:, ia] * c[:, ib]).astype(float)

    share_P = np.array([len(p.a & p.b) for p in train])
    groups = {"share no molecule": share_P == 0, "share >= 1 molecule": share_P > 0}
    rep.line(f"{len(train)} training pairs; {len(gnames)} functional groups and {len(fnames)} Morgan fragments in >= {MIN_MOLECULES} training molecules")

    rep.section("1. Substructure shared between the two mixtures")
    rows = []
    for g, m in groups.items():
        X = np.stack([subsim_P[m], groupsim_P[m], d_P[m]], 1)
        r, lo, hi, _ = corr_table(X, y_P[m], w_BxP[:, m])
        re_, lo_e, hi_e, _ = corr_table(X[:, :2], eps_P[m], w_BxP[:, m])
        rows.append([g, int(m.sum()), f"{np.median(subsim_P[m]):.2f} ({np.percentile(subsim_P[m], 10):.2f}-{np.percentile(subsim_P[m], 90):.2f})",
                     f"{r[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]", f"{re_[0]:+.2f} [{lo_e[0]:+.2f}, {hi_e[0]:+.2f}]",
                     f"{np.median(groupsim_P[m]):.2f}", f"{r[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}]", f"{r[2]:+.2f} [{lo[2]:+.2f}, {hi[2]:+.2f}]"])
    rep.table(["pairs", "n", "fragment overlap: median (10-90%)", "r(fragment overlap, y)", "r(fragment overlap, B1a residual)",
               "group overlap: median", "r(group overlap, y)", "r(B1a distance, y)"], rows)
    rep.line("overlap = fragments present in both mixtures / fragments present in either (Tanimoto). r < 0 expected: more shared -> rated closer.")

    rep.section("2. Human labels: spread")
    rows = []
    for g, m in groups.items():
        q = np.percentile(y_P[m], [5, 25, 50, 75, 95])
        rows.append([g, int(m.sum()), f"{y_P[m].std():.3f}"] + [f"{v:.2f}" for v in q] + [f"{d_P[m].std():.3f}"])
    rep.table(["pairs", "n", "sd y", "y 5%", "25%", "50%", "75%", "95%", "sd B1a distance"], rows)

    def unit_section(title, X_all, names, top=12):
        rep.section(title)
        summary = []
        for g, m in groups.items():
            X = X_all[m]
            ok = X.std(0) > 0
            r, lo, hi, p = corr_table(X[:, ok], y_P[m], w_BxP[:, m])
            re_, lo_e, hi_e, pe = corr_table(X[:, ok], eps_P[m], w_BxP[:, m])
            q, qe = bh(p), bh(pe)
            nm = [n for n, k in zip(names, ok) if k]
            summary.append([g, len(nm), int((q < 0.1).sum()), int(((q < 0.1) & (r > 0)).sum()), int((qe < 0.1).sum()),
                            f"{np.median(np.abs(r)):.3f}"])
            o = np.argsort(p + 1e-9 * -np.abs(r))[:top]
            rep.line(f"{g}: top {top} by p (r with y [95% CI], q; r with B1a residual, q)")
            rep.table(["unit", "r with y", "q", "r with residual", "q"],
                      [[nm[j], f"{r[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]", f"{q[j]:.3f}", f"{re_[j]:+.3f}", f"{qe[j]:.3f}"] for j in o])
        rep.table(["pairs", "units tested", "q < 0.1 with y", "of which r > 0 (differ -> farther)", "q < 0.1 with B1a residual", "median |r| with y"], summary)

    unit_section("3. Odour words: |u_w(A) - u_w(B)| vs human rating", dW_PxW, words_W)
    unit_section("4a. Functional groups: |share of molecules with g in A - in B|", np.abs(gA - gB), gnames)
    unit_section("4b. Morgan fragments: |share of molecules with f in A - in B|", np.abs(fA - fB), fnames)
    rep.save()


if __name__ == "__main__":
    main()
