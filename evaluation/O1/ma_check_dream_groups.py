# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules   P = Ma mixtures (198)   G = RDKit groups (24)
#   K = classes (9)   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""Check A: do the dominance weights B1a+ learns on DREAM agree with Ma's measured dominance?

B1a+ (mixdist/b1a_plus.py, b1a+ session) fitted on all 820 DREAM training pairs, lam 1e-3, + size:
    log W(m) = sum_g theta_g x_g(m)      (x from structure only, so it applies to any molecule)
Pieces: RDKit fr_ groups carried by >= 5 of the 235 DREAM molecules, identical-carrier groups merged
(exactly as evaluation/B1aplus/eval_b1a_plus.py); reference: the 9 SMARTS classes.
A Ma molecule carries a merged group if it carries any of the merged raw groups.
Nothing is refitted on Ma: the DREAM weights are applied as they are.

Declared outputs:
  1. per mixture: predicted L = log W(m1) - log W(m2) vs Ma's L = log(v1/v2), all 198 mixtures:
     RMSE of the raw prediction (always 0 as reference) and Pearson r.
  2. per molecule: log W(m) vs Ma's molecule score s (results/ma_dominance, saved): all 72, the 41 with
     >= 4 Ma mixtures, the 39 also in DREAM.
  CIs: 95%, 200 draws, each = one DREAM mixture-resample refit (as eval_b1a_plus.py, seed 0) x one
  Ma molecule resample (seed 1), so both datasets' uncertainty is included.

Env:     mist
Usage:   python findings/ma_check_dream_groups.py
Writes:  results/ma_check_dream_groups/<date_time>.{txt,json}
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

N_BOOT = 200


def groups_with_ma(dream_smiles, ma_smiles):
    """Rebuild the merged, >= 5-carrier RDKit groups on DREAM; return DREAM and Ma membership."""
    raw_d = pp.units_rdkit_groups(dream_smiles); raw_m = pp.units_rdkit_groups(ma_smiles)
    by_unit = defaultdict(set)
    for mid, us in raw_d.carriers.items():
        for u in us:
            by_unit[u].add(mid)
    merged = defaultdict(list)
    for u in range(len(raw_d.names)):
        merged[frozenset(by_unit[u])].append(u)
    sets = [(mols, us) for mols, us in merged.items() if len(mols) >= 5]
    ids_d, ids_m = list(dream_smiles), list(ma_smiles)
    Xd = np.array([[float(i in mols) for mols, _ in sets] for i in ids_d])
    Xm = np.array([[float(any(u in raw_m.carriers.get(i, ()) for u in us)) for _, us in sets] for i in ids_m])
    names = [raw_d.names[us[0]] for _, us in sets]
    return Xd, Xm, names


def wr(x, y, w):
    w = w / w.sum(); xm, ym = w @ x, w @ y
    return (w @ ((x - xm) * (y - ym))) / np.sqrt((w @ (x - xm) ** 2) * (w @ (y - ym) ** 2))


def main():
    rep = Report("ma_check_dream_groups", {"n_boot": N_BOOT, "lam": 1e-3})
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]; dsmi = {i: str(s) for i, s in zip(ids, op.smiles_M)}
    q = mixture.quantile_per_word(op.probs_MxW); A, B = membership(train, op.cids_M); S = size_features(train)

    mol = pd.read_csv(ROOT / "external/ma_2021/pyrfume_molecules.csv").sort_values("CID").reset_index(drop=True)
    msmi = {f"ma{c}": s for c, s in zip(mol.CID, mol.IsomericSMILES)}
    Xg_d, Xg_m, gnames = groups_with_ma(dsmi, msmi)
    Xg_ref, gref = piece_matrix(pp.merge_identical(pp.units_rdkit_groups(dsmi)), ids)
    Xc_d, cnames = piece_matrix(pp.units_smarts(dsmi, pp.CLASSES), ids)
    Xc_m = np.array([[float(i in pp.units_smarts(msmi, pp.CLASSES).carriers.get(k, ())) for i in range(len(pp.CLASSES))]
                     for k in msmi])
    Xc_m = Xc_m[:, [list(pp.CLASSES).index(n) for n in cnames]]

    r = pd.read_csv(ROOT / "external/ma_2021/pyrfume_behavior.csv"); r = r[(r.Subject != 47) & (r.IA > 0) & (r.IB > 0)]
    g = r.groupby(["Stimulus A", "Stimulus B"])[["IA", "IAmix", "IB", "IBmix"]].mean().reset_index()
    L_P = np.log((g.IAmix / g.IA) / (g.IBmix / g.IB)).values
    ix = {c: i for i, c in enumerate(mol.CID)}; a_P, b_P = g["Stimulus A"].map(ix).values, g["Stimulus B"].map(ix).values
    f = sorted(glob.glob(str(ROOT / "results/ma_dominance/*.json")))[-1]
    smol = next(x["value"] for x in json.load(open(f))["items"] if x.get("key") == "molecules")
    s_by = {int(x["cid"]): x for x in smol}
    s_M = np.array([s_by[c]["s"] for c in mol.CID]); n_M = np.array([s_by[c]["n_mix"] for c in mol.CID])
    dream_M = np.isin(mol.CID.values, op.cids_M.astype(int))

    rep.section("0  Audit")
    rep.line(f"groups rebuilt: {len(gnames)}; identical to eval_b1a_plus.py on DREAM: {Xg_ref.shape == Xg_d.shape and np.array_equal(Xg_ref, Xg_d)}")
    rep.line(f"Ma molecules carrying >= 1 group: {(Xg_m.sum(1) > 0).sum()} of {len(mol)}; >= 1 class: {(Xc_m.sum(1) > 0).sum()}")
    rep.table(["group", "DREAM molecules", "Ma molecules"], [[n, int(Xg_d[:, j].sum()), int(Xg_m[:, j].sum())] for j, n in enumerate(gnames)])
    rep.line(f"molecule score s: {f}")

    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); im = {m: i for i, m in enumerate(mixes)}
    ia = np.array([im[p.a] for p in train]); ib = np.array([im[p.b] for p in train])
    cnt = np.random.default_rng(0).multinomial(len(mixes), np.full(len(mixes), 1 / len(mixes)), size=N_BOOT)
    cm = np.random.default_rng(1).multinomial(len(mol), np.full(len(mol), 1 / len(mol)), size=N_BOOT)
    subsets = [("all 72", np.ones(len(mol), bool)), (">= 4 Ma mixtures", n_M >= 4), ("also in DREAM", dream_M)]

    for tag, Xd, Xm in [("RDKit groups + size", Xg_d, Xg_m), ("classes + size (reference)", Xc_d, Xc_m)]:
        th = B1aPlus(q, Xd, A, B, S, 1e-3).fit(np.arange(len(train)), y).theta.numpy()
        thB = np.array([B1aPlus(q, Xd, A, B, S, 1e-3).fit(np.arange(len(train)), y,
                        w_pairs=(cnt[b, ia] * cnt[b, ib]).astype(float) + 1e-12).theta.numpy() for b in range(N_BOOT)])
        lw_M = Xm @ th; pred_P = lw_M[a_P] - lw_M[b_P]
        rep.section(f"{tag}: DREAM weights applied to Ma")
        rB = []
        for b in range(N_BOOT):
            lwb = Xm @ thB[b]; w = (cm[b, a_P] * cm[b, b_P]).astype(float)
            if (w > 0).sum() > 20: rB.append(wr(lwb[a_P] - lwb[b_P], L_P, w))
        rows = [["always 0", len(L_P), f"{np.sqrt(np.mean(L_P ** 2)):.4f}", "-"],
                [f"DREAM {tag}", len(L_P), f"{np.sqrt(np.mean((pred_P - L_P) ** 2)):.4f}",
                 f"{np.corrcoef(pred_P, L_P)[0, 1]:+.3f} [{np.quantile(rB, .025):+.3f}, {np.quantile(rB, .975):+.3f}]"]]
        rep.line("1. per mixture: predicted L = log W(m1) - log W(m2) vs Ma's L (raw DREAM scale, no refit)")
        rep.table(["model", "mixtures", "RMSE", "r [95% CI]"], rows)
        rep.line("2. per molecule: log W(m) vs Ma's molecule score s")
        rows, out = [], {}
        for sub, k in subsets:
            rb = [wr((Xm @ thB[b])[k], s_M[k], cm[b, k].astype(float)) for b in range(N_BOOT) if (cm[b, k] > 0).sum() > 8]
            rr = np.corrcoef(lw_M[k], s_M[k])[0, 1]; out[sub] = [rr, np.quantile(rb, .025), np.quantile(rb, .975)]
            rows.append([sub, int(k.sum()), f"{lw_M[k].std():.3f}", f"{rr:+.3f} [{out[sub][1]:+.3f}, {out[sub][2]:+.3f}]"])
        rep.table(["molecules", "n", "sd of log W", "r [95% CI]"], rows)
        rep.value(tag, {"theta": th.tolist(), "per_mixture_r": float(np.corrcoef(pred_P, L_P)[0, 1]), "per_molecule": out})
        if tag.startswith("RDKit"):
            o = np.argsort(th)
            rep.line("DREAM theta per group (most masked -> most dominant), with Ma's mean s of carriers:")
            rep.table(["group", "theta", "DREAM mol", "Ma mol", "Ma mean s of carriers"],
                      [[gnames[j], f"{th[j]:+.3f}", int(Xg_d[:, j].sum()), int(Xg_m[:, j].sum()),
                        f"{s_M[Xg_m[:, j] == 1].mean():+.3f}" if Xg_m[:, j].sum() else "-"] for j in o])
    rep.save()


if __name__ == "__main__":
    main()
