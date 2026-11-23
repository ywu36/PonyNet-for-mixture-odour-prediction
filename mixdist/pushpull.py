# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   P = pairs analysed   X = mixtures in them   E = experiments   M = molecules   U = units
#   B = bootstrap draws
# ---------------------------------------------------------------------------------------------
"""Push/pull of sub-molecular units between two odour mixtures: a standalone analysis tool.

Question: when two mixtures are compared by people, which units (substructures, chemical
classes, odour words, ...) pull the pair together when both mixtures carry them, and which push
it apart when only one does, beyond what each mixture contributes on its own?

Model, fitted separately for every unit u on the chosen pairs:
    y(A, B) = experiment offset + m_A + m_B + gamma * f(u)
  m_X        one number per mixture (ridge, penalty ``alpha``): anything that is a property of a
             single mixture (size, intensity, "far from everything", and "carries u"), so gamma is
             about the PAIR only.
  presence   f = 1 if both mixtures carry u
  graded     s_X(u) = share of X's molecules carrying u; f = min(s_A, s_B)
  gamma < 0: sharing u pulls the pair closer than the two mixtures' own effects predict; gamma > 0:
             sharing u pushes it apart.
Control for overall overlap (default control="overlap"): pairs that share more structure overall
are rated closer, so almost every unit would show gamma < 0 for that reason alone. The model adds
o(A, B) = Jaccard of the two mixtures' unit sets (all units, u included: under "every shared unit
counts the same", u's share of o is exactly what it should contribute), so gamma is the effect of u
beyond its fair share of general overlap. The general effect itself is reported once (``general_sharing``).
Why one number, not a separate pull and push: "exactly one carries u" = inA + inB - 2 * both, and
inA, inB are single-mixture properties absorbed by m. So "one-sided pushes by x" and "shared pulls by
2x" fit identically (same for graded: |s_A - s_B| = s_A + s_B - 2 min). Pull vs push of the same unit
is not identifiable once mixture effects are removed; only their contrast, gamma, is.
Uncertainty: 95% CI and two-sided p by resampling MIXTURES with replacement (a pair's weight is
count(A) x count(B)), because pairs that share a mixture are not independent. Benjamini-Hochberg
q over the units tested. Every unit reports its support (pairs in each state) and the raw state
table: mean of y minus offset and mixture effects, for neither / one / both, before any unit term.

Settings:
  no-shared  pairs whose mixtures share no molecule (default)
  shared     pairs sharing >= 1 molecule; units are counted only on the molecules the two mixtures
             do NOT share, so a shared molecule cannot fake a pull
  all        every pair (units counted on all molecules)
Optional discovery / confirmation: mixtures split at random into two halves; units are ranked on
pairs inside half 1 and the top ones re-tested on pairs inside half 2 (pairs across halves unused).

Inputs (CLI):
  --pairs      CSV: mixture_a, mixture_b (molecule ids separated by ';'), rating, [experiment]
  --molecules  CSV: id, smiles
  --units      morgan:R1-R2 (unhashed Morgan fragments, radius range, e.g. morgan:1-3) |
               smarts:FILE (CSV name,smarts) | classes (built-in 9 chemical classes: molecules matching one SMARTS pattern, e.g. phenol = OH on an aromatic ring) |
               rdkit-groups (RDKit fr_* functional groups) | table:FILE (CSV id,unit1,unit2,... 0/1)
Example:
  python -m mixdist.pushpull --pairs pairs.csv --molecules molecules.csv --units morgan:1-2 \
      --setting no-shared --mode graded --out results/pushpull_run
Python:
  from mixdist import pushpull
  units = pushpull.units_morgan(smiles_by_id, radii=(1, 2))
  table = pushpull.analyse(pushpull.Pairs(a, b, y, exp), units, setting="no-shared", mode="graded")
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSES = {"phenol": "c[OX2H]", "sulfur": "[#16]", "ester": "[CX3](=O)[OX2][#6]", "aldehyde": "[CX3H1](=O)[#6]",
            "ketone": "[#6][CX3](=O)[#6]", "alcohol": "[CX4][OX2H]", "acid": "[CX3](=O)[OX2H1]",
            "aromatic N": "[n]", "alkene": "[C]=[C]"}


# ----------------------------------------------------------------------------- data
@dataclass
class Pairs:
    """Rated mixture pairs. a, b: molecule-id collections per pair; y: rating (larger = more
    different); experiment: label per pair (one offset each), optional."""
    a: list
    b: list
    y: np.ndarray
    experiment: list = None

    def __post_init__(self):
        self.a = [frozenset(x) for x in self.a]; self.b = [frozenset(x) for x in self.b]
        self.y = np.asarray(self.y, float)
        if self.experiment is None:
            self.experiment = ["all"] * len(self.y)

    def subset(self, idx):
        return Pairs([self.a[i] for i in idx], [self.b[i] for i in idx], self.y[idx], [self.experiment[i] for i in idx])


@dataclass
class Units:
    """Which molecule carries which unit: names (U,), carriers {molecule id: set of unit indices}."""
    names: list
    carriers: dict
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- units
def _fragment_name(m, atom, env):
    """Fragment SMILES with hydrogen counts on every atom, written from the centre atom (first)."""
    from rdkit import Chem
    atoms = {atom} | {x for bd in env for x in (m.GetBondWithIdx(bd).GetBeginAtomIdx(), m.GetBondWithIdx(bd).GetEndAtomIdx())}
    sym = []
    for a in m.GetAtoms():
        s = a.GetSymbol().lower() if a.GetIsAromatic() else a.GetSymbol()
        h = a.GetTotalNumHs()
        sym.append(f"[{s}{'H' + str(h) if h > 1 else 'H' if h == 1 else ''}]")
    return Chem.MolFragmentToSmiles(m, atomsToUse=sorted(atoms), bondsToUse=list(env) or None, atomSymbols=sym,
                                    rootedAtAtom=atom, canonical=False)


def units_morgan(smiles_by_id: dict, radii=(1, 2), min_molecules=5, restrict_to=None, detailed_names=False) -> Units:
    """Unhashed Morgan fragments of the given radii (an atom and everything within r bonds),
    kept if carried by >= min_molecules molecules (of ``restrict_to`` if given); fragments carried
    by exactly the same molecules are merged into one unit. ``detailed_names``: names carry the
    hydrogen count of every atom, centre atom first (short names can collide)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    carriers, name = defaultdict(set), {}
    for mid, s in smiles_by_id.items():
        m = Chem.MolFromSmiles(s); bi = {}
        AllChem.GetMorganFingerprint(m, max(radii), bitInfo=bi)
        for k, envs in bi.items():
            for atom, r in envs:
                if r not in radii:
                    continue
                carriers[(r, k)].add(mid)
                if (r, k) not in name:
                    env = Chem.FindAtomEnvironmentOfRadiusN(m, r, atom)
                    if detailed_names:
                        name[(r, k)] = _fragment_name(m, atom, env)
                    else:
                        name[(r, k)] = Chem.MolToSmiles(Chem.PathToSubmol(m, env)) if env else m.GetAtomWithIdx(atom).GetSymbol()
    pool = set(smiles_by_id) if restrict_to is None else set(restrict_to)
    groups = defaultdict(list)
    for key, mols in carriers.items():
        if len(mols & pool) >= min_molecules:
            groups[frozenset(mols)].append(key)
    names, carry = [], defaultdict(set)
    for u, (mols, keys) in enumerate(sorted(groups.items(), key=lambda kv: sorted(kv[1])[0])):
        keys = sorted(keys)
        names.append(f"{name[keys[0]]} (r{keys[0][0]}" + (f", +{len(keys) - 1} same carriers" if len(keys) > 1 else "") + ")")
        for mid in mols:
            carry[mid].add(u)
    return Units(names, dict(carry), {"kind": "morgan", "radii": list(radii), "min_molecules": min_molecules})


def units_smarts(smiles_by_id: dict, patterns: dict) -> Units:
    from rdkit import Chem
    qs = {n: Chem.MolFromSmarts(s) for n, s in patterns.items()}
    names = list(qs); carry = defaultdict(set)
    for mid, s in smiles_by_id.items():
        m = Chem.MolFromSmiles(s)
        for u, n in enumerate(names):
            if m.HasSubstructMatch(qs[n]):
                carry[mid].add(u)
    return Units(names, dict(carry), {"kind": "smarts", "patterns": patterns})


def units_rdkit_groups(smiles_by_id: dict) -> Units:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    fns = [(n.removeprefix("fr_"), f) for n, f in Descriptors.descList if n.startswith("fr_")]
    carry = defaultdict(set)
    for mid, s in smiles_by_id.items():
        m = Chem.MolFromSmiles(s)
        for u, (_, f) in enumerate(fns):
            if f(m) > 0:
                carry[mid].add(u)
    return Units([n for n, _ in fns], dict(carry), {"kind": "rdkit-groups"})


def units_table(names: list, has_by_id: dict) -> Units:
    """Any 0/1 table, e.g. odour words: has_by_id {molecule id: iterable of 0/1 per name}."""
    return Units(list(names), {mid: {u for u, v in enumerate(row) if v} for mid, row in has_by_id.items()}, {"kind": "table"})


def merge_identical(units: Units) -> Units:
    """Units carried by exactly the same molecules cannot be told apart: merge them into one
    (named by the first, with the others listed), so they are tested and corrected for once."""
    by_unit = defaultdict(set)
    for mid, us in units.carriers.items():
        for u in us:
            by_unit[u].add(mid)
    groups = defaultdict(list)
    for u in range(len(units.names)):
        groups[frozenset(by_unit[u])].append(u)
    names, carry = [], defaultdict(set)
    for new, (mols, us) in enumerate(groups.items()):
        names.append(units.names[us[0]] + (f" (= {', '.join(units.names[v] for v in us[1:])})" if len(us) > 1 else ""))
        for mid in mols:
            carry[mid].add(new)
    return Units(names, dict(carry), {**units.info, "merged_from": len(units.names)})


# ----------------------------------------------------------------------------- core
def _bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def _select(pairs: Pairs, setting: str):
    shared = np.array([len(a & b) > 0 for a, b in zip(pairs.a, pairs.b)])
    if setting == "no-shared":
        return np.where(~shared)[0]
    if setting == "shared":
        return np.where(shared)[0]
    if setting == "all":
        return np.arange(len(pairs.y))
    raise ValueError(f"setting must be no-shared, shared or all, not {setting!r}")


def _design(pairs: Pairs):
    exps = sorted(set(pairs.experiment)); mixes = sorted(set(pairs.a) | set(pairs.b), key=sorted)
    ie, im = {e: i for i, e in enumerate(exps)}, {m: i for i, m in enumerate(mixes)}
    n = len(pairs.y)
    E = np.zeros((n, len(exps))); X = np.zeros((n, len(mixes)))
    for r in range(n):
        E[r, ie[pairs.experiment[r]]] = 1; X[r, im[pairs.a[r]]] += 1; X[r, im[pairs.b[r]]] += 1
    ia = np.array([im[x] for x in pairs.a]); ib = np.array([im[x] for x in pairs.b])
    return E, X, ia, ib, len(mixes)


def _solve(D, y, w, pen):
    """Weighted ridge with per-column penalty ``pen`` (0 = unpenalised) and a free intercept."""
    sw = w / w.sum()
    Dm, ym = sw @ D, sw @ y
    Dc, yc = D - Dm, y - ym
    A = (Dc * w[:, None]).T @ Dc + np.diag(pen)
    try:
        return np.linalg.solve(A, (Dc * w[:, None]).T @ yc)
    except np.linalg.LinAlgError:                         # a bootstrap draw left the unit column empty
        return np.full(D.shape[1], np.nan)


def _shares(pairs: Pairs, units: Units, exclude_shared: bool):
    U = len(units.names)
    def share(mix, other):
        mols = [m for m in mix if not (exclude_shared and m in other)]
        v = np.zeros(U)
        for m in mols:
            for u in units.carriers.get(m, ()):
                v[u] += 1
        return v / max(len(mols), 1), len(mols)
    sA, sB = zip(*[(share(a, b)[0], share(b, a)[0]) for a, b in zip(pairs.a, pairs.b)])
    return np.array(sA), np.array(sB)


def _prepare(pairs, units, setting, alpha, n_boot, seed):
    P = pairs.subset(_select(pairs, setting))
    E, X, ia, ib, n_mix = _design(P)
    sA, sB = _shares(P, units, exclude_shared=(setting == "shared"))
    base = np.hstack([E, X]); pen_base = np.full(base.shape[1], alpha)
    c = np.random.default_rng(seed).multinomial(n_mix, np.full(n_mix, 1 / n_mix), size=n_boot)
    return P, sA, sB, base, pen_base, (c[:, ia] * c[:, ib]).astype(float)


def _fit_last(D, y, pen, w_BxP, n_boot):
    ones = np.ones(len(y))
    g = _solve(D, y, ones, pen)[-1]
    gg = np.array([_solve(D, y, w, pen)[-1] if w.sum() > 0 else np.nan for w in w_BxP])
    lo, hi = np.nanpercentile(gg, [2.5, 97.5])
    p = max(min(1, 2 * min(np.nanmean(gg <= 0), np.nanmean(gg >= 0))), 1 / n_boot)
    return float(g), float(lo), float(hi), float(p)


def general_sharing(pairs: Pairs, units: Units, setting="no-shared", alpha=1.0, n_boot=500, seed=0) -> dict:
    """The general effect: y = offset + m_A + m_B + beta * o, o = Jaccard of the two mixtures' unit sets."""
    P, sA, sB, base, pen_base, w_BxP = _prepare(pairs, units, setting, alpha, n_boot, seed)
    inA, inB = sA > 0, sB > 0
    o = (inA & inB).sum(1) / np.maximum((inA | inB).sum(1), 1)
    b, lo, hi, p = _fit_last(np.hstack([base, o[:, None]]), P.y, np.r_[pen_base, 0], w_BxP, n_boot)
    return {"pairs": len(P.y), "overlap_mean": float(o.mean()), "beta": b, "lo": lo, "hi": hi, "p": p}


def analyse(pairs: Pairs, units: Units, setting="no-shared", mode="presence", control="overlap", alpha=1.0,
            n_boot=500, min_support=10, seed=0) -> list[dict]:
    """Per-unit sharing-effect table (list of dicts, sorted by p). Tested if >= min_support pairs
    have u on both sides and >= min_support on exactly one side."""
    P, sA, sB, base, pen_base, w_BxP = _prepare(pairs, units, setting, alpha, n_boot, seed)
    if mode == "graded":
        f = np.minimum(sA, sB)
    elif mode == "presence":
        f = ((sA > 0) & (sB > 0)).astype(float)
    else:
        raise ValueError("mode must be graded or presence")
    if control not in ("overlap", "none"):
        raise ValueError("control must be overlap or none")
    inA, inB = sA > 0, sB > 0
    state = inA.astype(int) + inB.astype(int)                     # 0 neither, 1 one, 2 both
    o = (inA & inB).sum(1) / np.maximum((inA | inB).sum(1), 1)
    ones = np.ones(len(P.y))
    resid = P.y - (base - ones @ base / len(ones)) @ _solve(base, P.y, ones, pen_base) - P.y.mean()
    rows = []
    for u, name in enumerate(units.names):
        n_state = [int((state[:, u] == k).sum()) for k in range(3)]
        if n_state[2] < min_support or n_state[1] < min_support:
            continue
        if control == "overlap":
            D = np.hstack([base, o[:, None], f[:, [u]]]); pen = np.r_[pen_base, 0, 0]
        else:
            D = np.hstack([base, f[:, [u]]]); pen = np.r_[pen_base, 0]
        g, lo, hi, p = _fit_last(D, P.y, pen, w_BxP, n_boot)
        raw = [float(resid[state[:, u] == k].mean()) if n_state[k] else float("nan") for k in range(3)]
        rows.append({"unit": name, "pairs_neither": n_state[0], "pairs_one": n_state[1], "pairs_both": n_state[2],
                     "raw_neither": raw[0], "raw_one": raw[1], "raw_both": raw[2],
                     "gamma": g, "lo": lo, "hi": hi, "p": p})
    if rows:
        for r, q in zip(rows, _bh([r["p"] for r in rows])):
            r["q"] = float(q)
    rows.sort(key=lambda r: r["p"])
    return rows


def discover_confirm(pairs: Pairs, units: Units, top=5, seed=0, **kw) -> dict:
    """Rank units on pairs inside one random half of the mixtures, re-test the top ``top`` per sign
    (gamma < 0 sharing pulls, gamma > 0 sharing pushes) on pairs inside the other half."""
    mixes = sorted(set(pairs.a) | set(pairs.b), key=sorted)
    half = dict(zip(mixes, np.random.default_rng(seed).permutation(len(mixes)) % 2))
    h = [half[a] if half[a] == half[b] else -1 for a, b in zip(pairs.a, pairs.b)]
    d_idx = [i for i, x in enumerate(h) if x == 0]; c_idx = [i for i, x in enumerate(h) if x == 1]
    disc = analyse(pairs.subset(d_idx), units, **kw)
    picks = {}
    for k, sign in (("pull", -1), ("push", +1)):
        picks[k] = [r["unit"] for r in disc if np.sign(r["gamma"]) == sign][:top]
    keep = sorted(set(picks["pull"]) | set(picks["push"]))
    sub = Units([n for n in units.names if n in keep], {}, units.info)
    pos = {n: i for i, n in enumerate(sub.names)}; old = {i: n for i, n in enumerate(units.names)}
    sub.carriers = {m: {pos[old[u]] for u in us if old[u] in pos} for m, us in units.carriers.items()}
    conf = analyse(pairs.subset(c_idx), sub, **kw)
    return {"pairs_discovery": len(d_idx), "pairs_confirmation": len(c_idx), "picked": picks, "discovery": disc, "confirmation": conf}


# ----------------------------------------------------------------------------- output
def format_table(rows: list[dict], top=15) -> str:
    head = f"{'unit':40s} {'pairs n/1/b':>12s}  {'raw neither/one/both':>24s}  {'gamma [95% CI]':>26s} {'q':>5s}"
    lines = [head, "-" * len(head)]
    for r in rows[:top]:
        lines.append(f"{r['unit'][:40]:40s} {r['pairs_neither']:>3d}/{r['pairs_one']:>3d}/{r['pairs_both']:>3d}  "
                     f"{r['raw_neither']:+.3f}/{r['raw_one']:+.3f}/{r['raw_both']:+.3f}  "
                     f"{r['gamma']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {r['q']:5.2f}")
    return "\n".join(lines)


def _read_inputs(pairs_csv, molecules_csv):
    with open(molecules_csv) as f:
        smiles = {r["id"]: r["smiles"] for r in csv.DictReader(f)}
    a, b, y, e = [], [], [], []
    with open(pairs_csv) as f:
        for r in csv.DictReader(f):
            a.append(r["mixture_a"].split(";")); b.append(r["mixture_b"].split(";"))
            y.append(float(r["rating"])); e.append(r.get("experiment") or "all")
    return Pairs(a, b, y, e), smiles


def _units_from_arg(arg, smiles, pool):
    if arg.startswith("morgan:"):
        lo, hi = map(int, arg.split(":")[1].split("-"))
        return units_morgan(smiles, radii=tuple(range(lo, hi + 1)), restrict_to=pool)
    if arg == "classes":
        return units_smarts(smiles, CLASSES)
    if arg.startswith("smarts:"):
        with open(arg.split(":", 1)[1]) as f:
            return units_smarts(smiles, {r["name"]: r["smarts"] for r in csv.DictReader(f)})
    if arg == "rdkit-groups":
        return units_rdkit_groups(smiles)
    if arg.startswith("table:"):
        with open(arg.split(":", 1)[1]) as f:
            rd = csv.reader(f); hdr = next(rd)
            return units_table(hdr[1:], {r[0]: [int(float(v)) for v in r[1:]] for r in rd})
    raise ValueError(f"unknown --units {arg!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sharing effect (pull/push) of sub-molecular units between odour mixtures")
    ap.add_argument("--pairs", required=True); ap.add_argument("--molecules", required=True)
    ap.add_argument("--units", default="morgan:1-2")
    ap.add_argument("--setting", default="no-shared", choices=["no-shared", "shared", "all"])
    ap.add_argument("--mode", default="presence", choices=["presence", "graded"])
    ap.add_argument("--control", default="overlap", choices=["overlap", "none"],
                    help="overlap: gamma is the unit's effect beyond the pair's overall unit overlap")
    ap.add_argument("--n-boot", type=int, default=500); ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--confirm", action="store_true", help="discovery on one half of the mixtures, confirmation on the other")
    ap.add_argument("--out", default=None, help="write <out>.json and <out>.csv")
    args = ap.parse_args(argv)
    pairs, smiles = _read_inputs(args.pairs, args.molecules)
    pool = set().union(*pairs.a, *pairs.b)
    units = merge_identical(_units_from_arg(args.units, smiles, pool))
    kw = dict(setting=args.setting, mode=args.mode, control=args.control, n_boot=args.n_boot, min_support=args.min_support)
    gen = general_sharing(pairs, units, setting=args.setting, n_boot=args.n_boot)
    print(f"general sharing ({gen['pairs']} pairs): rating change from no shared units to all shared (Jaccard 0 -> 1) "
          f"{gen['beta']:+.3f} [{gen['lo']:+.3f}, {gen['hi']:+.3f}], p {gen['p']:.3f}; mean overlap {gen['overlap_mean']:.2f}")
    if args.confirm:
        out = {"general": gen, **discover_confirm(pairs, units, **kw)}
        print(f"discovery pairs {out['pairs_discovery']}, confirmation pairs {out['pairs_confirmation']}; picked {out['picked']}")
        print(format_table(out["confirmation"], top=50)); rows = out["confirmation"]
    else:
        rows = analyse(pairs, units, **kw); out = {"general": gen, "rows": rows}
        print(f"{units.info.get('merged_from', len(units.names))} units -> {len(units.names)} after merging identical carriers, "
              f"{len(rows)} tested ({args.setting}, {args.mode}); {sum(r['q'] < 0.1 for r in rows)} with q < 0.1")
        print(format_table(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"args": vars(args), "units_info": units.info, **out}, open(f"{args.out}.json", "w"), indent=1, default=str)
        if rows:
            with open(f"{args.out}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
