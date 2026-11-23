# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = molecules (235 DREAM + new)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Predict DREAM mixture-pair ratings (perceptual distance, the DREAM label scale) with packaged model (b)
(checkpoints/final_o5b, written by package_o5b.py; finding 6 sec. 14). Average over the 10 members.
Molecules: the 235 DREAM molecules by CID (stored vectors), or any other molecule given with its SMILES (new-molecule
path: open-POM's featurizer in the openpom env -> frozen O5 network -> the same scale and strength units as the stored
vectors; needs the per-member constants added by `package_o5b.py --constants-only`).

Env:     mist (the new-molecule path also calls ~/miniforge3/envs/openpom/bin/python for featurizing)
Usage:   python evaluation/O3/predict_o5b.py pairs.csv [out.csv] [--smiles smiles.csv]
         pairs.csv columns: mixture_a, mixture_b   (molecule ids separated by ';', e.g. "702;7410;8094")
         smiles.csv columns: id, smiles            (only for ids that are not DREAM CIDs)
As a library: predict(list_of_id_lists_a, list_of_id_lists_b, smiles={id: SMILES}) -> numpy array of ratings.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402
from mixdist import o2  # noqa: E402

PK = ROOT / "checkpoints/final_o5b"
OPENPOM_PY = Path.home() / "miniforge3/envs/openpom/bin/python"


class SmilesGraphs(o2.Graphs):
    """Graph file written by featurize_smiles.py, with o2.Graphs batching."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.ids = [str(i) for i in z["ids"]]
        self.node, self.edge, self.edge_index = z["node"], z["edge"], z["edge_index"]
        self.node_off, self.edge_off = z["node_off"], z["edge_off"]


def featurize(smiles):
    """{id: SMILES} -> SmilesGraphs (runs featurize_smiles.py in the openpom env)."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.txt", Path(d) / "out.npz"
        src.write_text("".join(f"{k} {v}\n" for k, v in smiles.items()))
        subprocess.run([str(OPENPOM_PY), "-I", str(ROOT / "evaluation/O3/featurize_smiles.py"), str(src), str(dst)], check=True)
        return SmilesGraphs(dst)


def new_vectors(G, seed, r, dev):
    """O5 member `seed` on new molecules -> (V, H, mol, s) in the stored units (hidden / scale; atom scores / strength SD)."""
    import train_o3m
    from eval_attpool import attpool_atoms
    n = train_o3m.load(ROOT / f"checkpoints/o3m_attpool/member_{seed}.pt", dev)
    with torch.no_grad():
        H, sc, mol, V = attpool_atoms(n, G, np.arange(len(G.ids)), dev)
    return V / r["scale"], H / r["scale"], mol, sc / r["strength_sd"]


def predict(mix_a, mix_b, smiles=None):
    ecd.DIRECTION_ONLY = True
    z = np.load(PK / "dream_vectors.npz"); cids = [str(int(c)) for c in z["cids"]]
    used = sorted({str(c) for m in list(mix_a) + list(mix_b) for c in m})
    new = [c for c in used if c not in set(cids)]
    miss = [c for c in new if c not in (smiles or {})]
    if miss:
        raise ValueError(f"ids not among the 235 DREAM CIDs and without SMILES: {miss}")
    G = featurize({c: smiles[c] for c in new}) if new else None
    names = cids + new; row = {c: i for i, c in enumerate(names)}
    A = np.zeros((len(mix_a), len(names))); B = np.zeros_like(A)
    for i, (a, b) in enumerate(zip(mix_a, mix_b)):
        A[i, [row[str(c)] for c in a]] = 1; B[i, [row[str(c)] for c in b]] = 1
    dev = None
    if new:
        from mixdist.device import pick_device
        dev = pick_device()
    preds = []
    for s in eo.SEEDS:
        r = torch.load(PK / f"readout_member_{s}.pt", weights_only=False)
        V, H, mol, sa = z[f"V_{s}"], z[f"H_{s}"], z[f"mol_{s}"], z[f"s_{s}"]
        if new:
            assert "scale" in r, "run package_o5b.py --constants-only first"
            Vn, Hn, moln, sn = new_vectors(G, s, r, dev)
            V, H, mol, sa = np.r_[V, Vn], np.r_[H, Hn], np.r_[mol, moln + len(cids)], np.r_[sa, sn]
        f = eo.PriorAtom(V, H, mol, A, B, sa, r["q0"].numpy(), "shrink")
        f.beta, f.delta, f.t, f.kappa, f.lin, f.b = r["beta"], r["delta"], r["t"], r["kappa"], r["lin"], torch.zeros((), dtype=eo.F64)
        with torch.no_grad():
            f.lw = f.smax(f.score(f.beta, f.delta), f.t)
        preds.append(f.predict(np.arange(len(A))))
    return np.mean(preds, 0)


def main():
    import pandas as pd
    args = sys.argv[1:]; smiles = None
    if "--smiles" in args:
        i = args.index("--smiles"); sm = pd.read_csv(args[i + 1]); smiles = dict(zip(sm["id"].astype(str), sm["smiles"])); del args[i:i + 2]
    df = pd.read_csv(args[0]); sp = lambda s: [c.strip() for c in str(s).split(";") if c.strip()]  # noqa: E731
    df["predicted_rating"] = predict([sp(s) for s in df.mixture_a], [sp(s) for s in df.mixture_b], smiles)
    out = args[1] if len(args) > 1 else None
    print(df.to_csv(index=False)) if out is None else df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
