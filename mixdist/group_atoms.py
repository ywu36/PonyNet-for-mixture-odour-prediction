# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = molecules in the graph file   V = atoms over all molecules
#   K = chemical groups kept   D = distance bins (0, 1, 2, 3, 4+ bonds)
# ---------------------------------------------------------------------------------------------
"""Per-atom group-distance counts (the user, 2026-11-19): for each atom, how many groups of each kind sit 0, 1, 2, 3
or 4+ bonds away. Groups = RDKit's 85 functional-group patterns (rdkit.Chem.Fragments, the fr_* counters), each match
counted once; an atom's distance to a group match = the shortest bond path to any atom of that match.
Groups kept: present in >= `min_frac` of the GS-LF TRAINING molecules (chosen without DREAM or held-out molecules).
Appended to open-POM's 134 atom features; the network learns what they mean. Nothing reaches the mixture side.

Atom order: open-POM's featurizer (DeepChem) renumbers atoms canonically: RenumberAtoms(mol, CanonicalRankAtoms(mol)).
Checked 2026-11-19 on 1,044 molecules: atoms and bonds match the graph file 1,044 / 1,044 (raw RDKit order: 3 / 1,044).
build() also checks every atom's element against the stored one-hot.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Fragments, rdmolfiles, rdmolops

BINS = 5   # 0, 1, 2, 3, 4+ bonds


def patterns():
    """[(name, SMARTS mol)] for RDKit's 85 fr_* groups, in file order."""
    out = []
    with open(Fragments.defaultPatternFileName) as f:
        for line in f:
            if line and line[0] != "#":
                s = line.rstrip("\n").split("\t")
                if len(s) >= 3:
                    out.append((s[0].replace("=", "_").replace("-", "_"), Chem.MolFromSmarts(s[2])))
    return out


ELEMENT_COL = 8 + 7 + 6 + 6   # open-POM atom features: valence 8, degree 7, hydrogens 6, charge 6, then element (Z - 1)


def counts_VxKD(smiles, n_atoms, pats, stored_VxF=None):
    """One molecule -> (n_atoms x K*BINS) counts; row a, column k*BINS + b = matches of group k at distance bin b."""
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(str(smiles))
    if m is not None:
        m = rdmolops.RenumberAtoms(m, list(rdmolfiles.CanonicalRankAtoms(m)))
    if m is None or m.GetNumAtoms() != n_atoms:
        raise ValueError(f"atom order mismatch for {smiles}: RDKit {None if m is None else m.GetNumAtoms()}, graph {n_atoms}")
    if stored_VxF is not None:
        z = np.array([a.GetAtomicNum() - 1 for a in m.GetAtoms()])
        if not np.array_equal(z, np.argmax(stored_VxF[:, ELEMENT_COL:ELEMENT_COL + 101], 1)):
            raise ValueError(f"element order mismatch for {smiles}")
    dist = Chem.GetDistanceMatrix(m) if n_atoms > 1 else np.zeros((1, 1))
    out = np.zeros((n_atoms, len(pats) * BINS), np.float32)
    for k, (_, p) in enumerate(pats):
        for match in m.GetSubstructMatches(p, uniquify=True):
            d = dist[:, list(match)].min(1)
            b = np.minimum(d, BINS - 1).astype(int)
            out[np.arange(n_atoms), k * BINS + b] += 1
    return out


def build(G, train_ids, min_frac=0.01):
    """-> (extra_VxF aligned with G.node, kept group names). Groups kept by presence in the training molecules only."""
    pats = patterns(); n_atoms = np.diff(G.node_off)
    per_mol = [counts_VxKD(G.smiles_N[i], int(n_atoms[i]), pats, G.node[G.node_off[i]:G.node_off[i + 1]]) for i in range(len(G.smiles_N))]
    present_N = np.array([(c.reshape(len(c), len(pats), BINS)[:, :, 0].sum(0) > 0) for c in per_mol])   # group occurs
    keep = np.where(present_N[train_ids].mean(0) >= min_frac)[0]
    cols = np.concatenate([np.arange(k * BINS, (k + 1) * BINS) for k in keep])
    extra = np.concatenate([c[:, cols] for c in per_mol])
    assert len(extra) == len(G.node)
    return extra, [pats[k][0] for k in keep]


def attach(G, train_ids, min_frac=0.01):
    """Appends the counts to G.node in place (log1p-scaled); returns the kept group names."""
    extra, names = build(G, train_ids, min_frac)
    G.node = np.concatenate([G.node, np.log1p(extra)], 1).astype(np.float32)
    return names
