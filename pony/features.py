"""Molecule graphs from SMILES, with exactly the atom and bond features of open-POM's ``GraphFeaturizer``.

Atom features (134): one-hot total valence (0-6), total degree (0-5), total hydrogens (0-4), formal charge
(-1, -2, 1, 2, 0), atomic number (1-100) and hybridisation (SP, SP2, SP3, SP3D, SP3D2), each with one extra slot for
"other". Bond features (6): a padding flag (always 0), single, double, triple, aromatic, in ring. Every bond appears
twice, once per direction. Atoms are put in RDKit's canonical order, as DeepChem does, so the same molecule always
gives the same graph. Only RDKit is needed (no DeepChem).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops

ATOM_DIM = 134
BOND_DIM = 6
_VALENCE = list(range(7))
_DEGREE = list(range(6))
_NUM_HS = list(range(5))
_CHARGE = [-1, -2, 1, 2, 0]
_ATOMIC_NUM = list(range(100))
_HYBRID = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]


def _one_hot(value, allowed) -> list[int]:
    out = [0] * (len(allowed) + 1)
    out[allowed.index(value) if value in allowed else len(allowed)] = 1
    return out


def atom_features(atom: Chem.Atom) -> list[int]:
    return (_one_hot(atom.GetTotalValence(), _VALENCE) + _one_hot(atom.GetTotalDegree(), _DEGREE)
            + _one_hot(atom.GetTotalNumHs(), _NUM_HS) + _one_hot(atom.GetFormalCharge(), _CHARGE)
            + _one_hot(atom.GetAtomicNum() - 1, _ATOMIC_NUM) + _one_hot(str(atom.GetHybridization()), _HYBRID))


def bond_features(bond: Chem.Bond) -> list[int]:
    t = bond.GetBondType()
    return [0, int(t == Chem.BondType.SINGLE), int(t == Chem.BondType.DOUBLE), int(t == Chem.BondType.TRIPLE),
            int(t == Chem.BondType.AROMATIC), int(bond.IsInRing())]


@dataclass
class MolGraph:
    """One molecule: ``node`` (atoms x 134), ``edge`` (directed bonds x 6), ``edge_index`` (2 x directed bonds)."""
    smiles: str
    node: np.ndarray
    edge: np.ndarray
    edge_index: np.ndarray

    @property
    def num_atoms(self) -> int:
        return len(self.node)


def canonical_smiles(smiles: str) -> str:
    """RDKit canonical SMILES; raises ValueError if RDKit cannot parse ``smiles``."""
    mol = _parse(smiles)
    return Chem.MolToSmiles(mol)


def _parse(smiles: str) -> Chem.Mol:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    RDLogger.EnableLog("rdApp.*")
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    return mol


def featurize(smiles: str) -> MolGraph:
    """SMILES -> MolGraph (heavy atoms only, canonical atom order)."""
    mol = _parse(smiles)
    mol = rdmolops.RenumberAtoms(mol, list(rdmolfiles.CanonicalRankAtoms(mol)))
    node = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    src, dst, edge = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_features(b)
        src += [i, j]; dst += [j, i]; edge += [f, f]
    edge_arr = np.asarray(edge, dtype=np.float32).reshape(-1, BOND_DIM)
    return MolGraph(smiles, node, edge_arr, np.asarray([src, dst], dtype=np.int64).reshape(2, -1))


def batch(graphs: list[MolGraph]):
    """Concatenate graphs -> (node, edge, src, dst, graph_of_atom) numpy arrays with global atom indices."""
    offs = np.cumsum([0] + [g.num_atoms for g in graphs])
    node = np.concatenate([g.node for g in graphs])
    edge = np.concatenate([g.edge for g in graphs]) if graphs else np.zeros((0, BOND_DIM), np.float32)
    src = np.concatenate([g.edge_index[0] + o for g, o in zip(graphs, offs)])
    dst = np.concatenate([g.edge_index[1] + o for g, o in zip(graphs, offs)])
    graph_of_atom = np.concatenate([np.full(g.num_atoms, k) for k, g in enumerate(graphs)])
    return node, edge, src, dst, graph_of_atom
