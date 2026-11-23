"""``Pony``: the full model of the paper (frozen network copies + fitted mixture readout), ready to use.

    from pony import Pony
    model = Pony.from_pretrained()                      # 10 copies, readout fitted on the DREAM training pairs
    model.predict([["CCO", "CC(=O)OCC"]], [["CC(C)=O"]])   # perceived difference, 0 (same) .. 1 (maximally different)

Every copy predicts on its own and the ratings are averaged over the copies, as in the paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from . import features as fz
from . import readout as ro
from .network import PonyNet

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "pony_dream.pt"
Mixture = Sequence[str]


class Member:
    """One trained copy: frozen network, its constants (scale, own atom scorer u0) and its fitted readout."""

    def __init__(self, arch: dict, state_dict: dict, scale: float, u0, readout: ro.ReadoutParams | None, device="cpu"):
        self.arch, self.scale, self.u0, self.readout = dict(arch), float(scale), np.asarray(u0, float), readout
        self.net = PonyNet(**self.arch); self.net.load_state_dict(state_dict); self.net.float().to(device).eval(); self.device = device

    @torch.no_grad()
    def encode(self, graphs: list[fz.MolGraph], chunk: int = 256):
        """-> V (molecules x 64), H (atoms x 64), mol (atom -> molecule), word logits (molecules x 138); V, H divided by scale."""
        Vs, Hs, mols, logits, base = [], [], [], [], 0
        for k in range(0, len(graphs), chunk):
            g = graphs[k:k + chunk]; x, e, s, d, gid = fz.batch(g)
            t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device)  # noqa: E731
            H, _, v, lg = self.net(t(x, torch.float32), t(e, torch.float32), t(s, torch.long), t(d, torch.long), t(gid, torch.long), len(g))
            Vs.append(v.double().cpu().numpy()); Hs.append(H.double().cpu().numpy()); mols.append(gid + base); logits.append(lg.cpu().numpy())
            base += len(g)
        return (np.concatenate(Vs) / self.scale, np.concatenate(Hs) / self.scale, np.concatenate(mols), np.concatenate(logits))

    def table(self, graphs):
        V, H, mol, _ = self.encode(graphs)
        return ro.MoleculeTable(V, H, mol, self.u0)

    def state(self):
        return {"arch": self.arch, "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()}, "scale": self.scale,
                "u0": self.u0.tolist(), "readout": None if self.readout is None else self.readout.to_dict()}


class Pony:
    def __init__(self, members: list[Member], words: list[str], meta: dict | None = None):
        self.members, self.words, self.meta = members, list(words), dict(meta or {})
        self._graphs: dict[str, fz.MolGraph] = {}

    # ---------------------------------------------------------------- loading and saving
    @classmethod
    def from_pretrained(cls, path: str | Path | None = None, device: str = "cpu") -> "Pony":
        """Load a saved model (default: the paper's model, shipped with the package)."""
        ck = torch.load(Path(path or DEFAULT_WEIGHTS), map_location="cpu", weights_only=True)
        members = [Member(m["arch"], m["state_dict"], m["scale"], m["u0"],
                          None if m["readout"] is None else ro.ReadoutParams.from_dict(m["readout"]), device) for m in ck["members"]]
        return cls(members, ck["words"], ck.get("meta"))

    @classmethod
    def from_networks(cls, nets: list[PonyNet], words: Sequence[str], train_smiles: Sequence[str],
                      reference_smiles: Sequence[str], device: str = "cpu", meta: dict | None = None) -> "Pony":
        """Build a model from Stage-1 networks (``pony.train.train_network``); the readout is fitted next with ``fit``.
        train_smiles: the networks' training molecules (for the hidden scale); reference_smiles: the molecules your
        mixtures are made of (for the own atom scorer u0; the paper uses the 235 DREAM molecules). See pony.train.calibrate."""
        from .train import calibrate
        members = []
        for net in nets:
            scale, u0 = calibrate(net, train_smiles, reference_smiles)
            members.append(Member(net.arch, {k: v.cpu() for k, v in net.state_dict().items()}, scale, u0, None, device))
        return cls(members, words, meta)

    def save(self, path: str | Path):
        torch.save({"members": [m.state() for m in self.members], "words": self.words, "meta": self.meta}, Path(path))

    # ---------------------------------------------------------------- molecules
    def _graph(self, smiles: str) -> fz.MolGraph:
        key = fz.canonical_smiles(smiles)
        if key not in self._graphs:
            self._graphs[key] = fz.featurize(smiles)
        return self._graphs[key]

    def _index(self, mixtures: Iterable[Mixture]):
        """Unique molecules (canonical SMILES) of all mixtures and a membership matrix (mixtures x molecules)."""
        mixtures = [list(m) for m in mixtures]
        for m in mixtures:
            if len(m) == 0:
                raise ValueError("a mixture must contain at least one molecule")
        keys = [[fz.canonical_smiles(s) for s in m] for m in mixtures]
        uniq = sorted({k for m in keys for k in m}); row = {k: i for i, k in enumerate(uniq)}
        X = np.zeros((len(mixtures), len(uniq)))
        for i, m in enumerate(keys):
            X[i, [row[k] for k in m]] = 1.0
        return uniq, X

    def odour_words(self, smiles: Sequence[str]):
        """Predicted probability of each of the 138 odour words for single molecules, averaged over the copies.
        Returns (molecules x 138 array, list of words)."""
        graphs = [self._graph(s) for s in smiles]
        P = np.mean([1 / (1 + np.exp(-m.encode(graphs)[3].astype(float))) for m in self.members], 0)
        return P, self.words

    # ---------------------------------------------------------------- mixtures
    def embed(self, mixtures: Sequence[Mixture]) -> np.ndarray:
        """Each mixture as a point in the hidden layer, one per copy (copies x mixtures x 64). For copy k,
        rating(A, B) = a_k |point_k(A) - point_k(B)| + c_k; ``predict`` averages this over the copies."""
        self._require_readout()
        uniq, X = self._index(mixtures); graphs = [self._graph(s) for s in uniq]
        return np.stack([ro.points(m.table(graphs), X, m.readout) for m in self.members])

    def line(self) -> np.ndarray:
        """(a_k, c_k) of every copy: rating = a_k x distance + c_k (copies x 2)."""
        self._require_readout()
        return np.array([[m.readout.a, m.readout.c] for m in self.members])

    def predict(self, mixtures_a: Sequence[Mixture], mixtures_b: Sequence[Mixture]) -> np.ndarray:
        """Predicted perceived difference for each pair (mixtures_a[i], mixtures_b[i]); each mixture is a list of SMILES.
        Ratings are on the DREAM scale, 0 (indistinguishable) to 1 (maximally different)."""
        if len(mixtures_a) != len(mixtures_b):
            raise ValueError("mixtures_a and mixtures_b must have the same length")
        self._require_readout()
        uniq, X = self._index(list(mixtures_a) + list(mixtures_b)); n = len(mixtures_a)
        graphs = [self._graph(s) for s in uniq]
        return np.mean([ro.predict(m.table(graphs), X[:n], X[n:], m.readout) for m in self.members], 0)

    def fit(self, mixtures_a: Sequence[Mixture], mixtures_b: Sequence[Mixture], ratings, lam: float = ro.LAM) -> "Pony":
        """Stage 2: refit the 133 readout numbers of every copy on rated pairs (the networks stay frozen)."""
        y = np.asarray(ratings, float)
        if not (len(mixtures_a) == len(mixtures_b) == len(y)):
            raise ValueError("mixtures_a, mixtures_b and ratings must have the same length")
        uniq, X = self._index(list(mixtures_a) + list(mixtures_b)); n = len(mixtures_a)
        graphs = [self._graph(s) for s in uniq]
        for m in self.members:
            m.readout = ro.fit(m.table(graphs), X[:n], X[n:], y, lam=lam)
        return self

    def explain(self, mixture: Mixture) -> dict:
        """How the molecules of one mixture are weighted: share pi_A(m) of each molecule (averaged over the copies) and
        the mixture length g(A) per copy."""
        self._require_readout()
        uniq, X = self._index([mixture]); graphs = [self._graph(s) for s in uniq]; shares, lengths = [], []
        for m in self.members:
            tab = m.table(graphs); _, beta, kappa, t, delta, gamma = m.readout.tensors(tab.V.shape[1])
            Xt = torch.as_tensor(X, dtype=ro.F64)
            with torch.no_grad():
                _, pi, g = ro.mixture_points(tab, Xt, ro.context(tab, Xt), ro.strengths(tab, beta, delta, t), kappa, gamma)
            shares.append(pi[0].numpy()); lengths.append(g.item())
        return {"molecules": uniq, "shares": np.mean(shares, 0), "length_per_copy": np.array(lengths)}

    def _require_readout(self):
        if any(m.readout is None for m in self.members):
            raise RuntimeError("this model has no fitted readout; call fit(...) first")

    def __repr__(self):
        n = sum(p.numel() for p in self.members[0].net.parameters())
        return f"Pony({len(self.members)} copies, {n:,} network parameters each, readout {'fitted' if self.members[0].readout else 'not fitted'})"
