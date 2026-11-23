"""Pony: a small graph neural network that predicts how different two odour mixtures smell.

Paper: "Structure over Scale: Mixing Odours in a Pony Network like the Human Receptors".

    from pony import Pony
    model = Pony.from_pretrained()
    model.predict([["CCO", "CC(=O)OCC"]], [["CC(C)=O"]])
"""
from .features import canonical_smiles, featurize
from .model import Pony
from .network import PonyNet

__all__ = ["Pony", "PonyNet", "featurize", "canonical_smiles"]
__version__ = "1.0.0"
