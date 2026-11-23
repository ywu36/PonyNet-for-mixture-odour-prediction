"""Compute per-molecule MIST odour features and cache them.

Runs the frozen MIST odour model (dream2_data/mist_odour, a SMILES transformer) on the 235
molecules: 135 odour-word probabilities (and logits) and a 512-d embedding per molecule.
Only embeds; checks are in check_mist.py.

Env:     mist (torch, transformers, rdkit; uses the GPU with most free memory, or OSMO_DEVICE)
Usage:   python evaluation/embed_mist.py
Writes:  features/mist.npz, results/embed_mist/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import features, mist  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    rep = Report("embed_mist")
    arrays = mist.compute()
    f = features.save("mist", arrays)
    rep.line(f"device {arrays['device']}")
    rep.line(features.load("mist").describe())
    rep.line(f"wrote {f}")
    rep.line("check it with: python evaluation/check_mist.py")
    rep.save()


if __name__ == "__main__":
    main()
