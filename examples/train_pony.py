"""Train a Pony model on your own data, from files to a saved model (README, "Train Pony on your own data").

    python examples/train_pony.py --molecules molecules.csv --smiles-column smiles --pairs pairs.csv --out my_pony.pt
    python examples/train_pony.py --pairs pairs.csv --out my_pony.pt        # Stage 2 only, on the paper's networks

molecules.csv: a SMILES column and one 0/1 column per odour word (Stage 1).
pairs.csv:     mixture_a, mixture_b, rating; a mixture is its SMILES joined by ';' (Stage 2).
"""
import argparse

import numpy as np
import torch

from pony import Pony
from pony.data import holdout_mixtures, read_molecules, read_pairs, subset
from pony.train import train_network


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--molecules", help="single-molecule word labels; omit to keep the paper's networks")
    ap.add_argument("--smiles-column", default="smiles")
    ap.add_argument("--pairs", required=True, help="rated mixture pairs")
    ap.add_argument("--out", required=True, help="where to save the model")
    ap.add_argument("--copies", type=int, default=10, help="network copies (paper: 10)")
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of mixtures held out for the check")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    # Step 2: load the files
    mix_a, mix_b, y = read_pairs(a.pairs)
    print(f"{len(y)} rated pairs")
    if a.molecules:
        smiles, labels, words = read_molecules(a.molecules, a.smiles_column)
        print(f"{len(smiles)} molecules, {len(words)} odour words")
        # Step 3 (Stage 1): train the network copies on single molecules
        nets = [train_network(smiles, labels, seed=s, device=a.device) for s in range(a.copies)]
        # Step 4: freeze them into a model
        reference = sorted({s for m in mix_a + mix_b for s in m})
        model = Pony.from_networks(nets, words, smiles, reference)
    else:
        model = Pony.from_pretrained()

    # Step 5: hold out whole mixtures
    test = holdout_mixtures(mix_a, mix_b, a.holdout); train = ~test
    print(f"fit on {train.sum()} pairs, check on {test.sum()} pairs (every pair touching a held-out mixture)")
    # Step 6 (Stage 2): fit the mixture readout
    model.fit(subset(mix_a, train), subset(mix_b, train), y[train])
    # Step 7: check on the held-out pairs
    pred = model.predict(subset(mix_a, test), subset(mix_b, test))
    print(f"held-out pairs: r {np.corrcoef(pred, y[test])[0, 1]:.3f}, RMSE {np.sqrt(np.mean((pred - y[test]) ** 2)):.4f}")
    # Step 8: refit on all pairs and save
    model.fit(mix_a, mix_b, y); model.save(a.out)
    print(f"saved {a.out}; load it with Pony.from_pretrained({a.out!r})")


if __name__ == "__main__":
    main()
