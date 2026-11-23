"""Train a new Pony model.

Stage 1 trains the network on single molecules labelled with odour words (open-POM's recipe), and then freezes it.
Stage 2 fits the mixture readout on rated mixture pairs (``Pony.fit``).

    from pony import Pony
    from pony.train import train_network

    nets = [train_network(smiles, labels, seed=s, device="cuda") for s in range(10)]      # Stage 1, 10 copies
    model = Pony.from_networks(nets, words, smiles)                                       # frozen copies
    model.fit(mixtures_a, mixtures_b, ratings)                                            # Stage 2
    model.save("my_pony.pt")                                                              # Pony.from_pretrained("my_pony.pt")

``smiles``: the training molecules; ``labels``: molecules x words, 1 if the word describes the molecule, else 0;
``words``: the word names. The paper trains on open-POM's curated GS-LF data (3,975 training molecules, 138 words).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from . import features as fz
from .network import PonyNet

# open-POM's training recipe (batch, epochs, Adam learning rate, step decay of the learning rate, L1 + L2 weight penalty)
RECIPE = {"batch": 128, "epochs": 62, "lr": 1e-3, "decay": 0.5, "decay_steps": 640, "weight_decay": 1e-5}
ARCH = {"d": 64, "h": 64, "steps": 3, "edge_h": 16}          # the paper's network (116,603 parameters with 138 words)


def word_loss(logits, y, word_weight):
    """open-POM's multi-label loss as run by DeepChem: cross-entropy over [1 - z, z] on the logits z, weighted per word
    by log(1 + count / largest count), summed over words, averaged over molecules."""
    two = torch.stack([1 - logits, logits], 1)
    ce = nn.functional.cross_entropy(two, y.long(), reduction="none")
    return (ce * word_weight).sum(1).mean()


def weight_penalty(net: nn.Module, wd: float):
    """wd x (sum |w| + sum ||w||_2) over all weights except biases (open-POM)."""
    l1 = sum(p.abs().sum() for n, p in net.named_parameters() if "bias" not in n)
    l2 = sum(p.norm(2) for n, p in net.named_parameters() if "bias" not in n)
    return wd * (l1 + l2)


def _to(batch, device):
    x, e, s, d, g = batch
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return t(x, torch.float32), t(e, torch.float32), t(s, torch.long), t(d, torch.long), t(g, torch.long)


def train_network(smiles: Sequence[str], labels, seed: int = 0, device: str = "cpu", recipe: dict | None = None,
                  arch: dict | None = None, log: Callable[[str], None] = print) -> PonyNet:
    """Stage 1: train one network copy on single molecules and return it frozen (eval mode, on the CPU).

    smiles: N molecules; labels: N x W array of 0/1 (W odour words). Every epoch visits each molecule once in a random
    order; the last batch is filled up with randomly chosen molecules, as DeepChem does."""
    recipe = {**RECIPE, **(recipe or {})}; arch = {**ARCH, **(arch or {})}
    Y = np.asarray(labels, np.float32)
    if Y.ndim != 2 or len(Y) != len(smiles):
        raise ValueError("labels must be a (molecules x words) array with one row per SMILES")
    graphs = [fz.featurize(s) for s in smiles]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = PonyNet(**arch, n_words=Y.shape[1]).float().to(device)
    Yt = torch.as_tensor(Y, device=device); ids = np.arange(len(graphs))
    counts = Y.sum(0); word_weight = torch.as_tensor(np.log1p(counts / counts.max()), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=recipe["lr"], betas=(0.9, 0.999), eps=1e-8)
    bs, step = recipe["batch"], 0; n_batches = math.ceil(len(ids) / bs)
    for epoch in range(recipe["epochs"]):
        net.train(); perm = rng.permutation(ids)
        perm = np.concatenate([perm, rng.choice(ids, n_batches * bs - len(perm), replace=False)])
        total = 0.0
        for k in range(n_batches):
            for group in opt.param_groups:
                group["lr"] = recipe["lr"] * recipe["decay"] ** (step // recipe["decay_steps"])
            b = perm[k * bs:(k + 1) * bs]
            logits = net.word_logits(*_to(fz.batch([graphs[i] for i in b]), device), len(b))
            loss = word_loss(logits, Yt[b], word_weight) + weight_penalty(net, recipe["weight_decay"])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1; total += loss.item()
        if log and ((epoch + 1) % 10 == 0 or epoch == recipe["epochs"] - 1):
            log(f"seed {seed}: epoch {epoch + 1}/{recipe['epochs']}, loss {total / n_batches:.4f}")
    net = net.cpu().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def _encode(net: PonyNet, graphs, chunk: int = 256):
    """-> H (atoms x h), atom scores s, v (molecules x h), molecule index of each atom; eval mode, float64 numpy."""
    net.eval(); dev = next(net.parameters()).device; Hs, ss, vs, mols, base = [], [], [], [], 0
    for k in range(0, len(graphs), chunk):
        g = graphs[k:k + chunk]; batch = _to(fz.batch(g), dev)
        H, s, v, _ = net(*batch, len(g))
        Hs.append(H.double().cpu().numpy()); ss.append(s.double().cpu().numpy()); vs.append(v.double().cpu().numpy())
        mols.append(batch[4].cpu().numpy() + base); base += len(g)
    return np.concatenate(Hs), np.concatenate(ss), np.concatenate(vs), np.concatenate(mols)


def calibrate(net: PonyNet, scale_smiles: Sequence[str], reference_smiles: Sequence[str], n_scale: int = 2000, seed: int = 0):
    """The two constants of a frozen copy, as in the paper.

    scale: mean length of the hidden vector v over up to ``n_scale`` training molecules (hidden vectors are divided by it).
    u0:    the network's own atom scorer read off the hidden layer: least squares of atom score / SD on the atom hidden
           vectors H / scale (with an intercept, dropped), over the reference molecules; SD = spread across reference
           molecules of each molecule's log mean exp of its atom scores. The readout's atom scores start from u0.
    The paper uses the GS-LF training molecules for scale and the 235 DREAM molecules as reference."""
    idx = np.random.default_rng(seed).choice(len(scale_smiles), min(n_scale, len(scale_smiles)), replace=False)
    _, _, v, _ = _encode(net, [fz.featurize(scale_smiles[i]) for i in idx])
    scale = float(np.linalg.norm(v, axis=1).mean())
    H, s, _, mol = _encode(net, [fz.featurize(x) for x in reference_smiles]); H = H / scale
    n = np.bincount(mol); mx = np.full(len(n), -np.inf); np.maximum.at(mx, mol, s)
    se = np.zeros(len(n)); np.add.at(se, mol, np.exp(s - mx[mol])); strength = mx + np.log(se / n)
    Xa = np.c_[H, np.ones(len(H))]
    u0 = np.linalg.lstsq(Xa, s / strength.std(), rcond=None)[0][:-1]
    return scale, u0
