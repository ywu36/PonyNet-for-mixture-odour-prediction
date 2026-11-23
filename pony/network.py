"""The Pony network: a small message-passing network trained on single molecules (Stage 1), then frozen.

    atoms (134) -> Linear + ReLU -> 64
    3 rounds of message passing: NNConv (bond features 6 -> 16 -> 64 x 64, sum over neighbours, residual) + GRU
    atom attention: one linear scorer per atom, soft max within the molecule, weighted sum of atom states
    Linear 64 -> 64, centring (BatchNorm without scale or shift)         = hidden layer v(m), 64 units
    word head: one Linear 64 -> 138 odour words (sigmoid probabilities)

Parameter names match the training checkpoints, so ``load_state_dict`` works on them directly.
"""
from __future__ import annotations

import torch
from torch import nn


class NNConv(nn.Module):
    """Edge-conditioned convolution with sum aggregation and identity residual: h'_j = sum_{i->j} h_i W(e_ij) + h_j + b."""

    def __init__(self, n_in: int, n_out: int, edge_in: int, edge_hidden: int):
        super().__init__()
        self.edge_func = nn.Sequential(nn.Linear(edge_in, edge_hidden), nn.ReLU(), nn.Linear(edge_hidden, n_in * n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, h, W, src, dst):
        msg = torch.bmm(h[src].unsqueeze(1), W).squeeze(1)
        return torch.zeros_like(h).index_add_(0, dst, msg) + h + self.bias


class PonyNet(nn.Module):
    def __init__(self, d: int = 64, h: int = 64, steps: int = 3, edge_h: int = 16, dropout: float = 0.1,
                 atom_f: int = 134, bond_f: int = 6, n_words: int = 138):
        super().__init__()
        self.d, self.h, self.steps = d, h, steps
        self.arch = {"d": d, "h": h, "steps": steps, "edge_h": edge_h, "dropout": dropout, "atom_f": atom_f, "bond_f": bond_f, "n_words": n_words}
        self.atom = nn.Sequential(nn.Linear(atom_f, d), nn.ReLU())
        self.conv = NNConv(d, d, bond_f, edge_h)
        self.gru = nn.GRU(d, d)
        self.att = nn.Linear(d, 1)
        self.ffn = nn.Sequential()
        self.lin = nn.Linear(d, h)
        self.centre = nn.BatchNorm1d(h, affine=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, n_words)

    def atom_states(self, x, e, src, dst):
        """Atom states after message passing (atoms x d)."""
        x = self.atom(x); hidden = x.unsqueeze(0)
        W = self.conv.edge_func(e).view(-1, self.d, self.d)
        for _ in range(self.steps):
            x = torch.relu(self.conv(x, W, src, dst))
            x, hidden = self.gru(x.unsqueeze(0), hidden); x = x.squeeze(0)
        return x

    def molecule_vectors(self, xa, graph, n_graphs: int):
        """Atom scores s and molecule hidden vectors v = centre(lin(attention-weighted sum of atom states))."""
        s = self.att(xa).squeeze(1)
        mx = s.new_full((n_graphs,), -1e30).scatter_reduce(0, graph, s, reduce="amax", include_self=True)
        w = torch.exp(s - mx[graph]); w = w / s.new_zeros(n_graphs).index_add_(0, graph, w)[graph]
        pooled = xa.new_zeros(n_graphs, self.d).index_add_(0, graph, w.unsqueeze(1) * xa)
        return s, self.centre(self.lin(self.ffn(pooled)))

    def word_logits(self, x, e, src, dst, graph, n_graphs: int):
        """Word logits only (n_graphs x n_words); used for Stage-1 training, where the centring must see molecules only."""
        _, v = self.molecule_vectors(self.atom_states(x, e, src, dst), graph, n_graphs)
        return self.head(self.drop(v))

    def forward(self, x, e, src, dst, graph, n_graphs: int):
        """Returns atom hidden vectors H (atoms x h), atom scores s (atoms), molecule hidden vectors v (n_graphs x h)
        and word logits (n_graphs x n_words). v is the attention-weighted mean of the atoms' H."""
        if self.training:
            raise RuntimeError("forward() is for the frozen network (eval mode); train with word_logits()")
        xa = self.atom_states(x, e, src, dst)
        s, v = self.molecule_vectors(xa, graph, n_graphs)
        H = self.centre(self.lin(xa))
        return H, s, v, self.head(self.drop(v))
