# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  V = atoms   E = directed bonds   F = feature size
# ---------------------------------------------------------------------------------------------
"""Message-passing ablation of the open-POM layout (user 2026-11-22), after train_o2c_nolstm.py found the LSTM pooling
plays no part in open-POM's memorization gap. One change at a time from O2-C (checkpoints/o2c members 0-2; 5 steps,
width 100, set2set and 3-layer read-out kept), same graphs, held-out molecules, recipe and seeds 0-2:
  steps0 / steps1 / steps3 / steps8   number of message-passing steps (weights are shared across steps, so this
                                      changes reach, not parameters; at 0 the atom-bond fold still adds direct neighbours)
  w50 / w200                          atom width of the message passing (100 -> 50 / 200); sizes the bond network
Scoring as train_o2c_nolstm.py (single-molecule AUROC and gap; DREAM with Table 5's open-POM rule).

Env:     mist (GPU)
Usage:   python training/train_o2c_mp.py
Writes:  checkpoints/o2c_<variant>/member_<s>.pt, results/train_o2c_mp/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]
from mixdist import o2  # noqa: E402
import train_o2c_nolstm as base  # noqa: E402


class MPNNVar(nn.Module):
    def __init__(self, atom_f, bond_f, width, steps):
        super().__init__()
        self.project_node_feats = nn.Sequential(nn.Linear(atom_f, width), nn.ReLU())
        self.gnn_layer = o2.NNConv(width, width, bond_f, o2.EDGE_HIDDEN)
        self.gru = nn.GRU(width, width); self.width, self.steps = width, steps

    def forward(self, x_VxF, e_ExF, src_E, dst_E):
        h = self.project_node_feats(x_VxF); hidden = h.unsqueeze(0)
        W = self.gnn_layer.edge_func(e_ExF).view(-1, self.width, self.width)
        for _ in range(self.steps):
            h = torch.relu(self.gnn_layer(h, W, src_E, dst_E))
            h, hidden = self.gru(h.unsqueeze(0), hidden); h = h.squeeze(0)
        return h


SPEC = {"steps0": (100, 0), "steps1": (100, 1), "steps3": (100, 3), "steps8": (100, 8), "w50": (50, 5), "w200": (200, 5)}


class MPVar(o2.O2C):
    def __init__(self, variant, atom_f=134, bond_f=6, n_words=138):
        super().__init__(atom_f, bond_f, n_words)
        width, steps = SPEC[variant]; d = width + o2.EDGE_OUT
        self.mpnn = MPNNVar(atom_f, bond_f, width, steps)
        self.readout_set2set = o2.Set2Set(d, o2.S2S_ITERS, o2.S2S_LAYERS)
        self.centre = nn.BatchNorm1d(2 * d, affine=False); self.ffn = o2.FFN(2 * d, n_words)


base.VARIANTS = {"steps0": "message passing 0 steps", "steps1": "message passing 1 step", "steps3": "message passing 3 steps",
                 "steps8": "message passing 8 steps", "w50": "message passing width 50", "w200": "message passing width 200"}
base.NoLSTM = MPVar
base.Report = (lambda R: (lambda name, *a, **k: R("train_o2c_mp", *a, **k)))(base.Report)

if __name__ == "__main__":
    base.main()
