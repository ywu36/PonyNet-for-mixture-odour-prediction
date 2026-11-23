# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  V = atoms   N = held-out GS-LF molecules (1,008)   W = words (138)
#   M = DREAM molecules (235)   P = pairs   R = members
# ---------------------------------------------------------------------------------------------
"""Ablation of open-POM's LSTM pooling (user 2026-11-22): checks the paper sentence "Open-POM's LSTM pooling and deep
dense read-out raise single-molecule accuracy (0.890 against 0.860) but also the memorization gap (0.069 against 0.025),
without improving mixture prediction".

Paired with checkpoints/o2c members 0-2 (open-POM layout, set2set = LSTM query + attention over atoms, 3-layer read-out):
same graphs, same official held-out molecules, same recipe (o2.RECIPE), same seeds. One change per variant:
  mean        set2set -> plain mean over atoms (LSTM removed)
  query       set2set -> attention over atoms with one learned query, one pass (= set2set without the LSTM)
  mean_lin    mean over atoms AND the 3-layer read-out -> one linear layer
Scoring: single molecules (held-out AUROC of the 3-member mean prediction; gap = mean training AUROC on a fixed 1,000
training-molecule sample minus mean held-out AUROC, as train_groupatoms.py) and DREAM with Table 5's open-POM rule
(word profiles of the 3-member mean, quantile per word, mean per mixture, length 1, Euclidean, line only):
T46 / V50 (fit on the 820 training pairs) and CV 50 x 5 on the 514 development pairs.

Env:     mist (GPU)
Usage:   python training/train_o2c_nolstm.py            (trains missing members, then scores)
Writes:  checkpoints/o2c_<variant>/member_<s>.pt, results/train_o2c_nolstm/<date_time>.{txt,json}
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "training", ROOT / "evaluation/O1"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402

SEEDS = (0, 1, 2)
D = o2.NODE_OUT + o2.EDGE_OUT          # atom vector size after the radius-0 fold (200)


class MeanPool(nn.Module):
    def forward(self, feat_VxF, graph_V, n_graphs):
        s = feat_VxF.new_zeros(n_graphs, feat_VxF.shape[1]).index_add_(0, graph_V, feat_VxF)
        return s / feat_VxF.new_zeros(n_graphs).index_add_(0, graph_V, torch.ones_like(graph_V, dtype=feat_VxF.dtype))[:, None]


class QueryPool(nn.Module):
    """set2set without the LSTM: one learned query, softmax over each molecule's atoms, one pass."""

    def __init__(self, dim):
        super().__init__(); self.q = nn.Parameter(torch.randn(dim) / dim ** 0.5)

    def forward(self, feat_VxF, graph_V, n_graphs):
        e_V = feat_VxF @ self.q
        emax = feat_VxF.new_full((n_graphs,), -float("inf")).scatter_reduce(0, graph_V, e_V, "amax", include_self=True)
        ex = torch.exp(e_V - emax[graph_V]); den = feat_VxF.new_zeros(n_graphs).index_add_(0, graph_V, ex)
        return feat_VxF.new_zeros(n_graphs, feat_VxF.shape[1]).index_add_(0, graph_V, feat_VxF * (ex / den[graph_V])[:, None])


class LinearHead(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__(); self.lin = nn.Linear(d_in, n_out)

    def forward(self, x):
        return x, self.lin(x)


class NoLSTM(o2.O2C):
    """O2-C with set2set replaced (and optionally the read-out); everything else unchanged."""

    def __init__(self, variant, atom_f=134, bond_f=6, n_words=138):
        super().__init__(atom_f, bond_f, n_words)
        self.readout_set2set = QueryPool(D) if variant == "query" else MeanPool()
        self.centre = nn.BatchNorm1d(D, affine=False)
        self.ffn = LinearHead(D, n_words) if variant == "mean_lin" else o2.FFN(D, n_words)


VARIANTS = {"mean": "LSTM pooling -> mean over atoms", "query": "LSTM pooling -> one learned query",
            "mean_lin": "mean over atoms + linear read-out"}


def load(variant, s, dev):
    net = NoLSTM(variant).float(); net.load_state_dict(torch.load(ROOT / f"checkpoints/o2c_{variant}/member_{s}.pt",
                                                                   map_location="cpu", weights_only=False)["model_state_dict"])
    return net.to(dev).eval()


def auroc(p, y):
    return o2.mean_auroc(p, y)


def main():
    dev = pick_device(); rep = Report("train_o2c_nolstm", {"seeds": SEEDS, "device": dev, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    tr_s = np.random.default_rng(0).choice(tr, 1000, replace=False); rows_D = G.dream_rows()
    rep.line(f"device {dev}; GS-LF training {len(tr)}, held out {len(te)}; DREAM molecules {len(rows_D)}")
    for v in VARIANTS:
        ck = ROOT / f"checkpoints/o2c_{v}"; ck.mkdir(parents=True, exist_ok=True)
        for s in SEEDS:
            if (ck / f"member_{s}.pt").exists():
                continue
            t0 = time.time(); net, steps = o2.train_member(G, tr, s, dev, log=rep.line, make_net=lambda: NoLSTM(v))
            torch.save({"model_state_dict": net.state_dict(), "variant": v, "seed": s, "steps": steps}, ck / f"member_{s}.pt")
            rep.line(f"{v} seed {s}: {steps} steps, {time.time() - t0:.0f} s")

    nets = {"open-POM layout (LSTM pooling, deep read-out)": [o2.load_members([ROOT / f"checkpoints/o2c/member_{s}.pt"], dev)[0] for s in SEEDS]}
    nets.update({VARIANTS[v]: [load(v, s, dev) for s in SEEDS] for v in VARIANTS})
    REF = next(iter(nets))
    single, probs_D = [], {}
    for k, ns in nets.items():
        pt = [o2.predict_probs(n, G, te, dev) for n in ns]; a_te = [auroc(p, G.Y[te]) for p in pt]
        a_tr = [auroc(o2.predict_probs(n, G, tr_s, dev), G.Y[tr_s]) for n in ns]
        probs_D[k] = np.mean([o2.predict_probs(n, G, rows_D, dev) for n in ns], 0).astype(float)
        single.append([k, f"{sum(p.numel() for p in ns[0].parameters()) / 1e6:.2f} M", " ".join(f"{x:.4f}" for x in a_te),
                       f"{auroc(np.mean(pt, 0), G.Y[te]):.4f}", f"{np.mean(a_tr):.4f}", f"{np.mean(a_tr) - np.mean(a_te):+.4f}"])
    rep.section("1. Single molecules, official held-out (1,008): AUROC, mean over 138 words")
    rep.table(["network", "parameters", "held-out per seed (0, 1, 2)", "3-member mean held-out", "training (1,000)", "gap (training - held-out)"], single)

    op = features.load("openpom"); cids = np.asarray(op.cids_M)
    order = {int(c): i for i, c in enumerate(G.dream_cids)}; perm = np.array([order[int(c)] for c in cids])
    feats = {"released open-POM (Table 5 row)": np.asarray(op.probs_MxW, float)}
    feats.update({k: p[perm] for k, p in probs_D.items()})

    def line(d):
        return lambda a, b: np.polyval(np.polyfit(d[a], yy[a], 1), d[b])

    rep.section("2. DREAM, Table 5 rule (word profiles averaged, line only): RMSE / r; change from open-POM layout r [95% CI]")
    train = data.load_train(); res_rows = []
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); a = np.arange(n); b = np.arange(n, len(allp))
        yy = np.r_[evaluate.labels(train), hs.y_P]; preds = {k: line(mixture.paper_distance(allp, cids, f))(a, b) for k, f in feats.items()}
        for k, p in preds.items():
            c = "" if k == REF or k not in probs_D else (lambda ci: f"{metrics.pearson(p, hs.y_P) - metrics.pearson(preds[REF], hs.y_P):+.3f} [{ci[0][0]:+.3f}, {ci[0][1]:+.3f}]")(paired(p, preds[REF], hs.y_P))
            res_rows.append([hs.name, k, f"{metrics.rmse(p, hs.y_P):.4f} / {metrics.pearson(p, hs.y_P):.3f}", c])
    P = dev_pairs(); yy = evaluate.labels(P)
    cv = {k: evaluate.cross_validate(P, line(mixture.paper_distance(P, cids, f))) for k, f in feats.items()}
    for k, r_ in cv.items():
        ref = None if (k == REF or k not in probs_D) else cv[REF]; s = evaluate.score_cv(P, r_, reference=ref)
        c = "" if ref is None else f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"
        res_rows.append(["CV 50 x 5 (514)", k, f"{s['rmse']:.4f} / {s['r']:.3f}", c])
    rep.table(["set", "network", "RMSE / r", "change in r from open-POM layout"], res_rows)
    rep.value("single", single); rep.value("dream", res_rows); rep.save()


if __name__ == "__main__":
    main()
