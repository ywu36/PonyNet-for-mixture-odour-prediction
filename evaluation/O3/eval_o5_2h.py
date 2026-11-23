# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""O5-2h: two attentions over atoms (user, 2026-11-20: "double attention ... one like the counts, another that trades
off the molecule strength"). Network (checkpoints/o3m_attpool2, train_o3m.py --config attpool2): head 1 pools atoms
into the molecule's smell direction (centred, length 1); head 2's atom scores give the molecule's strength
(log mean exp over its atoms) and set the hidden vector's length: v = 8 exp(strength) x direction. The linear word head
reads both, so both heads learn from single molecules only (GS-LF word loss).
Mixing: unit merge on the directions; molecule weight ∝ exp(scale x strength + odd-one-out); line. 4 numbers on DREAM.
Rows (fixed before running), 3 seeds averaged:
  O1.2b (27); plain O3 + unit merge + atom soft max (68, current best); O5 + own atom scores (4)
  O5-2h: equal weights (3); head-2 strength (4); head-1 strength (4, check: is the strength head needed?)
CV: 514 development pairs, 50 x 5, all subsets; --heldout: fit on 820 training pairs, T46 then V50. No overlap term,
no routing, lockbox not used.

Env:     mist
Usage:   python evaluation/O3/eval_o5_2h.py [--heldout]
Writes:  results/eval_o5_2h/<date_time>[_heldout].{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from eval_unit_merge import LAM  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_attpool import backbones, LearnedStrength, REF, CUR  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
O5 = "O5: own atom scores + odd-one-out (4)"


@torch.no_grad()
def two_head(dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); rows = G.dream_rows(); out = []
    for s in SEEDS:
        net = train_o3m.load(ROOT / f"checkpoints/o3m_attpool2/member_{s}.pt", dev); net.eval()
        x, e, s_, d, g = G.batch(rows, dev)
        xa = net.atom(x); hidden = xa.unsqueeze(0); W = net.conv.edge_func(e).view(-1, net.d, net.d)
        for _ in range(net.steps):
            xa = torch.relu(net.conv(xa, W, s_, d)); xa, hidden = net.gru(xa.unsqueeze(0), hidden); xa = xa.squeeze(0)
        st1 = net.strength(xa, g, len(rows), head=1).double().cpu().numpy(); st2 = net.strength(xa, g, len(rows), head=2).double().cpu().numpy()
        V = net.latent(net.encode(x, e, s_, d, g, len(rows)))[0].double().cpu().numpy()
        assert np.allclose(np.log(np.linalg.norm(V, axis=1) / 8.0), np.clip(st2, -5, 5), atol=1e-3)
        out.append((V, st1, st2))
    return out


def models(BB, T2, q_op, X, A, B):
    def ens(items, make):
        return lambda tr, te, y: np.mean([make(*b).fit(tr, y[tr]).predict(te) for b in items], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            CUR: ens(BB["plain"], lambda V, H, mol, st: UnitAtom(V, H, mol, A, B)),
            O5: ens(BB["attpool"], lambda V, H, mol, st: LearnedStrength(V, st, A, B, True)),
            "O5-2h: equal weights + odd-one-out (3)": ens(T2, lambda V, s1, s2: LearnedStrength(V, s2, A, B, False)),
            "O5-2h: head-2 strength + odd-one-out (4)": ens(T2, lambda V, s1, s2: LearnedStrength(V, s2, A, B, True)),
            "O5-2h: head-1 strength + odd-one-out (4, check)": ens(T2, lambda V, s1, s2: LearnedStrength(V, s1, A, B, True))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_o5_2h", {"seeds": SEEDS, "lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    BB = backbones(dev); T2 = two_head(dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); nat = np.bincount(G.batch(G.dream_rows(), "cpu")[4].numpy())
    for i, (V, s1, s2) in enumerate(T2):
        rep.line(f"O5-2h member {i}: head-2 strength sd {s2.std():.3f}, corr with head-1 strength {np.corrcoef(s1, s2)[0, 1]:+.3f}, "
                 f"with log atom count {np.corrcoef(s2, np.log(nat))[0, 1]:+.3f}")
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
               "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "setting 2 (share >= 1)": k_P >= 1,
               "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        res = {}
        for nm, fp in models(BB, T2, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), "" if nm in (REF, CUR) else ch(sc(r_, sub, res[CUR])),
                             ch(sc(r_, sub, res[O5])) if nm.startswith("O5-2h") else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from current best", "change from O5"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, T2, q_op, X, A, B).items()}
            y = hs.y_P; ref = preds[REF]; rows = []
            rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
            for nm, p in preds.items():
                s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
                if nm != REF:
                    cr, ce = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} [{ce[0]:+.4f}, {ce[1]:+.4f}] / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE [CI] / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
