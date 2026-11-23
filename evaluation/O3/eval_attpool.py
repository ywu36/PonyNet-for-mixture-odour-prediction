# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   T = atoms
# ---------------------------------------------------------------------------------------------
"""Who dominates a mixture, learned from single molecules (user, 2026-11-20: "train plain O3 with attention pooling
over atoms (word loss only, single molecules), keep unit merge, and fit only a few numbers on DREAM").
New network (checkpoints/o3m_attpool, train_o3m.py --config attpool): plain O3 with the mean over atoms replaced by
attention pooling: atom score s_a = g . x_a (g learned by the word loss on GS-LF singles), weights soft max within the
molecule, v = weighted mean (same 64-number hidden layer, linear word head). Molecule strength for mixing =
log mean_a exp(s_a) (soft max of its atom scores, size-neutral), standardised over the 235 DREAM molecules.
Rows (fixed before running), 3 seeds averaged, unit merge everywhere, angle comparison:
  O1.2b (reference, 27)
  plain O3: unit merge, attention (67); unit merge + atom soft max (68) = current best (other session's UnitAtom)
  attention-pooled O3: equal weights + odd-one-out (3)
  attention-pooled O3: learned strength x scale + odd-one-out (4)        <- the step asked for: 4 numbers on DREAM
  attention-pooled O3: unit merge, attention (67); unit merge + atom soft max fitted on DREAM (68)  (separates the
      new network's directions from its learned weights)
CV: 514 development pairs, 50 x 5, all subsets; --heldout: fit on 820 training pairs, T46 then V50. No overlap term,
no routing, lockbox not used.

Env:     mist
Usage:   [O5_SEEDS="0 1 ... 9"] python evaluation/O3/eval_attpool.py [--heldout] [--lean]
Writes:  results/eval_attpool/<date_time>[_heldout].{txt,json}
"""
import argparse
import os
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
from eval_unit_merge import UnitMerge, LAM  # noqa: E402
from eval_unit_atom import UnitAtom  # noqa: E402
from eval_atom_attn import atom_vectors  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
SEEDS = tuple(int(x) for x in os.environ.get("O5_SEEDS", "0 1 2").split())  # O5_SEEDS="0 1 ... 9" for 10 members
LEAN = False  # --lean: only O1.2b, plain O3 unit merge (67), current best (68), O5 equal weights (3), O5 own scores (4)
REF = "O1.2b (reference, 27)"
CUR = "plain O3: unit merge + atom soft max (68) [current best]"


@torch.no_grad()
def attpool_atoms(net, G, rows, dev):
    """Atom vectors in the hidden space H_a = centre(lin(x_a)), atom scores s_a, molecule index; v = sum_a w_a H_a."""
    net.eval(); x, e, s, d, g = G.batch(rows, dev)
    xa = net.atom(x); hidden = xa.unsqueeze(0); W = net.conv.edge_func(e).view(-1, net.d, net.d)
    for _ in range(net.steps):
        xa = torch.relu(net.conv(xa, W, s, d)); xa, hidden = net.gru(xa.unsqueeze(0), hidden); xa = xa.squeeze(0)
    H = net.centre(net.lin(xa)); sc = net.atom_scores(xa); w = net.atom_weights(xa, g, len(rows))
    v = net.latent(net.encode(x, e, s, d, g, len(rows)))[0]
    v2 = torch.zeros_like(v).index_add_(0, g, w.unsqueeze(1) * H)
    assert torch.allclose(v, v2, atol=1e-4), (v - v2).abs().max()
    return H.double().cpu().numpy(), sc.double().cpu().numpy(), g.cpu().numpy(), v.double().cpu().numpy()


def backbones(dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = {"plain": [], "attpool": []}
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        H, mol = atom_vectors(n, G, rows_D, dev); H = H / scale
        V = np.zeros((len(rows_D), H.shape[1])); np.add.at(V, mol, H); V /= np.bincount(mol)[:, None]
        out["plain"].append((V, H, mol, None))
        n = train_o3m.load(ROOT / f"checkpoints/o3m_attpool/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        H, sc, mol, V = attpool_atoms(n, G, rows_D, dev); H, V = H / scale, V / scale
        cnt = np.bincount(mol); mx = np.full(len(cnt), -np.inf); np.maximum.at(mx, mol, sc)
        se = np.zeros(len(cnt)); np.add.at(se, mol, np.exp(sc - mx[mol])); strength = mx + np.log(se / cnt)
        out["attpool"].append((V, H, mol, strength))
    return out


class LearnedStrength(ecd.ContextReadoutQ):
    """Unit merge; molecule log-weight = c x learned strength (c fitted; c fixed 0 = equal weights) + odd-one-out."""

    def __init__(self, V, strength, A, B, use_strength):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=LAM); self.U = self.V / self.V.norm(dim=1, keepdim=True)
        st = np.asarray(strength, float); self.S = torch.as_tensor((st - st.mean()) / st.std(), dtype=F64); self.use = use_strength

    def dist(self, kappa, idx):
        def out(Mx, C):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(lw); z = (W @ self.U) / W.sum(1, keepdim=True)
            return z / z.norm(dim=1, keepdim=True)
        return torch.sqrt(((out(self.A, self.CA) - out(self.B, self.CB)) ** 2).sum(1) + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        c = torch.zeros((), dtype=F64, requires_grad=self.use); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        with torch.no_grad():
            self.lw = c * self.S; d = self.dist(kappa, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, kappa] + ([c] if self.use else []), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = c * self.S
            loss = ((lin[0] * self.dist(kappa, idx) + lin[1] - y) ** 2).mean() + self.lam * (kappa ** 2 + c ** 2)
            loss.backward(); return loss
        opt.step(closure)
        self.c, self.kappa, self.lin = c.detach(), kappa.detach(), lin.detach()
        with torch.no_grad():
            self.lw = self.c * self.S
        return self

    def predict(self, idx):
        with torch.no_grad():
            return (self.lin[0] * self.dist(self.kappa, torch.as_tensor(np.asarray(idx))) + self.lin[1]).numpy()


def models(BB, q_op, X, A, B):
    d = _models(BB, q_op, X, A, B)
    if not LEAN:
        return d
    return {k: v for k, v in d.items() if k in (REF, CUR, "plain O3: unit merge, attention (67)") or "odd-one-out" in k}


def _models(BB, q_op, X, A, B):
    def ens(k, make):
        return lambda tr, te, y: np.mean([make(*b).fit(tr, y[tr]).predict(te) for b in BB[k]], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            "plain O3: unit merge, attention (67)": ens("plain", lambda V, H, mol, st: UnitMerge(V, A, B, False)),
            CUR: ens("plain", lambda V, H, mol, st: UnitAtom(V, H, mol, A, B)),
            "attention-pooled O3: equal weights + odd-one-out (3)": ens("attpool", lambda V, H, mol, st: LearnedStrength(V, st, A, B, False)),
            "attention-pooled O3: learned strength + odd-one-out (4)": ens("attpool", lambda V, H, mol, st: LearnedStrength(V, st, A, B, True)),
            "attention-pooled O3: unit merge, attention (67)": ens("attpool", lambda V, H, mol, st: UnitMerge(V, A, B, False)),
            "attention-pooled O3: unit merge + atom soft max fitted on DREAM (68)": ens("attpool", lambda V, H, mol, st: UnitAtom(V, H, mol, A, B))}


def main():
    global LEAN
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); ap.add_argument("--lean", action="store_true")
    a = ap.parse_args(); LEAN = a.lean
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_attpool", {"seeds": SEEDS, "lean": LEAN, "lam": LAM, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    BB = backbones(dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    nat = np.bincount(BB["attpool"][0][2])
    for i, (V, H, mol, st) in enumerate(BB["attpool"]):
        rep.line(f"member {i}: learned strength sd {st.std():.3f}; corr with log atom count {np.corrcoef(st, np.log(nat))[0, 1]:+.3f}")

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
               "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "setting 2 (share >= 1)": k_P >= 1,
               "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
        fits = [LearnedStrength(V, st, A, B, True).fit(np.arange(len(P)), y) for V, H, mol, st in BB["attpool"]]
        rep.line("learned-strength readout fitted on all development pairs: scale c " + ", ".join(f"{m.c.item():+.3f}" for m in fits)
                 + "; odd-one-out " + ", ".join(f"{m.kappa.item():+.3f}" for m in fits))
        res = {}
        for nm, fp in models(BB, q_op, X, A, B).items():
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
                             "" if nm == REF else ch(s_), "" if nm in (REF, CUR) else ch(sc(r_, sub, res[CUR]))])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from current best"], rows)
    else:
        train = data.load_train()
        for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
            allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
            y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)
            preds = {nm: fp(tr, te, y_all) for nm, fp in models(BB, q_op, X, A, B).items()}
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
