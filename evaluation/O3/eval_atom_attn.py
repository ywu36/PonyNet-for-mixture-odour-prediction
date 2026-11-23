# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
#   T = atoms of the 235 DREAM molecules (all together)
# ---------------------------------------------------------------------------------------------
"""Strength from structure: attention over a molecule's ATOMS sets its weight in the mixture; the direction (flavor)
stays the plain O3 molecule vector (user, 2026-11-20: "how strong a molecule is depends on its structure; not the
molecule as a whole; an attention layer from the structure side").
Plain O3 (mean pooling, no FFN): v_m = mean over atoms of h_a, h_a = centre(lin(x_a)) (lin and eval BatchNorm are
affine), so every atom has a 64-number vector in the same space as v.
    atom score  s_a = q . h_a                         (q: 64 numbers, as the current query)
    mean (current):     log w_m = mean_a s_a = q . v_m
    soft max (new):     log w_m = tau * log( mean_a exp(s_a / tau) )   tau learned (1 number); tau -> inf gives the mean,
                        small tau: the strongest atom (part) decides
    sum (new):          log w_m = sum_a s_a = n_atoms * q . v_m        (larger molecules carry more)
    shares ~ w_m exp(kappa cos(v_m, others)); z = sum share v; angle comparison; line.
Rows (fixed before running), 3 seeds averaged:
  O1.2b (reference, 27) | atom mean = current attention (67) | atom soft max (68) | atom sum (67)
Penalty 1e-3 on q and kappa; tau = exp(t) floored at 1e-4 (numerical), t unpenalised, starts at tau = 10 x the SD of atom scores at a random q
(effectively the mean). CV: development pairs 50 x 5, six subsets, change from O1.2b and from atom mean (same folds).
--heldout: T46 then V50. Fitted tau on all development pairs reported (relative to the SD of fitted atom scores).

Env:     mist
Usage:   python evaluation/O3/eval_atom_attn.py [--heldout]
Writes:  results/eval_atom_attn/<date_time>.{txt,json}
"""
import argparse
import pickle
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
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
LAM = 1e-3
SEEDS = (0, 1, 2)
REF, K_MEAN, K_SMAX, K_SUM = "O1.2b (reference, 27)", "atom mean = current attention (67)", "atom soft max (68)", "atom sum (67)"


@torch.no_grad()
def atom_vectors(net, G, rows, dev):
    """Per-atom h_a = centre(lin(x_a)) for molecules ``rows`` -> (H_TxD, mol_T); mean over a molecule's atoms = v_m."""
    assert net.pool == "mean" and len(net.ffn) == 0
    net.eval(); x, e, s, d, g = G.batch(rows, dev)
    xa = net.atom(x); hidden = xa.unsqueeze(0); W = net.conv.edge_func(e).view(-1, net.d, net.d)
    for _ in range(net.steps):
        xa = torch.relu(net.conv(xa, W, s, d)); xa, hidden = net.gru(xa.unsqueeze(0), hidden); xa = xa.squeeze(0)
    H = net.centre(net.lin(xa))
    v = net.latent(net.encode(x, e, s, d, g, len(rows)))[0]
    v_from_atoms = torch.zeros_like(v).index_add_(0, g, H) / torch.bincount(g, minlength=len(rows)).unsqueeze(1)
    assert torch.allclose(v, v_from_atoms, atol=1e-4), (v - v_from_atoms).abs().max()
    return H.double().cpu().numpy(), g.cpu().numpy()


def backbone(dev):
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); _, tr, _, _ = train_groupatoms.graphs_with_groups()
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); out = []
    for s in SEEDS:
        n = train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item()
        H, mol = atom_vectors(n, G, rows_D, dev); H = H / scale
        V = np.zeros((len(rows_D), H.shape[1])); np.add.at(V, mol, H); V /= np.bincount(mol)[:, None]
        out.append((V, H, mol))
    return G, out


class AtomAttn(ecd.ContextReadoutQ):
    def __init__(self, V, H, mol, A, B, mode):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=LAM); self.mode = mode
        self.H = torch.as_tensor(H, dtype=F64); self.mol = torch.as_tensor(mol); self.M = len(V)
        self.n = torch.bincount(self.mol, minlength=self.M).double()
        rq = torch.as_tensor(np.random.default_rng(0).normal(size=H.shape[1]) / np.sqrt(H.shape[1]), dtype=F64)
        self.t0 = float(np.log(10 * (self.H @ rq).std().item()))

    def mol_lw(self, q, t):
        s = self.H @ q
        if self.mode == "mean":
            return torch.zeros(self.M, dtype=F64).index_add(0, self.mol, s) / self.n
        if self.mode == "sum":
            return torch.zeros(self.M, dtype=F64).index_add(0, self.mol, s)
        tau = torch.exp(t).clamp_min(1e-4); z = s / tau   # floor: tau -> 0 gave NaN in some folds (2026-11-20)
        mx = torch.full((self.M,), -1e30, dtype=F64).scatter_reduce(0, self.mol, z, reduce="amax", include_self=True)
        se = torch.zeros(self.M, dtype=F64).index_add(0, self.mol, torch.exp(z - mx[self.mol]))
        return tau * (mx + torch.log(se / self.n))

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        t = torch.tensor(self.t0, dtype=F64, requires_grad=self.mode == "softmax"); b0 = torch.zeros((), dtype=F64)
        with torch.no_grad():
            self.lw = self.mol_lw(q, t); d = self.distance(kappa, b0, idx)
            lin = torch.linalg.lstsq(torch.stack([d, torch.ones_like(d)], 1), y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, q, kappa] + ([t] if self.mode == "softmax" else []), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.mol_lw(q, t)
            loss = ((lin[0] * self.distance(kappa, b0, idx) + lin[1] - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.mol_lw(q, t).detach()
        self.q, self.kappa, self.t, self.b, self.lin = q.detach(), kappa.detach(), t.detach(), b0, lin.detach(); return self


def models(BB, q_op, X, A, B):
    def ens(mode):
        return lambda tr, te, y: np.mean([AtomAttn(V, H, mol, A, B, mode).fit(tr, y[tr]).predict(te) for V, H, mol in BB], 0)
    return {REF: lambda tr, te, y: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
            K_MEAN: ens("mean"), K_SMAX: ens("softmax"), K_SUM: ens("sum")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heldout", action="store_true"); a = ap.parse_args()
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_atom_attn", {"lam": LAM, "seeds": SEEDS, "heldout": a.heldout, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="heldout" if a.heldout else "")
    G, BB = backbone(dev); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    rep.line(f"atoms per DREAM molecule: median {np.median(np.bincount(BB[0][2])):.0f}, range {np.bincount(BB[0][2]).min()}-{np.bincount(BB[0][2]).max()}"
             " (heavy atoms); molecule vector = mean of atom vectors (checked)")

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    if not a.heldout:
        P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
        k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
        SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
               "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
        fits = [AtomAttn(V, H, mol, A, B, "softmax").fit(np.arange(len(P)), y) for V, H, mol in BB]
        rep.line("soft max fitted on all development pairs: tau / SD of atom scores "
                 + ", ".join(f"{np.exp(m.t.item()) / (m.H @ m.q).std().item():.2f}" for m in fits)
                 + "  (>> 1: like the mean; ~1 or below: the strongest atoms decide); weight range max/min "
                 + ", ".join(f"{np.exp((m.lw.max() - m.lw.min()).item()):.1f}" for m in fits))
        res = {}
        for nm, fp in models(BB, q_op, X, A, B).items():
            res[nm] = evaluate.cross_validate(P, lambda tr, te, fp=fp: fp(tr, te, y)); rep.line(f"done: {nm}")
            rep.line(f"  non-finite predictions: {int((~np.isfinite(res[nm].pred_N)).sum())}")
        pk = ROOT / "results/eval_atom_attn" / f"{rep.stamp if hasattr(rep, 'stamp') else 'latest'}_cv.pkl"
        pk.parent.mkdir(parents=True, exist_ok=True); pickle.dump({"res": res}, open(pk, "wb")); rep.line(f"CV predictions saved: {pk.name}")

        def sc(r_, sub, against):
            m = SUB[sub]
            return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
        rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r; change [95% CI], positive r change = better")
        rows = []
        for sub in SUB:
            for nm, r_ in res.items():
                s_ = sc(r_, sub, None if nm == REF else res[REF])
                rows.append([nm, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                             "" if nm == REF else ch(s_), ch(sc(r_, sub, res[K_MEAN])) if nm in (K_SMAX, K_SUM) else ""])
        rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from atom mean (same folds)"], rows)
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
                    cr, _ = paired(p, ref, y)
                    c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
                rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
            rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]"], rows)
            rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
