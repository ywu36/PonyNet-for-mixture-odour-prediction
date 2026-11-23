# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   H = hidden layer (64)   P = DREAM development pairs (514)
# ---------------------------------------------------------------------------------------------
"""Normalisation against whitening in the hidden-layer attention readout (the user, 2026-11-19: "for a lot of molecules
the arc should be averaged out somehow towards a mean"; why_setting1.py: hidden templates turn toward a common direction
with size, cos 0.37 -> 0.71, much more than word profiles).

mu = average hidden vector of 2000 GS-LF training single molecules (never DREAM). Attention shares w_m as C17
(share ~ exp(q . v_m + kappa cos(v_m, mean of the others))). Arms, all frozen networks, fitted on DREAM training folds:
  base         z = sum w_m v_m; angle only                                         q 64 + kappa + line = 67
  centred      z = sum w_m (v_m - mu); angle only (mixture mean no longer drifts to mu)                   67
  centred+n    centred, z / sqrt(sum w_m^2) (= x sqrt(n) for equal shares: length no longer shrinks with size),
               angle + length weight b                                                                    68
  partial      z = sum w_m (v_m - g mu), g fitted; angle only                                             68
  pool         as centred / centred+n, but mu = average of the molecules in development mixtures (no labels): the
               point big DREAM mixtures actually drift to (the GS-LF average is near 0)                67 / 68
Networks: count network (ga_w64, node-enriched) and plain network (o3m_w64), 3 members each, predictions averaged.
Paired 50 x 5 (same folds): change from base (same network) and from O1.2b; setting-1 residual vs size mismatch;
whiteness (cos to mean template direction, small -> big mixtures) after centring.

Env:     mist
Usage:   python evaluation/O3/eval_white_norm.py [--repeats 50]
Writes:  results/eval_white_norm/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import K0, O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
F64 = torch.float64
ARMS = {"base": dict(centre=False, scale=False, length=False, partial=False),
        "centred": dict(centre=True, scale=False, length=False, partial=False),
        "centred + sqrt(n) + length": dict(centre=True, scale=True, length=True, partial=False),
        "partial centring (g fitted)": dict(centre=False, scale=False, length=False, partial=True),
        "centred on DREAM pool": dict(centre=True, scale=False, length=False, partial=False, pool=True),
        "centred on DREAM pool + sqrt(n) + length": dict(centre=True, scale=True, length=True, partial=False, pool=True)}
NUMBERS = {"base": 67, "centred": 67, "centred + sqrt(n) + length": 68, "partial centring (g fitted)": 68,
           "centred on DREAM pool": 67, "centred on DREAM pool + sqrt(n) + length": 68}


class WhiteNorm(ecd.ContextReadoutQ):
    def __init__(self, V, mu, A, B, centre, scale, length, partial):
        mu = np.asarray(mu, float)
        super().__init__(V - mu if centre else V, A, B, np.zeros(len(V)), True)
        self.mu = torch.as_tensor(mu, dtype=F64); self.scale, self.length, self.partial = scale, length, partial
        self.g = torch.zeros((), dtype=F64)

    def distance(self, kappa, b, idx):
        Vg = self.V - self.g * self.mu if self.partial else self.V

        def out(Mx, C):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(lw); W = W / W.sum(1, keepdim=True); z = W @ Vg
            if self.scale:
                z = z / torch.sqrt((W ** 2).sum(1, keepdim=True))
            n = z.norm(dim=1, keepdim=True)
            return z / n, torch.log(n[:, 0])
        uA, hA = out(self.A, self.CA); uB, hB = out(self.B, self.CB)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + b ** 2 * (hA - hB) ** 2 + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        b = torch.tensor(K0 if self.length else 0.0, dtype=F64, requires_grad=self.length)
        self.g = torch.tensor(0.5 if self.partial else 0.0, dtype=F64, requires_grad=self.partial)
        with torch.no_grad():
            self.lw = self.V @ q; d = self.distance(kappa, b, idx); D = torch.stack([d, torch.ones_like(d)], 1)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        params = [lin, q, kappa] + ([b] if self.length else []) + ([self.g] if self.partial else [])
        opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.V @ q; d = self.distance(kappa, b, idx)
            loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = (self.V @ q).detach()
        self.q, self.kappa, self.b, self.lin, self.g = q.detach(), kappa.detach(), b.detach(), lin.detach(), self.g.detach()
        return self


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=evaluate.CV_REPEATS); a_ = ap.parse_args()
    dev = pick_device(); rep = Report("eval_white_norm", {"repeats": a_.repeats, "seeds": SEEDS, "lam": LAM, "device": dev, "arms": ARMS},
                                      tag="smoke" if a_.repeats < 5 else "")
    Gg, tr, _, _ = train_groupatoms.graphs_with_groups(); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    V, MU = {}, {}
    for name, nets, g in (("count network", [train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg),
                          ("plain network", [train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G)):
        V[name], MU[name] = [], []
        for n in nets:
            with torch.no_grad():
                h = n.latent(o2.encode_all(n, g, tr_s, dev))[0]; sc = h.norm(dim=1).mean().item()
                MU[name].append((h.mean(0) / sc).double().cpu().numpy())
                V[name].append(n.latent(o2.encode_all(n, g, g.dream_rows(), dev))[0].double().cpu().numpy() / sc)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    nA, nB = A.sum(1), B.sum(1); mis = np.abs(np.log(nA) - np.log(nB)); s1 = k_P == 0
    SUB = {"all pairs": None, "setting 1 (share none)": s1, "setting 2 (share >= 1)": k_P >= 1, "share >= 3": k_P >= 3}
    for net in V:
        rep.line(f"{net}: |mu| / mean |v| = {np.mean([np.linalg.norm(m) for m in MU[net]]):.3f} (GS-LF training molecules)")

    # whiteness after centring, no fitting
    mixes = sorted({tuple(sorted(m)) for p in P for m in (p.a, p.b)}); cid = {int(c): i for i, c in enumerate(op.cids_M)}
    Mx = np.zeros((len(mixes), len(cid)))
    for i, m in enumerate(mixes):
        Mx[i, [cid[int(c)] for c in m]] = 1
    used = Mx.sum(0) > 0; POOL = {net: [v[used].mean(0) for v in V[net]] for net in V}
    for net in V:
        rep.line(f"{net}: |DREAM pool mean| / mean |v| = {np.mean([np.linalg.norm(m) for m in POOL[net]]):.3f} ({used.sum()} molecules)")
    sz = Mx.sum(1); rows = []
    for net in V:
        for lab, E in (("raw", np.mean(V[net], 0)), ("centred", np.mean([v - m for v, m in zip(V[net], MU[net])], 0)),
                       ("centred on DREAM pool", np.mean([v - m for v, m in zip(V[net], POOL[net])], 0))):
            Z = (Mx @ E) / sz[:, None]; L = np.linalg.norm(Z, axis=1); U = Z / L[:, None]; c0 = U.mean(0); c0 /= np.linalg.norm(c0); cw = U @ c0
            Ls = L * np.sqrt(sz)
            rows.append([net, lab, f"{cw[sz <= 3].mean():.3f} -> {cw[sz >= 10].mean():.3f}", f"{np.corrcoef(cw, np.log(sz))[0, 1]:+.3f}",
                         f"{np.corrcoef(np.log(L), np.log(sz))[0, 1]:+.3f}", f"{np.corrcoef(np.log(Ls), np.log(sz))[0, 1]:+.3f}"])
    rep.section(f"Whiteness, plain mean templates of {len(mixes)} development mixtures (no fitting)")
    rep.table(["network", "vectors", "cos to mean direction: <= 3 -> >= 10 molecules", "r(cos, log size)", "r(log length, log size)",
               "r(log length x sqrt(n), log size)"], rows)

    REF = "O1.2b (reference, 27)"
    res = {REF: evaluate.cross_validate(P, lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM)
                                        .fit(tr_, y[tr_]).predict(te_), repeats=a_.repeats)}
    gs = {}
    for net in V:
        for arm, kw in ARMS.items():
            k = f"{net}, {arm} ({NUMBERS[arm]})"; gs[k] = []

            kw = dict(kw); Ms0 = POOL[net] if kw.pop("pool", False) else MU[net]

            def f(tr_, te_, Vs=V[net], Ms=Ms0, kw=kw, k=k):
                ms = [WhiteNorm(v, m, A, B, **kw).fit(tr_, y[tr_]) for v, m in zip(Vs, Ms)]; gs[k] += [float(m.g) for m in ms]
                return np.mean([m.predict(te_) for m in ms], 0)
            res[k] = evaluate.cross_validate(P, f, repeats=a_.repeats); rep.line(f"done: {k}")
            if kw["partial"]:
                rep.line(f"  fitted g: median {np.median(gs[k]):.2f}, range {np.min(gs[k]):.2f} to {np.max(gs[k]):.2f}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({a_.repeats} x 5): RMSE / r; changes paired on the same folds")
    out = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None)
            base = None if k == REF else k.split(", ")[0] + ", base (67)"
            out.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                        "" if k == REF else ch(sc(r_, sub, res[REF])), "" if base in (None, k) else ch(sc(r_, sub, res[base]))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from base (same network)"], out)

    rows = []
    for k, r_ in res.items():
        pr = np.zeros(len(P)); cnt = np.zeros(len(P)); np.add.at(pr, r_.pair_N, r_.pred_N); np.add.at(cnt, r_.pair_N, 1)
        pr = pr / np.maximum(cnt, 1)
        rows.append([k, f"{np.corrcoef(pr[s1] - y[s1], mis[s1])[0, 1]:+.3f}", f"{np.corrcoef(pr[s1], mis[s1])[0, 1]:+.3f}"])
    rep.section(f"Setting 1, out-of-fold: size mismatch |log nA - log nB| (label vs size mismatch r {np.corrcoef(y[s1], mis[s1])[0, 1]:+.3f})")
    rep.table(["model", "r(residual, size mismatch)", "r(prediction, size mismatch)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
