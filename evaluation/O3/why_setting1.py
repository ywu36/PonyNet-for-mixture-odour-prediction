# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   W = words (138)   P = DREAM development pairs (514)
#   X = mixtures in the development pairs
# ---------------------------------------------------------------------------------------------
"""Why does the count network + attention fail on setting 1 (pairs sharing no molecule)? (the user, 2026-11-19:
"is it missing normalization / whitening when too many molecules, or failing to attend to the important one?")

Spaces (frozen, seeds 0-2): count network hidden layer (ga_w64), plain network hidden layer (o3m_w64), count network
word profiles (quantile per word, as O1.2b's input).
A. CV arms (50 x 5, fitted on all development pairs as usual, scored per subset):
   hidden, plain mean, angle only (2) / + length weight (3); attention, angle only (67) / + length weight (68);
   word profiles + O1.2b rule (27).  Paired on setting 1 against attention, angle only.
B. Whitening, no fitting: per mixture, cos(template, mean template direction over all development mixtures) vs
   log(size), and template length vs log(size) (template = plain mean), per space.
C. Setting 1 errors (out-of-fold, CV arm A): correlation of the residual (prediction - label) with the size mismatch
   |log n_A - log n_B| and with mean log size; label vs size mismatch alone.
D. Attention concentration (fits on all pairs): effective number of molecules 1 / sum share^2 vs mixture size.

Env:     mist
Usage:   python evaluation/O3/why_setting1.py [--repeats 50]
Writes:  results/why_setting1/<date_time>.{txt,json}
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
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)


class Attn(ecd.ContextReadoutQ):
    """ContextReadoutQ with the length weight switched per instance (direction_only False adds b, 68 numbers)."""

    def __init__(self, V, A, B, direction_only, fit_q=True):
        super().__init__(V, A, B, np.zeros(len(V)), fit_q); self.direction_only, self.fit_q = direction_only, fit_q

    def fit(self, idx, y, max_iter=300):
        ecd.DIRECTION_ONLY = self.direction_only
        if self.fit_q:
            return super().fit(idx, y, max_iter)
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))     # plain mean: q = 0, kappa = 0
        kappa = torch.zeros((), dtype=torch.float64); b = torch.tensor(0.0 if self.direction_only else 0.1, dtype=torch.float64, requires_grad=not self.direction_only)
        with torch.no_grad():
            d = self.distance(kappa, b, idx); D = torch.stack([d, torch.ones_like(d)], 1); lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True); opt = torch.optim.LBFGS([lin] + ([] if self.direction_only else [b]), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); loss = ((lin[0] * self.distance(kappa, b, idx) + lin[1] - y) ** 2).mean(); loss.backward(); return loss
        opt.step(closure); self.q, self.kappa, self.b, self.lin = torch.zeros(self.V.shape[1], dtype=torch.float64), kappa, b.detach(), lin.detach()
        return self


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=evaluate.CV_REPEATS); a_ = ap.parse_args()
    dev = pick_device(); rep = Report("why_setting1", {"repeats": a_.repeats, "seeds": SEEDS, "lam": LAM, "device": dev}, tag="smoke" if a_.repeats < 5 else "")
    Gg, tr, _, _ = train_groupatoms.graphs_with_groups(); G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    V, Q = {}, None
    for name, nets, g in (("count network", [train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS], Gg),
                          ("plain network", [train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS], G)):
        V[name] = []
        for n in nets:
            with torch.no_grad():
                sc = n.latent(o2.encode_all(n, g, tr_s, dev))[0].norm(dim=1).mean().item()
                V[name].append(n.latent(o2.encode_all(n, g, g.dream_rows(), dev))[0].double().cpu().numpy() / sc)
        if name == "count network":
            Q = mixture.quantile_per_word(np.mean([o2.predict_probs(n, g, g.dream_rows(), dev) for n in nets], 0).astype(float))
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    nA, nB = A.sum(1), B.sum(1); mis = np.abs(np.log(nA) - np.log(nB)); msz = 0.5 * (np.log(nA) + np.log(nB))
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1, "share >= 3": k_P >= 3}
    s1 = k_P == 0
    rep.line(f"development pairs {len(P)}; setting 1 {s1.sum()}; setting 1 mixture sizes: median {np.median(np.r_[nA[s1], nB[s1]]):.0f}, "
             f"range {int(np.r_[nA[s1], nB[s1]].min())}-{int(np.r_[nA[s1], nB[s1]].max())}; label vs size mismatch in setting 1: "
             f"r {np.corrcoef(mis[s1], y[s1])[0, 1]:+.3f}")

    # A. CV arms
    def ens(Vs, direction_only, fit_q):
        return lambda tr_, te_: np.mean([Attn(v, A, B, direction_only, fit_q).fit(tr_, y[tr_]).predict(te_) for v in Vs], 0)
    REF = "O1.2b (reference, 27)"; BASE = "count network, attention, angle only (67)"
    arms = {REF: lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_),
            "count network, word profiles + O1.2b rule (27)": lambda tr_, te_: O12b(Q, X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_)}
    for net in ("count network", "plain network"):
        arms[f"{net}, plain mean, angle only (2)"] = ens(V[net], True, False)
        arms[f"{net}, plain mean + length (3)"] = ens(V[net], False, False)
        arms[f"{net}, attention, angle only (67)"] = ens(V[net], True, True)
        arms[f"{net}, attention + length (68)"] = ens(V[net], False, True)
    res = {}
    for k, f in arms.items():
        res[k] = evaluate.cross_validate(P, f, repeats=a_.repeats); rep.line(f"done: {k}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"A. DREAM development pairs ({a_.repeats} x 5): RMSE / r; change from the count network's attention (angle only), paired")
    out = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None)
            out.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                        "" if k == BASE else ch(sc(r_, sub, res[BASE]))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from count attention, angle only"], out)

    # C. setting 1 out-of-fold residuals vs size
    rows = []
    for k, r_ in res.items():
        pr = np.zeros(len(P)); cnt = np.zeros(len(P))
        np.add.at(pr, r_.pair_N, r_.pred_N); np.add.at(cnt, r_.pair_N, 1); pr = pr / np.maximum(cnt, 1); e = pr - y
        rows.append([k, f"{np.corrcoef(e[s1], mis[s1])[0, 1]:+.3f}", f"{np.corrcoef(e[s1], msz[s1])[0, 1]:+.3f}",
                     f"{np.corrcoef(pr[s1], mis[s1])[0, 1]:+.3f}"])
    rep.section("C. Setting 1, out-of-fold: residual (prediction - label) vs size mismatch |log nA - log nB| and mean log size; "
                f"prediction vs size mismatch (label vs size mismatch: {np.corrcoef(mis[s1], y[s1])[0, 1]:+.3f})")
    rep.table(["model", "r(residual, size mismatch)", "r(residual, mean size)", "r(prediction, size mismatch)"], rows)

    # B. whitening, no fitting
    mixes = sorted({tuple(sorted(m)) for p in P for m in (p.a, p.b)}); cid = {int(c): i for i, c in enumerate(op.cids_M)}
    Mx = np.zeros((len(mixes), len(cid))); [Mx.__setitem__((i, [cid[int(c)] for c in m]), 1) for i, m in enumerate(mixes)]
    lsz = np.log(Mx.sum(1)); rows = []
    spaces = {"count network hidden": np.mean([v for v in V["count network"]], 0), "plain network hidden": np.mean([v for v in V["plain network"]], 0),
              "count network words (quantile)": Q, "open-POM words (O1.2b input)": mixture.quantile_per_word(op.probs_MxW)}
    for nm, E in spaces.items():
        Z = (Mx @ E) / Mx.sum(1, keepdims=True); L = np.linalg.norm(Z, axis=1); U = Z / L[:, None]; c0 = U.mean(0); c0 /= np.linalg.norm(c0)
        cw = U @ c0; big, small = lsz >= np.log(10), lsz <= np.log(3)
        rows.append([nm, f"{np.corrcoef(cw, lsz)[0, 1]:+.3f}", f"{cw[small].mean():.3f} -> {cw[big].mean():.3f}",
                     f"{np.corrcoef(np.log(L), lsz)[0, 1]:+.3f}"])
    rep.section(f"B. Whitening, no fitting ({len(mixes)} development mixtures): template = plain mean; 'white' = mean direction over all mixtures")
    rep.table(["space", "r(cos to white, log size)", "cos to white: <= 3 molecules -> >= 10", "r(log length, log size)"], rows)

    # D. attention concentration, fits on all pairs
    allp = np.arange(len(P)); rows = []
    for net in ("count network", "plain network"):
        effs, tops, kap = [], [], []
        for v in V[net]:
            m = Attn(v, A, B, True, True).fit(allp, y); kap.append(float(m.kappa))
            lw = (torch.as_tensor(v) @ m.q).numpy(); C = m._context(torch.as_tensor(Mx)).numpy()
            for row, c in zip(Mx, C):
                s = row > 0; l_ = lw[s] + m.kappa.item() * c[s]
                w = np.exp(l_ - l_.max()); w = w / w.sum(); effs.append(1 / (w ** 2).sum() / row.sum()); tops.append(w.max())
        rep.line(f"{net}: fitted kappa per seed {', '.join(f'{k:+.2f}' for k in kap)}")
        effs, tops = np.array(effs).reshape(len(V[net]), -1).mean(0), np.array(tops).reshape(len(V[net]), -1).mean(0)
        for lo, hi in ((1, 3), (4, 9), (10, 43)):
            sel = (Mx.sum(1) >= lo) & (Mx.sum(1) <= hi)
            rows.append([net, f"{lo}-{hi}", int(sel.sum()), f"{effs[sel].mean():.2f}", f"{tops[sel].mean():.2f}"])
    rep.section("D. Attention concentration (q fitted on all development pairs): effective molecules / size, top molecule's share")
    rep.table(["network", "mixture size", "mixtures", "effective molecules / size", "top share"], rows)
    rep.save()


if __name__ == "__main__":
    main()
