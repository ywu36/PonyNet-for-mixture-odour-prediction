# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   W = words (138)   K = mixtures (800)
#   P = pairs (820)   G = RDKit groups (24)
# ---------------------------------------------------------------------------------------------
"""Sanity check (O1.2b owner's proposal, user asked 2026-11-18): O2-N merged vector keeps its own length into the
decoder; direction and length are split only in the distance. Frozen O2-N members, standard CV (50 x 5).

  "direction in" (current O2-N):  decoder input = sqrt(400) * z / |z|
  "keep length" (proposal):       decoder input = sqrt(400) * z         (|z| = rho: 1 for one molecule, < 1 if they differ)
Rows (all change from B1a++, same folds; existing O2-N rows read from results/eval_o2n/cv_cache):
  O2-out                                      retrained O2, mean of decoded singles (paired check of the normalization cost)
  keep length, direction-only distance
  keep length + read-out length               + beta (h_A - h_B)^2, h = log length of quantile-mapped profile (O1.2b)
  keep length + group gain + read-out length  theta, beta, line
  direction in + group gain + read-out length (like-for-like partner of the row above)
Check: spread after the first decoder BatchNorm and read-out by mixture size for "keep length".

Env:     mist (GPU)
Usage:   python evaluation/O2/eval_o2n_keeplen.py
Writes:  results/eval_o2n_keeplen/<date_time>.{txt,json}
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus")); sys.path.insert(0, str(ROOT / "evaluation/O2"))
from mixdist import data, evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402


class KeepLength:
    """O2-N member whose merge keeps the merged vector's length (no 1/rho rescaling)."""

    def __init__(self, net):
        self.net = net

    def merge(self, v, member, gain=None):
        z = self.net.pool(v, member, gain)
        return self.net.scale * z, z.norm(dim=1)

    def readout(self, x):
        return self.net.readout(x)


def main():
    dev = pick_device(); rep = Report("eval_o2n_keeplen", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    X, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(train, op.cids_M); S = size_features(train); q_rel = mixture.quantile_per_word(op.probs_MxW)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    nnets, nv, _, nprobs = members(G, "checkpoints/o2n", dev)
    _, _, _, dprobs = members(G, "checkpoints/o2", dev)
    knets = [KeepLength(n) for n in nnets]
    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); mi = {m: k for k, m in enumerate(mixes)}
    row = {int(c): i for i, c in enumerate(op.cids_M)}
    M = np.zeros((len(mixes), len(op.cids_M)), np.float32)
    for k, m in enumerate(mixes):
        M[k, [row[c] for c in m]] = 1
    pa = np.array([mi[p.a] for p in train]); pb = np.array([mi[p.b] for p in train]); n_K = M.sum(1)
    nq = o2.QuantileMap(nprobs, dev)

    def head(nets, theta, beta):
        return lambda tr, te: o2.O2Head(nets, nv, X, M, pa, pb, nq, dev, theta, beta, LAM, length="readout").fit(tr, y[tr]).predict(te)

    rep.section("0  Checks (member 0 for the spread; 10-member read-out by size)")
    with torch.no_grad():
        net0 = nnets[0]; Mt = torch.as_tensor(M, device=dev); big = torch.as_tensor(n_K >= 21, device=dev)
        bn = net0.ffn.batchnorms[0]; zs = lambda a: (net0.ffn.linears[0](a) - bn.running_mean) / bn.running_var.sqrt()  # noqa: E731
        x_dir, _ = net0.merge(nv[0], Mt); x_keep, _ = KeepLength(net0).merge(nv[0], Mt)
        rep.line(f"spread after first decoder BatchNorm: single {zs(net0.scale * nv[0]).std(0).mean():.2f}; merged 21-43 "
                 f"direction in {zs(x_dir[big]).std(0).mean():.2f}, keep length {zs(x_keep[big]).std(0).mean():.2f} (O2: 0.21)")
        zero_t = torch.zeros(len(gn), dtype=torch.float64, device=dev); rows = []
        for name, nets in (("direction in", nnets), ("keep length", knets)):
            Y = nq(o2.O2Head(nets, nv, X, M, pa, pb, nq, dev, False, False).profiles(zero_t)).cpu().numpy()
            mean_mol = (M @ nq.qt.transform(nprobs)) / n_K[:, None]
            u = Y / np.linalg.norm(Y, axis=1, keepdims=True); w = u.mean(0); w /= np.linalg.norm(w)
            cm = (Y * mean_mol).sum(1) / np.linalg.norm(Y, axis=1) / np.linalg.norm(mean_mol, axis=1)
            for a, b in [(1, 1), (2, 4), (5, 10), (11, 20), (21, 43)]:
                m = (n_K >= a) & (n_K <= b)
                rows.append([name, f"{a}-{b}", int(m.sum()), f"{np.linalg.norm(Y, axis=1)[m].mean():.2f}", f"{cm[m].mean():.3f}", f"{(u[m] @ w).mean():+.3f}"])
        rep.table(["decoder input", "molecules", "mixtures", "profile length", "cos(own mean of molecules)", "cos(common direction)"], rows)

    zero = np.zeros(len(train))
    variants = {
        "O2-out (retrained O2, mean of decoded singles)": evaluate.linear_calibration(mixture.paper_distance(train, op.cids_M, dprobs), y, zero),
        "keep length, direction-only distance": head(knets, False, False),
        "keep length + read-out length": head(knets, False, True),
        "keep length + group gain + read-out length": head(knets, True, True),
        "direction in + group gain + read-out length": head(nnets, True, True),
    }
    old = ["O2N-out", "O2N-in", "O2N-in + agreement", "O2N-in + read-out length", "O2N-in + group gain + agreement"]
    ref = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q_rel, X, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res, cache, oldc = {}, ROOT / "results" / rep.script / "cv_cache", ROOT / "results/eval_o2n/cv_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in old:
        res[f"(earlier) {name}"] = pickle.load(open(oldc / (name.replace(" ", "_") + ".pkl"), "rb"))
    for name, f in variants.items():
        cf = cache / (name.replace(" ", "_").replace("/", "-").replace(",", "") + ".pkl")
        if cf.exists():
            res[name] = pickle.load(open(cf, "rb")); continue
        t0 = time.time(); fb0 = o2.O2Head.fallbacks
        res[name] = evaluate.cross_validate(train, f); pickle.dump(res[name], open(cf, "wb"))
        rep.line(f"{name}: {time.time() - t0:.0f} s, fallback fits {o2.O2Head.fallbacks - fb0}, NaN {int(np.isnan(res[name].pred_N).sum())}")

    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5), change from B1a++ on the same pairs; then change from O2N-out / O2N-in")
    rows = []
    for subset in ("all pairs", "share no molecule"):
        mk = lambda r_: None if subset == "all pairs" else r_.no_overlap_N  # noqa: E731
        s0 = evaluate.score_cv(train, ref, mask_N=mk(ref))
        rows.append(["B1a++ (baseline)", subset, f"{s0['rmse']:.4f}", f"{s0['r']:.3f}", "", ""])
        for name, r_ in res.items():
            s = evaluate.score_cv(train, r_, reference=ref, mask_N=mk(r_))
            partner = "(earlier) O2N-out" if name.startswith("O2-out") else "(earlier) O2N-in"
            s2 = evaluate.score_cv(train, r_, reference=res[partner], mask_N=mk(r_)) if name != partner and not name.startswith("(earlier)") else None
            rows.append([name, subset, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                         "" if s2 is None else f"vs {partner[10:]}: {s2['r_change']:+.3f} [{s2['r_change_ci'][0]:+.3f}, {s2['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]", "paired r change"], rows)
    rep.save()


if __name__ == "__main__":
    main()
