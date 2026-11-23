# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  R = O2 members (10)   M = DREAM molecules (235)   W = words (138)
#   H = encoder output (400)   G = RDKit groups (24)   K = mixtures in the training pairs   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""O2 step 1, standard CV (design/o2_design.md): open-POM retrained in PyTorch (training/train_o2.py) with the
mixture pooled inside the network (after set2set, before the feed-forward read-out).

Rows, all declared in the design before running and all reported (change from B1a++ on the same folds):
  references: B1a (released open-POM), O1.2b (theta + length)
  O2-out                         mean of O2's single-molecule profiles, B1a distance, line
  O2-in                          pooled inside (gain 1), read out, B1a distance, line
  O2-in + group gain + length    G = exp(theta . x), 24 RDKit groups, O1.2b distance
  O2-in + encoder gain + length  G = exp(v . k), k = 24 principal components of e (members stacked), sd 1
Penalty lam 1e-3 on theta / v; beta = b^2, b starts at 0.1. Subsets: all pairs, share no molecule.
Checks: torch quantile map = sklearn's; read-out check (cos of pooled-inside profile with mean of molecules'
profiles, and length, by mixture size). V50 / T46 not scored.

Env:     mist (GPU)
Usage:   python evaluation/O2/eval_o2.py
Writes:  results/eval_o2/<date_time>.{txt,json}
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402

LAM = 1e-3
N_PC = 24


def main():
    dev = pick_device(); rep = Report("eval_o2", {"lam": LAM, "n_pc": N_PC, "repeats": evaluate.CV_REPEATS, "device": dev})
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    X_MxG, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(train, op.cids_M); S = size_features(train)
    q_rel = mixture.quantile_per_word(op.probs_MxW)

    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    assert np.array_equal(G.dream_cids.astype(int), np.asarray(op.cids_M, int))
    paths = sorted((ROOT / "checkpoints/o2").glob("member_*.pt"))
    nets = o2.load_members(paths, dev); rows_M = G.dream_rows()
    with torch.no_grad():
        e_RxMxH = torch.stack([o2.encode_all(n, G, rows_M, dev) for n in nets])
        probs_MxW = torch.stack([torch.sigmoid(n.readout(e)[1]) for n, e in zip(nets, e_RxMxH)]).mean(0).double()
    probs_np = probs_MxW.cpu().numpy()
    rep.line(f"O2 members: {len(nets)} ({', '.join(p.stem for p in paths)}); groups {len(gn)}; device {dev}")

    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); mi = {m: k for k, m in enumerate(mixes)}
    row = {int(c): i for i, c in enumerate(op.cids_M)}
    mix_KxM = np.zeros((len(mixes), len(op.cids_M)), np.float32)
    for k, m in enumerate(mixes):
        mix_KxM[k, [row[c] for c in m]] = 1
    pa_P = np.array([mi[p.a] for p in train]); pb_P = np.array([mi[p.b] for p in train])
    n_K = mix_KxM.sum(1)

    rep.section("0  Checks")
    qmap = o2.QuantileMap(probs_np, dev)
    test = np.vstack([probs_np, mix_KxM @ probs_np / n_K[:, None]])
    diff = np.abs(qmap(torch.as_tensor(test, device=dev)).cpu().numpy() - qmap.qt.transform(test)).max()
    rep.line(f"torch quantile map vs sklearn (235 molecules + {len(mixes)} mean profiles): max |difference| {diff:.2e}")
    head0 = o2.O2Head(nets, e_RxMxH, X_MxG, mix_KxM, pa_P, pb_P, qmap, dev, False, False)
    with torch.no_grad():
        r_in_KxW = head0.profiles(torch.zeros(len(gn), dtype=torch.float64, device=dev))
        y_in = qmap(r_in_KxW).cpu().numpy()
    y_out = (mix_KxM @ qmap.qt.transform(probs_np)) / n_K[:, None]
    single = n_K == 1
    rep.line(f"single-molecule mixtures ({single.sum()}): max |pooled-inside profile - molecule profile| "
             f"{np.abs(y_in[single] - qmap.qt.transform(probs_np)[mix_KxM[single].argmax(1)]).max():.2e}")
    cos = (y_in * y_out).sum(1) / np.linalg.norm(y_in, axis=1) / np.linalg.norm(y_out, axis=1)
    bins = [(1, 1), (2, 4), (5, 10), (11, 20), (21, 43)]
    rep.table(["molecules", "mixtures", "cos(inside, mean of molecules)", "length inside", "length of mean"],
              [[f"{a}-{b}", int(((n_K >= a) & (n_K <= b)).sum()),
                f"{cos[(n_K >= a) & (n_K <= b)].mean():.3f}",
                f"{np.linalg.norm(y_in, axis=1)[(n_K >= a) & (n_K <= b)].mean():.2f}",
                f"{np.linalg.norm(y_out, axis=1)[(n_K >= a) & (n_K <= b)].mean():.2f}"] for a, b in bins])

    E = e_RxMxH.permute(1, 0, 2).reshape(len(rows_M), -1).cpu().numpy()
    E = E - E.mean(0); U, s, _ = np.linalg.svd(E, full_matrices=False)
    k_MxN = U[:, :N_PC] * s[:N_PC]; k_MxN = k_MxN / k_MxN.std(0)
    rep.line(f"encoder PCs: first {N_PC} explain {(s[:N_PC] ** 2).sum() / (s ** 2).sum():.3f} of the variance of e "
             f"({E.shape[1]} columns)")

    b1a_d = mixture.paper_distance(train, op.cids_M, op.probs_MxW)
    o2out_d = mixture.paper_distance(train, op.cids_M, probs_np)
    zero = np.zeros(len(train))
    variants = {
        "B1a (released open-POM)": evaluate.linear_calibration(b1a_d, y, zero),
        "O1.2b theta + length": lambda tr, te: O12b(q_rel, X_MxG, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
        "O2-out": evaluate.linear_calibration(o2out_d, y, zero),
        "O2-in": lambda tr, te: o2.O2Head(nets, e_RxMxH, X_MxG, mix_KxM, pa_P, pb_P, qmap, dev, False, False).fit(tr, y[tr]).predict(te),
        "O2-in + group gain + length": lambda tr, te: o2.O2Head(nets, e_RxMxH, X_MxG, mix_KxM, pa_P, pb_P, qmap, dev, True, True, LAM).fit(tr, y[tr]).predict(te),
        "O2-in + encoder gain + length": lambda tr, te: o2.O2Head(nets, e_RxMxH, k_MxN, mix_KxM, pa_P, pb_P, qmap, dev, True, True, LAM).fit(tr, y[tr]).predict(te),
    }
    ref = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q_rel, X_MxG, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res = {}
    for name, f in variants.items():
        t0 = time.time(); res[name] = evaluate.cross_validate(train, f); print(f"  done {name} {time.time() - t0:.0f} s", flush=True)

    rep.section(f"1  Standard CV ({evaluate.CV_REPEATS} x 5, whole mixtures held out), change from B1a++ on the same pairs")
    rows, out = [], {}
    for subset in ("all pairs", "share no molecule"):
        def mask(r_):
            return None if subset == "all pairs" else r_.no_overlap_N
        s0 = evaluate.score_cv(train, ref, mask_N=mask(ref))
        rows.append(["B1a++ (baseline)", subset, s0["pairs"], f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
        for name in variants:
            s = evaluate.score_cv(train, res[name], reference=ref, mask_N=mask(res[name]))
            out[f"{name} | {subset}"] = {k: s[k] for k in ("rmse", "r", "r_ci", "rmse_change", "r_change", "r_change_ci")}
            rows.append([name, subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]"], rows)

    rep.section("2  Full-data fits (parameters only, not scores)")
    allp = np.arange(len(train))
    fg = o2.O2Head(nets, e_RxMxH, X_MxG, mix_KxM, pa_P, pb_P, qmap, dev, True, True, LAM).fit(allp, y)
    fe = o2.O2Head(nets, e_RxMxH, k_MxN, mix_KxM, pa_P, pb_P, qmap, dev, True, True, LAM).fit(allp, y)
    th = fg.theta.cpu().numpy(); order = np.argsort(th)
    rep.line(f"group gain: beta {fg.beta:.3f}; louder {', '.join(f'{gn[i]} {np.exp(th[i]):.2f}' for i in order[::-1][:4])}; "
             f"quieter {', '.join(f'{gn[i]} {np.exp(th[i]):.2f}' for i in order[:4])}")
    ge = np.exp(k_MxN @ fe.theta.cpu().numpy()); gg = np.exp(X_MxG @ th)
    rep.line(f"encoder gain: beta {fe.beta:.3f}; G spans x{np.percentile(ge / np.median(ge), 10):.2f} to x{np.percentile(ge / np.median(ge), 90):.2f} "
             f"(10th-90th pct); r(log G encoder, log G groups) over 235 molecules {np.corrcoef(np.log(ge), np.log(gg))[0, 1]:+.3f}")
    rep.value("cv", out); rep.value("quantile_check", float(diff))
    rep.save()


if __name__ == "__main__":
    main()
