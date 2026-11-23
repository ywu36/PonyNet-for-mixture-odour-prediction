# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  R = members (10)   M = DREAM molecules (235)   W = words (138)
#   D = latent size (VAE 32) or encoder size (O2 400)   G = RDKit groups (24)   K = mixtures (800)   P = pairs (820)
# ---------------------------------------------------------------------------------------------
"""O2-VAE step C, standard CV (design/o2_vae_design.md): mixtures merged in the VAE bottleneck, then decoded.

Rows declared in the design before running, all reported (change from B1a++ on the same folds):
  references: B1a (released open-POM), O1.2b theta + length, O2-in (deterministic O2, gain 1)
  VAE-out                              mean of decoded single molecules, B1a distance, line
  VAE-in mean                          decode(mean of mu), B1a distance, line
  VAE-in precision                     decode(sum P mu / sum P), B1a distance, line
  VAE-in precision + length            as above, O1.2b distance (beta), line
  VAE-in mean + group gain + length    G = exp(theta . x), O1.2b distance (theta, beta), line
lam 1e-3, b0 0.1 as O2. Read-out check by mixture size: length, cos with the mean of the molecules' profiles,
cos with the common direction over the 800 mixtures, latent norm. V50 / T46 not scored.

Env:     mist (GPU)
Usage:   python evaluation/O2/eval_o2_vae.py
Writes:  results/eval_o2_vae/<date_time>.{txt,json}
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


def members(G, folder, dev):
    nets = o2.load_members(sorted((ROOT / folder).glob("member_*.pt")), dev); rows = G.dream_rows()
    lat = [o2.latent_all(n, G, rows, dev) for n in nets]
    mu = torch.stack([m for m, _ in lat]); prec = None if lat[0][1] is None else torch.stack([p for _, p in lat])
    with torch.no_grad():
        probs = torch.stack([torch.sigmoid(n.readout(m)[1]) for n, m in zip(nets, mu)]).mean(0).double().cpu().numpy()
    return nets, mu, prec, probs


def main():
    dev = pick_device(); rep = Report("eval_o2_vae", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    train = data.load_train(); y = evaluate.labels(train); op = features.load("openpom")
    ids = [str(int(c)) for c in op.cids_M]
    X_MxG, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(train, op.cids_M); S = size_features(train); q_rel = mixture.quantile_per_word(op.probs_MxW)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    assert np.array_equal(G.dream_cids.astype(int), np.asarray(op.cids_M, int))
    vnets, vmu, vprec, vprobs = members(G, "checkpoints/o2_vae", dev)
    dnets, dmu, _, dprobs = members(G, "checkpoints/o2", dev)
    rep.line(f"O2-VAE members {len(vnets)} (latent {vmu.shape[-1]}); O2 members {len(dnets)}; device {dev}")

    mixes = sorted({m for p in train for m in (p.a, p.b)}, key=sorted); mi = {m: k for k, m in enumerate(mixes)}
    row = {int(c): i for i, c in enumerate(op.cids_M)}
    mix_KxM = np.zeros((len(mixes), len(op.cids_M)), np.float32)
    for k, m in enumerate(mixes):
        mix_KxM[k, [row[c] for c in m]] = 1
    pa_P = np.array([mi[p.a] for p in train]); pb_P = np.array([mi[p.b] for p in train]); n_K = mix_KxM.sum(1)
    vq, dq = o2.QuantileMap(vprobs, dev), o2.QuantileMap(dprobs, dev)

    def head(prec, use_theta=False, use_beta=False):
        return lambda: o2.O2Head(vnets, vmu, X_MxG, mix_KxM, pa_P, pb_P, vq, dev, use_theta, use_beta, LAM, prec_RxMxH=prec)

    rep.section("0  Read-out check by mixture size (quantile-mapped profiles)")
    zero_t = torch.zeros(len(gn), dtype=torch.float64, device=dev)
    with torch.no_grad():
        prof = {"VAE-in mean": vq(head(None)().profiles(zero_t)).cpu().numpy(),
                "VAE-in precision": vq(head(vprec)().profiles(zero_t)).cpu().numpy(),
                "O2-in": dq(o2.O2Head(dnets, dmu, X_MxG, mix_KxM, pa_P, pb_P, dq, dev, False, False).profiles(zero_t)).cpu().numpy()}
        zmean = torch.stack([n.pool(m, torch.as_tensor(mix_KxM, device=dev)) for n, m in zip(vnets, vmu)]).norm(dim=2).mean(0).cpu().numpy()
    prof["VAE-out (mean of molecules)"] = (mix_KxM @ vq.qt.transform(vprobs)) / n_K[:, None]
    ref_mean = prof["VAE-out (mean of molecules)"]
    bins = [(1, 1), (2, 4), (5, 10), (11, 20), (21, 43)]
    rows = []
    for name, Y in prof.items():
        u = Y / np.linalg.norm(Y, axis=1, keepdims=True); white = u.mean(0); white /= np.linalg.norm(white)
        cos_mean = (Y * ref_mean).sum(1) / np.linalg.norm(Y, axis=1) / np.linalg.norm(ref_mean, axis=1)
        for a, b in bins:
            m = (n_K >= a) & (n_K <= b)
            rows.append([name, f"{a}-{b}", int(m.sum()), f"{np.linalg.norm(Y, axis=1)[m].mean():.2f}", f"{cos_mean[m].mean():.3f}",
                         f"{(u[m] @ white).mean():+.3f}", f"{zmean[m].mean():.2f}" if name == "VAE-in mean" else ""])
    rep.table(["profile", "molecules", "mixtures", "length", "cos(mean of molecules)", "cos(common direction)", "|z| (VAE mean)"], rows)
    lv = torch.log(1 / vprec).mean(0).cpu().numpy()
    rep.line(f"VAE posterior: mean sigma {np.exp(0.5 * lv).mean():.3f}; active latent dims (mean sigma < 0.9) "
             f"{int((np.exp(0.5 * lv).mean(0) < 0.9).sum())} of {lv.shape[1]}")

    b1a_d = mixture.paper_distance(train, op.cids_M, op.probs_MxW); zero = np.zeros(len(train))
    variants = {
        "B1a (released open-POM)": evaluate.linear_calibration(b1a_d, y, zero),
        "O1.2b theta + length": lambda tr, te: O12b(q_rel, X_MxG, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
        "O2-in (deterministic, gain 1)": lambda tr, te: o2.O2Head(dnets, dmu, X_MxG, mix_KxM, pa_P, pb_P, dq, dev, False, False).fit(tr, y[tr]).predict(te),
        "VAE-out": evaluate.linear_calibration(mixture.paper_distance(train, op.cids_M, vprobs), y, zero),
        "VAE-in mean": lambda tr, te: head(None)().fit(tr, y[tr]).predict(te),
        "VAE-in precision": lambda tr, te: head(vprec)().fit(tr, y[tr]).predict(te),
        "VAE-in precision + length": lambda tr, te: head(vprec, False, True)().fit(tr, y[tr]).predict(te),
        "VAE-in mean + group gain + length": lambda tr, te: head(None, True, True)().fit(tr, y[tr]).predict(te),
    }
    ref = evaluate.cross_validate(train, lambda tr, te: B1aPlus(q_rel, X_MxG, A, B, S, LAM).fit(tr, y[tr]).predict(te))
    res, cache = {}, ROOT / "results" / rep.script / "cv_cache"
    cache.mkdir(parents=True, exist_ok=True)
    import pickle
    for name, f in variants.items():
        cf = cache / (name.replace(" ", "_").replace("/", "-") + ".pkl")
        if cf.exists():
            res[name] = pickle.load(open(cf, "rb")); print(f"  cached {name}", flush=True); continue
        t0 = time.time(); fb0 = o2.O2Head.fallbacks
        res[name] = evaluate.cross_validate(train, f)
        nan = int(np.isnan(res[name].pred_N).sum())
        print(f"  done {name} {time.time() - t0:.0f} s; fallbacks {o2.O2Head.fallbacks - fb0}; NaN predictions {nan}", flush=True)
        rep.line(f"{name}: fallback fits {o2.O2Head.fallbacks - fb0} of {5 * evaluate.CV_REPEATS}; NaN predictions {nan}")
        pickle.dump(res[name], open(cf, "wb"))

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
    for name, h in (("VAE-in precision + length", head(vprec, False, True)()), ("VAE-in mean + group gain + length", head(None, True, True)())):
        h.fit(allp, y); rep.line(f"{name}: beta {h.beta:.3f}")
    rep.value("cv", out)
    rep.save()


if __name__ == "__main__":
    main()
