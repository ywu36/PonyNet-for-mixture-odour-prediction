# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   W = words (138)   K = mixtures   P = development pairs (514)
#   G = RDKit groups (24)   H = merge-space size (400)
# ---------------------------------------------------------------------------------------------
"""Everything in one table (user, 2026-11-18): O1.2b (reference, the best neuron model) and every O2 setting,
development pairs only (CV.md rule 7), each row as change from O1.2b on the same folds (paired, mixture resampling); subsets by shared molecules (input-defined):
all pairs, share none, share 1-2 (the unexplained setting-2 pairs), share >= 3.

Part "frozen" (standard CV 50 x 5; networks trained on single molecules only, 10 members):
  O2-out, O2N-out, O2C-out                                   mean of decoded singles, B1a distance, line
  O2N-in + agreement, O2N-in + group gain + agreement        (the best O2 row so far, rerun on development pairs)
  O2C-in; O2C-in + agreement; O2C-in + group gain + agreement; O2C-in + group gain + read-out length
  O2C-out [head form]; O2C-out + group gain                  weighted mean of decoded singles (G = 1 or exp(theta . x)),
                                                             quantile map, direction distance, line
Part "trained" (5 folds x 1 repeat = the folds the networks were trained on; 3 members per fold; final-epoch weights):
  O1.2b and the frozen O2-C rows on these folds, then per trained arm:
  in [own head, final], in [own head, best inner-val epoch], out [final, B1a distance]
  in [head, agreement] (learned gain kept), in + group gain + agreement (G = learned gain x exp(theta . x))
  out [head form], out + group gain
Group gain: G(m) = exp(theta . x(m)), x = 24 RDKit groups, theta fitted per fold on its training pairs (lam 1e-3),
as in eval_o2n.py / O1.2b. Paired comparisons (group gain on vs off; in vs out) in a second table.

Env:     mist (GPU)
Usage:   python evaluation/O2/eval_o2_all.py --part frozen
         python evaluation/O2/eval_o2_all.py --part trained      (after the stage-2 rerun; reuses nothing from part frozen)
Writes:  results/eval_o2_all/<date_time>.{txt,json}, results/eval_o2_all/cv_cache/*.pkl
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus")); sys.path.insert(0, str(ROOT / "evaluation/O2"))
sys.path.insert(0, str(ROOT / "training"))
from mixdist import cv, evaluate, features, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM, members  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402

ARMS = [("O2-C-Single+DREAM, alpha 0", "checkpoints/o2d_es/o2c_a0"),
        ("O2-C-Single+DREAM, alpha 0.1", "checkpoints/o2d_es/o2c_a0.1"),
        ("O2-C-Single+DREAM, alpha 1", "checkpoints/o2d_es/o2c_a1"),
        ("O2-C-Single+Scent+Ma+DREAM, direction only", "checkpoints/o2all_es/o2c_dir"),
        ("O2-C-Single+Scent+Ma+DREAM, direction + length", "checkpoints/o2all_es/o2c_len")]


class OutNet:
    """'Merge after the read-out' in O2Head form: decode each molecule, weighted mean of the probabilities."""

    def __init__(self, net):
        self.net = net

    def merge(self, v, mix, g):
        p = torch.sigmoid(self.net.readout(v)[1]); w = mix * g.unsqueeze(0)
        return (w @ p) / w.sum(1, keepdim=True), torch.ones(len(mix), device=v.device)

    def readout(self, p):
        return None, torch.logit(p.clamp(1e-6, 1 - 1e-6))


class LearnedGain:
    """Trained member merged inside with its own learned gain times the group gain."""

    def __init__(self, net, gain_M):
        self.net, self.gain = net, gain_M

    def merge(self, v, mix, g):
        return self.net.merge(v, mix, g * self.gain)

    def readout(self, z):
        return self.net.readout(z)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["frozen", "trained"], required=True)
    a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_o2_all", {"part": a.part, "lam": LAM, "device": dev})
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); S = size_features(P); q_rel = mixture.quantile_per_word(op.probs_MxW)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); assert np.array_equal(G.dream_cids.astype(int), np.asarray(op.cids_M, int))
    mixes = sorted({m for p in P for m in (p.a, p.b)}, key=sorted); mi = {m: k for k, m in enumerate(mixes)}
    rw = {int(c): i for i, c in enumerate(op.cids_M)}
    M = np.zeros((len(mixes), len(op.cids_M)), np.float32)
    for k, m in enumerate(mixes):
        M[k, [rw[c] for c in m]] = 1
    pa = np.array([mi[p.a] for p in P]); pb = np.array([mi[p.b] for p in P]); zero = np.zeros(len(P))
    repeats = evaluate.CV_REPEATS if a.part == "frozen" else 1
    rep.line(f"development pairs {len(P)} (lockbox untouched); folds: 5 x {repeats} (seed 0)")

    cnets, cv_, _, cprobs = members(G, "checkpoints/o2c", dev); cq = o2.QuantileMap(cprobs, dev)
    _, _, _, dprobs = members(G, "checkpoints/o2", dev)
    nnets, nv, _, nprobs = members(G, "checkpoints/o2n", dev); nq = o2.QuantileMap(nprobs, dev)

    def head(nets, v, q, theta, beta, length="agreement"):
        return lambda tr, te: o2.O2Head(nets, v, X, M, pa, pb, q, dev, theta, beta, LAM, length=length).fit(tr, y[tr]).predict(te)

    def out_paper(probs):
        return evaluate.linear_calibration(mixture.paper_distance(P, op.cids_M, probs), y, zero)

    cout = [OutNet(n) for n in cnets]
    rows_f = {"O1.2b (reference)": lambda tr, te: O12b(q_rel, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
              "O2C-out": out_paper(cprobs),
              "O2C-in + agreement": head(cnets, cv_, cq, False, True),
              "O2C-in + group gain + agreement": head(cnets, cv_, cq, True, True),
              "O2C-out [head form]": head(cout, cv_, cq, False, False),
              "O2C-out + group gain": head(cout, cv_, cq, True, False)}
    if a.part == "frozen":
        rows_f = {**{k: rows_f[k] for k in ("O1.2b (reference)",)},
                  "O2-out (retrained O2)": out_paper(dprobs), "O2N-out": out_paper(nprobs), "O2C-out": rows_f["O2C-out"],
                  "O2N-in + agreement": head(nnets, nv, nq, False, True),
                  "O2N-in + group gain + agreement": head(nnets, nv, nq, True, True),
                  "O2C-in, direction only": head(cnets, cv_, cq, False, False),
                  "O2C-in + agreement": rows_f["O2C-in + agreement"],
                  "O2C-in + group gain + agreement": rows_f["O2C-in + group gain + agreement"],
                  "O2C-in + group gain + read-out length": head(cnets, cv_, cq, True, True, "readout"),
                  "O2C-out [head form]": rows_f["O2C-out [head form]"], "O2C-out + group gain": rows_f["O2C-out + group gain"]}
    cache = ROOT / "results" / rep.script / "cv_cache"; cache.mkdir(parents=True, exist_ok=True)
    res = {}

    def run(name, f):
        cf = cache / f"{a.part}_{name.replace(' ', '_').replace('/', '-').replace(',', '').replace('+', 'p')}.pkl"
        if cf.exists():
            res[name] = pickle.load(open(cf, "rb")); return
        t0 = time.time(); fb0 = o2.O2Head.fallbacks
        res[name] = evaluate.cross_validate(P, f, repeats=repeats); pickle.dump(res[name], open(cf, "wb"))
        rep.line(f"{name}: {time.time() - t0:.0f} s, fallback fits {o2.O2Head.fallbacks - fb0}, NaN {int(np.isnan(res[name].pred_N).sum())}")

    for name, f in rows_f.items():
        run(name, f)

    if a.part == "trained":
        folds = list(cv.mixture_holdout_folds(P, k=5, repeats=1, seed=0)); fold_of = {tuple(f.test): k for k, f in enumerate(folds)}
        rows_ = G.dream_rows()
        for arm, d in ARMS:
            saved = {}
            for f in sorted((ROOT / d).glob("fold*_seed*.pt")):
                c = torch.load(f, map_location="cpu", weights_only=False); saved.setdefault(c["fold"], []).append(f)
            if sorted(saved) != list(range(5)) or any(len(v) != 3 for v in saved.values()):
                rep.line(f"{arm}: incomplete, skipped"); continue
            per = {}                                           # fold -> (in nets with learned gain, out nets, v, qmap, ckpts)
            for k, fs in saved.items():
                ms, cs = zip(*[o2j.load(f, dev) for f in fs])
                with torch.no_grad():
                    enc = [o2.encode_all(m.net, G, rows_, dev) for m in ms]
                    v = torch.stack([m.net.latent(e)[0] for m, e in zip(ms, enc)])
                    gl = [torch.exp(m.gain(e)[:, 0]) for m, e in zip(ms, enc)]
                probs = np.mean([c["probs_out"] for c in cs], 0).astype(np.float64)
                per[k] = ([LearnedGain(m.net, g_) for m, g_ in zip(ms, gl)], [OutNet(m.net) for m in ms], v, o2.QuantileMap(probs, dev), cs)

            def by_fold(make):
                return lambda tr, te: make(fold_of[tuple(te)], tr, te)

            def own(key):
                return by_fold(lambda k, tr, te: np.mean([c[key] for c in per[k][4]], 0))

            def outp(k, tr, te):
                d_ = mixture.paper_distance(P, op.cids_M, np.mean([c["probs_out"] for c in per[k][4]], 0).astype(np.float64))
                a_, c_ = evaluate.fit_line(d_[tr], y[tr]); return a_ * d_[te] + c_

            def hd(which, theta, beta):
                return by_fold(lambda k, tr, te: o2.O2Head(per[k][which], per[k][2], X, M, pa, pb, per[k][3], dev, theta, beta, LAM,
                                                           length="agreement").fit(tr, y[tr]).predict(te))
            for tag, f in ((f"{arm}: in [own head, final]", own("pred")), (f"{arm}: in [own head, best inner-val epoch]", own("pred_best")),
                           (f"{arm}: out [final]", by_fold(outp)), (f"{arm}: in [head, agreement]", hd(0, False, True)),
                           (f"{arm}: in + group gain + agreement", hd(0, True, True)),
                           (f"{arm}: out [head form]", hd(1, False, False)), (f"{arm}: out + group gain", hd(1, True, False))):
                run(tag, f)

    ref = res["O1.2b (reference)"]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])          # shared molecules per pair (input-defined subsets)
    SUBSETS = {"all pairs": None, "share no molecule": lambda n: k_P[n] == 0, "share 1-2 molecules": lambda n: (k_P[n] >= 1) & (k_P[n] <= 2),
               "share >= 3 molecules": lambda n: k_P[n] >= 3}
    rep.line(f"pairs by shared molecules: 0 {int((k_P == 0).sum())}, 1-2 {int(((k_P >= 1) & (k_P <= 2)).sum())} "
             f"(1: {int((k_P == 1).sum())}, 2: {int((k_P == 2).sum())}), >= 3 {int((k_P >= 3).sum())}")

    def mask(subset, r_):
        f = SUBSETS[subset]
        return None if f is None else f(r_.pair_N)
    rep.section(f"1  Development pairs, 5 x {repeats} folds: every row as change from O1.2b (paired, same folds)")
    rows = []
    for subset in SUBSETS:
        mk = lambda r_: mask(subset, r_)  # noqa: E731
        for name, r_ in res.items():
            s = evaluate.score_cv(P, r_, reference=None if r_ is ref else ref, mask_N=mk(r_))
            ch = "" if r_ is ref else f"{s['rmse_change']:+.4f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"
            rows.append([name, subset, s["pairs"], f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)

    comp = [("O2C-in + group gain + agreement", "O2C-in + agreement", "group gain, O2-C in"),
            ("O2C-out + group gain", "O2C-out [head form]", "group gain, O2-C out"),
            ("O2N-in + group gain + agreement", "O2N-in + agreement", "group gain, O2-N in"),
            ("O2C-in + agreement", "O2C-out [head form]", "in minus out, O2-C, no group gain"),
            ("O2C-in + group gain + agreement", "O2C-out + group gain", "in minus out, O2-C, with group gain")]
    for arm, _ in ARMS:
        comp += [(f"{arm}: in + group gain + agreement", f"{arm}: in [head, agreement]", f"group gain, {arm}, in"),
                 (f"{arm}: out + group gain", f"{arm}: out [head form]", f"group gain, {arm}, out"),
                 (f"{arm}: in [head, agreement]", f"{arm}: out [head form]", f"in minus out, {arm}, no group gain"),
                 (f"{arm}: in + group gain + agreement", f"{arm}: out + group gain", f"in minus out, {arm}, with group gain"),
                 (f"{arm}: in + group gain + agreement", "O2C-in + group gain + agreement", f"trained minus frozen, {arm}, in + group gain"),
                 (f"{arm}: out + group gain", "O2C-out + group gain", f"trained minus frozen, {arm}, out + group gain")]
    rows = []
    for x1, x0, label in comp:
        if x1 in res and x0 in res:
            for subset in SUBSETS:
                s = evaluate.score_cv(P, res[x1], reference=res[x0], mask_N=mask(subset, res[x1]))
                rows.append([label, subset, f"{s['rmse_change']:+.4f}", f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.section("2  Paired comparisons (same folds): first row minus second")
    rep.table(["comparison", "subset", "RMSE change", "r change [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
