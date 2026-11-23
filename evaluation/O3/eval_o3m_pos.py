# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   H = hidden layer (64)   P = development pairs (514)
# ---------------------------------------------------------------------------------------------
"""Receptor-like hidden layer (checkpoints/o3m_w64pos: v = softplus(.) >= 0, not centred) on DREAM, frozen network,
no group weights. Mixtures:
  mean        z = mean of v                                   (as before; 3 numbers: beta + line)
  saturation  z = s / (k + s), s = SUM of the members' v      (receptors saturate: a few strong molecules dominate,
              k = per-unit median activation over GS-LF singles, fixed, no DREAM label; 3 numbers)
d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2) on z; line. Mean over the 3 members; 50 x 5; paired vs O1.2b, the open-POM
baseline and the centred width-64 network's plain mean; 4 subsets.

Env:     mist
Usage:   python evaluation/O3/eval_o3m_pos.py
Writes:  results/eval_o3m_pos/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import train_o3m  # noqa: E402


class Saturating(O12b):
    """O1.2b's distance on z = s / (k + s), s = sum of member vectors (no group weights)."""

    def __init__(self, v_MxH, k_H, x, A, B, lam):
        super().__init__(v_MxH, x, A, B, False, True, lam=lam); self.k = torch.as_tensor(k_H)

    def distance(self, theta, kk, idx):
        beta = kk ** 2

        def out(Mx):
            s = Mx[idx] @ self.q; z = s / (self.k + s); n = z.norm(dim=1, keepdim=True)
            return z / n, torch.log(n[:, 0])
        uA, hA = out(self.A); uB, hB = out(self.B)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + beta * (hA - hB) ** 2 + 1e-12)


def vectors(folder, G, dev):
    V, K = [], []
    for f in sorted((ROOT / f"checkpoints/{folder}").glob("member_*.pt")):
        net = train_o3m.load(f, dev)
        with torch.no_grad():
            V.append(net.latent(o2.encode_all(net, G, G.dream_rows(), dev))[0].double().cpu().numpy())
            K.append(np.median(net.latent(o2.encode_all(net, G, np.arange(G.n_gslf), dev))[0].double().cpu().numpy(), 0))
    return V, K


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--folder", default="o3m_w64pos"); a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_o3m_pos", {"folder": a.folder, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev}, tag=a.folder)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)  # API only
    A, B = membership(P, op.cids_M)
    Vc, _ = vectors("o3m_w64", G, dev); Vp, Kp = vectors(a.folder, G, dev); Kp = [np.maximum(k, 1e-3) for k in Kp]
    rep.line(f"positive network: members {len(Vp)}; DREAM activations mean {np.mean([v.mean() for v in Vp]):.3f}, "
             f"fraction near 0 (< 0.05) [{a.folder}] {np.mean([(v < 0.05).mean() for v in Vp]):.3f}; k (median per unit) mean {np.mean(Kp):.3f}")
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    q_op = mixture.quantile_per_word(op.probs_MxW)
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)),
           "open-POM baseline (2 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, False, False, lam=LAM).fit(tr, y[tr]).predict(te)),
           "width 64 centred, mean (3 numbers)": evaluate.cross_validate(
               P, lambda tr, te: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in Vc], 0)),
           "width 64 positive, mean (3 numbers)": evaluate.cross_validate(
               P, lambda tr, te: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in Vp], 0)),
           "width 64 positive, saturating sum (3 numbers)": evaluate.cross_validate(
               P, lambda tr, te: np.mean([Saturating(v, k, X, A, B, LAM).fit(tr, y[tr]).predict(te) for v, k in zip(Vp, Kp)], 0))}
    ref, opb, cm = res["O1.2b (reference, 27 numbers)"], res["open-POM baseline (2 numbers)"], res["width 64 centred, mean (3 numbers)"]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if r_ is ref else ch(s_), "" if r_ is opb else ch(sc(r_, sub, opb)),
                         "" if any(r_ is x for x in (ref, opb, cm)) else ch(sc(r_, sub, cm))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from open-POM baseline", "change from centred mean"], rows)
    rep.save()


if __name__ == "__main__":
    main()
