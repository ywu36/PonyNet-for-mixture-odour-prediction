# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   H = hidden layer (64)   C = latent directions (24)
#   P = development pairs (514)
# ---------------------------------------------------------------------------------------------
"""O1.2b's mixture side on the frozen width-64 network, with the 24 hand-made groups replaced by 24 LATENT directions of
the network's own hidden vector (the user, 2026-11-19: learn the group level latently; mixture side <= 27 numbers).

  profile of a molecule  = its hidden vector v (64, centred)            (O1.2b: quantile word profile of open-POM)
  weight in a mixture    = exp(theta . c_m), c_m = the molecule's scores on the top 24 principal directions of v,
                           directions fixed from all GS-LF single molecules (no DREAM label used)
  mixture z = weighted mean of v; d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2); line  -> 24 + beta + line = 27 numbers
Fitted per fold on DREAM training pairs exactly as O1.2b (same code, same ridge), 50 x 5, mean over the 3 members.
Rows: O1.2b, open-POM baseline, width 64 plain mean (3 numbers), width 64 + latent group weights (27 numbers).

Env:     mist
Usage:   python evaluation/O3/eval_o3m_latent_groups.py [--n-dir 4 8 16 24]
Writes:  results/eval_o3m_latent_groups/<date_time>.{txt,json}
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

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n-dir", type=int, nargs="+", default=[24])
    ap.add_argument("--folder", default="o3m_w64"); a = ap.parse_args()
    for n in a.n_dir:
        run(n, a.folder)


def run(N_DIR, folder="o3m_w64"):
    dev = pick_device(); rep = Report("eval_o3m_latent_groups", {"n_directions": N_DIR, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev}, tag=f"{folder}_dir{N_DIR}")
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M)
    VD, CD, expl = [], [], []
    for f in sorted((ROOT / f"checkpoints/{folder}").glob("member_*.pt")):
        net = train_o3m.load(f, dev)
        with torch.no_grad():
            Vs = net.latent(o2.encode_all(net, G, np.arange(G.n_gslf), dev))[0].double().cpu().numpy()
            Vd = net.latent(o2.encode_all(net, G, G.dream_rows(), dev))[0].double().cpu().numpy()
        mu = Vs.mean(0); _, s, Wt = np.linalg.svd(Vs - mu, full_matrices=False)
        C = (Vd - mu) @ Wt[:N_DIR].T; C = C / C.std(0)      # unit-scale scores, like the 0/1 groups' scale
        VD.append(Vd); CD.append(C); expl.append((s[:N_DIR] ** 2).sum() / (s ** 2).sum())
    rep.line(f"members {len(VD)}; top {N_DIR} directions explain {np.mean(expl):.3f} of the hidden-vector variance (GS-LF singles)")
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    q_op = mixture.quantile_per_word(op.probs_MxW)
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)),
           "open-POM baseline (2 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, False, False, lam=LAM).fit(tr, y[tr]).predict(te)),
           "width 64, plain mean (3 numbers)": evaluate.cross_validate(
               P, lambda tr, te: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in VD], 0)),
           f"width 64, latent group weights ({N_DIR} directions, 27 numbers)": evaluate.cross_validate(
               P, lambda tr, te: np.mean([O12b(v, c, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te) for v, c in zip(VD, CD)], 0)),
           "width 64, hand-made group weights (24 RDKit groups, 27 numbers; reference only)": evaluate.cross_validate(
               P, lambda tr, te: np.mean([O12b(v, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te) for v in VD], 0))}
    ref, opb, pm = res["O1.2b (reference, 27 numbers)"], res["open-POM baseline (2 numbers)"], res["width 64, plain mean (3 numbers)"]

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
                         "" if any(r_ is x for x in (ref, opb, pm)) else ch(sc(r_, sub, pm))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from open-POM baseline", "change from plain mean"], rows)
    rep.save()


if __name__ == "__main__":
    main()
