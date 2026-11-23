# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer
#   W = words (138)
# ---------------------------------------------------------------------------------------------
"""O3-M backbones (minimal, singles only) vs O2-C-Single, on single molecules and on DREAM, with NO group weights.

Mixture = plain mean of the molecules' hidden vectors v (a molecule's pull = its length |v|; nothing chemical is
given to the mixture side). d = sqrt(|u_A - u_B|^2 + beta (h_A - h_B)^2), u = z / |z|, h = log |z|; DREAM fits beta and
the line only (3 numbers). Two places to compare:
  hidden: z = mean v                       (the add-up layer itself)
  POM layer (O2-C only): open-POM's 256 embedding (after 3 FFN layers), averaged there; one layer from the words
  head:   C z = mean C v  (C = the linear word head without its bias; exact for O3-M; for O2-C the deep decoder is not
          linear, so O2-C has the hidden row only)
Prediction = mean over members (the same number, 3, for every backbone). Paired, development pairs, 50 x 5, 4 subsets: change from O1.2b (27 numbers,
reference) and from the O2-C-Single hidden mean (same 3-number form).

Env:     mist
Usage:   python evaluation/O3/eval_o3m.py [--folders checkpoints/o3m_w32 ...]
Writes:  results/eval_o3m/<date_time>.{txt,json}
"""
import argparse
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
    ap = argparse.ArgumentParser(); ap.add_argument("--folders", nargs="+", default=["checkpoints/o3m_w64", "checkpoints/o3m_w128"])
    ap.add_argument("--base", default="o2c", help="backbone folder name whose hidden mean is the second paired reference")
    ap.add_argument("--no-o2c", action="store_true", help="skip O2-C-Single (when it is not the base)")
    ap.add_argument("--members", type=int, default=3, help="members per backbone (same count for all, fair ensembles)")
    a = ap.parse_args(); dev = pick_device()
    rep = Report("eval_o3m", {"folders": a.folders, "base": a.base, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev},
                 tag="" if a.base == "o2c" else f"vs_{a.base}")
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); rows_D = G.dream_rows()
    gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    tr_sample = np.random.default_rng(0).choice(tr, 1000, replace=False)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)  # API only: no group weights used
    A, B = membership(P, op.cids_M)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}

    spaces, single_rows = {}, []
    for folder in ([] if a.no_o2c else ["checkpoints/o2c"]) + a.folders:
        files = sorted((ROOT / folder).glob("member_*.pt"))[:a.members]; name = Path(folder).name
        nets = o2.load_members(files, dev) if name == "o2c" else [train_o3m.load(f, dev) for f in files]
        n_par = sum(p.numel() for p in nets[0].parameters())
        te_auc = [o2.mean_auroc(o2.predict_probs(n, G, te, dev), G.Y[te]) for n in nets]
        tr_auc = [o2.mean_auroc(o2.predict_probs(n, G, tr_sample, dev), G.Y[tr_sample]) for n in nets]
        with torch.no_grad():
            V = [n.latent(o2.encode_all(n, G, rows_D, dev))[0] for n in nets]
            spaces[f"{name} hidden mean"] = [v.double().cpu().numpy() for v in V]
            if name == "o2c":   # open-POM's own map: the 256 embedding after the FFN's 3 layers (the "principal odor map")
                spaces[f"{name} POM layer (256) mean"] = [n.readout(v)[0].double().cpu().numpy() for v, n in zip(V, nets)]
            if name != "o2c":
                spaces[f"{name} head space (C x mean)"] = [(v @ n.head.weight.T).double().cpu().numpy() for v, n in zip(V, nets)]
            L = np.array([v.norm(dim=1).cpu().numpy() for v in V])
        single_rows.append([name, len(nets), n_par, V[0].shape[1], f"{np.mean(te_auc):.4f} ({min(te_auc):.4f}-{max(te_auc):.4f})",
                            f"{np.mean(tr_auc):.4f}", f"{np.mean(tr_auc) - np.mean(te_auc):+.4f}",
                            f"{np.mean(L.std(1) / L.mean(1)):.3f}"])
    rep.section("Single molecules (GS-LF): official held-out vs training-molecule AUROC; spread of |v| over DREAM molecules")
    rep.table(["backbone", "members", "parameters", "hidden size", "held-out AUROC mean (min-max)", "training AUROC", "gap",
               "|v| spread (std / mean)"], single_rows)

    q_op = mixture.quantile_per_word(op.probs_MxW)
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(
        P, lambda tr_, te_: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_)),
           "open-POM baseline (= B1a open-POM, 2 numbers)": evaluate.cross_validate(
        P, lambda tr_, te_: O12b(q_op, X, A, B, False, False, lam=LAM).fit(tr_, y[tr_]).predict(te_))}
    for k, Vs in spaces.items():
        res[k + " (3 numbers)"] = evaluate.cross_validate(
            P, lambda tr_, te_, Vs=Vs: np.mean([O12b(v, X, A, B, False, True, lam=LAM).fit(tr_, y[tr_]).predict(te_) for v in Vs], 0))
        rep.line(f"done {k}")
    ref = res["O1.2b (reference, 27 numbers)"]; base = res[f"{a.base} hidden mean (3 numbers)"]; opb = res["open-POM baseline (= B1a open-POM, 2 numbers)"]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5), no group weights: RMSE / r change [95% CI], positive r change = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if r_ is ref else ch(s_), "" if r_ is opb else ch(sc(r_, sub, opb)),
                         "" if (r_ is ref or r_ is base or r_ is opb) else ch(sc(r_, sub, base))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from open-POM baseline",
               f"change from {a.base} hidden mean"], rows)
    rep.save()


if __name__ == "__main__":
    main()
