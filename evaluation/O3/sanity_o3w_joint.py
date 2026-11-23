# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = GoodScents + Ma molecules   M = DREAM molecules (235)   G = RDKit groups (24)
#   P = DREAM development pairs (514)   K = members
# ---------------------------------------------------------------------------------------------
"""Sanity checks for eval_o3w_joint.py (fits only, no cross-validation; the user 2026-11-19: "have a sanity test").

1. Reproduce: DREAM term off -> the joint code must give the earlier separate fits (eval_o3w.py, Adam):
   GoodScents lean held-out AUROC ~0.588 (GoodScents only), ~0.479 (GoodScents + Ma).
2. Dial: c = 0, 1, 3, 10, 100 with all 514 DREAM pairs -> lean AUROC should move from the DREAM-only value toward the
   GoodScents + Ma value, and DREAM training RMSE should rise (DREAM numbers are training fit, not a test).
Per row: GoodScents lean held-out AUROC (both molecules unseen), Ma sign AUROC (training mixtures), DREAM training RMSE /
r, and the correlation of the group weights with the earlier GoodScents-only and GoodScents + Ma weights.

Env:     mist
Usage:   python evaluation/O3/sanity_o3w_joint.py [--jobs 5]
Writes:  results/sanity_o3w_joint/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o3w_joint import Joint  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o3t import scent_rel_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=5); a_ = ap.parse_args()
    dev = pick_device(); rep = Report("sanity_o3w_joint", {"encoder_device": dev, "fit_device": "cpu"})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    lab = scent_rel_labels(G, S)
    q_tr = np.where((S.pair_split == 0) & ~np.isnan(lab))[0]; q_te = np.where((S.pair_split == 1) & ~np.isnan(lab))[0]
    used = np.unique(np.r_[S.pair_a[q_tr], S.pair_b[q_tr], S.pair_a[q_te], S.pair_b[q_te], S.ma_a, S.ma_b])
    pos = {u: i for i, u in enumerate(used)}; re_ = np.vectorize(pos.get)
    XN = o3.group_matrix(z["smiles"][used], gn).astype(float)
    nets = o2.load_members(sorted((ROOT / "checkpoints/o2c").glob("member_*.pt")), dev)
    VN, VM = [], []
    for n in nets:
        v = np.zeros((len(used), 400))
        for i in range(0, len(used), 512):
            u = used[i:i + 512]; x, e, s, d, g = S.batch(u, dev)
            with torch.no_grad():
                v[i:i + len(u)] = n.latent(n.encode(x, e, s, d, g, int(g.max()) + 1))[0].double().cpu().numpy()
        VN.append(v); VM.append(n.latent(o2.encode_all(n, G, G.dream_rows(), dev))[0].double().cpu().numpy())
    del nets; torch.cuda.empty_cache()
    L = S.ma_L; y_ma = (L > 0).astype(int)
    aux0 = dict(x=XN, qa=re_(S.pair_a[q_tr]), qb=re_(S.pair_b[q_tr]), qy=lab[q_tr], ta=re_(S.ma_a), tb=re_(S.ma_b), ty=y_ma.astype(float))
    auxK = [dict(aux0, v=v) for v in VN]
    test_q = dict(aux0, qa=re_(S.pair_a[q_te]), qb=re_(S.pair_b[q_te]), qy=lab[q_te])
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M); allp = np.arange(len(P))
    old = np.load(features.features_dir() / "o3w_group_weights.npz")
    rep.line(f"members {len(VM)}; GoodScents lean pairs train {len(q_tr)}, held-out {len(q_te)}; Ma {len(L)}; DREAM pairs {len(P)} (all used for fitting)")

    from joblib import Parallel, delayed
    pool = Parallel(n_jobs=a_.jobs, backend="loky")

    def one(v, vn, ax, c, dream, ma):
        torch.set_num_threads(1)
        m = Joint(v, Xd, A, B, ax, c, dream=dream, ma=ma).fit(allp, y)
        tq = Joint(v, Xd, A, B, dict(test_q, v=vn), 0.0)          # only to score held-out GoodScents and Ma
        return (m.theta.numpy(), m.predict(allp), Joint.fast_score(tq.sets[0], m.theta).numpy(), Joint.fast_score(tq.sets[1], m.theta).numpy())

    configs = [("GoodScents only, DREAM off", 0.0, False, False), ("GoodScents + Ma, DREAM off", 0.0, False, True)] + \
              [(f"joint DREAM + GoodScents + Ma, c = {c:g}", c, True, True) for c in (0, 1, 3, 10, 100)] + \
              [("joint DREAM + GoodScents only, c = 10", 10.0, True, False)]
    rows = []
    for name, c, dream, ma in configs:
        out = pool(delayed(one)(v, vn, ax, c, dream, ma) for v, vn, ax in zip(VM, VN, auxK))
        th = np.mean([o[0] for o in out], 0); pr = np.mean([o[1] for o in out], 0)
        dq = np.mean([o[2] for o in out], 0); dm = np.mean([o[3] for o in out], 0)
        rows.append([name, f"{roc_auc_score(lab[q_te], dq):.3f}", f"{roc_auc_score(y_ma, dm):.3f}",
                     f"{np.sqrt(np.mean((pr - y) ** 2)):.4f} / {np.corrcoef(pr, y)[0, 1]:.3f}" if dream else "-",
                     f"{np.corrcoef(th, old['goodscents'])[0, 1]:+.2f}", f"{np.corrcoef(th, old['goodscents_ma'])[0, 1]:+.2f}"])
        rep.line(f"{name}: done")
    rep.section("Sanity: fits on all data (no CV). Expected: rows 1-2 reproduce 0.588 / 0.479 lean AUROC; c moves rows from DREAM-only toward rows 1-2")
    rep.table(["fit", "GoodScents lean AUROC (held-out)", "Ma sign AUROC (training)", "DREAM training RMSE / r",
               "group weights corr. with earlier GoodScents fit", "with earlier GoodScents + Ma fit"], rows)
    rep.save()


if __name__ == "__main__":
    main()
