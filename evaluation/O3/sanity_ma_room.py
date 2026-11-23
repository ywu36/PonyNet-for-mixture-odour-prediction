# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   T = Ma mixtures (198)   P = DREAM development pairs (514)
# ---------------------------------------------------------------------------------------------
"""Where can Ma's rule be learned without costing DREAM? (training fits only, all 514 development pairs; no CV)

Readout x backbone, Ma direction loss weight c = 0 / 1 / 10 (Ma only, no GoodScents):
  group weights (O1.2b form on the hidden template, 24 group weights + length weight + line; eval_o3w_ma.JointMa)
      on O2-C-Single (o2c400, 10 members) and on the node-enriched width 64 (ga_w64, 3 members)
  attention query (C17 readout, q 64 + kappa + line; eval_o3s_attn_ma.AttnMa) on ga_w64 (= O3s)
Per row: DREAM training r / RMSE, Ma training loss (T fitted; at c = 0 T = 0.1), Ma training AUROC.

Env:     mist
Usage:   python evaluation/O3/sanity_ma_room.py
Writes:  results/sanity_ma_room/<date_time>.{txt,json}
"""
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
from eval_readout_latent import backbone, vectors  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_o3s_attn_ma as att  # noqa: E402
import eval_o3w_ma as gw  # noqa: E402
import train_groupatoms  # noqa: E402

F64 = torch.float64


def main():
    dev = pick_device(); rep = Report("sanity_ma_room", {"device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    XN = o3.group_matrix(z["smiles"], gn).astype(float)
    L = S.ma_L; y_ma = (L > 0).astype(float); used = np.unique(np.r_[S.ma_a, S.ma_b])
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M); allp = np.arange(len(P))

    # backbones
    nets, fn = backbone("o2c400", dev)
    VN_c = [v.numpy() for v in vectors(nets, fn, S, used, len(z["smiles"]), dev)]
    VM_c = [v[G.dream_rows()].numpy() for v in vectors(nets, fn, G, G.dream_rows(), len(G.smiles_N), dev)]
    del nets
    Gg, tr, _, names = train_groupatoms.graphs_with_groups(); att.attach_named(S, z["smiles"], names, used)
    nets = [train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in att.SEEDS]
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); VM_g, VN_g = [], []
    for n in nets:
        with torch.no_grad():
            sc = n.latent(o2.encode_all(n, Gg, tr_s, dev))[0].norm(dim=1).mean().item()
            VM_g.append(n.latent(o2.encode_all(n, Gg, Gg.dream_rows(), dev))[0].double().cpu().numpy() / sc)
            v = np.zeros((len(z["smiles"]), VM_g[-1].shape[1])); v[used] = n.latent(o2.encode_all(n, S, used, dev))[0].double().cpu().numpy() / sc
            VN_g.append(v)
    del nets; torch.cuda.empty_cache()

    T0 = torch.tensor(float(np.log(0.1)))
    rows = []
    for name, VM, VN in (("group weights, O2-C-Single", VM_c, VN_c), ("group weights, node-enriched w64", VM_g, VN_g)):
        s_all = [gw.ma_stats(v, XN, S.ma_a, S.ma_b, y_ma) for v in VN]
        for c in (0.0, 1.0, 10.0):
            ms = [gw.JointMa(v, Xd, A, B, s, c).fit(allp, y) for v, s in zip(VM, s_all)]
            pr = np.mean([m.predict(allp) for m in ms], 0)
            ml = np.mean([gw.ma_loss(s, m.theta, T0).item() for m, s in zip(ms, s_all)])
            au = roc_auc_score(y_ma, np.mean([gw.ma_score(s, m.theta).numpy() for m, s in zip(ms, s_all)], 0))
            rows.append([name, f"{c:g}", f"{np.corrcoef(pr, y)[0, 1]:.3f}", f"{np.sqrt(np.mean((pr - y) ** 2)):.4f}", f"{ml:.3f}", f"{au:.3f}"])
            rep.line(f"{name}, c = {c:g}: done")
    s_all = [att.ma_stats(v, S.ma_a, S.ma_b, y_ma) for v in VN_g]
    for c in (0.0, 1.0, 10.0):
        ms = [att.AttnMa(v, A, B, s, c).fit(allp, y) for v, s in zip(VM_g, s_all)]
        pr = np.mean([m.predict(allp) for m in ms], 0)
        ml = np.mean([att.ma_loss(s, m.q, T0.double()).item() for m, s in zip(ms, s_all)])
        au = roc_auc_score(y_ma, np.mean([att.ma_score(s, m.q).numpy() for m, s in zip(ms, s_all)], 0))
        rows.append(["attention query, node-enriched w64 (O3s)", f"{c:g}", f"{np.corrcoef(pr, y)[0, 1]:.3f}", f"{np.sqrt(np.mean((pr - y) ** 2)):.4f}", f"{ml:.3f}", f"{au:.3f}"])
    rep.section("Training fits on all 514 DREAM pairs (not a test). Ma loss at T = 0.1 for every row (comparable); chance 0.693")
    rep.table(["readout, backbone", "Ma weight c", "DREAM training r", "DREAM training RMSE", "Ma loss (T 0.1)", "Ma AUROC (training)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
