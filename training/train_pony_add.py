# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = held-out GS-LF molecules (1,008)   M = DREAM molecules (235)   H = hidden (h)
# ---------------------------------------------------------------------------------------------
"""Open-POM ablation from the other side (user 2026-11-22): add ONE open-POM part at a time to Pony (w/o attention)
= O3M width 64 (checkpoints/o3m_w64 seeds 0-2: d 64, h 64, 3 steps, bond network 16, mean over atoms, linear head).
Same graphs, held-out molecules, recipe (o2.RECIPE) and seeds. Complements train_o2c_nolstm.py / train_o2c_mp.py, which
remove one part at a time from open-POM.
  mp      open-POM's message passing: atom width 100, 5 steps, bond network 75 (hidden layer stays 64)
  s2s     set2set (LSTM) pooling instead of the mean
  ffn     open-POM's deep read-out layers [392, 392] between the pooled vector and the hidden layer
  h256    hidden layer 256 instead of 64
  d128    atom width 128 (hidden layer stays 64)
  steps5  5 message-passing steps instead of 3
Scoring: single molecules (3-member held-out AUROC; gap as train_groupatoms.py) and DREAM with Table 6's Pony rule
(unit hidden-layer directions averaged per mixture, length 1, Euclidean, line only): T46 / V50 and CV 50 x 5.

Env:     mist (GPU)
Usage:   python training/train_pony_add.py
Writes:  checkpoints/pony_add_<variant>/member_<s>.pt, results/train_pony_add/<date_time>.{txt,json}
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "training", ROOT / "evaluation/O1"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, o2  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402
from final_heldout import paired  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
BASE = {"d": 64, "h": 64, "steps": 3, "edge_h": 16}
VARIANTS = {"mp": ("+ open-POM message passing (width 100, 5 steps, bond net 75)", {"d": 100, "steps": 5, "edge_h": 75}),
            "s2s": ("+ LSTM (set2set) pooling", {"pool": "set2set"}),
            "ffn": ("+ deep read-out layers (392, 392)", {"ffn": [392, 392]}),
            "h256": ("+ hidden layer 256", {"h": 256}),
            "d128": ("+ atom width 128", {"d": 128}),
            "steps5": ("+ 5 message-passing steps", {"steps": 5})}


def unit_dist(V_MxH, A_PxM, B_PxM):
    U = V_MxH / np.linalg.norm(V_MxH, axis=1, keepdims=True)
    za = A_PxM @ U / A_PxM.sum(1, keepdims=True); zb = B_PxM @ U / B_PxM.sum(1, keepdims=True)
    return np.linalg.norm(za / np.linalg.norm(za, axis=1, keepdims=True) - zb / np.linalg.norm(zb, axis=1, keepdims=True), axis=1)


def main():
    dev = pick_device(); rep = Report("train_pony_add", {"seeds": SEEDS, "device": dev, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    gslf = np.arange(G.n_gslf); tr, te = gslf[~G.audit_test], gslf[G.audit_test]
    tr_s = np.random.default_rng(0).choice(tr, 1000, replace=False); rows_D = G.dream_rows()
    for v, (_, ch) in VARIANTS.items():
        ck = ROOT / f"checkpoints/pony_add_{v}"; ck.mkdir(parents=True, exist_ok=True); arch = {**BASE, **ch}
        for s in SEEDS:
            if (ck / f"member_{s}.pt").exists():
                continue
            t0 = time.time(); net, steps = o2.train_member(G, tr, s, dev, log=rep.line, make_net=lambda: train_o3m.O3M(**arch))
            torch.save({"model_state_dict": net.state_dict(), "arch": arch, "seed": s, "steps": steps}, ck / f"member_{s}.pt")
            rep.line(f"{v} seed {s}: {time.time() - t0:.0f} s")

    nets = {"Pony (w/o attention)": [train_o3m.load(ROOT / f"checkpoints/o3m_w64/member_{s}.pt", dev) for s in SEEDS]}
    nets.update({VARIANTS[v][0]: [train_o3m.load(ROOT / f"checkpoints/pony_add_{v}/member_{s}.pt", dev) for s in SEEDS] for v in VARIANTS})
    REF = next(iter(nets)); single, V = [], {}
    for k, ns in nets.items():
        pt = [o2.predict_probs(n, G, te, dev) for n in ns]; a_te = [o2.mean_auroc(p, G.Y[te]) for p in pt]
        a_tr = [o2.mean_auroc(o2.predict_probs(n, G, tr_s, dev), G.Y[tr_s]) for n in ns]
        with torch.no_grad():
            V[k] = [n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy() for n in ns]
        single.append([k, f"{sum(p.numel() for p in ns[0].parameters()) / 1e6:.3f} M", " ".join(f"{x:.4f}" for x in a_te),
                       f"{o2.mean_auroc(np.mean(pt, 0), G.Y[te]):.4f}", f"{np.mean(a_tr):.4f}", f"{np.mean(a_tr) - np.mean(a_te):+.4f}"])
    rep.section("1. Single molecules, official held-out (1,008): AUROC, mean over 138 words")
    rep.table(["network", "parameters", "held-out per seed (0, 1, 2)", "3-member mean held-out", "training (1,000)", "gap (training - held-out)"], single)

    op = features.load("openpom"); cids = np.asarray(op.cids_M)
    order = {int(c): i for i, c in enumerate(G.dream_cids)}; perm = np.array([order[int(c)] for c in cids])

    def dist(k, P):
        A, B = membership(P, cids); A, B = np.asarray(A, float), np.asarray(B, float)
        return np.mean([unit_dist(v[perm], A, B) for v in V[k]], 0)   # members averaged (as Table 6)

    def line(d, yy):
        return lambda a, b: np.polyval(np.polyfit(d[a], yy[a], 1), d[b])

    rep.section("2. DREAM, Table 6 Pony rule (unit hidden directions averaged, line only): RMSE / r; change from Pony r [95% CI]")
    train = data.load_train(); rows = []
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); a = np.arange(n); b = np.arange(n, len(allp))
        yy = np.r_[evaluate.labels(train), hs.y_P]; preds = {k: line(dist(k, allp), yy)(a, b) for k in nets}
        for k, p in preds.items():
            c = "" if k == REF else (lambda ci: f"{metrics.pearson(p, hs.y_P) - metrics.pearson(preds[REF], hs.y_P):+.3f} [{ci[0][0]:+.3f}, {ci[0][1]:+.3f}]")(paired(p, preds[REF], hs.y_P))
            rows.append([hs.name, k, f"{metrics.rmse(p, hs.y_P):.4f} / {metrics.pearson(p, hs.y_P):.3f}", c])
    P = dev_pairs(); yy = evaluate.labels(P); cv = {k: evaluate.cross_validate(P, line(dist(k, P), yy)) for k in nets}
    for k, r_ in cv.items():
        ref = None if k == REF else cv[REF]; s = evaluate.score_cv(P, r_, reference=ref)
        c = "" if ref is None else f"{s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"
        rows.append(["CV 50 x 5 (514)", k, f"{s['rmse']:.4f} / {s['r']:.3f}", c])
    rep.table(["set", "network", "RMSE / r", "change in r from Pony (w/o attention)"], rows)
    rep.value("single", single); rep.value("dream", rows); rep.save()


if __name__ == "__main__":
    main()
