# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = held-out molecules / pairs   W = words (138)   R = members
# ---------------------------------------------------------------------------------------------
"""Train O2-J (design/o2j_design.md): backbone (O2-N or O2-VAE) trained jointly on single molecules (GS-LF), GoodScents
blender pairs (learning to rank over the 79 mapped words) and Ma dominance (Siamese). No DREAM labels.

Checks in --report: (1) single-molecule held-out 20% AUROC; (2) GoodScents held-out pairs (both molecules unseen):
mean per-word AUROC over the mapped words, O2-J merged vs O2-J mean of singles vs untrained backbone (merged, mean of
singles); (3) Ma on unseen molecules: members trained with --ma-holdout k (k = 0-4), pooled r(L-hat, L).

Env:     mist (GPU)
Usage:   python training/train_o2j.py --arch o2n --seeds 0 1 2 3 4
         python training/train_o2j.py --arch o2n --ma-holdout 0 1 2 3 4      (one member per Ma molecule fold)
         python training/train_o2j.py --arch o2n --report
Writes:  checkpoints/o2j_<arch>/{member_<seed>,mafold_<k>}.pt, results/train_o2j/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixdist import features, o2, o2j  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402

BACKBONE = {"o2n": "checkpoints/o2n", "vae": "checkpoints/o2_vae"}


def word_auroc(logits_NxW, Y_NxW, words):
    from sklearn.metrics import roc_auc_score
    vals = [roc_auc_score(Y_NxW[:, w] > 0, logits_NxW[:, w]) for w in words if 0 < (Y_NxW[:, w] > 0).sum() < len(Y_NxW)]
    return float(np.mean(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["o2n", "vae"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[]); ap.add_argument("--ma-holdout", type=int, nargs="*", default=[])
    ap.add_argument("--report", action="store_true"); ap.add_argument("--epochs", type=int, default=o2.RECIPE["epochs"])
    a = ap.parse_args()
    dev = pick_device(); recipe = {**o2.RECIPE, "epochs": a.epochs}
    rep = Report("train_o2j", {**vars(a), "device": dev, **recipe})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    gslf = np.arange(G.n_gslf); train_ids, test_ids = gslf[~G.audit_test], gslf[G.audit_test]
    ck = ROOT / f"checkpoints/o2j_{a.arch}"; ck.mkdir(parents=True, exist_ok=True)
    rep.line(f"GoodScents train pairs {(S.pair_split == 0).sum()}, held-out {(S.pair_split == 1).sum()}; Ma mixtures {len(S.ma_L)}")

    jobs = [("member", s, None) for s in a.seeds] + [("mafold", k, k) for k in a.ma_holdout]
    for kind, s, hold in jobs:
        t0 = time.time()
        m, steps = o2j.train(G, S, train_ids, s if kind == "member" else 100 + s, dev, arch=a.arch, ma_holdout=hold, log=rep.line, recipe=recipe)
        auc = o2.mean_auroc(o2.predict_probs(m.net, G, test_ids, dev), G.Y[test_ids])
        torch.save({"model_state_dict": m.state_dict(), "arch": a.arch, "latent": 32, "seed": s, "ma_holdout": hold, "steps": steps,
                    "held_out_auroc": auc}, ck / f"{kind}_{s}.pt")
        rep.line(f"{kind} {s}: steps {steps}, single-molecule held-out AUROC {auc:.4f}, {time.time() - t0:.0f} s")

    if a.report:
        files = sorted(ck.glob("member_*.pt")); ms = [o2j.load(f, dev)[0] for f in files]
        base = [o2j.as_joint(n, a.arch, dev) for n in o2.load_members(sorted((ROOT / BACKBONE[a.arch]).glob("member_*.pt")), dev)]
        rep.section("1  Single molecules: held-out 20% GS-LF AUROC")
        pr = [o2.predict_probs(m.net, G, test_ids, dev) for m in ms]; single = [o2.mean_auroc(p, G.Y[test_ids]) for p in pr]
        pb = [o2.predict_probs(m.net, G, test_ids, dev) for m in base]
        rep.line(f"O2-J members {len(ms)}: single {min(single):.4f}-{max(single):.4f}, ensemble {o2.mean_auroc(np.mean(pr, 0), G.Y[test_ids]):.4f}; "
                 f"backbone ensemble {o2.mean_auroc(np.mean(pb, 0), G.Y[test_ids]):.4f}")
        rep.section("2  GoodScents held-out pairs (both molecules unseen): per-word AUROC over the mapped words, ensemble")
        hq = np.where(S.pair_split == 1)[0]; Yh = S.pair_Y[hq]; rows = []
        for name, mm, merged in [("O2-J merged", ms, True), ("O2-J mean of singles", ms, False),
                                 ("backbone merged (no joint training)", base, True), ("backbone mean of singles", base, False)]:
            lg = np.mean([o2j.pair_logits(m, S, hq, dev, merged) for m in mm], 0)
            v, n = word_auroc(lg, Yh, S.targets); rows.append([name, f"{v:.4f}", n])
        rep.table(["profile", "mean AUROC", "words scored"], rows)
        mf = sorted(ck.glob("mafold_*.pt"))
        if mf:
            rep.section("3  Ma on unseen molecules (one member per held-out molecule fold)")
            Lh, Lt, rows = [], [], []
            for f in mf:
                m, c = o2j.load(f, dev); _, te = S.ma_split(c["ma_holdout"])
                p, _ = o2j.ma_predict(m, S, te, dev); Lh.append(p); Lt.append(S.ma_L[te])
                rows.append([c["ma_holdout"], len(te), f"{np.corrcoef(p, S.ma_L[te])[0, 1]:+.3f}"])
            rep.table(["fold", "test mixtures", "r(L-hat, L)"], rows)
            Lh, Lt = np.concatenate(Lh), np.concatenate(Lt); r = np.corrcoef(Lh, Lt)[0, 1]
            rng = np.random.default_rng(0); bs = [np.corrcoef(*(lambda i: (Lh[i], Lt[i]))(rng.integers(0, len(Lh), len(Lh))))[0, 1] for _ in range(2000)]
            rep.line(f"pooled r {r:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] over {len(Lh)} test mixtures "
                     f"(mixture resampling); RMSE {np.sqrt(np.mean((Lh - Lt) ** 2)):.3f} (always 0: {np.sqrt(np.mean(Lt ** 2)):.3f}); "
                     f"references: DREAM group gains no refit r +0.27, Ma own seen-molecule score r +0.34")
    rep.save()


if __name__ == "__main__":
    main()
