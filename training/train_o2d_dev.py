# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = DREAM development pairs (514)   N = pooled test predictions
# ---------------------------------------------------------------------------------------------
"""O2-N-Single+DREAM with a dial on what DREAM may update (design/o2j_design.md, "DREAM update dial").

alpha = fraction of the DREAM gradient that reaches the encoder (message passing + Set2Set):
    alpha 0    encoder learns from single molecules only; DREAM updates decoder, gain, head (the user's proposal)
    alpha 0.1  DREAM nudges the encoder at 1/10 strength
    alpha 1    full joint training (the earlier O2-D, rerun on development pairs)
Single molecules update everything in every arm. Development pairs only (evaluation/CV.md rule 7; lockbox untouched).
5 folds x 1 repeat over the development pairs (whole mixtures, seed 0), 3 members per fold (seeds 0-2), ensemble mean.

Env:     mist (GPU)
Usage:   python training/train_o2d_dev.py --alpha 0 --fold 0 1 2 3 4
         python training/train_o2d_dev.py --report
Writes:  checkpoints/o2d_dev/a<alpha>/fold<k>_seed<s>.pt, results/train_o2d_dev/<date_time>.{txt,json}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import cv, data, evaluate, features, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402

SEEDS = (0, 1, 2)
ALPHAS = (0.0, 0.1, 1.0)
CK = ROOT / "checkpoints/o2d_es"   # with validation lines (earlier runs without: checkpoints/o2d_dev)


def save_member(path, m, tk, te, meta):
    c = tk.curve[-1]
    torch.save({"model_state_dict": m.state_dict(), **meta, "test_idx": te, "inner_train": tk.itr, "inner_val": tk.iva,
                "pred": tk.final, "pred_best": tk.best[2], "best_epoch": tk.best[1], "curve": tk.curve,
                "probs_out": tk.final_probs, "probs_out_best": tk.best[3],
                "train_r": c["train_r"], "test_r": c["test_r_diagnostic"], "single_auroc": c["single_auroc"]}, path)


def describe(tk):
    c, b = tk.curve[-1], tk.curve[tk.best[1] - 1]
    return (f"inner train/val {len(tk.itr)}/{len(tk.iva)} pairs; final epoch: train r {c['train_r']:.3f}, val r {c['val_r']:.3f}, "
            f"held-out r {c['test_r_diagnostic']:.3f}, single AUROC {c['single_auroc']:.4f}; best val epoch {tk.best[1]}: val r {b['val_r']:.3f}, "
            f"held-out r {b['test_r_diagnostic']:.3f}, single AUROC {b['single_auroc']:.4f}")


def dev_pairs():
    train = data.load_train()
    return [train[i] for i in cv.lockbox_split(train).train]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float); ap.add_argument("--fold", type=int, nargs="*", default=[])
    ap.add_argument("--arch", choices=["o2n", "o2c"], default="o2c")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("train_o2d_dev", {**vars(a), "device": dev, "seeds": SEEDS, **o2.RECIPE})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); P = dev_pairs(); y = evaluate.labels(P)
    gslf = np.arange(G.n_gslf); single_ids, single_test = gslf[~G.audit_test], gslf[G.audit_test]
    folds = list(cv.mixture_holdout_folds(P, k=5, repeats=1, seed=0))
    rep.line(f"development pairs {len(P)} (lockbox untouched); held-out pairs per fold {[len(f.test) for f in folds]}")
    if a.fold:
        D = o2j.DreamStream(G, P, [int(c) for c in G.dream_cids], dev)
        out = CK / f"{a.arch}_a{a.alpha:g}"; out.mkdir(parents=True, exist_ok=True)
        for k in a.fold:
            tr, te = np.array(folds[k].train), np.array(folds[k].test)
            for s in SEEDS:
                if (out / f"fold{k}_seed{s}.pt").exists():
                    rep.line(f"alpha {a.alpha:g} fold {k} seed {s}: exists, skipped"); continue
                t0 = time.time(); itr, iva = o2j.inner_split(P, tr)
                tk = o2j.Tracker(G, D, y, itr, iva, te, single_test, dev, length=True)
                m = o2j.train_dream(G, single_ids, D, y, itr, s, dev, log=rep.line, enc_grad=a.alpha, on_epoch=tk, arch=a.arch)
                save_member(out / f"fold{k}_seed{s}.pt", m, tk, te, {"arch": a.arch, "alpha": a.alpha, "fold": k, "seed": s})
                rep.line(f"alpha {a.alpha:g} fold {k} seed {s}: " + describe(tk) + f", {time.time() - t0:.0f} s")
    if a.report:
        from mixdist.b1a_plus import B1aPlus, membership, size_features
        from eval_b1a_plus import piece_matrix
        op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
        X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
        A, B = membership(P, op.cids_M); S = size_features(P); q = mixture.quantile_per_word(op.probs_MxW)
        ref = evaluate.cross_validate(P, lambda tr, te: B1aPlus(q, X, A, B, S, 1e-3).fit(tr, y[tr]).predict(te), repeats=1)
        arms, side, curves, all_curves, marks, in_out = [], [], [], {}, [], []
        for name, d in [(f"O2-C-Single+DREAM, alpha {al:g}", CK / f"o2c_a{al:g}") for al in ALPHAS] + \
                       [("O2-C-Single+Scent+Ma+DREAM, alpha 0, direction only", ROOT / "checkpoints/o2all_es/o2c_dir"),
                        ("O2-C-Single+Scent+Ma+DREAM, alpha 0, direction + length", ROOT / "checkpoints/o2all_es/o2c_len")]:
            saved = {}
            for f in d.glob("fold*_seed*.pt"):
                c = torch.load(f, map_location="cpu", weights_only=False); saved.setdefault(c["fold"], []).append(c)
            if sorted(saved) != list(range(5)) or any(len(v) != len(SEEDS) for v in saved.values()):
                rep.line(f"{name}: incomplete {[(k, len(v)) for k, v in sorted(saved.items())]}, skipped"); continue
            cs = [c for v in saved.values() for c in v]
            for when, key, okey in (("final epoch", "pred", "probs_out"), ("best inner-val epoch", "pred_best", "probs_out_best")):
                by_test = {tuple(folds[k].test): np.mean([c[key] for c in saved[k]], 0) for k in saved}
                r_in = evaluate.cross_validate(P, lambda tr, te, b=by_test: b[tuple(te)], repeats=1)
                arms.append((f"{name} [in, {when}]", r_in))
                if all(okey in c for c in cs):   # "out" on the same weights: mean of decoded single molecules, B1a distance, line on the fold's training pairs
                    o = {}
                    for k in saved:
                        d_ = mixture.paper_distance(P, op.cids_M, np.mean([c[okey] for c in saved[k]], 0).astype(np.float64))
                        tr_, te_ = np.array(folds[k].train), np.array(folds[k].test); a_, c_ = evaluate.fit_line(d_[tr_], y[tr_])
                        o[tuple(te_)] = a_ * d_[te_] + c_
                    r_out = evaluate.cross_validate(P, lambda tr, te, b=o: b[tuple(te)], repeats=1)
                    arms.append((f"{name} [out, {when}]", r_out)); in_out.append((f"{name} [{when}]", r_in, r_out))
            ep = [c["best_epoch"] for c in cs]; E = len(cs[0]["curve"])
            marks = [e for e in (1, 2, 5, 10, 20, 30, 40, 50, E) if e <= E]
            curves.append([name, f"{np.median(ep):.0f} [{min(ep)}-{max(ep)}]"] + [
                " / ".join(f"{np.mean([c['curve'][e - 1][f] for c in cs]):.2f}" for f in ("train_r", "val_r", "test_r_diagnostic", "single_auroc"))
                for e in marks])
            all_curves[name] = [{**c["curve"][i], "fold": c["fold"], "seed": c["seed"]} for c in cs for i in range(E)]
            side.append([name, f"{np.mean([c['train_r'] for c in cs]):.3f}", f"{np.mean([c['test_r'] for c in cs]):.3f}",
                         f"{min(c['single_auroc'] for c in cs):.4f}-{max(c['single_auroc'] for c in cs):.4f}"])
        rows = []
        for subset in ("all pairs", "share no molecule"):
            mk = lambda r_: None if subset == "all pairs" else r_.no_overlap_N  # noqa: E731
            s0 = evaluate.score_cv(P, ref, mask_N=mk(ref))
            rows.append(["B1a++ (baseline)", subset, int(s0.get("n", 0)) or "", f"{s0['rmse']:.4f}", f"{s0['r']:.3f} [{s0['r_ci'][0]:.3f}, {s0['r_ci'][1]:.3f}]", ""])
            for name, res in arms:
                s1 = evaluate.score_cv(P, res, reference=ref, mask_N=mk(res))
                rows.append([name, subset, "", f"{s1['rmse']:.4f}", f"{s1['r']:.3f} [{s1['r_ci'][0]:.3f}, {s1['r_ci'][1]:.3f}]",
                             f"{s1['rmse_change']:+.4f} / {s1['r_change']:+.3f} [{s1['r_change_ci'][0]:+.3f}, {s1['r_change_ci'][1]:+.3f}]"])
        rep.section("DREAM development pairs, 5 folds x 1 repeat (whole mixtures held out; same folds for all rows)")
        rep.table(["model", "subset", "n", "RMSE", "r [95% CI]", "change from B1a++: RMSE / r [95% CI]"], rows)
        io = []
        for name, r_in, r_out in in_out:
            for subset in ("all pairs", "share no molecule"):
                s2 = evaluate.score_cv(P, r_in, reference=r_out, mask_N=None if subset == "all pairs" else r_in.no_overlap_N)
                io.append([name, subset, f"{s2['rmse_change']:+.4f}", f"{s2['r_change']:+.3f} [{s2['r_change_ci'][0]:+.3f}, {s2['r_change_ci'][1]:+.3f}]"])
        base = {n.split("[")[1]: r_ for n, r_ in arms if n.startswith("O2-C-Single+DREAM, alpha 0 [")}   # DREAM-only, alpha 0
        for name, r_ in arms:
            tag = name.split("[")[1]
            if name.startswith("O2-C-Single+Scent+Ma+DREAM") and tag in base:
                for subset in ("all pairs", "share no molecule"):
                    s3 = evaluate.score_cv(P, r_, reference=base[tag], mask_N=None if subset == "all pairs" else r_.no_overlap_N)
                    io.append([f"{name} minus DREAM-only alpha 0", subset, f"{s3['rmse_change']:+.4f}",
                               f"{s3['r_change']:+.3f} [{s3['r_change_ci'][0]:+.3f}, {s3['r_change_ci'][1]:+.3f}]"])
        rep.section("Paired, same folds: merged inside minus after the read-out (same weights); four datasets minus DREAM-only (GoodScents + Ma added)")
        rep.table(["model", "subset", "RMSE change", "r change [95% CI]"], io)
        rep.section("Validation lines: mean over members of train r / inner-val r / held-out r (diagnostic, not used) / single AUROC, by epoch")
        rep.table(["model", "best inner-val epoch, median [range]"] + [f"epoch {e}" for e in marks], curves)
        rep.value("curves", all_curves)
        rep.section("Overfitting and single-molecule memory (per member, final epoch)")
        rep.table(["model", "mean train r", "mean held-out r (single member)", "single-molecule held-out AUROC (O2-N 0.875-0.880)"], side)
    rep.save()


if __name__ == "__main__":
    main()
