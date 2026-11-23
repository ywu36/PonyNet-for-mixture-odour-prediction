# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   W = words (138)   P = DREAM development pairs (514)
#   T = Ma mixtures   K = members (3, seeds 0-2)
# ---------------------------------------------------------------------------------------------
"""Does GoodScents (O2-J word ranking, plain A + B) help the best backbone? (training: training/train_o3g.py)

Backbones (frozen, seeds 0-2, all node-enriched width 64):
  ga_w64                         singles only
  ga_w64_rank                    + POM-order loss (the best backbone so far)
  ga_w64_rank_scent0.1 / 0.3     + GoodScents word ranking, weight 0.1 / 0.3
  ga_w64_rank_scent0.3_shuf      control: GoodScents labels permuted across pairs, weight 0.3
Per backbone: single-molecule held-out AUROC; GoodScents held-out pairs (both molecules unseen) word-ranking AUROC with
plain A + B; Ma (198 mixtures, no fitting) AUROC of cos(z, v_A) - cos(z, v_B) with z = plain mean; DREAM development
pairs 50 x 5 with two readouts fitted on DREAM folds only:
  O3s readout: hidden-layer attention, direction only (C17; q 64 + kappa + line = 67 numbers)
  O1.2b rule on the backbone's word profiles (27 numbers; "O3 + counts" in the counts table)
Paired, same folds: change from O1.2b; each backbone minus ga_w64_rank (same readout); GoodScents 0.3 minus shuffled.

Env:     mist
Usage:   python evaluation/O3/eval_o3g.py [--repeats 50] [--set scent|ma|pos|gqn|gqn2]
Writes:  results/eval_o3g/<date_time>.{txt,json} (+ _cv.pkl)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report, results_dir  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o3s_attn_ma as att  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3g  # noqa: E402

ecd.DIRECTION_ONLY = True
SEEDS = (0, 1, 2)
FOLDERS = {"singles only (ga_w64)": "ga_w64", "+ POM-order (ga_w64_rank, best)": "ga_w64_rank",
           "+ POM-order + GoodScents 0.1": "ga_w64_rank_scent0.1", "+ POM-order + GoodScents 0.3": "ga_w64_rank_scent0.3",
           "+ POM-order + GoodScents 0.3, labels shuffled": "ga_w64_rank_scent0.3_shuf"}
BEST = "+ POM-order (ga_w64_rank, best)"
SETS = {"scent": (FOLDERS, BEST, "+ POM-order + GoodScents 0.3", "+ POM-order + GoodScents 0.3, labels shuffled"),
        "ma": ({"no Ma (ga_w64_ma0)": "ga_w64_ma0", "+ Ma 0.3": "ga_w64_ma0.3", "+ Ma 1": "ga_w64_ma1",
                "+ Ma 1, signs shuffled": "ga_w64_ma1_shuf"}, "no Ma (ga_w64_ma0)", "+ Ma 1", "+ Ma 1, signs shuffled"),
        "pos": ({"counts, centred (ga_w64)": "ga_w64", "counts, positive (ga_w64_pos)": "ga_w64_pos",
                 "counts, positive + sparse (ga_w64_sparse1)": "ga_w64_sparse1"}, "counts, centred (ga_w64)", None, None),
        "gqn": ({"counts, BatchNorm hidden (ga_w64)": "ga_w64", "counts, global quantile layer (ga_w64_gqn)": "ga_w64_gqn"},
                "counts, BatchNorm hidden (ga_w64)", None, None),
        "gqn2": ({"counts, BatchNorm (ga_w64)": "ga_w64", "counts + POM-rank, BatchNorm (ga_w64_rank)": "ga_w64_rank",
                  "counts, quantile refreshed every 5 steps": "ga_w64_gqn_steps", "counts + POM-rank, quantile every 5 steps": "ga_w64_rank_gqn_steps",
                  "counts, BatchNorm then quantile": "ga_w64_gqn_bn", "counts + POM-rank, BatchNorm then quantile": "ga_w64_rank_gqn_bn",
                  "counts, BatchNorm then quantile every 5 steps": "ga_w64_gqn_bn_steps",
                  "counts + POM-rank, BatchNorm then quantile every 5 steps": "ga_w64_rank_gqn_bn_steps"},
                 "counts, BatchNorm (ga_w64)", None, None)}


def load_net(path, dev):
    """Checkpoints with a full layout ("arch") load with O3M(**arch) (positive / sparse); others via train_groupatoms."""
    c = torch.load(path, map_location="cpu", weights_only=False)
    if "arch" in c and c["arch"].get("gqn"):
        import train_gqn
        return train_gqn.load(path, dev)
    if "arch" in c and ("positive" in c["arch"] or "sparse_l1" in c["arch"]):
        from train_o3m import O3M
        n = O3M(**c["arch"]).float(); n.load_state_dict(c["model_state_dict"]); return n.to(dev).eval()
    return train_groupatoms.load(path, dev)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=evaluate.CV_REPEATS)
    ap.add_argument("--set", choices=list(SETS), default="scent"); a_ = ap.parse_args()
    global FOLDERS, BEST
    FOLDERS, BEST, REAL, SHUF = SETS[a_.set]
    dev = pick_device(); rep = Report("eval_o3g", {"repeats": a_.repeats, "seeds": SEEDS, "lam": LAM, "device": dev, "folders": FOLDERS},
                                      tag=a_.set + ("_smoke" if a_.repeats < 5 else ""))
    Gg, tr, te, names = train_groupatoms.graphs_with_groups()
    S = o2j.Store(features.features_dir() / "o2j_data.npz"); z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    q_tr, q_te, _ = train_o3g.scent_pairs(Gg, S, z["smiles"], te)
    ma_rows = np.unique(np.r_[S.ma_a, S.ma_b]); need = np.unique(np.r_[S.pair_a[q_te], S.pair_b[q_te], ma_rows])
    att.attach_named(S, z["smiles"], names, need)
    y_ma = (S.ma_L > 0).astype(float); rows_D = Gg.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(Gg.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
           "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}

    VM, QW, rows = {}, {}, []
    for lab, fo in FOLDERS.items():
        nets = [load_net(ROOT / f"checkpoints/{fo}/member_{s}.pt", dev) for s in SEEDS]
        VM[lab], single, scent, ma = [], [], [], []
        for n in nets:
            with torch.no_grad():
                sc = n.latent(o2.encode_all(n, Gg, tr_s, dev))[0].norm(dim=1).mean().item()
                VM[lab].append(n.latent(o2.encode_all(n, Gg, rows_D, dev))[0].double().cpu().numpy() / sc)
                vn = np.zeros((len(z["smiles"]), VM[lab][-1].shape[1])); vn[ma_rows] = n.latent(o2.encode_all(n, S, ma_rows, dev))[0].double().cpu().numpy()
            single.append(o2.mean_auroc(o2.predict_probs(n, Gg, te, dev), Gg.Y[te]))
            scent.append(train_o3g.scent_rank_auc(n, S, q_te, S.pair_Y, S.targets, dev))
            ma.append(att.ma_score(att.ma_stats(vn, S.ma_a, S.ma_b, y_ma), torch.zeros(vn.shape[1], dtype=torch.float64)).numpy())
        QW[lab] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, Gg, rows_D, dev) for n in nets], 0).astype(float))
        mm = np.mean(ma, 0); _, t_te = S.ma_split(0)
        rows.append([lab, f"{np.mean(single):.4f} ({', '.join(f'{x:.3f}' for x in single)})", f"{np.mean(scent):.3f}",
                     f"{roc_auc_score(y_ma, mm):.3f}", f"{roc_auc_score(y_ma[t_te], mm[t_te]):.3f}"])
        del nets; torch.cuda.empty_cache()
    rep.section(f"Backbones (3 members): single held-out AUROC; GoodScents held-out word ranking ({len(q_te)} pairs, plain A + B); "
                "Ma plain-mean direction AUROC (no fitting)")
    rep.table(["backbone", "single AUROC (per seed)", "GoodScents rank AUROC", "Ma AUROC, plain mean (all 198)",
               "Ma AUROC, mixtures with unseen fold-0 molecules"], rows)

    REF = "O1.2b (reference, 27 numbers)"
    res = {REF: evaluate.cross_validate(P, lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM)
                                        .fit(tr_, y[tr_]).predict(te_), repeats=a_.repeats)}
    for lab in FOLDERS:
        k1 = f"{lab} | O3s attention (67)"
        res[k1] = evaluate.cross_validate(P, lambda tr_, te_, Vs=VM[lab]: np.mean(
            [ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(tr_, y[tr_]).predict(te_) for v in Vs], 0), repeats=a_.repeats)
        rep.line(f"done: {k1}")
        k2 = f"{lab} | O1.2b rule on words (27)"
        res[k2] = evaluate.cross_validate(P, lambda tr_, te_, q=QW[lab]: O12b(q, X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_),
                                          repeats=a_.repeats)
        rep.line(f"done: {k2}")
    (results_dir() / "eval_o3g").mkdir(parents=True, exist_ok=True)
    with open(results_dir() / "eval_o3g" / f"{rep.time:%Y-%m-%d_%H%M%S}_{a_.set}_cv.pkl", "wb") as fh:
        pickle.dump({"res": res, "k_P": k_P}, fh)

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({a_.repeats} x 5): RMSE / r; change [95% CI], positive r change = better")
    out = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if k == REF else res[REF])
            best = None if k == REF else k.replace(k.split(" | ")[0], BEST)
            out.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                        "" if k == REF else ch(s_), ch(sc(r_, sub, res[best])) if best and best != k else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "minus best backbone (same readout)"], out)
    if REAL is None:
        rep.save(); return
    rep.section(f"{REAL} minus {SHUF} (same readout, same folds)")
    out = []
    for ro in ("O3s attention (67)", "O1.2b rule on words (27)"):
        for sub in SUB:
            out.append([ro, sub, ch(sc(res[f"{REAL} | {ro}"], sub, res[f"{SHUF} | {ro}"]))])
    rep.table(["readout", "subset", "real minus shuffled"], out)
    rep.save()


if __name__ == "__main__":
    main()
