# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Branch by how many molecules the two mixtures share (user, 2026-11-19: "share 0 -> method 1, share 1-2 -> method 2,
share >= 3 -> method 3 ... not cheating"). The route is read from the input (the two molecule lists), so it is available
at test time. To avoid choosing each route's method from the same pairs it is scored on, the choice is made INSIDE
every outer training fold: an inner 4-fold mixture-holdout CV on the training pairs only scores every candidate per
route (r), the best one is refitted on all training pairs and predicts that route's test pairs.
Candidates (frozen networks trained on single molecules only; DREAM fits only the readout):
  M1 counts + POM order, O1.2b rule (27)        M2 counts, O1.2b rule (27)
  M3 counts, attention (67)                      M4 one attention compared in both spaces, POM space centred (67)
  M5 width 64 without counts, attention (67)     M6 width 64 + POM order without counts, O1.2b rule (27)
  (M5 / M6 added after the other session found the counts' share-none gain does not carry over to T46 / V50)
Rows on the SAME outer folds (10 x 5 to keep nested fitting affordable): O1.2b; each candidate alone; routed (nested
choice); routed with a fixed choice taken from earlier 50 x 5 results (M1 / M2 / M3; optimistic, chosen on these pairs).

Env:     mist
Usage:   python evaluation/O3/eval_o3p_route.py [--repeats 10]
Writes:  results/eval_o3p_route/<date_time>.{txt,json}
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import cv, evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
from eval_o3p_twoview import BothPlaces  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
K, INNER_K, SEED = 5, 4, 0
REF = "O1.2b (reference, 27)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=10); a_ = ap.parse_args(); R = a_.repeats
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_route", {"repeats": R, "k": K, "inner_k": INNER_K, "seeds": SEEDS, "lam": LAM, "device": dev})
    G, tr_g, te_g, names = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows()
    tr_s = np.random.default_rng(0).choice(tr_g, 2000, replace=False); V, Vc, Q = {}, {}, {}
    G0 = o2.Graphs(features.features_dir() / "o2_graphs.npz")
    for k, folder in (("counts", "ga_w64"), ("pom", "ga_w64_rank"), ("w64", "o3m_w64"), ("w64pom", "o3p_rank")):
        ga = folder.startswith("ga_"); g_ = G if ga else G0
        ns = [(train_groupatoms.load if ga else train_o3m.load)(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev) for s in SEEDS]; V[k], Vc[k] = [], []
        for n in ns:
            with torch.no_grad():
                Ts = n.latent(o2.encode_all(n, g_, tr_s, dev))[0].double().cpu().numpy()
                D = n.latent(o2.encode_all(n, g_, rows_D, dev))[0].double().cpu().numpy()
            V[k].append(D / np.linalg.norm(Ts, axis=1).mean()); mu = Ts.mean(0); Vc[k].append((D - mu) / np.linalg.norm(Ts - mu, axis=1).mean())
        Q[k] = mixture.quantile_per_word(np.mean([o2.predict_probs(n, g_, rows_D, dev) for n in ns], 0).astype(float))
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    route = np.where(k_P == 0, 0, np.where(k_P <= 2, 1, 2)); RN = ["share none", "share 1-2", "share >= 3"]
    cand = {"M1 counts + POM order, O1.2b rule (27)": lambda a, b: O12b(Q["pom"], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b),
            "M2 counts, O1.2b rule (27)": lambda a, b: O12b(Q["counts"], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b),
            "M3 counts, attention (67)": lambda a, b: np.mean(
                [ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V["counts"]], 0),
            "M4 attention in both spaces, POM centred (67)": lambda a, b: np.mean(
                [BothPlaces(w, p, A, B).fit(a, y[a]).predict(b) for w, p in zip(V["counts"], Vc["pom"])], 0),
            "M5 width 64 no counts, attention (67)": lambda a, b: np.mean(
                [ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V["w64"]], 0),
            "M6 width 64 + POM order no counts, O1.2b rule (27)": lambda a, b: O12b(Q["w64pom"], X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b)}
    names_c = list(cand); fixed = [names_c[0], names_c[1], names_c[2]]
    ref = lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b)  # noqa: E731
    out = {n: ([], []) for n in [REF] + names_c + ["routed (choice made inside each training fold)", "routed, fixed choice M1 / M2 / M3 (optimistic)"]}
    chosen = [Counter() for _ in RN]
    folds = cv.mixture_holdout_folds(P, k=K, repeats=R, seed=SEED)
    for fi, f in enumerate(folds):
        tr, te = np.array(f.train), np.array(f.test)
        sub = [P[i] for i in tr]; inner = cv.mixture_holdout_folds(sub, k=INNER_K, repeats=1, seed=1000 + fi)
        ipred = {n: ([], []) for n in names_c}
        for g in inner:
            itr, ite = tr[np.array(g.train)], tr[np.array(g.test)]
            if len(ite) == 0:
                continue
            for n, fn in cand.items():
                ipred[n][0].append(ite); ipred[n][1].append(fn(itr, ite))
        pick = []
        for r_ in range(3):
            best, bscore = None, -np.inf
            for n in names_c:
                idx = np.concatenate(ipred[n][0]); pr = np.concatenate(ipred[n][1]); m = route[idx] == r_
                sc = np.corrcoef(pr[m], y[idx][m])[0, 1] if m.sum() > 2 else -np.inf
                if sc > bscore:
                    best, bscore = n, sc
            pick.append(best); chosen[r_][best] += 1
        full = {n: fn(tr, te) for n, fn in cand.items()}
        out[REF][0].append(te); out[REF][1].append(ref(tr, te))
        for n in names_c:
            out[n][0].append(te); out[n][1].append(full[n])
        rt = route[te]
        out["routed (choice made inside each training fold)"][0].append(te)
        out["routed (choice made inside each training fold)"][1].append(np.choose(rt, [full[pick[0]], full[pick[1]], full[pick[2]]]))
        out["routed, fixed choice M1 / M2 / M3 (optimistic)"][0].append(te)
        out["routed, fixed choice M1 / M2 / M3 (optimistic)"][1].append(np.choose(rt, [full[fixed[0]], full[fixed[1]], full[fixed[2]]]))
        if (fi + 1) % 5 == 0:
            rep.line(f"outer fold {fi + 1}/{len(folds)}; picks so far: " + "; ".join(f"{RN[i]} {dict(chosen[i])}" for i in range(3)))
    res = {}
    for n, (ii, pp_) in out.items():
        idx = np.concatenate(ii)
        res[n] = evaluate.CVResult(idx, np.concatenate(pp_), y[idx], np.array([P[i].dataset for i in idx]),
                                   np.array([P[i].jaccard == 0 for i in idx]), K, R, SEED)
    SUB = {"all pairs": None, "share none": k_P == 0, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}

    def sc(r_, s, against):
        m = SUB[s]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section("Method picked per route inside the training folds (count over outer folds)")
    rep.table(["route", "picks"], [[RN[i], ", ".join(f"{n.split(' ')[0]} {c}" for n, c in chosen[i].most_common())] for i in range(3)])
    rep.section(f"DREAM development pairs ({R} x {K}, same outer folds for every row): RMSE / r change [95% CI]; positive r = better")
    rows = []
    for s in SUB:
        for n, r_ in res.items():
            s_ = sc(r_, s, None if n == REF else res[REF])
            rows.append([n, s, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", "" if n == REF else ch(s_)])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b"], rows)
    rep.save()


if __name__ == "__main__":
    main()
