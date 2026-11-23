# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = GoodScents + Ma molecules   M = DREAM molecules (235)   D = vector size
#   Q = GoodScents pairs   T = Ma mixtures   P = DREAM development pairs (514)   K = members
# ---------------------------------------------------------------------------------------------
"""Latent readout (the user, 2026-11-19: this side = only the readout, "the last layer that O1.2b did"; group level
learned latently, no RDKit groups). Works on any frozen vector set (--space):
  o2c400   O2-C-Single centred set2set output (400)          o2c256   O2-C-Single POM embedding after the FFN (256)
  o3m_w<d> the other session's minimal backbone (d)
Readout: importance of molecule m = e^(q . s_m), s_m = v_m / (mean length of v over molecules) (scale only);
mixture template z = sum_m importance_m v_m / sum importance (a molecule's pull = its length x importance);
two mixtures compared by direction + length (O1.2b's form), then a line. Nothing decoded; q (D numbers) is the only
learned part beyond the length weight and the line.
q is learned from the new data with the other session's set-vs-single losses: d = cos(z, v_A) - cos(z, v_B) for the set
{A, B}; GoodScents "which ingredient the blend leans to" (labels from real notes, train_o3t.scent_rel_labels) and Ma
"which molecule dominates" (sign of L); softplus(-(2 label - 1) d / temperature) + 1e-3 |q|^2.
Checks: GoodScents lean held-out AUROC (both molecules unseen); Ma 5 unseen-molecule folds AUROC; DREAM development
pairs 50 x 5 vs O1.2b (paired, 4 subsets): plain mean (q = 0, 3 numbers), q from GoodScents + Ma (3 numbers), and q
fitted on DREAM (D + 3 numbers; only when D <= 64, the budget).

Env:     mist
Usage:   python evaluation/O3/eval_readout_latent.py --space o3m_w32 [--repeats 50]
Writes:  results/eval_readout_latent/<date_time>_<space>.{txt,json}
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
from mixdist import evaluate, features, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from eval_o3w import Template, fit_weights, score  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o3t import scent_rel_labels  # noqa: E402

F = torch.float64


def backbone(space, dev):
    """-> (nets, vector function net, graph batch -> B x D)."""
    if space.startswith("o2c"):
        nets = o2.load_members(sorted((ROOT / "checkpoints/o2c").glob("member_*.pt")), dev)
        if space == "o2c400":
            return nets, lambda n, b: n.latent(n.encode(*b, int(b[4].max()) + 1))[0]
        return nets, lambda n, b: n.readout(n.latent(n.encode(*b, int(b[4].max()) + 1))[0])[0]
    import train_o3m
    nets = [train_o3m.load(p, dev) for p in sorted((ROOT / f"checkpoints/{space}").glob("member_*.pt"))]
    return nets, lambda n, b: n.latent(n.encode(*b, int(b[4].max()) + 1))[0]


@torch.no_grad()
def vectors(nets, fn, store, ids, n_rows, dev):
    out = []
    for n in nets:
        v = None
        for i in range(0, len(ids), 512):
            u = ids[i:i + 512]; x = fn(n, store.batch(u, dev)).double().cpu()
            if v is None:
                v = torch.zeros(n_rows, x.shape[1], dtype=F)
            v[u] = x
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--space", required=True); ap.add_argument("--repeats", type=int, default=evaluate.CV_REPEATS)
    a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_readout_latent", {"space": a.space, "lam": LAM, "repeats": a.repeats, "device": dev}, tag=a.space)
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False); op = features.load("openpom")
    lab = scent_rel_labels(G, S)
    q_all = np.where((S.pair_split == 0) & ~np.isnan(lab))[0]
    q_tr = np.sort(np.random.default_rng(0).choice(q_all, min(20000, len(q_all)), replace=False)); q_te = np.where((S.pair_split == 1) & ~np.isnan(lab))[0]
    L = S.ma_L; ma_folds = [S.ma_split(f) for f in range(5)]
    nets, fn = backbone(a.space, dev)
    used = np.unique(np.r_[S.pair_a[q_tr], S.pair_b[q_tr], S.pair_a[q_te], S.pair_b[q_te], S.ma_a, S.ma_b])
    VN = vectors(nets, fn, S, used, len(z["smiles"]), dev)
    rows_M = G.dream_rows(); VM = [v[rows_M].numpy() for v in vectors(nets, fn, G, rows_M, len(G.smiles_N), dev)]
    scaleN = [float(v[used].norm(dim=1).mean()) for v in VN]
    SN = [v / s for v, s in zip(VN, scaleN)]; SM = [vm / np.linalg.norm(vm, axis=1).mean() for vm in VM]
    D = VN[0].shape[1]
    rep.line(f"space {a.space}: {len(nets)} members, vector size {D}; GoodScents lean train {len(q_all)} (20,000 used per fit), held-out {len(q_te)} "
             f"(both molecules unseen); Ma {len(L)} mixtures, 5 unseen-molecule folds; learned: q ({D}) + 2 temperatures")
    tt = torch.as_tensor
    sc_tr = (tt(S.pair_a[q_tr]), tt(S.pair_b[q_tr]), tt(lab[q_tr]))
    zero = torch.zeros(D, dtype=F)

    def fit(i, sets):
        return fit_weights(VN[i], SN[i], sets)

    def ma_sets(idx):
        return (tt(S.ma_a[idx]), tt(S.ma_b[idx]), tt((L[idx] > 0).astype(float)))
    K = len(VN)
    q_sc = [fit(i, [sc_tr]) for i in range(K)]; q_both = [fit(i, [sc_tr, ma_sets(np.arange(len(L)))]) for i in range(K)]
    d_te = lambda qs: np.mean([score(VN[i], SN[i], tt(S.pair_a[q_te]), tt(S.pair_b[q_te]), qs[i]).numpy() for i in range(K)], 0)  # noqa: E731
    rep.section("1. GoodScents: which ingredient the blend leans to, held-out pairs (both molecules unseen), AUROC")
    rep.table(["readout", "AUROC"], [["plain mean (pull = length)", f"{roc_auc_score(lab[q_te], d_te([zero] * K)):.3f}"],
                                     ["q from GoodScents", f"{roc_auc_score(lab[q_te], d_te(q_sc)):.3f}"],
                                     ["q from GoodScents + all Ma", f"{roc_auc_score(lab[q_te], d_te(q_both)):.3f}"]])
    pred = {k: np.zeros(len(L)) for k in ("plain mean", "q from Ma", "q from GoodScents", "q from GoodScents + Ma")}; cnt = np.zeros(len(L))
    for f, (tr, te) in enumerate(ma_folds):
        cnt[te] += 1
        qs = {"plain mean": [zero] * K, "q from Ma": [fit(i, [ma_sets(tr)]) for i in range(K)], "q from GoodScents": q_sc,
              "q from GoodScents + Ma": [fit(i, [sc_tr, ma_sets(tr)]) for i in range(K)]}
        for k, qq in qs.items():
            pred[k][te] += np.mean([score(VN[i], SN[i], tt(S.ma_a[te]), tt(S.ma_b[te]), qq[i]).numpy() for i in range(K)], 0)
    y_ma = (L > 0).astype(int); rng = np.random.default_rng(0); rows = []
    for k, p_ in pred.items():
        p_ = p_ / cnt; bs = [roc_auc_score(y_ma[i], p_[i]) for i in (rng.integers(0, len(L), len(L)) for _ in range(2000))]
        rows.append([k, f"{roc_auc_score(y_ma, p_):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"])
    rep.section("2. Ma: which molecule dominates, 5 unseen-molecule folds (198 mixtures), AUROC [95% CI, mixtures resampled]")
    rep.table(["readout", "AUROC"], rows); rep.line("for scale (same folds): chemical groups alone (ridge) 0.644 [0.567, 0.726]")

    from joblib import Parallel, delayed
    P = dev_pairs(); y = evaluate.labels(P); ids = [str(int(c)) for c in op.cids_M]
    Xd, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    pool = Parallel(n_jobs=min(K, 10), backend="loky")

    def one(i, q0, use, tr, te):
        torch.set_num_threads(1)
        return Template(VM[i], SM[i], A, B, q0, use).fit(tr, y[tr]).predict(te)

    def ens(qs, use):
        return lambda tr, te: np.mean(pool(delayed(one)(i, qs[i].numpy(), use, tr, te) for i in range(K)), 0)
    rows_def = {"O1.2b (reference, word level)": lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), Xd, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
                "plain mean, pull = length (3 numbers)": ens([zero] * K, False),
                "q from GoodScents + Ma (3 numbers)": ens(q_both, False)}
    if D <= 64:
        rows_def[f"q fitted on DREAM ({D} + 3 numbers)"] = ens([zero] * K, True)
    res = {}
    for k, f in rows_def.items():
        res[k] = evaluate.cross_validate(P, f, repeats=a.repeats); rep.line(f"{k}: done")
    ref = res["O1.2b (reference, word level)"]; base = res["plain mean, pull = length (3 numbers)"]

    def sc_(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
    rep.section(f"3. DREAM development pairs ({a.repeats} x 5), frozen {a.space}: change from O1.2b (paired); and minus plain mean")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc_(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            st = ""
            if r_ is not ref and r_ is not base:
                s2 = sc_(r_, sub, base); st = f"{s2['r_change']:+.3f} [{s2['r_change_ci'][0]:+.3f}, {s2['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch, st])
    rep.table(["readout", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]", "r minus plain mean [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
