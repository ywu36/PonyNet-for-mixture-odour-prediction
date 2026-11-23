# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = GoodScents + Ma molecules   M = DREAM molecules (235)   H = middle layer (400)
#   G = RDKit groups (24)   Q = GoodScents pairs   T = Ma mixtures   P = DREAM development pairs (514)   K = members
# ---------------------------------------------------------------------------------------------
"""O3-W step 1 (the user, 2026-11-19): learn ONLY how chemical groups set a molecule's importance in a mixture, from the
new data, with the mixture losses learned in the other session; the network is frozen; merging in the hidden layer
(no word-level merge, nothing decoded).

Frozen O2-C-Single (10 members), hidden vector v (centred middle layer, 400). Mixture template
z = sum_m w_m v_m / sum_m w_m, importance w_m = exp(group weights . x_m) (24 group weights, shared by every molecule).
Losses (set vs single, as train_o3t.py L4 / L6), score d = cos(z, v_A) - cos(z, v_B) for a two-molecule set {A, B}:
  GoodScents lean: label = which ingredient the blend leans to, from real notes (train_o3t.scent_rel_labels, gap 0.05)
  Ma: label = sign(L) (which molecule dominates)
  loss = softplus(-(2 label - 1) d / temperature), one temperature per dataset; + 1e-3 |group weights|^2.
Trained numbers: 24 group weights + 2 temperatures, per member (Adam, full batch). Nothing else changes.

Checks:
  1. GoodScents lean, official held-out pairs (both molecules unseen): AUROC of d (group weights 0 / GoodScents / both)
  2. Ma, 5 unseen-molecule folds (fold molecules never in training): AUROC of d; group weights 0 / Ma only /
     GoodScents only (Ma never seen) / GoodScents + Ma
  3. DREAM development pairs, frozen template, 50 x 5, change from O1.2b (paired), 4 subsets:
     group weights 0 + length weight + line (3 numbers); group weights from GoodScents (3); from GoodScents + Ma (3);
     all 27 fitted on DREAM pulled toward 0 (= plain template); all 27 pulled toward the GoodScents + Ma weights.

Env:     mist
Usage:   python evaluation/O3/eval_o3w.py
Writes:  results/eval_o3w/<date_time>.{txt,json}, features/o3w_group_weights.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o3t import scent_rel_labels  # noqa: E402

F = torch.float64


def score(v, xg, a, b, gw):
    """d = cos(z, v_a) - cos(z, v_b), z = importance-weighted mean of v_a, v_b."""
    wa, wb = torch.exp(xg[a] @ gw).unsqueeze(1), torch.exp(xg[b] @ gw).unsqueeze(1)
    z = (wa * v[a] + wb * v[b]) / (wa + wb); c = torch.nn.functional.cosine_similarity
    return c(z, v[a], dim=1) - c(z, v[b], dim=1)


def fit_weights(v, xg, sets, lam=1e-3, steps=500):
    """sets: list of (a, b, label 0/1 tensor). -> group weights (24)."""
    gw = torch.zeros(xg.shape[1], dtype=F, device=v.device, requires_grad=True)
    lt = torch.full((len(sets),), np.log(0.1), dtype=F, device=v.device, requires_grad=True)
    opt = torch.optim.Adam([gw, lt], lr=0.05)
    for _ in range(steps):
        loss = lam * (gw ** 2).sum()
        for k, (a, b, y) in enumerate(sets):
            loss = loss + torch.nn.functional.softplus(-(2 * y - 1) * score(v, xg, a, b, gw) / torch.exp(lt[k])).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return gw.detach()


class Template(O12b):
    """O1.2b's form on hidden vectors, importance = exp((g0 + fitted) . x); the penalty pulls the fitted part to 0, i.e.
    the group weights toward g0. use_theta False: group weights fixed at g0 (DREAM fits length weight + line only)."""

    def __init__(self, v, x, A, B, g0, use_theta, lam=LAM):
        super().__init__(v, x, A, B, use_theta, True, lam=lam); self.g0 = torch.as_tensor(np.asarray(g0, float))

    def distance(self, theta, k, idx):
        return super().distance(theta + self.g0, k, idx)


def main():
    dev = pick_device(); rep = Report("eval_o3w", {"lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, gn = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    XN = torch.as_tensor(o3.group_matrix(z["smiles"], gn), dtype=F)
    lab = scent_rel_labels(G, S)
    q_all = np.where((S.pair_split == 0) & ~np.isnan(lab))[0]
    q_tr = np.sort(np.random.default_rng(0).choice(q_all, min(20000, len(q_all)), replace=False)); q_te = np.where((S.pair_split == 1) & ~np.isnan(lab))[0]
    L = S.ma_L; ma_folds = [S.ma_split(f) for f in range(5)]
    nets = o2.load_members(sorted((ROOT / "checkpoints/o2c").glob("member_*.pt")), dev)
    used = np.unique(np.r_[S.pair_a[q_tr], S.pair_b[q_tr], S.pair_a[q_te], S.pair_b[q_te], S.ma_a, S.ma_b])
    VN, VM = [], []
    for n in nets:
        v = torch.zeros(len(z["smiles"]), 400, dtype=F)
        for i in range(0, len(used), 512):
            u = used[i:i + 512]; x, e, s, d, g = S.batch(u, dev)
            with torch.no_grad():
                v[u] = n.latent(n.encode(x, e, s, d, g, int(g.max()) + 1))[0].double().cpu()
        VN.append(v.to(dev)); VM.append(n.latent(o2.encode_all(n, G, G.dream_rows(), dev))[0].double().cpu().numpy())
    rep.line(f"members {len(nets)}; GoodScents lean labels: train {len(q_all)} (random 20,000 used per fit), held-out {len(q_te)} (both molecules unseen; "
             f"leans to A {np.nanmean(lab[q_tr]):.3f}); Ma mixtures {len(L)} in 5 unseen-molecule folds; trained numbers: 24 group weights + 2 temperatures")
    XN = XN.to(dev); tt = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    sc_tr = (tt(S.pair_a[q_tr]), tt(S.pair_b[q_tr]), tt(lab[q_tr]))
    ma_all = (tt(S.ma_a), tt(S.ma_b), tt((L > 0).astype(float)))

    def ma_sets(idx):
        return (tt(S.ma_a[idx]), tt(S.ma_b[idx]), tt((L[idx] > 0).astype(float)))

    # 1 + 2: fits per member
    zero = torch.zeros(len(gn), dtype=F, device=dev)
    gw_sc = [fit_weights(v, XN, [sc_tr]) for v in VN]
    gw_both = [fit_weights(v, XN, [sc_tr, ma_all]) for v in VN]
    rep.line("GoodScents and GoodScents + Ma fits done")

    def sc_auc(gws):
        d = np.mean([score(v, XN, tt(S.pair_a[q_te]), tt(S.pair_b[q_te]), g).detach().cpu().numpy() for v, g in zip(VN, gws)], 0)
        return roc_auc_score(lab[q_te], d)
    rep.section("1. GoodScents: which ingredient the blend leans to, held-out pairs (both molecules unseen), AUROC")
    rep.table(["group weights", "AUROC"], [["0 (hidden vectors only)", f"{sc_auc([zero] * len(VN)):.3f}"], ["from GoodScents", f"{sc_auc(gw_sc):.3f}"],
                                           ["from GoodScents + all Ma", f"{sc_auc(gw_both):.3f}"]])

    pred = {k: np.zeros(len(L)) for k in ("0", "Ma", "GoodScents", "GoodScents + Ma")}; cnt = np.zeros(len(L))
    for f, (tr, te) in enumerate(ma_folds):
        cnt[te] += 1
        for k, mk in (("0", lambda v: zero), ("Ma", lambda v: fit_weights(v, XN, [ma_sets(tr)])),
                      ("GoodScents", None), ("GoodScents + Ma", lambda v: fit_weights(v, XN, [sc_tr, ma_sets(tr)]))):
            gws = gw_sc if k == "GoodScents" else [mk(v) for v in VN]
            pred[k][te] += np.mean([score(v, XN, tt(S.ma_a[te]), tt(S.ma_b[te]), g).detach().cpu().numpy() for v, g in zip(VN, gws)], 0)
        rep.line(f"Ma fold {f}: done")
    rng = np.random.default_rng(0); y_ma = (L > 0).astype(int)
    rows = []
    for k, p_ in pred.items():
        p_ = p_ / cnt; auc = roc_auc_score(y_ma, p_)
        bs = [roc_auc_score(y_ma[i], p_[i]) for i in (rng.integers(0, len(L), len(L)) for _ in range(2000)) if 0 < y_ma[i].mean() < 1]
        rows.append([k, f"{auc:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"])
    rep.section("2. Ma: which molecule dominates, 5 unseen-molecule folds (198 mixtures), AUROC [95% CI, mixtures resampled]")
    rep.table(["group weights learned from", "AUROC"], rows)
    rep.line("for scale (other session, same folds): chemical groups alone (ridge) 0.644 [0.567, 0.726]")

    gsc, gboth = torch.stack(gw_sc).mean(0).cpu().numpy(), torch.stack(gw_both).mean(0).cpu().numpy()
    np.savez(features.features_dir() / "o3w_group_weights.npz", groups=np.array(gn), goodscents=gsc, goodscents_ma=gboth)
    rep.section("Group weights (mean over members): importance factor = e^(sum of the molecule's group weights)")
    order = np.argsort(-np.abs(gboth))
    rep.table(["group", "from GoodScents", "from GoodScents + Ma"], [[gn[g], f"{gsc[g]:+.3f}", f"{gboth[g]:+.3f}"] for g in order])

    # 3: DREAM
    from joblib import Parallel, delayed
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    pool = Parallel(n_jobs=len(VM), backend="loky")

    def one(v, g0, use, tr, te):
        torch.set_num_threads(1)
        return Template(v, Xd, A, B, g0, use).fit(tr, y[tr]).predict(te)

    def ens(g0, use):
        return lambda tr, te: np.mean(pool(delayed(one)(v, g0, use, tr, te) for v in VM), 0)
    z24 = np.zeros(len(gn))
    rows_def = {"O1.2b (reference, word level)": lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), Xd, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te),
                "template, group weights 0 (3 numbers)": ens(z24, False),
                "template, group weights from GoodScents (3)": ens(gsc, False),
                "template, group weights from GoodScents + Ma (3)": ens(gboth, False),
                "template, 27 fitted on DREAM, pulled to 0": ens(z24, True),
                "template, 27 fitted on DREAM, pulled to GoodScents + Ma": ens(gboth, True)}
    res = {}
    for k, f in rows_def.items():
        res[k] = evaluate.cross_validate(P, f); rep.line(f"{k}: done")
    ref = res["O1.2b (reference, word level)"]

    def sc_(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
    rep.section(f"3. DREAM development pairs ({evaluate.CV_REPEATS} x 5), frozen hidden template: change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc_(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)
    rep.section("Paired steps (same folds)")
    rows = []
    for hi, lo in (("template, group weights from GoodScents (3)", "template, group weights 0 (3 numbers)"),
                   ("template, group weights from GoodScents + Ma (3)", "template, group weights 0 (3 numbers)"),
                   ("template, 27 fitted on DREAM, pulled to GoodScents + Ma", "template, 27 fitted on DREAM, pulled to 0")):
        for sub in SUB:
            s_ = sc_(res[hi], sub, res[lo])
            rows.append([f"{hi}  minus  {lo}", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["comparison", "subset", "RMSE change", "r change [95% CI]"], rows)
    rep.save()


if __name__ == "__main__":
    main()
