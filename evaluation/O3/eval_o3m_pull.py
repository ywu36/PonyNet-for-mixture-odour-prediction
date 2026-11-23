# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = GoodScents + Ma molecules   M = DREAM molecules (235)   H = hidden layer (64)
#   Q = GoodScents pairs   T = Ma mixtures   P = development pairs (514)   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Fully fixed width-64 network (checkpoints/o3m_w64, 3 members); only a "pull" add-on is learned:
    weight of molecule m in a mixture  w_m = exp(u . v_m)     u = one direction in the hidden layer (64 numbers)
    mixture z = sum_m w_m v_m / sum_m w_m                    (u = 0: plain mean, a molecule's pull = |v_m| only)
The strength is read from the network's own hidden vector, not from hand-made groups (the user, 2026-11-19).
u is fitted (L-BFGS, ridge 1e-3 |u|^2) with frozen network and frozen linear head on:
    Ma who-wins   logistic(s_1 (cos(z, v_A) - cos(z, v_B))) vs sign(L)            (Ma folds 1-4; fold 0 held out)
    GS lean-to    logistic(s_2 (cos(z, v_A) - cos(z, v_B))) vs "blend leans to A"   (count-neutral label "excess")
    GS notes      head(z) / tau vs the blend's real notes (rank loss over 79)        (official training pairs)
(s_1, s_2, tau: one positive number each.) Fits: u = 0; Ma; lean; Ma + lean; Ma + lean + notes; the last with all
training labels shuffled (control). Checks on items never fitted: official GoodScents held-out pairs (both molecules
unseen), Ma fold 0. Then DREAM: u frozen, z as above, O1.2b's distance (direction + length), 3 numbers fitted (beta +
line), mean over members; paired vs O1.2b, the open-POM baseline and u = 0; 50 x 5; 4 subsets.

Env:     mist
Usage:   python evaluation/O3/eval_o3m_pull.py
Writes:  results/eval_o3m_pull/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, o3, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from eval_o3e import O3Fixed  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o2j import word_auroc  # noqa: E402
from train_o3s_scent import MA_HOLDOUT_FOLD  # noqa: E402
from train_o3t import scent_rel_labels  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
cos = torch.nn.functional.cosine_similarity


@torch.no_grad()
def hidden_all(net, store, ids, dev, batch=512):
    out = []
    for i in range(0, len(ids), batch):
        b = ids[i:i + batch]; x, e, s, d, g = store.batch(b, dev); out.append(net.latent(net.encode(x, e, s, d, g, len(b)))[0])
    return torch.cat(out).double().cpu()


def pair_d(V, u, a, b):
    va, vb = V[a], V[b]; wa, wb = torch.exp(va @ u).unsqueeze(1), torch.exp(vb @ u).unsqueeze(1)
    z = (wa * va + wb * vb) / (wa + wb)
    return z, cos(z, va, dim=1) - cos(z, vb, dim=1)


def fit(V, Wh, data, use, lam=1e-3):
    """L-BFGS on the chosen losses. data: dict with ma (a, b, y), lean (a, b, y), notes (a, b, Y, T)."""
    H = V.shape[1]; u = torch.zeros(H, dtype=F64, requires_grad="u" in use)
    ls = torch.zeros(3, dtype=F64, requires_grad=True)       # log s_1, log s_2, log tau
    params = ([u] if "u" in use else []) + [ls]
    opt = torch.optim.LBFGS(params, max_iter=300, line_search_fn="strong_wolfe")
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def closure():
        opt.zero_grad(); loss = lam * (u ** 2).sum()
        if "ma" in use:
            a, b, y = data["ma"]; _, d = pair_d(V, u, a, b); loss = loss + bce(torch.exp(ls[0]) * d, y)
        if "lean" in use:
            a, b, y = data["lean"]; _, d = pair_d(V, u, a, b); loss = loss + bce(torch.exp(ls[1]) * d, y)
        if "notes" in use:
            a, b, Y, T = data["notes"]; z, _ = pair_d(V, u, a, b)
            loss = loss + o3.rank_loss((z @ Wh[0].T + Wh[1]) / torch.exp(ls[2]), Y, T).mean()
        loss.backward(); return loss
    opt.step(closure)
    return u.detach(), ls.detach()


def main():
    dev = pick_device(); rep = Report("eval_o3m_pull", {"lam": 1e-3, "repeats": evaluate.CV_REPEATS, "device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    n_S = len(S.node_off) - 1
    rel = scent_rel_labels(G, S, measure="excess"); T = torch.as_tensor(S.targets)
    tr_q, te_q = np.where(S.pair_split == 0)[0], np.where(S.pair_split == 1)[0]
    lab_tr, lab_te = tr_q[~np.isnan(rel[tr_q])], te_q[~np.isnan(rel[te_q])]
    ma_tr, ma_te = S.ma_split(MA_HOLDOUT_FOLD); ma_y = (S.ma_L > 0).astype(float)
    rng = np.random.default_rng(0); notes_tr = rng.choice(tr_q, 40000, replace=False)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)  # API only
    A, B = membership(P, op.cids_M)
    rep.line(f"GoodScents lean-to labels: training {len(lab_tr)}, held-out {len(lab_te)}; notes: training sample {len(notes_tr)}, "
             f"held-out {len(te_q)}; Ma training {len(ma_tr)}, held-out fold 0 {len(ma_te)}")
    FITS = {"u = 0 (plain mean)": set(), "Ma": {"u", "ma"}, "GS lean-to": {"u", "lean"}, "Ma + GS lean-to": {"u", "ma", "lean"},
            "Ma + GS lean-to + GS notes": {"u", "ma", "lean", "notes"}, "same, labels shuffled (control)": {"u", "ma", "lean", "notes", "shuffle"}}
    nets = [train_o3m.load(f, dev) for f in sorted((ROOT / "checkpoints/o3m_w64").glob("member_*.pt"))]
    rows, U = [], {k: [] for k in FITS}; VD = []
    for k_m, net in enumerate(nets):
        V = hidden_all(net, S, np.arange(n_S), dev); VD.append(hidden_all(net, G, G.dream_rows(), dev).numpy())
        Wh = (net.head.weight.detach().double().cpu(), net.head.bias.detach().double().cpu())
        t = lambda a: torch.as_tensor(a, dtype=F64)  # noqa: E731
        for name, use in FITS.items():
            sh = np.random.default_rng(100 + k_m)
            ym, yl, Yn = ma_y[ma_tr].copy(), rel[lab_tr].copy(), S.pair_Y[notes_tr].astype(float).copy()
            if "shuffle" in use:
                ym, yl, Yn = ym[sh.permutation(len(ym))], yl[sh.permutation(len(yl))], Yn[sh.permutation(len(Yn))]
            data = {"ma": (S.ma_a[ma_tr], S.ma_b[ma_tr], t(ym)), "lean": (S.pair_a[lab_tr], S.pair_b[lab_tr], t(yl)),
                    "notes": (S.pair_a[notes_tr], S.pair_b[notes_tr], t(Yn), T)}
            u, ls = fit(V, Wh, data, use - {"shuffle"}) if use else (torch.zeros(V.shape[1], dtype=F64), torch.zeros(3, dtype=F64))
            U[name].append(u.numpy())
            with torch.no_grad():
                _, dm = pair_d(V, u, S.ma_a[ma_te], S.ma_b[ma_te]); _, dl = pair_d(V, u, S.pair_a[lab_te], S.pair_b[lab_te])
                z, _ = pair_d(V, u, S.pair_a[te_q], S.pair_b[te_q]); lg = z @ Wh[0].T + Wh[1]
                zt, _ = pair_d(V, u, S.pair_a[notes_tr[:5000]], S.pair_b[notes_tr[:5000]]); lt = zt @ Wh[0].T + Wh[1]
                grid = torch.logspace(-2, 1.5, 71, dtype=F64); Yt = t(S.pair_Y[notes_tr[:5000]])
                tau = grid[int(torch.stack([o3.rank_loss(lt / g, Yt, T).mean() for g in grid]).argmin())]
                nl = o3.rank_loss(lg / tau, t(S.pair_Y[te_q]), T).mean().item()
            rows.append([k_m, name, f"{roc_auc_score(ma_y[ma_te], dm.numpy()):.3f}", f"{roc_auc_score(rel[lab_te], dl.numpy()):.3f}",
                         f"{word_auroc(lg.numpy(), S.pair_Y[te_q], S.targets)[0]:.4f}", f"{nl:.4f}", f"{u.norm().item():.3f}",
                         f"{float(np.std((V @ u).numpy())):.3f}"])
            rep.line(f"member {k_m} {name}: done")
    rep.section("Items never fitted: Ma fold 0 (48 mixtures), GoodScents official held-out pairs (both molecules unseen)")
    rep.table(["member", "fit", "Ma who-wins AUROC", "GS lean-to AUROC", "GS notes word AUROC", "GS notes loss (tau on training)",
               "|u|", "spread of log weight over molecules (std)"], rows)

    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    q_op = mixture.quantile_per_word(op.probs_MxW)
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te)),
           "open-POM baseline (2 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(q_op, X, A, B, False, False, lam=LAM).fit(tr, y[tr]).predict(te))}
    for name in ("u = 0 (plain mean)", "Ma + GS lean-to", "Ma + GS lean-to + GS notes", "same, labels shuffled (control)"):
        xs = [(VD[k] @ U[name][k])[:, None] for k in range(len(nets))]
        res[f"width 64, pull {name} (3 numbers)"] = evaluate.cross_validate(
            P, lambda tr, te, xs=xs: np.mean([O3Fixed(VD[k], xs[k], A, B, [1.0]).fit(tr, y[tr]).predict(te) for k in range(len(nets))], 0))
        rep.line(f"DREAM done: {name}")
    ref, opb, zero = res["O1.2b (reference, 27 numbers)"], res["open-POM baseline (2 numbers)"], res["width 64, pull u = 0 (plain mean) (3 numbers)"]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if r_ is ref else ch(s_), "" if r_ is opb else ch(sc(r_, sub, opb)),
                         "" if (r_ is ref or r_ is opb or r_ is zero) else ch(sc(r_, sub, zero))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from open-POM baseline", "change from u = 0"], rows)
    rep.save()


if __name__ == "__main__":
    main()
