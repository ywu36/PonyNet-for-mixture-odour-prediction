# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   T = Ma mixtures (198)
#   P = DREAM development pairs (514)   K = members (3)
# ---------------------------------------------------------------------------------------------
"""O3s readout + Ma direction regulariser (the user, 2026-11-19).

O3s = node-enriched width-64 network (checkpoints/ga_w64: open-POM atom features + log1p counts of 36 RDKit groups at
0-4 bonds), trained on GS-LF single molecules only, frozen; readout = hidden-layer attention (clue C17,
eval_context_dream.ContextReadoutQ, direction only):
    share_m ~ exp(q . s_m + kappa cos(v_m, mean of the other molecules)),  z = share-weighted mean of v,
    d = angle between z_A and z_B (|u_A - u_B|), line.   q 64 + kappa + line = 67 numbers, fitted on DREAM folds.
Ma direction regulariser (the user's form), same q: for Ma mixture {A, B} (kappa terms cancel for two molecules)
    z = a v_A + (1 - a) v_B, a = sigmoid(q . (s_A - s_B));  m = cos(z, v_A) - cos(z, v_B);
    loss_Ma = softplus(-y m / T), y = +1 if A dominates (Ma L > 0) else -1, T one temperature.
    loss = DREAM MSE + 1e-3 (|q|^2 + kappa^2) + c var(y_train) loss_Ma;  c = 0 is the C17 readout (DREAM only).
No GoodScents (its valid use, the O2-J word ranking, trains the encoder, not this readout).
Checks:
  1. Ma learnable through q? 5 unseen-molecule folds, q fitted on Ma alone (1e-3 |q|^2): AUROC of m (q = 0: chance).
  2. Does the loss go down? Fits on all DREAM pairs: DREAM training r, Ma training loss and AUROC, per c.
  3. DREAM development pairs, 50 x 5, change from O1.2b and from c = 0 (paired, same folds), 4 subsets.
Ma molecules get the same 36 group counts (same names and order as the backbone's training; element order checked).

Env:     mist
Usage:   python evaluation/O3/eval_o3s_attn_ma.py [--c 1 10] [--repeats 50]
Writes:  results/eval_o3s_attn_ma/<date_time>.{txt,json}
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
from mixdist import evaluate, features, group_atoms, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402

F64 = torch.float64
SEEDS = (0, 1, 2)
ecd.DIRECTION_ONLY = True


def attach_named(S, smiles_N, names, rows):
    """Append log1p group-distance counts (groups `names`, in that order) to S.node for molecules `rows`; others 0."""
    pats = [p for p in group_atoms.patterns() if p[0] in set(names)]
    assert [p[0] for p in pats] == list(names), "group order differs from the backbone's"
    n_atoms = np.diff(S.node_off); extra = np.zeros((len(S.node), len(pats) * group_atoms.BINS), np.float32)
    for i in rows:
        a0, a1 = S.node_off[i], S.node_off[i + 1]
        extra[a0:a1] = group_atoms.counts_VxKD(smiles_N[i], int(n_atoms[i]), pats, S.node[a0:a1])
    S.node = np.concatenate([S.node, np.log1p(extra)], 1).astype(np.float32)


def ma_stats(V, a, b, y):
    V = torch.as_tensor(V, dtype=F64); va, vb = V[a], V[b]
    return ((va * va).sum(1), (vb * vb).sum(1), (va * vb).sum(1), va - vb, torch.as_tensor(2.0 * y - 1, dtype=F64))


def ma_score(s, q):
    aa, bb, ab, dv, _ = s
    al = torch.sigmoid(dv @ q)
    zz = (al ** 2 * aa + 2 * al * (1 - al) * ab + (1 - al) ** 2 * bb).clamp_min(1e-12).sqrt()
    return (al * aa + (1 - al) * ab) / (zz * aa.clamp_min(1e-12).sqrt()) - (al * ab + (1 - al) * bb) / (zz * bb.clamp_min(1e-12).sqrt())


def ma_loss(s, q, lt):
    return torch.nn.functional.softplus(-s[4] * ma_score(s, q) / torch.exp(lt)).mean()


def fit_ma_only(s, D, lam=1e-3):
    q = torch.zeros(D, dtype=F64, requires_grad=True); lt = torch.tensor(float(np.log(0.1)), dtype=F64, requires_grad=True)
    opt = torch.optim.LBFGS([q, lt], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); loss = lam * (q ** 2).sum() + ma_loss(s, q, lt); loss.backward(); return loss
    opt.step(closure)
    return q.detach()


class AttnMa(ecd.ContextReadoutQ):
    """C17 attention readout (q, kappa, line on DREAM), the same q also answering Ma's direction (c > 0)."""

    def __init__(self, V, A, B, s_ma, c, lam=LAM):
        super().__init__(V, A, B, np.zeros(len(V)), True, lam=lam); self.s_ma, self.c = s_ma, c

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float)); vy = float(y.var())
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True); kappa = torch.zeros((), dtype=F64, requires_grad=True)
        b = torch.tensor(0.0, dtype=F64); lt = torch.tensor(float(np.log(0.1)), dtype=F64, requires_grad=True)
        with torch.no_grad():
            self.lw = self.V @ q; d = self.distance(kappa, b, idx); D = torch.stack([d, torch.ones_like(d)], 1)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True)
        opt = torch.optim.LBFGS([lin, q, kappa] + ([lt] if self.c > 0 else []), max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.V @ q; d = self.distance(kappa, b, idx)
            loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2)
            if self.c > 0:
                loss = loss + self.c * vy * ma_loss(self.s_ma, q, lt)
            loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.V @ q
        self.q, self.kappa, self.b, self.lin, self.lt = q.detach(), kappa.detach(), b, lin.detach(), lt.detach(); self.lw = self.lw.detach()
        return self


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--c", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--repeats", type=int, default=evaluate.CV_REPEATS); a_ = ap.parse_args()
    dev = pick_device()
    rep = Report("eval_o3s_attn_ma", {"c": a_.c, "lam": LAM, "repeats": a_.repeats, "seeds": SEEDS, "device": dev}, tag="smoke" if a_.repeats < 5 else "")
    Gg, tr, te, names = train_groupatoms.graphs_with_groups()
    S = o2j.Store(features.features_dir() / "o2j_data.npz"); z = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False)
    used = np.unique(np.r_[S.ma_a, S.ma_b]); attach_named(S, z["smiles"], names, used)
    assert S.node.shape[1] == Gg.node.shape[1]
    nets = [train_groupatoms.load(ROOT / f"checkpoints/ga_w64/member_{s}.pt", dev) for s in SEEDS]
    rows_D = Gg.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)   # as eval_context_counts.py
    VM, VN = [], []
    Q_cnt = mixture.quantile_per_word(np.mean([o2.predict_probs(n, Gg, rows_D, dev) for n in nets], 0).astype(float))   # as eval_groupatoms
    for n in nets:
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, Gg, tr_s, dev))[0].norm(dim=1).mean().item()
            VM.append(n.latent(o2.encode_all(n, Gg, rows_D, dev))[0].double().cpu().numpy() / scale)
            vn = np.zeros((len(z["smiles"]), VM[-1].shape[1])); vn[used] = n.latent(o2.encode_all(n, S, used, dev))[0].double().cpu().numpy() / scale
            VN.append(vn)
    del nets; torch.cuda.empty_cache()
    L = S.ma_L; y_ma = (L > 0).astype(float); ma_folds = [S.ma_split(f) for f in range(5)]
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(Gg.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
           "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
    rep.line(f"O3s: ga_w64 members {SEEDS}, groups {len(names)}, hidden {VM[0].shape[1]}; Ma mixtures {len(L)} on {len(used)} molecules "
             f"(A dominates {y_ma.mean():.3f}); DREAM development pairs {len(P)}; GoodScents not used")

    # 1. Ma learnable through q (unseen molecules)
    pred = {"q = 0 (plain mean)": np.zeros(len(L)), "q fitted on Ma (training folds)": np.zeros(len(L))}; cnt = np.zeros(len(L))
    for trf, tef in ma_folds:
        cnt[tef] += 1
        for v in VN:
            s_tr, s_te = ma_stats(v, S.ma_a[trf], S.ma_b[trf], y_ma[trf]), ma_stats(v, S.ma_a[tef], S.ma_b[tef], y_ma[tef])
            pred["q = 0 (plain mean)"][tef] += ma_score(s_te, torch.zeros(v.shape[1], dtype=F64)).numpy() / len(VN)
            pred["q fitted on Ma (training folds)"][tef] += ma_score(s_te, fit_ma_only(s_tr, v.shape[1])).numpy() / len(VN)
    rng = np.random.default_rng(0); rows = []
    for k, p_ in pred.items():
        p_ = p_ / cnt; bs = [roc_auc_score(y_ma[i], p_[i]) for i in (rng.integers(0, len(L), len(L)) for _ in range(2000)) if 0 < y_ma[i].mean() < 1]
        rows.append([k, f"{roc_auc_score(y_ma, p_):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"])
    rep.section("1. Ma: which molecule dominates, 5 unseen-molecule folds, AUROC [95% CI, mixtures resampled]")
    rep.table(["readout", "AUROC"], rows)

    s_all = [ma_stats(v, S.ma_a, S.ma_b, y_ma) for v in VN]
    # 2. loss goes down? fits on all DREAM pairs
    rows = []
    for c in [0.0] + list(a_.c):
        ms = [AttnMa(v, A, B, s, c).fit(np.arange(len(P)), y) for v, s in zip(VM, s_all)]
        pr = np.mean([m.predict(np.arange(len(P))) for m in ms], 0)
        mloss = np.mean([ma_loss(s, m.q, torch.tensor(float(np.log(0.1)), dtype=F64) if c == 0 else m.lt).item() for m, s in zip(ms, s_all)])
        mauc = roc_auc_score(y_ma, np.mean([ma_score(s, m.q).numpy() for m, s in zip(ms, s_all)], 0))
        rows.append([f"c = {c:g}", f"{np.corrcoef(pr, y)[0, 1]:.3f}", f"{np.sqrt(np.mean((pr - y) ** 2)):.4f}", f"{mloss:.3f}", f"{mauc:.3f}",
                     ", ".join(f"{m.kappa.item():+.2f}" for m in ms)])
    rep.section("2. Fits on all 514 DREAM pairs (training numbers, not a test): does the Ma loss go down, what does DREAM lose? "
                "(Ma loss at c = 0 uses T = 0.1; chance = 0.693 at m = 0)")
    rep.table(["Ma weight", "DREAM training r", "DREAM training RMSE", "Ma loss", "Ma AUROC (training)", "kappa per member"], rows)

    # 3. DREAM CV
    def ens(c):
        return lambda tr_, te_: np.mean([AttnMa(v, A, B, s, c).fit(tr_, y[tr_]).predict(te_) for v, s in zip(VM, s_all)], 0)
    REF, BASE = "O1.2b (reference, 27 numbers)", "O3s, DREAM only (c = 0; 67 numbers)"
    res = {REF: evaluate.cross_validate(P, lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM)
                                        .fit(tr_, y[tr_]).predict(te_), repeats=a_.repeats)}
    CNT = "O3 + counts: O1.2b rule on node-enriched word profiles (27)"
    res[CNT] = evaluate.cross_validate(P, lambda tr_, te_: O12b(Q_cnt, X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_), repeats=a_.repeats)
    rep.line(f"done: {CNT}")
    res[BASE] = evaluate.cross_validate(P, ens(0.0), repeats=a_.repeats); rep.line(f"done: {BASE}")
    for c in a_.c:
        k = f"O3s + Ma direction (c = {c:g})"; res[k] = evaluate.cross_validate(P, ens(c), repeats=a_.repeats); rep.line(f"done: {k}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"3. DREAM development pairs ({a_.repeats} x 5): RMSE / r; change [95% CI], positive r change = better")
    out = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if k == REF else res[REF])
            out.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                        "" if k == REF else ch(s_), ch(sc(r_, sub, res[BASE])) if k not in (REF, BASE) else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "minus O3s DREAM only (same folds)"], out)
    rep.section("Paired against O3 + counts (O1.2b rule on node-enriched words), same folds")
    out = [[k, sub, ch(sc(r_, sub, res[CNT]))] for k, r_ in res.items() if k not in (REF, CNT) for sub in SUB]
    rep.table(["model", "subset", "change from O3 + counts"], out)
    import pickle
    from mixdist.report import results_dir
    (results_dir() / "eval_o3s_attn_ma").mkdir(parents=True, exist_ok=True)
    with open(results_dir() / "eval_o3s_attn_ma" / f"{rep.time:%Y-%m-%d_%H%M%S}_cv.pkl", "wb") as fh:
        pickle.dump({"res": res, "k_P": k_P}, fh)
    rep.save()


if __name__ == "__main__":
    main()
