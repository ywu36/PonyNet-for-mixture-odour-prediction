# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  N = GoodScents + Ma molecules   M = DREAM molecules (235)   D = hidden size (64)
#   Q = GoodScents pairs   T = Ma mixtures   P = DREAM development pairs (514)   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Context readout on the FROZEN width-64 backbone (encoder and word head never change; the user, 2026-11-19: molecules
should see the other molecules when mixed, without many parameters).

Mixture vector z(K) = sum_m w_m v_m / sum_m w_m over the molecules m of mixture K, with
    log w_m = q . s_m  +  kappa * cos(v_m, mean of the OTHER molecules' v in K)
s_m = v_m / (mean |v| over GoodScents + Ma molecules). q (64 numbers): how strongly a molecule pulls, read from its own
vector. kappa (1 number): > 0 a molecule like its company counts more; < 0 a molecule unlike its company stands out.
For a two-molecule mixture both molecules' company term is cos(v_A, v_B), so kappa cancels: Ma and GoodScents cannot
teach it; only DREAM (1-43 molecules) can, so kappa is fitted on DREAM training folds only.

q is learned per member on real two-molecule data with the losses that held up (findings/clues.md); encoder and head
frozen; Adam, full batch, 400 steps; 1e-3 |q|^2:
  GoodScents notes:  rank loss of head(z) / tau vs the blend's real notes (head linear: head(z) = sum w head(v) / sum w)
  GoodScents lean:   softplus(-s (cos(z, v_A) - cos(z, v_B)) / t1); s from real notes (train_o3t.scent_rel_labels)
  Ma who wins:       softplus(-s (cos(z, v_A) - cos(z, v_B)) / t2); s = sign(L)
Training: GoodScents official training pairs (20,000 sampled), Ma mixtures without a fold-0 molecule. Checks on data
never used: GoodScents official held-out pairs (both molecules unseen), Ma fold 0. Control: the same with notes, lean
labels and Ma signs permuted across training items.
DREAM (development pairs, 50 x 5, 4 subsets, RMSE and r; prediction = mean over the 3 members):
  d = sqrt(|u_A - u_B|^2 + b^2 (h_A - h_B)^2), u = z / |z|, h = log |z|, then a line. Fitted per fold: b + line (3),
  + kappa (4) where marked.

Env:     mist
Usage:   python evaluation/O3/eval_context_readout.py [--folder checkpoints/o3m_w64]
Writes:  results/eval_context_readout/<date_time>.{txt,json}
"""
import argparse
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
from mixdist.o12 import K0, O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
from train_o3t import scent_rel_labels  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
cos = torch.nn.functional.cosine_similarity


def learn_q(V_N, S, head_W, head_b, tr_q, ma_tr, rel_Q, shuffle=False, use_q=True, seed=0, steps=400):
    """-> q (D), temperatures. V_N frozen hidden vectors of the Store molecules (scaled: s = V / mean |V|)."""
    rng = np.random.default_rng(seed); V = torch.as_tensor(V_N, dtype=F64)
    q_lab = tr_q[~np.isnan(rel_Q[tr_q])]; q_all = rng.choice(tr_q, min(20000, len(tr_q)), replace=False)
    q_lab = rng.choice(q_lab, min(20000, len(q_lab)), replace=False)
    PY = torch.as_tensor(S.pair_Y[q_all], dtype=F64); lab = torch.as_tensor(rel_Q[q_lab], dtype=F64)
    ma_y = torch.as_tensor((S.ma_L[ma_tr] > 0).astype(float))
    if shuffle:
        PY = PY[torch.as_tensor(rng.permutation(len(PY)))]; lab = lab[torch.as_tensor(rng.permutation(len(lab)))]
        ma_y = ma_y[torch.as_tensor(rng.permutation(len(ma_y)))]
    Wh, bh = torch.as_tensor(head_W, dtype=F64), torch.as_tensor(head_b, dtype=F64); T = torch.as_tensor(S.targets)
    q = torch.zeros(V.shape[1], dtype=F64, requires_grad=use_q); lt = torch.zeros(3, dtype=F64, requires_grad=True)
    opt = torch.optim.Adam([q, lt] if use_q else [lt], lr=0.05)

    def mix(a, b):
        w = torch.sigmoid((V[a] - V[b]) @ q).unsqueeze(1); return w * V[a] + (1 - w) * V[b]

    def side(a, b, y, t):
        z = mix(a, b); d = cos(z, V[a], dim=1) - cos(z, V[b], dim=1)
        return torch.nn.functional.softplus(-(2 * y - 1) * d / torch.exp(t)).mean()
    for _ in range(steps):
        opt.zero_grad()
        z = mix(S.pair_a[q_all], S.pair_b[q_all])
        l_notes = o3.rank_loss((z @ Wh.T + bh) / torch.exp(lt[0]), PY, T).mean()
        loss = l_notes + side(S.pair_a[q_lab], S.pair_b[q_lab], lab, lt[1]) + side(S.ma_a[ma_tr], S.ma_b[ma_tr], ma_y, lt[2]) + 1e-3 * (q ** 2).sum()
        loss.backward(); opt.step()
    return q.detach(), lt.detach()


@torch.no_grad()
def checks(V_N, S, head_W, head_b, te_q, ma_te, rel_Q, q, lt):
    V = torch.as_tensor(V_N, dtype=F64); out = {}

    def score(a, b):
        w = torch.sigmoid((V[a] - V[b]) @ q).unsqueeze(1); z = w * V[a] + (1 - w) * V[b]
        return z, (cos(z, V[a], dim=1) - cos(z, V[b], dim=1)).numpy()
    k = te_q[~np.isnan(rel_Q[te_q])]
    out["lean_auroc"] = roc_auc_score(rel_Q[k], score(S.pair_a[k], S.pair_b[k])[1])
    z, _ = score(S.pair_a[te_q], S.pair_b[te_q])
    lg = z @ torch.as_tensor(head_W, dtype=F64).T + torch.as_tensor(head_b, dtype=F64)
    out["notes_loss"] = float(o3.rank_loss(lg / torch.exp(lt[0]), torch.as_tensor(S.pair_Y[te_q], dtype=F64), torch.as_tensor(S.targets)).mean())
    out["ma_auroc"] = roc_auc_score(S.ma_L[ma_te] > 0, score(S.ma_a[ma_te], S.ma_b[ma_te])[1])
    return out


class ContextReadout:
    """DREAM pair model on fixed molecule vectors; fit(tr, y) / predict(te) like O12b."""

    def __init__(self, V_MxD, A_PxM, B_PxM, logw_M, fit_kappa, lam=LAM):
        self.V = torch.as_tensor(V_MxD, dtype=F64); self.lw = torch.as_tensor(logw_M, dtype=F64)
        self.A, self.B = torch.as_tensor(A_PxM, dtype=F64), torch.as_tensor(B_PxM, dtype=F64)
        self.CA, self.CB = self._context(self.A), self._context(self.B); self.fit_kappa, self.lam = fit_kappa, lam

    def _context(self, Mx):
        """cos(v_m, mean of the other members) for each pair's mixture (0 for a lone molecule or a non-member)."""
        n = Mx.sum(1, keepdim=True); tot = Mx @ self.V                                          # P x D
        others = (tot.unsqueeze(1) - self.V.unsqueeze(0)) / (n - 1).clamp_min(1).unsqueeze(2)   # P x M x D
        c = cos(others, self.V.unsqueeze(0).expand_as(others), dim=2)
        return torch.where((Mx > 0) & (n > 1), c, torch.zeros_like(c))

    def distance(self, kappa, b, idx):
        def out(Mx, C):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(lw); z = (W @ self.V) / W.sum(1, keepdim=True); n = z.norm(dim=1, keepdim=True)
            return z / n, torch.log(n[:, 0])
        uA, hA = out(self.A, self.CA); uB, hB = out(self.B, self.CB)
        return torch.sqrt(((uA - uB) ** 2).sum(1) + b ** 2 * (hA - hB) ** 2 + 1e-12)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        kappa = torch.zeros((), dtype=F64, requires_grad=self.fit_kappa); b = torch.tensor(K0, dtype=F64, requires_grad=True)
        with torch.no_grad():
            d = self.distance(kappa, b, idx); D = torch.stack([d, torch.ones_like(d)], 1)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True); params = [lin, b] + ([kappa] if self.fit_kappa else [])
        opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); d = self.distance(kappa, b, idx)
            loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + self.lam * kappa ** 2; loss.backward(); return loss
        opt.step(closure); self.kappa, self.b, self.lin = kappa.detach(), b.detach(), lin.detach(); return self

    def predict(self, idx):
        with torch.no_grad():
            return (self.lin[0] * self.distance(self.kappa, self.b, torch.as_tensor(np.asarray(idx))) + self.lin[1]).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folder", default="checkpoints/o3m_w64"); a = ap.parse_args()
    dev = pick_device(); rep = Report("eval_context_readout", {"folder": a.folder, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z_ = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False); ok = np.where(z_["ok"].astype(bool))[0]
    rel_Q = scent_rel_labels(G, S); tr_q, te_q = np.where(S.pair_split == 0)[0], np.where(S.pair_split == 1)[0]
    ma_tr, ma_te = S.ma_split(0)
    nets = [train_o3m.load(f, dev) for f in sorted((ROOT / a.folder).glob("member_*.pt"))]
    rep.line(f"{len(nets)} frozen members ({a.folder}); GoodScents training pairs {len(tr_q)} (20,000 sampled per loss), "
             f"held-out {len(te_q)} (lean labels {int((~np.isnan(rel_Q[te_q])).sum())}); Ma training {len(ma_tr)}, fold 0 {len(ma_te)}")
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}

    VARIANTS = {"q = 0": dict(use_q=False), "q from GoodScents + Ma": dict(), "q from shuffled labels (control)": dict(shuffle=True)}
    member, rows = [], []
    for i, net in enumerate(nets):
        with torch.no_grad():
            V_N = np.zeros((len(z_["smiles"]), net.h)); V_N[ok] = net.latent(o2.encode_all(net, S, ok, dev))[0].double().cpu().numpy()
            V_M = net.latent(o2.encode_all(net, G, G.dream_rows(), dev))[0].double().cpu().numpy()
        scale = np.linalg.norm(V_N[ok], axis=1).mean(); V_N, V_M = V_N / scale, V_M / scale
        Wh, bh = net.head.weight.detach().double().cpu().numpy() * scale, net.head.bias.detach().double().cpu().numpy()
        qs = {}
        for name, kw in VARIANTS.items():
            q, lt = learn_q(V_N, S, Wh, bh, tr_q, ma_tr, rel_Q, seed=i, **kw); qs[name] = q.numpy()
            c = checks(V_N, S, Wh, bh, te_q, ma_te, rel_Q, q, lt)
            rows.append([i, name, f"{c['lean_auroc']:.3f}", f"{c['notes_loss']:.4f}", f"{c['ma_auroc']:.3f}", f"{np.abs(q.numpy()).max():.3f}"])
        member.append((V_M, qs))
        rep.line(f"member {i} done")
    rep.section("Data never used for q: GoodScents official held-out pairs (both molecules unseen), Ma fold 0")
    rep.table(["member", "q", "GoodScents lean-to AUROC (0.5 chance)", "GoodScents notes held-out rank loss (lower better)",
               "Ma who-wins AUROC (fold 0)", "max |q|"], rows)

    def ens(qname, fit_kappa):
        return lambda tr_, te_: np.mean([ContextReadout(V_M, A, B, V_M @ qs[qname], fit_kappa).fit(tr_, y[tr_]).predict(te_)
                                         for V_M, qs in member], 0)
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(
        P, lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_))}
    spec = [("plain mean (q = 0), 3 numbers", "q = 0", False), ("q = 0 + kappa, 4", "q = 0", True),
            ("q from GoodScents + Ma, 3", "q from GoodScents + Ma", False), ("q from GoodScents + Ma + kappa, 4", "q from GoodScents + Ma", True),
            ("q from shuffled labels (control), 3", "q from shuffled labels (control)", False)]
    for label, qn, fk in spec:
        res[label] = evaluate.cross_validate(P, ens(qn, fk)); rep.line(f"DREAM done: {label}")
    ref = res["O1.2b (reference, 27 numbers)"]
    comp = {"q = 0 + kappa, 4": "plain mean (q = 0), 3 numbers", "q from GoodScents + Ma, 3": "plain mean (q = 0), 3 numbers",
            "q from GoodScents + Ma + kappa, 4": "q from GoodScents + Ma, 3", "q from shuffled labels (control), 3": "plain mean (q = 0), 3 numbers"}

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    out = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            out.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                        "" if r_ is ref else ch(s_), f"vs {comp[k].split(',')[0]}: " + ch(sc(r_, sub, res[comp[k]])) if k in comp else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "paired step"], out)
    kap = [ContextReadout(V_M, A, B, V_M @ qs["q from GoodScents + Ma"], True).fit(np.arange(len(P)), y).kappa.item() for V_M, qs in member]
    rep.line(f"kappa fitted on all development pairs (q from GoodScents + Ma), per member: {', '.join(f'{k:+.3f}' for k in kap)} "
             "(> 0: like-company counts more; < 0: odd one out stands out)")
    rep.save()


if __name__ == "__main__":
    main()
