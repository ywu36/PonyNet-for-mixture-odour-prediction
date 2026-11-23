"""Attention pooling for mixtures: can a context-dependent molecule weight beat B1a++, in POM word space and in the
frozen O1.4 hidden cell layer (where every molecule activates every cell and per-cell normalization cancels)?

Context. POMMix/CheMix (external/pommix_repo/src/chemix/model.py) pools molecules with sigmoid self-attention and an
attention aggregation whose query is the mixture MEAN: a molecule's weight depends on how it relates to its company.
Our hand-made version (shared contrast) gave beta +0.15 [+0.04, +0.71]. CheMix has 1.6 M parameters; here each head is
low rank (few hundred numbers, penalised) on top of B1a++'s weights, so 820 pairs can fit it.
Recent attention ideas used:
  pommix   CheMix-style mean-query attention with a SIGMOID gate (no competition inside the score; Ramapuram et al.
           2024 sigmoid attention; CheMix uses it):  w_i = G_i * 2 sigma(s_i),  s_i = (U z_i).(U zbar) / sqrt(r)
  diff     DIFFERENTIAL attention (Ye et al., Differential Transformer, ICLR 2025): two maps, the second subtracted to
           cancel what all molecules share (common mode), leaving what is distinctive:
           a = softmax(log G + s1) - lam softmax(log G + s2),  lam = 0.8 sigmoid(l), s_k = (u_k.z_i)(u_k.zbar)
  sparse   SPARSE weights (sparsemax; Martins & Astudillo 2016; entmax family): w = sparsemax(log G / tau), tau learned.
           Exact zeros: only the dominant molecules count; addresses "everything active, everything averages".
G_i = exp(theta . groups_i) (B1a++), z = features z-scored over the 235 molecules, zbar = unweighted mixture mean.
At init every variant equals its base (s = 0 -> sigma = 1/2 x 2; diff lam small; sparse tau large -> uniform).
Spaces:
  words  values = open-POM quantile profiles q (B1a++).
  cells  values = frozen O1.4 dense cells (results/eval_o14/cells_q.pt, second session, loaded read-only), pooled in the
         cell layer, decoded to words with its frozen read-out; then B1a++'s length-1 distance + size line.
         Base "cellsbase" = G-weighted mean in cells, no attention.
  scells the SELECTIVE O1.4 cells (results/eval_o14/cells_q_sparse.pt: top 8 of 64 per molecule, overlap 0.12,
         held-out R^2 0.932; second session). Added after the second session noted the dense cells overlap 0.88 and
         decode near-linearly, so attention in dense cells may act like word space.
Fixed before running: standard CV (same folds as B1a++; B1a++ CV predictions from
results/eval_setting2_shared_fraction/b1a_pp_cv_pred.npz, same seed); change from B1a++ (and for cells variants
from cellsbase); primary subset all pairs; claim only if r-change CI > 0 and within-study CI > 0. Settings 1 and 2
reported. Penalty lam 1e-3 on theta and on the attention matrices; U init N(0, 0.05^2), seed 0.

Env:     mist
Usage:   python evaluation/B1aplus/eval_attention_mix.py {pommix|diff|sparse|cellsbase} {words|cells|scells} [cpu|cuda:0|cuda:1]
Writes:  results/eval_attention_mix_<variant>_<space>/<date_time>.{txt,json} (+ cv_pred.npz)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "evaluation/B1aplus"))
from mixdist import data, evaluate, features, mixture, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import B1aPlus, membership, size_features  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_setting2_step1 import within_change  # noqa: E402

torch.set_default_dtype(torch.float64)
LAM = 1e-3; N_BOOT = 2000; RANK = 2


def sparsemax(z, dim=1):
    zs, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.shape[dim] + 1, dtype=z.dtype, device=z.device).view(1, -1, *([1] * (z.dim() - 2)))
    cs = zs.cumsum(dim)
    support = (1 + k * zs) > cs
    kz = support.sum(dim, keepdim=True).clamp_min(1)
    tau = (cs.gather(dim, kz - 1) - 1) / kz
    return torch.clamp(z - tau, min=0)


class AttnMix(B1aPlus):
    """Weights in pair x molecule form (P x M, 0 for non-members); mixture = W @ V / sum W."""

    def __init__(self, q, x, A, B, S, lam, variant, Z, V, dec=None, device="cpu"):
        super().__init__(q, x, A, B, S, lam)
        self.dev = device
        for nm in ("q", "x", "A", "B", "S"):
            setattr(self, nm, getattr(self, nm).to(device))
        self.variant, self.dec = variant, (None if dec is None else dec.to(device))
        self.Z = torch.tensor(np.asarray(Z, float), device=device); self.V = torch.tensor(np.asarray(V, float), device=device)
        self.ZA = (self.A @ self.Z) / self.A.sum(1, keepdim=True); self.ZB = (self.B @ self.Z) / self.B.sum(1, keepdim=True)
        D = Z.shape[1]; g = torch.Generator().manual_seed(0)
        if variant == "pommix":
            self.U0 = 0.05 * torch.randn(RANK, D, generator=g); self.extra0 = torch.zeros(0)
        elif variant == "diff":
            self.U0 = 0.05 * torch.randn(2, D, generator=g); self.extra0 = torch.tensor([-3.0])   # lam = 0.8 sigmoid(-3) = 0.04
        elif variant == "sparse":
            self.U0 = torch.zeros(0, D); self.extra0 = torch.tensor([np.log(20.0)])                 # tau = 20: ~uniform
        else:
            self.U0 = torch.zeros(0, D); self.extra0 = torch.zeros(0)
        self.U0, self.extra0 = self.U0.to(device), self.extra0.to(device)

    def weights(self, theta, U, ex, Mx, Zbar):
        lg = (self.x @ theta)[None, :].expand(Mx.shape[0], -1)          # P x M   log G
        mem = Mx > 0; neg = torch.tensor(-1e4, device=self.dev)
        if self.variant == "pommix":
            s = (self.Z @ U.T)[None, :, :] * (Zbar @ U.T)[:, None, :]   # P x M x r
            w = torch.exp(lg) * 2 * torch.sigmoid(s.sum(-1) / np.sqrt(RANK))
            return torch.where(mem, w, torch.zeros((), device=self.dev))
        if self.variant == "diff":
            s = (self.Z @ U.T)[None, :, :] * (Zbar @ U.T)[:, None, :]   # P x M x 2
            lam = 0.8 * torch.sigmoid(ex[0])
            a1 = torch.softmax(torch.where(mem, lg + s[..., 0], neg), 1)
            a2 = torch.softmax(torch.where(mem, lg + s[..., 1], neg), 1)
            return torch.where(mem, a1 - lam * a2, torch.zeros((), device=self.dev))
        if self.variant == "sparse":
            tau = torch.exp(ex[0].clamp(np.log(0.01), np.log(100.0)))      # bounded: unbounded tau -> 0 overflowed to NaN in 3/250 fits
            return sparsemax(torch.where(mem, lg / tau, torch.tensor(-1e12, device=self.dev)))
        return torch.where(mem, torch.exp(lg), torch.zeros((), device=self.dev))

    def mix(self, theta, U, ex, Mx, Zbar):
        w = self.weights(theta, U, ex, Mx, Zbar)
        p = (w @ self.V) / w.sum(1, keepdim=True)
        if self.dec is not None:
            p = self.dec(p)
        return p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)

    def dist(self, theta, U, ex, idx):
        return (self.mix(theta, U, ex, self.A[idx], self.ZA[idx]) - self.mix(theta, U, ex, self.B[idx], self.ZB[idx])).norm(dim=1)

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev); y = torch.tensor(np.asarray(y, float), device=self.dev)
        theta = torch.zeros(self.x.shape[1], device=self.dev, requires_grad=True)
        U = self.U0.clone().requires_grad_(True); ex = self.extra0.clone().requires_grad_(True)
        with torch.no_grad():
            D = self._design(self.dist(theta, U, ex, idx), idx)
            lin0 = torch.linalg.lstsq(D.cpu(), y.cpu()[:, None]).solution[:, 0].to(self.dev)
        lin = lin0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([theta, U, ex, lin], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = self._design(self.dist(theta, U, ex, idx), idx) @ lin
            loss = ((pred - y) ** 2).mean() + self.lam * ((theta ** 2).sum() + (U ** 2).sum())
            loss.backward()
            return loss
        opt.step(closure)
        self.theta, self.U, self.ex, self.lin = theta.detach(), U.detach(), ex.detach(), lin.detach()
        return self

    def predict(self, idx):
        idx = torch.as_tensor(np.asarray(idx), device=self.dev)
        with torch.no_grad():
            return (self._design(self.dist(self.theta, self.U, self.ex, idx), idx) @ self.lin).cpu().numpy()


def b1a_pp_cv(train):
    """B1a++ CV result rebuilt from the saved predictions (same seed, folds and order as evaluate.cross_validate)."""
    f = np.load(ROOT / "results/eval_setting2_shared_fraction/b1a_pp_cv_pred.npz")
    pair_N, pred_N = f["pair_N"], f["pred_N"]
    y = evaluate.labels(train)
    return evaluate.CVResult(pair_N, pred_N, y[pair_N], np.array([train[i].dataset for i in pair_N]),
                             np.array([train[i].jaccard == 0 for i in pair_N]), 5, evaluate.CV_REPEATS, 0)


def main():
    variant, space = sys.argv[1], sys.argv[2]
    assert variant in ("pommix", "diff", "sparse", "cellsbase") and space in ("words", "cells", "scells")
    rep = Report(f"eval_attention_mix_{variant}_{space}", {"variant": variant, "space": space, "lam": LAM, "rank": RANK, "repeats": evaluate.CV_REPEATS, "n_boot": N_BOOT})
    train = data.load_train(); y = evaluate.labels(train)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    q = mixture.quantile_per_word(op.probs_MxW)
    A, B = membership(train, op.cids_M); S = size_features(train)
    Xg, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    dec = None
    if space == "words":
        V = q
    else:
        net = torch.load(ROOT / ("results/eval_o14/cells_q_sparse.pt" if space == "scells" else "results/eval_o14/cells_q.pt"), weights_only=False).double()
        with torch.no_grad():
            V = net.cells(torch.tensor(np.asarray(q, float))).numpy()
        dec = net.dec
        rep.line(f"O1.4 cells: {V.shape[1]} units; active (> 0.1) {(V > 0.1).mean():.0%}; read-out R^2 of a molecule's own words "
                 f"{1 - ((dec(torch.tensor(V)).detach().numpy() - q) ** 2).sum() / ((q - q.mean(0)) ** 2).sum():.3f}")
    Z = (V - V.mean(0)) / (V.std(0) + 1e-9)
    dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    make = lambda: AttnMix(q, Xg, A, B, S, LAM, variant, Z, V, dec, dev)  # noqa: E731

    base = b1a_pp_cv(train)
    res = evaluate.cross_validate(train, lambda tr, te: make().fit(tr, y[tr]).predict(te))
    assert np.array_equal(base.pair_N, res.pair_N)
    print("  done CV", flush=True)
    bad = ~np.isfinite(res.pred_N)
    rep.line(f"non-finite CV predictions: {bad.sum()} of {len(bad)}")
    if bad.any():
        raise SystemExit("non-finite predictions; not scored")
    out = ROOT / f"results/eval_attention_mix_{variant}_{space}"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "cv_pred.npz", pred_N=res.pred_N, pair_N=res.pair_N)
    refs = [("B1a++", base)]
    cb = ROOT / f"results/eval_attention_mix_cellsbase_{space}/cv_pred.npz"
    if space != "words" and variant != "cellsbase" and cb.exists():
        f = np.load(cb)
        refs.append(("cellsbase", evaluate.CVResult(f["pair_N"], f["pred_N"], y[f["pair_N"]], base.study_N, base.no_overlap_N, 5, evaluate.CV_REPEATS, 0)))
    s1 = base.no_overlap_N; ok = False
    for rname, ref in refs:
        rows = []
        for subset, mask in [("all pairs (primary)", np.ones(len(s1), bool)), ("setting 1", s1), ("setting 2", ~s1)]:
            s0 = evaluate.score_cv(train, ref, mask_N=mask, n_boot=N_BOOT)
            s = evaluate.score_cv(train, res, reference=ref, mask_N=mask, n_boot=N_BOOT)
            d0, (lo, hi), _ = within_change(train, ref, res, mask)
            if rname == "B1a++" and subset.startswith("all"):
                ok = s["r_change_ci"][0] > 0 and lo > 0
            rows.append([subset, s["pairs"], f"{s0['rmse']:.4f} / {s0['r']:.3f}", f"{s['rmse']:.4f} / {s['r']:.3f}",
                         f"{s['rmse_change']:+.4f} [{s['rmse_change_ci'][0]:+.4f}, {s['rmse_change_ci'][1]:+.4f}] / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]",
                         f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}]"])
        rep.section(f"{variant} in {space} space vs {rname} (same folds)")
        rep.table(["subset", "pairs", f"{rname} RMSE / r", "new RMSE / r", "change: RMSE [CI] / r [CI]", "within-study r change [CI]"], rows)
    rep.line(f"claim rule (all pairs, vs B1a++): {'MET' if ok else 'NOT met'}")

    m = make().fit(np.arange(len(train)), y)
    rep.section("Fitted on all 820 pairs")
    if variant == "sparse":
        w = m.weights(m.theta, m.U, m.ex, m.iA, m.mA).numpy(); nz = (w > 0).sum(1); n = m.mA.numpy().sum(1)
        rep.line(f"tau = {float(torch.exp(m.ex[0].clamp(np.log(0.01), np.log(100.0)))):.3f}; molecules with weight > 0: median {np.median(nz):.0f} of median {np.median(n):.0f}")
    if variant == "diff":
        rep.line(f"lambda = {float(0.8 * torch.sigmoid(m.ex[0])):.3f}")
    if variant in ("pommix", "diff"):
        rep.line(f"|U| = {float(m.U.norm()):.3f} (init {float(m.U0.norm()):.3f})")
        with torch.no_grad():
            w = m.weights(m.theta, m.U, m.ex, m.A, m.ZA).cpu().numpy(); lg = np.exp((m.x @ m.theta).cpu().numpy())[None, :] * np.ones_like(w)
        mk = m.A.cpu().numpy() > 0
        wn = w / w.sum(1, keepdims=True); gn = np.where(mk, lg, 0); gn = gn / gn.sum(1, keepdims=True)
        rep.line(f"attention moves the weights: mean |w - G| per molecule {np.abs(wn - gn)[mk].mean():.4f} (mean weight {wn[mk].mean():.4f})")
    rep.value("result", {"variant": variant, "space": space, "claim_met": bool(ok)})
    rep.save()


if __name__ == "__main__":
    main()
