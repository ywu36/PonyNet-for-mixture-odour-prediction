# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   X = DREAM development mixtures (unique)
#   P = development pairs (514)   H = middle layer (400)   W = words (138)   N = GoodScents molecules   K = members
# ---------------------------------------------------------------------------------------------
"""Does merging in the middle layer work as intended, and is it at least as good as merging after decoding (O1.2b)?
(the user, 2026-11-19). Frozen networks (default: O2-C-Single, 10 members); no training, nothing fitted on DREAM
beyond the stated numbers.

  in   z(A) = mean of the members' middle vectors v (O2-C: centred), decoded once: p_in(A) = sigmoid(readout(z))
  out  p_out(A) = mean of the members' decoded profiles (O1.2b's place of merging)
Why it can fail: the read-out is non-linear and was trained on single molecules only; a mean of different molecules is
shorter than any of them (rho = |z| / mean |v| < 1) and may lie where the read-out never saw an input.

(a) Distortion on DREAM development mixtures: rho; share of mixtures whose |z| is below the 1st percentile of single
    |v|; cos(p_in, p_out) on probabilities and on quantile profiles; by mixture size.
(b) DREAM development pairs, 50 x 5, group gain off, only beta + line fitted (3 numbers), d = sqrt(|uA - uB|^2 +
    beta (hA - hB)^2): out vs in, on probabilities and on quantile profiles (in: decoded mixture mapped with the
    quantile map fitted on the single molecules). Change from O1.2b (27 numbers) and paired in minus out.
(c) Real blends with known notes: GoodScents held-out pairs (both molecules unseen). Per member theta + tau fitted on
    30k training pairs for each form (out: logit(w p_A + (1 - w) p_B); in: readout(w v_A + (1 - w) v_B)); rank loss and
    word AUROC on held-out pairs, mean over members.

Env:     mist (GPU)
Usage:   python evaluation/O3/diag_middle_merge.py [--folder checkpoints/o2c]
Writes:  results/diag_middle_merge/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

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
from train_o2j import word_auroc  # noqa: E402
from train_o3e import dream_groups  # noqa: E402

F32 = torch.float32
BETAS = np.r_[0.0, np.logspace(-3, 2, 26)]


def three_numbers(PA, PB, y):
    """fit_predict with beta (grid) + line fitted on training pairs, from fixed mixture profiles PA, PB (P x W)."""
    uA, uB = PA / np.linalg.norm(PA, axis=1, keepdims=True), PB / np.linalg.norm(PB, axis=1, keepdims=True)
    du2 = ((uA - uB) ** 2).sum(1); dh2 = (np.log(np.linalg.norm(PA, axis=1)) - np.log(np.linalg.norm(PB, axis=1))) ** 2

    def fp(tr, te):
        best = None
        for b in BETAS:
            d = np.sqrt(du2 + b * dh2); a, c = np.polyfit(d[tr], y[tr], 1); e = ((a * d[tr] + c - y[tr]) ** 2).mean()
            if best is None or e < best[0]:
                best = (e, b, a, c)
        _, b, a, c = best
        return a * np.sqrt(du2[te] + b * dh2[te]) + c
    return fp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folder", default="checkpoints/o2c")
    ap.add_argument("--renorm", action="store_true", help="merged middle vector rescaled to the members' weighted mean length (direction averaged, length kept)")
    a = ap.parse_args()
    dev = pick_device(); rep = Report("diag_middle_merge", {"folder": a.folder, "renorm": a.renorm, "device": dev, "repeats": evaluate.CV_REPEATS},
                                       tag=Path(a.folder).name + ("_renorm" if a.renorm else ""))
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    Xd, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M)
    nets = o2.load_members(sorted((ROOT / a.folder).glob("member_*.pt")), dev); rows_M = G.dream_rows()
    mix_X, inv = np.unique(np.r_[A, B], axis=0, return_inverse=True); inv = inv.ravel(); iA, iB = inv[:len(P)], inv[len(P):]
    Mx = torch.as_tensor(mix_X, dtype=F32, device=dev); size_X = mix_X.sum(1)
    rep.line(f"members {len(nets)} from {a.folder}; development pairs {len(P)}, unique mixtures {len(mix_X)} (sizes {sorted(set(size_X.astype(int).tolist()))})")

    # ---- (a) distortion
    p_single, p_in, rho, oor = [], [], [], []
    with torch.no_grad():
        for n in nets:
            v = n.latent(o2.encode_all(n, G, rows_M, dev))[0]                         # M x H
            ps = torch.sigmoid(n.readout(v)[1]); p_single.append(ps)
            z = n.pool(v, Mx)
            if a.renorm:
                z = z / z.norm(dim=1, keepdim=True) * (Mx @ v.norm(dim=1) / Mx.sum(1)).unsqueeze(1)
            p_in.append(torch.sigmoid(n.readout(z)[1]))
            r = n.pool(v, Mx).norm(dim=1) / (Mx @ v.norm(dim=1) / Mx.sum(1)); rho.append(r)
            oor.append(z.norm(dim=1) < torch.quantile(v.norm(dim=1), 0.01))
    ps_M = torch.stack(p_single).mean(0).double().cpu().numpy(); pin_X = torch.stack(p_in).mean(0).double().cpu().numpy()
    pout_X = mix_X @ ps_M / size_X[:, None]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=100).fit(ps_M)
    q_M = qt.transform(ps_M); qout_X = mix_X @ q_M / size_X[:, None]; qin_X = qt.transform(pin_X)
    rho_X = torch.stack(rho).mean(0).cpu().numpy(); oor_X = torch.stack(oor).float().mean(0).cpu().numpy()
    cosf = lambda u, w: (u * w).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1)
    c_p, c_q = cosf(pin_X, pout_X), cosf(qin_X, qout_X)
    rows = []
    for lo, hi, nm in ((1, 1, "1"), (2, 10, "2-10"), (11, 30, "11-30"), (31, 99, "> 30")):
        m = (size_X >= lo) & (size_X <= hi)
        if m.any():
            rows.append([nm, int(m.sum()), f"{rho_X[m].mean():.3f}", f"{oor_X[m].mean():.2f}", f"{c_p[m].mean():.4f} (min {c_p[m].min():.4f})",
                         f"{c_q[m].mean():.4f} (min {c_q[m].min():.4f})", f"{np.linalg.norm(pin_X[m], axis=1).mean() / np.linalg.norm(pout_X[m], axis=1).mean():.3f}"])
    rep.section("(a) Distortion: middle-layer merge vs merge after decoding, DREAM development mixtures (no fitting)")
    rep.table(["mixture size", "mixtures", "rho = |z| / mean |v|", "share of members with |z| below single 1st percentile",
               "cos(p_in, p_out), probabilities", "cos, quantile profiles", "|p_in| / |p_out|"], rows)

    # ---- (b) DREAM, 3 numbers
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    forms = {"out, quantile profiles (O1.2b place, group gain off)": (qout_X, ), "in, quantile profiles": (qin_X, ),
             "out, probabilities": (pout_X, ), "in, probabilities": (pin_X, )}
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(P, lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), Xd, A, B, True, True, lam=LAM).fit(tr, y[tr]).predict(te))}
    for k, (PX,) in forms.items():
        res[k] = evaluate.cross_validate(P, three_numbers(PX[iA], PX[iB], y))
    ref = res["O1.2b (reference, 27 numbers)"]

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])
    rep.section(f"(b) DREAM development pairs ({evaluate.CV_REPEATS} x 5), beta + line only (3 numbers): change from O1.2b (paired)")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if r_ is ref else ref)
            ch = "" if r_ is ref else f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]", ch])
    rep.table(["merge", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [95% CI]"], rows)
    rows = []
    for hi, lo in (("in, quantile profiles", "out, quantile profiles (O1.2b place, group gain off)"), ("in, probabilities", "out, probabilities")):
        for sub in SUB:
            s_ = sc(res[hi], sub, res[lo])
            rows.append([f"{hi}  minus  {lo}", sub, f"{s_['rmse_change']:+.4f}", f"{s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"])
    rep.table(["comparison (same folds)", "subset", "RMSE change", "r change [95% CI]"], rows)

    # ---- (c) GoodScents held-out blends
    z_ = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False); gn = dream_groups()
    XN = torch.as_tensor(o3.group_matrix(z_["smiles"], gn), dtype=F32, device=dev)
    tr_q, te_q = np.where(S.pair_split == 0)[0], np.where(S.pair_split == 1)[0]
    tr_q = tr_q[np.random.default_rng(0).permutation(len(tr_q))[:30000]]
    mols = np.unique(np.r_[S.pair_a[np.r_[tr_q, te_q]], S.pair_b[np.r_[tr_q, te_q]]])
    PY = torch.as_tensor(S.pair_Y, dtype=F32, device=dev); T = torch.as_tensor(S.targets, device=dev)
    out = {"out": [], "in": []}
    for n in nets:
        with torch.no_grad():
            V = torch.zeros(int(mols.max()) + 1, 400, dtype=F32, device=dev)
            for i in range(0, len(mols), 512):
                x, e, s, d, g = S.batch(mols[i:i + 512], dev); V[mols[i:i + 512]] = n.latent(n.encode(x, e, s, d, g, int(g.max()) + 1))[0]
            PN = torch.sigmoid(n.readout(V)[1])
        for form in ("out", "in"):
            def logits(q, th, lt):
                a_, b_ = S.pair_a[q], S.pair_b[q]; w = torch.sigmoid((XN[a_] - XN[b_]) @ th).unsqueeze(1)
                if form == "out":
                    lg = torch.logit((w * PN[a_] + (1 - w) * PN[b_]).clamp(1e-6, 1 - 1e-6))
                else:
                    zz = w * V[a_] + (1 - w) * V[b_]
                    if a.renorm:
                        zz = zz / zz.norm(dim=1, keepdim=True) * (w * V[a_].norm(dim=1, keepdim=True) + (1 - w) * V[b_].norm(dim=1, keepdim=True))
                    lg = n.readout(zz)[1]
                return lg / torch.exp(lt)
            th = torch.zeros(len(gn), dtype=F32, device=dev, requires_grad=True); lt = torch.zeros((), dtype=F32, device=dev, requires_grad=True)
            opt = torch.optim.Adam([th, lt], lr=0.05); rng = np.random.default_rng(0)
            for _ in range(300):
                q = rng.choice(tr_q, 4096, replace=False)
                loss = o3.rank_loss(logits(q, th, lt), PY[q], T).mean() + 1e-3 * (th ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                lg = logits(te_q, th, lt); l_ = o3.rank_loss(lg, PY[te_q], T).mean().item()
            out[form].append((l_, word_auroc(lg.cpu().numpy(), S.pair_Y[te_q], S.targets)[0]))
    rep.section(f"(c) GoodScents held-out blends ({len(te_q)} pairs, both molecules unseen): theta + tau fitted per form, mean over {len(nets)} members")
    o_, i_ = np.array(out["out"]), np.array(out["in"])
    rep.table(["merge", "rank loss (mean, sd over members)", "word AUROC"],
              [["out (after decoding)", f"{o_[:, 0].mean():.4f} ({o_[:, 0].std():.4f})", f"{o_[:, 1].mean():.4f}"],
               ["in (middle layer)", f"{i_[:, 0].mean():.4f} ({i_[:, 0].std():.4f})", f"{i_[:, 1].mean():.4f}"],
               ["in minus out, per member", f"{(i_[:, 0] - o_[:, 0]).mean():+.4f} (range {(i_[:, 0] - o_[:, 0]).min():+.4f} to {(i_[:, 0] - o_[:, 0]).max():+.4f})",
                f"{(i_[:, 1] - o_[:, 1]).mean():+.4f}"]])
    rep.save()


if __name__ == "__main__":
    main()
