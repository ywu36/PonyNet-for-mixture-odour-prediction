# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = DREAM development pairs
# ---------------------------------------------------------------------------------------------
"""Do molecules seeing each other help, when the attention query q is fitted on DREAM? Frozen width-64 backbone.

    share_m in mixture K  ~  exp( q . s_m  +  kappa * cos(v_m, mean of the OTHER molecules' v in K) )
    z(K) = sum share_m v_m;  d = sqrt(|u_A - u_B|^2 + b^2 (h_A - h_B)^2);  prediction = line(d)
Fitted per fold on DREAM training pairs (least squares on similarity, as O1.2b; 1e-3 |q|^2 + 1e-3 kappa^2):
    attention, self only:        q (64) + b + line             = 67 numbers   (kappa = 0)
    attention, sees the others:  q (64) + kappa + b + line     = 68 numbers
Paired, development pairs, 50 x 5, 4 subsets, RMSE and r; change from O1.2b; "sees the others" minus "self only".
--direction-only: no length weight (b fixed 0): two mixtures compared by the angle between z_A and z_B only
(d = |u_A - u_B| = sqrt(2 - 2 cos)); the line only maps that onto the rating scale (r does not depend on it).

Env:     mist
Usage:   python evaluation/O3/eval_context_dream.py [--folder checkpoints/o3m_w64]
Writes:  results/eval_context_dream/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, o2j, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import K0, O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_context_readout import ContextReadout  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import train_o3m  # noqa: E402

F64 = torch.float64
DIRECTION_ONLY = False


class ContextReadoutQ(ContextReadout):
    """ContextReadout with the attention query q fitted on DREAM too."""

    def fit(self, idx, y, max_iter=300):
        idx = torch.as_tensor(np.asarray(idx)); y = torch.as_tensor(np.asarray(y, float))
        q = torch.zeros(self.V.shape[1], dtype=F64, requires_grad=True)
        kappa = torch.zeros((), dtype=F64, requires_grad=self.fit_kappa)
        b = torch.tensor(0.0 if DIRECTION_ONLY else K0, dtype=F64, requires_grad=not DIRECTION_ONLY)
        with torch.no_grad():
            self.lw = self.V @ q; d = self.distance(kappa, b, idx); D = torch.stack([d, torch.ones_like(d)], 1)
            lin = torch.linalg.lstsq(D, y[:, None]).solution[:, 0].clone()
        lin.requires_grad_(True); params = [lin, q] + ([] if DIRECTION_ONLY else [b]) + ([kappa] if self.fit_kappa else [])
        opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); self.lw = self.V @ q; d = self.distance(kappa, b, idx)
            loss = ((lin[0] * d + lin[1] - y) ** 2).mean() + self.lam * ((q ** 2).sum() + kappa ** 2); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            self.lw = self.V @ q
        self.q, self.kappa, self.b, self.lin = q.detach(), kappa.detach(), b.detach(), lin.detach(); self.lw = self.lw.detach()
        return self


def main():
    global DIRECTION_ONLY
    ap = argparse.ArgumentParser(); ap.add_argument("--folder", default="checkpoints/o3m_w64")
    ap.add_argument("--direction-only", action="store_true"); a = ap.parse_args(); DIRECTION_ONLY = a.direction_only
    n_extra = 0 if DIRECTION_ONLY else 1
    dev = pick_device(); rep = Report("eval_context_dream", {"folder": a.folder, "direction_only": DIRECTION_ONLY, "lam": LAM,
                                                             "repeats": evaluate.CV_REPEATS, "device": dev}, tag="direction_only" if DIRECTION_ONLY else "")
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); S = o2j.Store(features.features_dir() / "o2j_data.npz")
    z_ = np.load(features.features_dir() / "o2j_data.npz", allow_pickle=False); ok = np.where(z_["ok"].astype(bool))[0]
    nets = [train_o3m.load(f, dev) for f in sorted((ROOT / a.folder).glob("member_*.pt"))]
    V_M = []
    for net in nets:
        with torch.no_grad():
            scale = net.latent(o2.encode_all(net, S, ok, dev))[0].norm(dim=1).mean().item()
            V_M.append(net.latent(o2.encode_all(net, G, G.dream_rows(), dev))[0].double().cpu().numpy() / scale)
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share no molecule": k_P == 0, "share 1-2 molecules": (k_P >= 1) & (k_P <= 2), "share >= 3 molecules": k_P >= 3}
    rep.line(f"{len(nets)} frozen members ({a.folder}); development pairs {len(P)}")

    def ens(fk):
        return lambda tr_, te_: np.mean([ContextReadoutQ(v, A, B, np.zeros(len(v)), fk).fit(tr_, y[tr_]).predict(te_) for v in V_M], 0)
    res = {"O1.2b (reference, 27 numbers)": evaluate.cross_validate(
        P, lambda tr_, te_: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(tr_, y[tr_]).predict(te_))}
    k_self, k_ctx = f"attention, self only ({66 + n_extra})", f"attention, sees the others ({67 + n_extra})"
    res[k_self] = evaluate.cross_validate(P, ens(False)); rep.line("done: self only")
    res[k_ctx] = evaluate.cross_validate(P, ens(True)); rep.line("done: sees the others")
    ref, base = res["O1.2b (reference, 27 numbers)"], res[k_self]

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
                        "" if r_ is ref else ch(s_), ch(sc(r_, sub, base)) if "sees the others" in k else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "sees the others minus self only"], out)
    kap = [ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(np.arange(len(P)), y).kappa.item() for v in V_M]
    rep.line(f"kappa fitted on all development pairs, per member: {', '.join(f'{k:+.3f}' for k in kap)} "
             "(> 0: a molecule like its company counts more; < 0: the odd one out stands out)")
    rep.save()


if __name__ == "__main__":
    main()
