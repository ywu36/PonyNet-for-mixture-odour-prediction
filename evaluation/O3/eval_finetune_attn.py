# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = DREAM development pairs
#   B = POM-dataset molecules in a batch   E = open-POM map (256)   W = words (138)
# ---------------------------------------------------------------------------------------------
"""One unified model (user, 2026-11-19: "we need a unified model"; no routing by shared molecules): the counts +
POM-order backbone (checkpoints/ga_w64_rank, trained on POM-dataset single molecules: word loss + POM-order loss) with
the hidden-layer attention readout, FINE-TUNED on DREAM while the POM-dataset losses keep the middle layer in place.

Per CV fold, per member (seeds 0-2), all settings fixed before running:
  1. frozen: attention + kappa fitted on the fold's DREAM training pairs (as eval_context_counts.py; 67 numbers).
  2. fine-tune from there, 200 Adam steps, BatchNorm statistics fixed, no dropout:
       loss = W_DREAM * MSE(DREAM training pairs)                       (attention readout, angle only)
            + word loss on a batch of 128 POM-dataset training molecules (open-POM's loss; the strict readout)
            + POM-order loss on the same batch (hidden layer keeps open-POM's distance order; train_o3p.order_loss)
            + L2SP * sum (backbone weights - start)^2  +  1e-3 (|q|^2 + kappa^2)
       backbone lr 1e-4, readout (q, kappa, line) lr 1e-3.
  3. control: the same with the fold's DREAM training ratings shuffled (must not help).
Scored: development pairs, 10 x 5 folds (fine-tuning per fold; 50 x 5 too slow), RMSE and r, subsets all / setting 1 /
setting 2 / share 1 / share 2 / share >= 3; change from O1.2b and from the frozen readout on the same folds.
Also per fold: POM-dataset held-out single-molecule AUROC after fine-tuning (does the backbone stay a singles model?).

Env:     mist (GPU)
Usage:   python evaluation/O3/eval_finetune_attn.py [--repeats 10] [--smoke]
Writes:  results/eval_finetune_attn/<date_time>.{txt,json}
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import evaluate, features, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from eval_o2n import LAM  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import train_groupatoms  # noqa: E402
from train_o3p import order_loss, pom_targets  # noqa: E402

SEEDS = (0, 1, 2)
W_DREAM, L2SP, STEPS, LR_NET, LR_READ, BATCH = 100.0, 1e-2, 200, 1e-4, 1e-3, 128
REF, FROZEN, FT, SHUF = "O1.2b (reference, 27)", "frozen + attention (67)", "fine-tuned + attention (67)", "fine-tuned, DREAM ratings shuffled (control)"


def attn_pred(S_MxD, q, kappa, lin, A, B, idx):
    """Angle-only attention readout (as eval_context_readout.ContextReadout, b = 0), differentiable in S."""
    def side(Mx):
        Mx = Mx[idx]; n = Mx.sum(1, keepdim=True); tot = Mx @ S_MxD
        others = (tot.unsqueeze(1) - S_MxD.unsqueeze(0)) / (n - 1).clamp_min(1).unsqueeze(2)
        C = torch.nn.functional.cosine_similarity(others, S_MxD.unsqueeze(0).expand_as(others), dim=2)
        C = torch.where((Mx > 0) & (n > 1), C, torch.zeros_like(C))
        lw = (S_MxD @ q).unsqueeze(0) + kappa * C; lw = lw - lw.max(1, keepdim=True).values
        W = Mx * torch.exp(lw); z = (W @ S_MxD) / W.sum(1, keepdim=True)
        return torch.nn.functional.normalize(z, dim=1)
    d = torch.sqrt(((side(A) - side(B)) ** 2).sum(1) + 1e-12)
    return lin[0] * d + lin[1]


class Unified:
    def __init__(self, nets, G, gs_tr, gs_te, Pemb, A, B, dev):
        self.nets, self.G, self.gs_tr, self.gs_te, self.dev = nets, G, gs_tr, gs_te, dev
        self.Pemb = torch.as_tensor(Pemb, device=dev); self.Y = torch.as_tensor(G.Y, device=dev)
        cnt = G.Y[gs_tr].sum(0); self.cw = torch.as_tensor(np.log1p(cnt / cnt.max()), dtype=torch.float32, device=dev)
        self.batch_D = G.batch(G.dream_rows(), dev); self.n_D = len(G.dream_rows())
        self.A = torch.as_tensor(A, dtype=torch.float32, device=dev); self.B = torch.as_tensor(B, dtype=torch.float32, device=dev)
        self.A64, self.B64 = A, B
        tr_s = np.random.default_rng(0).choice(gs_tr, 2000, replace=False); self.scale = []
        for n in nets:
            with torch.no_grad():
                self.scale.append(n.latent(o2.encode_all(n, G, tr_s, dev))[0].norm(dim=1).mean().item())
        self.auroc_after = []

    def S(self, net, k):
        return net.latent(net.encode(*self.batch_D, self.n_D))[0] / self.scale[k]

    def fold(self, tr, te, y_tr, finetune, seed):
        out = []
        for k, net0 in enumerate(self.nets):
            with torch.no_grad():
                S0 = self.S(net0, k).double().cpu().numpy()
            fr = ecd.ContextReadoutQ(S0, self.A64, self.B64, np.zeros(len(S0)), True).fit(tr, y_tr)
            if not finetune:
                out.append(fr.predict(te)); continue
            net = copy.deepcopy(net0).train(); start = [p.detach().clone() for p in net.parameters()]
            for mod in net.modules():   # train mode only for the GRU's backward; BatchNorm statistics fixed, no dropout
                if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.Dropout)):
                    mod.eval()
            q = fr.q.float().to(self.dev).requires_grad_(True); kap = fr.kappa.float().to(self.dev).requires_grad_(True)
            lin = fr.lin.float().to(self.dev).requires_grad_(True)
            opt = torch.optim.Adam([{"params": list(net.parameters()), "lr": LR_NET}, {"params": [q, kap, lin], "lr": LR_READ}])
            rng = np.random.default_rng(seed * 100 + k); idx = torch.as_tensor(tr, device=self.dev)
            yt = torch.as_tensor(y_tr, dtype=torch.float32, device=self.dev)
            for _ in range(STEPS):
                b = rng.choice(self.gs_tr, BATCH, replace=False)
                v, logits = net(self.G.batch(b, self.dev))
                loss = (W_DREAM * ((attn_pred(self.S(net, k), q, kap, lin, self.A, self.B, idx) - yt) ** 2).mean()
                        + o2.pom_loss(logits, self.Y[b], self.cw) + order_loss(v, self.Pemb[b])
                        + L2SP * sum(((p - s) ** 2).sum() for p, s in zip(net.parameters(), start))
                        + 1e-3 * ((q ** 2).sum() + kap ** 2))
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                out.append(attn_pred(self.S(net, k), q, kap, lin, self.A, self.B, torch.as_tensor(te, device=self.dev)).double().cpu().numpy())
                self.auroc_after.append(o2.mean_auroc(o2.predict_probs(net, self.G, self.gs_te, self.dev), self.G.Y[self.gs_te]))
        return np.mean(out, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=10); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); reps = 1 if a.smoke else a.repeats; ecd.DIRECTION_ONLY = True
    dev = pick_device(); torch.manual_seed(0)
    rep = Report("eval_finetune_attn", {"repeats": reps, "W_DREAM": W_DREAM, "L2SP": L2SP, "steps": STEPS, "lr_net": LR_NET,
                                         "lr_read": LR_READ, "batch": BATCH, "seeds": SEEDS, "device": dev}, tag="smoke" if a.smoke else "")
    G, gs_tr, gs_te, _ = train_groupatoms.graphs_with_groups(); Pemb = pom_targets(G)
    nets = [train_groupatoms.load(ROOT / f"checkpoints/ga_w64_rank/member_{s}.pt", dev) for s in SEEDS]
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "setting 1 (share none)": k_P == 0, "setting 2 (share >= 1)": k_P >= 1,
           "share 1": k_P == 1, "share 2": k_P == 2, "share >= 3": k_P >= 3}
    U = Unified(nets, G, gs_tr, gs_te, Pemb, A, B, dev)
    rep.line(f"backbone checkpoints/ga_w64_rank (counts + POM order), members {SEEDS}; device {dev}; folds {reps} x 5; "
             f"POM-dataset training molecules {len(gs_tr)}, held-out {len(gs_te)}")
    fold_no = [0]

    def ft(shuffle):
        def fp(tr, te):
            fold_no[0] += 1; y_tr = y[tr].copy()
            if shuffle:
                y_tr = np.random.default_rng(fold_no[0]).permutation(y_tr)
            return U.fold(tr, te, y_tr, True, fold_no[0])
        return fp
    res = {REF: evaluate.cross_validate(P, lambda tr, te: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM)
                                        .fit(tr, y[tr]).predict(te), repeats=reps)}
    res[FROZEN] = evaluate.cross_validate(P, lambda tr, te: U.fold(tr, te, y[tr], False, 0), repeats=reps); rep.line("done: frozen")
    res[FT] = evaluate.cross_validate(P, ft(False), repeats=reps); auc_ft = list(U.auroc_after); rep.line("done: fine-tuned")
    U.auroc_after = []
    res[SHUF] = evaluate.cross_validate(P, ft(True), repeats=reps); auc_sh = list(U.auroc_after); rep.line("done: shuffled control")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({reps} x 5): RMSE / r; change [95% CI], positive r change = better")
    rows = []
    for sub in SUB:
        for k, r_ in res.items():
            s_ = sc(r_, sub, None if k == REF else res[REF])
            rows.append([k, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if k == REF else ch(s_), ch(sc(r_, sub, res[FROZEN])) if k in (FT, SHUF) else ""])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from frozen (same folds)"], rows)
    base = [n_ for n_ in (torch.load(ROOT / f"checkpoints/ga_w64_rank/member_{s}.pt", map_location="cpu", weights_only=False)["single_auroc"] for s in SEEDS)]
    rep.line(f"POM-dataset held-out single AUROC: before {np.mean(base):.4f}; after fine-tuning {np.mean(auc_ft):.4f} "
             f"[min {np.min(auc_ft):.4f}]; after shuffled fine-tuning {np.mean(auc_sh):.4f}")
    rep.save()


if __name__ == "__main__":
    main()
