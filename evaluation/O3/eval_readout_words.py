# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
#   W = words (138)
# ---------------------------------------------------------------------------------------------
"""Same frozen network, different use (user, 2026-11-20: "not use the attention vector directly for distance ... but the
merged attention vector after passing the readout to be the new word predictions, and use that for final comparison").
  attention, hidden angle (67): attention weights the molecules' hidden vectors, mixture = weighted mean of hidden
      vectors, distance = angle between the two mixture vectors (current plain O3 + attention).
  attention, through the word head (67): same attention weights and weighted mean in the hidden layer; the mixture's
      hidden vector is passed through the network's linear word head -> the mixture's 138 word logits (head is linear, so
      this equals the weighted mean of its molecules' word logits); each word put on an equal scale (minus its mean,
      divided by its spread over 2000 GS-LF training molecules: no DREAM data); distance = angle between the two
      mixtures' 138-word vectors. 0 extra numbers.
  + length (68) (user, 2026-11-20: "we still need to use both"): distance = sqrt(direction difference^2 + length weight x
      (log length A - log length B)^2), as in O1.2b; length = sqrt(sum of squares) of the mixture vector.
  soft normalization (68) (user, 2026-11-20: "if things cancel out ... the direction may still keep the identity, but we
      want it deleted"): mixture vector -> vector / (softness + its length), softness fitted (1 number; O1.2's form);
      distance = difference of the two. A long vector (clear smell) ~ its direction; a short one (molecules cancelled)
      stays short, so its leftover direction barely counts. Direction-only rescales a cancelled leftover to full size.
Networks: plain O3 (o3m_w64) and O4 (checkpoints/o4, first 3 members), frozen, single molecules only. Earlier run of the
word-head version on the counts network only: eval_o3p_space (row 5). One model for all pairs; subsets for reporting only.

Env:     mist
Usage:   python evaluation/O3/eval_readout_words.py
Writes:  results/eval_readout_words/<date_time>.{txt,json}
"""
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
from eval_o3p_twoview import TwoView  # noqa: E402
import train_o3m  # noqa: E402

SEEDS = (0, 1, 2)
REF = "O1.2b (reference, 27)"
NETS = {"plain O3": "o3m_w64", "O4": "o4"}
RO = ("attention, hidden angle (67)", "attention, through the word head (67)",
      "attention, through the word head + length (68)", "attention, hidden, soft normalization (68)",
      "attention, through the word head, soft normalization (68)")


def fitted_extra(cls):
    """Same readout with its extra number b switched on: ContextReadoutQ fits b when DIRECTION_ONLY is False."""
    class E(cls):
        def fit(self, idx, y, max_iter=300):
            old = ecd.DIRECTION_ONLY; ecd.DIRECTION_ONLY = False
            try:
                return super().fit(idx, y, max_iter)
            finally:
                ecd.DIRECTION_ONLY = old
    return E


def _mix(self, Mx, C, idx, Vx):
    lw = self.lw.unsqueeze(0) + self.kappa_now * C[idx]; lw = lw - lw.max(1, keepdim=True).values
    W = Mx[idx] * torch.exp(lw); return (W @ Vx) / W.sum(1, keepdim=True)


class WordsLength(TwoView):
    """Shares from hidden, mixture in the scaled word space; direction + length (length weight = b^2)."""

    def distance(self, kappa, b, idx):
        self.kappa_now = kappa
        zA, zB = _mix(self, self.A, self.CA, idx, self.Vp), _mix(self, self.B, self.CB, idx, self.Vp)
        nA, nB = zA.norm(dim=1, keepdim=True), zB.norm(dim=1, keepdim=True)
        return torch.sqrt(((zA / nA - zB / nB) ** 2).sum(1) + b ** 2 * (torch.log(nA[:, 0]) - torch.log(nB[:, 0])) ** 2 + 1e-12)


class Soft(TwoView):
    """Shares from V, mixture built from Vp; soft normalization z / (softness + |z|), softness = b^2."""

    def distance(self, kappa, b, idx):
        self.kappa_now = kappa
        zA, zB = _mix(self, self.A, self.CA, idx, self.Vp), _mix(self, self.B, self.CB, idx, self.Vp)
        yA = zA / (b ** 2 + zA.norm(dim=1, keepdim=True)); yB = zB / (b ** 2 + zB.norm(dim=1, keepdim=True))
        return torch.sqrt(((yA - yB) ** 2).sum(1) + 1e-12)


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_readout_words", {"nets": NETS, "seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G = o2.Graphs(features.features_dir() / "o2_graphs.npz"); gslf = np.arange(G.n_gslf); tr = gslf[~G.audit_test]
    rows_D = G.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); V, L = {}, {}
    for k, folder in NETS.items():
        V[k], L[k] = [], []
        for s in SEEDS:
            n = train_o3m.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
            with torch.no_grad():
                Ts = n.latent(o2.encode_all(n, G, tr_s, dev))[0].double().cpu().numpy()
                D = n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy()
            Wt = n.head.weight.detach().double().cpu().numpy(); bt = n.head.bias.detach().double().cpu().numpy()
            lt, lD = Ts @ Wt.T + bt, D @ Wt.T + bt
            V[k].append(D / np.linalg.norm(Ts, axis=1).mean()); L[k].append((lD - lt.mean(0)) / lt.std(0).clip(1e-6))
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(np.array([len(p.a) - len(p.b) for p in P]))
    SUB = {"all pairs": None, "share none": k_P == 0, "share none, same size (gap <= 1)": (k_P == 0) & (gap <= 1),
           "share none, sizes differ (gap >= 2)": (k_P == 0) & (gap >= 2), "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}
    res = {REF: evaluate.cross_validate(
        P, lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b))}
    for k in NETS:
        res[f"{k}: {RO[0]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in V[k]], 0))
        res[f"{k}: {RO[1]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([TwoView(v, l, A, B).fit(a, y[a]).predict(b) for v, l in zip(V[k], L[k])], 0))
        WLn, Sf = fitted_extra(WordsLength), fitted_extra(Soft)
        res[f"{k}: {RO[2]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([WLn(v, l, A, B).fit(a, y[a]).predict(b) for v, l in zip(V[k], L[k])], 0))
        res[f"{k}: {RO[3]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([Sf(v, v, A, B).fit(a, y[a]).predict(b) for v in V[k]], 0))
        res[f"{k}: {RO[4]}"] = evaluate.cross_validate(
            P, lambda a, b, k=k: np.mean([Sf(v, l, A, B).fit(a, y[a]).predict(b) for v, l in zip(V[k], L[k])], 0))
        rep.line(f"done: {k}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for n, r_ in res.items():
            s_ = sc(r_, sub, None if n == REF else res[REF]); base = None
            if n != REF and not n.endswith(RO[0]):
                base = res[n.split(": ", 1)[0] + ": " + RO[0]]
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if n == REF else ch(s_), "" if base is None else ch(sc(r_, sub, base))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from hidden angle (same network)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
