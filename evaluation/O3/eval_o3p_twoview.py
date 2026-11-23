# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = development pairs (514)   M = DREAM molecules (235)   H = hidden layer (64)
# ---------------------------------------------------------------------------------------------
"""Make the two pieces compatible inside ONE attention readout (user, 2026-11-19: "attention must work; make it
different by adding another attention layer or something").
Diagnosis (scratchpad why.py): the POM-order loss pulls every molecule toward one shared direction (mean cosine to the
average direction 0.22 -> 0.39); good for placing different molecules (share none), bad for telling apart mixtures that
share most molecules (share >= 3). Attention decides WHO counts (needs strength); POM order decides WHERE molecules
sit (needs placement). So split the two jobs:
  two-view attention   shares from the counts network (q . s_m + kappa cos(s_m, others), s = counts hidden vector),
                       template z = sum share_m p_m with p = the counts + POM-order hidden vector (paired by seed).
  centred              p minus its average over GS-LF training molecules (no DREAM label): removes the shared direction
                       POM order adds, keeping its ordering of differences (the face-recognition mean-subtraction).
All rows direction only (angle between mixtures), q 64 + kappa + line = 67 numbers, fitted on DREAM training folds;
networks frozen and trained on single molecules only. 50 x 5; 4 subsets; RMSE and r; change from O1.2b and from
counts + attention (the current best on share >= 3).

Env:     mist
Usage:   python evaluation/O3/eval_o3p_twoview.py
Writes:  results/eval_o3p_twoview/<date_time>.{txt,json}
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
import train_groupatoms  # noqa: E402

SEEDS = (0, 1, 2)
REF = "O1.2b (reference, 27)"
BASE = "counts: attention (67)"


class TwoView(ecd.ContextReadoutQ):
    """Shares (q, kappa) read from V (the weighting network); the mixture template is built from Vp (placement)."""

    def __init__(self, V_MxH, Vp_MxH, A, B):
        super().__init__(V_MxH, A, B, np.zeros(len(V_MxH)), True); self.Vp = torch.as_tensor(Vp_MxH, dtype=ecd.F64)

    def distance(self, kappa, b, idx):
        def out(Mx, C):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(lw); z = (W @ self.Vp) / W.sum(1, keepdim=True)
            return z / z.norm(dim=1, keepdim=True)
        return torch.sqrt(((out(self.A, self.CA) - out(self.B, self.CB)) ** 2).sum(1) + 1e-12)   # direction only


class BothPlaces(TwoView):
    """One attention (shares from V, as TwoView) and the mixture compared in BOTH spaces at once: angle distance in the
    counts space and in the POM-order space, added in quadrature with equal weight (no extra number): what averaging the
    two models' predictions did, inside one 67-number readout."""

    def distance(self, kappa, b, idx):
        def out(Mx, C, Vx):
            lw = self.lw.unsqueeze(0) + kappa * C[idx]; lw = lw - lw.max(1, keepdim=True).values
            W = Mx[idx] * torch.exp(lw); z = (W @ Vx) / W.sum(1, keepdim=True)
            return z / z.norm(dim=1, keepdim=True)
        d1 = ((out(self.A, self.CA, self.V) - out(self.B, self.CB, self.V)) ** 2).sum(1)
        d2 = ((out(self.A, self.CA, self.Vp) - out(self.B, self.CB, self.Vp)) ** 2).sum(1)
        return torch.sqrt(d1 + d2 + 1e-12)


def main():
    ecd.DIRECTION_ONLY = True
    dev = pick_device(); rep = Report("eval_o3p_twoview", {"seeds": SEEDS, "lam": LAM, "repeats": evaluate.CV_REPEATS, "device": dev})
    G, tr, te, names = train_groupatoms.graphs_with_groups(); rows_D = G.dream_rows()
    tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False); V, Vc = {}, {}
    for k, folder in (("counts", "ga_w64"), ("pom", "ga_w64_rank")):
        V[k], Vc[k] = [], []
        for s in SEEDS:
            n = train_groupatoms.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
            with torch.no_grad():
                Ts = n.latent(o2.encode_all(n, G, tr_s, dev))[0].double().cpu().numpy()
                D = n.latent(o2.encode_all(n, G, rows_D, dev))[0].double().cpu().numpy()
            V[k].append(D / np.linalg.norm(Ts, axis=1).mean())
            mu = Ts.mean(0); Vc[k].append((D - mu) / np.linalg.norm(Ts - mu, axis=1).mean())
    cosm = lambda M: float(((M / np.linalg.norm(M, axis=1, keepdims=True)) @ (lambda u: u / np.linalg.norm(u))((M / np.linalg.norm(M, axis=1, keepdims=True)).mean(0))).mean())  # noqa: E731
    rep.line("mean cosine to the average direction (DREAM molecules): " + ", ".join(
        f"{k} {np.mean([cosm(m) for m in V[k]]):.3f} / centred {np.mean([cosm(m) for m in Vc[k]]):.3f}" for k in V))
    P = dev_pairs(); y = evaluate.labels(P); op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    A, B = membership(P, op.cids_M); assert list(G.dream_cids) == [int(c) for c in op.cids_M]
    k_P = np.array([len(set(p.a) & set(p.b)) for p in P])
    SUB = {"all pairs": None, "share none": k_P == 0, "share 1-2": (k_P >= 1) & (k_P <= 2), "share >= 3": k_P >= 3}

    def att(Vs):
        return lambda a, b: np.mean([ecd.ContextReadoutQ(v, A, B, np.zeros(len(v)), True).fit(a, y[a]).predict(b) for v in Vs], 0)

    def two(Vw, Vp):
        return lambda a, b: np.mean([TwoView(w, p, A, B).fit(a, y[a]).predict(b) for w, p in zip(Vw, Vp)], 0)
    def both(Vw, Vp):
        return lambda a, b: np.mean([BothPlaces(w, p, A, B).fit(a, y[a]).predict(b) for w, p in zip(Vw, Vp)], 0)
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--both-only", action="store_true"); only = ap.parse_args().both_only
    rows_m = {REF: lambda a, b: O12b(mixture.quantile_per_word(op.probs_MxW), X, A, B, True, True, lam=LAM).fit(a, y[a]).predict(b),
              BASE: att(V["counts"]),
              "counts + POM order: attention (67)": att(V["pom"]),
              "counts + POM order, centred: attention (67)": att(Vc["pom"]),
              "two-view: shares from counts, places from counts + POM order (67)": two(V["counts"], V["pom"]),
              "two-view: shares from counts, places from counts + POM order centred (67)": two(V["counts"], Vc["pom"]),
              "both places: shares from counts, compared in counts AND POM-order spaces (67)": both(V["counts"], V["pom"]),
              "both places, POM-order space centred (67)": both(V["counts"], Vc["pom"])}
    if only:
        rows_m = {k: v for k, v in rows_m.items() if k in (REF, BASE) or k.startswith("both")}
    res = {}
    for n, f in rows_m.items():
        res[n] = evaluate.cross_validate(P, f); rep.line(f"done: {n}")

    def sc(r_, sub, against):
        m = SUB[sub]
        return evaluate.score_cv(P, r_, reference=against, mask_N=None if m is None else m[r_.pair_N])

    def ch(s_):
        return f"{s_['rmse_change']:+.4f} / {s_['r_change']:+.3f} [{s_['r_change_ci'][0]:+.3f}, {s_['r_change_ci'][1]:+.3f}]"
    rep.section(f"DREAM development pairs ({evaluate.CV_REPEATS} x 5): RMSE / r change [95% CI]; positive r change = better")
    rows = []
    for sub in SUB:
        for n, r_ in res.items():
            s_ = sc(r_, sub, None if n == REF else res[REF])
            rows.append([n, sub, s_["pairs"], f"{s_['rmse']:.4f}", f"{s_['r']:.3f} [{s_['r_ci'][0]:.3f}, {s_['r_ci'][1]:.3f}]",
                         "" if n == REF else ch(s_), "" if n in (REF, BASE) else ch(sc(r_, sub, res[BASE]))])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from O1.2b", "change from counts: attention"], rows)
    rep.save()


if __name__ == "__main__":
    main()
