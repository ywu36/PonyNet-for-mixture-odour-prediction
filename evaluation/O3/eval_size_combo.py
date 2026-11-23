# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs   T = atoms
# ---------------------------------------------------------------------------------------------
"""Step 3 (user 2026-11-20 "you can do 1-3"): size loss on both complementary models, then their average.
  (b)            O5 (o3m_attpool, 10 members) + atom query shrunk toward O5's own attention (69; eval_o5_softmax.PriorAtom)
  counts model   O5 + counts + POM order 0.3 (ga_w64_att_rank03, 3 members, the other session's network), same readout
  size loss      penalty 1.0 x var(y) x corr(distance, mean log size)^2 in the DREAM fit (eval_size_robust.SizeRobust)
  average        mean of the two models' predicted ratings (no new numbers; finding 7 sec. 8: best CV, 0.459)
Rows (fixed before running): O1.2b | (b) | (b) + size loss | counts model | counts model + size loss |
  average of (b) and counts model | average of both with size loss.
Held-out only (T46 then V50); CV of the winner later.

Env:     mist
Usage:   python evaluation/O3/eval_size_combo.py
Writes:  results/eval_size_combo/<date_time>_heldout.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "evaluation/B1aplus", ROOT / "evaluation/O1", ROOT / "evaluation/O2", ROOT / "evaluation/O3", ROOT / "training"):
    sys.path.insert(0, str(p))
from mixdist import data, evaluate, features, metrics, mixture, o2, pushpull as pp  # noqa: E402
from mixdist.b1a_plus import membership  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.o12 import O12b  # noqa: E402
from mixdist.report import Report  # noqa: E402
from eval_b1a_plus import piece_matrix  # noqa: E402
from final_heldout import paired  # noqa: E402
from eval_attpool import attpool_atoms  # noqa: E402
from eval_size_robust import SizeRobust  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

LAM = 1e-3
REF = "O1.2b (reference, 27)"


def counts_members(dev, rep):
    Gg, tr, _, _ = train_groupatoms.graphs_with_groups(); rows_D = Gg.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    out = []
    for s in (0, 1, 2):
        n = train_o3m.load(ROOT / f"checkpoints/ga_w64_att_rank03/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, Gg, tr_s, dev))[0].norm(dim=1).mean().item()
        H, sc, mol, V = attpool_atoms(n, Gg, rows_D, dev); H, V = H / scale, V / scale
        cnt = np.bincount(mol); mx = np.full(len(cnt), -np.inf); np.maximum.at(mx, mol, sc)
        se = np.zeros(len(cnt)); np.add.at(se, mol, np.exp(sc - mx[mol])); strength = mx + np.log(se / cnt)
        s_atoms = sc / strength.std(); Xa = np.c_[H, np.ones(len(H))]; w = np.linalg.lstsq(Xa, s_atoms, rcond=None)[0]
        rep.line(f"counts model member {s}: strength SD {strength.std():.3f}")
        out.append({"V": V, "H": H, "mol": mol, "strength": strength, "s": s_atoms, "q0": w[:-1]})
    return out


def main():
    ecd.DIRECTION_ONLY = True; dev = pick_device()
    rep = Report("eval_size_combo", {"lam": LAM, "device": dev}, tag="heldout")
    O5 = eo.backbones(dev, rep)[1]["o5"]; CM = counts_members(dev, rep)
    op = features.load("openpom"); ids = [str(int(c)) for c in op.cids_M]; q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    train = data.load_train()
    for hs in sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46"):
        allp = list(train) + list(hs.pairs); n = len(train); tr = np.arange(n); te = np.arange(n, len(allp))
        y_all = np.r_[evaluate.labels(train), hs.y_P]; A, B = membership(allp, op.cids_M)

        def ens(ms, size_loss):
            P_ = np.array([SizeRobust(m["V"], m["H"], m["mol"], A, B, m["s"], m["q0"], False, False, size_loss).fit(tr, y_all[tr]).predict(te) for m in ms])
            bad = ~np.isfinite(P_).all(1); rep.line(f"{hs.name}: members with non-finite predictions dropped: {int(bad.sum())} of {len(ms)}"); return P_[~bad].mean(0)
        preds = {REF: O12b(q_op, X, A, B, True, True, lam=LAM).fit(tr, y_all[tr]).predict(te),
                 "(b) (69)": ens(O5, False), "(b) + size loss": ens(O5, True),
                 "counts model (69)": ens(CM, False), "counts model + size loss": ens(CM, True)}
        preds["average of (b) and counts model"] = (preds["(b) (69)"] + preds["counts model (69)"]) / 2
        preds["average of both, with size loss"] = (preds["(b) + size loss"] + preds["counts model + size loss"]) / 2
        y = hs.y_P; ref = preds[REF]; rows = []
        rep.section(f"{hs.name}: {len(y)} pairs (all share no molecule)" + (f"; noise ceiling r {hs.ceiling:.3f}" if hs.ceiling else ""))
        for nm, p in preds.items():
            s = metrics.score(p, y, ceiling=hs.ceiling); c = ""
            if nm != REF:
                cr, _ = paired(p, ref, y); c = f"{s['rmse'] - metrics.rmse(ref, y):+.4f} / {s['r'] - metrics.pearson(ref, y):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
            rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{p.mean() - y.mean():+.3f}", c])
        rep.table(["model", "RMSE", "r [95% CI]", "bias", "change from O1.2b: RMSE / r [CI]"], rows)
        rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, y), "r": metrics.pearson(p, y)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
