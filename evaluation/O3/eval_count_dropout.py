# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  M = DREAM molecules (235)   D = hidden size (64)   P = pairs
# ---------------------------------------------------------------------------------------------
"""Item 1 of "do all of them" (user 2026-11-20): counts with dropout on the count features (train_o3p att_cdrop05 /
att_cdrop08: O5 attention pooling + atom group-distance counts, each count zeroed with probability p in training).
Finding 6 sec. 22: counts memorise single molecules (gap 0.048 vs O5 0.025). Backbones (all frozen, GS-LF only):
  O5 (10)                      o3m_attpool               no counts
  O5 + counts (3)              ga_w64_att                counts, no dropout
  O5 + counts + POM 0.3 (3)    ga_w64_att_rank03         the other session's counts model
  O5 + counts, dropout 0.5 (3) ga_w64_att_cdrop05
  O5 + counts, dropout 0.8 (3) ga_w64_att_cdrop08
Part 1, 0 DREAM numbers (own attention weights, unit merge, angle), direct r on the 514 development pairs by shared-
molecule count and on T46 / V50; with and without centring (finding 6 sec. 20).
Part 2, (b) readout (69 numbers, fitted on all 820 training pairs): T46 and V50, RMSE and r vs O1.2b.

Env:     mist
Usage:   python evaluation/O3/eval_count_dropout.py
Writes:  results/eval_count_dropout/<date_time>.{txt,json}
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
from eval_size_robust import centred_units  # noqa: E402
from eval_zero_param import mix_dist  # noqa: E402
from train_o2d_dev import dev_pairs  # noqa: E402
import eval_context_dream as ecd  # noqa: E402
import eval_o5_softmax as eo  # noqa: E402
import train_groupatoms  # noqa: E402
import train_o3m  # noqa: E402

LAM = eo.LAM; REF = "O1.2b (reference, 27)"


def count_members(folder, dev, rep, Gg, rows_D, tr_s):
    out = []
    for s in (0, 1, 2):
        ck = torch.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", weights_only=False)
        n = train_o3m.load(ROOT / f"checkpoints/{folder}/member_{s}.pt", dev)
        with torch.no_grad():
            scale = n.latent(o2.encode_all(n, Gg, tr_s, dev))[0].norm(dim=1).mean().item()
        H, sc, mol, V = attpool_atoms(n, Gg, rows_D, dev); H, V = H / scale, V / scale
        cnt = np.bincount(mol); mx = np.full(len(cnt), -np.inf); np.maximum.at(mx, mol, sc)
        se = np.zeros(len(cnt)); np.add.at(se, mol, np.exp(sc - mx[mol])); strength = mx + np.log(se / cnt)
        s_atoms = sc / strength.std(); Xa = np.c_[H, np.ones(len(H))]; w = np.linalg.lstsq(Xa, s_atoms, rcond=None)[0]
        rep.line(f"{folder} member {s}: single held-out AUROC {ck.get('single_auroc', float('nan')):.4f}, "
                 f"gap {ck.get('train_auroc', float('nan')) - ck.get('single_auroc', float('nan')):+.4f}, strength SD {strength.std():.3f}")
        out.append({"V": V, "H": H, "mol": mol, "strength": strength, "s": s_atoms, "q0": w[:-1]})
    return out


def main():
    ecd.DIRECTION_ONLY = True; dev = pick_device(); rep = Report("eval_count_dropout", {"lam": LAM, "device": dev})
    Gg, tr, _, _ = train_groupatoms.graphs_with_groups(); rows_D = Gg.dream_rows(); tr_s = np.random.default_rng(0).choice(tr, 2000, replace=False)
    NETS = {"O5 (10)": eo.backbones(dev, rep)[1]["o5"]}
    for nm, f in [("O5 + counts (3)", "ga_w64_att"), ("O5 + counts + POM 0.3 (3)", "ga_w64_att_rank03"),
                  ("O5 + counts, dropout 0.5 (3)", "ga_w64_att_cdrop05"), ("O5 + counts, dropout 0.8 (3)", "ga_w64_att_cdrop08")]:
        NETS[nm] = count_members(f, dev, rep, Gg, rows_D, tr_s)
    op = features.load("openpom"); held = sorted(evaluate.held_out_sets(), key=lambda h: h.name != "T46")

    # part 1: 0 DREAM numbers
    P = dev_pairs(); y = evaluate.labels(P); A, B = membership(P, op.cids_M)
    k = np.array([len(set(p.a) & set(p.b)) for p in P]); gap = np.abs(A.sum(1) - B.sum(1))
    SUB = {"all": k >= 0, "share 0": k == 0, "share 0 same size": (k == 0) & (gap <= 1), "share 1": k == 1, "share 2": k == 2, "share >= 3": k >= 3}
    hm = [(h.name, membership(list(h.pairs), op.cids_M), h.y_P) for h in held]
    rep.section("0 DREAM numbers: r (direct, own attention weights); development pairs by shared-molecule count, then T46, V50")
    rep.line("n per subset: " + ", ".join(f"{s} {int(m.sum())}" for s, m in SUB.items())); rows = []
    for nm, ms in NETS.items():
        for centre in (False, True):
            def d_of(m, A_, B_):
                U = centred_units(m["V"]) if centre else m["V"] / np.linalg.norm(m["V"], axis=1, keepdims=True)
                return mix_dist(U, A_, B_, np.exp(m["strength"] - m["strength"].max()))
            d = np.mean([d_of(m, A, B) for m in ms], 0)
            vals = [metrics.pearson(d[mm], y[mm]) for mm in SUB.values()] + [metrics.pearson(np.mean([d_of(m, Ah, Bh) for m in ms], 0), yh) for _, (Ah, Bh), yh in hm]
            rows.append([nm, "centre" if centre else "none"] + [f"{v:+.3f}" for v in vals])
    rep.table(["backbone", "fix"] + list(SUB) + [h for h, _, _ in hm], rows)

    # part 2: (b) readout, 69 numbers, held-out
    ids = [str(int(c)) for c in op.cids_M]; q_op = mixture.quantile_per_word(op.probs_MxW)
    X, _ = piece_matrix(pp.merge_identical(pp.units_rdkit_groups({i: str(s) for i, s in zip(ids, op.smiles_M)})), ids)
    train = data.load_train()
    for hs in held:
        allp = list(train) + list(hs.pairs); n = len(train); tr_ = np.arange(n); te = np.arange(n, len(allp))
        y_all = np.r_[evaluate.labels(train), hs.y_P]; A2, B2 = membership(allp, op.cids_M)
        preds = {REF: O12b(q_op, X, A2, B2, True, True, lam=LAM).fit(tr_, y_all[tr_]).predict(te)}
        for nm, ms in NETS.items():
            P_ = np.array([eo.PriorAtom(m["V"], m["H"], m["mol"], A2, B2, m["s"], m["q0"], "shrink").fit(tr_, y_all[tr_]).predict(te) for m in ms])
            ok = np.isfinite(P_).all(1); preds[f"{nm}, (b) readout (69)"] = P_[ok].mean(0)
            if (~ok).any():
                rep.line(f"{hs.name} {nm}: {int((~ok).sum())} members with non-finite predictions dropped")
        yh = hs.y_P; ref = preds[REF]; rows = []
        rep.section(f"{hs.name}: {len(yh)} pairs, (b) readout fitted on all 820 training pairs")
        for nm, p in preds.items():
            s = metrics.score(p, yh, ceiling=hs.ceiling); c = ""
            if nm != REF:
                cr, _ = paired(p, ref, yh); c = f"{s['rmse'] - metrics.rmse(ref, yh):+.4f} / {s['r'] - metrics.pearson(ref, yh):+.3f} [{cr[0]:+.3f}, {cr[1]:+.3f}]"
            rows.append([nm, f"{s['rmse']:.4f}", f"{s['r']:.3f} [{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", c])
        rep.table(["model", "RMSE", "r [95% CI]", "change from O1.2b: RMSE / r [CI]"], rows)
        rep.value(hs.name, {nm: {"rmse": metrics.rmse(p, yh), "r": metrics.pearson(p, yh)} for nm, p in preds.items()})
    rep.save()


if __name__ == "__main__":
    main()
