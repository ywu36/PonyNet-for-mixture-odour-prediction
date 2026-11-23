"""Morgan fingerprints as a frozen, training-free mixture representation vs B1a open-POM.

Fixed before running (no selection among them; all reported):
  Morgan B1a        count fingerprint radius 2 (2048 folded bins) per molecule -> B1a steps
                    (per-bin quantile over molecules, mean per mixture, length 1, Euclidean)
  overlap r1/r2/r3  set of fragment bins present in any molecule of the mixture (radius 1, 2, 3 ->
                    1, 2, 3 bonds around each atom) -> distance 1 - |A and B| / |A or B| (Tanimoto)
Each distance -> one line (2 numbers). Standard CV (evaluation/CV.md), all pairs and pairs sharing
no molecule, change from B1a open-POM on the same folds; then V50 and T46 scored once.

Env:     mist
Usage:   python evaluation/B1/eval_morgan_b1a.py
Writes:  results/eval_morgan_b1a/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import data, evaluate, features, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402

N_BITS = 2048


def morgan(smiles_M, radius, counts):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    out = np.zeros((len(smiles_M), N_BITS))
    for i, s in enumerate(smiles_M):
        fp = AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(s), radius, nBits=N_BITS)
        for k, v in fp.GetNonzeroElements().items():
            out[i, k] = v if counts else 1
    return out


def tanimoto_distance(pairs, cids_M, bits_MxB):
    row = {int(c): i for i, c in enumerate(cids_M)}
    def union(mix):
        return bits_MxB[[row[c] for c in mix]].max(0) > 0
    A = np.stack([union(p.a) for p in pairs]); B = np.stack([union(p.b) for p in pairs])
    return 1 - (A & B).sum(1) / np.maximum((A | B).sum(1), 1)


def main():
    rep = Report("eval_morgan_b1a", {"n_bits": N_BITS, "repeats": evaluate.CV_REPEATS})
    train = data.load_train()
    y_P = evaluate.labels(train)
    zero = np.zeros(len(train))
    op = features.load("openpom")
    cids, smiles = op.cids_M, [str(s) for s in op.smiles_M]
    cnt2 = morgan(smiles, 2, True); cnt2 = cnt2[:, cnt2.any(0)]
    bits = {r: morgan(smiles, r, False) for r in (1, 2, 3)}
    dist = {"B1a open-POM words": lambda P: mixture.paper_distance(P, cids, op.probs_MxW),
            "Morgan B1a (r2 counts)": lambda P: mixture.paper_distance(P, cids, cnt2),
            **{f"fragment overlap r{r}": (lambda P, r=r: tanimoto_distance(P, cids, bits[r])) for r in (1, 2, 3)}}
    rep.line(f"Morgan r2 count bins used by our 235 molecules: {cnt2.shape[1]}; fragments per molecule (r2) median {int(np.median((bits[2] > 0).sum(1)))}")

    rep.section(f"1. Standard CV ({evaluate.CV_REPEATS} x 5, every pair once, CI by resampling mixtures)")
    res = {k: evaluate.cross_validate(train, evaluate.linear_calibration(f(train), y_P, zero)) for k, f in dist.items()}
    ref = res["B1a open-POM words"]
    rows = []
    for subset, m in [("all pairs", None), ("share no molecule", ref.no_overlap_N)]:
        for k in dist:
            s = evaluate.score_cv(train, res[k], reference=None if k == "B1a open-POM words" else ref, mask_N=m)
            rows.append([k, subset, s["pairs"], f"{s['rmse']:.3f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         "" if k == "B1a open-POM words" else f"{s['rmse_change']:+.3f} / {s['r_change']:+.3f} [{s['r_change_ci'][0]:+.3f}, {s['r_change_ci'][1]:+.3f}]"])
    rep.table(["model", "subset", "pairs", "RMSE", "r [95% CI]", "change from B1a open-POM: RMSE / r [95% CI]"], rows)

    rep.section("2. V50 and T46: line fitted on all 820 training pairs, scored once")
    rows = []
    for hs in evaluate.held_out_sets():
        for k, f in dist.items():
            a, b = evaluate.fit_line(f(train), y_P)
            s = evaluate.score_held_out(hs, a * f(hs.pairs) + b)
            rows.append([hs.name, k, s["n"], f"{s['rmse']:.3f}", f"{s['r']:.3f} [{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]",
                         f"{s['r_ref']:.3f}", f"[{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]"])
    rep.table(["set", "model", "pairs", "RMSE", "r [95% CI]", "published POM r", "r - POM r (95% CI)"], rows)
    rep.save()


if __name__ == "__main__":
    main()
