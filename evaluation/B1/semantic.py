# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns), e.g. probs_MxW, dist_P
#   M = molecules with SMILES                          235
#   W = odour words (probability columns)              MIST 135, open-POM 138
#   E = embedding size (columns of emb)                MIST 512, open-POM 256
#   P = pairs in the set being processed               train 820, V50 50, T46 46
# ---------------------------------------------------------------------------------------------
"""B1 semantic baseline (PLAN.md step 3), shared by the eval_semantic_*.py scripts. Not a script.

No training. Per molecule: odour-word probabilities from a frozen pretrained model. Mixture: mean
over its molecules. Distance: cosine distance between the two mixture profiles. Calibration: line
y = a*d + c fitted on training pairs, one line for all studies or one per group (rule 5, see
mixdist.evaluate.CALIBRATIONS).

B1a: ``distance="paper"`` swaps the cosine distance for the paper's open-POM baseline recipe
(mixdist.mixture.paper_distance: quantile per word, length 1, Euclidean); the rest is unchanged.
``inputs="emb"`` uses the model's embedding (MIST: first-token embedding, 512-d) in place of the
odour-word probabilities; with ``distance="paper"`` each embedding dimension is quantile-transformed.
B1b: B1a with ``calibration="overlap"``: one line for pairs whose mixtures share no molecule and
one for the rest, so the level of "unrelated mixtures" comes from training pairs of that kind.

Steps of ``run``: 1. embeddings (features/<model>.npz, computed by evaluation/embed_<model>.py in
its own env if missing); 2. cross-validation on the training set; 3. with --final only, V50 and
T46, scored once.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import data, evaluate, features, metrics, mixture  # noqa: E402
from mixdist.report import Report  # noqa: E402


def parse_args(doc: str, final: bool = True) -> argparse.Namespace:
    """The options every B1 script has: --recompute-embeddings, and --final where it applies."""
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    if final:
        ap.add_argument("--final", action="store_true", help="score V50 and T46 (do this once)")
    ap.add_argument("--recompute-embeddings", action="store_true", help="rerun the pretrained model first")
    return ap.parse_args()


DISTANCES = {"cosine": mixture.mean_cosine_distance, "paper": mixture.paper_distance}


def molecule_inputs(fs, inputs: str) -> np.ndarray:
    """probs_MxW (``inputs="probs"``) or emb_MxE (``inputs="emb"``) of the cached features."""
    return {"probs": fs.probs_MxW, "emb": fs.emb_MxE}[inputs]


def load(model: str, rep: Report, recompute: bool = False, distance: str = "cosine", inputs: str = "probs"):
    """Section 1: the model's cached features -> (FeatureSet, training pairs, dist_P, y_P)."""
    rep.section("1. Embeddings")
    fs = features.ensure(model, recompute=recompute, log=rep.line)
    rep.line(fs.describe())
    rep.line(f"file {features.path(model)} (computed {features.computed_at(model)})")
    rep.line(f"per molecule: {inputs} ({molecule_inputs(fs, inputs).shape[1]} columns)")
    train = data.load_train()
    return fs, train, DISTANCES[distance](train, fs.cids_M, molecule_inputs(fs, inputs)), evaluate.labels(train)


def cross_validate(train, dist_P, y_P, calibration: str) -> evaluate.CVResult:
    group_P = evaluate.calibration_groups(train, calibration)
    return evaluate.cross_validate(train, evaluate.linear_calibration(dist_P, y_P, group_P))


def score_held_out(rep: Report, fs, train, dist_P, y_P, calibration: str, distance: str = "cosine",
                   inputs: str = "probs") -> None:
    """Section 3: fit each held-out set's line on all training pairs of its group, score V50 and T46."""
    rep.section("3. Held-out sets, scored once with the calibration fitted on all training pairs")
    earlier = rep.previous_runs(final=True)
    if earlier:
        rep.line(f"NOTE: --final was already run at {', '.join(earlier)}")
    group_P = evaluate.calibration_groups(train, calibration)
    rows, lines, preds = [], {}, {}
    for hs in evaluate.held_out_sets():
        group_H = evaluate.calibration_groups(hs.pairs, calibration)
        dist_H = DISTANCES[distance](hs.pairs, fs.cids_M, molecule_inputs(fs, inputs))
        pred_P, lines[hs.name] = np.empty(len(hs.pairs)), []
        for g in np.unique(group_H):
            a, c = evaluate.fit_line(dist_P[group_P == g], y_P[group_P == g])
            studies = sorted({p.dataset for p, gp in zip(train, group_P) if gp == g})
            rep.line(f"{hs.name}: {(group_H == g).sum()} pairs use the line fitted on {(group_P == g).sum()} "
                     f"training pairs ({', '.join(studies)}): y = {a:.3f} * d + {c:.3f}")
            lines[hs.name].append({"group": int(g), "a": a, "c": c, "studies": studies})
            pred_P[group_H == g] = a * dist_H[group_H == g] + c
        preds[hs.name] = pred_P.tolist()                      # in the order of hs.pairs
        rows.append((hs.name, evaluate.score_held_out(hs, pred_P)))
        rmse_med, r_med = metrics.bootstrap_medians(pred_P, hs.y_P)
        rep.line(f"{hs.name}: bootstrap medians (as SI Table S4) RMSE {rmse_med:.3f}, r {r_med:.3f}")
    rep.value("calibration", lines)
    rep.value("predictions", preds)
    rep.scores(rows, label="set")
    rep.line("reference = published POM predictions (paired bootstrap); "
             + "; ".join(f"{hs.name}: {hs.note}" for hs in evaluate.held_out_sets() if hs.note))


def run(script: str, model: str, doc: str, calibration: str = "shared", distance: str = "cosine",
        inputs: str = "probs") -> None:
    """One B1 experiment end to end: a frozen ``model``, one ``calibration`` scheme, one ``distance``,
    per molecule its odour-word probabilities or its embedding (``inputs``)."""
    args = parse_args(doc)
    rep = Report(script, {"model": model, "calibration": calibration, "distance": distance, "inputs": inputs, **vars(args)},
                 tag="final" * args.final)
    fs, train, dist_P, y_P = load(model, rep, args.recompute_embeddings, distance, inputs)
    result = cross_validate(train, dist_P, y_P, calibration)
    rep.section(f"2. Cross-validation on the training set ({result.describe()})")
    rep.scores(result.score_rows(), label="subset")
    if args.final:
        score_held_out(rep, fs, train, dist_P, y_P, calibration, distance, inputs)
    else:
        rep.line()
        rep.line("V50 and T46 not scored (pass --final).")
    rep.save()
