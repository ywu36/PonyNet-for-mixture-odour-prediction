# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes
#   P = training pairs (820)      N = cross-validation predictions, all training pairs pooled
#   H = pairs of one held-out set (T46 46, V50 50)
# ---------------------------------------------------------------------------------------------
"""B2-ft: MIST fine-tuned on mixture distances (MIST's own mixture recipe), next to frozen MIST (B1).

The model (mixdist/mist_mixture.py) fine-tunes the MIST encoder with a learned weighted sum per
mixture and a scaled cosine distance, as MIST's MixtureSimilarityModel does. Two starting encoders:
``base`` (pretrained on SMILES only, as MIST's config) and ``odour`` (the odour model B1 uses frozen).
Versions:
  1  MIST's recipe as configured (head starts near 0, loss 1 - R^2, no line); retrained on all pairs.
  2  B1a's normalisation built in (fixed z-score per CLS dimension, mixtures scaled to length 1,
     Euclidean); stage A: encoder frozen, weighted sum and head trained (head started at the label
     mean, squared error); stage B: then the encoder unfrozen at a 10 x lower learning rate. Each stage
     is the early-stopped model, reported raw and with a line fitted on its inner validation pairs.
     B1a readouts: each encoder's molecule outputs (before training, after stage A, after stage B)
     through B1a's recipe and one line on the fold's training pairs.
Compared on the same pairs with one baseline, B1 MIST (frozen odour model, mean of odour-word
probabilities, cosine, one line); every other row is its change from B1 MIST:
  cross-validation   the 820 training pairs, whole mixtures held out (mixdist.cv), 5 folds x 3
                     repeats, the same folds for every row; pooled over all training pairs
  T46, V50           trained on all 820 training pairs, each scored once (--final)
Also shown: B1a MIST, and the released POMMix checkpoints (POM graph network pretrained on single
molecules, then fine-tuned end to end on its own 865 mixture pairs) from the latest
evaluation/check_pommix.py --final run; POMMix has no score on our folds. Saved rows of other versions
are shown too.

Training runs once per variant and version and is saved under checkpoints/mist_mixture/<variant>/
(version 1) or checkpoints/mist_mixture/v<version>/<variant>/ (cross-validation predictions, final model
and its held-out predictions); later runs load them. Without --final, random numbers stand in for the
held-out labels (dry run). --cv-only skips the final models and the held-out sets.

Env:     mist (torch, transformers, smirk, rdkit); one GPU per variant with 4 GB free
Usage:   python evaluation/B2ft/eval_mist_mixture.py --cv-only  version 2, cross-validation only
         python evaluation/B2ft/eval_mist_mixture.py            also the final models, dry-run held-out scores
         python evaluation/B2ft/eval_mist_mixture.py --final    score T46 and V50 (do this once)
         --version 1|2, --variants odour base, --retrain (train again even if checkpoints exist)
Writes:  checkpoints/mist_mixture/[v2/]<variant>/{cv.json,final.pt,held_out.json},
         results/eval_mist_mixture/<date_time>[_final].{txt,json}
"""
import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mixdist import data, evaluate, features, metrics, mixture  # noqa: E402
from mixdist.report import Report, results_dir  # noqa: E402

K, REPEATS, SEED = 5, 3, 0
STAGES = "AB"
CHECKPOINTS = REPO / "checkpoints/mist_mixture"
BASELINE = "B1 MIST (frozen odour model, cosine, line)"
LABEL = {"odour": "B2-ft MIST from the odour model", "base": "B2-ft MIST from the SMILES-pretrained encoder"}
STAGE_NAME = {"A": "stage A head only", "B": "stage B joint"}
B1A = "B1a MIST (quantile, length 1, Euclidean, line)"
POMMIX = "POMMix, released checkpoints (mean of 5)"


def folder_of(variant: str, version: int) -> Path:
    return CHECKPOINTS / variant if version == 1 else CHECKPOINTS / f"v{version}" / variant


def train_variant(variant: str, device: str, retrain: bool, version: int, cv_only: bool) -> dict:
    """In its own process: cross-validation and the final model of one variant, trained or loaded."""
    import torch
    from mixdist import mist_mixture as mm
    folder = folder_of(variant, version)
    folder.mkdir(parents=True, exist_ok=True)
    cv_file, held_file = folder / "cv.json", folder / "held_out.json"
    train, held = data.load_train(), {"T46": data.load_test(), "V50": data.load_validation()}
    if retrain or not cv_file.exists():
        infos, t = [], time.time()
        for f in ("train_log.tsv", "fit_summary.tsv"):         # live logs of this training (version 2)
            (folder / f).unlink(missing_ok=True)
        fp = mm.fit_predict(variant, train, device, SEED, infos) if version == 1 else \
            mm.fit_predict_v2(variant, train, device, SEED, infos, STAGES, k=K, log_dir=folder)
        res = evaluate.cross_validate(train, fp, k=K, repeats=REPEATS, seed=SEED)
        cv_file.write_text(json.dumps({"pair_N": res.pair_N.tolist(), "pred_N": res.pred_N.tolist(), "fits": infos,
                                       "minutes": (time.time() - t) / 60, "version": version,
                                       "settings": mm.SETTINGS if version == 1 else mm.SETTINGS_V2}))
    if not cv_only and (retrain or not held_file.exists()):
        t = time.time()                                       # labels of the held-out sets are not read here
        if version == 1:
            model, tokens, info = mm.train(variant, train, device, SEED)
            torch.save(model.state_dict(), folder / "final.pt")
            preds = {hs: mm.predict(model, tokens, pairs, device).tolist() for hs, pairs in held.items()}
            out = {"predictions": preds, "fit": info}
        else:
            out = {"stages": mm.train_v2(variant, train, device, SEED, STAGES, held["T46"] + held["V50"])}
        held_file.write_text(json.dumps({**out, "minutes": (time.time() - t) / 60}))
    return {"variant": variant, "device": device, "cv": json.loads(cv_file.read_text()),
            "held_out": json.loads(held_file.read_text()) if held_file.exists() else None}


def run_variants(variants, retrain: bool, version: int, cv_only: bool) -> dict:
    """One process per variant, each on its own GPU."""
    import torch
    free = [i for i in range(torch.cuda.device_count()) if torch.cuda.mem_get_info(i)[0] > 4 * 2**30]
    assert len(free) >= 1, "no GPU with 4 GB free"
    jobs = [(v, f"cuda:{free[i % len(free)]}", retrain, version, cv_only) for i, v in enumerate(variants)]
    with multiprocessing.get_context("spawn").Pool(min(len(jobs), len(free))) as pool:
        return {r["variant"]: r for r in pool.starmap(train_variant, jobs)}


def load_saved(variant: str, version: int) -> dict | None:
    f = folder_of(variant, version) / "cv.json"
    if not f.exists():
        return None
    held = folder_of(variant, version) / "held_out.json"
    return {"variant": variant, "cv": json.loads(f.read_text()),
            "held_out": json.loads(held.read_text()) if held.exists() else None}


def rows_of(r: dict, version: int, n_held: dict) -> list[dict]:
    """Rows of one trained variant: label, cross-validation predictions, held-out predictions (or None),
    and notes (head, epochs, minutes)."""
    cv, held, label = r["cv"], r["held_out"], LABEL[r["variant"]]
    if version == 1:
        fits = cv["fits"]
        return [{"label": f"{label}, v1 raw", "pred_N": np.array(cv["pred_N"]),
                 "held": {hs: np.array(p) for hs, p in held["predictions"].items()} if held else None,
                 "fits": fits, "minutes": cv["minutes"]}]

    def split(pred):                                          # T46 then V50, as passed to train_v2
        return {"T46": np.array(pred[:n_held["T46"]]), "V50": np.array(pred[n_held["T46"]:])}
    out = []
    for stage in STAGES:
        fits = [f[stage] for f in cv["fits"]]
        raw_N = np.concatenate([f["pred"] for f in fits])
        line_N = np.concatenate([f["line"][0] * np.array(f["pred"]) + f["line"][1] for f in fits])
        assert np.array_equal(raw_N, np.array(cv["pred_N"])) or stage != STAGES[-1]
        h = held["stages"][stage] if held else None
        minutes = sum(f["minutes"] for f in fits)
        base = f"{label}, v2 {STAGE_NAME[stage]}"
        out.append({"label": base + " raw", "pred_N": raw_N, "held": split(h["pred"]) if h else None,
                    "fits": fits, "minutes": minutes})
        out.append({"label": base + " + line", "pred_N": line_N, "fits": fits, "minutes": minutes,
                    "held": {hs: h["line"][0] * p + h["line"][1] for hs, p in split(h["pred"]).items()} if h else None})
    for part in ("frozen", *STAGES):                          # B1a readouts; stage A's mean ones equal the frozen ones
        for name in cv["fits"][0][part].get("readout", {}):
            if (part == "frozen" and "learned" in name) or (part == "A" and "mean" in name):
                continue
            out.append({"label": f"B1a readout of {r['variant']} MIST {part if part == 'frozen' else 'after ' + STAGE_NAME[part]}: "
                                 f"{name}", "pred_N": np.concatenate([f[part]["readout"][name] for f in cv["fits"]]),
                        "held": None, "fits": None})
    return out


def describe(row: dict) -> list[str]:
    """Head after training, epochs, inner validation loss and minutes of a row's cross-validation fits."""
    fits = row["fits"]
    lines = []
    heads = [f["head"] for f in fits if "head" in f]
    if heads:
        w, b = np.array(heads).T
        lines.append(f"head weight {np.median(w):.2f} ({w.min():.2f} to {w.max():.2f}), bias {np.median(b):.2f} "
                     f"({b.min():.2f} to {b.max():.2f}), median (range) over {len(heads)} fits")
    else:
        lines.append("head: not recorded per fit")
    first = [f["inner_val_loss"][0] for f in fits]
    best = [min(f["inner_val_loss"]) for f in fits]
    ep = [f["epochs"] for f in fits]
    lines.append(f"epochs chosen {min(ep)}-{max(ep)} (median {np.median(ep):.0f}), run "
                 f"{min(f['epochs_run'] for f in fits)}-{max(f['epochs_run'] for f in fits)}; inner validation loss "
                 f"first epoch {min(first):.3f}-{max(first):.3f}, best {min(best):.3f}-{max(best):.3f}; "
                 f"{row['minutes']:.0f} min for {len(fits)} fits")
    if "line" in fits[0]:
        a, c = np.array([f["line"] for f in fits]).T
        lines.append(f"line slope {np.median(a):.2f} ({a.min():.2f} to {a.max():.2f}), intercept {np.median(c):.2f} "
                     f"({c.min():.2f} to {c.max():.2f})")
    if "start_mean_d" in fits[0]:
        lines.append(f"start: mean (1 - cos) of fit pairs {np.median([f['start_mean_d'] for f in fits]):.4f}, "
                     f"output {np.median([f['start_output'] for f in fits]):.3f}")
    return lines


def baselines(train, y_P) -> dict:
    """B1 and B1a MIST: label -> (distance function, dist_P)."""
    fs = features.ensure("mist")
    return {label: (fn, fn(train, fs.cids_M, fs.probs_MxW))
            for label, fn in ((BASELINE, mixture.mean_cosine_distance), (B1A, mixture.paper_distance))}, fs


def pommix_predictions() -> dict:
    """Mean of the 5 released POMMix checkpoints per held-out set, from the latest check_pommix --final run."""
    for f in sorted((results_dir() / "check_pommix").glob("*_final.json"), reverse=True) or \
            sorted((REPO / "results/check_pommix").glob("*_final.json"), reverse=True):
        values = {i["key"]: i["value"] for i in json.loads(f.read_text())["items"] if i["type"] == "value"}
        return {"T46": values["t46_predictions"]["mean of the 5"], "V50": values["v50_predictions"]["mean of the 5"],
                "file": f.name}
    raise SystemExit("run evaluation/check_pommix.py --final first")


def change(s: dict, base: dict) -> list:
    signed = lambda x: "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"      # noqa: E731
    return [signed(s["rmse"] - base["rmse"]), signed(s["r"] - base["r"]) + "*" * (s.get("p_vs_ref", 1.0) < 0.05)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", action="store_true", help="score T46 and V50 (do this once)")
    ap.add_argument("--cv-only", action="store_true", help="cross-validation only, no held-out sets")
    ap.add_argument("--version", type=int, choices=(1, 2), default=2, help="model version to train (default 2)")
    ap.add_argument("--variants", nargs="+", choices=list(LABEL), default=list(LABEL), help="encoders to train")
    ap.add_argument("--retrain", action="store_true", help="train again even if checkpoints exist")
    args = ap.parse_args()
    assert not (args.final and args.cv_only), "--final needs the final models"
    rep = Report("eval_mist_mixture", {"k": K, "repeats": REPEATS, **vars(args)}, tag="final" * args.final)
    train = data.load_train()
    y_P = evaluate.labels(train)
    held = {} if args.cv_only else {"T46": data.load_test(), "V50": data.load_validation()}
    n_held = {"T46": len(data.load_test()), "V50": len(data.load_validation())}

    rep.section(f"1. Training version {args.version} (one process per variant)")
    trained = {args.version: run_variants(args.variants, args.retrain, args.version, args.cv_only)}
    for version in (1, 2):                                    # other saved versions and variants, for comparison
        for v in LABEL:
            if v not in trained.get(version, {}) and (r := load_saved(v, version)):
                trained.setdefault(version, {})[v] = r
    from mixdist import mist_mixture as mm
    for version in sorted(trained):
        settings = mm.SETTINGS if version == 1 else mm.SETTINGS_V2
        rep.line(f"v{version} settings: " + ", ".join(f"{k} {v}" for k, v in settings.items()))
    rows = [row for v in LABEL for version in sorted(trained) if v in trained[version]
            for row in rows_of(trained[version][v], version, n_held)]

    rep.section(f"2. Cross-validation predictions ({K} folds x {REPEATS} repeats, whole mixtures held out, seed {SEED})")
    dists, fs = baselines(train, y_P)
    cv_N = {label: evaluate.cross_validate(train, evaluate.linear_calibration(d_P, y_P, np.zeros(len(y_P))),
                                           k=K, repeats=REPEATS, seed=SEED) for label, (_, d_P) in dists.items()}
    pair_N, truth_N = cv_N[BASELINE].pair_N, cv_N[BASELINE].truth_N
    pred_N = {label: res.pred_N for label, res in cv_N.items()}
    for version, by_variant in trained.items():
        for r in by_variant.values():
            assert r["cv"]["pair_N"] == pair_N.tolist(), "folds differ from the baseline's"
    for row in rows:
        pred_N[row["label"]] = row["pred_N"]
    order = [BASELINE, B1A, *[row["label"] for row in rows], POMMIX]     # B1a: the bar a fine-tune must beat
    rep.line(f"{len(pair_N)} predictions of {len(set(pair_N.tolist()))} distinct pairs, all studies pooled")
    rep.line("+ line: each fold's line is fitted on that model's predictions for its inner validation pairs "
             "(never on test pairs)")
    rep.line("B1a readout: quantile per dimension over the 235 molecules, mean per mixture (or the model's learned "
             "weights), length 1, Euclidean, one line on the fold's training pairs. CLS = first-token embedding; "
             "words = the odour model's 135 odour-word probabilities on that encoder")

    if args.cv_only:
        rep.section("3. Cross-validation only (--cv-only): T46 and V50 not used")
        y_H = {}
    elif args.final:
        rep.section("3. T46 and V50, scored once")
        earlier = rep.previous_runs(final=True)
        if earlier:
            rep.line(f"NOTE: --final was already run at {', '.join(earlier)}")
        y_H = {hs: evaluate.labels(pairs) for hs, pairs in held.items()}
    else:
        rep.section("3. DRY RUN: random numbers in place of the T46 and V50 labels (pass --final to score)")
        rng = np.random.default_rng(0)
        y_H = {hs: rng.uniform(0.2, 0.9, len(pairs)) for hs, pairs in held.items()}
    pred_H = {hs: {} for hs in held}
    for label, (fn, d_P) in dists.items():
        a, c = evaluate.fit_line(d_P, y_P)
        for hs, pairs in held.items():
            pred_H[hs][label] = a * fn(pairs, fs.cids_M, fs.probs_MxW) + c
    for row in rows:
        for hs in held:
            if row["held"] is not None:
                pred_H[hs][row["label"]] = row["held"][hs]
    if held:
        pom = pommix_predictions()
        rep.line(f"POMMix: {pom['file']} of check_pommix")
        for hs in held:
            pred_H[hs][POMMIX] = np.array(pom[hs])

    sets = [*held, "CV"]
    scores = {}
    for name, preds, y in [*((hs, pred_H[hs], y_H[hs]) for hs in held), ("CV", pred_N, truth_N)]:
        scores[name] = {label: metrics.score(p, y, reference=None if label == BASELINE else preds[BASELINE])
                        for label, p in preds.items()}
    rep.line("RMSE change: + is worse.  r change: - is worse.  * = r change with p < 0.05 (paired bootstrap).")
    base = {name: s[BASELINE] for name, s in scores.items()}
    table = [["baseline: " + BASELINE, *[f"{base[n][m]:.3f}" for n in sets for m in ("rmse", "r")]]]
    for label in order[1:]:
        if any(label in scores[n] for n in sets):
            table.append([label, *[x for n in sets for x in
                                   (change(scores[n][label], base[n]) if label in scores[n] else ["n/a", "n/a"])]])
    rep.table(["", *[f"{n} {m}" for n in sets for m in ("RMSE", "r")]], table)
    if held:
        rep.line("CV: POMMix trains on its own pairs, so it has no score on these folds.")

    rep.section("4. Cross-validation predictions: level and spread; head, epochs and time")
    rep.line(f"labels: mean {truth_N.mean():.3f}, sd {truth_N.std():.3f}")
    for label in order:
        if label in pred_N:
            rep.line(f"{label}: predictions mean {pred_N[label].mean():.3f}, sd {pred_N[label].std():.3f}")
    for row in rows:
        if row["fits"] is not None and not row["label"].endswith(" + line"):
            rep.line(row["label"].removesuffix(" raw") + ":")
            for line in describe(row):
                rep.line("  " + line)

    rep.section("5. Convergence: median over cross-validation fits (inner validation loss per epoch)")
    rep.line("loss: v2 mean squared error; v1 1 - R^2. Live logs: checkpoints/mist_mixture/v2/<variant>/train_log.tsv "
             "(per epoch) and fit_summary.tsv (per fit)")
    conv = []
    for row in rows:
        if row["fits"] is None or row["label"].endswith(" + line"):
            continue
        fits = row["fits"]
        med = lambda xs: float(np.median(xs))                              # noqa: E731
        conv.append([row["label"].removesuffix(" raw"), len(fits),
                     f"{med([f['inner_val_loss'][0] for f in fits]):.4f}", f"{med([min(f['inner_val_loss']) for f in fits]):.4f}",
                     f"{med([f['inner_val_loss'][-1] for f in fits]):.4f}",
                     f"{med([f['train_loss_at_best'] for f in fits]):.4f}" if "train_loss_at_best" in fits[0] else "n/a",
                     f"{med([f['epochs'] for f in fits]):.0f}", f"{med([f['epochs_run'] for f in fits]):.0f}"])
    rep.table(["", "fits", "val first", "val best", "val last", "train at best", "best epoch", "epochs run"], conv)

    for name in sets:
        rep.section(f"Absolute scores, {name} (RMSE / r, 95% CI of r)")
        rep.scores([(label, scores[name][label]) for label in order if label in scores[name]], label="")
    if held:
        rep.value("predictions", {hs: {label: np.asarray(p).tolist() for label, p in pred_H[hs].items()} for hs in held})
    rep.value("cv", {"pair_N": pair_N.tolist(), "truth_N": truth_N.tolist(),
                     "pred_N": {label: p.tolist() for label, p in pred_N.items()}})
    rep.save()


if __name__ == "__main__":
    main()
