# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = training pairs (820)   W = odour words (138)
# ---------------------------------------------------------------------------------------------
"""O1.0 (competition on open-POM's frozen odour words, mixdist/o10.py) scored by zero-overlap
cross-validation on the DREAM training pairs, for one variant:

  full              competition (s_w learned) + per-word distance weights v_w + length-1 step
  competition_only  s_w learned, v_w = 1
  weights_only      plain mean (B1a's mixing), v_w learned
  no_unit           full, without the length-1 step
  o12               O1.2: the molecules strongest on a word dominate it (softmax over molecules, tau_w); no v_w
  o11               O1.1: occupancy depends on the molecule, r_w = sum e / (1 + s_w sum q); no v_w

Training: DREAM pair distances only, squared error, full batch, fixed defaults (mixdist.o10.Settings10,
not tuned). Reference on identical folds: B1a open-POM (the open-POM baseline recipe), which is
also O1.0 before training.
V50/T46 only with --final (no CV then): trained on all training pairs, both sets scored once.

Shared by the eval_o10_<variant>.py scripts in this folder. Not a script.
"""
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import data, evaluate, features, metrics, mixture, o10  # noqa: E402
from mixdist.report import Report  # noqa: E402

VARIANTS = {"full": {}, "competition_only": {"weights": False}, "weights_only": {"competition": False},
            "no_unit": {"unit": False}, "o11": {"occupancy": True, "weights": False},
            "o12": {"dominance": True, "weights": False}}


def run(script: str, variant: str, doc: str, repeats: int = 10):
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=repeats, help="CV repeats of 5 folds")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 repeat, 100 steps")
    ap.add_argument("--live", action="store_true", help="print training progress every log_every steps")
    ap.add_argument("--final", action="store_true", help="skip CV; train on all training pairs, score V50 and T46 once")
    args = ap.parse_args()
    s = o10.Settings10(**VARIANTS[variant])
    if args.quick:
        args.repeats, s.steps = 1, 100
    rep = Report(script, {"variant": variant, **vars(args), **asdict(s)}, tag="quick" * args.quick or "final" * args.final)

    rep.section("1. Data")
    train = data.load_train()
    fs = features.load("openpom")
    rep.line(f"open-POM word probabilities (frozen): {fs.probs_MxW.shape[0]} molecules x {fs.probs_MxW.shape[1]} words; "
             f"DREAM training pairs {len(train)}")
    live = (lambda line: print(line, flush=True)) if args.live else None
    trainer = o10.Trainer10(fs.cids_M, fs.probs_MxW, train, s, progress=live)

    if args.final:
        final(rep, trainer, train, fs.words_W)
        rep.save()
        return

    rep.section(f"2. Cross-validation ({args.repeats} x 5 folds, whole mixtures held out)")
    logs = []

    def fit_predict(train_idx, test_idx):
        trainer.tag = f"fold {len(logs) + 1}/{5 * args.repeats}"
        model, log = trainer.fit(train_idx)
        logs.append(log)
        if len(logs) <= 5:
            rep.line(f"fold {len(logs)}: step 0 " + fmt(log[0]) + " | last " + fmt(log[-1]))
        return trainer.predict(model, test_idx)

    t0 = time.time()
    result = evaluate.cross_validate(train, fit_predict, k=5, repeats=args.repeats)
    rep.line(f"{len(logs)} folds in {time.time() - t0:.0f} s")
    rep.scores(result.score_rows(), label={"o11": "O1.1", "o12": "O1.2"}.get(variant, f"O1.0 {variant}"))

    rep.section("3. Reference on the same folds (= O1.0 before training)")
    y_P = evaluate.labels(train)
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(
        mixture.paper_distance(train, fs.cids_M, fs.probs_MxW), y_P, np.zeros(len(train))), k=5, repeats=args.repeats)
    rep.scores(ref.score_rows(), label="B1a open-POM")
    rep.value("training_log", logs)
    rep.value("cv_predictions", {"pair": result.pair_N.tolist(), "pred": result.pred_N.tolist(),
                                 "b1a": ref.pred_N.tolist()})
    rep.save()


def fmt(entry):
    return ", ".join(f"{k} {v:.3f}" for k, v in entry.items() if k != "step")


def final(rep, trainer, train, words_W):
    rep.section("2. Held-out sets, scored once with the model trained on all training pairs")
    earlier = rep.previous_runs(final=True)
    if earlier:
        rep.line(f"NOTE: --final was already run at {', '.join(earlier)}")
    trainer.tag = "final"
    model, log = trainer.fit(np.arange(len(train)))
    rep.line("step 0: " + fmt(log[0]) + " | last: " + fmt(log[-1]))
    rows, preds = [], {}
    for hs in evaluate.held_out_sets():
        pred_P = trainer.predict_pairs(model, hs.pairs)
        preds[hs.name] = pred_P.tolist()
        rows.append((hs.name, evaluate.score_held_out(hs, pred_P)))
        rmse_med, r_med = metrics.bootstrap_medians(pred_P, hs.y_P)
        rep.line(f"{hs.name}: bootstrap medians RMSE {rmse_med:.3f}, r {r_med:.3f}; "
                 f"bias {pred_P.mean() - hs.y_P.mean():+.3f} (mean label {hs.y_P.mean():.3f})")
    rep.scores(rows, label="set")
    s_W, v_W = model.binding_strength().detach().cpu().numpy(), model.nu.detach().exp().cpu().numpy()
    o = np.argsort(s_W)
    rep.line("smallest s (sum-like): " + ", ".join(f"{words_W[i]} {s_W[i]:.2f}" for i in o[:8]))
    rep.line("largest s (mean-like): " + ", ".join(f"{words_W[i]} {s_W[i]:.2f}" for i in o[::-1][:8]))
    o = np.argsort(v_W)[::-1]
    rep.line("largest distance weight v: " + ", ".join(f"{words_W[i]} {v_W[i]:.2f}" for i in o[:8]))
    rep.value("training_log", log)
    rep.value("learned", {"words": list(map(str, words_W)), "s": s_W.tolist(), "v": v_W.tolist()})
    rep.value("predictions", preds)
