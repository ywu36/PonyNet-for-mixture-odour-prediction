# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = training pairs (820)
# ---------------------------------------------------------------------------------------------
"""O1 (receptor-competition model, mixdist/o1.py) scored by zero-overlap cross-validation on the
DREAM training pairs, for one training-data setting:

  strict  DREAM 2 distances only (+ GS-LF odour words of single molecules, as in every setting)
  joint   pretrain on the GoodScents blend pairs (D2Smell's pair source, external/odor_pair), then
          train on DREAM distances with the pair loss kept on
  pairs   pretrain on the blend pairs only; DREAM is used only for a line y = a*d + c on the
          unweighted receptor distance (like B1)

Blend pairs go through the same receptor layer as 2-molecule mixtures, with an asymmetric loss so
that groups the catalogue did not list are not pushed to 0 (PLAN.md, "Pairwise blend data").
Encoder: open-POM embeddings, frozen. Settings are fixed defaults (mixdist.o1.Settings), not tuned.
Reference on identical folds: B1a open-POM (= open-POM baseline recipe).

The training log (losses, train RMSE/r every log_every steps, per fold; pretraining losses and the
blend-group AUROC on molecule-disjoint pairs) is saved in the run's json. V50/T46 are used only with
--final (no CV then): the model is trained on all training pairs and both sets are scored once.

Shared by the eval_o1_<setting>.py scripts in this folder. Not a script.
"""
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mixdist import data, evaluate, features, metrics, mixture, o1  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402


def run(script: str, setting: str, doc: str, inner_calibration: bool = False, xcal: bool = False,
        encoder: str = "openpom", repeats: int = 10):
    """CV of O1 trained with one data setting ("strict", "joint" or "pairs"); optionally with
    early stopping and a line fitted on an inner split of each training fold."""
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=repeats, help="CV repeats of 5 folds")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 repeat, 200 steps")
    ap.add_argument("--live", action="store_true", help="print the training losses every log_every steps while running")
    ap.add_argument("--final", action="store_true",
                    help="skip CV; train on all training pairs and score V50 and T46 (do this once)")
    args = ap.parse_args()
    s = o1.Settings(inner_calibration=inner_calibration, xcal=xcal)
    if args.quick:
        args.repeats, s.steps, s.pretrain_steps = 1, 200, 200
    rep = Report(script, {"setting": setting, "encoder": encoder, **vars(args), **asdict(s)}, tag="quick" * args.quick or "final" * args.final)

    rep.section("1. Data")
    train = data.load_train()
    tables = o1.load_tables(seed=s.seed, encoder=encoder)
    rep.line(f"encoder {encoder} (frozen); DREAM training pairs {len(train)}; {tables.describe()}")
    rep.line(f"blend groups ({len(tables.group_names)}): {', '.join(tables.group_names)}")
    device = pick_device()
    t0 = time.time()
    live = (lambda line: print(line, flush=True)) if args.live else None
    trainer = o1.Trainer(tables, train, setting, s, device, progress=live)
    if trainer.pretrained is not None:
        model = trainer.new_model()
        model.load_state_dict(trainer.pretrained)
        rep.line(f"pretraining on blend pairs: {s.pretrain_steps} steps, {time.time() - t0:.0f} s; "
                 f"last losses {trainer.pretrain_log[-1]}; blend-group macro AUROC on molecule-disjoint "
                 f"monitor pairs {trainer.pair_monitor_auroc(model):.3f}")

    if args.final:
        final(rep, trainer, train)
        rep.save()
        return

    rep.section(f"2. Cross-validation ({args.repeats} x 5 folds, whole mixtures held out), device {device}")
    logs = []

    def fit_predict(train_idx, test_idx):
        trainer.tag = f"fold {len(logs) + 1}/{5 * args.repeats}"
        model, log = trainer.fit(train_idx)
        logs.append(log)
        if log and len(logs) <= 5:
            last = log[-1]
            rep.line(f"fold {len(logs)}: " + ", ".join(f"{k} {v:.4f}" for k, v in last.items() if k != "step"))
        return trainer.predict(model, test_idx)

    t0 = time.time()
    result = evaluate.cross_validate(train, fit_predict, k=5, repeats=args.repeats)
    rep.line(f"{len(logs)} folds in {time.time() - t0:.0f} s")
    rep.scores(result.score_rows(), label=f"O1 {setting}" + " cal" * inner_calibration + " xcal" * xcal + f" {encoder}" * (encoder != "openpom"))

    rep.section("3. Reference on the same folds: B1a open-POM (open-POM baseline recipe)")
    fs = features.load("openpom")
    dist_P = mixture.paper_distance(train, fs.cids_M, fs.probs_MxW)
    ref = evaluate.cross_validate(train, evaluate.linear_calibration(dist_P, evaluate.labels(train), np.zeros(len(train))),
                                  k=5, repeats=args.repeats)
    rep.scores(ref.score_rows(), label="B1a open-POM")

    rep.value("training_log", {"pretrain": trainer.pretrain_log, "folds": logs})
    rep.value("cv_predictions", {"pair": result.pair_N.tolist(), "pred": result.pred_N.tolist()})
    rep.save()


def final(rep, trainer, train):
    """Train on all training pairs, then score V50 and T46 once (PLAN.md rules 1 and 6)."""
    rep.section("2. Held-out sets, scored once with the model trained on all training pairs")
    earlier = rep.previous_runs(final=True)
    if earlier:
        rep.line(f"NOTE: --final was already run at {', '.join(earlier)}")
    trainer.tag = "final"
    model, log = trainer.fit(np.arange(len(train)))
    if log:
        rep.line("last log entries: " + "; ".join(str({k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()}) for e in log[-2:]))
    rows, preds = [], {}
    for hs in evaluate.held_out_sets():
        pred_P = trainer.predict_pairs(model, hs.pairs)
        preds[hs.name] = pred_P.tolist()
        rows.append((hs.name, evaluate.score_held_out(hs, pred_P)))
        rmse_med, r_med = metrics.bootstrap_medians(pred_P, hs.y_P)
        rep.line(f"{hs.name}: bootstrap medians (as SI Table S4) RMSE {rmse_med:.3f}, r {r_med:.3f}; "
                 f"bias {pred_P.mean() - hs.y_P.mean():+.3f} (mean label {hs.y_P.mean():.3f})")
    rep.scores(rows, label="set")
    rep.value("training_log", {"pretrain": trainer.pretrain_log, "final": log})
    rep.value("predictions", preds)
