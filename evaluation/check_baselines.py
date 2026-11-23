# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes:  P = T46 pairs (46), in the order of mixdist.data.load_test()
# ---------------------------------------------------------------------------------------------
"""Every model we have run on V50 and T46, one table per set, from saved results only (nothing is
rescored).

Rows and where their predictions come from (latest saved run of each script):
  D2Smell, PL21       T46: check_d2smell.py / check_pl21.py (team's saved models, rerun with their
                      code). V50: the published predictions (pnas_repo/Predictions, the organisers'
                      run of the team's model); the teams' code has no V50 path
  POMMix              T46 only: check_pommix.py, mean of its 5 released checkpoints
  open-POM baseline   B1/eval_semantic_openpom_b1a.py --final (= B1a open-POM; same recipe as the
                      paper's notebook in check_openpom_official.py, predictions agree to 0.001)
  B1a MIST            B1/eval_semantic_mist_odour_b1a.py --final
  B1b MIST, open-POM  B1/eval_semantic_mist_odour_b1b.py, eval_semantic_openpom_b1b.py --final
                      (B1a with one line for pairs sharing no molecule, one for the rest)

Columns: RMSE and Pearson r (bootstrap medians, as SI Table S4) with the 95% CI on r, then the
bias = mean prediction - mean label, and the RMSE left after removing that bias. The last one
uses the set's own mean, so it is a diagnostic of where the error comes from, never a score or a
choice. Below, mean labels: where a line fitted on training pairs takes its level from.

Env:     mist
Usage:   python evaluation/check_baselines.py
         (needs a run of each script above; the B1 runs with --final)
Writes:  results/check_baselines/<date_time>.{txt,json}
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data, evaluate, metrics  # noqa: E402
from mixdist.report import Report, results_dir  # noqa: E402

TRAINED_ON = {
    "D2Smell": "trees on 780 measured + 19,798 synthetic pairs; features from a pair GNN trained on GoodScents blends",
    "PL21": "single-molecule ratings; distance: positive Lasso on 60 Bushdid pairs",
    "POMMix": "its own 865 pairs (Snitz, Ravia, Bushdid), end to end",
    "open-POM baseline (= B1a open-POM)": "frozen; one line on all 820 training pairs",
    "B1a MIST": "frozen; one line on all 820 training pairs",
    "B1b open-POM": "frozen; line on the 325 training pairs sharing no molecule",
    "B1b MIST": "frozen; line on the 325 training pairs sharing no molecule",
}
OURS = {"open-POM baseline (= B1a open-POM)": "eval_semantic_openpom_b1a", "B1a MIST": "eval_semantic_mist_odour_b1a",
        "B1b open-POM": "eval_semantic_openpom_b1b", "B1b MIST": "eval_semantic_mist_odour_b1b"}


def latest(script: str, final: bool = False) -> tuple[dict, str]:
    """Newest saved run of ``script`` (with --final if ``final``) -> (run, file name)."""
    runs = [f for f in sorted((results_dir() / script).glob("*.json"))
            if not final or json.loads(f.read_text()).get("args", {}).get("final")]
    if not runs:
        raise SystemExit(f"no saved run of {script}{' with --final' if final else ''}; see Usage")
    return json.loads(runs[-1].read_text()), runs[-1].name


def value(run: dict, key: str):
    return [i["value"] for i in run["items"] if i["type"] == "value" and i["key"] == key][-1]


def predictions(hs, sources: set) -> dict:
    """{model: pred_P} for held-out set ``hs`` (an evaluate.HeldOutSet), in the order of hs.pairs."""
    preds = {}
    if hs.name == "T46":
        for name, script in (("D2Smell", "check_d2smell"), ("PL21", "check_pl21")):
            run, f = latest(script)                         # saved in load_test order = hs.pairs order
            preds[name] = value(run, "predictions")
            sources.add(f"{script}/{f}")
        run, f = latest("check_pommix")
        preds["POMMix"] = value(run, "t46_predictions")["mean of the 5"]
        sources.add(f"check_pommix/{f}")
    else:
        published = data.load_baseline_predictions("V50")
        for name in ("D2Smell", "PL21"):
            preds[name] = [published[name][tuple(sorted((p.id_a, p.id_b)))] for p in hs.pairs]
        sources.add("pnas_repo/Predictions/Validation_set_Prediction_top6_Teams.csv")
    for name, script in OURS.items():
        run, f = latest(script, final=True)
        preds[name] = value(run, "predictions")[hs.name]
        sources.add(f"{script}/{f}")
    return {k: np.asarray(v, dtype=float) for k, v in preds.items()}


def main():
    rep = Report("check_baselines")
    sources, saved = set(), {}
    for hs in evaluate.held_out_sets():
        preds = predictions(hs, sources)
        rows = []
        for name, pred_P in preds.items():
            s = metrics.score(pred_P, hs.y_P)
            rmse, r = metrics.bootstrap_medians(pred_P, hs.y_P)
            bias = pred_P.mean() - hs.y_P.mean()
            rows.append([name, rmse, r, f"[{s['r_ci'][0]:.2f}, {s['r_ci'][1]:.2f}]", f"{bias:+.3f}",
                         metrics.rmse(pred_P - bias, hs.y_P), TRAINED_ON[name]])
        rows.sort(key=lambda row: row[1])
        kind = "similarity ratings" if hs.name == "V50" else "triangle tests"
        rep.section(f"{hs.name} ({len(hs.pairs)} pairs, {kind}, every pair shares no molecule): by RMSE")
        rep.table(["model", "RMSE", "r", "r 95% CI", "bias", "RMSE w/o bias", "trained on"], rows)
        rep.line()
        rep.line(f"bias = mean prediction - mean {hs.name} label ({hs.y_P.mean():.3f}); 'RMSE w/o bias' is a diagnostic only"
                 + (f"; noise ceiling r = {hs.ceiling:.3f}" if hs.ceiling else ""))
        saved[hs.name] = {k: v.tolist() for k, v in preds.items()}

    train = data.load_train()
    y_tr = evaluate.labels(train)
    triangle = np.array([p.dataset in evaluate.TRIANGLE_STUDIES for p in train])
    disjoint = np.array([p.jaccard == 0 for p in train])
    rep.section("Mean label: where a line fitted on training pairs takes its level from")
    rows = [[f"{hs.name}", len(hs.pairs), hs.y_P.mean()] for hs in evaluate.held_out_sets()]
    for label, m in (("training, all", np.ones(len(y_tr), bool)),
                     ("training, sharing no molecule", disjoint),
                     ("training, sharing no molecule, triangle tests", disjoint & triangle),
                     ("training, sharing no molecule, similarity ratings", disjoint & ~triangle),
                     ("training, sharing a molecule", ~disjoint)):
        rows.append([label, m.sum(), y_tr[m].mean()])
    rep.table(["pairs", "n", "mean label"], rows)
    rep.line()
    rep.line("from: " + ", ".join(sorted(sources)))
    rep.value("predictions", saved)
    rep.save()


if __name__ == "__main__":
    main()
