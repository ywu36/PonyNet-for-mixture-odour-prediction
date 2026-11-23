# ---------------------------------------------------------------------------------------------
# Shape letters used as name suffixes (rows x columns)
#   M = molecules with SMILES (235)
#   W = open-POM odour words (138)
#   P = T46 pairs (46)
# ---------------------------------------------------------------------------------------------
"""Everything about open-POM in one place: check our downloaded model, then reproduce the paper's
open-POM baseline on the test set (T46) with it.

Part 1, the standard model checks of mixdist.checks (same as check_mist.py): embeddings computed
(runs embed_openpom.py in the openpom env if missing), the SMILES are our molecules, odour-word
outputs vs real GS-LF labels (AUROC), embedding linear probe, agreement with MIST.

Part 2, the paper's baseline:

The paper's notebook (pnas_repo/SOTA/2-openPOM/POM_to_Perception-Transform.ipynb) is run twice,
its code unchanged except for the three things in ``notebook_as_script``:
  A. with the paper's own open-POM output file (openpom_ensemble_predictions_results.csv):
     must give the paper's number exactly;
  B. with that file replaced by the outputs of the open-POM model we downloaded and ran
     (features/openpom.npz, released 10-model ensemble): the same method on our model.
What the notebook does:
  1. per molecule: 138 open-POM odour-word probabilities, each word quantile-transformed to a
     normal distribution over all molecules;
  2. per mixture: mean over its molecules; per pair: correlation, cosine similarity, Euclidean
     distance and angle between the two mixtures;
  3. per measure: a straight line and a log curve fitted on all training pairs, applied to T46.
The paper's "POM" baseline is the Euclidean distance with the straight line; all 8 variants are
reported. The last section shows how far our model's per-molecule outputs are from the paper's
file, which is where any difference between A and B comes from.

As published, the notebook fits its lines on a precomputed table (data/processed/
mixture_similarity_results.csv) rather than on the table its own step 2 writes; here the fit
reads the table step 2 just wrote, so every number comes from the run.

Env:     mist to start it; the notebook runs in the conda environment pommix
         (pandas, scikit-learn, scipy, matplotlib, seaborn)
Usage:   python evaluation/check_openpom_official.py
Writes:  results/check_openpom_official/<date_time>.{txt,json}
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import checks, data, features, metrics, openpom, teams  # noqa: E402
from mixdist.report import Report  # noqa: E402

PAPER = (0.114, 0.397)                                       # SI Table S4: median RMSE, median Pearson r
ENV = "pommix"
NOTEBOOK = "2-openPOM/POM_to_Perception-Transform.ipynb"
OPENPOM_FILE = "data/raw/openpom_ensemble_predictions_results.csv"
FILES = (NOTEBOOK, "util/plot.py", "data/raw/Mixure_Definitions_All.csv",
         "data/raw/TrainingData_mixturedist.csv", OPENPOM_FILE)
PUBLISHED_TABLE = "data/processed/mixture_similarity_results.csv"
OUR_TABLE = "data/processed/mixture_similarity_results_POM.csv"
MEASURES = ("Correlation", "Cosine_Similarity", "Euclidean_Distance", "Angle")
SHAPES = (("linear", "linear"), ("log", "Logarithmic"))      # (our label, the notebook's column suffix)
PAPER_VARIANT = "euclidean distance, linear"


def notebook_as_script(work: Path) -> str:
    """The notebook's code cells, unchanged except: no plot windows, ``display`` = print, and the
    line fits read the similarity table this run computed (see the docstring)."""
    cells = json.loads((work / NOTEBOOK).read_text())["cells"]
    code = "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    fits_published = f"read_csv('../{PUBLISHED_TABLE}')"
    assert code.count(fits_published) == len(MEASURES), "notebook changed: expected one fit per measure"
    assert f"to_csv('../{OUR_TABLE}'" in code
    code = code.replace(fits_published, f"read_csv('../{OUR_TABLE}')").replace("plt.show()", "plt.close('all')")
    return "display = print\n" + code


def write_our_openpom(path: Path) -> None:
    """Replace the paper's open-POM file with our model's outputs, in the columns the notebook reads."""
    fs = features.load("openpom")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CID", *fs.words_W])
        for cid, probs_W in zip(fs.cids_M, fs.probs_MxW):
            w.writerow([int(cid), *(f"{p:.10g}" for p in probs_W)])


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]


def run_notebook(source: Path, work: Path, ours: bool) -> tuple[dict, float]:
    """Copy the notebook's inputs to ``work``, optionally swap in our open-POM outputs, run it.
    Returns ({variant: pred_P}, y_P) in the row order of the published T46 prediction file, and
    the seconds it took."""
    if work.exists():
        shutil.rmtree(work)
    for rel in FILES:
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, work / rel)            # no metadata: the sync drive refuses xattr reads
    (work / "data/processed").mkdir(parents=True, exist_ok=True)
    (work / "output").mkdir()
    if ours:
        write_our_openpom(work / OPENPOM_FILE)
    (work / "2-openPOM/run.py").write_text(notebook_as_script(work))
    seconds = teams.run_in_env(ENV, ["run.py"], work / "2-openPOM")

    preds, y_P = {}, None
    for m in MEASURES:
        out = read_rows(work / f"output/POM_{m}_POM_Predictions.csv")
        y_P = np.array([float(r["Experimental Values"]) for r in out])
        for shape, col in SHAPES:
            preds[f"{m.replace('_', ' ').lower()}, {shape}"] = np.array(
                [float(r[f"Predicted_POM {m.replace('_', ' ')}-{col}"]) for r in out])
    assert np.allclose(np.sort(y_P), np.sort([p.y for p in data.load_test()])), "output rows are not T46"
    return {"preds": preds, "y_P": y_P}, seconds


def main():
    rep = Report("check_openpom_official")
    checks.run_standard_checks(rep, "openpom")
    source = data.locate("pnas_repo", needs="SOTA/" + NOTEBOOK) / "SOTA"
    runs = {}
    rep.section("Run")
    rep.line(f"paper code  {source / NOTEBOOK}")
    for key, ours, label in (("paper_file", False, "A. paper's open-POM file"),
                             ("our_model", True, "B. our downloaded open-POM (features/openpom.npz)")):
        work = teams.runs_dir() / f"openpom_official_{key}"
        runs[key], seconds = run_notebook(source, work, ours)
        rep.line(f"{label:52s} ran in {work} ({seconds:.0f} s, env {ENV})")

    rows = []
    for variant in runs["paper_file"]["preds"]:
        cells = [variant + ("   <- paper's POM" if variant == PAPER_VARIANT else "")]
        for key in runs:
            cells += metrics.bootstrap_medians(runs[key]["preds"][variant], runs[key]["y_P"])
        rows.append(cells)
    rep.section("T46, all 8 variants (bootstrap medians, as SI Table S4)")
    rep.table(["distance, line", "A RMSE", "A r", "B RMSE", "B r"], rows)

    a_P, b_P = runs["paper_file"]["preds"][PAPER_VARIANT], runs["our_model"]["preds"][PAPER_VARIANT]
    published_P = np.array(list(data.load_t46_sota_predictions()["POM"].values()))
    a, b = (metrics.bootstrap_medians(p, runs["paper_file"]["y_P"]) for p in (a_P, b_P))
    rep.section("The paper's open-POM baseline on T46")
    rep.table(["", "RMSE", "Pearson r"], [["paper (SI Table S4)", *PAPER],
                                          ["A. rerun, paper's open-POM file", *a],
                                          ["B. rerun, our downloaded open-POM", *b]])
    rep.line()
    rep.line(f"A vs the published POM predictions, largest difference over the 46 pairs: {np.abs(a_P - published_P).max():.1e}")

    fs = features.load("openpom")
    cmp = openpom.compare_with_published(fs.cids_M, fs.probs_MxW)
    rep.section("Where B differs from A: our open-POM outputs vs the paper's file, per molecule and word")
    rep.table(["quantity", "value"], [
        ["molecules in both files", cmp["n_common"]],
        ["Pearson over all values", f"{cmp['pearson_all_values']:.4f}"],
        ["mean |ours - paper|", f"{cmp['mean_abs_diff']:.4f}"],
        ["max |ours - paper|", f"{cmp['max_abs_diff']:.4f}"],
    ])
    rep.line("the paper's file (Jan 2025) predates the released checkpoints (Oct 2025): a different ensemble")
    rep.value("paper_file_predictions", a_P.tolist())
    rep.value("our_model_predictions", b_P.tolist())
    rep.value("vs_published_file", cmp)
    rep.save()


if __name__ == "__main__":
    main()
