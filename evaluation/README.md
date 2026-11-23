# evaluation/

Every experiment of the project, one script per experiment. Scripts follow `ARCHITECTURE.md`: a docstring with the
purpose, `Env:`, `Usage:` and `Writes:`; output through `mixdist.report.Report`, saved under `results/<script>/`.
`CV.md` is the cross-validation protocol (whole mixtures held out, 50 x 5, each pair counted once, mixture-resampling
intervals).

Development names used in the code, and what they are in the paper:

| Code name | Paper |
|---|---|
| B1, B1a | open-POM (or MIST) averaged, compared by angle, one fitted line: the DREAM baseline, "open-POM avg" / "MIST avg" |
| B1a+, O1.2b | Session 1: open-POM + chemical-group weights + magnitude in the distance ("open-POM + both factors") |
| O3, `o3m_w64` | Pony (w/o attention): the small network with mean pooling over atoms |
| O5, `o3m_attpool` | the Pony network: the small network with attention pooling over atoms |
| model (b) | Pony without the mixture length (build-up step 4, Table 7) |
| O6, "(b) + hidden length, learned (133)" | Pony |

| Folder | What it holds |
|---|---|
| top level | `embed_<model>.py` (run a frozen model on every molecule), `check_<model>.py` (reproduce other teams' and models' published numbers: open-POM, MIST, POMMix, D2Smell, PL21), `eval_b1a_cv.py` |
| `B1/` | the averaging baselines (open-POM, MIST, Uni-Mol 2, Morgan fingerprints) and their calibration |
| `B1aplus/`, `B1b/`, `Substructure/` | Session 1: chemical groups, mixture size and magnitude terms added to the baseline, and the variants tried along the way |
| `B2ft/` | MIST fine-tuned on mixture ratings (Section 2.1) |
| `D2Smell/` | ablation of team D2Smell's model |
| `O1/` | receptor-layer readouts on frozen open-POM, the Session 1 reference model and the paired held-out scoring helpers (`final_heldout.py`) |
| `O2/` | the PyTorch rebuild of open-POM and its audits, open-POM retrained variants, graph export (`export_o2_graphs.py`) |
| `O3/` | the small network, attention pooling, the mixture readout and every step that led to Pony: `eval_attpool.py` (Table 6), `eval_o5_softmax.py` and `eval_size_tune.py` (Table 7, the final model), `eval_pony_ablation.py` (Table 8), plus diagnostics |

`reproduce/REPRODUCE.md` lists the exact commands behind each table of the paper.
