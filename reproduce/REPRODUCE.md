# Reproducing the paper

Every table of the paper, with the command that produces it and the file its numbers are written to. Commands run
from the repository root. `Env` is the conda environment named in each script's docstring (`mist` for almost all;
`openpom`, `pommix`, `d2smell`, `pl21` only where a released model must run in its own environment). Wrap heavy runs
in a memory limit if your machine is shared.

## 0. Data and cached inputs

| Step | Command | Writes |
|---|---|---|
| DREAM data | put the DREAM challenge repository in `external/pnas_repo/` (or point `OSMO_DATA` at a folder holding it) | — |
| open-POM word profiles of the 235 DREAM molecules | `python evaluation/embed_openpom.py` (env openpom) | `features/openpom.npz` |
| MIST word profiles | `python evaluation/embed_mist.py` | `features/mist.npz` |
| Molecule graphs (GS-LF single molecules + DREAM molecules) | `python evaluation/O2/export_o2_graphs.py` (env openpom) | `features/o2_graphs.npz` |

## 1. Stage 1: the networks

| Network | Command | Writes |
|---|---|---|
| Pony network (attention pooling), 10 copies | `python training/train_o3m.py --config attpool --seeds 0 1 2 3 4 5 6 7 8 9` | `checkpoints/o3m_attpool/` |
| Pony (w/o attention), 10 copies | `python training/train_o3m.py --width 64 --seeds 0 1 2 3 4 5 6 7 8 9` | `checkpoints/o3m_w64/` |
| open-POM layout retrained (Tables A1, A3) | `python training/train_o2c.py --seeds 0 1 2 3 4` | `checkpoints/o2c/` |

The shipped model file is then built, and checked against the research pipeline, by
`python reproduce/build_pony_weights.py` (writes `pony/weights/pony_dream.pt`; T46 0.593, V50 0.622).

## 2. Tables

Cross-validation (CV) is 50 repeats x 5 folds on the 514 development pairs, whole mixtures held out
(`evaluation/CV.md`). Long CV runs are split into chunks of repeats (`--reps a:b`) or rows (`--rows`) and joined with
`--merge`/`--score`; the joined result is the same as one run.

| Table | Rows | Command | Result file |
|---|---|---|---|
| 1, 4 | open-POM avg, MIST avg (T46, V50) | `python evaluation/O1/final_heldout.py` | `results/final_heldout/` |
| 1, 4 | D2Smell, PL21 (T46: rerun of the released models) | `python evaluation/check_d2smell.py`, `python evaluation/check_pl21.py` | `results/check_d2smell/`, `results/check_pl21/` |
| 1, 4 | D2Smell, PL21 (V50: the teams' published predictions) | `python evaluation/check_baselines.py` | `results/check_baselines/` |
| 1, 3, 4 | POMMix | `python evaluation/check_pommix.py --final` (env pommix) | `results/check_pommix/` |
| 2 | baseline, + chemical groups, + size count, + both (size count) | `python reproduce/session1_factors.py`, `python reproduce/session1_size_terms.py` (print only) | screen |
| 2, 3, 4 | + both factors (magnitude in distance): T46, V50 | `python evaluation/O1/final_heldout.py` | `results/final_heldout/` |
| 2, 4 | + both factors: CV | `python evaluation/O3/eval_o5_softmax.py` (CV parts `--rows 0,2,3`, `--rows 1`, `--rows 4`, `--rows 5`, then `--merge <part pickles>`) | `results/eval_o5_softmax/` |
| 4, 6 | open-POM avg and MIST avg: CV | `python reproduce/baselines_cv.py` (print only; reads the CV pickles of `eval_o5_softmax.py`) | screen |
| 3 | MIST fine-tuned on mixtures (26.7 M parameters) | `python evaluation/B2ft/eval_mist_mixture.py --cv-only --version 2` | `results/eval_mist_mixture/` |
| 4, 7, 8 | Pony | `python evaluation/O3/eval_pony_ablation.py --heldout`; CV: `--reps 0:13 --rows 0,1` (and `13:25`, `25:38`, `38:50`), then `--merge results/eval_pony_ablation/*_reps*.pkl` | `results/eval_pony_ablation/` |
| 6 | Pony (w/o attention), Pony network | `python evaluation/O3/eval_ablation.py --heldout`; CV: `--rows A0,A1,A2,A3,A4,A5`, `--rows A6`, `--rows S1`, then `--score <the three pickles>` (rows A2, A3) | `results/eval_ablation/` |
| 7 | step 1, + odd one out | `eval_ablation.py` rows A3, A4 (as above) | `results/eval_ablation/` |
| 7 | + strength, scale β only | `eval_pony_ablation.py --heldout --rows 7`; CV `--reps a:b --rows 7` (four chunks), `--merge` with the row 0-1 pickles | `results/eval_pony_ablation/` |
| 7, 8 | + anchored correction (= − mixture length), − odd one out, − anchored correction, − strength, − L2 penalty | `eval_pony_ablation.py` rows 1-5: `--heldout`; CV `--reps a:b --rows 2,3,4,5` (four chunks), `--merge` | `results/eval_pony_ablation/` |
| 8 | − odd one out and anchored correction | `eval_pony_ablation.py` row 6: `--heldout --rows 6`; CV `--reps a:b --rows 6`, `--merge` | `results/eval_pony_ablation/` |
| A1 | open-POM: held-out AUROC, memorization gap | `python training/train_o2c_nolstm.py` (row "open-POM layout") | `results/train_o2c_nolstm/` |
| A1 | Pony (w/o attention): held-out AUROC, gap | `python training/train_o3m.py --width 64 --seeds 0 1 2` (mean over the three seeds) | `results/train_o3m/` |
| A3 | pooling and read-out variants of open-POM | `python training/train_o2c_nolstm.py` | `results/train_o2c_nolstm/` |
| A3 | message-passing rounds and width of open-POM | `python training/train_o2c_mp.py` | `results/train_o2c_mp/` |
| A3 | one open-POM component added to Pony | `python training/train_pony_add.py` | `results/train_pony_add/` |
| A2 | the 24 chemical groups and their molecule counts | computed inside the Session 1 scripts (`mixdist.pushpull.units_rdkit_groups`, `merge_identical`) | `results/eval_b1a_plus_exp/` |

## Notes

* **Copies and averaging.** Pony rows average 10 network copies, each with its own fitted readout. In Table A1, Pony's held-out
  AUROC is the mean over its three copies (their ensemble reaches 0.870) and open-POM's is its three-copy ensemble
  (mean over copies 0.880); both memorization gaps are means over copies.
* **Readout fit.** In most copies the fitted temperature τ ends near its floor (10⁻⁴), where L-BFGS is sensitive to
  rounding. Refitting the readout (for example with `pony.Pony.fit`) on another machine or with a different molecule
  order can end at a slightly different point; on the DREAM training pairs, predictions moved by at most 7·10⁻⁴.
* **Rerunning other teams' training.** `training/train_d2smell.py` and `training/train_pl21.py` retrain D2Smell and
  PL21 from scratch with the teams' own code (envs d2smell, pl21).
