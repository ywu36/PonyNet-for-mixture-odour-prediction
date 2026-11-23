# training/

Scripts that train models from scratch. Same conventions as `evaluation/` (see ARCHITECTURE.md,
section 1).

| Script | Trains | Compared with |
|---|---|---|
| `train_pl21.py` | team PL21, the team's own notebook with TRAINING = True | `evaluation/check_pl21.py` (team's saved models), SI Table S4 |
| `train_d2smell.py` | team D2Smell, the team's own training script per seed | `evaluation/check_d2smell.py` (team's saved models), SI Table S4 |

These retrain OTHER teams' models with their own code (level 3: the method works when we run it,
not only their saved outputs). They run the team's code in a copy under `~/.cache/osmo-runs/`,
remove the team's saved models from that copy first, and score T46 only as the team did.

Rules for training scripts, ours included:

- **Name:** `train_<model>.py`, e.g. `train_mixture_mlp.py`.
- **Docstring:** purpose, `Env:`, `Usage:`, `Writes:`; a short `main()`; output through
  `mixdist.report.Report`, so every run is saved under `results/train_<model>/`.
- **Inputs:** per-molecule embeddings from `mixdist.features.ensure(<model>)` (computed on demand
  by `evaluation/embed_<model>.py`) and pairs from `mixdist.data.load_train()`. Never load V50 or
  T46 here.
- **Outputs:** checkpoints in `checkpoints/<model>/` (git-ignored), plus the run record in `results/`.
- **Scoring is not done here.** Put the model code in `mixdist/` with a
  `fit_predict(train_idx, test_idx)` function, so that an `evaluation/eval_<model>.py` scores it with
  `mixdist.evaluate.cross_validate`: the same zero-overlap folds, metrics and tables as B1. A
  trained model is then compared with B1 on identical splits.
