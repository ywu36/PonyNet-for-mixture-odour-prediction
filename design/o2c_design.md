# O2-C (centred only): keep length through the network, split direction / length only in the distance

2026-11-18. Written before any O2-C run. User's decision after the keep-length check and the O1.2b owner's review.

## Why

- O2-N's per-molecule unit scaling costs r −0.092 [−0.186, −0.007] on share-no-molecule pairs (O2N-out vs O2-out,
  paired, standard CV on all 820 pairs, run before rule 7).
- On frozen O2-N, keeping the merged length into the decoder: all pairs −0.011, share no molecule +0.104 / +0.131
  (direction only / + read-out length) vs direction in (all 820 pairs; the frozen decoder never saw short inputs).
- Rule learned: keep direction and length through the network; split them only at the final distance
  √(|u_A − u_B|² + β (h_A − h_B)²) (O1.2b form). Angle alone for "like A or B" questions (Ma).

## Network

e(m) 400 -> centre + scale per unit (BatchNorm, no affine) -> v(m) (own length kept) -> z(A) = Σ G v / Σ G (length
kept) -> decoder (open-POM FFN) -> 138 words. Agreement ρ(A) = |z(A)| / (Σ G |v| / Σ G): 1 for one molecule.
Note: trained on single molecules only, O2-C equals O2 up to parametrisation (per-unit affine map is absorbed by the
decoder's first linear layer and commutes with the mean); the backbone matters once mixtures are trained.

## Steps (in order; each only if the one before passes)

1. O2-C-Single: 10 members (seeds 0-9), O2 recipe and molecules. Fit check: held-out 20% AUROC.
2. Gate: O2C-out vs O2-out, paired, 50 × 5 on the development pairs (CV.md rule 7). Pass = the paired r change CI
   includes 0 or is above it, on all pairs and on share no molecule.
3. If passed: O2C-in, direction-only distance; O2C-in + read-out length (O1.2b); O2C-in + agreement (log ρ).
   Change from B1a++ and from O2C-out, same development folds.
4. If 3 looks good: O2-C-Single+Scent+Ma+DREAM (α = 0, direction + length), with validation lines.

Code: mixdist/o2.py (O2C), training/train_o2c.py, evaluation/O2/eval_o2c.py.

## BatchNorm fix (2026-11-18, before the stage-2 rerun)

The first stage-2 runs (checkpoints/*_bnbug, results/train_o2d_dev/2026-11-18_190615) normalised mixtures by the
statistics of their own batch while training, but by the running (mostly single-molecule) statistics when scored. One
member: inner-train r 0.97 with batch statistics vs 0.61 as scored; predictions 0.12 too low (RMSE +0.05-0.09 vs B1a++
in every arm). Those runs are not used. Fix (mixdist/o2j.py single_stats): every mixture pass (DREAM, GoodScents, Ma)
uses the single-molecule BatchNorm statistics and does not update them. Dropout stays on; the remaining offset (about
-0.04) is removed by refitting the line a, c on the inner-training pairs as scored (Tracker), as the frozen and "out"
rows get. Earlier O2-N mixture runs (O2-D, O2-J step 1, o2n_a*) had the same flaw.
New report rows: "out" on the same trained weights (mean of decoded DREAM molecules, B1a distance, line on the fold's
training pairs) and the paired in-minus-out change.
