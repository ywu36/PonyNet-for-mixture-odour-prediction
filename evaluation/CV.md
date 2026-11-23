# Cross-validation protocol (from 2026-11-17)

Training pairs only (820). V50 and T46 are never used to design or tune the CV.

## Rules

1. **Hold out whole mixtures.** In each split, 1/5 of the mixtures are held out. A pair is
   *held out* if both its mixtures are held out; *training* if neither is; *unused* if exactly
   one is (dropped in that split). No mixture is ever on both sides. (`mixdist/cv.py`, unchanged.)
2. **50 repeats × 5 folds.** 50 is the smallest of 10 / 20 / 50 at which every pair is held out
   at least once (10 repeats left 82 pairs never held out). Seed 0. Not tuned further.
3. **Every pair counts once.** A pair held out t times gets weight 1/t on each of its t
   predictions. Without this, pairs whose mixtures appear in only one pair (held out every
   repeat) dominate the score.
4. **CI by resampling mixtures.** Pairs that share a mixture are not independent, so the 95% CI
   resamples mixtures with replacement and weights a pair by count(A) × count(B).
5. **Report** r and RMSE on all pairs and on pairs sharing no molecule; every other model as a
   change from B1a on the same folds, with the paired CI. No study is reported on its own.
6. **Inner choices** (e.g. a ridge penalty) are made inside the training side of each split.

Code: `mixdist/evaluate.py` — `cross_validate(..., repeats=CV_REPEATS)`, then `score_cv`.

## Why it changed

The earlier protocol (10 repeats, every prediction counted, bootstrap over predictions) counted a
pair held out 10 times as 10 independent pairs: CIs were too narrow, 86% of the score came from
half of the pairs, and 82 pairs were never held out (findings/cv_audit.py, finding 3 section 4f).
Results produced before 2026-11-17 use the earlier protocol.

## Rule 7: DREAM lockbox (fixed 2026-11-18, before any model was scored on it)

7. **One fixed lockbox inside the 820 training pairs.** `cv.lockbox_split(pairs)`:
   `mixture_holdout_folds(k=4, repeats=1, seed=20261118)[0]`. That means 1/4 of mixtures, held out by rules 1 and 4's component logic.
   The fraction and seed were chosen once, from sizes alone, before anything was scored.
   Split file: `features/dream_lockbox.json`.

   | part | pairs | share no molecule | mixtures |
   |---|---|---|---|
   | lockbox (both mixtures held out) | 129 | 24 | 195 |
   | development (no lockbox mixture) | 514 | 186 | 599 |
   | unused (one lockbox mixture) | 177 | 115 | n/a |

   - **Development.** All work from now on (O2 family: training on DREAM, head fitting, choosing between
     variants) uses only the development pairs, with rules 1–6 (50 × 5, seed 0). One development pair is never
     held out.
   - **Lockbox.** It is scored only after development is finished. Each candidate on a list fixed beforehand is
     trained or fitted on all 514 development pairs and scored once on the lockbox. Report all pairs plus
     share no molecule, as a change from B1a++, with a mixture-resampling CI.
   - **Order.** No V50 or T46 scoring until after the lockbox.
   - **Not split by study.** The study mix is given for disclosure only. Lockbox: Bushdid 56, Ravia 4 18,
     Ravia 5 21, Ravia 6 12, Snitz 2 10, Ravia 2 6, Snitz 1 4, Ravia 1 1, Metamer 1.
   - **Limits.**
     - Share no molecule has only 24 lockbox pairs, so the CI on that subset is wide. The all-pairs number is the
       main lockbox result; V50 and T46 remain the setting-1 test.
     - Choices made before 2026-11-18 (B1a++, O1.2b, θ groups, the O2 backbone CVs) used all 820 pairs, so they
       are not clean on the lockbox. B1a++ is refitted on development pairs and serves as the reference only.
