# O2 (trained on single molecules: O2-Single): open-POM with the mixture pooled inside the network

2026-11-18. Design, written before any O2 run. Step 1 only (training on single molecules + standard CV);
later steps (Ma survival, masked-molecule pretraining) are not part of this file.

## Idea

Every model so far reads molecules out to 138 odour words first and pools afterwards (B1a, O1.2b), or bolts a
pooling layer on after a frozen encoder (MIST, POMMix). O2 pools **inside** open-POM: the encoder (graph network)
gives one vector per molecule; the mixture is pooled there; the rest of open-POM reads the pooled vector out.
A single molecule is a set of size 1, so O2 at n = 1 is open-POM.

What O2 takes from O1.2b (findings 4, 5): one gain per molecule, one shared divisor (global competition only),
distance = direction + length on the read-out profile.

## Symbols

| symbol | meaning | size |
|---|---|---|
| A, B | the two mixtures of a pair; n_A = molecules in A | |
| m | a molecule | |
| e(m) | encoder output of m: open-POM's message passing + set2set (atoms -> molecule) | 400 |
| G(m) | gain of m (> 0) | 1 |
| z(A) | pooled vector of A = Σ_m G(m) e(m) / Σ_m G(m) | 400 |
| f | read-out: open-POM's feed-forward part, 400 -> 392 -> 392 -> 256 -> 138 words (sigmoid) | |
| r(A) | read-out profile of A = f(z(A)), mean over the R = 10 ensemble members | 138 |
| q(.) | B1a's quantile map per word (fitted on the 235 DREAM molecules' single-molecule r, applied to any profile) | 138 |
| u(A), h(A) | direction q(r(A)) / ‖q(r(A))‖ and length log ‖q(r(A))‖ | 138, 1 |
| x(m) | the 24 RDKit groups of O1.2 / B1a++ | 24 |
| θ, β, a, c | gain weights, length weight, output line, as O1.2b | |

## Network

```
 molecule m ── atoms/bonds ──► message passing (5 steps) ──► set2set ──► e(m) (400)      ENCODER (open-POM)
                                                                          │
 mixture A = {m1..mn} ───────────────────────────────────────────────────►│ z(A) = Σ G e / Σ G      SET LAYER
                                                                          ▼
                                       392 ─► 392 ─► 256 ─► 138 words ─► r(A)                    READ-OUT (open-POM)
                                                                          ▼
                    q per word ─► u(A) = direction, h(A) = log length ─► d(A,B) = sqrt(|uA-uB|^2 + β (hA-hB)^2) ─► a d + c
```

Pooling point: after set2set (open-POM's own atom -> molecule pooling), before the feed-forward part (called L1 in
the discussion). Two nonlinear layers + the word layer read the pooled vector out.

## Step 1: training (single molecules only)

- Code: open-POM rebuilt in plain PyTorch (`mixdist/o2.py`) so it trains on GPU (the openpom env's DGL is CPU-only);
  molecule graphs from open-POM's own featurizer (`evaluation/O2/export_o2_graphs.py`, openpom env).
- Data: open-POM's training file (curated GS-LF, 4,983 molecules, 138 words); no DREAM, Ma or V50/T46 labels.
- Recipe = open-POM's (`examples/ensemble_cv.ipynb`): batch 128, 62 epochs, Adam lr 1e-3 halved every 640 steps
  (staircase), dropout 0.12 in the feed-forward part, loss = open-POM's class-weighted loss (weights log(1 + count /
  max count); cross-entropy over [1 − z, z] on the logits z, as in their code), + 1e-5 × (L1 + L2 norm of all
  non-bias weights); BatchNorm after each hidden feed-forward layer (open-POM's default). 10 members, seeds 0-9,
  trained on 80% of GS-LF (3,975; iterative stratification), as the released ensemble (its checkpoints stop at
  step 1,984 = 62 epochs × 32 batches, i.e. its 80% split of 3,989); the other 20% (1,008) is held out for audit 2.
- G(m) is not trained in step 1: with one molecule per set it cancels. Set to 1 at training.

## Audits (before any DREAM score)

1. Rebuild = open-POM: the released 10 checkpoints loaded into `mixdist/o2.py` reproduce `features/openpom.npz`
   (probs and embeddings per member; max |difference| reported).
2. Recipe = open-POM: held-out 20% AUROC (mean over words) of each of our 10 members and of their ensemble, vs the
   released members on their own 20% split (0.873-0.881 single, ensemble_benchmark.ipynb).
3. Pooling layer at n = 1 equals the plain network (max |difference|).

## Step 1: evaluation (standard CV, 50 × 5, whole mixtures held out; V50 / T46 not scored)

Rows declared now; all reported. Change from **B1a++** (current best) on the same folds, RMSE / r [95% CI], all pairs
and pairs sharing no molecule; B1a and O1.2b as references.

| row | profile | gain G(m) | distance | fitted per fold |
|---|---|---|---|---|
| O2-out | mean of single-molecule r (pool at the output) | 1 | B1a (unit length, Euclidean) | line |
| O2-in | r(A) = f(pooled e) | 1 | B1a | line |
| O2-in + group gain + length | r(A) = f(pooled e) | exp(θ·x(m)), 24 groups | O1.2b | θ, β, line |
| O2-in + encoder gain + length | r(A) = f(pooled e) | exp(v·k(m)), k = first 24 principal components of e(m) (all 10 members stacked, 235 molecules, no labels), each scaled to sd 1 | O1.2b | v, β, line |

Implementation: the quantile map q of a mixture profile uses the molecules' fitted quantiles (sklearn's
QuantileTransformer formula, rebuilt in torch so θ / v get gradients; checked equal to sklearn). The read-out runs
in eval mode (BatchNorm running statistics, no dropout), float32 on GPU; the head in float64.

- O2-out vs B1a: does the retrained ensemble match the released one at the DREAM level?
- O2-in vs O2-out: the effect of pooling inside, nothing else changed.
- The last two: O2 with O1.2b's mechanisms; the encoder-gain row is the "everything from the encoder" version,
  capacity matched to the 24 groups. Penalty lam 1e-3 on θ and v (as O1.2b); β = b², b starts at 0.1.
- Read-out check (O1.4's): cos(q(r(A)) with pooling inside, mean of single-molecule q) and length vs n_A
  (1 to 43 molecules).

## Not in step 1

Mixture-level training of the read-out (Ma survival, masked-molecule pretraining on odor_pair, DREAM pairs),
learned sink token, log-n temperature, pooling at other depths. Each needs this step first.
