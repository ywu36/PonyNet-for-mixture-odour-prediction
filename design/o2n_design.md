# O2-N (trained on single molecules: O2-N-Single): O1.2b's normalization inside the merge layer

2026-11-18. Design, written before any O2-N run. Same rebuilt open-POM, data and recipe as O2 (design/o2_design.md);
single molecules only.

## Why

O2 merges raw encoder vectors e (400). They share a large common offset: merging 21-43 molecules keeps the norm
(|e| 4.3 single vs |z| 3.6 merged, member 0) while the spread after the decoder's first BatchNorm drops 5x (sd 1.05
-> 0.21). Disagreement among the molecules is invisible inside the network and every large mixture decodes to the
same confident "average molecule" (cos +0.89 with the common direction). Face-recognition template pooling and
O1.2b both say: centre, use unit directions, average, and keep direction and length apart. O2-N does that at the
merge layer instead of after the read-out.

## Network

```
 m ─► encoder (as O2) ─► e(m) 400 ─► centre + scale per unit (BatchNorm, no affine) ─► v(m) = unit vector     per molecule
 mixture A:  z(A) = Σ G(m) v(m) / Σ G(m)          ρ(A) = |z(A)|  (agreement: 1 for one molecule, < 1 when they differ)
 decoder input = √400 · z(A) / |z(A)|   (direction only, same scale as a single molecule)
 decoder = open-POM's feed-forward part 400 -> 392 -> 392 -> 256 -> 138 words
```

G = 1 during training (one molecule per set). The centring statistics are learned on single molecules (running
mean / variance, frozen at evaluation).

## Training and fit check

10 members (seeds 0-9), O2 training molecules (3,975), open-POM's recipe. Fit check: held-out 20% AUROC vs O2 (0.875-
0.883 single, 0.894 ensemble) and released open-POM (0.873-0.881). Per the user, close is enough; reported either way.

## Standard CV (50 × 5, whole mixtures held out; V50 / T46 not scored)

Rows declared now; all reported as change from B1a++ (RMSE / r [95% CI]), all pairs and share no molecule; B1a,
O1.2b, O2-in, VAE-in mean as references.

| row | profile | gain | length term | fitted per fold |
|---|---|---|---|---|
| O2N-out | mean of decoded single molecules | 1 | none (B1a distance) | line |
| O2N-in | decode(normalized merge) | 1 | none (B1a distance) | line |
| O2N-in + agreement | decode(normalized merge) | 1 | β (log ρ_A − log ρ_B)² | β, line |
| O2N-in + read-out length | decode(normalized merge) | 1 | β (h_A − h_B)², h = log length of the quantile-mapped profile (O1.2b) | β, line |
| O2N-in + group gain + agreement | decode(normalized merge) | exp(θ·x), 24 groups | agreement | θ, β, line |

lam 1e-3, b0 0.1 as O2. Checks: read-out by size (length, cos with mean of molecules, cos with common direction)
and ρ by size; spread after the first decoder BatchNorm, single vs merged (the O2 diagnostic).
