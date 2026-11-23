# O2-VAE (trained on single molecules: O2-VAE-Single): open-POM with a variational bottleneck; mixtures merged in the bottleneck

2026-11-18. Design, written before any O2-VAE run. Builds on O2 (design/o2_design.md): same rebuilt open-POM encoder,
same data, same training recipe; trained on single molecules only.

## Why

O2 merges molecules in a hidden layer whose decoder never saw merged points (O1.4 showed what that does: decoding is
near-linear or wrongly specific). A VAE's KL term fills the latent space as a smooth Gaussian, so averages of molecule
latents decode to plausible profiles without any mixture labels. Two predictions follow:
- averaging many dissimilar molecules moves toward the origin, which decodes to the most generic profile: mixture
  length should fall and the direction whiten with size by itself (finding 5: cos with the common direction +0.42 at
  21-43 molecules; length r −0.70 with log n);
- the encoder's uncertainty σ(m) gives a gain without groups: precision-weighted merging (product of Gaussians).

## Symbols

| symbol | meaning | size |
|---|---|---|
| e(m) | O2 encoder output (message passing + set2set) | 400 |
| μ(m), σ²(m) | bottleneck mean and variance: e -> 392 -> 392 -> (μ, log σ²) (BatchNorm, ReLU, dropout 0.12) | D |
| z | latent; training: z = μ + σ ε; evaluation: z = μ | D |
| decoder | z -> 392 -> 256 -> 138 words (BatchNorm, ReLU, dropout 0.12, as open-POM's last layers) | |
| KL | 0.5 Σ_d (μ² + σ² − log σ² − 1), mean over the batch | |
| β_KL | KL weight, warmed up linearly over the first 10 epochs | |
| P(m) | precision exp(−log σ²(m)), per dimension | D |

Loss = open-POM's word loss + β_KL · KL + 1e-5 (L1 + L2) of non-bias weights. Recipe otherwise as O2 (62 epochs,
batch 128, Adam 1e-3 halved every 640 steps).

## Merging a mixture A (inside each member, then decoded; profiles averaged over the 10 members)

- mean:       z(A) = Σ_m G(m) μ(m) / Σ_m G(m)
- precision:  z_d(A) = Σ_m G(m) P_d(m) μ_d(m) / Σ_m G(m) P_d(m)     (product of Gaussians when G = 1)

## Step A: choose D and β_KL on GS-LF only (no DREAM data)

Inner split: the O2 training molecules (80%, 3,975) split again 80 / 20 at random (seed 0); words with no positive
in the inner validation part are skipped. One member per configuration (seed 0): deterministic O2 (reference) and
VAE with D ∈ {32, 64} × β_KL ∈ {1e-3, 1e-2, 1e-1} (6). Rule, fixed now: among VAE configurations whose inner-validation
AUROC is within 0.005 of the deterministic O2's, take the largest β_KL (most regularised latent), then the smaller D.
If none qualifies, stop and report: the VAE does not fit as well as open-POM.

## Step B: gate (must match open-POM before any DREAM score)

10 members (seeds 0-9) of the chosen configuration on the O2 training molecules; held-out 20% (1,008) AUROC.
Gate: mean single-member AUROC ≥ 0.873 (lowest released member) and ensemble ≥ 0.889 (O2's 0.894 − 0.005).
If the gate fails, stop and report; no DREAM evaluation.

Outcome (2026-11-18): step A chose D 32, β_KL 1e-2 (inner AUROC 0.8505 vs det 0.8538). Step B: single mean 0.8702
(0.8671-0.8731), ensemble 0.8826 -> formally FAIL (0.003 below on single, 0.006 below on ensemble). The user decided the
fit is close enough ("fit ok is fine, no need to be exact"); step C runs with this disclosed.

## Step C: standard CV (50 × 5, whole mixtures held out; V50 / T46 not scored)

Rows declared now; all reported as change from B1a++ on the same folds (RMSE / r [95% CI]), all pairs and share no
molecule; B1a, O1.2b, O2-in as references.

| row | profile | gain | distance | fitted per fold |
|---|---|---|---|---|
| VAE-out | mean of the decoded single molecules | 1 | B1a | line |
| VAE-in mean | decode(mean merge) | 1 | B1a | line |
| VAE-in precision | decode(precision merge) | 1 (σ is the gain) | B1a | line |
| VAE-in precision + length | decode(precision merge) | 1 | O1.2b (direction + length) | β, line |
| VAE-in mean + group gain + length | decode(mean merge) | exp(θ·x), 24 groups | O1.2b | θ, β, line |

Checks: (1) read-out by mixture size: length of the quantile-mapped profile, cos with the mean of the molecules'
profiles, cos with the common direction over all 800 mixtures, and the latent norm |z(A)|; (2) the quantile map,
lam 1e-3, b0 0.1 as O2.
