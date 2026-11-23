# O1.2 (revised 2026-11-18): gain-weighted pooling with divisive normalization

Design only; nothing in this file has been run. Replaces the first O1.2 (one dominance temperature per POM word,
`evaluation/O1/eval_o12.py`, now called **O1.2-word**: r −0.104 vs B1a on pairs sharing no molecule), whose two
assumptions the data contradict (dominance is per word; dominance comes from POM). Evidence for every choice below:
`findings/finding_4_notes.md` and finding 4 (`findings/finding_4.html`).

## Symbols

| symbol | meaning | size |
|---|---|---|
| A, B | the two mixtures of a pair | |
| m | a molecule; "m ∈ A" = the molecules of A | |
| n_A | number of molecules in A | |
| j | a receptor-proxy unit = one open-POM smell word | J = 138 |
| q(m, j) | quality of molecule m on unit j: open-POM probability, quantile-transformed per word (as B1a) | 138 per molecule |
| g | an RDKit group (fr_ functional-group counters, ≥ 5 of the 235 DREAM molecules, merged) | 24 |
| x_g(m) | 1 if m carries group g | 24 per molecule |
| θ_g | learned log-gain of group g | 24 |
| G(m) | gain of molecule m = exp(Σ_g θ_g x_g(m)) | 1 per molecule |
| p_j(A) | pooled drive of unit j in mixture A | 138 per mixture |
| s(A) | pooled strength of A = ‖p(A)‖ | 1 per mixture |
| κ | learned semi-saturation constant of the normalization (κ ≥ 0) | 1 |
| y_j(A) | normalized output of unit j | 138 per mixture |
| a, c | the output line (as B1a) | 2 |

Learned: 24 θ + κ + a + c = **27 numbers**. Frozen: open-POM, RDKit group detectors.

## Neuron layout

```
 INPUT                      LAYER 1 (frozen, per molecule)          LAYER 2 (learned, per molecule)
 mixture A = {m1, m2, ...}
   m1 ──► open-POM ──► quality units  q(m1, 1..138)  ─────────────────────────────┐
     └──► RDKit    ──► group detectors x(m1, 1..24) ──► gain unit  G(m1) = exp(θ·x(m1)) ──┐
   m2 ──► open-POM ──► q(m2, 1..138) ──────────────────────────────────────────────┤      │
     └──► RDKit    ──► x(m2, 1..24)  ──► gain unit  G(m2) = exp(θ·x(m2)) ──────────┼──┐   │
                                                                                     ▼  ▼   ▼
 LAYER 3 (computed, per mixture): 138 pooling units ("glomeruli")
        p_j(A) = Σ_m G(m) q(m, j)  /  Σ_m G(m)        one shared divisor for all 138 units
                                                     = global competition for one budget
 LAYER 4 (computed, per mixture): one normalization unit + 138 outputs
        s(A) = ‖p(A)‖                                pooled strength (shrinks as n grows)
        y_j(A) = p_j(A) / (κ + s(A))                 divisive normalization with semi-saturation κ
 LAYER 5 (comparison, per pair)
        d(A, B) = ‖y(A) − y(B)‖ ;   prediction = a · d(A, B) + c
```

Same weights for every molecule and for both mixtures of a pair. Three kinds of unit:

1. **Quality units (138 per molecule, frozen).** What the molecule smells like: open-POM's word profile, the
   receptor proxy also used by B1a, O1.1 and O1.2-word. Quality is the part POM already does well, so it stays frozen.
2. **Gain unit (1 per molecule, 24 learned weights).** How strongly the molecule's signal enters the mixture,
   whatever it smells like. Its input is chemical structure (the 24 group detectors), not POM, and its output is one
   number shared by all 138 units. Groups add inside the exponent (log G = Σ θ_g x_g).
3. **Pooling units (138 per mixture) and one normalization unit (per mixture).** Each pooling unit j sums the
   molecules' quality on j weighted by their gains and divides by the total gain (the same divisor for every unit).
   The normalization unit then divides the whole pattern by κ + its own strength.

Biological reading (structure only, no values taken from biology): quality units ≈ the receptor-neuron pattern a
molecule evokes; gain ≈ how strongly the molecule drives those neurons overall; pooling ≈ convergence onto glomeruli;
the shared divisor and the normalization unit ≈ global inhibition across glomeruli (divisive normalization).

## Why each piece is there (what we learned → where it sits)

| learned (finding 4, Ma) | design element |
|---|---|
| Dominance is one weight per molecule, stable against any partner | one **gain unit** per molecule, not per word or per pair |
| Dominance is not in POM (~9%; 0 of 138 words; 138 per-word weights null) | gain is fed by **structure (RDKit groups)**, not by q; quality stays frozen POM |
| Classes too coarse; groups carry the signal; Morgan fragments unlearnable | gain inputs = the **24 RDKit groups** |
| Groups add inside a molecule (0 of 54 within-molecule pairs) | **log G = Σ θ_g x_g**, no group×group terms |
| No pair-specific effects (0 of 8, 0 of 40, 0 of 300) | the only interaction between molecules is the **shared divisor** Σ G |
| Similarity of the two molecules does not change suppression or the winner (Ma and DREAM group graph) | the divisor is **global** (same for all 138 units), not per receptor; no similarity term |
| Every molecule loses something even at n = 2 (Ma ×0.86); suppression is generic | **normalization unit** dilutes each molecule as the mixture grows |
| Size mismatch is the largest DREAM effect (+0.036 per unit \|log n_A − log n_B\|), total size ~0 | **κ**: the pooled strength s(A) falls with n (single molecules 12.9, 21–43 molecules 2.2; ρ −0.68 over 800 mixtures); B1a's length-1 step throws this away, κ > 0 keeps part of it, so a large ("whiter") mixture sits nearer the origin than a small one |
| Biology gives structure, not values | κ, θ learned on DREAM only; Ma used only as an outside check |

## Special cases (each a model we already have)

| setting | equals |
|---|---|
| θ = 0, κ = 0 | **B1a exactly** (mean → length 1 → Euclidean → line) |
| θ learned, κ = 0 | B1a+ groups (without the size covariates) |
| θ = 0, κ learned | B1a with size handled inside the model instead of as covariates |
| θ, κ learned | **O1.2** |

So every comparison is a change from B1a on the same folds, and each part can be switched off.

## Outputs beyond the distance (for checks)

- **Share of molecule m in mixture A:** G(m) / Σ G. For a two-molecule Ma mixture, log share(m1) − log share(m2) =
  θ·(x(m1) − x(m2)), directly comparable with Ma's L = log(v1 / v2) (no refit).
- **Whiteness of A:** κ / (κ + s(A)), the fraction of the pattern lost to normalization; should grow with n_A.

## Evaluation plan (declared now; not started)

Standard CV (50 × 5, whole mixtures held out), every row as a change from B1a on the same pairs, RMSE and r with
mixture-resampling CIs, all pairs and pairs sharing no molecule; θ penalty λ = 1e-3 (as B1a+), κ unpenalized,
initialized at 0. Rows, all reported:

1. B1a (baseline)
2. O1.2 κ only (θ = 0)
3. O1.2 θ only (κ = 0)
4. O1.2 θ + κ
5. reference: B1a+ groups + size (covariates), to see whether κ replaces the size covariates

Checks: (a) Ma, no refit: r of θ·(x(m1) − x(m2)) with L; (b) fitted κ and the whiteness of mixtures by n;
(c) whether rows 2 and 4 reproduce the size-mismatch effect (residual correlation with |log n_A − log n_B|).
V50/T46 only with an explicit go-ahead.

## Deliberately left out (tested, no signal at the size the data can see)

Pair-specific antagonism, group×group terms between molecules, within-molecule group pairs, similarity-driven
collaboration/counteraction, per-word dominance, a learned latent receptor layer. A latent receptor layer (the
original O1.0, R binding/efficacy units) can be put back in place of the frozen quality units later, keeping the gain
unit and the normalization unit unchanged.

## Result of the declared CV (2026-11-18, `evaluation/O1/eval_o12_gain.py`, results/eval_o12_gain/2026-11-18_013622)

Change from B1a (RMSE / r [95% CI]):

| model | all pairs (820) | share no molecule (325) | share 1, audit (126) |
|---|---|---|---|
| O1.2 κ only | +0.0002 / −0.003 | +0.0003 / −0.007 | +0.0005 / −0.007 |
| O1.2 θ only (gain unit) | −0.0010 / +0.017 [−0.018, +0.053] | −0.0025 / **+0.116 [+0.035, +0.213]** | −0.0022 / +0.096 [−0.048, +0.239] |
| O1.2 θ + κ | −0.0008 / +0.015 | −0.0021 / +0.111 [+0.027, +0.213] | −0.0018 / +0.091 |
| reference B1a+ groups + size | −0.0027 / +0.040 | −0.0059 / +0.188 [+0.053, +0.311] | −0.0024 / +0.063 |

- **Gain unit works:** θ alone (no size) gives +0.116 on pairs sharing no molecule, CI above 0; its θ predicts Ma's L at
  r +0.283 without refit.
- **κ fails as the size mechanism:** fitted on all pairs κ goes to ~0 from both starts (0.01 / 0.03; 0 / 0), so no
  mixture becomes "white" and the residual still tracks |log n_A − log n_B| (r +0.17, as B1a +0.18). In-sample grid:
  larger κ raises r on pairs sharing no molecule (0.131 at κ 0 → 0.203 at κ 64) but lowers r on all pairs
  (0.431 → 0.355): pairs with large overlap need the normalized direction, pairs with none need the profile length.
  One κ cannot serve both.
- **Revision proposed (not run):** keep both instead of trading them: d² = ‖u(A) − u(B)‖² + β (h(A) − h(B))², with
  u = p / ‖p‖ (B1a's direction) and h = log ‖p‖ (how specific vs "white" the mixture is; shrinks as n grows),
  β ≥ 0 learned (β = 0 gives B1a / O1.2 θ only). This replaces κ.

## O1.2b result (2026-11-18): direction + profile length replaces κ

`evaluation/O1/eval_o12b.py` (results/eval_o12b/2026-11-18_014324), change from B1a (RMSE / r [95% CI]):

| model | all pairs | share no molecule | share 1 (audit) |
|---|---|---|---|
| O1.2b length only | −0.0017 / +0.024 [−0.009, +0.057] | −0.0040 / +0.123 [−0.023, +0.246] | +0.0001 / −0.053 |
| **O1.2b θ + length** | **−0.0032 / +0.046 [+0.000, +0.090]** | **−0.0069 / +0.215 [+0.081, +0.345]** | −0.0019 / +0.050 |
| O1.2 θ only | −0.0010 / +0.017 | −0.0025 / +0.116 [+0.035, +0.213] | −0.0022 / +0.096 |
| B1a+ groups + size (B1a++) | −0.0027 / +0.040 | −0.0059 / +0.188 [+0.053, +0.311] | −0.0024 / +0.063 |

- β stable from two starts (length only 0.89; θ + length 1.09). The length term absorbs the size-mismatch effect:
  r(residual, |log n_A − log n_B|) 0.180 (B1a) → 0.027 (θ + length). r(log|p|, log n) = −0.70 over 800 mixtures.
- **On top of B1a++** (`evaluation/O1/eval_o12b_on_b1a2.py`, results/eval_o12b_on_b1a2/2026-11-18_014638): no gain
  (setting 1 Δr +0.007 [−0.030, +0.047], within-study −0.003; setting 2 −0.005; claim rule not met); β 0.59 [0.000, 6.0];
  the mismatch covariate halves (0.0357 → 0.0183) when length is present. Length and the size covariates carry the
  same information.
- Reading: O1.2b reproduces the size effect from inside the model (2 size covariates → 1 β, no counting of molecules
  needed), with group gain it is the best row against B1a, but it is not an extra signal beyond size.
- Share-1 thesis (whiteness hides the signal) **not supported**: length does not help share-1 pairs.

## Next: the metric side (proposed 2026-11-18, not run)

Observation: large mixtures shrink (‖p‖ 12.9 → 2.2) and turn toward a common "white" direction (cos +0.42 at 21–43
molecules; Weiss et al. 2012: ~30 components smell alike). O1.2b already splits a mixture into direction (what it
smells of) and length (how specific vs white). Face-recognition metric learning after ArcFace makes the same split and
offers ways to use it:

| idea (source) | in face recognition | here | extra numbers |
|---|---|---|---|
| angle instead of chord (ArcFace, CosFace) | identity = angle on the sphere | d uses φ = arccos(u_A·u_B) instead of ‖u_A − u_B‖ | 0 |
| norm = quality (MagFace 2021, AdaFace 2022) | feature norm tracks image quality / recognizability; AdaFace adapts the margin by norm | length h = specificity; already a coordinate in O1.2b | (done) |
| uncertainty-aware matching (Probabilistic Face Embeddings, Shi & Jain 2019) | each face a Gaussian (mean, variance); match by mutual likelihood score ‖μ_A − μ_B‖²/(σ_A² + σ_B²) + log(σ_A² + σ_B²) | mean = direction u, variance from whiteness σ² = exp(τ − 2h): two white mixtures are compared as uncertain, not as full-strength smells | τ, γ (weight of the log term) |
| hyperbolic embedding (Poincaré ball; hyperbolic metric learning, Hyp-UML 2023; odor space is hyperbolic: Zhou, Smith & Sharpee 2018; 2-D hyperbolic olfactory perceptual space: Medbouhi 2025 abstract) | norm = level in a hierarchy / certainty; generic near the centre, specific near the edge | radius ρ(A) = tanh(s ‖p(A)‖), x = ρ u, Poincaré distance: white mixtures sit near the centre and are close to each other whatever their angle; specific mixtures sit near the edge and differ by angle | s |
| factor out a shared component (centering; sub-centres) | remove what all samples share before comparing | split u into its coordinate along the common white direction w (from training folds) and the residual direction; compare residual angle + Δ along w + Δh | γ |

Proposed declared test (all with the group gain θ, standard CV, change from B1a; O1.2b and B1a++ as comparison rows;
share-1 as audit): M1 angle; M2 uncertainty (PFE-style); M3 hyperbolic; M4 white factor. ArcFace's margin itself is a
training trick for class separation; with distance ratings it would become a perceptual threshold d = max(0, φ − m),
kept as an optional M5.
