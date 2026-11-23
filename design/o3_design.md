# O3: a lean O2 that avoids memorizing and focuses on the underlying interaction

2026-11-18. Design, written before any O3 run. Named by the user: "a compressed version of what we have", "a lean version of O2 to avoid memorizing and focus on the underlying interaction". Runs only
after O2's conclusions are reported (O2-C stage-2 rerun + evaluation/O2/eval_o2_all.py). Development pairs only
(evaluation/CV.md rule 7); lockbox untouched; V50 / T46 not scored.

## Why

O2 lets mixture data bend a ~446k-parameter decoder, plus a 401-parameter learned gain and a 141-parameter DREAM head,
while DREAM gives ~280 noisy pairs per fold over 231 molecules (median mixture size 10). That capacity can memorize
molecules (401 gain parameters > 231 molecules) and forgets single molecules (held-out AUROC 0.879 -> ~0.81). The
user's rules: no memorization, no forgetting, generalize from structure (memory: forbid-memorization), no synthetic
data. Prior work controls this freedom the same way: additive null + small deviation (drug synergy, Bliss/Loewe), one
shared low-rank interaction rule (factorization machines, Decagon's shared basis, SSI-DDI), pair parameters fitted on
binary mixtures and composed to many components (UNIFAC), mechanistic competition (Singh et al. 2019).

## Symbols

| symbol | meaning |
|---|---|
| p(m) | 138-word profile of molecule m, decoded alone by the frozen O2-C-Single members (mean of 10) |
| x_m | 24 RDKit group indicators of m (as B1a++) |
| θ | group gain, 24; G(m) = exp(θ·x_m) |
| s, r | rank-1 company suppression, 24 + 24: s_g = how strongly group g suppresses, r_h = how susceptible h is |
| G(m\|A) | context weight: log G(m\|A) = θ·x_m − (r·x_m) · mean_{m′∈A, m′≠m}(s·x_m′) |
| q | B1a's quantile map over molecules (fixed) |
| d | sqrt(\|u_A − u_B\|² + β (h_A − h_B)²), u = unit(q(profile)), h = log length |

## Ladder (each step a paired change from the one before; same folds)

| step | model | fitted on | new parameters |
|---|---|---|---|
| O3.0 | additive core: profile(A) = Σ G(m) p(m) / Σ G(m), G = exp(θ·x); distance d; line | DREAM (θ, β, a, c) | 27 |
| O3.1 | + rank-1 company suppression G(m\|A) | s, r on **GoodScents** (frozen network, learning to rank as O2-J), checked on Ma; DREAM refits only θ, β, line | 48 (none from DREAM) |
| O3.2 | + merge inside with a mixture-only correction: decoder(z) + (1 − ρ) U Vᵀ z, rank 2 | U, V on **GoodScents** | ~1.1k (none from DREAM) |

O3.1 runs only if O3.0 is reported; O3.2 only if O3.1 passes the acceptance rule.

## Acceptance rule (fixed now)

A step is kept only if all hold:
1. Paired r change CI > 0 vs the previous step (all pairs or share no molecule), on development pairs, 50 x 5.
2. Structure-shuffle control: refit with structures shuffled among DREAM molecules (compositions kept); the step's gain
   must disappear (a gain that survives shuffling is memorization).
3. Parts fitted on GoodScents / Ma improve their unseen-molecule splits (GoodScents held-out pairs, both molecules
   unseen; Ma molecule folds).
4. Single-molecule held-out AUROC unchanged (network frozen; checked).

## Reporting

Every row as change from B1a++ (RMSE and r, 95% CI, mixture resampling); subsets all pairs, share no molecule,
share 1-2, share >= 3 (input-defined). Parameter count per data source in every table. Also: the O2 free-trained arms
as the "fully free" reference row.

## Implementation (2026-11-18, written before any O3 run)

Code: mixdist/o3.py (O3 = B1aPlus with context weights; GoodScents fit), evaluation/O3/eval_o3.py --step 0 | 1.
Refinement fixed now: O3.0 uses **B1a++'s exact form** (quantile profile per molecule -> group-weighted mean -> unit
length -> Euclidean -> line + the two size terms) with profiles from the frozen O2-C-Single ensemble instead of the
released open-POM; so O3.0 vs B1a++ isolates the profile source, and each later step isolates one mechanism. DREAM
parameters: theta 24 + line 2 + size 2 = 28. (The direction + length form remains an O2 row; O1.2b on top of B1a++
added nothing.)
Unit checks passed: group_matrix reproduces DREAM's 24 groups exactly; O3 with s = r = 0 equals B1aPlus (max
difference 2e-16); a single molecule's weight is exp(theta . x) whatever s, r are.
GoodScents fit: equal weights / group gain / group gain + s, r, each with one temperature tau, L-BFGS on the mean rank
loss over 79 target words + 1e-3 L2; scored on held-out pairs (both molecules unseen). Ma: predicted log w1 - log w2
vs L, no refit, all mixtures and mixtures whose molecules are absent from GoodScents train pairs.
Shuffle control: 5 permutations of the DREAM molecules' (profile, groups), 10 x 5 folds each.

## Rebased on O1.2b (user, 2026-11-18 late; before any O3 result)

Reference model is now **O1.2b** (the best neuron model), not B1a++. O3.0 = **O1.2b's form** (gain-weighted mean of
quantile profiles, direction + length distance, theta 24 + beta + line = 27 numbers) with profiles from the frozen O2-C
ensemble; "O3.0 without group gain" = O1.2b length only. O3.1 adds the company term inside O1.2b's pooling.
Unit check: O3 with s = r = 0 equals O12b (max difference 4e-16). Every row is reported as change from O1.2b.

Earlier tries of related ideas (disclosed; O3.1 must be judged against them):
- O1.2c (finding 5d): a shared molecule's weight depending on company **size** (alpha + gamma log n): no gain.
- O1.2e (finding 5f): low-rank structure shift of the profile (q + x V U): lead at a lam picked after seeing; did not
  survive the clean test.
- O1.2k (finding 5l): kernel set distance: no gain over O1.2b.
- Finding 4: group x group competition tested pair by pair (DREAM 0/300, Ma 0/40 at q < 0.1); Ma suggested a molecule's
  weight is the same in every company at n = 2.
O3.1 differs: the company effect depends on the company's **chemistry** (rank-1 over groups, all pairs pooled into 48
numbers, not 300 separate tests), and it is fitted on **GoodScents**, not DREAM, so DREAM spends no parameters on it.
