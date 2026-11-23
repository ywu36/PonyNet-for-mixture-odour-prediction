# O2-N-Single+Scent+Ma (working name O2-J): joint training on single molecules + mixtures (GoodScents blends, Ma dominance)

Names (user, 2026-11-18): `<backbone>-<training data joined by +>`; Single = GS-LF single molecules, DREAM = DREAM
pairs trained in-fold, Scent = GoodScents blender pairs, Ma = Ma 2021. O2-D = O2-N-Single+DREAM; O2-J =
O2-N-Single+Scent+Ma (or O2-VAE-Single+Scent+Ma if the VAE wins the backbone rule).

2026-11-18. Plan, written before any O2-J run. The backbone (O2-VAE or O2-N) is chosen by the rule below once both
CVs report; everything else is fixed now. No DREAM labels in training (step 1); DREAM in-fold training is step 2.

## Backbone rule (fixed before the two CVs report)

Highest all-pairs r (standard CV) among the gain-1 merge-inside rows: "VAE-in mean" and "O2N-in". If their 95% CIs
overlap, take O2-N when its r is higher, else the VAE (O2-N keeps open-POM's single-molecule fit; the VAE costs 0.01
ensemble AUROC). O2 (deterministic, raw merge) is out: O2-in r 0.325 vs B1a++ 0.459.

## Symbols

| symbol | meaning |
|---|---|
| e(m) | encoder output (rebuilt open-POM), 400 |
| v(m) | the backbone's merge-space vector of m: O2-N unit direction (400), VAE μ (32) |
| g(m) | gain logit = linear(e(m)), new, starts at 0 (G = 1); G(m) = exp(g(m)) |
| z(A) | Σ G v / Σ G (VAE: weighted mean of μ; O2-N: then direction to the decoder, ρ = agreement) |
| f | the backbone's decoder -> 138 logits |
| T | the 79 open-POM words that GoodScents blend groups map to (features/o2j_data.npz) |

## Three streams, one update per step

| stream | per step | loss | weight |
|---|---|---|---|
| single molecules (GS-LF, 3,975 O2 training molecules) | 128 | open-POM's word loss (+ KL for the VAE), as the backbone was trained | 1 |
| GoodScents blender pairs (134,791 train pairs, molecule-disjoint from 1,487 held-out pairs) | 128 pairs | learning to rank: −Σ_{w∈T} p_w log softmax_{T}(f(z(m1,m2)))_w, p = the pair's group counts normalised; words outside T get no feedback, nothing is forced to 0 | 1 |
| Ma (198 two-molecule mixtures, 72 molecules) | 32 mixtures | Siamese: Δs = cos(z(m1,m2), v(m1)) − cos(z(m1,m2), v(m2)) (VAE: −|z − μ|); L̂ = a Δs + c (a, c learned); loss = (L̂ − L)² / var(L) | 1 |

Recipe otherwise as the backbone: 62 epochs of 32 single-molecule batches (1,984 steps), Adam 1e-3 halved every 640
steps, dropout 0.12, 1e-5 (L1 + L2). Initialised from scratch (seeds 0-9), not from the trained backbone, so all
three streams shape the encoder from the start. Face-recognition lessons (finding 5c): direction + length; σ only in
pooling; near-white mixtures keep full weight.

## Checks (all reported)

1. Single-molecule fit: held-out 20% GS-LF AUROC (backbone: O2-N 0.878 single / 0.892 ensemble; VAE 0.870 / 0.883).
2. GoodScents held-out pairs (both molecules unseen): mean AUROC of the listed words among T, for (i) O2-J merged,
   (ii) the untrained backbone merged, (iii) mean of the decoded single molecules (does merging add anything beyond
   the singles' own notes?).
3. Ma on unseen molecules: 5 molecule folds (features/o2j_data.npz ma_fold); a mixture is test in fold k if either
   molecule is in fold k, train mixtures have no fold-k molecule; 1 member per fold trained without the fold's Ma
   mixtures (GoodScents and singles unchanged). r(L̂, L) pooled; references: DREAM group gains no refit r +0.27,
   Ma's own seen-molecule score r +0.34, zero.
4. DREAM standard CV (50 × 5), network frozen (10 members), change from B1a++ and from the untrained backbone on the
   same folds, all pairs and share no molecule:
   - O2J-in (learned gain G, line)
   - O2J-in + length (O2-N: agreement ρ; VAE: read-out length), β, line
   - O2J-in + group gain θ on top + length, θ, β, line
   plus the read-out check by mixture size.

## Step 2: DREAM pairs as a stream, inside the CV

5 folds × 1 repeat × 3 members; per fold the fold's training DREAM pairs join as a stream (distance head, MSE);
compared with step 1 and B1a++ on the same 5 folds.

### O2-N-Single+DREAM (working name O2-D; run first, at the user's request, 2026-11-18)

To see what training the network on DREAM does when single molecules anchor it (other teams fine-tune on DREAM;
finding 2: MIST's fine-tune gained on seen mixtures only). O2-N architecture, from scratch per fold; streams: single
molecules (open-POM loss, 128 per step) + 64 DREAM training pairs per step: all 235 DREAM molecules encoded, each
mixture merged (gain G = exp(linear(e)), O2-N normalization), decoded; u = unit(logits − offset), h = log ρ,
d = sqrt(|uA − uB|² + b² (hA − hB)²), prediction a d + c; loss MSE / var(y_train). 62 epochs (1,984 steps), recipe as
O2. Folds: cv.mixture_holdout_folds(k=5, repeats=1, seed=0); 3 members (seeds 0-2), ensemble mean. Scored vs B1a++
on the same 5 folds (paired, mixture resampling); single-molecule held-out AUROC reported for every member.
Code: training/train_o2d.py, mixdist/o2j.py (DreamStream, train_dream).
