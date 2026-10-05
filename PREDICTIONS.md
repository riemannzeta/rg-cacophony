# PREDICTIONS.md — pre-registration

**Status:** DRAFT, awaiting sign-off. It will be committed before any model comparison is run.

**Frozen inputs:**
- Stimuli at commit `ab041f6`: `stimuli/ambiguous.json` and `stimuli/wikitext_manifest.json` (sha256_all `57149fb6…`).
- Analysis constants in `src/rgc/config.py`. Their hash is recorded in every result file.

**Rules:**
- Nothing below changes after results are seen.
- Any change made later is logged in DECISIONS.md, marked *post hoc*, and the pre-registered version is reported alongside it.
- Each sub-criterion is scored and reported separately, and no multiple-comparison correction is applied.

**Models:**
- Pythia-160M at steps 0, 512, 4000 and 143000 ("trained").
- Three weight-shuffled copies of step 143000 (seeds 1001–1003).
- Precision and inputs: float32, unprocessed weights, BOS prepended.

---

## Measurement 1: symmetry breaking

### Definitions
- **Slices:** slice L ∈ {0…12}. For L ≤ 11 it is `blocks[L].ln1.hook_normalized` (LayerNorm output before the affine step). Slice 12 is `ln_final.hook_normalized`. Block L maps slice L to slice L+1. All slices are read at the target position.
- **Slice 0** is the target token's embedding, which is identical across all prompts. d′(0) ≡ 0, and its zero variance serves as a check.
- **Sense direction:** for each word w and slice L, v = mean(train, sense A) − mean(train, sense B), normalised to unit length.
- **d′:** project the test prompts onto v. Then d′ = (m_A − m_B) / √((s²_A + s²_B)/2), with sample variances. The sign is positive when sense A projects higher.
- **Control d′:** d′_ctrl is computed the same way on the control-token prompts, which are identical except that `" thing"` replaces the target. It has its own direction, fit on control-train.
- **Order parameter (primary):** Δd′(L) = d′_target(L) − d′_ctrl(L), averaged over the 10 words.
- **Secondary:** raw d′_target(L), d′_ctrl(L), and per-word curves.
- **CIs:** within each word, resample train pairs (20) and test pairs (10) with replacement, refit the directions, and recompute. 10,000 resamples; 95% percentile intervals. Resampling is by minimal pair, so both senses of a frame stay together.
- **Breaking block:**
  - J(L) = Δd′(L+1) − Δd′(L) for blocks L = 0…11, from the pooled curve.
  - L* = argmax J.
  - Total rise R = max_L Δd′(L).
  - The **band** is the shortest run of consecutive blocks whose summed J is at least 0.5·R. Its width is w.
- **Attention** at query = target position, keys 0…target (n = target_pos + 1):
  - normalised entropy H = −Σ a log a / log n (primary);
  - BOS-excluded entropy, with weights renormalised over keys 1…target and divided by log(n−1) (secondary);
  - **cue mass** M = Σ_{k ∈ cue tokens} a_k, plus M / (|cue| / n) relative to uniform attention (secondary).
- **Head contribution c_{L,h}:**
  - Take the head's output at the target, o = z_{L,h} W_O^{L,h} (raw residual space). Mean-centre it and divide by the LayerNorm scale of slice L+1 for that prompt. Project the result onto the slice-(L+1) sense direction.
  - c = (mean_A − mean_B of that projection) / (pooled SD of the full slice-(L+1) projection). This is the head's direct share of d′(L+1).
  - Computed on test prompts and averaged over words.
- **Selected heads:** the top **6** of 144 by mean cue mass over **train** prompts (both senses, all words). The set is fixed before any test-split statistic is computed.
- **Mean ablation:**
  - Replace each selected head's `hook_z` with its mean over all train target-prompt tokens at positions 1…target_pos (BOS excluded).
  - The replacement applies at positions ≥ 1; BOS is left intact.
  - Then **refit** the directions on ablated train and evaluate on ablated test, for both target and control prompts.
  - Effect E = 1 − Δd′_abl(12) / Δd′(12).
- **Nulls:**
  - 200 random 6-head sets drawn uniformly from 144 heads;
  - 200 layer-matched random sets: the same multiset of layers as the selected heads, with random heads within those layers, excluding the selected heads.

### Predictions (user's) and pass criteria
- **(a) Trained model:** d′ rises sharply over a narrow band of layers that coincides with low-entropy, high-cue-mass heads, and ablating those heads collapses d′. Scored as:
  - **a1** The band width w ≤ 3 blocks, *and* the 95% CI of peak Δd′ lies above 0.
  - **a2-i** The selected heads are over-represented in band blocks. The test is hypergeometric (population 144, 12·w successes, 6 draws), with p < 0.05.
  - **a2-ii** Spearman ρ(c, cue mass) > 0 across the 144 heads, permutation p < 0.05 (10,000 permutations).
  - **a2-iii** Spearman ρ(c, entropy) < 0 across the 144 heads, permutation p < 0.05.
  - **a3** E ≥ 0.5, *and* E exceeds the 95th percentile of both null distributions.
  - (a) passes only if a1–a3 all pass.
- **(b) Step 0 and each shuffle seed:** no such structure. For each model:
  - **b1** Either the 95% CI of peak Δd′ includes 0, *or* peak Δd′ is less than 0.2 × the trained model's point estimate.
  - **b2** The model's own top-6 cue-mass heads give an ablation effect E that does *not* exceed its random-null 95th percentile.
  - (b) passes only if b1 and b2 hold for step 0 and for all three shuffle seeds. Each model is reported separately.
- **Neutral prompts (user's expectation, secondary):** at slice 12 in the trained model, compute normalised positions t = (s − m_B)/(m_A − m_B) on the target direction. The prediction is that the 95% CI of mean t lies inside (0, 1). The fraction of neutral prompts with t ∈ (0, 1) is also reported.
- **Not a prediction:** steps 512 and 4000 are reported descriptively for M1.

### Claude's auxiliary expectation (from the plan, recorded so it can be checked)
- Cue-identity leakage is removed by the disjoint split.
- Target position still differs by up to 1.6 tokens between senses within a word, so raw d′_target at step 0 may be above 0 through length and position information. The control correction removes this.
- I make no confident directional claim here; raw d′ is reported for every model.

---

## Measurement 2: flow (PCI analogue)

### Definitions
- **Passages:** the 64 WikiText passages in the manifest. T = 257 positions, with BOS at 0.
- **Perturbation:** at slice-k resid_pre and position p, resid += ε·‖resid‖·u.
  - u is one of 8 fixed unit directions, mean-removed, seed 11. The same directions are used for every passage, k, p and model, so comparisons are paired.
  - k ∈ {0…11} and p ∈ {16, 64, 128}.
- **Response:**
  - r(L, q) = ‖Δresid‖/‖resid‖ for L = k…12 (slice 12 = final resid_post).
  - Gain G = r/ε.
  - Clean baselines are row-matched within the batch.
- **ε*, the linear regime (per model, chosen before computing PCI):**
  - Calibration uses passages 0–7, the first 4 directions, all k and p, and the full ε grid 1e-6…1 in half-decades.
  - The reference gain G_ref comes from a float64 finite difference at ε = 1e-6.
  - ε* is the largest grid ε at which the median of |G_f32(ε) − G_ref|/G_ref, over sites with G_ref ≥ 0.01, is ≤ 0.05.
  - Also reported: the 95th percentile of that deviation, and the agreement of binarised lattices at θ = 0.1.
- **PCI:**
  - Binarise G ≥ θ over slices k+1…12 and positions q > p. The self-column q = p is excluded and reported separately.
  - Flatten layer by layer, with positions in natural order.
  - PCI = c_LZ76(n)·log₂n / (n·H(p₁)). PCI = 0 when p₁ ∈ {0, 1}.
- **Threshold:**
  - θ_primary = **0.1**, on the gain.
  - Grid θ ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}, plus the area under PCI(log₁₀θ) over the grid as a threshold-free summary.
  - **Fallback rule** (label-blind, applied once, before comparisons): if at θ = 0.1, 90% or more of all lattices pooled over *all* models have p₁ < 0.01 or p₁ > 0.99, the primary θ moves to the nearest grid value at which that pooled fraction is below 90%. Any move is logged.
- **Primary PCI per model:** the mean over 64 passages × 8 directions × k ∈ {0…9} × 3 values of p, at ε*.
- **PCI CIs:** a crossed bootstrap that resamples passages and directions independently, 10,000 resamples. Model differences use a *paired* bootstrap, with the same resampled passages and directions applied to both models.
- **Spread and the dynamic exponent:**
  - Primary cell: k = 0, p = 64, so t = L for L = 1…12.
  - S(t) = #{q > p : G(t, q) ≥ θ_primary}, averaged over the 8 directions for each passage.
  - **Usable points:** 0 < S < 0.9·192. A passage needs at least 4 usable points to be fitted; otherwise its fit counts as a failure.
  - **Fits:** ordinary least squares in log space. Power law: log S = a + b·log t, with **z = 1/b**. Exponential: log S = a + t/τ.
  - **Model comparison:** AIC = n·log(RSS/n) + 4. ΔAIC = AIC_exp − AIC_pow, so positive values favour the power law.
  - **Pooled z:** from the passage-mean S(t), with a bootstrap over passages and directions.
  - **Secondary:** every other (k, p) cell, every θ, and the threshold-free participation spread S_eff(t) = (Σ_q G)²/Σ_q G².

### Predictions (user's) and pass criteria
- **(c) PCI:** clearly higher for trained than for step 0 and shuffled, with the early checkpoints in between.
  - **c1** PCI(143000) − PCI(step 0) > 0, *and* PCI(143000) − PCI(shuffled s) > 0 for each of the three seeds. Every paired 95% CI must exclude 0.
  - **c2** For each e ∈ {512, 4000}: PCI(e) − PCI(step 0) > 0 *and* PCI(143000) − PCI(e) > 0, with both paired CIs excluding 0.
- **(d) z is well defined, i.e. stable across prompts in the linear regime.** All of the following must hold for the trained model at the primary cell, θ_primary and ε*:
  - At least 80% of passages are fittable, with b > 0.
  - The coefficient of variation of per-passage z is ≤ 0.25.
  - The half-width of the 95% CI of pooled z is ≤ 15% of the estimate.
  - The median R² of the power-law fit is ≥ 0.9.
  - The winning functional form is reported separately. If the exponential wins by ΔAIC > 4 in most passages, the report will say "z is stable but a power law is disfavoured".
  - z is also reported for every model, each (k, p) cell and each θ. That is robustness reporting, not part of the criterion.

### Gating test (links M1 and M2)
- **Injection:** use L* from the **trained** model's M1, and apply it to every model.
  - **Before:** inject at slice k_b = L* − 2. The read window is slices k_b+1…k_b+3, which contains block L*.
  - **After:** inject at slice k_a = L* + 1. The window is k_a+1…k_a+3, which excludes block L*.
  - If a boundary is hit, both windows are shrunk to the same length (logged).
- **Inputs:** all 600 ambiguous sense prompts, injected at target_pos, at ε*.
  - Directions: (i) the 8 random directions; (ii) the M1 sense direction at slice k, mean-removed and normalised.
  - Positions read: the continuation, q > target_pos (28–29 tokens).
- **Metrics:** PCI over the window, and mean S over the window, at θ_primary. Contrast: after − before, with a bootstrap over prompt pairs and directions.
- **Prediction:** **[USER TO FILL: direction of after − before for PCI and for spread, or "exploratory"]**. If left blank, the gating test is labelled *exploratory* and no claim is scored.

---

## Sanity expectations (instrument checks, not hypotheses)
- d′(0) has zero variance for every model.
- ε = 0 gives zero response and q < p gives zero response, both bitwise (verified in Stage 0 on CPU; to be re-verified on GPU).
- Runs reproduce bitwise.
- The user's "step 0 shows no sense separation" is checked with **Δd′** (b1). Raw d′ is reported beside it.
