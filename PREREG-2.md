# PREREG-2.md — addendum pre-registration

**Status:** SIGNED OFF by the user on 2026-10-05. Committed before any data for these tests were generated, and before Stage 3, so before any PCI lattice of any model had been seen.

**The user's answers (2026-10-05):**
- **A:** agrees with A1–A4.
- **B:** agrees with B1–B4 (B3 stays labelled exploratory, as drafted).
- **C:** no prediction, so C1–C3 are **exploratory**: reported with CIs, not scored.
- **D:** no prediction, so D1 is **exploratory**: reported with CIs, not scored.

**Relationship to PREDICTIONS.md:**
- PREDICTIONS.md is unchanged, and its results are scored as registered.
- This file adds new tests. They reuse the frozen instruments: the perturbation engine, ε* rule, θ grid, LZ76/PCI, bootstrap scheme and seeds in `src/rgc/config.py`. The only new constants are listed below.
- **Rules:** the same as PREDICTIONS.md. Nothing here changes after data are seen. Any change is logged in DECISIONS.md as *post hoc*. Each sub-criterion is reported separately.
- **What has already been seen:** M1 v1 results for step 143000 (Stage 2), and M1 for the control variants (which run before this file is committed). No M2/PCI result for any model.

---

## A. Anaesthesia analogue: does PCI track differentiated integration, or just reach?

**Motivation.** Human PCI is low both in deep sleep and anaesthesia (responses stay local and stereotyped) and in generalized seizures (responses spread widely but uniformly). PCI earns its meaning by falling in both cases while ordinary activity persists. In a transformer:
- Cutting attention range limits *reach*, so PCI must fall by construction. That makes it a **positive control**, not a validation.
- Flattening attention keeps reach (everything still attends to everything) but makes every query average its context the same way. That removes *differentiation*. **This is the informative test.**

**Model:** Pythia-160M step 143000.

**Manipulations:** applied at `hook_attn_scores` in **all 12 blocks**, to the clean and perturbed runs alike (`src/rgc/attention_manip.py`). Unmasked scores are multiplied by β, and masked entries stay masked.
- **Flattening (inverse temperature β):** β ∈ {1 (intact), 0.5, 0.25, 0}. β = 0 gives uniform causal attention.
- **Local window w** (keys within the last w positions, plus BOS so the attention sink survives): w ∈ {∞ (intact), 64, 16, 4}.

**Protocol:** exactly the M2 protocol of PREDICTIONS.md for each condition:
- its own ε* (calibration rule unchanged);
- 64 passages × 12 values of k × 3 values of p × 8 directions;
- θ_primary = 0.1, the same fallback rule, and the primary PCI as defined there.

**Measures per condition:**
- primary PCI;
- **reach** = mean p₁ (fraction of downstream sites with G ≥ θ_primary) over the same lattices;
- participation spread;
- **local response** = mean self-column gain at the final slice (the perturbed position's own response);
- next-token cross-entropy on the passages, as a descriptive manipulation check only (it is an output measure).

**All contrasts:** paired crossed bootstrap over passages and directions (10,000 resamples), each condition vs. intact.

**Predictions** (confirmed by the user; each scored separately):
- **A1 (positive control):** PCI falls monotonically as the window shrinks: PCI(∞) > PCI(64) > PCI(16) > PCI(4), with each adjacent paired difference's CI excluding 0.
- **A2 (key test):** PCI(β = 0) < PCI(intact), with the CI excluding 0, **and** PCI falls proportionally more than reach: PCI(β=0)/PCI(intact) < reach(β=0)/reach(intact), with the bootstrap CI of the difference of the two ratios excluding 0.
- **A3 (dose-response):** PCI(1) ≥ PCI(0.5) ≥ PCI(0.25) ≥ PCI(0) (point estimates), with PCI(1) − PCI(0.25) > 0 (CI excluding 0).
- **A4 (local activity persists):** at β = 0, the local response is at least 50% of intact.

---

## B. Symmetry breaking under conflicting and graded cues (M1 v2, new prompts)

**Motivation.** M1 v1 applied a strong external "field" (an unambiguous cue), which measures the response to a field. A sharper symmetry-breaking test asks how the representation at the ambiguous word behaves when cues are weak or conflict:
- **Averaging:** the representation sits in between.
- **Commitment:** it snaps to one reading.

In the language of physics, we look for **saturation** of the response to the field, where a single cue already moves the word as far as two cues do, and for decisiveness under conflict.

**Stimuli (to be written after sign-off, before any run):**
- **Words:** the same 10 words.
- **Frames:** 16 **new** frames, each with two cue slots before the target (e.g. "Between {c1} and {c2}, my cousin could only think about the"). Frames 1–8 are train and 9–16 are test.
- **Cue phrases:** per word and sense, 8 **new** cue noun phrases, plus 8 sense-neutral filler phrases shared across words. None of the cue content words may appear in the v1 stimuli for that word; this is checked programmatically.
- **Field conditions** per frame, h = (#A cues − #B cues):
  - h = +2: (A,A)
  - h = −2: (B,B)
  - h = +1: (A,N) and (N,A)
  - h = −1: (B,N) and (N,B)
  - h = 0, conflict: (A,B) and (B,A)
  - h = 0, none: (N,N)
- **Totals:** 9 conditions × 16 frames = 144 prompts per word, 1,440 in all, each with its `" thing"` control twin. Cue phrases rotate across frames by a fixed Latin-square assignment.

**Order parameter:**
- At each slice L, fit the sense direction on train-frame (A,A) vs (B,B) prompts.
- On test frames, compute normalised position t = (s − m_BB)/(m_AA − m_BB), using test-frame (A,A)/(B,B) means. So t = 1 means "reads as A" and t = 0 means "reads as B".
- This is done separately for target and control prompts. The pooled value is the mean over words.

**Quantities per slice:**
- **Saturation ratio** R(L) = [t̄(+1) − t̄(−1)] / [t̄(+2) − t̄(−2)]:
  - ≈ 0.5 means linear (each cue adds the same amount);
  - → 1 means saturated (one cue already commits).
  - Reported only where the denominator's CI excludes 0.
- **Conflict decisiveness** D(L) = mean |t − 0.5| over h = 0 conflict prompts, minus the same for h = 0 none prompts. The second term corrects for prior bias, which the neutral prompts revealed.
- **Bimodality coefficient** BC(L) of t over the conflict prompts, pooled over words after per-word standardisation (BC > 0.555 suggests bimodality).
- **Recency** ρ(L) = t̄(A,B) − t̄(B,A): negative means the later cue wins.
- **CIs:** bootstrap over frames within word (10,000 resamples).

**Predictions** (confirmed by the user):
- **B1 (saturation):** in the trained model, the maximum over slices of R_target(L) − R_control(L) is > 0, with the CI excluding 0. The ambiguous word saturates more than the neutral control token.
- **B2 (decisiveness):** at some slice, D_target − D_control > 0 with the CI excluding 0. That slice is identified from the pooled curve, and its CI is taken at that fixed slice.
- **B3 (bimodality):** BC_target > 0.555 at some slice. **Exploratory:** BC has low power with 16 conflict prompts per word.
- **B4 (controls):** for step 0 and each shuffle seed, the B1 and B2 differences have CIs that include 0.
- **Recency:** exploratory; no prediction.

---

## C. Transformer vs. recurrent architecture with matched training: Pythia vs. Mamba

**Motivation.** The paper holds that systems with stronger recurrent connectivity score higher on intrinsic causal-structure indicators. Mamba (`state-spaces/mamba-130m-hf`, `mamba-370m-hf`) is a recurrent state-space model trained on the **same Pile data with the same GPT-NeoX tokenizer** as Pythia. That makes it a matched comparison of architecture, not of data. Its recurrence runs over sequence positions, which is lateral recurrence, not feedback across depth. The write-up will say so.

**Models:**
- Mamba-130M (24 layers, d=768) vs. Pythia-160M (12 blocks, d=768);
- Mamba-370M (48 layers, d=1024) vs. Pythia-410M (24 blocks, d=1024);
- a weight-shuffled Mamba-130M (seeds 1001–1003) as its control.

**Running Mamba:** HF `MambaForCausalLM`, pure-PyTorch path, so no new CUDA packages.
- **Perturbation and reading:** PyTorch forward hooks on the residual stream entering each Mamba layer, plus the input to the final norm.
- **Clean baselines:** row-matched, as for Pythia.
- **Before any measurement**, the same instrument checks as Stage 0 must pass: ε=0 gives exactly zero, q<p gives exactly zero, and runs are bitwise reproducible.

**Depth matching:** a Mamba layer has about half a transformer block's parameters.
- **Primary:** Mamba lattices use **every second residual slice**, with injection at even layers, giving the same number of depth steps as the matched Pythia.
- **Secondary:** full resolution.

**Protocol:** otherwise identical to M2, including the ε* rule, θ_primary, and a primary PCI over k ≤ (depth steps − 3).

**Predictions:** none. The user chose to make no prediction. All of the following are reported with paired CIs and are **exploratory**:
- **C1:** PCI(Mamba-130M) − PCI(Pythia-160M).
- **C2:** the same for 370M vs. 410M.
- **C3:** PCI(Mamba-130M) − PCI of each shuffled Mamba-130M.

---

## D. Is z stable across scale ("universality")?

- **Models:** Pythia-410M and Pythia-1B at step 143000, with the M2 protocol unchanged (primary cell k=0, p=64, θ_primary).
- **D1 (exploratory; the user made no prediction):** is z the same across 160M, 410M and 1B? Operationally, the pairwise bootstrap CIs of the z differences all include 0, *and* each model separately meets the PREDICTIONS.md (d) stability criteria.
- **If (d) fails for 160M,** D1 is reported as not testable rather than as failed.

---

## Deferred to PREREG-3 (after engineering pilots on held-out passages, with no indicator values inspected)
- **A recurrent (autoregressive) perturbation protocol:** perturb at token t and track internal states across later generated tokens, with the loop closed vs. open (teacher-forced).
- **Coarse-graining / causal emergence:** Rosas et al.'s Ψ across grains (neurons → heads → layer → sense direction), using depth as time. This aims at the paper's question of which grain matters.

## New constants (in `src/rgc/config_prereg2.py`, so the PREDICTIONS.md config hash is unchanged)
- **Section A:** β grid (1, 0.5, 0.25, 0); w grid (∞, 64, 16, 4); local-response floor 0.5.
- **Section B:** 16 frames (8 train / 8 test); 8 cue phrases per sense; 8 neutral fillers; BC threshold 0.555.
- **Section C:** model IDs; depth stride 2; shuffle seeds 1001–1003.
- **Section D:** model IDs.
