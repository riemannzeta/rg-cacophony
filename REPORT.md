# REPORT — Output-free indicators in transformers: symmetry breaking, perturbational complexity, and the edge of chaos

*Michael Frank Martin, with Claude (Anthropic) as research assistant. October 2026. Code, data, pre-registrations and every deviation: https://github.com/riemannzeta/rg-cacophony*

## 1. Summary

**Motivation.** Chandaria, Seth, Shevlin, Shanahan, Rosas and colleagues (arXiv:2609.35618, *From cacophony to hierarchy*) argue that behavioural evidence about AI consciousness is confounded: a system trained to reproduce human output produces human-like signatures "whether or not the states they ordinarily signal are present". They call for operationalisable indicators at deeper levels of description. We tried to build two such indicators for transformers. Both are measured from the inside, without looking at outputs, and both are motivated by a renormalisation-group reading of the forward pass.

**Discipline.**
- **Pre-registration:** five pre-registrations, each committed and tagged before its data existed (`prereg`, `prereg2` … `prereg5`).
- **Controls:** untrained (step 0) and weight-shuffled networks throughout.
- **Determinism:** bitwise-deterministic runs.
- **Deviation log:** every deviation in `DECISIONS.md`.

**What held up:**
1. **A PCI analogue that behaves like clinical PCI under an "anaesthesia" manipulation.** Flattening attention lowers the perturbational complexity of the response by about a third, while the response's reach is preserved and its threshold-free spread more than triples. This was found exploratorily and then **replicated on fresh data with all four pre-registered predictions passing** (PREREG-3). In PREREG-5 the same fall appeared in all four models tested (Pythia-160M, 410M and 1B, and GPT-2 small; a pre-specified secondary measure, reported but not scored).
2. **An edge-of-chaos signature, in small models only.** The perturbation-growth exponent λ of trained Pythia-160M is ≈ 0 at its natural attention temperature, while every control is clearly positive (pre-registered S1: pass). Across temperatures, λ rises smoothly through zero almost exactly at β = 1 (exploratory, PREREG-4 T3). The confirmatory test (PREREG-5, §4.6) replicated the zero crossing in Pythia-160M on fresh text and in GPT-2 small, but **not** in Pythia-410M or 1B, where trained λ is positive at every temperature tested. The pre-registered headline rule (crossing in all four models) therefore fails.
3. **A word-sense signal beyond context leakage** that appears with training and is absent in controls (pre-registered (b): pass). Head ablations confirm its causal basis at the layer where it peaks (post hoc).

**What failed or was not testable:**
- The "awareness as a sharp mid-depth symmetry-breaking event" picture: no mid-depth snap, no low-entropy heads, and near-linear evidence integration under conflicting cues.
- A dynamic exponent z: attention is non-local, so there is no spreading front.
- PCI at the originally pre-registered threshold.
- Chimera-index predictions.
- Several temperature-shape predictions.

Two blind spots matter for anyone adopting these measures. **LZ-based PCI and the chimera index both rate random networks as "rich"** (clearly at 160M and 410M; in PREREG-5 the shuffled Pythia-1B and GPT-2 scored below their trained models), and only λ separated trained from random dynamics in every model, by a modest margin at 410M and 1B. **Recurrence did not raise PCI**: a recurrent state-space model (Mamba) trained on the same data scored about the same as Pythia.

## 2. Background and indicators

The Cacophony framework orders functional descriptions in five levels: behavioural, computational, intrinsic causal-structure, organismic, and organism–environment. Our two indicators sit at levels 2 and 3:

- **Measurement 1, "symmetry breaking" (level 2, computational).** Does an ambiguous word's representation commit to one reading at a narrow band of depth, carried by focused, cue-attending heads?
- **Measurement 2, perturbational complexity (level 3, intrinsic causal structure).** Inject a small perturbation into the residual stream at (layer k, position p). Record the relative response at every later layer and position (gain G). Binarise at a threshold θ, and compute normalised Lempel–Ziv complexity: a PCI analogue (Casali et al., 2013). The same response maps give a **growth exponent** λ: the per-layer log growth of the cross-position response, which is negative if perturbations die out, positive if they amplify, and ≈ 0 if poised.

## 3. Methods (short; details in `PLAN.md`, `PREDICTIONS.md`, `PREREG-2…5.md`)

- **Models:**
  - Pythia-160M at steps 0, 512, 4000 and 143000;
  - three weight-shuffled copies (every parameter tensor permuted within itself);
  - Pythia-410M and Pythia-1B;
  - Mamba-130M and Mamba-370M (same Pile data and tokenizer) and their shuffled copies;
  - GPT-2 small.
- **Loading and numerics:** TransformerLens 3.9 with unprocessed weights, float32 and deterministic kernels. Every variant's logits are checked against Hugging Face.
- **M1 stimuli:**
  - 10 single-token ambiguous words × 30 minimal-pair frames, with lexically disjoint train and test cue vocabularies;
  - a control twin with `" thing"` in place of the target, giving the order parameter Δd′ = d′(target) − d′(control);
  - 1,440 new two-cue prompts for the conflicting/graded-cue test.
- **M2 data:** 64 WikiText-103 passages × 12 injection layers × 3 positions × 8 directions. Three further fresh, never-before-analysed passage sets were used for PREREG-3, PREREG-4 and PREREG-5.
- **Linear regime:** float32 finite differences against a float64 reference, per condition.
- **Statistics:** crossed bootstrap over passages × directions (10,000 resamples). Paired comparisons use identical resamples.
- **Scoring:** mechanical, by `scripts/s4_compare.py`; output in `results/SCORECARD.md`.

## 4. Results

### 4.1 Measurement 1: word-sense commitment (Fig. 1)
- **Trained model:** Δd′ jumps at the very first block (to ≈ 1.15), plateaus (peak 1.36 [1.02, 2.01] at slice 6), and fades in the last two blocks as context spreads to every token.
- **Controls:**
  - untrained and shuffled: ≈ 0 everywhere (peaks 0.04–0.18, all CIs include 0);
  - step 512: ≈ 0;
  - step 4000: already ≈ 0.95.
- **Pre-registered (a): FAIL.**
  - The narrow band is block 0, not mid-depth.
  - The selected heads are not over-represented in the band (p = 0.08).
  - Contribution correlates with cue mass (ρ = 0.41, pass) but with *higher*, not lower, entropy (ρ = +0.41; it fails).
  - Ablation at the final layer is not better than random heads (E = 0.52; random-null p = 0.15).
- **Pre-registered (b): PASS.**
- **Post hoc:** ablation at the peak layer halves Δd′ (1.36 → 0.63) and beats both nulls (p = 0.005 and 0.015).
- **Conflicting cues (PREREG-2 B): B1 and B2 fail.** Their CIs were uninformative because two words have near-zero separation at slices 2–4 (D21). In the stable layers, the representation integrates cues almost linearly:
  - saturation ratio 0.56–0.59, versus 0.50 for linear;
  - conflicting cues cancel rather than force commitment;
  - no bimodality (BC ≤ 0.42).

### 4.2 Measurement 2: perturbational complexity (Figs. 2 and 3)
- **No spreading front.** One block after injection every later position is already affected, weakly (gain ≈ 0.01–0.03). At θ = 0.1 the number of positions reached is flat with depth, so z is undefined and **pre-registered (d) FAILS** (also at 410M and 1B; PREREG-2 D is not testable).
- **(c): PASS** at the pre-registered θ = 0.1. The fallback rule was not triggered (pooled degenerate fraction 0.76).
  - Trained − step 0: +0.89. Trained − shuffled: +0.29 to +0.32. The early checkpoints lie in between.
  - **Caveat:** at θ = 0.01, where maps are non-degenerate, shuffled networks score *higher* than trained (0.73–0.74 vs 0.66). LZ rewards randomness.
- **Anaesthesia.**
  - At θ = 0.1, A1–A3 fail: the maps are nearly empty (1.6% of sites on).
  - At θ = 0.01, the predicted dissociation appeared exploratorily and then **replicated on fresh data (PREREG-3: R1–R4 all PASS):**
    - PCI 0.648 → 0.424 under fully flattened attention (Δ = 0.225 [0.201, 0.247]);
    - reach 0.22 → 0.21;
    - threshold-free spread 39 → 151;
    - monotone dose-response;
    - window restriction lowers PCI step by step;
    - the density-matched PCI shows the same drop (+0.287 [0.278, 0.297]).
- **Integration–differentiation plane (Fig. 2):**
  - trained models sit at moderate reach and high PCI;
  - flattened attention keeps reach but loses PCI;
  - local windows lose reach;
  - untrained networks are low on both;
  - **shuffled Mamba is "seizure-like"**: reach 0.80–0.84 with low PCI.
- **Pythia vs Mamba (PREREG-2 C, exploratory):**
  - Mamba-130M ≈ Pythia-160M (+0.02 [−0.03, 0.06]);
  - Mamba-370M slightly below Pythia-410M (−0.06 [−0.10, −0.01]);
  - trained Mamba far above shuffled Mamba (+0.61 to +0.66).

### 4.3 Criticality and chimera states (PREREG-4; Figs. 4 and 5)
- **S1: PASS.** At the natural temperature, trained λ = −0.0005. Controls: step 0 +0.127; shuffled +0.066 to +0.071; steps 512 and 4000 +0.15 and +0.11.
- **T3 (exploratory; Fig. 4):** trained λ(β) = −0.121, −0.085, −0.037, **−0.0005**, +0.066, +0.126 for β = 0.25 … 2. The shuffled model stays positive at every β (0.068–0.107).
- **T1/T2: not testable as pre-registered.**
  - At β = 4 there is no linear regime (D23).
  - Post hoc, PCI rises from the disordered side (0.46 → 0.64) and peaks just above the natural temperature (0.71 at β = 1.5–2). It barely falls at β = 4 (0.65).
- **S2: FAIL.** The shuffled model's PCI is flat across temperature.
- **C1: FAIL.** The chimera index of the trained model is ≈ untrained, and below shuffled.
- **C2: FAIL.** χ decreases monotonically with β and anti-correlates with PCI (ρ = −0.94).
- **C3: PASS (borderline).** The effective-dimensionality "waist" is at slice 8 in the trained model (range 4–8) and at slice 11 untrained.

### 4.4 What λ ≈ 0 means: the growth curves (exploratory; Fig. 6)
- **Estimator sensitivity.** The crossing location depends on how λ is estimated:
  - the average of per-perturbation log-growth slopes (the pre-registered, Lyapunov-style estimator) crosses at β ≈ 1.0;
  - the slope of the averaged response curve crosses near β ≈ 1.6;
  - a late-layer-only slope (t ≥ 3) stays negative up to β = 2.
- **Mechanism.** In the trained model, an early build-up of cross-position influence is balanced by late-layer **contraction**. The controls show no late contraction. λ ≈ 0 is a balance of two regimes rather than uniform marginality.

### 4.5 Scorecard
See `results/SCORECARD.md`; it is generated mechanically.

### 4.6 PREREG-5: confirmatory test of the zero crossing across scale and family (Figs. 7 and 8)
- **Design:** 32 fresh passages × 4 directions. Trained models at β ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}; weight-shuffled copies (seed 1001) at β ∈ {0.25, 1, 2}; untrained step 0 at β = 1 (Pythia only). All 39 conditions completed; no condition needed the fallback ε (ε* ∈ {0.03, 0.1, 0.32}).
- **Scored results** (λ with crossed-bootstrap 95% CIs):

| model | λ(0.75) | λ(1) | λ(1.5) | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|---|---|
| Pythia-160M | −0.043 [−0.062, −0.023] | −0.007 | +0.055 [+0.029, +0.082] | PASS | PASS | PASS | PASS |
| Pythia-410M | +0.064 [+0.055, +0.072] | +0.073 | +0.089 [+0.080, +0.099] | FAIL | FAIL | PASS | PASS |
| Pythia-1B | +0.071 [+0.063, +0.079] | +0.090 | +0.126 [+0.115, +0.138] | FAIL | PASS | PASS | PASS |
| GPT-2 small | −0.015 [−0.027, −0.0003] | −0.012 | +0.017 [+0.004, +0.031] | PASS | FAIL | PASS | PASS |

- **Headline rule: FAILS.** P1 passes in Pythia-160M and GPT-2 small, and fails in Pythia-410M and 1B. The result is reported model by model.
- **P2 failures:**
  - Pythia-410M: λ(0.5) = +0.038 is just below λ(0.25) = +0.042, a near-tie.
  - GPT-2: the curve is U-shaped. λ(0.25) = +0.075, with a minimum near β = 0.75–1.
- **P3: PASS everywhere.** Every shuffled copy is positive at every β tested (+0.05 to +0.29), and every untrained Pythia model is positive at β = 1 (+0.085 to +0.126).
- **P4: PASS everywhere,** but the margin over step 0 is small at 410M (|λ| difference 0.013 [0.003, 0.022]) and at 1B (0.022 [0.012, 0.031]).
- **Secondary (pre-specified, not scored):**
  - **PCI at θ = 0.01 falls with flattened attention in every model** (Fig. 7). From β = 1 to 0.25: 0.648 → 0.457 (160M), 0.473 → 0.362 (410M), 0.468 → 0.359 (1B), 0.525 → 0.255 (GPT-2); the 95% CIs over passages do not overlap. Reach holds in Pythia (0.22–0.26) and rises in GPT-2 (0.21 → 0.56).
  - **Trained vs shuffled PCI at β = 1:** 0.648 vs 0.735 (160M), 0.473 vs 0.513 (410M), 0.468 vs 0.414 (1B), 0.525 vs 0.472 (GPT-2). Untrained: 0.20, 0.13 and 0.17.
  - **Late-layer λ (t ≥ 3) at β = 1:** −0.142 (160M), +0.033 (410M), +0.037 (1B), −0.079 (GPT-2). The late-layer contraction that balances early build-up (§4.4) is present in the two models whose λ crosses zero and absent at 410M and 1B.
  - **Averaged-curve λ** (slope of the passage- and direction-averaged response curve; `scripts/p4b_growth_curves.py fresh3`): at β = 1, −0.039 (160M), +0.069 (410M), +0.097 (1B), −0.003 (GPT-2). This estimator agrees that 410M and 1B amplify at every β (+0.02 to +0.15) and that GPT-2 sits near zero at its natural temperature. For Pythia-160M it again places the crossing above β = 1 (between 1.5 and 2), as in §4.4. Every control is positive (+0.04 to +0.27).
- **Reading:** λ ≈ 0 at the natural temperature is a property of the two small models tested (124M and 160M, from two families), not of trained transformers in general. We have not tested why the larger Pythia models amplify.

## 5. What this means for the Cacophony framework
1. **Level-3 perturbational indicators are well defined in transformers and measurable without outputs.** They inherit the strengths and limitations of their neuroscience counterparts.
   - They respond to within-system manipulations as clinical PCI does: anaesthesia-like flattening (in all four PREREG-5 models), and a seizure-like shuffled Mamba.
   - Like LZ-based measures in neuroscience, they do not by themselves separate structured from random dynamics. Clinical PCI avoids that problem because it compares states within one brain.
2. **Criticality discriminates in small models but is not yet a general indicator.** λ separated trained from untrained and shuffled networks where PCI and the chimera index did not, and it placed trained Pythia-160M at the order–chaos boundary of its own temperature family. PREREG-5 confirmed the crossing in a second family (GPT-2 small) and on fresh text, but not at 410M or 1B. There, trained models amplify at every temperature and sit only modestly closer to zero than their controls. The anaesthesia signature in PCI, by contrast, held in all four models.
3. **Recurrence did not help.** Mamba's lateral recurrence did not raise PCI. Our perturbation lattice treats depth as time, so a recurrent-generation protocol, in which perturbations are followed across generated tokens, remains an open test.
4. **The level-2 "symmetry breaking" picture did not survive.** Sense information arrives at the first attention layer, integrates cues nearly linearly, and is later diluted into context. This is graded field response, not a commitment event. It argues against reading ignition-like dynamics into transformer word processing, at least at this scale.

## 6. Caveats
- **Scope:** the core results are at 160M parameters; PREREG-5 extends to 410M, 1B and GPT-2 small. One text domain (Wikipedia) and synthetic manipulations throughout. "Anaesthesia" here means attention temperature, not a pharmacological analogue.
- **Thresholds:** θ = 0.01 was chosen after exploratory data, then confirmed on fresh data. The originally pre-registered θ = 0.1 results are reported as scored.
- **Interpretation:** λ ≈ 0 depends on the estimator (§4.4), reflects a balance of early build-up and late contraction, and does not hold at 410M or 1B (§4.6).
- **What is not claimed:** nothing here is evidence for or against consciousness. These are tests of whether candidate indicators are well defined, robust and discriminating.

## 7. Reproducing
- **Setup and checks:** `uv sync`; `uv run pytest` (about 60 instrument tests); `scripts/s0_env_check.py`.
- **Measurements:**
  - M1: `scripts/s2_m1.py`, with post-hoc analysis in `s2b_m1_posthoc.py` and the conflicting-cue test in `s2c_conflict.py`;
  - M2: `scripts/s3_m2.py` (see `--help` for the PREREG-specific options);
  - Mamba runs through the same script with `--mamba`.
- **Scoring, results and figures:** `scripts/s4_compare.py`, then `s5_results.py` and `s5_figures.py`.
- **Provenance:** every result file records the git commit, config hash, package versions and GPU. The raw perturbation lattices are attached to the GitHub release.

## 8. Engineering notes (bugs that would have corrupted results)
- **Silent checkpoint reload:** TransformerLens 3.9 silently re-downloads checkpoints when `checkpoint_value` is set. That would have turned every shuffled control back into the trained model (D4).
- **Runtime Triton kernels:** torch 2.14 routes some eager CUDA ops to Triton kernels compiled at run time. These were disabled in favour of the standard ATen kernels (D11).
- **Hidden float32 in Mamba:** HF Mamba computes parts of a float64 model in float32 (the residual cast, RMSNorm, and the scan), which corrupted the float64 linearity reference until fixed (D18).
- **Determinism checks:** every major re-run reproduced bitwise, including across two LZ76 implementations proven equivalent.
