# PREREG-5.md — does λ cross zero at the natural temperature, across scale and model family?

**Status:** SIGNED OFF by the user on 2026-10-09 ("Approved!"): the draft is approved as written, with P1–P4 as the user's predictions, and the compute is approved. Committed (tag `prereg5`) before any run.

**Why this exists.** In PREREG-4 (exploratory T3), the trained Pythia-160M's perturbation-growth exponent λ rose smoothly with attention inverse temperature β and crossed zero almost exactly at its natural temperature (β = 1, λ = −0.0005):
- flattened attention gave λ < 0, so perturbations die out (ordered);
- sharpened attention gave λ > 0, so perturbations amplify (chaotic).

The shuffled control stayed positive at every β, and partially trained checkpoints were positive. This edge-of-chaos signature was exploratory and found on one model, so it is tested here confirmatorily.

**Estimator sensitivity, disclosed in advance** (exploratory diagnostic, `results/m2_fresh2/growth_curves.json`). The location of the zero crossing depends on the estimator:
- the PREREG-4 estimator (the average of per-perturbation log-growth slopes, i.e. a Lyapunov-style average) puts it at β ≈ 1.0;
- the slope of the averaged response curve puts it near β ≈ 1.6;
- a late-layers-only slope (t ≥ 3) stays negative up to β = 2.

In the trained model an early build-up of cross-position influence is balanced by late-layer contraction, and the controls show no late contraction. This test therefore keeps the **PREREG-4 estimator as primary**, because it is the one that generated the hypothesis and the one matching the Lyapunov definition, and reports the others as secondary.

**Unchanged:** all earlier results stand as scored.

## Fresh data
- **Source:** WikiText-103-raw train split, qualifying articles after number 128 (after the PREREG-3 and PREREG-4 sets), keeping only those with at least 256 tokens under both the Pythia and the GPT-2 tokenizers. The first 64 such articles form the set; this test uses the first 32.
- **Manifest:** `stimuli/wikitext_fresh3_manifest.json`, sha256_all `48d20f1b…`. The first article is "Ohio State Route 319"; zero overlap with any earlier set.
- **Tokenisation:** Pythia models use the stored GPT-NeoX token IDs. GPT-2 re-tokenises the stored text and keeps its first 256 tokens.

## Models and conditions
**Trained models:**

| Model | Layers | Injection layers |
|---|---|---|
| Pythia-160M (step 143000) | 12 | every layer, k ≤ 9 |
| Pythia-410M (step 143000) | 24 | every 2nd layer, k ∈ {0, 2, …, 20} |
| Pythia-1B (step 143000) | 16 | every layer, k ≤ 13 |
| **GPT-2 small** (124M) | 12 | every layer, k ≤ 9 |

GPT-2 small is a different family: different training data (WebText), a different tokenizer, and learned absolute position embeddings.

**Temperatures:**
- Trained models: β ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}, applied at `hook_attn_scores` in every block.
- Controls: each model's weight-shuffled copy (seed 1001) at β ∈ {0.25, 1, 2}, and untrained step 0 at β = 1 (Pythia only; GPT-2 has no released step-0 checkpoint).

**Main grid:** 32 passages × 4 directions (the first 4 of the 8 seeded directions, seed 11) × p ∈ {16, 64, 128}.

**Calibration (ε\*), per condition:**
- the first 4 passages × 2 directions × injection layers {0, ⌊n/3⌋, ⌊2n/3⌋};
- the rule is unchanged: the largest ε whose median deviation from the float64 reference is ≤ 5%;
- **pre-registered fallback:** if no ε qualifies, use the ε with the smallest median deviation, flagged in the output and reported.

**Totals:** 39 conditions, about 9–12 GPU-hours. The queue ends with an automatic instance stop.

## Primary measure
- **λ:** for each (passage, direction, injection layer k, position p), the OLS slope over depth t of ½·log E(t), where E(t) = mean over q > p of G(k+t, q)². This is averaged over k and p per (passage, direction), then averaged.
- **CIs:** crossed bootstrap over passages × directions, 10,000 resamples. For |λ| comparisons the bootstrap is paired.

## Predictions (approved by the user as written; each scored separately, per model)
- **P1 (zero crossing near the natural temperature):** λ(0.75) < 0 and λ(1.5) > 0, with both 95% CIs excluding 0.
- **P2 (monotone):** λ point estimates increase strictly across β = 0.25 → 2.
- **P3 (controls never cross):** the shuffled copy has λ > 0 (CI excluding 0) at β = 0.25, 1 and 2, and Pythia step 0 has λ(1) > 0 (CI excluding 0).
- **P4 (trained is closest to the edge):** |λ(trained, β=1)| < |λ(control, β=1)| for each control, with paired CIs excluding 0.

**Headline rule, fixed now:** "the zero crossing replicates" only if P1 passes in **all four** models. Otherwise the result is reported model by model.

## Secondary (reported, not scored)
- the late-layer λ (t ≥ 3) per (passage, direction);
- the averaged-curve λ (from the raw lattices);
- PCI at θ = 0.01 across β;
- any fallback ε used.
