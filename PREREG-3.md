# PREREG-3.md — confirmatory replication of the anaesthesia pattern on fresh data

**Status:** SIGNED OFF by the user on 2026-10-05. The user confirmed R1–R4 as their predictions and approved the compute. Committed (tag `prereg3`) together with the fresh-passage manifest, before any fresh-data run.

**Why this exists.** Stage 3 tested PREREG-2 part A at the pre-registered θ = 0.1. A1–A3 failed, but at that threshold almost every lattice is near-empty: only 1.6% of sites are on, and half of all lattices have p₁ < 1%.

At θ = 0.01, which is part of the pre-registered *secondary* grid, the exploratory pattern was the one A predicted:
- flattening attention lowered PCI by about 35% (0.66 → 0.42);
- reach barely changed (0.21 → 0.18);
- the threshold-free spread more than doubled (74 → 166);
- the dose-response was monotone;
- shrinking the window lowered PCI from ∞ to 64 to 16.

That pattern was found by looking at data, so it is a hypothesis, not a finding. This file tests it on data no analysis has touched.

**What is NOT changed:** the PREDICTIONS.md and PREREG-2 results stand as scored, including the θ fallback rule evaluated in Stage 4.

## Fresh data
- **Source:** WikiText-103-raw **train** split. Take the first 64 articles, in dataset order, with at least 256 body tokens.
- **Same rule as before:** same header parser and detokeniser; the first 256 tokens of each article; BOS prepended.
- **Freshness:** none of these articles has been used by any analysis in this project.
- **Manifest:** written to `stimuli/wikitext_fresh_manifest.json` (token IDs and SHA256) and committed before any run.

## Model, manipulations, protocol (unchanged from PREREG-2 A)
- **Model:** Pythia-160M at step 143000.
- **Conditions:** intact; β ∈ {0.5, 0.25, 0}; window w ∈ {64, 16, 4}. All applied at `hook_attn_scores` in every block.
- **Protocol:** the M2 protocol, unchanged. Each condition gets its own ε* (calibrated on the first 8 fresh passages). The rest is as before: 12 values of k, p ∈ {16, 64, 128}, the same 8 directions (seed 11), PCI over k ≤ 9.
- **CIs:** paired crossed bootstrap over passages and directions, 10,000 resamples.

## Primary threshold
- **θ = 0.01**, fixed now. It is the value at which the pattern was observed, so this is a direct replication.

## Measures (per passage × direction, averaged over k ≤ 9 and all p)
- **PCI** at θ = 0.01.
- **Reach** = mean p₁ at θ = 0.01.
- **Threshold-free spread** = mean participation spread S_eff over the same lattices and all downstream slices.

## Predictions (confirmed by the user; each scored separately)
- **R1:** PCI(β=0) < PCI(intact), with the paired CI excluding 0.
- **R2 (the dissociation):** PCI falls proportionally more than reach.
  - The CI of PCI(β=0)/PCI(intact) − reach(β=0)/reach(intact) lies below 0.
  - **and** the threshold-free spread does not fall: S_eff(β=0) − S_eff(intact) > 0, with the CI excluding 0.
- **R3 (dose-response):** the point estimates satisfy PCI(1) > PCI(0.5) > PCI(0.25) > PCI(0), and PCI(1) − PCI(0) has a CI excluding 0.
- **R4 (window positive control):** PCI(∞) > PCI(64) > PCI(16), with each adjacent paired CI excluding 0. w = 4 has no prediction; it was non-monotone in the exploratory data.

## Secondary (pre-registered, reported but not part of R1–R4)
- **Density-matched PCI:** each lattice is binarised at its own top 20% of downstream sites, so p₁ is exactly 0.2. PCI then measures arrangement only, independent of reach and sparsity. Prediction: PCI_dm(β=0) < PCI_dm(intact), with the CI excluding 0.
- The full θ grid for every condition.

## Run
- **Where:** rgc-gpu, alongside the Stage 4 queue, from a separate checkout. Calibration is chunked (52 rows), and outputs go to `results/m2_fresh/`.
- **Implementation:** `scripts/s3_m2.py --passages stimuli/wikitext_fresh_manifest.json --out-root results/m2_fresh --dm-frac 0.2`, then scoring in `scripts/s4_compare.py` (section PREREG-3). Constants are in `src/rgc/config_prereg3.py`.
- **Manifest:** `stimuli/wikitext_fresh_manifest.json`, sha256_all `b0f6de978e6b204b…`. The first article is "Valkyria Chronicles III"; zero token-level overlap with the original 64 passages (asserted).
- **Cost:** about 2.5–3 GPU-hours.
