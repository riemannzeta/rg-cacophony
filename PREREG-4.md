# PREREG-4.md — criticality and chimera states (fresh data)

**Status:** SIGNED OFF by the user on 2026-10-08, with the compute approved. Committed (tag `prereg4`) before any run.

**The user's answers:**
- **Agree:** T1, T2, S1, S2, C1, C2, C3.
- **Exploratory:** T3. It is reported and not scored.

**Motivation.** The user's essay ("What art is doing", symmetrybroken.com) models healthy processing as poised at a critical point:
- **ordered side:** attention "commits, conviction crystallises" (hard argmax, frozen);
- **disordered side:** attention "diffuses" (everything averaged);
- **at the critical point:** metastable chimera states, with some token clusters synchronised and others productively divergent.

Two literatures behind the Cacophony paper use matching instruments:
- **Criticality:** the conscious brain sits near the edge of chaos, and anaesthesia pushes it away.
- **Shanahan's chimera and metastability indices.**

Results so far bear on this directly:
- PREREG-3 replicated that flattening attention (β < 1) lowers PCI at θ = 0.01.
- Stage 4 showed that PCI alone cannot tell structured from random dynamics: shuffled Pythia scores *higher* at θ = 0.01.

This addendum tests the essay's claims with an explicit order–chaos axis and with Shanahan's measures.

**Unchanged:** all earlier results stand as scored.

## Fresh data
- **Source:** the next 64 qualifying articles of the WikiText-103-raw **train** split (qualifying numbers 65–128), in dataset order.
- **Method:** the same parser, detokeniser and 256-token truncation as before.
- **Freshness:** zero overlap with any earlier passage set (asserted).
- **Manifest:** `stimuli/wikitext_fresh2_manifest.json`, sha256_all `93be3062…`. The first article is "Corn crake".

## Conditions
- **Models:** Pythia-160M at step 143000 (trained); step 0, step 512 and step 4000; three weight-shuffled copies (seeds 1001–1003).
- **Temperature β:** attention scores are multiplied by β in every block (`hook_attn_scores`). β > 1 sharpens attention toward argmax (ordered); β < 1 flattens it (disordered); β = 1 is the model as trained.
- **β grid:** {0.25, 0.5, 0.75, 1, 1.5, 2, 4}, for the trained model **and** for shuffled seed 1001. Every other model runs at β = 1 only.
- **Protocol:** the M2 protocol as in PREREG-3:
  - each condition gets its own ε*;
  - 12 values of k; p ∈ {16, 64, 128}; the same 8 directions;
  - θ = 0.01 for PCI and reach;
  - the density-matched PCI (top 20% of sites) as secondary.

## Measures
- **PCI** at θ = 0.01; **PCI_dm** (density-matched).
- **Growth exponent λ:** the per-layer slope over depth of ½·log(mean over q > p of G²), averaged over k ≤ 9, the three p values and the 8 directions.
  - λ < 0: perturbations shrink relative to the state (ordered).
  - λ > 0: perturbations grow (chaotic).
  - λ ≈ 0: marginal (critical).
- **Chimera index χ and metastability index m** (Shanahan 2010), with depth as time (slices 1–12) and token positions as oscillators:
  - each slice's LayerNorm-normalised residuals are mean-centred across positions and unit-normalised;
  - communities are contiguous 16-token blocks (primary) or sentences (secondary);
  - R_c(L) = ‖mean unit vector of community c‖;
  - χ = mean over L of the variance across communities of R_c(L);
  - m = mean over c of the variance across L of R_c(L).
- **Effective dimensionality D(L):** the participation ratio of the centred token cloud at each slice. The "waist" is the slice with minimum D.
- **CIs:** paired crossed bootstrap (passages × directions; for χ and m, passages only), 10,000 resamples.

## Predictions (from the essay; confirmed by the user except T3, which is exploratory; each scored separately)
**Criticality in temperature (trained model)**
- **T1 (inverted U in PCI):** PCI(1) − PCI(0.25) > 0 **and** PCI(1) − PCI(4) > 0, with both paired CIs excluding 0.
- **T2 (peak at the natural temperature):** the β that maximises the PCI point estimates is in {0.75, 1, 1.5}.
- **T3 (edge of chaos; EXPLORATORY by the user's choice):** λ increases with β (monotone point estimates), and |λ| is smallest at a β in {0.75, 1, 1.5}.

**Self-tuning (trained vs. controls, at β = 1)**
- **S1:** the trained model is closest to the edge of chaos: |λ(trained)| < |λ(step 0)| and |λ(trained)| < |λ(each shuffle)|, with every paired CI excluding 0.
- **S2:** the natural temperature is special only after training: for shuffled seed 1001, the β that maximises PCI is **not** in {0.75, 1, 1.5}.

**Chimera states**
- **C1:** χ(trained) > χ(step 0) and χ(trained) > χ(each shuffle), with paired CIs excluding 0.
- **C2 (inverted U):** χ(1) > χ(0.25) and χ(1) > χ(4), with paired CIs excluding 0.
- **C3 (the waist):** the trained model's minimum of D(L) over slices 1–11 lies in the middle third (slices 4–8); step 0's does not.

**Exploratory (reported, not scored)**
- Spearman correlation between PCI and χ across the 7 β conditions.
- Metastability m.
- The sentence-community versions.
- Steps 512 and 4000.
- PCI_dm in every comparison.

## Compute
- **Runs:** trained β-sweep (7) + controls at β = 1 (6) + shuffled β-sweep (6) = 19 conditions.
- **Cost:** about 15 minutes each, about 5 GPU-hours (about $5).
- **Where:** rgc-gpu, from a checkout pinned to tag `prereg4`. The queue ends with an automatic instance stop.
