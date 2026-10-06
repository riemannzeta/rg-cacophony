# DECISIONS.md

An append-only log of every choice that departs from, or fills a gap in, the approved plan (PLAN.md). Each entry says what was decided, why, and whether results had been seen when it was made.

---

## D1 — 2026-10-04 — Dedicated GPU instance (no results seen)
- **What:** this project gets its own g5.xlarge (`rgc-gpu`) instead of co-running on `ml-dev`, which is busy with ladder-witnesses. This was the user's request.
- **Quota:** an increase for `L-DB2E81BA` (4 → 8 vCPUs, us-west-2) was filed 2026-10-04 and is now support case <support-case-id>.
- **Meanwhile:** Stage 0 CPU checks and Stage 1 run on the Mac. All reported numbers will come from the GPU.

## D2 — 2026-10-04 — Python 3.12, isolated uv project (no results seen)
- **Python:** pinned to 3.12 to match the GPU box (system Python 3.12.3).
- **Isolation:** `~/pyproject.toml` declared a uv workspace that listed `rg-cacophony` as a member, which would have shared one environment with projects requiring Python ≥3.13. I moved `rg-cacophony` from `members` to `exclude` in `~/pyproject.toml`; a backup of the original is in the session scratchpad. The project now has its own `.venv` and `uv.lock`.

## D3 — 2026-10-04 — TransformerLens pinned to 3.9.0 (no results seen)
- **Why:** TransformerLens 4.0.0 (the latest release) removed `HookedTransformer` in favour of `TransformerBridge`. The spec names `HookedTransformer`, so I pinned 3.9.0, the last release that has it.
- **Pins:** torch 2.14.1, transformers 5.18.0, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, datasets 5.0.1, pytest 9.1.1.

## D4 — 2026-10-04 — Checkpoints are loaded through `hf_model`, not `checkpoint_value` (no results seen)
- **Bug found by the Stage 0 sanity check.** With `checkpoint_value=…` set, TransformerLens 3.9 re-downloads the checkpoint from the Hub and *silently ignores* any `hf_model` passed in. The weight-shuffled controls therefore ran the *unshuffled* final model. Two signals exposed it:
  - TL-vs-HF logits differed by 880% relative;
  - the shuffled variants' perturbation numerics were bit-identical to the original model's.
- **Fix:** every variant is now built as `AutoModelForCausalLM.from_pretrained(repo, revision="step{N}")`, shuffled if required, and then passed to `HookedTransformer.from_pretrained_no_processing(name, hf_model=…)` with no `checkpoint_value`.
- **Effect:** same weights as the spec's `from_pretrained(checkpoint_value=N)` for unshuffled checkpoints, now also correct for the controls.
- **Guard:** Stage 0 checks TL-vs-HF logits (relative max diff < 1e-5) and records a weight fingerprint for every variant.

## D5 — 2026-10-04 — Unprocessed weights and explicit BOS (no results seen; as planned)
- **Unprocessed weights:** `from_pretrained_no_processing` is used in float32, with no LN folding and no centering of writing weights or unembed. Residual norms are therefore the original model's.
- **Explicit BOS:** BOS (`<|endoftext|>`, id 0) is prepended by our own `tokenize()`, and TL's automatic prepending is turned off. Sequence position 0 is always BOS. All positions in config (e.g. `p_grid`) are sequence positions.

## D6 — 2026-10-04 — Row-matched clean baselines for perturbations (no results seen)
- **What:** each perturbed batch of R rows is compared with a clean batch of R identical rows, row by row. It is not compared against a single clean forward.
- **Why:** the two rows then go through identical kernels, so any nonzero response comes from the perturbation and not from batch-position-dependent rounding.
- **Verified on CPU:** ε=0 rows give exactly zero; q<p gives exactly zero; the injected site's relative size equals ε to within 1e-5.

## D7 — 2026-10-04 — Shuffled control covers all parameters (no results seen; as planned)
- **What:** every `named_parameter` is permuted within itself, including embeddings, unembedding, LN weights and biases, and attention and MLP biases.
- **Excluded:** buffers (rotary `inv_freq`) are not parameters and are untouched.
- **Check:** all 148/148 tensors change and each is an exact permutation of the original.
- **Constant tensors:** a tensor whose values are all equal (e.g. step-0 LN weights of all ones) is unchanged by construction.

## D8 — 2026-10-04 — " crane" replaced by " palm" (no results seen)
- **Why:** in the Pythia tokenizer `" crane"` is two tokens (`' cr'` + `'ane'`), so it fails the single-token requirement. The other nine candidates are single tokens.
- **Replacement:** `" palm"` (single token, id 19992), with the noun senses *hand* and *tree*. It was chosen to keep 10 words, before any model forward pass on stimuli.
- **Senses used:**
  - bank: river / money
  - bat: animal / sport
  - pitcher: baseball / jug
  - seal: animal / closure
  - spring: season / coil
  - match: contest / fire
  - mole: animal / skin
  - pen: writing / enclosure
  - bark: tree / dog
  - palm: hand / tree
- The tokenizer vocabulary was verified identical at steps 0, 512, 4000 and 143000.

## D9 — 2026-10-04 — WikiText passages: test split, then validation (no results seen)
- **Why:** the WikiText-103 test split has only 62 articles, and 61 of them have ≥256 tokens, so the planned "64 test passages, one per article" cannot be met from test alone.
- **Rule:** take all qualifying test articles in order, then validation articles in order, until there are 64. That gives 61 test + 3 validation.
- **Unchanged:**
  - one passage per article, so passages are independent units for the bootstrap;
  - each passage is the first 256 tokens of the article body;
  - text is detokenised with the standard WikiText detokeniser from lm-evaluation-harness.
- **Recorded:** the token IDs, titles and SHA256 hashes are in `stimuli/wikitext_manifest.json`.

## D10 — 2026-10-04 — Stimulus design details (no results seen)
- **Minimal pairs:**
  - Each word has 30 frames (20 train, 10 test). Each frame is filled once with a sense-A cue phrase and once with a sense-B cue phrase.
  - Frame words are therefore identical across senses, and target positions differ by 0.7–2.3 tokens on average within a pair.
  - Both members of a pair get the same continuation.
- **Split disjointness:**
  - Test frames differ from train frames.
  - After stemming, *every* content word in a word's test cue phrases (not only the bracketed cues) is absent from that word's train cue phrases. This is checked by `stimuli.build()`, and validation fails on any overlap.
  - Near-overlaps (shared prefixes of 4 or more letters, e.g. cave/cavern, fire/firewood) were removed by hand.
  - No cue word is used for both senses of a word.
- **Cue marking:** square brackets mark words that on their own point toward one sense. Cue mass is computed over the tokens of those words. Generic words (e.g. "window", "radio") were un-bracketed before any measurement.
- **Neutral prompts:** 12 shared cue-free frames, used for every word.
- **Control token:** `" thing"` (id 2181). The control prompt is identical to the target prompt except at the target position.
- **Continuations:** four sense-neutral continuations of 28–29 tokens.
- **Prompt length:** 15–19 tokens including BOS, at the target position.
- **Known orthographic overlap:** `" pencil"` (a writing cue for *pen*) contains "pen". It is a different token and a legitimate cue, so it is kept.

## D11 — 2026-10-05 — torch native DSL op overrides disabled (no results seen)
- **Problem:** on CUDA, torch 2.14.1 routes some eager ops through runtime-compiled kernels instead of ATen:
  - Triton `bmm` (hit by the HF rotary embedding);
  - `native` `_foreach_mm` and `linalg_polar`.
- **Symptom:** on the fresh AMI, the Triton build failed for lack of Python headers.
- **Fix:** rather than install build tooling, `set_determinism()` deregisters every DSL override with `torch._native.registry.deregister_op_overrides`. All ops then run on standard ATen kernels, whose determinism under `use_deterministic_algorithms(True)` is documented.
- **Recorded:** the disabled DSLs (`["native", "triton"]`) are written into every result's env block.

## D12 — 2026-10-05 — Dedicated instance launched (no results seen)
- **Instance:** `rgc-gpu` = <instance-id>, g5.xlarge (A10G), cloned from ml-dev's DL AMI, subnet, SG and key, with a 100 GB gp3 volume. Launched after the quota increase to 8 vCPUs was granted.
- **Tooling:** uv 0.12.23, matching ml-dev.
- **Code flow:** code reaches the box only via `git push gpu main`, so every run is at a committed SHA. `scripts/gpu.sh` handles start/stop/push/pull.
- **Stage 0 on CUDA:** all checks pass for all 7 variants (`results/stage0/sanity_cuda.json`), including bitwise reproducibility, exact-zero ε=0 and causality responses, and unit tests run on the GPU.

## D13 — 2026-10-05 — Stage 2 re-run for clean provenance (results seen only as integrity flags)
- **What happened:** the first Stage 2 run on step 143000 executed the pre-registered commit `1c25bf8` exactly, but recorded its SHA as `-dirty`. The cause was that its own log file was untracked in the checkout.
- **Fix:** `git_sha()` now ignores untracked files (commit `bf3791a`). That commit changes no analysis code, and the config hash is unchanged.
- **Re-run:** Stage 2 was run again. The re-run's summary is compared field by field with the first run, as an end-to-end determinism check.
- **What was seen before the re-run:** only integrity flags (finite values, slice-0 identity, counts, provenance). No measurement was inspected.

## D14 — 2026-10-05 — Sequencing change and addendum pre-registration (after Stage 2 results)
- **M1 controls run early.** M1 on step 0, 512, 4000 and the three shuffle seeds runs now, before Stage 3. Every analysis choice for these runs is already fixed in PREDICTIONS.md (b), so this changes only the order of work, not any analysis.
- **New tests go in a separate pre-registration.** The additions proposed after Stage 2 get their own file (PREREG-2), committed before any of their data are generated:
  - an attention-flattening ("anaesthesia") test, plus a local-window positive control, for PCI;
  - a conflicting/graded-cue susceptibility test of symmetry breaking, on NEW prompts;
  - later: a recurrent (autoregressive) perturbation protocol, coarse-graining / causal-emergence measures, a Pythia-vs-Mamba comparison, and scale.
- **Why PREREG-2 comes before Stage 3:** anything involving PCI must be registered before the trained model's PCI lattices are seen.
- **PREDICTIONS.md is unchanged.** Its results are reported as scored, whatever PREREG-2 finds.

## D15 — 2026-10-05 — M1 control runs: provenance note (results not yet inspected)
- **SHAs:** the six control variants ran at commits `d66d38b` (step 0, step 512) and `9e7c4df` (the rest). The trained model ran at `bf3791a`.
- **Same instrument:** `git diff` shows the M1 code (`m1_order.py`, `m1_ablation.py`, `stats.py`, `stimuli.py`, `models.py`, `determinism.py`, `config.py`, `scripts/s2_m1.py`, `stimuli/`) is identical across these commits, and all runs share config hash `26fef58ef05e1363`.
- **Post-hoc analysis:** `results/m1/pythia-160m_step143000/posthoc.json` (exploratory, D14) reproduces the pre-registered final-slice ablation effect exactly (asserted in the script).

## D16 — 2026-10-05 — PREREG-2 part B stimulus details (no B data generated)
- **Cue phrases:**
  - Train frames (0–7) use cue phrases 0–3 and test frames (8–15) use phrases 4–7, so train and test cue vocabularies are disjoint, as in v1. Within a split, the Latin-square rotation fixes which phrase fills which slot.
  - The validator rejected 3 v2 cue words that v1 had already used for the same word (`bounced`, `fishing`, `sticks`), 1 filler word that also appeared as a cue (`ride`), and 14 near-overlaps with v1 (e.g. `cigar`/`cigarette`, `waterfall`/`water`). All were replaced before any run. The final set has no errors or warnings.
- **Saturation ratio R:** t is normalised by the test-set (A,A) and (B,B) means, so the ratio's denominator t̄(+2) − t̄(−2) is exactly 1. The pre-registered "denominator CI excludes 0" condition is therefore applied to the raw separation m_AA − m_BB, for both target and control. This is implementation detail only; the pre-registered quantity is unchanged.
- **Prompt endings:** prompts end at the target, with no continuation, because part B measures only at the target.
- **Validation:** synthetic ground-truth tests confirm R = 0.5 for a linear response and R = 1 for a step response.

## D17 — 2026-10-05 — Faster LZ76 (identical output) and the Mamba runner (no new results seen)
- **LZ76:** `rgc.pci.lz76` now uses a vectorised exhaustive-history parse (each new phrase is one symbol longer than the longest earlier match, overlap allowed).
  - Its phrase counts equal the Kaspar–Schuster reference (`lz76_ks`, kept) exactly on 2,371 test strings: lengths 1–2304, densities 0–1, plus periodic strings. A permanent test enforces this.
  - About 8.5× faster (6.4 ms vs 55 ms per 2,304-bit string).
  - Because outputs are identical integers, runs made with either implementation are interchangeable. The intact Stage 3 run (in progress) uses the reference implementation; later runs use the fast one.
- **Mamba runner (`rgc/mamba.py`, PREREG-2 part C):** HF `MambaForCausalLM` on the pure-torch path (neither the `kernels` nor the `mamba_ssm` package is installed). Perturbation and reading use forward pre-hooks.
  - Stage-0-style checks on mamba-130m (CPU) pass: bitwise reproducible, ε=0 gives exactly zero, q<p gives exactly zero, and the injected site equals ε to 1e-3. Shuffled copies are exact per-tensor permutations.

## D18 — 2026-10-05 — Mamba float64 reference: three hidden float32 casts removed (no indicator values seen)
- **Symptom:** an engineering pilot (passage 0 only; calibration curves only, no PCI) showed a 40–60% "plateau" between Mamba-130M's float32 gains and its float64 finite-difference reference. Clean states agreed to 2e-6, and float32 and float64 gains agreed for ε ≥ 1e-4. The fault was therefore in the reference: the float64 model at ε = 1e-8 disagreed with itself at ε = 1e-6.
- **Cause:** HF's Mamba silently computes parts of a float64 model in float32:
  - `residual_in_fp32` casts the residual stream to float32;
  - `MambaRMSNorm` normalises in float32;
  - the torch selective scan casts B and its input with `.float()`.

  Each puts a float32 noise floor of about 0.06 gain units on a reference taken at ε = 1e-6.
- **Fix:** in the float64 *reference copy only*, residuals stay float64 and RMSNorm keeps its input dtype. The scan's two `.float()` casts become `.to(input_dtype)` module-wide; this is bitwise identical for float32, as a test verifies against the original function. Casts on float32-valued *parameters* (A_log, D, biases) are exact and left alone.
- **After the fix:** the float64 copy is linear to 1e-6 between ε = 1e-8 and 1e-6 (test), and float32 vs float64 agree to 0.6% at ε = 1e-2.
- **Pythia/TransformerLens check:** the same self-consistency test on Pythia-160M shows its float64 reference is genuinely float64 (ε = 1e-8 vs 1e-6 agree to 4e-5), so the Stage 3 calibration is unaffected.
- **Observation, unchanged rule:** Pythia's float32 noise floor is high at small ε, because late-layer residual norms are hundreds of times the injection-site norm. Its usable linear window sits near ε ≈ 1e-2–1e-1, with float32 noise and nonlinearity each about 3–4%, which is within the pre-registered 5% tolerance.

## D19 — 2026-10-05 — Gating test boundary (exploratory test; integrity seen, results not inspected)
- **Clipping:** the trained model's L* = 0 (block 0), so k_before = L* − 2 = −2 is clipped to slice 0, as the pre-registered boundary rule provides. k_after = 1. Both windows keep 3 downstream slices.
- **Sense direction:** it is undefined at slice 0, because every prompt of a word has the identical embedding there. The "before" condition therefore uses only the 8 random directions. The "after" condition uses random directions and the sense direction.
- The headline contrast (after − before) uses random directions only, as pre-registered.

## D20 — 2026-10-05 — Stage 4 approved; Mamba injections at even layers only (no C data generated)
- **Approval:** the user approved Stage 4 (about 20 GPU-hours).
- **Cost:** HF's pure-torch Mamba scan loops over positions in Python, about 3.5 s per 24-row response on the shared A10G. A full-resolution Mamba grid would cost about 30 GPU-hours.
- **What changes:**
  - Mamba runs inject only at even layers (`--k-stride 2`), which is exactly the set the **pre-registered primary** stride-2 analysis uses, so the primary analysis is unchanged.
  - The calibration (ε*) also uses even-k injections.
  - The PREREG-2 C **secondary** full-resolution analysis is therefore not run.
  - Calibration batches are chunked (24 rows for 130M, 12 for 370M) for memory; each chunk keeps its own row-matched clean baseline.
- **Status:** all of part C is exploratory (the user made no prediction), and nothing about it has been measured.

## D21 — 2026-10-06 — Notes from scoring (c) and PREREG-2 B (after results were seen; nothing re-scored)
- **θ fallback (pre-registered, mechanical):** the pooled degenerate fraction at θ = 0.1 over the 7 Pythia-160M variants is 0.765, below 0.9, so θ_primary stays 0.1. PREDICTIONS (c) and PREREG-2 A therefore stand as scored at θ = 0.1.
- **Determinism:** the Stage 4 re-run of the trained model's main phase reproduces run 1's PCI matrix and PCI-by-θ bitwise, even though run 1 used the reference LZ76 and the re-run the fast one.
- **B instability:**
  - The B1/B2 CIs are very wide (B1 diff 0.51, CI −2.6 to 3.0) because at slices 2–4 two words (seal, bark) have per-word raw separation m_AA − m_BB ≈ 0. Their t values explode (seal R = 5.0 at slice 4) and dominate the pooled mean.
  - The pre-registered implementation checked the *pooled* separation's CI rather than each word's, which did not guard against this. That is a flaw in my implementation of the pre-registration.
  - The pre-registered verdicts stand: B1 FAIL, B2 FAIL (uninformative rather than contrary). Any per-word-guarded re-analysis is exploratory only.

## D22 — 2026-10-06 — Stage 4 Mamba runs failed on a guard; re-run approved compute (no C data existed)
- **What failed:** all Mamba runs (part C) stopped before calibrating. My float64 sanity guard required float64 *logits*, but HF Mamba casts its final logits to float32 even when every internal computation is float64. The measured float32 vs float64 agreement was fine (3e-6 to 2e-5).
- **Fix:** `f64_probe()` now checks the reference model's *parameter* dtype plus output agreement. It is unit-tested for Mamba (float32 logits accepted; a float32 "reference" rejected) and for TransformerLens.
- **Re-run:** the Mamba part of the approved Stage 4 compute is re-run unchanged.
- **Idle box:** after the queue ended at 00:39 UTC the box sat idle for about 16 h (about $16), because nothing stopped it. From now on every queue ends with an automatic instance stop.
