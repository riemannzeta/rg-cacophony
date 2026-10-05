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
