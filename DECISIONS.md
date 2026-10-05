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
