# rg-cacophony

This repo holds output-free indicator measurements for language models, of the kind called for in Chandaria, Seth, Shevlin, Shanahan, Rosas et al., [*From cacophony to hierarchy*](https://arxiv.org/abs/2609.35618). Every measurement is taken inside the network and never looks at what the model writes.

**Main result: a transformer "zap test".** This is an analogue of the clinical Perturbational Complexity Index (Casali, Massimini et al., 2013):
- A small push goes into the residual stream at one word and one layer.
- The response at every later layer and word is recorded, binarised, and compressed with Lempel–Ziv.

Flattening attention, an anaesthesia-like manipulation, lowers this complexity by about a third while the response's reach is preserved, which is the signature of anaesthesia in the clinic.
- The pattern was found exploratorily in Pythia-160M, then **replicated on fresh text with all four pre-registered predictions passing**.
- It then appeared in all four models of a further pre-registered study: Pythia-160M, 410M and 1B, and GPT-2 small. There it was a pre-specified secondary measure.

**Other results:**
- **Edge of chaos, in small models only.** A perturbation-growth exponent λ puts trained Pythia-160M and GPT-2 small at λ ≈ 0, poised between order and chaos, at their natural attention temperature. Untrained and weight-shuffled controls amplify. The pre-registered confirmation failed at 410M and 1B, where trained models amplify at every temperature.
- **Not supported:** a mid-depth "symmetry-breaking" commitment to one word sense, a dynamic exponent z, and chimera-state predictions.

Everything, including the failures, is in [REPORT.md](REPORT.md). A short write-up of the zap test is in [site/symmetrybroken-post.md](site/symmetrybroken-post.md).

## How this was done
- **Pre-registration:** five rounds, each committed and tagged before its data existed.

  | file | tag |
  |---|---|
  | [PREDICTIONS.md](PREDICTIONS.md) | `prereg` |
  | [PREREG-2.md](PREREG-2.md) | `prereg2` |
  | [PREREG-3.md](PREREG-3.md) | `prereg3` |
  | [PREREG-4.md](PREREG-4.md) | `prereg4` |
  | [PREREG-5.md](PREREG-5.md) | `prereg5` |

- **Scoring:** mechanical, by `scripts/s4_compare.py`, written to [results/SCORECARD.md](results/SCORECARD.md).
- **Deviations:** every departure from plan is logged in [DECISIONS.md](DECISIONS.md). The original plan is in [PLAN.md](PLAN.md).
- **Controls:** untrained (step 0) and weight-shuffled networks throughout, plus early training checkpoints.
- **Determinism:** runs are bitwise deterministic. Every result file records the git commit, config hash, package versions and GPU.

## Layout
```
src/rgc/        instruments: models, perturbation engine, PCI/LZ76, lambda, attention manipulations, stats
scripts/        s0–s5 stage scripts, PREREG-specific builders, figures, results.json builder
stimuli/        ambiguous-word prompts and WikiText passage manifests (with token-id hashes)
results/        per-run summaries (JSON), scorecard, results.json, figures, logs
tests/          instrument tests (hooks, causality, LZ76, PCI, determinism, shuffling)
site/           the symmetrybroken.com write-up and its images
```

## Reproducing
```bash
uv sync                      # Python 3.12, pinned versions (see pyproject.toml / uv.lock)
uv run pytest                # instrument tests
uv run python scripts/s0_env_check.py --device cuda
uv run python scripts/s3_m2.py --help                 # perturbation runs (all PREREG options)
uv run python scripts/s4_compare.py                   # scorecard
uv run python scripts/s5_results.py && uv run python scripts/s5_figures.py
```
The numbers in the report come from an NVIDIA A10G.

## Raw data
The raw perturbation lattices (gain at every layer × position, for every passage, direction and injection site) are too large for git. They are attached to the [`data-v1` release](https://github.com/riemannzeta/rg-cacophony/releases/tag/data-v1) as split tar archives; the release notes explain how to reassemble them. The summaries in `results/` are computed from these lattices.

## Provenance note
Before publication, the git history was rewritten to remove infrastructure identifiers: IP addresses, cloud resource IDs and a key-file name. Commit dates, contents and pre-registration tags are otherwise unchanged. Result files record the commit SHA of the code that produced them, which is a pre-rewrite SHA. [COMMIT_MAP.tsv](COMMIT_MAP.tsv) maps each old SHA to its published equivalent.

## Licence
- Code: [MIT](LICENSE).
- Data, results and text: [CC BY 4.0](LICENSE-DATA.md). WikiText excerpts remain under CC BY-SA.

The work was carried out with Claude (Anthropic) as a research assistant.
