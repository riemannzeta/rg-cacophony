# rg-cacophony

This project tests two output-free indicator measurements on Pythia, motivated by a renormalization-group reading of the transformer forward pass:

1. **Symmetry breaking (awareness indicator):** the depth at which an ambiguous word's representation commits to one sense. It is measured with a cross-validated, context-leakage-corrected d′ and aligned with per-head attention entropy and cue mass. Head ablations test it causally.
2. **Flow (phenomenal-experience indicator):** a Perturbational Complexity Index analogue. Perturbations injected into the residual stream are followed across the (layer × position) lattice; the result is binarised and scored with normalised LZ76, and a spread-versus-depth exponent z is fitted.

Both are run on trained Pythia-160M, on early checkpoints, at initialisation (step 0), and on weight-shuffled controls.

- The approved plan is in [PLAN.md](PLAN.md).
- The pre-registered predictions are in [PREDICTIONS.md](PREDICTIONS.md).
- Every departure from the plan is logged in [DECISIONS.md](DECISIONS.md).

## Setup
```bash
uv sync                      # Python 3.12, pinned versions (see pyproject.toml / uv.lock)
uv run pytest                # instrument sanity tests
uv run python scripts/s0_env_check.py --device cuda   # per-variant sanity → results/stage0/
```
