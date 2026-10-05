# rg-cacophony — symmetry breaking and flow in Pythia: two indicator measurements

## Context
This project tests an RG reading of transformer computation against Chandaria, Shevlin, Seth et al. (arXiv:2609.35618).
- **Awareness** is read as symmetry breaking: a sharp, localised rise in sense separation d′ at an ambiguous word, carried by low-entropy, high-cue-mass heads.
- **Phenomenal flow** is read through a PCI analogue: LZ complexity of how a perturbation spreads over the (layer × position) lattice, plus a dynamic exponent z.

The goal is to show that both quantities are well defined, robust, measurable without looking at outputs, and able to tell trained networks from controls (step 0, weight-shuffled). Nothing gets reported until both measurements exist. Predictions are pre-registered, and every later choice is logged in DECISIONS.md.

## Environment facts (checked read-only)
- **I am running on the Mac, not the GPU box.** The project lives at `~/rg-cacophony` (contains only `uv init` stubs: main.py, a pyproject with `requires-python >=3.13`, and an empty README). It is *not* its own git repo yet: its git toplevel is `~`. I will `git init` inside `rg-cacophony` so nothing gets committed to the home-directory repo.
- **GPU box** (`ubuntu@<tailscale-ip>` over Tailscale, ~/<ssh-key>.pem): A10G with 23 GB, driver 580 / CUDA 13.0, 4 vCPU, 15 GB RAM, 53 GB disk free, uv 0.12.23 in ~/.local/bin, system Python 3.12.3. No HF model cache yet.
- **ml-dev is busy.** It is at 99% utilisation running ladder-witnesses (tmux `wave1` and `reeval`). **This project gets its own A10 instead (your call).**
- **Quota blocker:** on-demand G/VT is 4 vCPUs in us-west-2, and ml-dev uses all 4. Spot is 0, and other regions are 0.

## Dedicated GPU (`rgc-gpu`)
1. **Quota request:** us-west-2, quota code `L-DB2E81BA`, raising 4 → 8 vCPUs. Either you file it, or I file it as the first approved step.
2. **Wait for approval.** Small increases are often approved within minutes to hours, but some go to a support case that takes 1–2 business days.
3. **Launch** a g5.xlarge cloned from ml-dev:
   - same DL AMI `<ami-id>` (OSS Nvidia driver, Ubuntu 24.04);
   - subnet `<subnet-id>` in us-west-2b, SG `<security-group-id>`, key `ml-dev`;
   - 100 GB gp3 volume, tag `Name=rgc-gpu`.
4. **SSH** over the public IP, which the SG already allows from this Mac (<home-ip>). Tailscale is optional; joining needs one auth-URL click from you.
5. **Cost:** about $1/hr. I **stop the instance at every stage stop** and start it when we resume. Termination happens only when you ask.

While the quota is pending, Stage 0's CPU checks and all of Stage 1 run on the Mac. The GPU sanity checks (determinism, TL-vs-HF on CUDA) run on rgc-gpu before Stage 2.

## Workflow
- **Mac:** the canonical git repo, where I edit and commit. I add a bare remote on the box (`ubuntu@…:rg-cacophony.git`) and the box runs from a clone of it.
- **Provenance:** every result file records the git SHA, a hash of the frozen config, the package versions, the GPU, and the seeds.
- **Jobs:** run in tmux session `rgc` on rgc-gpu. Large raw arrays stay in `results/raw/`, which is gitignored. Summaries and figures are rsynced back and committed.
- **Local env:** the same uv lockfile serves CPU unit tests and smoke tests on the Mac. All reported numbers come from the GPU.
- **No GitHub push** unless you say so.

## Repo layout
```
rg-cacophony/
  pyproject.toml  uv.lock  .python-version(3.12)  .gitignore  README.md
  PLAN.md  PREDICTIONS.md  DECISIONS.md  REPORT.md
  src/rgc/
    config.py       # ALL frozen analysis constants (θ grid, ε grid, k/p grids, k_heads, seeds); hashed into outputs
    determinism.py  # seeds, torch deterministic flags, TF32 off, CUBLAS_WORKSPACE_CONFIG, env capture
    models.py       # load(name, step) via TL no-processing; shuffled controls; registry
    stimuli.py      # load/validate ambiguous prompts, single-token check, cue/target token indices
    wikitext.py     # deterministic passage selection → manifest
    m1_order.py     # normalised residuals, sense directions, cross-validated d′, control correction
    m1_attention.py # per-head entropy (all keys / BOS-excluded), cue mass
    m1_ablation.py  # mean-ablation, random and layer-matched random-head nulls
    m2_perturb.py   # perturbation hooks, r(L,q) lattices, ε sweep, JVP cross-check
    pci.py          # binarisation, LZ76 (Kaspar–Schuster), Casali normalisation
    spread.py       # S(t), power vs exponential fits, AIC, z
    stats.py        # hierarchical bootstrap, permutation tests
    plots.py
  stimuli/ambiguous.json  stimuli/wikitext_manifest.json
  scripts/ s0_env_check.py s1_build_stimuli.py s2_m1.py s3_m2.py s4_compare.py s5_figures.py
  tests/   test_hooks.py test_lz76.py test_pci.py test_determinism.py test_causality.py test_shuffle.py
  results/ stage{0..4}/ (figs + summary json)  results.json  raw/ (gitignored)
```

## Models and controls
- **pythia-160m** (12 layers × 12 heads, d=768, parallel attention+MLP, rotary) at **steps 0, 512, 4000, 143000**. All of these are standard Pythia revisions.
- **Weight-shuffled final model:** every parameter tensor is permuted independently within itself, including embeddings, unembedding, biases and LayerNorm parameters. This is done on the raw HF weights before loading into TL, which preserves each tensor's multiset of values exactly. I use **3 shuffle seeds**, following your "seeds before negative claims" rule.
- **Loading:** `HookedTransformer.from_pretrained_no_processing`, in float32. That means no LN folding and no writing-weight centering, so ||resid|| is the true model's residual norm. BOS (`<|endoftext|>`) is prepended to every input. Both choices are logged in DECISIONS.md.
- **Determinism:** `torch.use_deterministic_algorithms(True)`, TF32 off, matmul precision "highest", and fixed seeds for the shuffles, directions and splits.

## Measurement 1: symmetry breaking

**Stimuli (Stage 1)**
- Candidate words: bank, bat, crane, pitcher, seal, spring, match, mole, pen, bark. A word is kept only if `" word"` is a single token in the Pythia tokenizer; this is checked programmatically.
- For each word: **30 prompts per sense** with the cue *before* the target, plus **12 neutral** prompts.
- Every prompt carries annotated cue-word spans, which map to token indices, and one of **4 fixed, sense-neutral continuations** of about 30 tokens after the target. M1 measures at the target only; causality makes the continuation invisible there. The continuation exists so the gating test has positions q > p.

**⚠ Context-leakage confound (I recommend changing the spec)**
- Even at step 0, near-uniform attention mixes the cue tokens' embeddings into the target position. A difference-of-means direction will then separate the senses whenever cue words recur between train and test. As specified, the sanity check "step 0 shows no sense separation" will very likely fail. That would be a real effect, not a bug, and d′ alone would not isolate *disambiguation*.
- Fix 1 (both fixes are pre-registered): a **lexically disjoint split**. Each sense's cue vocabulary is partitioned so the test cues never appear in train, giving a 20/10 train/test split per sense.
- Fix 2: **matched control-token prompts**. The same prompts are run with the target replaced by a single-token neutral noun (e.g. `" thing"`, verified as a single token).
- **Primary order parameter:** Δd′(L) = d′_target(L) − d′_control(L). Each term is fit on train, evaluated on test, and computed from `blocks[L].ln1.hook_normalized` (pre-affine LN output; `ln_final` for L=12).
- Raw d′ is reported for every model as a secondary measure.
- Neutral prompts: I report the fraction whose projections fall between the two class means.

**Attention (at the target query)**
- Normalised entropy over all keys (your spec, primary). Entropy with BOS excluded is reported as secondary, because sink heads have low entropy for reasons unrelated to the cue.
- Cue mass, plus cue mass relative to its uniform-attention baseline.

**Alignment test (quantified)**
- Breaking layer L* = argmax_L [Δd′(L+1) − Δd′(L)]. The "narrow band" is the smallest window of consecutive layers that contains ≥50% of the total rise.
- **Head selection:** top **k=6** heads by cue mass, selected on the train split.
  - Test 1: are they concentrated in the band? This is a hypergeometric test. A head in layer L writes into resid L+1.
  - Test 2: across all 144 heads, Spearman ρ between each head's direct contribution to separation (its output projected on the sense direction) and its cue mass (predicted > 0) and its entropy (predicted < 0). Significance is by permutation test.
- **Causal check:** mean-ablate the 6 heads at all positions. The mean is taken over train-split (prompt, position) tokens. Δd′ is then re-measured on test. The nulls are 200 random 6-head sets and 200 layer-matched random sets.

## Measurement 2: flow and the PCI analogue

**Input and perturbation**
- **Passages:** 64 WikiText-103-raw *test* passages, each the first 256 tokens of an article with ≥256 tokens, selected deterministically. The manifest records titles, line indices and a SHA256 of the token ids.
- **Perturbation:** add δ = ε·||resid_pre[k,p]||·u at the layer-k, position-p residual.
  - Grids: k ∈ {0…11}, p ∈ {16, 64, 128}.
  - u is one of 8 random unit directions, mean-removed because LN discards the mean.
  - For the gating runs, u is instead the M1 sense direction, on the ambiguous prompts.
- **Clean and perturbed rows run in the same batch**, so the ε=0 difference is bitwise zero.

**Response and linear regime**
- r(L,q) = ||Δresid||/||resid|| for L>k, q≥p. The self-column q=p is reported separately and **excluded** from PCI and spread. q<p must be exactly 0; this is a test.
- **Linear regime:** sweep ε from 1e-6 to 1 in half-decade steps on 8 calibration passages. ε* is the largest ε whose gain G=r/ε stays within 5% of the small-ε plateau.
  - The float32 noise floor sets the lower edge. A float64 spot-check and a forward-mode JVP give exact linear-response cross-checks.
  - ε* is chosen per model *before* PCI is computed. All ε are kept to show the nonlinear regime.

**PCI**
- **Binarisation threshold:** binarise the **gain** G, not r, so the result is ε-invariant in the linear regime. A deterministic network has no spontaneous-activity null like Casali's, so the threshold is a convention fixed now.
  - Primary θ = **0.1**: a site counts as reached when its relative change is ≥10% of the injected relative change.
  - Grid θ ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}, plus a threshold-free area under PCI(log θ).
  - Fallback rule, pre-registered and blind to model labels: if θ=0.1 makes ≥90% of pooled lattices from *all* models degenerate (p₁<1% or >99%), the primary moves to the nearest non-degenerate grid value.
- **Score:** flatten layer by layer (L=k+1…n). Each layer contributes its binary vector over q>p in natural order. Then PCI = c(n)·log₂n / (n·H(p₁)), with LZ76 by Kaspar–Schuster.
- **Primary PCI** is the mean over k ≤ n_layers−3, all p, and random directions.

**Dynamic exponent**
- S(t) = #{q>p : G(k+t, q) ≥ θ}, averaged over directions and fit per passage.
- Power law: log S = a + (1/z)·log t. Exponential: log S = a + t/τ. AIC is used to choose between them.
- Fits exclude S=0 points and saturated points (S ≥ 0.9 × available positions).
- Primary cell: k=0, p=64. Secondary: all cells, plus a threshold-free participation-ratio spread S_eff(t) = (Σ_q G)²/Σ_q G².
- **Physics caveat stated up front:** attention is non-local, so a single layer can reach every q>p. S(t) may saturate at t=1 for low θ or stay flat for high θ. If z turns out to be ill-defined, that is a legitimate outcome for prediction (d), not something to engineer around.

**Gating test**
- Use the ambiguous prompts with their continuations, and inject at the target position.
- Compare k = L*−2 with L*+1. Both cases use an **equal 3-layer downstream window**, so the "after" case is not penalised for having fewer layers. Directions are random and the sense direction.

**CIs:** hierarchical bootstrap over passages or prompts and then directions, 10k resamples, giving percentile 95% CIs.

## Pre-registration (PREDICTIONS.md, committed at the end of Stage 1, before any model comparison)
Draft of how each prediction is made testable (to be refined with you at the Stage 1 stop):
- **(a)** In step 143000, all of the following hold:
  - the narrow band is ≤3 layers wide and peak test Δd′ has a 95% CI above 0;
  - the top-6 cue-mass heads are over-represented in the band (p<0.05), and ρ(cue mass, contribution) > 0 (p<0.05);
  - ablating them cuts final-layer Δd′ by ≥50% *and* by more than the 95th percentile of both random nulls.
- **(b)** For step 0 and for each shuffle seed: either the Δd′ peak CI includes 0 or the peak is less than 20% of the trained peak, *and* top-6 ablation does not differ from the random null (p>0.05). I also expect raw d′ at step 0 to be >0, which is the leakage effect described above, and say so in advance.
- **(c)** The primary PCI satisfies trained > step 0 and trained > every shuffle seed, with the bootstrap CI of each difference excluding 0. For the early checkpoints, PCI_control < PCI_512 and PCI_4000 < PCI_trained, with CIs on the differences.
- **(d)** At the primary cell and θ: the across-passage CV of z is ≤0.25, the 95% CI half-width is ≤15% of z, and the median R² is ≥0.9. The winning functional form is reported separately. If the exponential wins by ΔAIC>4 in most fits, I will say that z is stable but a power law is disfavoured.
- **Gating:** this prediction is yours. Without a direction from you, it is labelled *exploratory*.

## Stages (I stop at the end of each, ask for your prediction, and only then show results)
0. **Environment and sanity:** git init, uv env with exact pins (torch, transformer_lens, numpy, scipy, pandas, matplotlib, datasets), and a lockfile. Tests:
   - hook shapes;
   - TL-vs-HF logits match;
   - two runs are bitwise identical;
   - ε=0 gives exactly zero response;
   - q<p gives exactly zero response;
   - shuffled tensors are permutations of the originals;
   - LZ76 matches textbook sequences.
1. **Stimuli:** tokenizer verification, the written prompt set with disjoint cue splits, control-token versions, and the WikiText manifest. Then PREDICTIONS.md goes to you for sign-off and is committed.
2. **M1 on step 143000:** d′ / Δd′ across depth, entropy and cue-mass heatmaps, the alignment statistics, and the ablation with its nulls.
3. **M2 on step 143000:** ε calibration, r(L,q) lattices, PCI across θ, the S(t) fits and z, and the gating test.
4. **Controls and comparisons:** all of M1 and M2 on step 0, 512, 4000 and the 3 shuffles, plus the scorecard against PREDICTIONS.md.
5. **Write-up:** REPORT.md (what was found, what failed, caveats), results.json, and figures. Before citing arXiv:2609.35618 I will check that it exists and that its "grain" framing is described correctly.
- **Stretch, only after Stage 4 and only on your go:** M2 on pythia-410m and pythia-1b at the final checkpoint plus step 0, testing whether z holds up across scale.

## Compute estimates (dedicated A10G)
- **Downloads:** about 0.35 GB for each 160M revision. Stretch: about 0.9 GB for 410M and 2 GB for 1B.
- **Stage 0–1:** minutes.
- **M1 per model:** about 1.2k short prompts, plus about 80k ablation forwards, about 3–5 min. All 7 model variants take about 30 min in total.
- **M2 per model:** the calibration grid (8×12×3×4×13 ≈ 15k forwards of 257 tokens) plus the main grid (64×12×3×8 ≈ 18k) plus gating, about 3–6 min. All 7 variants take about 40 min in total. Raw lattices are about 250 MB per model and are gitignored.
- **Stretch:** about 10–25 min per model.
- **Total:** about 1–1.5 GPU-hours of compute (about $1–2), plus instance idle time while I develop, which stopping at each stage keeps down. No single job should pass 30 min, and I will ask before any job that might, including the stretch.

## Verification
- `uv run pytest` passes on the Mac (CPU) and on the box (GPU).
- `s0_env_check.py` writes `results/stage0/sanity.json` with every check from Stage 0, and that file is committed.
- Each stage script is idempotent, keyed by config hash and SHA. Re-running it reproduces the summary JSON bitwise.
- `results.json` validates against a schema listing every pre-registered number with its CI.

## Needs your sign-off (on approval of this plan)
1. Making **Δd′ (control-corrected, disjoint cue split)** the primary order parameter, with raw d′ secondary.
2. Primary θ = 0.1 on gain, with the degeneracy fallback rule. Primary PCI cell definition; primary z cell k=0, p=64.
3. Who files the quota request (`L-DB2E81BA`, 4 → 8 vCPUs, us-west-2); I will file it on approval unless you already have. Also the stop-at-every-stage policy for rgc-gpu.
4. A direction for the gating prediction, or leaving it exploratory. This can wait until the Stage 1 stop.
