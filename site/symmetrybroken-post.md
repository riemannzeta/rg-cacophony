*Short version:* I tried to build two indicators of the kind a recent paper on AI consciousness asks for: measurements taken from *inside* a language model, immune to the fact that these systems are trained to sound like us. I pre-registered predictions five times, with controls and fresh data each time. Most of my original ideas failed. But one instrument, a transformer analogue of the clinical “zap-and-listen” test, behaved like its neuroscience counterpart under an anaesthesia-like manipulation, and the replication on fresh data passed every prediction. A second measurement put trained models right at the edge between order and chaos, where untrained and scrambled networks are not. Everything (code, data, pre-registrations, every deviation) is public.

## Why measure from the inside?

Chandaria, Seth, Shevlin, Shanahan, Rosas and colleagues, in [*From cacophony to hierarchy*](https://arxiv.org/abs/2609.35618), separate theories of consciousness by which “grain” of functional description they take to matter: behavioural, computational, intrinsic causal-structure, organismic, or organism–environment. They point out a confound with no biological analogue: a system optimised to reproduce human behaviour and report will produce those signatures whether or not the underlying states are present. That puts weight on indicators below the behavioural level.

Following my reading of the forward pass as a renormalisation-group flow ([earlier essays](https://www.symmetrybroken.com/)), I built two:

- **A “decision moment” (computational level).** When the model reads an ambiguous word (“bank”), does its internal representation *snap* to one meaning at a particular depth, like symmetry breaking?
- **An “echo test” (intrinsic causal-structure level).** This copies the Perturbational Complexity Index used in the clinic (Casali, Massimini et al., 2013): give the brain a tiny magnetic zap and measure how rich the echo is. Here I give the model’s internal state a tiny push at one word and one layer, record how every later word at every later layer moves, and compress that pattern. A silent echo, or one that is identical everywhere, scores low. A widespread *and* differentiated echo scores high.

## How

The main subject was Pythia-160M at four points in training, with weight-*shuffled* copies as controls (same numbers, no learned structure), plus Pythia-410M/1B, GPT-2, and Mamba, a recurrent model trained on the same data as Pythia. Every comparison was pre-registered before its data existed: five rounds, each a tagged git commit. Each new test ran on *fresh* text no analysis had touched. Every deviation is logged. Runs are bitwise reproducible.

## What held up

### 1. The echo responds to “anaesthesia” the way clinical PCI does

In the clinic, PCI falls under anaesthesia and in deep sleep, when activity persists but stays local and stereotyped, and in generalised seizures, when activity is widespread but uniform. In a transformer, attention has a “temperature”. Turning it down makes every word attend to its whole context equally. The push still reaches everything, but it no longer reaches anything *in particular*.

![PCI falls, reach barely changes, and threshold-free spread grows as attention is flattened](/content/images/2026/10/fig3_anaesthesia_replication.png)

*Fresh-data replication (pre-registered). As attention is flattened (β from 1 to 0), the echo’s complexity falls by about a third. The share of sites the push reaches barely changes (note the axis starts at 0), and the threshold-free spread of the response more than triples. A density-matched version of PCI, which holds the number of active sites fixed, shows the same drop. All four predictions passed.*

So the echo measure tracks *differentiated integration*, not mere reach. This was first seen exploratorily, then re-tested on 64 new passages with predictions written down in advance. The numbers came back almost identical (0.657 → 0.648 for the intact model; 0.424 both times under full flattening).

![Reach versus PCI for every condition](/content/images/2026/10/fig2_integration_differentiation_plane.png)

*Where each condition sits. Trained models have moderate reach and high complexity. Flattened attention keeps reach but loses complexity. Narrow attention windows lose reach. Untrained networks are low on both. A shuffled Mamba is “seizure-like”: its push spreads almost everywhere, uniformly, with low complexity. The shuffled Pythias reveal a blind spot: random networks also look “complex”.*

### 2. Trained models sit at the edge of chaos

From the same echo maps I computed how a push grows or shrinks with depth: a per-layer growth exponent λ. It is negative if pushes die out, positive if they amplify, and near zero if the system is poised between the two, which is the edge-of-chaos signature criticality theorists look for in the awake brain.

![Growth exponent λ: trained near zero, controls positive; λ crosses zero at the natural temperature](/content/images/2026/10/fig4_growth_exponent.png)

*Left (pre-registered, passed): at its natural temperature the trained model’s λ is −0.0005. Untrained, early-training and shuffled networks are all clearly positive. Right (exploratory): as attention is flattened λ goes negative (pushes die out), and as it is sharpened λ goes positive (pushes amplify). It crosses zero almost exactly at the temperature the model was trained at. The shuffled model never crosses.*

> **Confirmation run (pre-registered across scale and model family):** [PREREG-5 RESULT: Pythia-160M on fresh text, Pythia-410M, Pythia-1B and GPT-2. To be filled in when the runs complete.]

One honest qualification. Where exactly the crossing lands depends on how λ is averaged. The pre-registered, Lyapunov-style estimator puts it at β ≈ 1. A curve-of-means estimator puts it nearer 1.6. Mechanistically, the trained model balances an early build-up of influence against late-layer contraction. The controls show no late contraction.

![Growth curves of a perturbation's footprint over depth](/content/images/2026/10/fig6_growth_curves.png)

*Exploratory. How a push’s cross-word footprint evolves with depth, for a push at the embedding and one mid-depth. Each curve is labelled with its own slope.*

## What did not hold up

- **The “decision moment”.** The model does separate word senses beyond simple context leakage, and untrained and shuffled controls do not. But the separation appears at the *first* layer, plateaus, and fades. It is carried by heads that attend broadly, not sharply. With conflicting clues (“between the canoe and the mortgage… the bank”) the representation integrates evidence almost linearly instead of committing. That is a graded response to a field, not symmetry breaking.
- **A spreading speed.** I expected the push to ripple outward with depth at a measurable rate. Attention connects every word to every earlier word in one step, so there is no front to track.
- **Chimera states.** Shanahan’s chimera index did not single out trained models (random ones score higher), and it fell steadily as attention sharpened rather than peaking.
- **Recurrence.** Mamba, a recurrent model trained on the same data, scored about the same as Pythia on the echo test.

![Word-sense separation across depth](/content/images/2026/10/fig1_m1_sense_separation.png)

*The decision-moment test. Sense separation beyond context leakage, across depth. It rises at the first layer and fades late. Controls are flat at zero.*

![PCI and chimera index across attention temperature](/content/images/2026/10/fig5_temperature_sweep.png)

*Across temperature. The echo’s complexity rises from the dissolved side and peaks slightly above the natural temperature. The predicted fall on the frozen side was small and could not be measured cleanly: at the sharpest setting the response is no longer smooth. The chimera index simply falls.*

## Scorecard

| Prediction | Result |
|---|---|
| Sharp mid-depth “decision moment” carried by focused, clue-watching heads | **fail** |
| No decision moment in untrained or shuffled models | **pass** |
| Trained echo richer than untrained and shuffled, early checkpoints in between | **pass** at the pre-registered threshold; reverses against shuffled at the sensitive threshold |
| A stable spreading exponent | **fail** |
| Anaesthesia lowers the echo more than its reach (original threshold) | **fail**; local response survives: **pass** |
| Conflicting clues: saturation and commitment | **fail**; controls null: **pass** |
| Fresh-data anaesthesia replication (four predictions) | **4 / 4 pass** |
| Trained model closest to the edge of chaos | **pass** |
| Inverted U of the echo across temperature | not testable as registered |
| Chimera states: trained > controls; inverted U | **fail**, **fail** |
| Mid-depth “waist” of compression | **pass** (borderline) |
| Edge-of-chaos crossing replicates across scale and family | [PREREG-5] |

## What this might mean for the Cacophony framework

- **Intrinsic causal-structure indicators are well defined in transformers and can be measured without looking at outputs.** They respond to within-system manipulations the way their clinical counterparts do. They also share their counterparts’ main limitation: complexity measures rate random dynamics as rich. In the clinic that never arises, because PCI compares states within one brain.
- **Criticality was the more discriminating axis.** The growth exponent separated trained from random networks where complexity and chimera measures did not. It also placed the trained model at the order–chaos boundary of its own temperature family.
- **The computational-level “commitment” picture did not survive.** At least at this scale, transformers integrate word-sense evidence gradually rather than through an ignition-like event.

> None of this is evidence for or against consciousness in language models. It tests whether candidate indicators are well defined, robust, and able to tell trained systems from controls. Scale was mostly 160M parameters, one text domain, and synthetic manipulations.

## Code, data, and every decision

Everything is on GitHub at [riemannzeta/rg-cacophony](https://github.com/riemannzeta/rg-cacophony): the five pre-registrations (tags `prereg`–`prereg5`), the mechanical scorecard, a full report, the deviation log, and the raw perturbation data as release assets. The research was carried out with Claude (Anthropic) as a research assistant.
