Neuroscientists have a test for consciousness that does not depend on anything the patient says. You give the brain a small magnetic zap and listen to the echo with EEG. In an awake or dreaming brain, the echo spreads widely *and* stays differentiated: different regions answer in different ways. In deep sleep, under anaesthesia, or in a vegetative state, the echo either stays local or spreads as one uniform wave. Compress the echo and the difference becomes a single number, the Perturbational Complexity Index (PCI), introduced by Casali, Massimini and colleagues in 2013.

I built a version of that test for language models, and it behaves the way the clinical one does.

## Why language models need a zap test

Asking a language model whether it is conscious mostly tells you about its training data. Chandaria, Seth, Shevlin, Shanahan, Rosas and colleagues make this point carefully in [*From cacophony to hierarchy*](https://arxiv.org/abs/2609.35618): a system optimised to reproduce human behaviour and report will produce those signatures whether or not the underlying states are present. That confound has no analogue in biology. Their framework therefore looks for indicators deeper than behaviour. One level they single out is *intrinsic causal structure*, which is exactly where PCI lives.

So the question is whether a PCI-style measurement can be defined for a transformer, measured without looking at any output, and shown to respond to the kinds of changes that make clinical PCI meaningful.

## The zap test for a transformer

- **Zap.** Pick one word in a passage and one layer of the network. Give the model's internal state at that spot (the residual stream) a tiny random push.
- **Listen.** Run the rest of the computation and record how much every later word, at every later layer, moved.
- **Mark.** Mark each spot that moved by at least 1% of the push.
- **Compress.** The result is a map, like the ones below. Compress it with the same Lempel–Ziv method used for clinical PCI. An empty map or a uniform map compresses well and scores low. A widespread *and* irregular map scores high.

Nothing in this procedure looks at what the model writes.

![Four example echo maps: trained, anaesthetised, untrained, scrambled](/content/images/2026/10/zap-fig1-echo-maps.png)

*One typical zap in each of four systems; each panel shows the zap closest to its condition's average. A trained model (Pythia-160M) answers with scattered, word-specific streaks that persist through depth. The same model with attention flattened answers in uniform bands. An untrained network barely answers beyond the zapped word's neighbourhood. A network with scrambled weights floods almost everything: the "seizure-like" pattern.*

## The anaesthesia test

What makes clinical PCI meaningful is how it behaves when consciousness is switched off while the brain stays active. Under anaesthesia, cortex still responds to the zap, but the response loses its differentiated structure, and PCI falls.

A transformer has a natural analogue. Attention has a "temperature" that controls how selectively each word attends to the others. Turn it down, and every word attends to its whole context equally. The push still reaches everything, but it no longer reaches anything *in particular*. If the zap test measures what clinical PCI measures, its score should fall even though the push still spreads.

I first saw this pattern in exploratory data. I then wrote down four predictions, committed them publicly, and re-ran the test on 64 passages of text that had never been analysed. All four predictions held:

- **PCI falls** by about a third as attention is flattened (0.65 → 0.42).
- **Reach barely changes:** the push still moves about a fifth of all downstream spots.
- **The threshold-free spread grows** more than threefold. Flattening spreads the echo more evenly; it doesn't silence it.
- **The dose-response is monotone:** each step of flattening lowers PCI further. A version of the score that holds the number of active spots fixed falls even further, so the drop is not an artefact of how many spots light up.

![PCI falls while reach holds and spread grows as attention is flattened](/content/images/2026/10/zap-fig2-anaesthesia.png)

*Pre-registered replication on fresh text. As attention is flattened (β from 1, as trained, to 0, uniform), the echo's complexity falls while its reach barely moves (axis from 0) and its spread grows. That dissociation is the signature of anaesthesia in the clinic.*

In other words, the zap test tracks *differentiated integration*, not how far a perturbation travels. The replication came back almost number for number: 0.657 → 0.648 for the intact model, and 0.424 under full flattening both times.

## Four models, new text

A single model can mislead. So in a further pre-registered study, on another set of passages nobody had analysed, I added Pythia-410M and Pythia-1B, and GPT-2 small, which comes from a different lab, was trained on different data, and uses a different tokenizer. There, PCI across attention temperatures was a pre-specified secondary measure: reported, not scored. Read it as strong supporting evidence rather than a confirmed prediction.

It came out cleanly. Flattening attention to a quarter of its trained sharpness lowered PCI in all four models, with no overlap between the confidence bands at the two ends: 0.65 → 0.46 in Pythia-160M, 0.47 → 0.36 in Pythia-410M, 0.47 → 0.36 in Pythia-1B, and 0.53 → 0.26 in GPT-2. Reach held steady in the Pythia models and rose in GPT-2.

![PCI falls as attention is flattened in all four models; reach holds](/content/images/2026/10/zap-fig3-four-models.png)

*Pre-specified secondary measure from a pre-registered study on 32 new passages; shaded bands are 95% confidence intervals over passages. Left: PCI falls as attention is flattened (β below 1) in every model. Right: reach holds. In the Pythia models it stays flat. In GPT-2 it more than doubles while complexity halves: the echo floods the network without differentiating, the seizure-like end of the same dissociation.*

## A map of states

Plotting every condition by reach (how widely the echo spreads) against PCI (how differentiated it is) gives a picture any clinician would recognise.

![Reach versus PCI for every condition](/content/images/2026/10/zap-fig4-state-map.png)

*Trained models sit at moderate reach and high complexity. Flattened attention keeps reach and loses complexity, like anaesthesia. Restricting attention to a local window shrinks reach. Untrained networks are low on both. Scrambled Mamba (a recurrent architecture) reaches almost everything with low complexity, like a generalised seizure.*

The map also shows the zap test's blind spot: scrambled *transformers* sit near the top. Compression rates randomness as complexity. Neuroscience has the same issue, and clinical PCI sidesteps it by comparing states of the same brain rather than different brains. A second reading of the same zap can help, at least in smaller models.

## Does the echo fade, hold, or explode?

From the same maps, I measured how the push grows or shrinks as it travels deeper: a growth exponent λ per layer.

- **Negative λ:** pushes die out. The system is ordered.
- **Positive λ:** pushes amplify. The system is chaotic.
- **λ near zero:** the system is poised between the two. That is the edge-of-chaos signature that criticality researchers associate with the awake brain.

In Pythia-160M, a pre-registered test found the trained model's λ almost exactly zero at the temperature it was trained at (−0.0005), while untrained, partially trained and scrambled copies all amplified. An exploratory sweep then showed that flattening attention pushes the trained model into the ordered regime and sharpening it pushes it into the chaotic one, so λ crosses zero right at its natural temperature. Because that crossing was first seen in exploratory data, I pre-registered it and tested it on new text in all four models.

![Growth exponent across attention temperatures in four models](/content/images/2026/10/zap-fig5-edge-of-chaos.png)

*Pre-registered test on new text. Coloured lines: the trained model across attention temperatures. Grey dashed lines: the same model with scrambled weights. Pink diamonds: the untrained network. Shaded bands are 95% confidence intervals over passages. In Pythia-160M and GPT-2 small, the trained model's λ crosses zero at its natural temperature (dotted line) while the controls stay positive. In Pythia-410M and 1B, the trained models amplify at every temperature.*

The result is split. The crossing replicated in Pythia-160M on new text and in GPT-2 small: in both, λ at the natural temperature is within about 0.01 of zero, negative just below it and positive just above. (GPT-2's curve also turns back up when attention is strongly flattened, which I had not predicted.) The crossing did not replicate in the larger Pythia models. At 410 million and 1 billion parameters, the trained models amplify pushes at every temperature tested (λ ≈ +0.07 and +0.09 at their natural temperature). They are still closer to zero than their scrambled and untrained copies, but only modestly so, and they are not poised. I don't yet know why criticality shows up in the small models and not the larger ones; that is the obvious next question.

So the edge-of-chaos reading is promising but depends on scale, and I would not lean on it alone. Where it holds, it patches the complexity measure's blind spot: rich *and* poised is the trained model, rich but chaotic is random weights. The robust result is the complexity one above.

## Why this could be useful

- **It reads causal structure, not behaviour.** The test never looks at what the model says, so it is immune to the imitation confound that makes behavioural evidence so hard to interpret.
- **It is a recipe, not a model-specific probe.** It pokes the residual stream and reads the residual stream, so it applies to any architecture that has one. I have run it unchanged on Pythia models from 160 million to 1 billion parameters, on GPT-2, and on Mamba, a recurrent architecture that is not a transformer at all. It is cheap, and the code is open.
- **Its strength is comparing states of one system,** which is exactly how clinical PCI earned its meaning. Natural next comparisons include:
  - a base model against its fine-tuned or RLHF'd versions;
  - checkpoints across training (in our data, the untrained network's echo is far less complex than the finished model's at every size tested);
  - quantised or pruned variants;
  - different context lengths or prompts.
- **For a framework like Cacophony's, it offers one operational indicator at the intrinsic-causal-structure level** that can be computed for any open-weight model, with calibration logic borrowed directly from the clinic.

## What it is not

It is not a consciousness detector. Clinical PCI earned its meaning by being calibrated against states we know independently to be conscious or not, in human beings. No such ground truth exists for machines. What the zap test offers is narrower and testable: a well-defined, reproducible measurement of differentiated integration and of criticality that behaves like its clinical counterpart under the manipulations we can perform. The core results are at 160 million parameters, with a follow-up up to 1 billion, all on one kind of text (Wikipedia articles) and with synthetic manipulations. And the criticality reading holds only in the smaller models.

## Code, data, and pre-registrations

Everything is public at [riemannzeta/rg-cacophony](https://github.com/riemannzeta/rg-cacophony): the code, the raw perturbation data, every pre-registration as a tagged commit made before its data existed, a full technical report, and a log of every analysis decision. The work was carried out with Claude (Anthropic) as a research assistant.
