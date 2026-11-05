# SLWM research findings — updated 2026-11-04

## Current decision — G1/G2 generation follow-up

**Keep the small core for timing alignment, preserve observed signals through
an explicit output path, and retain the cheaper raw residual audio head for
reconstruction. Do not scale this same recipe or claim useful cross-modal
generation.** Fifteen additional RTX 3090 runs now separate signal preservation,
output quality, appearance conditioning and core generalization.

G1 trains eleven heads on the frozen 4,741,089-weight M1 core: five
548,352-weight 100 Hz audio heads and six 213,699-weight visual heads conditioned
on ONE actual seed frame. G2 trains four signal-preserving residual audio heads,
raw versus frozen-core conditioning, in two matched seeds. Audio arms see
51,200 sampled 1.28s windows; visual arms see 25,600, matched within each task.
Total decoder training is about 6.8 GPU minutes, excluding preparation/scoring.
Cached frozen fields can include context outside a decoder crop, while raw
corrections see the crop directly. Core arms also pay extra extraction compute.
These comparisons match decoder/data/update budgets, not total compute or
identical temporal receptive fields.

All use unchanged 5,061 train / 210 validation / 240 test CREMA-D clips with
person-disjoint splits. The CREMA test is now reused. G1 freezes a new 64-clip,
four-person RAVDESS check with two new utterances, without external fitting,
normalization or selection. G2 responds to G1's external failure, so its external
scores are explicitly reused diagnostics rather than untouched confirmation.

### What the new experiments establish

- **A useful audio edge:** interpolation of observed audio plus bounded learned
  raw-feature corrections reduces test mel MSE **0.03024 → 0.02566 / 0.02563**
  in two seeds (15.1–15.2%). External error falls **0.06220 → 0.05179 / 0.05215**
  (16.2–16.7%). All-clip person-cluster mel gain intervals are positive.
- The matched core corrections have nearly identical CREMA error, but external
  error **0.06822 / 0.06794** is worse than interpolation. They add 4.74M frozen
  weights without reliable benefit. The raw-residual pilot gate passes in both
  seeds; frozen-core and core-benefit gates fail.
- Independent recognizer WER for interpolation versus raw residual heads is
  **73.2% → 49.0% / 48.2%** on 32 fixed CREMA cases, and **22.9% → 8.3% / 10.4%**
  on 16 external cases. External WER intervals include zero with four people;
  these diagnostic means are not a human intelligibility benchmark. CREMA's
  alphabetical subset concentrates on one prompt. Reconstruction is not novel
  speech generation or model transcription.
- **Core detail loss on a new domain:** G1's new frozen-audio-field head improves
  CREMA MSE **0.1645 → 0.0843** versus the old head on the same 100 Hz grid.
  External MSE is **2.0798**, versus **0.2667** for a matched raw-audio decoder;
  diagnostic WER is 102.1% versus 14.6%. A richer 100 Hz input diagnostic has
  good recognizer scores but poor level-sensitive external MSE: recognizer
  success and signal fidelity are different properties.
- **Video→speech fails:** test mel MSE **0.8453** for the new frozen-video-field
  head, **0.8367** for raw video and **0.7464** for the old interpolated head.
  Diagnostic WER remains near 100%, without useful wrong/null-video advantage.
- **Sharper images do not establish audio-driven motion:** one real seed plus
  audio reduces mouth MSE 21.7% versus copied-seed persistence, but a time-only
  head matches it. Correct versus wrong audio helps 3.67% [1.48, 6.63], yet the time-only control explains
  the apparent improvement. Predicted motion energy is 2.6% of recorded energy.
  Both full-model seeds fail the visual gate, including externally.

- **Partial timing transfer survives:** the original frozen core selects the
  correct shift on **53.1%** of 64 RAVDESS clips, person-cluster interval
  [32.8, 76.6], against 14.3% seven-way chance. This additional exploratory audit
  supports narrow correspondence, not successful generation.

An explicit evaluation erratum disables the shuffled-training flag when reading
EXP-T2-415 with correct test audio. Original metrics/media and the original
derived report are preserved; corrected readouts are separately named. Its
checkpoint is not retrained or reselected. The decisive time-only comparison
and all full-model gates are unaffected.

All generated media are actual predictions, with original/codec/interpolation,
copied-seed/time-only/shuffled, wrong-input and null controls. No pretrained
generator or source waveform supplies model output. Recognizer text is evaluation
only. One real seed and the entire simultaneous audio window are observed;
there are no autonomous future rollouts or policy/uncertainty claims.
**104 repository tests pass**, plus real-file CPU checks for all three input/output
paths. Video outputs are identical for pixel-identical videos with different
soundtracks; the visual first frame matches only the supplied seed.

All 15 checkpoints, source/config/data/checkpoint hashes, optimizer states,
curves, complete metrics, failure cases and actual WAV/GIF/WebM artifacts are
retained locally and on Fedora. See [G1 report](../../experiments/multimodal/t2/conditional-generation-20261104/report.md),
[G2 report](../../experiments/multimodal/t2/conditional-residual-20261104/report.md),
[G1 protocol](conditional_generation_protocol_20261104.md),
[G2 protocol](conditional_residual_followup_20261104.md), and
[reproduction/inference](../training/conditional_generation.md).

### Next allowed research step

Retain the raw residual edge as a baseline. Test a core objective that preserves
phonetic/spectral information across more people, recording channels and corpora,
while retaining timing alignment. Keep explicit raw bypass and matched raw
decoders; require a core advantage on a domain that has not guided architecture
choices. Use a controlled motion target and stronger dynamics baseline before
adding capacity. Text/code grounding, general generation and commitment remain
separate unsolved stages.

## Previous M1 decision and results (preserved)

**Retain the small motion-aware alignment core; improve generative heads before
increasing its size again.** The new physical-time recipe fixes a measurable
modality-matching problem, while generation remains incomplete. The user
explicitly authorized this redesign/scaling/exploration after the earlier
no-scaling result; none of the earlier negative results were overwritten.

Ten new registered RTX 3090 experiments (`EXP-T2-301`–`310`) compare
**4,741,089** and **15,232,225** trainable-parameter models, identical-feature
Transformer baselines, legacy packing, removal of local alignment, shuffled
pairs, two larger-model seeds, and a no-spectral arm. No pretrained neural
encoder supplies the learned representation. The training-only codec has
249,248 separately counted PCA/statistical values.

Data: 5,511 usable real CREMA-D recordings (5,061 train / 210 validation /
240 test), from 62 / 7 / 8 disjoint people. All eight test people were untouched
in earlier cycles; all earlier test people stay excluded from training.
Use 25 Hz physical timestamps, mouth appearance/motion features, random
1.28-second windows and same-recording shifted negatives. This is simultaneous
missing-modality translation and alignment, not causal future prediction.

## What improved

- The small model selects the correct timing offset on **95.5%** of 202 eligible
  new-person clips, versus **36.1%** for untrained audio-energy/motion correlation
  and **14.3%** chance among seven valid equal-length candidates. The two larger
  signal models score **93.6% / 94.1%**.
- The small no-local-alignment arm falls to **34.2%**, and shuffled pairs to
  **7.9%**. Legacy packing still scores 94.6%; the evidence favors improved data
  and the local objective more strongly than a unique advantage of the new packing.
- For the small full model, reversed audio drops timing accuracy to **18.3%**,
  another clip from the same person to **17.3%**, and zeroed audio to **3.5%**.
  The larger models pass the same person-cluster input-dependence checks. An
  envelope-only input retains 63.9–72.8%, so some alignment is coarse timing,
  while full spectral/temporal sound contributes beyond that diagnostic.
- Larger-model video→audio mel error is **0.6295 / 0.6320**, versus small-model
  **0.6563** and training-mean **1.0550**. Correct video improves error about
  **34%** over another clip from the same person, with positive person-cluster
  intervals. These are feature predictions, not proof of intelligible speech.
- Frozen closed-set probes on the selected larger model recognize **41.7%** of
  the 12 known sentence prompts from video, versus **14.6%** for a raw-feature
  probe and 24.2% for the matched larger vanilla core. Audio recognition is
  **86.2%** (raw-feature control 84.6%). These are supervised probe readouts,
  not general transcription or text generation.

## What did not improve

Increasing parameters 3.2× does not improve timing alignment in the controlled
seed-41 comparison: −1.98 percentage points, cluster interval [−5.53, +1.45].
It improves video→audio MSE only 4.1% at about 1.42× measured training time.
The larger models have two seeds; the small model has one, so this is a pilot
comparison, not a scaling law. The larger representation helps the bounded
recognition probe, but adding capacity does not solve generation.

**Audio→motion still fails the frozen gate.** Its error remains close to the
training mean, and correct audio improves over wrong audio by only about 0.4%.
The audio-conditioned mouth images tend toward average appearance; unknown
identity cannot be recovered from speech alone. Spectral/motion matching does
not by itself produce a useful generative head.

The gallery contains actual model-generated audio and mouth sequences, original
media, codec reconstructions, same-modality reconstructions and no-input controls.
An independent, evaluation-only Whisper-tiny.en recognizer reports selected-
gallery mean word-error rates of 34.7% on original audio, 27.8% on codec audio,
83.0% on audio reconstructions and **118.5% on video-conditioned audio**. WER
can exceed 100% through insertions. This small, fallible recognizer is a
diagnostic, not a human quality score, and never supplied generated audio.
The outputs do not reliably recover the words.

A separately fitted frozen-field linear decoder improves some observed-signal
reconstruction (larger-model audio MSE 0.0715 and visible-mouth appearance MSE
0.4909) but does not fix audio-conditioned motion. Its additional 221,760 fitted
weights are counted separately. Original heads, checkpoints and registered
metrics remain intact. This was explicitly post-hoc exploratory diagnosis.

## New artifacts and validation

- [Full architecture/scaling report](../../experiments/multimodal/t2/motion-alignment-20261104/report.md).
- [Conditional output gallery](../../experiments/multimodal/t2/motion-alignment-20261104/exploration/gallery.md).
- [Input-dependence controls](../../experiments/multimodal/t2/motion-alignment-20261104/input_dependence.png).
- [Small checkpoint](../../experiments/multimodal/t2/motion-alignment-20261104/EXP-T2-301/best.pt) and
  [selected larger checkpoint](../../experiments/multimodal/t2/motion-alignment-20261104/EXP-T2-308/best.pt).
- [Research protocol](motion_alignment_protocol_20261104.md) and
  [reproduction / standalone audio or video inference](../training/motion_alignment.md).

All ten runs, primary-source snapshots, configs, optimizer/RNG states, hashes,
curves, per-person/clip metrics, fitted probes, generated media and controls are
retained. Training-loop time totals 20.8 minutes, with preparation and diagnostics
extra. **96 regression tests pass**, including strict missing-modality access
and deterministic checkpoint/output checks.

The next objective is a conditional generative decoder with an explicit visual
starting state and uncertainty, while retaining this small alignment benchmark.
Confirm broader claims on a second audiovisual domain. Timing alignment is
supported in this narrow speech setting; general world modeling, physical
forecasting, realistic generation, policy and hallucination reduction are not.

---

# Earlier appearance-feature cycles — 2026-11-04

## Decision

**The preceding appearance-feature recipe failed its scale/translation gates.**
Small SLWM models learn real audio-feature continuation and beat the matched
Transformer by about 7–8% in the discovery cycle and 6.8% in the residual
follow-up. That is limited evidence for a useful audio predictor. Correctly
paired video does not add a reliable benefit, shuffled-pair training performs
as well or better, and neither forecast recipe beats visual persistence.
A useful multimodal world model has not been demonstrated.

## What was actually trained

Eighteen registered GPU runs completed on `db@192.168.1.200`, an RTX 3090.
Each full SLWM has **4,471,425 trainable parameters**, including 4,256,256 in
the processor and all adapters/embeddings/heads. Vanilla has 4,433,537;
the difference is below 1%. A shared frozen ImageNet ResNet18 feature encoder
adds **11,176,512 deployed parameters**, with external pretraining disclosed.

The data are genuine paired human speech video and audio from
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), pinned at
`1658cd342dff90010aa843eaeebd53610a08b1dc`. There are 1,649 distinct usable
recordings across both cycles: 1,019 training, 210 validation, 210 discovery
test and 210 fresh follow-up test. Training/validation people never enter
either test set; the two test sets also have different people. This is about
one hour of constrained acted speech, not diverse world experience.

Inputs are 32 synchronized normalized-time steps of 64-bin log-mel audio and
training-only PCA-compressed frame features. The model sees the first half,
updates a shared continuous latent field, and predicts hidden future features.
Audio FFT windows cannot cross the context boundary; frames are encoded
independently. Transcripts, emotions and identity are metadata only.
Text/code, raw-media decoders, uncertainty/policy behavior and persistent
memory were not trained in these cycles.

Both cycles ran nine arms: full SLWM, vanilla Transformer, no spectral,
no long convolution, shuffled-pair training, audio-only, video-only, and a
second seed for full/vanilla. Each arm received 600 steps × 32 clips.
Validation alone selected checkpoints. Neural train-loop time totaled
414.3 seconds (6.9 minutes), with preprocessing and final diagnostics extra.
Full models cost about 1.5 times the vanilla wall time. These comparisons
match parameters approximately and data/optimizer steps exactly; they do
not establish an advantage at equal wall time or FLOPs.

## Results

| Cycle and held-out people | Full audio MSE, seeds 41 / 42 | Vanilla audio MSE | Full video MSE | Video persistence | Decision |
|---|---:|---:|---:|---:|---|
| Absolute-feature discovery | 0.4569 / 0.4490 | 0.4968 / 0.4861 | 1.0764 / 1.1113 | 0.4650 | Multimodal gate failed |
| Residual forecast, fresh people | 0.5136 / 0.4791 | 0.5512 / 0.5141 | 0.4152 / 0.4067 | 0.3775 | Multimodal and visual-dynamics gates failed |

Scores from the two rows use different test people and must not be read as
a controlled before/after improvement. Within each row, all arms use the
same examples and preprocessing. The residual follow-up preserves the
discovery training/validation arrays and PCA/normalizer byte for byte.

The frozen gates required at least 5% prediction improvement over the
stronger simple control, at least 2% benefit from the correct other stream
over a within-person wrong stream with a positive person-cluster 95% interval,
and a repeatable advantage over vanilla. The follow-up additionally required
at least 5% visual improvement over persistence in both seeds.

Audio prediction and the full-versus-vanilla threshold pass, but reliable
paired-modality contribution does not. Independent retrieval is weak and
unstable near a 0.48% chance R@1 across 210 candidates; within-person scores
are close to chance. In the follow-up, no-spectral and no-long-convolution
variants beat the full model in seed 41, so those components are not reliably
necessary in this setup. Shuffled-pair training achieves audio/video MSE
0.4757/0.3884 versus correctly paired full seed 41's 0.5136/0.4152.

These controls reject scaling this particular recipe. They do not rule out
multimodal latent predictive modeling in general.

## Stronger exploratory controls

After the discovery result and some follow-up outputs had been seen, we
added training-time-profile means and fixed-ridge residual regressors. These
are explicitly **post-hoc exploratory controls**, with regularization fixed
at 100 and no test tuning; they do not alter the frozen decisions.

On fresh test people, the audio time-profile mean scores 0.5530 MSE. A linear
regressor using audio context scores **0.4711**, below both full SLWM seeds,
with about 2.10M fitted coefficients/biases. Both target streams retain their
own persistence anchors, so its visual result is not a modality-deletion test.
The linear joint-input model is worse (audio 0.6610). These observations make
the neural-baseline audio gain insufficient evidence of architectural value.
The fitted linear predictors are retained as reusable artifacts.

## Reproducibility, corrections and limitations

- Each run retains `best.pt`, optimizer/RNG state, config, source/config/data
  hashes, validation-selected step, curves, metrics, per-clip errors and registry.
  Both cycles archive their exact frozen-run training source in `source_snapshot/`.
- The old shared T2 runner used retrieval representations after paired modalities
  had already exchanged information. It now uses blind unimodal encodes. Old
  generated-fixture metrics remain historical mechanics evidence only.
- Original cards counted the complete 11,689,512-parameter ImageNet checkpoint,
  including its discarded 513,000-parameter classifier. Immutable card hashes
  remain; `parameter_accounting.json` records the deployed encoder correction.
- Discovery single-modality arms had the same sample budget but different sample
  order because dropout consumed different RNG draws. The follow-up fixes this;
  the primary full/vanilla/ablated/shuffled comparisons already shared schedules.
- Final source adds accounting and artifact-integrity guards after frozen runs.
  Prepared corpora are reused only after hash checks and never overwritten
  silently. Source snapshots reproduce the exact code used for the results.
- Only seven people appear in each test set. Confidence intervals resample people,
  not independent clips. Repeated sentences, normalized timing, identity and
  background shortcuts limit interpretation. Frozen image features may discard
  lip dynamics; predicting their future is not raw video generation.

The final regression suite passes **84 tests**. Saved checkpoints from both
cycles were loaded successfully and run on real held-out clips. All checkpoints,
compact corpora, reports, charts and example predictions are local; the complete
raw corpus also remains on Fedora. No training process is left running.

## Artifacts and next research step

- [Discovery report](../../experiments/multimodal/t2/real-av-20261104/report.md)
  (`EXP-T2-201`–`209`).
- [Fresh-person residual report](../../experiments/multimodal/t2/real-av-residual-20261104/report.md)
  (`EXP-T2-211`–`219`).
- [Fresh-person forecast comparison](../../experiments/multimodal/t2/real-av-residual-20261104/forecast_comparison.png)
  and [saved-checkpoint feature examples](../../experiments/multimodal/t2/real-av-residual-20261104/forecast_examples.png).
- [Research protocol](real_av_pilot_20261104.md) and
  [separate residual follow-up](real_av_residual_followup_20261104.md).
- [Reproduction and inference](../../training/README.md#real-recorded-audiovisual-pilots).

Keep these as a small reproducible benchmark and forecasting artifacts. The next
research objective should be **demonstrable event-level audio/video alignment**
at this scale, with within-person/time-shifted negatives and a video feature
representation sensitive to motion. Register that separately and retain a fresh
confirmation set. Do not increase parameter count, add text/code generation, or
claim general world-model capability on the present evidence.
