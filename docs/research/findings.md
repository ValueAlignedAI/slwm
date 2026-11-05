# SLWM research findings — updated 2026-11-05

## Current decision — content and timing improve; speech quality remains mixed

Ten new RTX 3090 runs (`EXP-T2-701`–`704`, `711`–`716`) test aligned lexical
supervision and observed-only video admission. Four separate text runs
(`EXP-T1-201`–`204`) test causal processors at GPT-2-small scale. Keep the useful
speech/text checkpoints and continue controlled experiments. Do not commit a
long multimodal scale run based on these short, separate studies.

### Speech content and rejection: fresh people and complete sentences

The original GRID recordings provide synchronized audio, actual mouth pixels
and official timed text. Training uses **12,704 clips / 10.59 hours / 16 people**.
Selection uses four different people (395 clips); threshold calibration uses
three more (278); the primary test reserves **749 clips from eight people**.
Global sentence-hash buckets also separate complete sentence combinations.
The vocabulary remains only 51 lexical words plus two pause codes. CREMA/Libri
windows retain broader training acoustics; their final checks are reused
retention diagnostics, not new confirmation. All normalizers remain fitted to
historical CREMA training data. GRID 33/34 remain unopened.

The source-quality alarm stopped preparation before training. A CRC/actual-frame
review verified 90 truncated video sources, all from training person 08
(88 training, two training-person diagnostics). The exact exclusion list and
failed logs are preserved; the 50-packet input rule stayed fixed, and no primary
holdout was removed. See [the source review](grid_source_quality_review_20261105.md).

The 5.13M speech base adds a timed-word decoder interface and a lexical loss on
processed fields. Both matched bases have the timed-text auxiliary path; the
ablation removes only the added unit cross-entropy. Audio/video candidates freeze
the base and add about 1.29M trained weights each. The **7,713,820-weight video
artifact** includes the strong frozen audio candidate. Every rejection returns
that candidate's spectrum, word logits and latent state exactly. Admission uses
only observed audio/video, with thresholds fixed on independent calibration
people and a mandatory aligned-video coverage floor. Training labels never enter
practical inference. Fixed ridge probes are fitted on training fields only.

| Primary effect (paired clip means; person-cluster 95% intervals) | Seed 71 | Seed 72 |
|---|---:|---:|
| Field lexical-word accuracy gain | **+9.72pp [8.21, 10.97]** | **+12.43pp [10.95, 13.92]** |
| Content-objective gap-spectrum improvement | 4.02% [2.81, 5.82] | 3.12% [1.47, 5.30] |
| Calibrated video improvement over strong audio | **12.26% [9.65, 14.86]** | **11.24% [8.24, 14.21]** |
| Aligned video accepted | 81.84% [77.33, 85.92] | 83.85% [79.53, 87.83] |
| +240ms / −240ms video accepted | 0.53% / 0.53% | 0.93% / 0.67% |

Content, calibrated rejection and spectral-video gates pass in both seeds.
Unfiltered ±240ms video increases spectral error about 33–35%; admission reduces
that harm to at most 0.15%. Untrained ±160ms shifts are also mostly rejected.
Wrong-utterance video is harder: about 12–13% is admitted and causes ~3.5% mean
harm. This is evidence for timing rejection, not calibrated semantic truth.
Shuffled-pair training loses the aligned video's feature benefit. Letters gain
only ~3.3pp in the content probe; digits gain ~15–16pp, exposing uneven learning.

**The practical speech gate fails.** On the 32 fixed recognizer cases, strong
audio→calibrated-video WER is **54.69%→51.04%** for seed 71 (paired interval
includes zero), and **48.96%→53.65%** for seed 72 (worse). Erasing the gap scores
52.60%. Reference audio scores 19.27%, and the deliberately target-provided
spectrum diagnostic 19.79%; the missing-content/acoustic prediction remains the
main issue. These are external recognizer diagnostics, not human ratings or
SLWM-generated text. A lower spectral loss has not yet become dependable speech.

[Corrected report](../../experiments/multimodal/t2/content-sync-20261105/report_v2.md),
[all fixed listening cases](../../experiments/multimodal/t2/content-sync-20261105/listen.html),
[training and ordinary-file inference](../training/content_and_scale.md).
Revision 2 excludes 19 short-pause frames incorrectly counted as lexical words
in the original primary score; it reuses the same fitted probes. Original metrics,
reports and evaluation sources are retained. No checkpoint, threshold, spectrum
or waveform changed. The initial evaluation/source snapshots remain immutable.

### Separate GPT-2-scale text comparison

The pinned, domain-disjoint FineWeb-Edu corpus contains **2.600B training tokens**,
14.57M validation and 10.01M test tokens, with normalized exact document,
long-prefix and canonical-URL deduplication. Residual near duplicates remain a
limitation. GPT-2-style models have 124,439,808 weights; causal SLWM has 125,131,008.
Shared modules start identically within seeds 81/82, with matched unique token
windows and no pretrained language model. Each run sees **16,777,216 tokens**.

| Seed | GPT-2 / SLWM validation NLL | GPT-2 / SLWM test perplexity | GPT-2 / SLWM minutes |
|---|---:|---:|---:|
| 81 | 5.6486 / 5.4517 | 262.84 / 216.33 | 7.95 / 11.35 |
| 82 | 5.6449 / 5.4688 | 260.83 / 218.47 | 7.98 / 11.31 |

The early guardrail passes, with 3.49% / 3.12% lower validation loss at equal
tokens, at ~40% higher time per token. A separately labeled post-hoc wall-time
interpolation gives only 0.82% / 0.55% lower loss; it is not measured equal-FLOP
performance. The corpus provides ~20.9 available tokens/parameter, but these runs
consume only ~0.13 tokens/parameter. They are severely undertrained, not evidence
of converged superiority. Only 512-token contexts were trained. Each validation/test score uses the same
fixed 131,072-token held-stream sample, not full-corpus perplexity. All fixed raw
continuations are retained, including poor/repetitive outputs. English text
results do not establish code skill, grounding or a learned output policy.

[Text report](../../experiments/text/t1/scale-20261105/report.md),
[compute diagnostic](../../experiments/text/t1/scale-20261105/compute_diagnostic.md),
[frozen scale protocol](text_scale_protocol_20261105.md).

### Retained evidence and next experiment

All 14 best optimizer/RNG checkpoints and final weights, source/data hashes,
validation choices, per-example scores and portable bundles are saved on Fedora
and locally. New measured training time totals **64.27 GPU-process minutes**
including validation, with data preparation, historical pretraining and final
evaluation additional. The full suite passes **133 tests**. The speech audit
verifies 14,636 source records, frozen components, actual hidden-waveform/label
isolation and **640 observed-PCM-exact WAVs**. A separate text audit verifies
prepared token/source hashes, disjoint domains and all selected checkpoints.
Portable bundles also run through ordinary-file CPU inference.

The next speech experiment should target lexical/acoustic decoding on broader
continuous speech with new reserved people and phrases. Keep exact audio fallback
and the coverage-aware timing filter. Register new data/selection choices before
examining another confirmation set. For text, use component ablations and longer
matched token/compute budgets before a full-corpus run. The small GRID grammar is
inappropriate as the sole dataset for a GPT-2-scale multimodal model.

## Previous T2-V1 decision — a narrow video benefit; broader repair gate fails

**Keep the audio-anchored visual correction artifact, but do not scale this
recipe yet.** Eleven additional RTX 3090 runs test whether synchronized video beats
an approximately capacity-matched extra audio branch. A two-stage architecture
freezes the previous content-preserving speech expert, adapts an audio anchor to
true waveform gaps, then learns only bounded missing-region corrections from
original 96×64 mouth pixels. The full video model has **7,990,547 neural weights**,
including **1,423,025 newly trained visual-branch weights**. Fixed codec/PCA
values are counted separately. This is additional supervised adaptation, not
training a world model from scratch or demonstrating a learned output policy.

The primary task erases a true 320ms waveform interval. Every 64ms input analysis
window touching it is removed, including the first packet after the nominal
hole; deltas are recomputed from observed-only interpolation. This closes the
boundary leakage in older feature-packet experiments. New actual WAV repairs
erase hidden samples before features or phase and preserve observed PCM samples
exactly. This passthrough task differs from earlier full-synthesis experiments.

### Frozen comparisons and result

Use 5,061 CREMA training clips plus 240 RAVDESS clips from four training people.
CREMA validation stays 210; the earlier 128 R1 RAV test clips explicitly become
validation. Reserve 128 new clips from eight other RAV people for final scoring.
All normalizers remain CREMA-training-only. Checkpoints and the stronger audio
baseline are selected by equal-corpus validation only. Seed61 selects the
**doubled-update audio control619**; seed62 uses612. Primary video614/615 and
all ablations are fixed before final tests. All models use the same positive
windows within a seed; visual negative examples add disclosed training compute.

Relative gains below are means of paired per-clip missing-spectrum MSE reductions,
with 2,000-draw person-cluster 95% intervals. The primary comparison requires at
least 5% and a positive interval in both seeds on both sets.

| Test | Seed61 video gain vs selected audio | Seed62 video gain vs selected audio |
|---|---:|---:|
| Reused CREMA,202 eligible clips/eight people | **9.30% [6.71,11.79]** | **9.11% [6.50,11.55]** |
| Fresh RAV people,128 clips/eight people | 5.00% [−1.68,11.10] | 4.87% [−2.65,11.44] |

- Correct video beats different-utterance, genuinely shifted±240ms and static
  video on CREMA. This supports a narrow temporal audiovisual contribution,
  beyond simply adding audio parameters. CREMA is reused and has limited phrases.
- Fresh RAV generalization is uncertain. A post-hoc all-person breakdown shows
  negative effects for actors 10 and 21 in both seeds, with larger positive
  effects for 09,22,23. No crop, model or threshold is adjusted on these scores.
- Sensitivity is not safe rejection: wrong or shifted video makes spectral
  repair **11–15% worse than the strong audio control on CREMA** and about
  **15–26% worse on fresh RAV**. The learned confidence gate still admits harmful
  corrections. Missing video returns the frozen audio anchor exactly, but that
  is a weaker fallback than the separately trained additional audio branch.
- The fixed 32-case external recognizer diagnostic gives CREMA mean WER
  **56.92% →41.96%** and **49.55% →41.07%** for audio→video. Only the first
  paired-seed interval excludes zero. Fresh RAV is unchanged at 15.63%/14.58%;
  silence inside the gap does better at 11.46%. Both broad feature and speech
  gates therefore **FAIL**. These are fallible recognizer results, not human
  listening evidence or model-generated text.
- The evaluation-only true-spectrum reconstruction reaches 18.53% CREMA WER
  (reference 20.09%) and 0% RAV (reference 0%). This suggests the phase decoder can
  support much better recognition when given the missing spectrum; decoder
  quality alone does not explain the learned repairs' remaining errors. The
  oracle never enters any model, selection or practical inference path.

### Artifacts and next research decision

All **11 best checkpoints with optimizer state**,11 final checkpoints, configs,
source/data hashes,160/480ms secondary gaps, all-clip checks,32 fixed listening
cases and **448 actual WAVs** are retained locally and on Fedora. Training takes
**16.18 measured GPU-process minutes**, including validation and excluding
historical expert training/data preparation. The audit checks 6,007 raw records,
frozen expert/anchor equality, hidden-waveform and target-dictionary invariance,
container-soundtrack isolation, real-file CPU inference and exact observed PCM.
The complete suite passes **124 tests**. The preexisting latent-field edit is
preserved.

The first render attempt exposed a terminal subframe padding bug. Its log,
partial outputs and original evaluation snapshot are retained; a separately
hashed revision2 applies the already established≤3-frame edge-padding rule to
the globally erased recording. No weight, feature metric, selected ID or gate
changes. The full suite is rerun after this correction.

The useful result is a small, genuine video correction model with a replicated
in-corpus spectral benefit and a clear generalization failure. The next research
priority is speech-content prediction and validation-calibrated rejection of
mistimed video before increasing parameter count. The reference-spectrum
condition argues for investigating the missing-content objective as well as
waveform decoding. A train-only audiovisual unit-prediction objective is a
candidate, not a validated result. Any new success claim needs a new
preregistered holdout with broader speech, rather than tuning these test actors.

- [Full frozen report](../../experiments/multimodal/t2/video-repair-20261105/report.md)
- [Every fixed listening example](../../experiments/multimodal/t2/video-repair-20261105/listen.html)
- [Audio/video checkpoint bundle](../../experiments/multimodal/t2/video-repair-20261105/bundle/README.md)
- [Post-hoc failure analysis](../../experiments/multimodal/t2/video-repair-20261105/failure_analysis.md)
- [Audit](../../experiments/multimodal/t2/video-repair-20261105/audit.json),
  [source and artifact manifest](../../experiments/multimodal/t2/video-repair-20261105/delivery_manifest.json)
- [Post-hoc crop inspection](video_repair_crop_inspection_20261105.md) — the central rectangle is not a complete lip ablation.
- [Protocol](video_repair_protocol_20261105.md),
  [render correction](video_repair_output_correction_20261105.md),
  [training and ordinary-file inference](../training/video_repair.md)

## Previous R1/R2 decision — retain an explicit content objective

**Keep the observed-signal preservation path AND train an auxiliary content
readout from the shared field.** Thirteen new RTX 3090 runs isolate why the
earlier bypass helped output fidelity while making sound harder to decode from
the core. The resulting 5,100,289-weight research model preserves linearly
decodable sound detail and timing; useful positive contribution from video to
speech repair remains unestablished. Do not scale the same recipe yet.

### What changed and passed

T2-R1 trains nine small models on the existing 5,061 CREMA training clips plus
1,278 real LibriSpeech clips from 32 training people. The 64 Libri validation clips
use eight other people. All codec/PCA/normalization remains CREMA-training-only.
Every update draws 24 CREMA and 8 Libri 1.28s windows; video is dropped for half
the CREMA cases. New held-out checks are 48 complete test-clean utterances from
12 people and 128 RAVDESS clips from eight people excluded from previous cycles.

The raw skip protects output but also removes pressure to retain observable
content inside the field. In a training-only fitted, frozen-field linear readout,
the R1 full core has test-clean MSE **0.2988 / 0.3183**, versus **0.1640** for
the original frozen core. Removing the skip improves that readout to **0.0976**
but makes actual decoded sound worse. This measures linear decodability; neither
an unsuccessful decoder nor a probe proves irreversible information loss.

T2-R2 adds a **16,448-weight linear auxiliary head**, trained to reconstruct
only observed coarse sound from the processed field, with no raw bypass into
that head. The primary decoder, underlying core size, data, sampled windows,
masking and new 1,600-update budget stay unchanged. Both primary seeds use
auxiliary weight .5; zero and weight-2 sensitivity controls are retained. All
restart from the same original M1 weights, not R1-trained checkpoints.

Reserve another fresh check before scoring R2: **48 complete LibriSpeech
test-other utterances from 12 different people** and harder recording conditions.
The earlier external sets become reused diagnostics for R2. No held-out data
fits normalizers, weights, ridge readouts or checkpoint selection.

- Fresh test-other frozen-field readout MSE drops **0.3335 → 0.06453** and
  **0.3715 → 0.06234** in matched seeds: **80.7% / 83.2% lower aggregate error**.
  Mean per-clip relative gains are 79.7% [75.1, 83.4] and 81.7% [76.6, 85.4],
  using person-cluster 95% intervals. This readout sees only audio-only core
  fields, never the raw skip. It establishes sound decodability, not semantics.
- On reused test-clean, readout error falls to **0.05151 / 0.04844**. Both
  primary seeds beat the old frozen core on fresh test-other as well
  (**0.16769 → 0.06453 / 0.06234**).
- Clean-output error on fresh test-other is **0.04704 / 0.04698**, within the
  preregistered 5% guardrail. Hidden-packet error on reused CREMA/test-clean
  also stays within its 5% guardrail. The content objective is not free: the
  representative CREMA timing score drops from R1's 98.5% to 94.1% in seed 51.
- Seven-shift timing remains **94.1% / 97.5%** on CREMA and **70.3% / 72.7%** on
  reused RAVDESS, against 14.3% chance. Both primary seeds pass all four R2
  gates: field content, clean output, repair guardrail, timing retention.
- The zero-auxiliary control exactly reproduces R1's primary test errors.
  The stronger auxiliary objective is explicitly a predeclared sensitivity run,
  with a single seed; it is not selected as a winner after test inspection.

### What remains unsolved

R1's full models improve CREMA hidden-packet MSE about 21% versus interpolation,
but do not pass the required advantage over both raw-AV and vanilla controls.
Correct video versus sound alone has per-clip gains **1.82% [-0.23, 4.00]** and
**−0.65% [-2.20, 1.10]**. On fresh R1 RAVDESS people, adding video worsens hole
prediction **4.23% / 6.92%** by that metric. Wrong video is worse still, but
wrong-input sensitivity does not establish that the correct modality is useful.
The new R2 content gate does not convert this failed multimodal gate into a pass.

Independent recognizer WER on fresh test-other is **79.7%** for interpolation,
**72.6%** for R1 full and **70.3% / 70.8%** for the primary R2 models. Original
audio scores 26.5%, target codec 31.2%; the evaluator itself is limited. This is
still poor speech repair, not reliable generated speech. A matched coarse-silence
control scores 87.0%. The original R1 silence diagnostic has richer 100 Hz
observations outside holes; it is retained separately and excluded from fair
model comparisons. Recognizer text never trains or supplies model output.

These are withheld **mel packets**, with centered 64ms codec windows that can
straddle boundaries, not sample-exact waveform gaps. All model audio is actual
inverse-mel/new-phase output, never copied source waveform samples. Full-clip
rendering applies one global mask before overlap windows, preventing a hidden
packet from returning in another window. There are no policy, free-text,
autonomous future-rollout or general understanding claims.

### Evidence and next step

New update training totals **11.74 RTX 3090 minutes**, excluding data preparation,
readout fitting, scoring and media rendering. Head/data/new update budgets match;
total model size, historical pretraining and compute differ. Every selected and
last checkpoint, best-state optimizer, config/source/data hash, curve, failure,
actual waveform and independent readout is retained locally and on Fedora.
All **116 repository tests pass**. Ordinary-file CPU inference passes for the
new core, raw control and audiovisual path. Pixel-identical video with a changed
soundtrack produces exactly identical predictions; the zero-auxiliary control's
primary weight tensors exactly match R1, not merely its aggregate scores.

- [R1 report](../../experiments/multimodal/t2/signal-restoration-20261105/report.md)
  and [registered protocol](signal_restoration_protocol_20261105.md).
- [R2 report and figure](../../experiments/multimodal/t2/content-conservation-20261105/report.md)
  and [registered protocol](content_conservation_protocol_20261105.md).
- [Standalone inference and reproduction](../training/signal_restoration.md).

Retain the auxiliary content objective in the next core experiment. Focus next
on when visual evidence actually helps, stronger waveform-level/perceptual
checks and output usefulness under uncertainty. Keep sound-only, wrong-video,
interpolation and observation-matched silence controls. More parameters are not
justified by these results. Text/code and commitment remain separate stages.

## Previous G1/G2 decision and results (preserved)

### G1/G2 generation follow-up

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
