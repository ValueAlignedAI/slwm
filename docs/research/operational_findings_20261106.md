# Operational and scaling findings, 2026-11-06

All22 primary runs completed: ten world fields, four separate language models
and eight sensing actors. Technical integrity passes; the three B1/P1 capability
gates fail. Keep these outcomes separate. Full evidence is in the
[world report](../../experiments/multimodal/t2/broad-stream-20261105/report.md),
[sensing report](../../experiments/policy/continuous-sensing-20261105/report.md),
[language report](../../experiments/text/t1/extended-throughput-20261105/report.md)
and [all-case media gallery](../../experiments/multimodal/t2/broad-stream-20261105/watch.html).

## GPU headroom and measured scale

The final23:30–00:46 training interval averages53.81% sampled NVIDIA GPU
activity, peaking at6,121MiB of24,576MiB. Activity is not achieved FLOP
utilization. Width256→512 increases neural storage from6.68M to23.98M for SLWM
and6.60M to23.81M for Transformer. At the same2000-update maximum, measured
training/validation time rises only1.275x/1.226x. This is a single-seed hardware
scale probe, not a quality scaling law. Validation-selected exposures vary;
the selected-state audit replays exact RNG and coverage for every saved step.

Fused text output loss and batch64 improved benchmark throughput about18–24%;
batch128 OOMs remain retained. Compilation gives1.458x/1.524x steady world-model
steps; matched400-update learning controls give1.174x/1.290x measured speedups
with a warmed cache and≤1% selection-loss degradation. Original cold startup
79.29s/37.63s is additional, and the original SLWM local numerical gate remains
failed. MPS concurrency also failed its trajectory check. Neither replaces the
frozen primary B1 execution. Final selected-weight filter folding gives1.150x
CUDA FP32 inference, with all-head error below0.000009. Its algebraic equivalence
means these B1 parallel linear branches add parameterization, not a different
31-tap filter function class. T1's sequential paths have a different structure.

## Useful speech and shared-state decisions remain unproved

Primary160ms gaps, mean clip WER (lower is better):

| Condition | Fresh test-clean | Fresh test-other |
|---|---:|---:|
| Silence |6.62%|13.67%|
| Linear waveform interpolation |6.53%|13.50%|
| SLWM seed91 |7.24%|15.66%|
| SLWM seed92 |7.14%|15.31%|
| Transformer seed91 |6.99%|14.90%|
| Transformer seed92 |6.99%|14.73%|

Every learned arm fails the registered practical repair gate. Preserving known
fine mel context at the renderer gives mixed secondary changes and does not
rescue the gate. More parameters did not improve this practical outcome.

Sensing counts every tick, including uncommitted forecasts. SLWM commitment
coverage is60.6%/65.2%, but both seed intervals versus the strongest fixed rule
cross zero. One seed narrowly beats the same-sized direct-statistics actor;
the other is worse. There is no replicated field-versus-raw benefit. The12 test
people share only three meetings. A continuously executing policy is therefore
demonstrated, while its added decision utility is not.

The audio phonetic readout is about25% accurate versus21–23% for selected
Transformers on43 eligible windows from seven people. Those limited diagnostics
are neither intelligible speech nor general semantic grounding. Both processors
use the same recurrent shared-field design; a generic field result is not an
extra-signal-path attribution.

## What the long outputs reveal

All248 fixed outputs are40.96–60.416s, with actual decoded frame/sample checks.
Generated video is64×64 face motion over a supplied training appearance.
Audio→video retains source sound. Video→audio uses no source audio input.
Text→audio/video receives CMU phones and train-estimated timing, separate from
the125M language weights. Unique frames alone are misleading: all four main
audio→video arms have worse motion error than a constant-frame control, despite
lower code error and nonconstant pixels. Continuous conditioning and30.037s
state-only continuations are separately retained.

Video→audio corpus WER is99.57% on nine nonempty-reference cases; provided-text
generation is100% on both passages/all four main arms. Source/oracle-renderer
WER is41.38%/44.40% on the same nonempty meeting references. This supports a
content-prediction bottleneck rather than a successful long speech generator.
These deterministic readouts do not establish knowledge of general scenes.

Empty owner annotations do not mean silent physical audio. A post-test audit
checks distinct ES2012a headset hashes, cross-channel correlation and an
independent single-example CPU decode: the quiet owner's headset contains the
other participant's speech. The report separates nonempty-reference WER from
empty-owner insertion counts, while preserving all original scores/cases/gates
byte-for-byte. Such insertions cannot automatically be called hallucinations.

## Next research decision

The125M text path merits a longer controlled data-budget study: both seeds have
3.64–3.67% lower test NLL at equal151M tokens, at1.47–1.49x continuation time.
Saved-checkpoint timer-matched validation gains shrink to0.10–1.04%; samples
remain repetitive. All four models reaching500M tokens costs about10.77 extra
GPU hours, excluding saving/final tests. A new schedule and uninspected test
windows must be frozen. See the [resource plan](scaling_budget_review_20261106.md).

For speech, prioritize finer observation masks and an exact practical fallback,
real phone alignment, and content-preserving speech units/prosody conditioning
against direct deterministic mel prediction. Broad Libri text-conditioned
training is already present; simply adding that mode is not a new intervention.
Compare revised edges through both processors on identical data before scaling
the joint core. Expand independent active paired sessions and account for whose
speech a microphone actually captures. Current training has100.58h Libri and
8.23h person tracks, but only2.05h of independent common meeting time. The
[access review](data_scaling_access_review_20261106.md) checks feasible sources.

For physical decisions, the DROID source probe validates two actual episodes,
three camera views and typed commanded-pose/gripper actions. It trains no robot
policy. The [next-stage contract](action_conditioned_next_stage_20261105.md)
requires action-conditioned transitions, scene/proprioception/goal edges,
episode/environment holdouts, proper rotation/cadence handling and multi-step
controls. Keep a fast control layer separate from the present1.365s semantic
chunk cycle. Scaling the unchanged face/phonetic decoder is not justified by
the current capability results.

## Delivery validation

The166-test suite passes. Actual-state audits additionally verify10 worlds,
four language exports and eight actors; hidden-input invariance, forecast
causality, chunk/state equivalence, source identity and exact actor binding.
All3,936 repaired WAVs retain observed PCM exactly; all248 long clocks agree.
Ordinary-file CPU inference passes every mode. The primary paced replay makes
30 automatic decisions in41.42s, with zero deadline misses and
unchanged weights. Portable recorded inputs and a second verified replay are
in `broad-stream-20261105/live_demo/`. Features are prepared before replay;
actual live capture and physical control are subsequent work.

Full states, optimizer/RNG, failed pilots, media, codecs and renderer are retained
locally and on Fedora. Curated evidence is versioned in focused commits; binary
artifacts are outside Git and covered by the final delivery manifest. Raw and
prepared training corpora remain on Fedora, with mirrored provenance metadata.
