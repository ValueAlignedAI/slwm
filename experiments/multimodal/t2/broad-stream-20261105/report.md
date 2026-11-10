# T2-B1 — broad speech and continuous multimodal fields

Randomly initialized shared audio/video/phonetic-text models versus paired Transformers, with matched data, window plans, optimizers and2000-update limits. Selection uses independent validation only; final speakers are disjoint from all prior SLWM exposures. Gates and every failed case remain visible.

Fitted codec/duration values: 266,508; external frozen Vocos:13,531,650 neural weights, separately counted. CMU pronunciation is a fixed external dictionary, not a learned semantic text encoder. The124/125M English language models have separate weights.

| Run | Variant / seed | Neural weights | Selected / total updates | Validation loss | Measured run minutes |
|---|---|---:|---:|---:|---:|
| EXP-T2-801 | slwm / 91 | 23,980,313 | 1800 / 2000 | 1.3215 | 17.69 |
| EXP-T2-802 | transformer / 91 | 23,805,209 | 1800 / 2000 | 1.3667 | 11.58 |
| EXP-T2-803 | slwm / 92 | 23,980,313 | 2000 / 2000 | 1.3220 | 17.62 |
| EXP-T2-804 | transformer / 92 | 23,805,209 | 1400 / 2000 | 1.3583 | 11.63 |
| EXP-T2-805 | late_fusion / 91 | 23,805,209 | 1800 / 2000 | 1.3571 | 14.06 |
| EXP-T2-806 | slwm / 91 | 23,980,313 | 2000 / 2000 | 1.3226 | 17.42 |
| EXP-T2-807 | slwm / 91 | 23,980,313 | 2000 / 2000 | 1.4444 | 17.38 |
| EXP-T2-808 | slwm / 91 | 23,980,313 | 2000 / 2000 | 1.3317 | 17.85 |
| EXP-T2-809 | slwm / 91 | 6,683,929 | 1800 / 2000 | 1.3723 | 13.87 |
| EXP-T2-810 | transformer / 91 | 6,596,377 | 1600 / 2000 | 1.3872 | 9.44 |

All runs share the maximum training budget, while validation-selected checkpoints can have different exposure. exposure_identity_audit.json replays exact sampled coverage for both final training states and selected weights, checking their saved data RNG; the selected/total column above must be retained in comparisons.

## Fresh-speaker true-gap speech

Primary160ms gap, all48 clips in each person-disjoint12-speaker set; actual waveform erasure before FFT. Mean clip WER is primary, corpus WER is also saved. Each baseline is scored identically by pinned external Whisper-tiny.en. Observed PCM is exact outside every gap. CIs resample people10,000 times.

| Set | Condition | Mean clip WER | Change versus stronger registered baseline |95% interval | Gate |
|---|---|---:|---:|---|---|
| test-clean | reference | 5.40% | — | — | control |
| test-clean | silence | 6.62% | — | — | control |
| test-clean | linear_wave | 6.53% | — | — | control |
| test-clean | interpolated_mel | 6.69% | — | — | control |
| test-clean | oracle_mel | 5.40% | — | — | control |
| test-clean | EXP-T2-801 | 7.24% | -9.34% relative | [-1.07,-0.24]pp | FAIL |
| test-clean | EXP-T2-801_conserved_context | 7.07% | — | — | secondary readout |
| test-clean | EXP-T2-802 | 6.99% | -5.57% relative | [-0.71,-0.05]pp | FAIL |
| test-clean | EXP-T2-802_conserved_context | 6.99% | — | — | secondary readout |
| test-clean | EXP-T2-803 | 7.14% | -7.88% relative | [-1.04,-0.10]pp | FAIL |
| test-clean | EXP-T2-803_conserved_context | 7.05% | — | — | secondary readout |
| test-clean | EXP-T2-804 | 6.99% | -5.57% relative | [-0.71,-0.05]pp | FAIL |
| test-clean | EXP-T2-804_conserved_context | 7.05% | — | — | secondary readout |
| test-clean | EXP-T2-805 | 6.99% | -5.57% relative | [-0.71,-0.05]pp | FAIL |
| test-clean | EXP-T2-806 | 7.07% | -6.83% relative | [-0.93,-0.06]pp | FAIL |
| test-clean | EXP-T2-807 | 6.99% | -5.57% relative | [-0.71,-0.05]pp | FAIL |
| test-clean | EXP-T2-808 | 7.32% | -10.51% relative | [-1.24,-0.28]pp | FAIL |
| test-clean | EXP-T2-809 | 6.94% | -4.84% relative | [-0.64,-0.02]pp | FAIL |
| test-clean | EXP-T2-810 | 6.99% | -5.57% relative | [-0.71,-0.05]pp | FAIL |
| test-other | reference | 12.17% | — | — | control |
| test-other | silence | 13.67% | — | — | control |
| test-other | linear_wave | 13.50% | — | — | control |
| test-other | interpolated_mel | 14.93% | — | — | control |
| test-other | oracle_mel | 12.04% | — | — | control |
| test-other | EXP-T2-801 | 15.66% | -14.55% relative | [-2.77,-1.24]pp | FAIL |
| test-other | EXP-T2-801_conserved_context | 15.46% | — | — | secondary readout |
| test-other | EXP-T2-802 | 14.90% | -8.98% relative | [-2.48,+0.51]pp | FAIL |
| test-other | EXP-T2-802_conserved_context | 15.13% | — | — | secondary readout |
| test-other | EXP-T2-803 | 15.31% | -12.02% relative | [-2.71,-0.45]pp | FAIL |
| test-other | EXP-T2-803_conserved_context | 15.50% | — | — | secondary readout |
| test-other | EXP-T2-804 | 14.73% | -7.73% relative | [-2.20,+0.59]pp | FAIL |
| test-other | EXP-T2-804_conserved_context | 15.48% | — | — | secondary readout |
| test-other | EXP-T2-805 | 14.75% | -7.90% relative | [-2.33,+0.64]pp | FAIL |
| test-other | EXP-T2-806 | 15.31% | -12.00% relative | [-2.50,-0.79]pp | FAIL |
| test-other | EXP-T2-807 | 14.79% | -8.21% relative | [-2.23,+0.20]pp | FAIL |
| test-other | EXP-T2-808 | 14.56% | -6.53% relative | [-2.13,+0.79]pp | FAIL |
| test-other | EXP-T2-809 | 15.18% | -11.02% relative | [-2.62,-0.11]pp | FAIL |
| test-other | EXP-T2-810 | 14.90% | -8.98% relative | [-2.45,+0.49]pp | FAIL |

The four `_conserved_context` conditions are pre-test secondary readouts: retain physically observed fine mel frames at Vocos input while keeping every hidden model proposal identical. They add no fitted weights and do not replace the original primary gate. context_conservation_summary.json reports paired effects against original readout and the stronger practical baseline. Fine masks and both mel arrays are retained for exact invariance checks.

## Same-model video contribution on new AMI people

These secondary40.96s cases compare identical weights/input gaps with audio only, correct video and wrong/static/±240ms video. This is not a fitted timing-rejection gate. The table displays corpus WER on nonempty owner references; raw all-case mean clip scores remain unchanged in av_repair_summary.json. Empty owner annotations can contain another person's audible speech, so their insertion counts are separate and are not automatically hallucinations. This presentation correction changes no case, model, score, threshold or gate.

| Run | Audio only | Aligned video | Static video | Wrong time | −240ms | +240ms |
|---|---:|---:|---:|---:|---:|---:|
| EXP-T2-801 | 44.83% | 44.83% | 43.97% | 44.83% | 44.83% | 44.83% |
| EXP-T2-802 | 44.83% | 44.40% | 43.10% | 44.40% | 44.40% | 44.40% |
| EXP-T2-803 | 44.40% | 44.83% | 44.83% | 44.83% | 44.83% | 44.83% |
| EXP-T2-804 | 44.83% | 44.40% | 43.97% | 44.83% | 44.40% | 44.40% |

Insertion counts in empty-owner-reference cases, retained for every condition:

| Run | Audio only | Aligned video | Static video | Wrong time | −240ms | +240ms |
|---|---:|---:|---:|---:|---:|---:|
| EXP-T2-801 | 182 | 187 | 187 | 187 | 187 | 187 |
| EXP-T2-802 | 180 | 187 | 182 | 182 | 187 | 187 |
| EXP-T2-803 | 182 | 182 | 187 | 182 | 182 | 182 |
| EXP-T2-804 | 182 | 182 | 182 | 182 | 182 | 182 |

## Shared representation and future prediction

Blind audio/video/text embeddings are encoded independently, with within-person candidates and no appearance input. Phonetic accuracy excludes silence/unknown targets. Forecasts receive both observed sensors and predict beyond the2-packet guard; persistence uses the same last observations.

| Run | Audio/video retrieval | Audio/text retrieval | Video/text retrieval | Future audio MAE / persistence | Future video MSE / persistence |
|---|---:|---:|---:|---:|---:|
| EXP-T2-801 | 44.3% | 67.9% | 44.3% | 0.6264 / 0.6938 | 1.0378 / 0.4980 |
| EXP-T2-802 | 45.2% | 70.2% | 42.6% | 0.6202 / 0.6938 | 1.0701 / 0.4980 |
| EXP-T2-803 | 43.8% | 76.2% | 43.5% | 0.6179 / 0.6938 | 1.0339 / 0.4980 |
| EXP-T2-804 | 45.5% | 71.4% | 43.2% | 0.6344 / 0.6938 | 1.1127 / 0.4980 |
| EXP-T2-805 | 42.6% | 66.7% | 42.6% | 0.6097 / 0.6938 | 1.0575 / 0.4980 |
| EXP-T2-806 | 43.5% | 77.1% | 43.5% | 0.6068 / 0.6938 | 0.9728 / 0.4980 |
| EXP-T2-807 | 44.3% | 64.9% | 44.3% | 0.6392 / 0.6938 | 1.0557 / 0.4980 |
| EXP-T2-808 | 36.6% | 70.5% | 44.3% | 0.6296 / 0.6938 | 0.9603 / 0.4980 |
| EXP-T2-809 | 45.2% | 71.7% | 45.5% | 0.6423 / 0.6938 | 1.2287 / 0.4980 |
| EXP-T2-810 | 47.6% | 73.8% | 39.6% | 0.6076 / 0.6938 | 1.2344 / 0.4980 |

Chance varies with eligible windows and is saved per person. Training-mean forecasts and paired person intervals are also in representation_summary.json. Alignment/field readouts do not establish language understanding. The smaller256-width arms are a single-seed exploratory size comparison; this is not enough to infer a scaling law. The late-fusion control has the same total six-block budget but only two layers per stream, versus six joint layers; this depth difference prevents a pure attribution to fusion timing. The trained reset-memory arm also changes embedding pooling to an average of normalized chunk readouts, so its retrieval difference is not a pure memory effect; the same-weight long reset control isolates runtime memory removal more directly. The no-added-content arm removes phone/CTC losses but retains phonetic input reconstruction and alignment. Its character head is untrained: poor CTC output from that arm alone cannot show that its latent state contains no speech content.

## Beyond thirty seconds

Every reserved AMI participant receives the same fixed80s start and40.96s interval. All main architecture/seed outputs and registered wrong-time, reset and prefix controls are in watch.html. Text prompts use training-estimated timing with no reference alignment. All audio is retained, including poor synthesis; external recognizer text is kept separate from actual model CTC text. field_content_summary.json reports actual CTC text from the model; long_condition_summary.json includes every control, source/oracle waveform recognition and no-reference insertion counts.

| Run | Audio→video code MSE / provided static | Motion MSE / static motion | Video→audio nonempty-reference corpus WER |
|---|---:|---:|---:|
| EXP-T2-801 | 1.6024 / 2.8493 | 0.04596 / 0.04252 | 99.6% |
| EXP-T2-802 | 1.6547 / 2.8493 | 0.04509 / 0.04252 | 99.6% |
| EXP-T2-803 | 1.5805 / 2.8493 | 0.04566 / 0.04252 | 99.6% |
| EXP-T2-804 | 1.7456 / 2.8493 | 0.04499 / 0.04252 | 99.6% |

The source/oracle-renderer corpus WER is41.38%/44.40% on9 nonempty-reference cases (232 normalized words). The3 empty-owner cases yield187 source and165 oracle insertions; crosstalk makes these different from acoustically silent controls. Token-limit flags and every raw transcript remain visible.

| Provided-text run | Corpus WER of rendered audio |
|---|---:|
| EXP-T2-801 | 100.0% |
| EXP-T2-802 | 100.0% |
| EXP-T2-803 | 100.0% |
| EXP-T2-804 | 100.0% |

Long spontaneous meetings contain silent participants and crosstalk. All fixed cases remain, and normalized reference-word/edit counts distinguish no-reference intervals from ordinary WER. No general-scene visual decoder or pretrained language model is used in these fields. Native video is64×64 and appearance is supplied. The text edge represents CMU phones, not arbitrary English semantics.

## Continuous decisions and research scope

Broad speech gate: **FAIL**. Continuous SLWM sensing gate: **FAIL**. Field actor versus direct-sensor actor gate: **FAIL**. Full paired cost/coverage results are in the P1-S1 report. Each tick runs automatically from persistent memory, with frozen world weights; sensing actions change observations only.

Both SLWM and Transformer share the same recurrent field/memory design. A shared-latent result is not automatically specific to the extra SLWM signal paths. Compare paired metrics, ablations, actual compute and both seeds before attributing a gain. Useful robot decisions additionally require action-conditioned physical dynamics, goals/reward, multi-step consequence tests and a faster control loop. The1.365s received chunk plus analysis support is unsuitable evidence for fast motor control. Waveform rendering additionally uses512ms of future predicted mel context and is performed after rollout; file-based stateful inference is not an end-to-end live microphone/camera transport.
