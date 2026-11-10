# T2-C1 — speech content and mistimed-video rejection (word evaluation revision2)

The original word metric included19 short-pause frames. Revision2 excludes both sil and sp, reusing the same training-fitted ridge coefficients. All spectra, weights, thresholds and waveforms remain unchanged; original metrics and report are retained.

All choices use training, selection or independent calibration people. Final people and exact sentence combinations were reserved before fitting. Primary gates remain fixed, including a coverage requirement so rejecting every video cannot pass.

| Gate | Both-seed outcome |
|---|---|
| Field content + spectral/clean guardrails | PASS |
| Mistimed-video rejection + aligned coverage | PASS |
| Useful calibrated video beyond strong audio | PASS |
| Independent speech diagnostic | FAIL |

## Content and video effects

Effects are paired per-clip means, with2,000-draw person-cluster95% intervals. Word readouts are separate training-fitted ridge classifiers of processed audio fields, with no raw bypass. Text labels are never supplied during practical audio/video inference.

| Seed | Content gap-MSE gain | Field word-accuracy gain | Clean-MSE ratio | Calibrated video gain vs audio | Aligned coverage |
|---|---:|---:|---:|---:|---:|
| 71 | 4.02% [2.81, 5.82] | 9.72pp [8.21, 10.97] | 1.009 | 12.26% [9.65, 14.86] | 81.84% [77.33, 85.92] |
| 72 | 3.12% [1.47, 5.30] | 12.43pp [10.95, 13.92] | 1.007 | 11.24% [8.24, 14.21] | 83.85% [79.53, 87.83] |

## Rejection with usefulness accounted for

The video model stores the frozen strong audio candidate. Every rejection returns that prediction and its unit/latent state exactly. Acceptance requires zero predicted offset and an observed-only quality threshold fixed on three separate calibration people. Audio/video candidates have approximately matched graph-connected trained capacity; the deployed video model has additional frozen fallback storage and extra negative-example compute.

| Seed / video | Accepted | Repair harm vs strong audio | Raw unfiltered harm |
|---|---:|---:|---:|
| 71 / shift6 | 0.53% [0.00, 1.09] | 0.04% [0.00, 0.10] | 34.10% [24.86, 43.40] |
| 71 / shift-6 | 0.53% [0.00, 1.08] | 0.14% [0.00, 0.36] | 34.77% [28.00, 41.51] |
| 71 / shift4 | 0.13% [0.00, 0.39] | 0.03% [0.00, 0.09] | 39.63% [32.89, 46.71] |
| 71 / shift-4 | 0.13% [0.00, 0.39] | 0.00% [0.00, 0.00] | 39.29% [32.57, 46.91] |
| 71 / wrong | 13.08% [10.32, 15.56] | 3.52% [1.69, 5.21] | 30.99% [24.97, 37.10] |
| 71 / static | 10.81% [7.46, 14.36] | 1.81% [0.83, 2.92] | 13.27% [10.67, 16.24] |
| 71 / blank | 4.67% [2.99, 6.38] | 0.37% [0.09, 0.66] | 14.97% [10.49, 19.70] |
| 71 / missing | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | -13.70% [-16.60, -10.77] |
| 72 / shift6 | 0.93% [0.00, 1.95] | 0.11% [-0.02, 0.27] | 34.13% [24.88, 43.17] |
| 72 / shift-6 | 0.67% [0.13, 1.43] | 0.13% [0.03, 0.27] | 33.11% [26.46, 40.23] |
| 72 / shift4 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 41.14% [34.38, 48.42] |
| 72 / shift-4 | 0.27% [0.00, 0.79] | 0.02% [0.00, 0.05] | 40.83% [33.81, 49.16] |
| 72 / wrong | 11.88% [9.47, 14.49] | 3.56% [1.75, 5.66] | 30.47% [24.75, 36.83] |
| 72 / static | 11.75% [9.05, 14.46] | 1.76% [1.24, 2.33] | 12.44% [10.05, 15.38] |
| 72 / blank | 9.75% [6.71, 12.34] | 1.24% [0.51, 2.06] | 11.78% [9.39, 14.62] |
| 72 / missing | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | -12.79% [-16.11, -9.45] |

Positive harm is worse than audio alone. ±240ms is primary; ±160ms is an untrained shift magnitude. Wrong utterances, static/blank video and missing-video behavior are diagnostics. Wrong sources differing only within an erased word can be observationally ambiguous; timing rejection is not calibrated semantic truth.

## Every trained artifact

| Run | Variant / seed | Selected / updates | Trainable / frozen | Total | Train minutes | Test320 MSE | Missing-word accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| EXP-T2-701 | base units=0.0 / 71 | 4000 / 4000 | 5,133,942 / 0 | 5,133,942 | 1.99 | 1.03930 | 1.54% |
| EXP-T2-702 | base units=0.2 / 71 | 4000 / 4000 | 5,133,942 / 0 | 5,133,942 | 2.00 | 0.98769 | 30.72% |
| EXP-T2-703 | base units=0.0 / 72 | 3200 / 4000 | 5,133,942 / 0 | 5,133,942 | 1.99 | 1.03240 | 1.41% |
| EXP-T2-704 | base units=0.2 / 72 | 3200 / 4000 | 5,133,942 / 0 | 5,133,942 | 2.01 | 0.99143 | 30.03% |
| EXP-T2-711 | audio units=0.2 / 71 | 2800 / 3000 | 1,294,570 / 5,133,942 | 6,428,512 | 1.38 | 0.94038 | 33.46% |
| EXP-T2-712 | audio units=0.2 / 72 | 3000 / 3000 | 1,294,570 / 5,133,942 | 6,428,512 | 1.37 | 0.94055 | 33.19% |
| EXP-T2-713 | video units=0.2 / 71 | 3000 / 3000 | 1,285,308 / 6,428,512 | 7,713,820 | 3.71 | 0.80564 | 41.31% |
| EXP-T2-714 | video units=0.2 / 72 | 2800 / 3000 | 1,285,308 / 6,428,512 | 7,713,820 | 3.70 | 0.81331 | 42.05% |
| EXP-T2-715 | video units=0.0 / 71 | 3000 / 3000 | 1,285,308 / 6,428,512 | 7,713,820 | 3.72 | 0.81179 | 34.18% |
| EXP-T2-716 | shuffled units=0.2 / 71 | 2800 / 3000 | 1,285,308 / 6,428,512 | 7,713,820 | 3.81 | 0.93838 | 32.80% |

Measured training wall time on the RTX3090: 25.68 minutes, including validation. Historical R2 pretraining and corpus preparation are additional. All best optimizer/RNG states, final weights, counts, sources/configs, curves, per-clip scores and160/480ms secondary results are retained.

## Actual speech and text diagnostics

All32 listening IDs were fixed before final scores. Observed PCM is preserved on the video-aligned decoded timeline; initial source stream trimming is recorded. Practical model features and phase erase original hidden samples before transforms. The true-spectrum and timed-text conditions deliberately receive reference information and are clearly labeled diagnostics, not practical repair results. Whisper-tiny.en is pinned, external and never trains a generator.

| Condition | Mean clip WER |
|---|---:|
| audio71 | 54.69% |
| audio72 | 48.96% |
| content71 | 51.56% |
| content72 | 52.60% |
| interpolation | 53.65% |
| missing_video71 | 54.69% |
| oracle_spectrum | 19.79% |
| plain71 | 55.21% |
| plain72 | 56.25% |
| raw_shift71 | 53.12% |
| raw_video71 | 47.92% |
| raw_wrong71 | 55.73% |
| reference | 19.27% |
| shift71 | 54.69% |
| silence | 52.60% |
| static71 | 56.25% |
| timed_text71 | 48.96% |
| video71 | 51.04% |
| video72 | 53.65% |
| wrong71 | 53.65% |

See `exploration/intelligibility.json` for paired intervals and raw WER. Digit/letter splitting and single-digit number normalization are identical for reference and hypotheses; no grammar/homophone correction is applied. The `word_probes.json` files instead contain actual SLWM closed-vocabulary outputs for the context, not recognizer text and not a learned output-policy commitment.

## Scope, data and failures

Training uses GRID people01–16 on sentence-hash buckets0–7. Selection17–20 and calibration22–24 use bucket8. Final25–32 use bucket9: both people and complete sentence combinations are new. Speaker21 has no video;33/34 remain unopened. The vocabulary/grammar stays restricted, so this is not novel-vocabulary or natural-world validation. CREMA/Libri training windows retain broader acoustics; their final checks are reused retention diagnostics. No codec is fitted to GRID holdouts.

The0.5% source alarm stopped preparation. Raw CRC/frame-count review verified90 genuinely short video files, all from training person08 (88train/2training-person diagnostic). The exact reviewed list passes the alarm; the50-packet input rule and all holdout people stay unchanged. Failed preparation logs and source review are retained.

The content objective is compared to the same architecture without its explicit unit CE; both have the timed-text auxiliary reconstruction interface. This is not an ablation removing all text supervision. Video pooling/translation, lexical objectives and the admission detector are an engineering package, with no claim that any one module alone causes all gains. Ungated results on the same weights isolate the admission decision.

T1-S2 separately measures causal English prediction at GPT-2 parameter scale on a much larger corpus. These are distinct trained weights. Neither study establishes a unified124M multimodal world model, future physical dynamics, general reasoning or a learned policy. A failed gate remains a negative/mixed result; further changes require a new preregistered holdout.
