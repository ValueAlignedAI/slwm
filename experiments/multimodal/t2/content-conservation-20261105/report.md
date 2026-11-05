# T2-R2 — conserve observed content in the shared field

**Retain the explicit content objective: both primary seeds pass its registered gates.** These are content/output/timing guardrails; the R1 positive-video-contribution gate remains failed.

Primary auxiliary=.5 two-seed gates: `{"clean_output_guardrail": true, "field_content_conservation": true, "repair_guardrail": true, "timing_retention": true}`.

The intervention adds only 16,448 linear-head weights and an observed-packet MSE objective. The primary decoder and raw preservation path are unchanged. All runs restart from the same original M1 source and use the exact R1 training data, sampled windows and hyperparameters. R1 already exposed test-clean/RAVDESS; only test-other is fresh here. Coefficient 2 is a predeclared sensitivity run, not a winner selected after test inspection.

## Models and compute

| ID | Auxiliary weight | Total neural weights | Selected step | Train seconds |
|---|---:|---:|---:|---:|
| EXP-T2-521 | 0.5 | 5,100,289 | 1200 | 74.3 |
| EXP-T2-522 | 0.5 | 5,100,289 | 1600 | 73.4 |
| EXP-T2-523 | 0.0 | 5,100,289 | 1200 | 73.4 |
| EXP-T2-524 | 2.0 | 5,100,289 | 1200 | 74.5 |

Training total 4.93 RTX 3090 minutes, preparation/scoring excluded. Zero-auxiliary control reproduces R1 paired-hole MSE with absolute differences: `{"crema_test": 0.0, "libri_test": 0.0, "ravdess_test": 0.0}`. The unused auxiliary head is retained in the parameter count for that control.

## Frozen-field decodability: audio-only core inputs, no raw bypass

| Model | CREMA | test-clean (reused) | RAVDESS (reused) | test-other (fresh) |
|---|---:|---:|---:|---:|
| R1 EXP-T2-509 | 0.11189 | 0.16402 | 0.18591 | 0.16769 |
| R1 EXP-T2-501 | 0.17371 | 0.29876 | 0.45728 | 0.33348 |
| R1 EXP-T2-502 | 0.17818 | 0.31833 | 0.54242 | 0.37150 |
| R1 EXP-T2-506 | 0.07278 | 0.09763 | 0.11285 | 0.10110 |
| EXP-T2-521 | 0.03819 | 0.05151 | 0.06536 | 0.06453 |
| EXP-T2-522 | 0.03711 | 0.04844 | 0.06170 | 0.06234 |
| EXP-T2-523 | 0.17371 | 0.29876 | 0.45728 | 0.33348 |
| EXP-T2-524 | 0.01811 | 0.02296 | 0.03265 | 0.03482 |

Paired mean per-clip readout improvements over the corresponding R1 full seed (ratios of aggregate MSE can differ):

| Model | Set | Relative gain | Person-cluster 95% interval |
|---|---|---:|---:|
| EXP-T2-521 | libri_test | 80.1% | [78.0, 82.2]% |
| EXP-T2-521 | other_test | 79.7% | [75.1, 83.4]% |
| EXP-T2-522 | libri_test | 81.6% | [79.3, 83.9]% |
| EXP-T2-522 | other_test | 81.7% | [76.6, 85.4]% |

## Primary output and timing guardrails

| Model | CREMA hole MSE | test-clean hole MSE | Fresh test-other hole MSE | Fresh clean MSE | CREMA timing | RAVDESS timing |
|---|---:|---:|---:|---:|---:|---:|
| EXP-T2-521 | 0.32991 | 0.82747 | 0.70086 | 0.04704 | 94.1% | 70.3% |
| EXP-T2-522 | 0.32421 | 0.83047 | 0.68766 | 0.04698 | 97.5% | 72.7% |
| EXP-T2-523 | 0.32126 | 0.82090 | 0.69418 | 0.04640 | 98.5% | 68.8% |
| EXP-T2-524 | 0.33303 | 0.83121 | 0.70057 | 0.04726 | 94.1% | 74.2% |

## Fresh test-other listening diagnostic

| Condition | Mean clip WER |
|---|---:|
| EXP-T2-501 | 72.6% |
| EXP-T2-507 | 78.3% |
| EXP-T2-509 | 73.8% |
| EXP-T2-521 | 70.3% |
| EXP-T2-522 | 70.8% |
| EXP-T2-523 | 72.6% |
| EXP-T2-524 | 69.1% |
| coarse_silence | 87.0% |
| codec100 | 31.2% |
| interpolation | 79.7% |
| reference | 26.5% |

No intelligibility claim follows a feature-MSE improvement alone. Compare the same-observation silence control and the untouched original/codec controls. All WAV files are actual predictions with estimated phase, and the fresh set has only 12 people. Reports retain all predeclared coefficients and both primary seeds, including failed gates.

## Data, observations and claim limits

CREMA-D keeps 5,061 train / 210 validation / 240 test clips, with disjoint people; its test is reused. LibriSpeech dev-clean adds 1,278 training clips from 32 people and 64 validation clips from eight other people. Long training clips use the first 5.12s; no transcript is a training target. Every update draws 24 CREMA + 8 Libri windows, 1,600×32 updates, exactly 32 packets per learned context. Half of CREMA video inputs are dropped. Only head/data/new update budgets match: total core parameters, pretraining and compute differ. Both signal and vanilla cores warmstart from their earlier corresponding 1,600-update M1 runs; raw controls have no such pretraining. Old unused reconstruction heads are removed. No wider cached world features enter this cycle.

R1 first reserves 48 complete test-clean utterances from 12 new Libri people and 128 RAVDESS clips from eight new people. These guide the separately labeled R2 repair and become reused diagnostics. R2 reserves 48 complete test-other utterances from 12 different Libri people as its fresh, harder recording-condition check. Neither external set fits statistics, weights, readouts or checkpoint selection. All codec/PCA/statistical values remain from CREMA training only (249,376 fitted values). Libri archives are checked against official MD5; RAVDESS uses selected ZIP-member CRC and individual media SHA256, with archive MD5 declarations retained rather than recomputed for the unfetched full ZIPs. Sources: [LibriSpeech](https://www.openslr.org/12), Panayotov et al. (2015), CC BY 4.0; [RAVDESS](https://zenodo.org/records/1188976), Livingstone & Russo (2018), CC BY-NC-SA 4.0, noncommercial evaluation only.

Two four-packet holes with observed boundary anchors remove 25% of each middle window. Deltas are computed after observed-only interpolation. Hidden target changes cannot affect any input or prediction. Video is simultaneous context, not a future observation. Reconstruction uses hidden-packet weight 4 and temporal L1; independent alignment never sees its paired modality. Gain ±12dB and smooth four-knot EQ ±6dB affect BOTH observations and target, so the model preserves the observed channel/level rather than guessing an unknowable original gain. These are mel-domain approximations, not real room simulation.

Known CREMA prompt IDs only exclude contrastive false negatives, and person groups define controlled negative examples. No transcript words or speaker ID is an encoder input; this is not text-generation training or a fully unlabeled data claim.

Important scope: 64ms centered codec windows can straddle packet boundaries. This withholds mel packets, not sample-exact waveform spans. Actual audio uses inverse mel and newly estimated Griffin-Lim phase; no original waveform samples supply model output. Listening uses one global mask before overlapping windows, preventing a hidden packet from returning in a neighboring window. Global interpolation can use observed anchors outside a render window equally for all models; primary window metrics use only their 32-packet context. Models have not learned policy/uncertainty commitment, free text or autonomous world dynamics.

Checkpoint selection uses ONLY the equal-corpus mean validation hole MSE every 400 updates. All seeds, checkpoints, last weights, optimizer-bearing best states, curves, hashes and failures remain. Paired confidence intervals resample people (2,000 draws), not frames. Ridge readouts fit 512 CREMA + 512 Libri training clips only, fixed regularization, 16,448 coefficient/bias values plus 256 input means, and see ONLY frozen world fields. Lower readout MSE is linear decodability, not irreversible information capacity or semantic understanding. Raw input already contains the observed coarse target, so copying is a trivial perfect baseline for that assay.

Research basis: [masked audiovisual representation learning](https://arxiv.org/abs/2201.02184) and [speech inpainting](https://www.isca-archive.org/interspeech_2023/montesinos23_interspeech.html). No pretrained generator from those works is used. The only external neural evaluator is pinned Whisper-tiny.en (37,760,256 weights, revision `87c7102498dcde7456f24cfd30239ca606ed9063`), never a training/generation component. Recognizer text is not model text; WER is fallible and can exceed 100%.
