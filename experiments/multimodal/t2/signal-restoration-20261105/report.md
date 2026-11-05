# T2-R1 — shared-field speech-packet restoration

**Reliable benefit from adding video is not established.** The model repairs spectral features and preserves timing, but wrong-video sensitivity is not sufficient: correct video must also beat sound alone. The frozen field readout motivates the separate R2 content objective.

## Frozen gates

| Gate | Result |
|---|---|
| clean_output_retention | PASS |
| repair_over_raw_av_and_vanilla | FAIL |
| positive_multimodal_contribution | FAIL |
| timing_retention | PASS |

## Models and compute

| ID | Arm | Total neural weights | Trainable | Frozen | Selected step | Train seconds |
|---|---|---:|---:|---:|---:|---:|
| EXP-T2-501 | signal | 5,083,841 | 5,083,841 | 0 | 1200 | 70.2 |
| EXP-T2-502 | signal | 5,083,841 | 5,083,841 | 0 | 1600 | 70.4 |
| EXP-T2-503 | vanilla | 5,027,521 | 5,027,521 | 0 | 1600 | 55.6 |
| EXP-T2-504 | no_alignment | 5,083,841 | 5,083,841 | 0 | 1200 | 35.5 |
| EXP-T2-505 | no_augmentation | 5,083,841 | 5,083,841 | 0 | 1600 | 69.2 |
| EXP-T2-506 | no_skip | 5,083,841 | 5,083,841 | 0 | 1600 | 68.8 |
| EXP-T2-507 | raw_audio | 548,352 | 548,352 | 0 | 1600 | 10.0 |
| EXP-T2-508 | raw_av | 622,848 | 622,848 | 0 | 1600 | 10.3 |
| EXP-T2-509 | frozen_core | 5,083,841 | 548,352 | 4,535,489 | 1200 | 19.0 |

New decoder/core training: 6.82 RTX 3090 minutes; data preparation, readout fitting, scoring and rendering excluded. Update budgets are matched, not wall time or total historical pretraining compute.

## crema_test

| ID | Paired hole MSE | Sound-only hole MSE | Clean-sound MSE | Timing accuracy |
|---|---:|---:|---:|---:|
| EXP-T2-501 | 0.32126 | 0.33766 | 0.02800 | 98.5% |
| EXP-T2-502 | 0.32302 | 0.32995 | 0.02789 | 97.5% |
| EXP-T2-503 | 0.32807 | 0.34301 | 0.02811 | 92.6% |
| EXP-T2-504 | 0.31589 | 0.33139 | 0.02775 | 98.0% |
| EXP-T2-505 | 0.31829 | 0.32777 | 0.02788 | 97.5% |
| EXP-T2-506 | 0.39337 | 0.38767 | 0.16696 | 97.0% |
| EXP-T2-507 | 0.33699 | 0.33699 | 0.02831 | — |
| EXP-T2-508 | 0.34142 | 0.34273 | 0.02836 | — |
| EXP-T2-509 | 0.33412 | 0.35147 | 0.02836 | 96.0% |

Correct-video versus sound-only/wrong-video mean per-clip gains (cluster intervals):

- EXP-T2-501: masked_audio: +1.82% [-0.23, +4.00]; wrong_video: +14.20% [+12.56, +15.74]
- EXP-T2-502: masked_audio: -0.65% [-2.20, +1.10]; wrong_video: +13.11% [+11.92, +14.33]

Independent timing controls (correct / wrong sound / zero sound / energy baseline):

- EXP-T2-501: 98.5% / 16.8% / 2.5% / 36.1%; correct interval [97.2, 100.0]%, chance 14.3%.
- EXP-T2-502: 97.5% / 19.8% / 4.5% / 36.1%; correct interval [95.8, 99.4]%, chance 14.3%.

Recognizer mean clip WER: reference 15.4%, target codec 24.3%, interpolation 85.9%, full seeds 71.4%/69.2%, raw audio 67.6%, observation-matched silence 113.2%.

## libri_test

| ID | Paired hole MSE | Sound-only hole MSE | Clean-sound MSE | Timing accuracy |
|---|---:|---:|---:|---:|
| EXP-T2-501 | 0.82090 | 0.82090 | 0.05801 | — |
| EXP-T2-502 | 0.83442 | 0.83442 | 0.05768 | — |
| EXP-T2-503 | 0.81953 | 0.81953 | 0.05797 | — |
| EXP-T2-504 | 0.81189 | 0.81189 | 0.05742 | — |
| EXP-T2-505 | 0.80279 | 0.80279 | 0.05770 | — |
| EXP-T2-506 | 0.87534 | 0.87534 | 0.25800 | — |
| EXP-T2-507 | 0.84956 | 0.84956 | 0.05980 | — |
| EXP-T2-508 | 0.84763 | 0.84763 | 0.05960 | — |
| EXP-T2-509 | 0.84429 | 0.84429 | 0.05900 | — |

Correct-video versus sound-only/wrong-video mean per-clip gains (cluster intervals):

- EXP-T2-501: masked_audio: +0.00% [+0.00, +0.00]; wrong_video: +0.00% [+0.00, +0.00]
- EXP-T2-502: masked_audio: +0.00% [+0.00, +0.00]; wrong_video: +0.00% [+0.00, +0.00]

Recognizer mean clip WER: reference 4.6%, target codec 6.7%, interpolation 54.1%, full seeds 47.1%/48.8%, raw audio 50.3%, observation-matched silence 59.8%.

## ravdess_test

| ID | Paired hole MSE | Sound-only hole MSE | Clean-sound MSE | Timing accuracy |
|---|---:|---:|---:|---:|
| EXP-T2-501 | 0.70359 | 0.69364 | 0.05614 | 68.8% |
| EXP-T2-502 | 0.70762 | 0.68058 | 0.05578 | 69.5% |
| EXP-T2-503 | 0.70859 | 0.69233 | 0.05644 | 59.4% |
| EXP-T2-504 | 0.70297 | 0.68456 | 0.05548 | 77.3% |
| EXP-T2-505 | 0.70251 | 0.68638 | 0.05591 | 63.3% |
| EXP-T2-506 | 1.03277 | 0.80166 | 0.29711 | 70.3% |
| EXP-T2-507 | 0.68921 | 0.68921 | 0.05739 | — |
| EXP-T2-508 | 0.69067 | 0.68934 | 0.05724 | — |
| EXP-T2-509 | 0.72725 | 0.72035 | 0.05711 | 62.5% |

Correct-video versus sound-only/wrong-video mean per-clip gains (cluster intervals):

- EXP-T2-501: masked_audio: -4.23% [-6.53, -1.76]; wrong_video: +3.44% [+0.80, +6.23]
- EXP-T2-502: masked_audio: -6.92% [-10.05, -3.83]; wrong_video: +2.90% [-0.11, +6.32]

Independent timing controls (correct / wrong sound / zero sound / energy baseline):

- EXP-T2-501: 68.8% / 24.2% / 3.1% / 4.7%; correct interval [55.4, 78.9]%, chance 14.3%.
- EXP-T2-502: 69.5% / 18.8% / 3.1% / 4.7%; correct interval [54.7, 82.0]%, chance 14.3%.

Recognizer mean clip WER: reference 0.0%, target codec 3.1%, interpolation 55.2%, full seeds 46.9%/50.0%, raw audio 47.9%, observation-matched silence 65.6%.

## Mechanism and failure analysis

The bypass protects decoded observations while allowing the field to specialize in corrections. The full skip model's linear coarse-sound readout is worse than the old frozen core; the no-skip arm's readout is better even though its generated signal is worse. This is evidence about decodability/incentives, not proof of irreversible information deletion. R2 tests an explicit observed-content gradient.

A matched silence control was added after the initial listening pass. The original `silence_gaps` retains 100Hz target detail outside holes and therefore has richer observations than the models; it remains as a diagnostic, never a fair model baseline. `coarse_silence.wav` instead blanks the same 25Hz-derived interpolation available to every model. Separate original and matched-control transcripts/results remain intact.

## Data, observations and claim limits

CREMA-D keeps 5,061 train / 210 validation / 240 test clips, with disjoint people; its test is reused. LibriSpeech dev-clean adds 1,278 training clips from 32 people and 64 validation clips from eight other people. Long training clips use the first 5.12s; no transcript is a training target. Every update draws 24 CREMA + 8 Libri windows, 1,600×32 updates, exactly 32 packets per learned context. Half of CREMA video inputs are dropped. Only head/data/new update budgets match: total core parameters, pretraining and compute differ. Both signal and vanilla cores warmstart from their earlier corresponding 1,600-update M1 runs; raw controls have no such pretraining. Old unused reconstruction heads are removed. No wider cached world features enter this cycle.

R1 first reserves 48 complete test-clean utterances from 12 new Libri people and 128 RAVDESS clips from eight new people. These guide the separately labeled R2 repair and become reused diagnostics. R2 reserves 48 complete test-other utterances from 12 different Libri people as its fresh, harder recording-condition check. Neither external set fits statistics, weights, readouts or checkpoint selection. All codec/PCA/statistical values remain from CREMA training only (249,376 fitted values). Libri archives are checked against official MD5; RAVDESS uses selected ZIP-member CRC and individual media SHA256, with archive MD5 declarations retained rather than recomputed for the unfetched full ZIPs. Sources: [LibriSpeech](https://www.openslr.org/12), Panayotov et al. (2015), CC BY 4.0; [RAVDESS](https://zenodo.org/records/1188976), Livingstone & Russo (2018), CC BY-NC-SA 4.0, noncommercial evaluation only.

Two four-packet holes with observed boundary anchors remove 25% of each middle window. Deltas are computed after observed-only interpolation. Hidden target changes cannot affect any input or prediction. Video is simultaneous context, not a future observation. Reconstruction uses hidden-packet weight 4 and temporal L1; independent alignment never sees its paired modality. Gain ±12dB and smooth four-knot EQ ±6dB affect BOTH observations and target, so the model preserves the observed channel/level rather than guessing an unknowable original gain. These are mel-domain approximations, not real room simulation.

Known CREMA prompt IDs only exclude contrastive false negatives, and person groups define controlled negative examples. No transcript words or speaker ID is an encoder input; this is not text-generation training or a fully unlabeled data claim.

Important scope: 64ms centered codec windows can straddle packet boundaries. This withholds mel packets, not sample-exact waveform spans. Actual audio uses inverse mel and newly estimated Griffin-Lim phase; no original waveform samples supply model output. Listening uses one global mask before overlapping windows, preventing a hidden packet from returning in a neighboring window. Global interpolation can use observed anchors outside a render window equally for all models; primary window metrics use only their 32-packet context. Models have not learned policy/uncertainty commitment, free text or autonomous world dynamics.

Checkpoint selection uses ONLY the equal-corpus mean validation hole MSE every 400 updates. All seeds, checkpoints, last weights, optimizer-bearing best states, curves, hashes and failures remain. Paired confidence intervals resample people (2,000 draws), not frames. Ridge readouts fit 512 CREMA + 512 Libri training clips only, fixed regularization, 16,448 coefficient/bias values plus 256 input means, and see ONLY frozen world fields. Lower readout MSE is linear decodability, not irreversible information capacity or semantic understanding. Raw input already contains the observed coarse target, so copying is a trivial perfect baseline for that assay.

Research basis: [masked audiovisual representation learning](https://arxiv.org/abs/2201.02184) and [speech inpainting](https://www.isca-archive.org/interspeech_2023/montesinos23_interspeech.html). No pretrained generator from those works is used. The only external neural evaluator is pinned Whisper-tiny.en (37,760,256 weights, revision `87c7102498dcde7456f24cfd30239ca606ed9063`), never a training/generation component. Recognizer text is not model text; WER is fallible and can exceed 100%.
