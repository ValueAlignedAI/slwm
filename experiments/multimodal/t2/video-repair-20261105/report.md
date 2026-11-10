# T2-V1 — Video versus audio for true-gap speech repair

Video fails the registered contribution gate; retain trained audio controls and report the negative or mixed result.

Eleven registered RTX 3090 runs; 16.18 measured GPU-process minutes including validation. Checkpoints selected on equal-corpus validation only; audio61 comparison is EXP-T2-619.

The speech expert is frozen. A new audio anchor adapts it to honest waveform gaps; the visual branch then learns only bounded residual corrections on that frozen anchor. Video uses original96×64 mouth pixels, differences, a temporal CNN, local cross-attention and observed-only confidence. Audio controls have approximately matched graph-connected trainable capacity and the same positive windows. Video receives additional negative examples and therefore more compute.

Every input STFT window touching hidden samples is removed. Actual WAV repairs erase originals before any transform and preserve observed PCM samples exactly. This is a new waveform-passthrough task, distinct from the older full-synthesis protocol.

## Primary320ms results

Relative effects are means of paired per-clip reductions, with2,000-draw person-cluster95% intervals. Positive means video improves the baseline. The registered gate needs≥5% and a positive interval in both seeds on both sets.

| Test | Seed | Audio MSE | Video MSE | Video gain vs audio [95%] | vs anchor [95%] |
|---|---:|---:|---:|---:|---:|
| crema_test (n=202) | 61 | 0.51216 | 0.45996 | 9.30% [6.71, 11.79] | 10.35% [7.86, 12.85] |
| crema_test (n=202) | 62 | 0.50815 | 0.45828 | 9.11% [6.50, 11.55] | 10.94% [8.06, 13.77] |
| rav_test (n=128) | 61 | 0.76673 | 0.71038 | 5.00% [-1.68, 11.10] | 7.03% [0.13, 13.40] |
| rav_test (n=128) | 62 | 0.75788 | 0.70666 | 4.87% [-2.65, 11.44] | 7.53% [-0.56, 14.21] |

## Temporal and pair controls

Correct video must beat each control by≥2% with a positive interval. Wrong video is the same person saying a different utterance. Shifts use real±240ms frames, with length≥50 eligibility. Missing video returns the frozen audio anchor exactly; that fallback is not claimed equal to the stronger two-branch audio control.

| Test / seed | Wrong | +240ms | −240ms | Static |
|---|---:|---:|---:|---:|
| crema_test/61 | 16.34% [12.65, 19.79] | 16.47% [13.14, 19.60] | 17.71% [15.53, 19.93] | 10.36% [7.87, 12.86] |
| crema_test/62 | 16.82% [13.13, 20.48] | 17.86% [14.26, 21.54] | 18.66% [15.65, 21.83] | 10.94% [8.06, 13.76] |
| rav_test/61 | 15.10% [8.72, 21.74] | 14.84% [9.27, 20.19] | 16.23% [7.60, 23.81] | 7.03% [0.13, 13.40] |
| rav_test/62 | 17.45% [12.05, 22.93] | 20.01% [12.94, 25.92] | 22.03% [12.32, 29.25] | 7.53% [-0.56, 14.21] |

## All trained artifacts

Total count includes the frozen speech expert/anchor, all frontend/decoder weights and inactive auxiliary heads. Active trainable counts measure graph participation in the first optimizer update; no inactive padding is used to claim a capacity match.

| ID | Variant | Selected / total updates | Active trainable | Total | Minutes | C320 MSE | R320 MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| EXP-T2-601 | anchor | 1600 / 1600 | 1,430,080 | 6,567,522 | 0.51 | 0.51680 | 0.78139 |
| EXP-T2-602 | anchor | 1600 / 1600 | 1,430,080 | 6,567,522 | 0.51 | 0.52017 | 0.78115 |
| EXP-T2-611 | audio | 2000 / 2400 | 1,430,080 | 8,034,755 | 0.92 | 0.51302 | 0.77518 |
| EXP-T2-612 | audio | 2400 / 2400 | 1,430,080 | 8,034,755 | 0.93 | 0.50815 | 0.75788 |
| EXP-T2-613 | pca_video | 2000 / 2400 | 1,068,833 | 7,636,355 | 1.31 | 0.49996 | 0.72540 |
| EXP-T2-614 | video | 2400 / 2400 | 1,423,025 | 7,990,547 | 2.19 | 0.45996 | 0.71038 |
| EXP-T2-615 | video | 2400 / 2400 | 1,423,025 | 7,990,547 | 2.18 | 0.45828 | 0.70666 |
| EXP-T2-616 | ungated_video | 2400 / 2400 | 1,398,256 | 7,990,547 | 2.15 | 0.46693 | 0.73174 |
| EXP-T2-617 | shuffled_video | 2400 / 2400 | 1,423,025 | 7,990,547 | 2.23 | 0.51079 | 0.75815 |
| EXP-T2-618 | no_negatives | 2000 / 2400 | 1,423,025 | 7,990,547 | 1.44 | 0.46255 | 0.70838 |
| EXP-T2-619 | audio | 4800 / 4800 | 1,430,080 | 8,034,755 | 1.81 | 0.51216 | 0.76673 |

Full per-clip metrics, confidence gates,160/480ms secondary results, actual PCM outputs, optimizer snapshots, configs and hashes are retained alongside this report. EXP-T2-619 has twice the update exposure; it is a stronger audio control, not data-budget matched.

## Speech diagnostic

Pinned Whisper-tiny.en (37,760,256 external evaluation parameters) evaluates the32 fixed examples. Its text is never SLWM output or training supervision. Mean per-clip WER can exceed100%. The speech gate needs≥5% relative WER reduction and positive person-cluster intervals against both selected audio and silence, in both seeds and both sets. Oracle spectrum is an evaluation-only decoder ceiling: true clean mel, observed-only phase, never a model input or candidate.

| Test | Original | Silence | Interpolation | Oracle spectrum | Audio61 | Video61 | Audio62 | Video62 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| crema_test (n=16) | 20.09% | 58.04% | 54.69% | 18.53% | 56.92% | 41.96% | 49.55% | 41.07% |
| rav_test (n=16) | 0.00% | 11.46% | 20.83% | 0.00% | 15.62% | 15.62% | 14.58% | 14.58% |

Feature gate: **FAIL**. Speech diagnostic gate: **FAIL**.

## Data, limitations and next decision

CREMA:5,061 training clips/62people,210 validation/7people,240 reused test/8people. Fresh RAVDESS:240 training clips from07,08,19,20;128 validation clips from earlier R1 test actors03,04,05,06,15,16,17,18;128 new test clips from09,10,11,12,21,22,23,24. Only length≥50 clips enter the main shifted-control comparison; exclusions and all-clip diagnostic scores are saved. Both corpora use acted speech and restricted phrases. RAV is new people in a partly trained domain, not new vocabulary. All normalizers/PCA stay fixed from CREMA training. No external generator supplies trained outputs.

The frozen expert includes the previous CREMA/Libri-trained R2 core and auxiliary content readout; this cycle is additional adaptation, not training from scratch. The visual branch is a diagnostic audio-conditioned correction head, not a newly demonstrated shared multimodal world model or learned output policy. Confidence and no-video fallback are engineering safeguards, not calibrated uncertainty or evidence of understanding.

Keep the true-gap masking, exact observation preservation, trained checkpoints and ablation record. A failed video gate calls for a new preregistered hypothesis and genuinely new holdout before claiming improvement; increasing parameter count alone is not justified. See the protocol and training guide for reproduction and license limits.
