# Speaker transfer and sustained decisions

A single fixed ridge head is fitted on eight training examples per command from one modality, then applied to another with no adaptation. The table averages six directional swaps on four development speakers. Balanced chance is 25%. The intervals resample whole speakers.

| Parameters | Seed | Hybrid | Dense GPT-2 | Hybrid gain, pp | Paired interval, pp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 116,502,025 | [511](supplemental/frozen-v5e-anchor-511.json) | 28.44% | 26.78% | 1.66 | 0.35 to 3.27 |
| 116,502,025 | [512](supplemental/frozen-v5e-anchor-512.json) | 26.08% | 25.45% | 0.64 | -0.49 to 1.76 |
| 202,346,377 | [511](supplemental/frozen-v5e-larger-511.json) | 32.18% | 24.15% | 8.02 | 6.40 to 8.44 |
| 202,346,377 | [512](supplemental/frozen-v5e-larger-512.json) | 26.48% | 23.03% | 3.46 | 2.37 to 4.11 |

At 116,502,025 parameters the two-seed shared-content gate was unmet. Each linked result includes all nine directional cells, source-specific raw baselines, within-speaker shuffles and retrieval.

At 202,346,377 parameters the two-seed shared-content gate passed. Each linked result includes all nine directional cells, source-specific raw baselines, within-speaker shuffles and retrieval.

## Independent speaker and sentence confirmation

The same heads are evaluated on eight different speakers and sentence combinations excluded from both support and development. Each readout retains its original support and requires no adaptation.

| Parameters | Seed | Hybrid | Dense | Hybrid gain, pp | Speaker-paired interval, pp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 202,346,377 | [511](supplemental/speakers-v5e-larger-511.json) | 31.04% | 23.98% | 7.05 | 6.50 to 8.04 |
| 202,346,377 | [512](supplemental/speakers-v5e-larger-512.json) | 25.81% | 22.81% | 3.00 | 1.71 to 4.44 |

The eight-speaker replication at 202,346,377 parameters passed the original two-seed shared-content gate.

## Final checkpoints on 32 unseen goal-reaching scenes

All rules ran 24 successive decisions per scene. Lower final distance is better. The speech diagnostic uses the same frozen recognizer and held development speakers as G5.

| Parameters | Seed | Hybrid, px | Dense, px | Prior, px | Hybrid reset, px | Dense reset, px |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 116,502,025 | [511](supplemental/terminal-v5e-anchor-511.json) | 81.26 | 144.13 | 152.56 | 73.78 | 74.49 |
| 202,346,377 | [511](supplemental/terminal-v5e-larger-511.json) | 85.97 | 167.20 | 152.56 | 65.54 | 72.42 |
| 202,346,377 | [512](supplemental/terminal-v5e-larger-512.json) | 117.59 | 121.70 | 152.56 | 71.30 | 63.70 |

## Visual contribution beyond the identical action plan

The controls use 512 additional independent scenes. Positive reductions favor correct video. The gate requires positive paired intervals for error and decision regret against both erased and wrong-scene input.

| Parameters | Seed | Core | Error reduction vs erased, px | Regret reduction vs erased, px | Visual gate |
| ---: | ---: | --- | ---: | ---: | --- |
| 116,502,025 | 511 | gpt2 | 10.56 | 3.11 | unmet |
| 116,502,025 | 511 | hybrid | 11.05 | 1.77 | passed |
| 202,346,377 | 511 | gpt2 | 16.26 | 3.88 | unmet |
| 202,346,377 | 511 | hybrid | 10.87 | 3.50 | passed |
| 202,346,377 | 512 | gpt2 | 14.25 | 2.26 | passed |
| 202,346,377 | 512 | hybrid | 10.87 | 2.11 | passed |

## Held text and decoded speech

The text score is bits per real next byte. The speech score is word error for audio generated from video. The linked JSON also contains transcript conditioning, original and codec speech controls, learned and empirical bigrams, and the earlier-prefix intervention.

| Parameters | Seed | Core | Full text BPB | Empirical bigram BPB | Video-to-speech WER |
| ---: | ---: | --- | ---: | ---: | ---: |
| 116,502,025 | 511 | gpt2 | 2.117 | 3.708 | 100.00% |
| 116,502,025 | 511 | hybrid | 2.002 | 3.708 | 100.00% |
| 202,346,377 | 511 | gpt2 | 2.202 | 3.708 | 100.00% |
| 202,346,377 | 511 | hybrid | 2.011 | 3.708 | 100.00% |
| 202,346,377 | 512 | gpt2 | 2.160 | 3.708 | 100.00% |
| 202,346,377 | 512 | hybrid | 2.012 | 3.708 | 100.00% |

[Frozen readout protocol](shared-state.md) · [Fresh decision protocol](decisions.md) · [Conditional generation protocol](generation.md)
