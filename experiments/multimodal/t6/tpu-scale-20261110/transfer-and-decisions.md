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

[Frozen readout protocol](shared-state.md) · [Fresh decision protocol](decisions.md) · [Conditional generation protocol](generation.md)
