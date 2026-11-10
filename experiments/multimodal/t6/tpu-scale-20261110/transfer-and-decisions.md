# Speaker transfer and sustained decisions

A single fixed ridge head is fitted on eight training examples per command from one modality, then applied to another with no adaptation. The table averages six directional swaps on four development speakers. Balanced chance is 25%. The intervals resample whole speakers.

| Parameters | Seed | Hybrid | Dense GPT-2 | Hybrid gain, pp | Paired interval, pp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 202,346,377 | [511](supplemental/frozen-v5e-larger-511.json) | 32.18% | 24.15% | 8.02 | 6.40 to 8.44 |

[Frozen readout protocol](shared-state.md) · [Fresh decision protocol](decisions.md) · [Conditional generation protocol](generation.md)
