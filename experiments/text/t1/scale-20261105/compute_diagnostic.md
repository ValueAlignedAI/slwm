# Text compute diagnostic

The primary comparison uses equal tokens. This supplementary analysis linearly interpolates the already fixed validation curves at equal process wall time. It includes validation/IO and can be affected by system load; it is not a matched-FLOP experiment, model-selection rule or directly measured intermediate checkpoint.

| Seed | Wall seconds | GPT-2 NLL / tokens | SLWM NLL / tokens | Interpolated NLL reduction |
|---|---:|---:|---:|---:|
| 81 | 474.8 | 5.6486 / 16.78M | 5.6024 / 11.73M | 0.82% |
| 82 | 476.8 | 5.6449 / 16.78M | 5.6141 / 11.88M | 0.55% |

SLWM spends more time per token. Early per-token improvements alone do not establish a compute-efficiency advantage or predict converged behavior. More training and component ablations are required before committing a long scale run.
