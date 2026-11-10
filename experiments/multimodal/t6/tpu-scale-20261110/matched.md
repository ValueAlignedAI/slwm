# Verified matched TPU checkpoints

Every row uses the same chip, data order, updates, and parameter budget for both cores. Both full optimizer states and inference weights have been verified on two Fedora disks. Lower action error and higher contact average precision are better.

| TPU | Seed | Parameters | Updates | Hybrid action error, px | Dense action error, px | Hybrid contact AP | Dense contact AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TPU v5 lite | 511 | 202,346,377 | [128](metrics/v5e-511-larger-128.json) | 21.604 | 21.496 | 0.05819 | 0.05806 |
| TPU v5 lite | 511 | 202,346,377 | [512](metrics/v5e-511-larger-512.json) | 20.180 | 21.511 | 0.05663 | 0.05943 |

These are development measurements. An architecture claim requires the two-seed, two-endpoint gate. [Study protocol](protocol.md)
