# Verified matched TPU checkpoints

Every row uses the same chip, data order, updates, and parameter budget for both cores. Both full optimizer states and inference weights have been verified on two Fedora disks. Lower action error and higher contact average precision are better.

| TPU | Seed | Parameters | Updates | Hybrid action error, px | Dense action error, px | Hybrid contact AP | Dense contact AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TPU v5 lite | 511 | 202,346,377 | [128](metrics/v5e-511-larger-128.json) | 21.604 | 21.496 | 0.05819 | 0.05806 |
| TPU v5 lite | 511 | 116,502,025 | [512](metrics/v5e-511-anchor-512.json) | 21.566 | 21.675 | 0.05707 | 0.05799 |
| TPU v5 lite | 511 | 202,346,377 | [512](metrics/v5e-511-larger-512.json) | 20.180 | 21.511 | 0.05663 | 0.05943 |
| TPU v5 lite | 511 | 116,502,025 | [2,048](metrics/v5e-511-anchor-2048.json) | 10.659 | 16.435 | 0.05106 | 0.05915 |
| TPU v5 lite | 511 | 202,346,377 | [2,048](metrics/v5e-511-larger-2048.json) | 12.037 | 16.819 | 0.05477 | 0.05897 |
| TPU v5 lite | 511 | 202,346,377 | [8,192](metrics/v5e-511-larger-8192.json) | 6.522 | 8.453 | 0.04203 | 0.04590 |
| TPU v5 lite | 512 | 202,346,377 | [128](metrics/v5e-512-larger-128.json) | 21.550 | 21.519 | 0.05814 | 0.05805 |
| TPU v5 lite | 512 | 116,502,025 | [512](metrics/v5e-512-anchor-512.json) | 21.456 | 21.744 | 0.05808 | 0.05828 |
| TPU v5 lite | 512 | 202,346,377 | [512](metrics/v5e-512-larger-512.json) | 21.118 | 21.415 | 0.05746 | 0.05805 |
| TPU v5 lite | 512 | 202,346,377 | [2,048](metrics/v5e-512-larger-2048.json) | 12.316 | 14.720 | 0.05766 | 0.06149 |

## Parameter effect at the same exposure

Positive values favor the 202.35M model in both columns. The linked records include paired 95% intervals.

| TPU | Seed | Core | Updates | Action error reduction, px | Contact AP gain |
| --- | ---: | --- | ---: | ---: | ---: |
| TPU v5 lite | 511 | gpt2 | [512](metrics/v5e-511-size-gpt2-512.json) | 0.164 | 0.00145 |
| TPU v5 lite | 511 | hybrid | [512](metrics/v5e-511-size-hybrid-512.json) | 1.386 | -0.00044 |
| TPU v5 lite | 511 | gpt2 | [2,048](metrics/v5e-511-size-gpt2-2048.json) | -0.384 | -0.00018 |
| TPU v5 lite | 511 | hybrid | [2,048](metrics/v5e-511-size-hybrid-2048.json) | -1.378 | 0.00370 |
| TPU v5 lite | 512 | gpt2 | [512](metrics/v5e-512-size-gpt2-512.json) | 0.328 | -0.00023 |
| TPU v5 lite | 512 | hybrid | [512](metrics/v5e-512-size-hybrid-512.json) | 0.339 | -0.00062 |
| TPU v5 lite | 512 | hybrid | [2,048](metrics/v5e-512-size-hybrid-2048.json) | -1.339 | -0.00055 |

These are development measurements. An architecture claim requires the two-seed, two-endpoint gate. [Study protocol](protocol.md)
