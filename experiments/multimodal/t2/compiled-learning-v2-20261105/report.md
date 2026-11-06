# Compiled stream learning pilot

Single-seed,400-update validation-only runtime study. Eager and compiled arms have identical400-update LR horizons, initialization, data and selection. The original2000-step B1 matrix is unchanged. No fresh test speaker is read.

| Variant | Eager loss | Compiled loss | Relative change | Cold-inclusive speedup | Pilot check |
|---|---:|---:|---:|---:|---|
| slwm | 1.431002 | 1.432473 | +0.103% | 1.174x | PASS |
| transformer | 1.430794 | 1.428834 | -0.137% | 1.290x | PASS |

The pilot gate is no more than1% loss degradation at the fixed400 endpoint. It is exploratory evidence about short learning, not fresh-person equivalence, convergence or additional primary capability evidence. Full best/resume/final states and every component curve remain available. First-gradient and steady-step measurements, including the failed original SLWM local loss gate, remain in the separate compilation benchmark. Future adoption needs a registered longer comparison and cold-cost accounting.
