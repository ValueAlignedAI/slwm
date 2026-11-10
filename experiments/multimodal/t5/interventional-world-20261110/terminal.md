# Terminal anchor comparison

All four inference weights and optimizer states were verified on two independent Fedora checkpoint volumes before opening the held targets.

| Seed | Hybrid endpoint error, px | Dense endpoint error, px | Raw fitted, px | Hybrid contact AP | Dense contact AP | Raw motion AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 511 | 5.47 | 7.27 | 21.07 | 0.0488 | 0.0558 | 0.0674 |
| 512 | 5.34 | 6.67 | 21.07 | 0.0531 | 0.0546 | 0.0674 |

The two-seed, two-endpoint architecture gate was unmet. The hybrid's separate comparison against both fitted raw controls was unmet.

The held simulator measures the consequences of issued actions. The real recordings measure visual evidence of contact timing. [Paired intervals, input ablations and controls](metrics/terminal-test.json) · [Frozen endpoints](metrics/terminal-lock.json) · [Storage verification](metrics/terminal-storage.json)
