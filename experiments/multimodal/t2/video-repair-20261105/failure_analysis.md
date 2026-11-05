# T2-V1 post-hoc failure analysis

These diagnostics explain the frozen result. They do not change model selection, primary gates or the fixed listening subset. Positive control harm below means that condition performs worse than the validation-selected additional audio branch.

## crema_test

| Seed / input video | Relative MSE harm vs selected audio [person-cluster95%] | Mean gate |
|---|---:|---:|
| 61 / correct | -9.30% [-11.79, -6.71] | 0.748 |
| 61 / wrong | 11.43% [8.54, 13.95] | 0.536 |
| 61 / shift_plus | 11.04% [8.61, 13.63] | 0.520 |
| 61 / shift_minus | 12.80% [10.71, 15.59] | 0.528 |
| 61 / static | 1.21% [0.92, 1.54] | 0.005 |
| 62 / correct | -9.11% [-11.55, -6.50] | 0.760 |
| 62 / wrong | 12.31% [9.41, 15.01] | 0.531 |
| 62 / shift_plus | 13.68% [11.17, 16.15] | 0.514 |
| 62 / shift_minus | 14.86% [13.50, 16.35] | 0.522 |
| 62 / static | 2.31% [1.75, 2.92] | 0.002 |

All people, including failures (descriptive averages; no within-person independence claim):

| Person | Seed61 video gain vs audio | Seed62 video gain vs audio |
|---|---:|---:|
| 1026 | 9.57% | 9.63% |
| 1029 | 4.90% | 5.78% |
| 1036 | 12.64% | 11.15% |
| 1039 | 10.68% | 11.85% |
| 1046 | 6.63% | 5.09% |
| 1054 | 5.65% | 6.89% |
| 1064 | 6.40% | 5.10% |
| 1081 | 15.53% | 15.21% |

## rav_test

| Seed / input video | Relative MSE harm vs selected audio [person-cluster95%] | Mean gate |
|---|---:|---:|
| 61 / correct | -5.00% [-11.10, 1.68] | 0.771 |
| 61 / wrong | 15.65% [7.73, 22.88] | 0.568 |
| 61 / shift_plus | 14.62% [12.01, 16.99] | 0.461 |
| 61 / shift_minus | 17.70% [9.25, 27.22] | 0.505 |
| 61 / static | 2.41% [0.92, 4.05] | 0.001 |
| 62 / correct | -4.87% [-11.44, 2.65] | 0.776 |
| 62 / wrong | 19.17% [11.15, 26.29] | 0.560 |
| 62 / shift_plus | 22.33% [19.79, 24.47] | 0.426 |
| 62 / shift_minus | 26.46% [18.99, 33.72] | 0.484 |
| 62 / static | 3.27% [0.98, 5.41] | 0.000 |

All people, including failures (descriptive averages; no within-person independence claim):

| Person | Seed61 video gain vs audio | Seed62 video gain vs audio |
|---|---:|---:|
| RAVDESS-09 | 14.15% | 18.93% |
| RAVDESS-10 | -9.22% | -9.20% |
| RAVDESS-11 | 7.26% | 8.30% |
| RAVDESS-12 | 1.95% | 3.19% |
| RAVDESS-21 | -6.90% | -11.00% |
| RAVDESS-22 | 13.74% | 11.32% |
| RAVDESS-23 | 16.56% | 14.75% |
| RAVDESS-24 | 2.45% | 2.71% |

A difference between correct and shifted video establishes sensitivity, not safe rejection of bad input. Compare every shifted/wrong condition against the strong audio baseline: a positive harm means the trained confidence gate has allowed damaging corrections. The exact no-video fallback alone does not resolve this. Calibrating rejection on training/validation mismatches is a future hypothesis; no threshold is tuned on these test actors.

The main report retains all architecture ablations,160/480ms gaps, all-clip diagnostics, original speech, silence, interpolation and the reference-spectrum reconstruction. Those comparisons are needed before attributing limitations solely to fusion or model size.
