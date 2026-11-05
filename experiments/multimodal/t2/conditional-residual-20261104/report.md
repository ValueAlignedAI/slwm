# T2-G2 — preserve observed sound at the edge

**Decision: retain the raw observation skip and small residual decoder. The frozen aligned core does not improve this reconstruction task.** This is a useful signal-preserving edge component, not evidence of general generation or world dynamics.

G1 showed new-domain degradation when a normalized shared field replaced observed audio detail. G2 separately preregisters a fixed 25→100 Hz physical-grid skip, plus a learned correction bounded to ±0.25 normalized mel units and initialized at zero. Four 1,600-step runs compare raw versus frozen-core corrections in seeds 41 and 42, with identical 548,352-weight decoders, data, crops and training budget. Frozen-core arms additionally cost 4,741,089 weights. No external generator supplies sound.

The same CREMA 5,061/210/240 split, 3.636 training hours, batch 32, 32-packet windows and 51,200 sampled windows per arm are retained. Validation selects checkpoints every 200 steps. All 64 RAVDESS clips are reused diagnostic data because the architecture responds to G1's external failure; no external clips fit parameters or normalization. The 48-case recognizer subset is unchanged, with only four external people and repeated prompts. Confidence intervals resample people, 2,000 draws.

Only decoder/data/update budgets are matched: cached frozen fields include overlap-averaged context from the full observed clip, potentially beyond a decoder crop; raw corrections see the crop directly. Core arms also pay additional extraction compute and parameters. No claim of matched total compute or identical temporal receptive field is made.

Total decoder-training wall time: 32.7 seconds on RTX 3090; preparation/scoring/rendering excluded. Raw skip has zero neural weights; codec/statistical values are counted separately (249,376 fitted values overall, 128 derived affine buffer values introduce no new fitting).

## test

| Condition | 100 Hz mel MSE | Gain vs interpolation | Temporal MSE | Mean clip WER |
|---|---:|---:|---:|---:|
| Deterministic interpolation | 0.030242 | — | 0.022252 | 73.2% |
| EXP-T2-421: raw + skip, seed 41 | 0.025664 | +15.1% | 0.019106 | 49.0% |
| EXP-T2-422: frozen core + skip, seed 41 | 0.025650 | +15.2% | 0.019320 | 43.2% |
| EXP-T2-423: raw + skip, seed 42 | 0.025632 | +15.2% | 0.019076 | 48.2% |
| EXP-T2-424: frozen core + skip, seed 42 | 0.025662 | +15.1% | 0.019328 | 51.3% |

Original / true 100 Hz codec / old audio head rendered at 100 Hz WER: 37.0% / 40.6% / 103.4%. Matching the renderer does not by itself fix the old head.

| Comparison | Mean per-clip mel gain | Person-cluster 95% interval |
|---|---:|---:|
| EXP-T2-422_vs_EXP-T2-421 | -0.18% | [-0.37, +0.02]% |
| EXP-T2-424_vs_EXP-T2-423 | -0.35% | [-0.59, -0.12]% |

All listening results are evaluator measurements, not SLWM transcription. They use mean per-clip WER, can exceed 100%, and remain a fallible selected diagnostic. In particular, external WER improvement intervals include zero for these four people; the means are not strong confirmation of intelligibility generalization. Mel improvements use all clips, not just the listening subset.

## external

| Condition | 100 Hz mel MSE | Gain vs interpolation | Temporal MSE | Mean clip WER |
|---|---:|---:|---:|---:|
| Deterministic interpolation | 0.062199 | — | 0.045811 | 22.9% |
| EXP-T2-421: raw + skip, seed 41 | 0.051793 | +16.7% | 0.038385 | 8.3% |
| EXP-T2-422: frozen core + skip, seed 41 | 0.068225 | -9.7% | 0.049975 | 19.8% |
| EXP-T2-423: raw + skip, seed 42 | 0.052152 | +16.2% | 0.038660 | 10.4% |
| EXP-T2-424: frozen core + skip, seed 42 | 0.067938 | -9.2% | 0.049970 | 15.6% |

Original / true 100 Hz codec / old audio head rendered at 100 Hz WER: 0.0% / 1.0% / 92.7%. Matching the renderer does not by itself fix the old head.

| Comparison | Mean per-clip mel gain | Person-cluster 95% interval |
|---|---:|---:|
| EXP-T2-422_vs_EXP-T2-421 | -33.45% | [-38.55, -28.34]% |
| EXP-T2-424_vs_EXP-T2-423 | -31.76% | [-36.33, -27.19]% |

All listening results are evaluator measurements, not SLWM transcription. They use mean per-clip WER, can exceed 100%, and remain a fallible selected diagnostic. In particular, external WER improvement intervals include zero for these four people; the means are not strong confirmation of intelligibility generalization. Mel improvements use all clips, not just the listening subset.

## Frozen gates and accounting

Raw-residual pilot gate: **PASS**. Frozen-core-residual pilot gate: **FAIL**. Core-benefit gate: **FAIL**. The preregistered gate requires both seeds to improve test mel error ≥5% and recognizer WER ≥5%, with no >2% relative external error regression. A core claim additionally requires ≥5% benefit over matched raw correction and a positive cluster interval. This is an exploratory repair, not a fresh external confirmation.

| ID | New trainable weights | Frozen core | Selected step | Train seconds | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| EXP-T2-421 | 548,352 | 0 | 1600 | 8.47 | `ff05880fb8153470cce9316e0f9b5949accd1dad85e284baf5843053b29c3427` |
| EXP-T2-422 | 548,352 | 4,741,089 | 1600 | 8.11 | `c92a789182e0bcc7471f13eda06b424f6d044aaa8ded00be30e47859878ddfcf` |
| EXP-T2-423 | 548,352 | 0 | 1600 | 8.04 | `f00907238842e0ca4717e132ca44cd30d14de6aef0dff68f6beb32648c6db2fc` |
| EXP-T2-424 | 548,352 | 4,741,089 | 1600 | 8.06 | `0b67c47980a88ac7fa45542d072a5430ea7ce770e35decf0c1340f0cb615c0ea` |

## Retained artifacts and next step

Every checkpoint includes optimizer/config/RNG state. Source snapshots, configuration and source hashes, complete per-clip mel/temporal metrics, wrong-correction controls and all actual WAV files are retained. G1 artifacts remain unchanged. The standalone `exploration.generation_infer` accepts G1 or G2 checkpoints and real audio/video inputs; G2 only reconstructs observed audio.

The next core experiment should test retaining phonetic/spectral information under varied speakers, channels and corpora while preserving alignment, with explicit raw bypass and frozen-core probes. Do not spend more parameters on this same narrow recipe, infer identity from audio, or treat the sharp copied seed as learned motion. Video→speech and audio-driven visual dynamics remain failed gates.
