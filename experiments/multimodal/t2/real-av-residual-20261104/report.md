# T2 real audiovisual residual follow-up — 2026-11-04

## Decision

Do not scale the hybrid on this evidence. At least one preregistered gate failed; keep the checkpoints and benchmark as negative or limited evidence.

Prediction gate (both seeds): **True**. Cross-modal contribution gate: **False**. Architecture gate: **True**.

## Question and provenance

Does a small shared signal processor predict real audio/video futures and benefit from the correctly paired other modality beyond simple baselines?

Protocol: `docs/research/real_av_residual_followup_20261104.md`. Cycle: `real-av-residual-20261104`. Base commit: `418ebc4`; exact changed-source hashes accompany every checkpoint.

Dataset: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), snapshot `1658cd342dff90010aa843eaeebd53610a08b1dc`. ODbL 1.0 database / DbCL 1.0 contents. 1439 recorded clips; split counts `{'train': 1019, 'validation': 210, 'test': 210}`; person counts `{'train': 34, 'validation': 7, 'test': 7}`. Failures: 0. Audio/video hours per split: `{'train': 0.7279202722222222, 'validation': 0.1433405538888889, 'test': 0.14553444277777777}`.

All trainable adapters, embeddings, processor and heads are counted below. The same frozen ImageNet ResNet18 encoder (11,176,512 deployed parameters) supplies frame features to all arms; PCA and normalization are fitted only on training people. Its external pretraining is not evidence for learning perception from scratch.

Accounting correction: original source cards/registries count the complete downloaded ImageNet checkpoint (11,689,512 parameters), including its discarded 513,000-parameter classifier. `parameter_accounting.json` records the deployed encoder count; original hashed cards remain intact.

## Frozen evaluation

Observe the first 16 of 32 normalized-time steps; predict only the remaining 16. Audio FFT windows cannot cross that boundary. Last-observation persistence and the training mean are measured explicitly. Validation alone selects checkpoint step. Retrieval uses independent encodes of each stream and every test clip as a candidate. Within-person shuffled controls preserve identity while changing the paired event. Confidence intervals resample whole people (2,000 bootstrap samples). Test outcomes never change the frozen matrix or budgets.

## All planned results

| Run | Arm | Seed | Trainable params | Audio MSE ↓ | Video MSE ↓ | A→V R@1 | V→A R@1 | Within-person AUC | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T2-211 | full | 41 | 4,471,425 | 0.5136 | 0.4152 | 1.4% | 1.0% | 0.523 | 30.9 |
| EXP-T2-212 | vanilla | 41 | 4,433,537 | 0.5512 | 0.4159 | 0.5% | 1.0% | 0.524 | 20.3 |
| EXP-T2-213 | no_spectral | 41 | 4,454,017 | 0.4931 | 0.4124 | 1.0% | 1.0% | 0.527 | 26.0 |
| EXP-T2-214 | no_longconv | 41 | 4,455,041 | 0.4999 | 0.4134 | 2.4% | 1.0% | 0.518 | 27.6 |
| EXP-T2-215 | shuffled | 41 | 4,471,425 | 0.4757 | 0.3884 | 0.5% | 0.5% | 0.489 | 30.8 |
| EXP-T2-216 | audio_only | 41 | 4,471,425 | 0.5660 | 2.0451 | 0.5% | 0.5% | 0.496 | 12.4 |
| EXP-T2-217 | visual_only | 41 | 4,471,425 | 0.5215 | 0.3871 | 0.5% | 1.0% | 0.499 | 12.4 |
| EXP-T2-218 | full | 42 | 4,471,425 | 0.4791 | 0.4067 | 0.5% | 0.5% | 0.507 | 30.4 |
| EXP-T2-219 | vanilla | 42 | 4,433,537 | 0.5141 | 0.4175 | 1.4% | 1.0% | 0.537 | 19.9 |

Full-split retrieval chance: 0.48% across 210 candidates.

## Strong simple controls

Residual heads start exactly at last-observation persistence. The train/validation arrays and PCA/normalizer are byte-identical to the discovery cycle; the seven test people were absent from every discovery split. The additional visual-dynamics gate (≥5% over persistence in both seeds) is **False**.

- audio: training-mean MSE 0.7446; last-observation persistence MSE 1.6357.
- visual: training-mean MSE 2.0571; last-observation persistence MSE 0.3775.

## Paired modality contribution and uncertainty

Positive relative improvement means the actual paired other stream predicts better than a within-person wrong stream.

| Full run | Target | Improvement | Person-cluster 95% interval | Gate |
|---|---|---:|---|---|
| EXP-T2-211 | audio | 0.12% | [-0.53%, 0.64%] | False |
| EXP-T2-211 | visual | 0.04% | [-0.24%, 0.29%] | False |
| EXP-T2-218 | audio | 0.75% | [-0.23%, 1.60%] | False |
| EXP-T2-218 | visual | 0.20% | [-0.29%, 0.69%] | False |

## Full model versus vanilla, matched seed

- Seed 41: audio error improvement 6.82%; video 0.15%; mean independent R@1 difference +0.48%; architecture gate True.
- Seed 42: audio error improvement 6.80%; video 2.59%; mean independent R@1 difference -0.71%; architecture gate True.

## Concrete failure cases

The table lists the largest video-error excess over persistence for each full-model seed. Every other clip's errors remain in `metrics.json`; these examples are selected to expose failures.

| Run | Held-out clip | Person | Model video MSE | Persistence MSE |
|---|---|---|---:|---:|
| EXP-T2-211 | 1077_IEO_DIS_MD | 1077 | 0.4135 | 0.2680 |
| EXP-T2-218 | 1061_IOM_SAD_XX | 1061 | 0.4997 | 0.3404 |

`forecast_examples.png` shows the first full-model seed's lowest, median and highest joint-error test clips. The audio panels distinguish observed context from predicted future; video curves compare feature-error against persistence. These diagnostic feature plots use the saved checkpoint in FP32; aggregate tables use the registered BF16 evaluation.

## Compute and artifacts

RTX 3090, BF16 autocast with FP32 weights/optimizer, AdamW, 600 steps × 32 clips per run. Total measured train-loop time 3.5 minutes; final diagnostics and preprocessing are additional. Data and optimizer-step budgets are matched; wall time is reported rather than claimed equal. Temporal mixer ablations remove small numbers of parameters; exact differences are in the table.

Each run directory contains `best.pt`, `config.json`, `registry.json`, `curves.json`, `metrics.json`; the best checkpoint includes optimizer and RNG state, validation-selected step, and source/config/dataset hashes. `comparison.csv` and `decision.json` are machine-readable. `learning_curves.png` and `controls.png` show the aggregate behavior.

Inference: `python -m training.t2_real_av_infer --checkpoint <run>/best.pt --clip <source.flv> --output <prediction.npz>`. The outputs are uncommitted internal feature futures; the model has no trained raw-media or language decoder.

## Failures and limits

The corpus contains acted speech, repeated sentence prompts, fixed backgrounds and only seven held-out people. Cross-modal retrieval can exploit demographic or clip-duration cues even when people are disjoint. Within-person controls address some identity shortcuts; they do not establish general grounding. Frozen image features may discard subtle lip motion, and normalized time does not test physical timing. A failed hybrid gate rejects scaling this recipe, not every possible latent predictive model. No evidence here supports text/code capability, general world understanding, hallucination reduction, policy selection, or raw-media generation.

## Next allowed step

Inspect per-person and temporal errors, retain the stronger simple baseline, and register a separate targeted experiment before any joint text/code run or larger model.

## Exploratory stronger controls

Added after discovery and some follow-up outputs were seen, these controls do not change the frozen gate or supply new confirmation. All fits use training clips only; ridge regularization is fixed at 100, without test tuning. The time-profile control predicts the training mean at each normalized time, rather than the global zero mean. Linear models predict future residuals from flattened observed features and retain both streams' own persistence anchors. The audio/visual labels identify residual-regressor inputs; these are not absent-modality tests.

| Control | Audio test MSE | Video test MSE |
|---|---:|---:|
| train_time_profile | 0.5530 | 2.0487 |
| linear_joint | 0.6610 | 0.5068 |
| linear_audio_only | 0.4711 | 0.4522 |
| linear_visual_only | 1.4312 | 0.4573 |

Fitted linear predictors and all control settings are saved under `posthoc_simple_controls/`. Their results narrow the value of gains against a vanilla neural baseline; they do not validate multimodal grounding.
