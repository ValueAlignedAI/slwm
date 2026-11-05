# T2 real audiovisual pilot — 2026-11-04

## Decision

Do not scale the hybrid on this evidence. At least one preregistered gate failed; keep the checkpoints and benchmark as negative or limited evidence.

Prediction gate (both seeds): **True**. Cross-modal contribution gate: **False**. Architecture gate: **True**.

## Question and provenance

Does a small shared signal processor predict real audio/video futures and benefit from the correctly paired other modality beyond simple baselines?

Protocol: `docs/research/real_av_pilot_20261104.md`. Cycle: `real-av-20261104`. Base commit: `418ebc4`; exact changed-source hashes accompany every checkpoint.

Dataset: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), snapshot `1658cd342dff90010aa843eaeebd53610a08b1dc`. ODbL 1.0 database / DbCL 1.0 contents. 1439 recorded clips; split counts `{'train': 1019, 'validation': 210, 'test': 210}`; person counts `{'train': 34, 'validation': 7, 'test': 7}`. Failures: 1. Audio/video hours per split: `{'train': 0.7279202722222222, 'validation': 0.1433405538888889, 'test': 0.15272416611111111}`.

All trainable adapters, embeddings, processor and heads are counted below. The same frozen ImageNet ResNet18 encoder (11,176,512 deployed parameters) supplies frame features to all arms; PCA and normalization are fitted only on training people. Its external pretraining is not evidence for learning perception from scratch.

Accounting correction: original source cards/registries count the complete downloaded ImageNet checkpoint (11,689,512 parameters), including its discarded 513,000-parameter classifier. `parameter_accounting.json` records the deployed encoder count; original hashed cards remain intact.

## Frozen evaluation

Observe the first 16 of 32 normalized-time steps; predict only the remaining 16. Audio FFT windows cannot cross that boundary. Last-observation persistence and the training mean are measured explicitly. Validation alone selects checkpoint step. Retrieval uses independent encodes of each stream and every test clip as a candidate. Within-person shuffled controls preserve identity while changing the paired event. Confidence intervals resample whole people (2,000 bootstrap samples). Test outcomes never change the frozen matrix or budgets.

## All planned results

| Run | Arm | Seed | Trainable params | Audio MSE ↓ | Video MSE ↓ | A→V R@1 | V→A R@1 | Within-person AUC | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-T2-201 | full | 41 | 4,471,425 | 0.4569 | 1.0764 | 2.4% | 1.0% | 0.543 | 30.3 |
| EXP-T2-202 | vanilla | 41 | 4,433,537 | 0.4968 | 1.1507 | 0.0% | 1.0% | 0.533 | 19.5 |
| EXP-T2-203 | no_spectral | 41 | 4,454,017 | 0.4753 | 1.1048 | 0.5% | 0.5% | 0.537 | 24.7 |
| EXP-T2-204 | no_longconv | 41 | 4,455,041 | 0.4721 | 1.1278 | 1.0% | 0.5% | 0.536 | 26.4 |
| EXP-T2-205 | shuffled | 41 | 4,471,425 | 0.4371 | 0.9900 | 0.0% | 1.4% | 0.491 | 29.3 |
| EXP-T2-206 | audio_only | 41 | 4,471,425 | 0.4438 | 2.4122 | 1.0% | 1.0% | 0.510 | 11.8 |
| EXP-T2-207 | visual_only | 41 | 4,471,425 | 0.6161 | 1.0047 | 0.0% | 0.0% | 0.497 | 12.2 |
| EXP-T2-208 | full | 42 | 4,471,425 | 0.4490 | 1.1113 | 1.0% | 1.4% | 0.525 | 30.0 |
| EXP-T2-209 | vanilla | 42 | 4,433,537 | 0.4861 | 1.1425 | 1.4% | 0.5% | 0.540 | 19.5 |

Full-split retrieval chance: 0.48% across 210 candidates.

## Strong simple controls

- audio: training-mean MSE 0.8458; last-observation persistence MSE 1.7675.
- visual: training-mean MSE 2.2944; last-observation persistence MSE 0.4650.

## Paired modality contribution and uncertainty

Positive relative improvement means the actual paired other stream predicts better than a within-person wrong stream.

| Full run | Target | Improvement | Person-cluster 95% interval | Gate |
|---|---|---:|---|---|
| EXP-T2-201 | audio | 1.10% | [-0.46%, 2.66%] | False |
| EXP-T2-201 | visual | -0.21% | [-0.34%, -0.10%] | False |
| EXP-T2-208 | audio | 2.02% | [-0.51%, 3.98%] | False |
| EXP-T2-208 | visual | 0.09% | [-0.12%, 0.32%] | False |

## Full model versus vanilla, matched seed

- Seed 41: audio error improvement 8.05%; video 6.46%; mean independent R@1 difference +1.19%; architecture gate True.
- Seed 42: audio error improvement 7.63%; video 2.73%; mean independent R@1 difference +0.24%; architecture gate True.

## Concrete failure cases

The table lists the largest video-error excess over persistence for each full-model seed. Every other clip's errors remain in `metrics.json`; these examples are selected to expose failures.

| Run | Held-out clip | Person | Model video MSE | Persistence MSE |
|---|---|---|---:|---:|
| EXP-T2-201 | 1049_IEO_FEA_HI | 1049 | 4.5021 | 0.5666 |
| EXP-T2-208 | 1049_TAI_SAD_XX | 1049 | 4.4499 | 0.4561 |

`forecast_examples.png` shows the first full-model seed's lowest, median and highest joint-error test clips. The audio panels distinguish observed context from predicted future; video curves compare feature-error against persistence. These diagnostic feature plots use the saved checkpoint in FP32; aggregate tables use the registered BF16 evaluation.

## Compute and artifacts

RTX 3090, BF16 autocast with FP32 weights/optimizer, AdamW, 600 steps × 32 clips per run. Total measured train-loop time 3.4 minutes; final diagnostics and preprocessing are additional. Data and optimizer-step budgets are matched; wall time is reported rather than claimed equal. Temporal mixer ablations remove small numbers of parameters; exact differences are in the table.

Each run directory contains `best.pt`, `config.json`, `registry.json`, `curves.json`, `metrics.json`; the best checkpoint includes optimizer and RNG state, validation-selected step, and source/config/dataset hashes. `comparison.csv` and `decision.json` are machine-readable. `learning_curves.png` and `controls.png` show the aggregate behavior.

Inference: `python -m training.t2_real_av_infer --checkpoint <run>/best.pt --clip <source.flv> --output <prediction.npz>`. The outputs are uncommitted internal feature futures; the model has no trained raw-media or language decoder.

## Failures and limits

The corpus contains acted speech, repeated sentence prompts, fixed backgrounds and only seven held-out people. Cross-modal retrieval can exploit demographic or clip-duration cues even when people are disjoint. Within-person controls address some identity shortcuts; they do not establish general grounding. Frozen image features may discard subtle lip motion, and normalized time does not test physical timing. A failed hybrid gate rejects scaling this recipe, not every possible latent predictive model. No evidence here supports text/code capability, general world understanding, hallucination reduction, policy selection, or raw-media generation.

## Next allowed step

Inspect per-person and temporal errors, retain the stronger simple baseline, and register a separate targeted experiment before any joint text/code run or larger model.

## Recorded execution limitation

The discovery single-modality arms used equal sample budgets but different sampled clip orders because their dropout path consumed fewer RNG draws. The full, vanilla, spectral, long-convolution and shuffled comparisons used the same sample/dropout schedule. The residual follow-up fixes this schedule for all arms.
