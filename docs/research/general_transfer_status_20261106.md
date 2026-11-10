# Shared-state research status — November6,2026

Status at14:20UTC/Lisbon. This is an intermediate research record; no G1 final observations or scores have been opened.

The current system is task-specialized. That is distinct from demonstrated statistical memorization. Better speech repair within one family would not, by itself, establish a reusable world representation.

## Completed selection-only alignment repair

At512 main-training updates, independently encoded image/sound/text projections were nearly parallel and their covariance participation was near one. Raw feature variance did not ensure diverse unit-vector directions. The aligned projection is an auxiliary branch; predictive/reconstruction heads can read the full field without using this pooled128D projection. A shared field therefore does not establish one sufficient shared semantic state.

A1 revision2 restarted the same512-update SCS/Transformer parents and compared the original objective with an added angular variance/decorrelation penalty. Each arm received512 identical additional COCO/Clotho calls. The original attempt failed during its initial diagnostic, before an optimizer update; the FP32 diagnostic revision preserves that failure.

| Processor | Source | Control hit@1 | Angular hit@1 | Control effective dimensions | Angular effective dimensions |
|---|---|---:|---:|---:|---:|
| SCS | coco | 2.34% | 1.95% | 1.28 | 1.14 |
| SCS | clotho | 5.08% | 3.91% | 1.04 | 1.12 |
| Transformer | coco | 1.56% | 1.95% | 1.35 | 1.95 |
| Transformer | clotho | 2.34% | 4.30% | 1.16 | 1.88 |

Exact uniform-random record retrieval is1.65% for COCO and2.01% for Clotho across these four fixed64-record batches. Repeated record IDs make this higher than the nominal one-positive1/64 baseline; the metadata-only chance audit retains each batch. These are four fixed selection batches/source, not a fresh generalization estimate. Total pilot active time: 11.10 minutes on the3090. The registered gate required better retrieval on both sources/processors, participation>=8 and no more than10% masked-error degradation. **All four source/processor gates fail.** Retain every endpoint and curve; do not promote this penalty or silently change G1. Lowering mean cosine alone did not remove low-rank collapse.

## Active primary studies

- T1-S5: four independent124–125M text models, two paired seeds, each continuing from151M toward500,006,912 tokens. Around300–309M tokens/model at this snapshot. Text data has2.6B prepared training tokens and a separate final domain slice; the independent text models are not shared multimodal weights.
- G1:14 registered runs, with four primary50.7–50.9M processors, smaller controls, shuffled alignment/no-future controls, fixed-modality Transformer and five family specialists. Primary runs are around768–896 schedule positions of12,000; specialist calls match target-family exposure. Early alignment/timing results remain weak; no final capability verdict yet.
- Diverse training covers images/captions, environmental sound/captions,176.1h of speech from443 Libri training speakers, grouped meeting video/audio, wearable sensors and131 robot training episodes. The sensor/robot sources are bounded pilots, not foundation-scale coverage. Data/source versions, training-only normalizers and all role identities were frozen before fitting.
- R1: completed289,860-parameter phonetic renderer and listening artifacts. Its oracle-phone diagnostic is a decoder ceiling, not evidence that the world predicts speech. Freshness is relative to our fitted cores/renderer; the upstream LibriTTS vocoder's exact held-speaker exposure is unverified.

## What will decide the next architecture

Freeze all cores and fit identical small readouts on training roles only. Hold new-source MHealth people and new-task robot action targets out of core pretraining. Compare shared last-state, matched pooled-field and aligned-projection features with raw observations, learned edges and specialists. Report both main seeds and independent-person/environment confidence intervals. Test the same actor with removed/shuffled goals and absent/history-reduced images.

Speech uses32 previously unexposed Libri final people and six new connected meeting components; calibration and final roles stay separate. Broad reuse requires the registered activity/action error improvement and speech noninferiority. Semantic sound→image retrieval uses unpaired category agreement and a specialist text bridge, not matched physical events. The main evaluation includes raw40.96s media failures, decoder ceilings and ongoing observation/proposal replay.

A structural action test confirms that the current future queries see each command independently: changing the preceding seven commands cannot alter the eighth query. The next dynamics architecture needs a causal action-prefix encoder or recurrent latent rollout, with verified physical cadence and actual feedback/goal costs before claiming operational decision quality. Current replay does not actuate a robot or update weights.

If the primary reuse gate fails, retain task-specific gains and redesign a shared predictive bottleneck rather than adding parameters to the same disconnected objectives. Check the matched pooling control before blaming only projection width, and test action-history dependence before claiming planning. Any changed main objective requires a new registered study and new confirmation data; no final-set tuning.

## Delivery and ETA

The bounded pilot is complete. Main500M text and cross-source evaluation target **November7 afternoon, Lisbon time**, conditional on the single-GPU queue and evaluation runtime. Detached training/evaluation workers are active. Byte-verified immutable milestones and fully scored media are archived to the Mac before removing redundant Fedora copies. Trained studies are delivered before reproducible prepared caches; cache copies preserve8GiB headroom and retain skipped sources on Fedora. Focused research commits are pushed on `training/general-transfer-20261106`.
