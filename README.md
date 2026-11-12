# SLWM

SLWM tests whether a shared predictive state can carry useful knowledge between text, audio, video, numeric observations and actions. The matched experiments give a recurrent–attention hybrid and a dense GPT-2-style Transformer the same training examples and parameter count within each comparison. Earlier studies varied the emitted state between 128 and 512 coordinates. G12 holds the emitted field at 128 while changing the processor and its width. The original SLWM-124M target and larger fit checks are documented separately. [Architecture](docs/model_spec.md)

## Verified studies

### X1B 1.008B spatiotemporal JEPA, Zero-Lag GatedDelta, and closed-loop MPC

Across three independent seeds (`2001, 2002, 2003`) on Google TPU `v6e-1`, the strict-parameter-matched `SLWM-X1B` (`1,008,454,410` parameters: `24`-slot `64×64` spatiotemporal field, `3:1` Multi-Head Householder `GatedDelta` + $\mathcal{O}(1)$ ring-buffer RoPE hybrid, decoupled instantaneous/differential pose & goal-register heads, 4-step unrolled Euler Residual-EMA-JEPA with `SIGReg`, and Mahalanobis `PolicyCommitGate`) achieved `100.0 ± 0.0%` (`192/192` pooled worlds) on out-of-distribution `T=76` sustained `0.4s` transient-goal `1-px` barrier MPC and `93.2 ± 3.9%` on mid-episode Goal-B overwrite MPC (`t=36..39 > T_train=32`, zero external goal), outperforming matched `DenseTransformer-X1B` (`6.8 ± 8.5%`, `+86.46` points, 99% CI `[+79.17, +92.71]`) and `LegacyScalarHybrid-X1B` (`73.4 ± 13.4%`, `+19.79` points, 99% CI `[+10.42, +29.17]`). On standard `N=288` closed-loop manipulation, `SLWM-X1B` (`71.9 ± 12.7%` overall, `55.6 ± 20.3%` `1-px` barrier) beat `DenseTransformer-X1B` (`47.2 ± 2.6%` overall, `0.7 ± 1.0%` barrier; `+54.86` barrier points) and G8 3-shot random shooting (`23.3 ± 2.0%`, `+48.61` points), while `SLWM-X450M` (`453M`, 3 seeds) reached the fixed-sequence Chinchilla frontier at `89.6 ± 6.1%` overall MPC (`83.3 ± 9.0%` barrier, `96.9 ± 2.6%` OOD goal-switch, `100.0 ± 0.0%` `T=1024` recall). At `32×` OOD `T=1024` (`24,576` slots), `SLWM-X1B` held streaming state constant at `14.91 MB` versus `3,312.54 MB` (`222.2×` larger) on `DenseTransformer-X1B` (`85.4 ± 20.6%` vs `33.3 ± 5.9%` and `2.1 ± 2.9%` scalar hybrid), lifted `0.4s` open-loop transient retention to `90.3 ± 6.4%`, and reached `94.3 ± 1.9%` real GreatestHits + STARSS23 audio-visual `R@1` (`1.0 ± 0.7%` scalar hybrid), `+30.3 ± 0.8%` 4-step forecast skill over persistence, and `0.295` camera-spoofing regret (`0.0%` unsupported claims). Unimodal short-horizon kinematic specialization on upgraded `LegacyScalarHybrid-X1B` (`79.2 ± 1.7%` short `T=38` MPC) and token-matched `1.008B` scaling remain documented boundaries across 120 dual-HDD SHA-256 verified cloud runs (`201.44` Colab units). [Full X1B study](experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111/report.md) · [Run the 1.008B models](experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111/inference/README.md) · [Watch closed-loop replays](experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111/examples/README.md) · [Registered protocol](docs/research/x1b_spatiotemporal_jepa_protocol_20261111.md)

### G14 spoken goals and continuous tabletop control

The frozen policies executed 27,200 branches across 320 held starts, with four real spoken alternatives per physical scene. On two new speakers, spatial SLWM achieved 53.3% selective placement, dense GPT-2 style 52.3%, and geometric feedback 53.9%. The spatial processor did not meet the registered three-seed superiority criterion. Pretrained speech recovered all three fields on 96.3% of fresh-speaker utterances. On a separate 64-world contact panel, moving-target error was 11.07 mm for spatial and 12.04 mm for dense. Replacing the 126 learned coordinates with their training mean reduced fresh placement success by 7.0 points at seed 1411. G14 includes 75 fits, six portable physical models, two original-audio continuous replays and few-shot calibration on real robot recordings. [Full physical and multimodal results](experiments/multimodal/t14/spoken-closed-loop-20261111/report.md) · [Run the models](experiments/multimodal/t14/spoken-closed-loop-20261111/inference/README.md) · [Watch continuous missions](experiments/multimodal/t14/spoken-closed-loop-20261111/examples/README.md)

### G13 grounded physical transfer

On a preregistered cohort of 3,368 real robot trajectories, the 700k-parameter spatial SLWM beat matched dense GPT-2 on four-command prediction by 4.07 mm (99% interval 3.31–4.85), with all five seeds favoring SLWM. On a fourth independent acquisition, a direct 128-coordinate readout combining a raw proposal with the learned field improved raw error from 96.97 to 94.37 mm; dense GPT-2 reached 94.35 mm with the same process. Preserving lexical knowledge also improved natural spoken component retrieval on held devices. G13 includes 182 fits, verified CPU/CUDA speech inference and three uninterrupted real-robot streams. [Full findings](experiments/multimodal/t13/grounded-transfer-20261111/report.md) · [Prospective comparison](experiments/multimodal/t13/grounded-transfer-20261111/prospective-findings.md) · [Run the models](experiments/multimodal/t13/grounded-transfer-20261111/inference/README.md)

### G12 spoken action sequences and directly visible device changes

On 373 fresh decisions from 48 unseen-family recordings, zero of 24 primary comparisons reliably benefited from correct speech over both an earlier same-scene directive and silence. Blind review yielded 77 directly visible device outcomes. Early joint training improved ordered accuracy in 28 of 48 paired comparisons. Widening the oscillatory processor improved printer/camera accuracy by 5.4 and 2.8 points in two seeds. Coffee/computer did not reproduce this gain. The release includes 112 matched fits, 24 runnable models and four chronological replays totaling 603 seconds. [G12 findings](experiments/multimodal/t12/instruction-sequences-20261111/report.md) · [Run inference](experiments/multimodal/t12/instruction-sequences-20261111/inference/README.md) · [Listen to the speech interventions](experiments/multimodal/t12/instruction-sequences-20261111/examples/README.md)

### G11 spoken instructions and physical effects on unseen devices

Across 838 prospective decisions in 96 new coffee, computer, printer and camera recordings, the correct recorded instruction failed the registered next-verb and changed-area gates in all 18 primary architecture–seed–fold cells. A swapped cue frequently changed the prediction. Of 259 near-depth measurements, only 156 pre-action centers were clearly on a device or component. The completed study includes dense GPT-2, gated, persistence and raw controls, a matched ~117M scale check, development-selected diagnostics and six runnable adapters with audible counterfactuals. [Full G11 study](experiments/multimodal/t11/instructions-physical-effects-20261111/report.md) · [Run the models](experiments/multimodal/t11/instructions-physical-effects-20261111/inference/README.md) · [Recorded examples](experiments/multimodal/t11/instructions-physical-effects-20261111/examples/README.md)

### G10 context training and public task recordings

On matched variable histories, SLWM led dense GPT-2 on uninterrupted simulated control in one of three seeds. A simpler gated recurrent control reached about 94% occupancy in all seeds and handled two actuator changes best. With 416 public HoloAssist recordings, paired SLWM audio-to-video retrieval reached 20.7 / 19.1 / 20.1% on familiar tasks and 12.3 / 14.1 / 12.3% on unseen coffee/computer tasks, against 12.5% chance. Brief goal instructions exposed a separate memory failure that a causal last-command cache repaired in the simulator. [Verified G10 study](experiments/multimodal/t10/context-alignment-20261111/report.md) · [Runnable models](experiments/multimodal/t10/context-alignment-20261111/inference/README.md) · [60-second examples](experiments/multimodal/t10/context-alignment-20261111/examples/README.md)

### G9 concurrent sensing and sustained decisions

With 4,096 updates, a 13M SLWM and a 128-coordinate shared output spent 94.5 / 94.8% of the time at the requested goal across two seeds in 51.2-second held missions. The matched dense GPT-2 reached 43.8 / 41.2%. Larger trunks and 512-coordinate outputs gave no consistent gain. Resetting both large wide-field models at their 32-step training length greatly narrowed the dense gap. On real room recordings and unseen robot contacts, the architecture ranking depended on the target and neither large wide model consistently beat the simple force reference. [Full matched study](experiments/multimodal/t9/concurrent-memory-20261110/report.md) · [Two complete 5.5 MB models](experiments/multimodal/t9/concurrent-memory-20261110/inference/README.md) · [51-second decisions and 40-second generation](experiments/multimodal/t9/concurrent-memory-20261110/examples/README.md)

### G8 acoustic coding and action-transfer study

A 652 kB trained acoustic codebook now has a portable encoder and decoder. At the same 2,250-bit/s payload, allocating bits by training distortion reduced word error from 93.5% to 46.7% on 33 new audiobook speakers. At 9,000 bits/s it reached 22.5%, close to the 22.4% original-audio result. A separate whole-utterance replication on 40 further speakers met the registered ±3-point WER margin versus original audio at 9 kbit/s for both Whisper and wav2vec2. The text- and video-conditioned predictors still failed the word-content gate. [Fresh-speaker results and listening examples](experiments/multimodal/t8/factorial-transfer-20261110/speech.md) · [Weights and usage](experiments/multimodal/t8/factorial-transfer-20261110/inference/README.md)

The completed G8 four-cell study missed its prespecified cross-modal decision criterion (1/4 required hybrid task–seed cases). At fixed simulator freshness, the average non-simulator effect was 0.013 / 0.023 pixels of realized goal regret on new Push-T starts and 0.103 / -0.126 on unseen puck-and-barrier dynamics. Lower regret is better. The study includes matched dense Transformers, 512-start articulated-arm readouts and 38.4-second closed-loop action traces. It used 8.90 additional project Colab units. [Factorial results](experiments/multimodal/t8/factorial-transfer-20261110/report.md) · [Executed decision replays](experiments/multimodal/t8/factorial-transfer-20261110/metrics/rollout-examples.json)

### G7 shared state, speech and control

On 14 additional speakers, a timing head picked the correct camera among four alternatives in 43–44% of trials with the hybrid and 40–43% with dense GPT-2. The same head applied to raw sensor features reached 65–69%. All alternatives used identical clocks. Moving encoded positions along with images had inflated an earlier alignment score. Supplying the true acoustic units still yielded 85% word error for acted speech and 99% for audiobooks. [Clock-controlled confirmation](experiments/multimodal/t7/continuation-alignment-control-20261110/physical-time.md) · [Speech synthesis controls](experiments/multimodal/t7/continuation-alignment-control-20261110/alignment-and-control.md)

On 512 new Reacher starts, both architectures retained the initial goal cue after it disappeared. Raw observations with the common physical prior gave the lowest supervised action regret. None of four RL policies reliably improved final goal success. Exact two-step planning reached 45–48% in two diagnostic cohorts, and the learned transition lost that advantage. [Decisions and independent transfer](experiments/multimodal/t7/continuation-alignment-control-20261110/transfer-and-planning.md)

Sixteen exact continuations completed at 16,384 updates on two named Colab v5e assignments, using 24.48 project compute units. On 512 fresh Push-T starts, the 116.50M hybrid reached 4.38 and 4.41 pixels with independent examples versus 4.77 and 4.87 on replay. The matched dense fresh-data models reached 5.48 and 5.89 pixels. The release includes the larger model, equal-TPU-time analyses, held-room results and 43.69-second cross-modal examples. [Matched scaling results](experiments/multimodal/t7/continuation-alignment-control-20261110/report.md) · [Inference previews](experiments/multimodal/t7/continuation-alignment-control-20261110/examples/README.md)

### G6 multimodal scaling

Eight models completed 8,192 updates at 116.50M and 202.35M parameters, with two training seeds per architecture and size. On 512 new Push-T starts, the 116.50M hybrid predicted joint action endpoints to 5.89 and 5.84 pixels. The 202.35M hybrid reached 6.12 and 6.01 pixels, compared with 8.14 and 7.41 for its matched dense control. Video helped with action-conditioned prediction in this simulator. Held real-event timing and generated speech remained weak. [Results and learning curves](experiments/multimodal/t6/tpu-scale-20261110/report.md) · [Transfer and decisions](experiments/multimodal/t6/tpu-scale-20261110/transfer-and-decisions.md)

The release includes verified inference and optimizer hashes, long cross-modal generations, source-substitution controls and the measured compute cost. The broader claim depends on replication across different environments and on intelligible generated content. [Artifact inventory](experiments/multimodal/t6/tpu-scale-20261110/artifact-inventory.json) · [Publication assessment](experiments/multimodal/t6/tpu-scale-20261110/publication-assessment.md)

## Core Idea

The planned architecture keeps modality-specific processing at the edges and a shared latent world field in the middle:

```text
modality encoders / codecs
        ↓
shared latent signal field
        ↓
signal world processor
        ↓
proposal / prediction / uncertainty heads
        ↓
policy / commit gate
        ↓
selected output decoders or no-op/internal-only behavior
```

Canonical tensor contract:

```python
Z: FloatTensor[B, T, D]
mask: BoolTensor[B, T]
```

## Required Modalities

- `text_code` — English text and code at the edge codec/decoder.
- `audio` — speech and general audio latents/features.
- `visual_video` — image/video patch, tube, or latent streams.
- `noop` — valid committed policy behavior.

Recorded active-sensing and persistent-memory experiments include closed-loop simulation. Physical-robot deployment remains an open research question.

## Documentation

Root Markdown is intentionally minimal:

- `AGENTS.md` — operational rules, required reading, sprint discipline, and branch workflow.
- `README.md` — project overview and current status.

Project documentation is organized under `docs/`:

- `docs/README.md` — documentation index.
- `docs/research/` — research plan, hypotheses, risks, literature map, and design decisions.
- `docs/architecture/` — architecture, inference, and policy/commitment design.
- `docs/training/` — preprocessing and sprint training notes.
- `docs/evaluation/` — baseline and evaluation-facing design notes.
- `docs/experiments/` — experiment registry schema and completed artifact references.
- `docs/exploration/` — diagnostic probe and exploration protocol.
- `docs/process/` — sprint playbook and agent prompts.

## Study history

| Sprint | Status | Main artifacts | Gate status |
|---|---|---|---|
| R0 — Hypotheses and falsification | Complete as research specification | `docs/research/hypotheses.md`, `docs/research/risks_and_assumptions.md`, `docs/experiments/experiment_registry.md` | Claims are measurable; hypotheses remain untested until registered experiments run. |
| R1 — Literature-to-design mapping | Complete as research/design specification | `docs/research/literature_map.md`, `docs/research/design_decisions.md` | Design choices are traceable to sources and R0 hypotheses; no empirical success claim is made. |
| I0 — Repo skeleton and contracts | Complete by local validation | `docs/model_spec.md`, `docs/data_contract.md`, module stubs, config/registry utilities, tests | Shape tests pass; dummy adapter → latent field → processor → head → policy path works. |
| I1 — Baselines | Implemented locally | `docs/evaluation/baselines.md`, baseline modules, baseline configs, tiny smoke registry entries | GPT-2-style and vanilla multimodal baselines overfit tiny batches and log metrics; no SLWM quality claim is made. |
| T1 — Text baseline and scale | Four independent text models reached 500,006,912 tokens each | [Matched text continuation](experiments/text/t1/500m-20261106/REPORT.md) | Across two seeds the signal models reached fresh test NLL 3.789 and 3.792, versus 3.923 and 3.911 for dense GPT-2. Active continuation took 3.68–3.73 hours versus 2.49–2.53. |
| T2 — Speech content and video timing | Ten new runs pass fresh-speaker content, timing rejection and spectral-video gates | `experiments/multimodal/t2/content-sync-20261105/report_v2.md`, `docs/training/content_and_scale.md` | 749 new GRID clips/eight people and unseen complete sentence combinations. Word readout +9.7–12.4pp; video gap-spectrum error −11–12%; ±240ms admission below 1%. Practical speech-recognition gate fails. Restricted grammar; 7.71M deployed video weights. |
| T2-B1 / P1-S1 — Broader shared fields and continuous sensing | Complete; all three capability gates fail | `docs/research/operational_findings_20261106.md` |24M fields and paired Transformers, fresh-speaker tests,248 long outputs and eight sensing actors. Technical audits pass; speech repairs worsen WER and decision benefit does not replicate. |
| T3 / T4 — Predictive state and native TPU comparisons | Matched transfer and modality probes completed | [Cross-modal report](experiments/multimodal/t4/crossmodal-state-tpu-20261109/report.md) | Source substitution and erasure exposed dependence on the task and modality. |
| G3 — Dense and sparse shared-state transfer | Primary, six-layer and structured 2-of-4 comparisons completed at 8,000 updates | [Unseen-task comparison](experiments/multimodal/t4/joint-transfer-20261109/summary-8000-test.md) · [Structured confirmation](experiments/multimodal/t4/joint-transfer-20261109/structured/structured-8000-test.json) | Spoken-digit transfer favored the dense signal model on both seeds. Both held-episode Push-T comparisons missed the physical action-transfer gate. |
| T5 / T6 — Action-conditioned multimodal scaling | Eight two-size controls completed | [G6 results](experiments/multimodal/t6/tpu-scale-20261110/report.md) | Visual action dynamics transferred on Push-T. Event timing and speech generation failed broader gates. |

## Earlier speech and text results — 2026-11-06

The new [151M-token text comparison](experiments/text/t1/extended-throughput-20261105/report.md)
is complete: SLWM test NLL is3.64–3.67% lower than the matched GPT-2-style model,
at1.47–1.49x continuation training time. Equal total timer-budget validation
gains are only0.10–1.04%, with checkpoint-cadence and timer limits reported.
All fixed raw continuations remain available and still show substantial
repetition. These124/125M language weights are separate from the24M audiovisual
fields and remain severely undertrained at roughly1.2 tokens per parameter.

The [completed operational study](docs/research/operational_findings_20261106.md)
finds real GPU headroom:3.6x more world parameters cost only23–28% more measured
training/validation time. Fresh-speaker speech repair and replicated sensing
utility nevertheless fail their gates. All248 long outputs and failures are in
the [media gallery](experiments/multimodal/t2/broad-stream-20261105/watch.html).
The [ordinary-file workflow](docs/training/operational_world.md) provides exact
world/actor exports and a portable recorded-input continuous replay. Read the
[scale budget](docs/research/scaling_budget_review_20261106.md) and
[data access review](docs/research/data_scaling_access_review_20261106.md) before
committing to a larger joint model. Physical action learning remains a separate,
explicitly typed [next stage](docs/research/action_conditioned_next_stage_20261105.md).

Read the [current findings](docs/research/findings.md),
[corrected speech report](experiments/multimodal/t2/content-sync-20261105/report_v2.md),
[32 fixed listening cases](experiments/multimodal/t2/content-sync-20261105/listen.html)
and [early text-scale report](experiments/text/t1/scale-20261105/report.md).
[Run and artifact instructions](docs/training/content_and_scale.md) cover portable
speech/text checkpoints and both-seed full optimizer states. Code, compact evidence
and reports are versioned; large model/media files remain in the local and Fedora
experiment directories, with SHA256 delivery manifests.

The speech and large text studies use separate trained weights. These results
support further controlled research; they do not establish a unified multimodal
124M world model or dependable speech repair. The original word-scoring result is
retained beside revision 2, which excludes silence and short-pause labels without
changing weights, calibration, spectra or waveforms.

## Validation

Current local validation evidence:

```bash
pytest
```

Latest complete suite on Fedora (2026-11-05):

```text
166 passed
```

The previous completed content/timing study passed133 tests.

Earlier result from the T1/T2 foundation branch:

```text
74 passed
```

Targeted T2 validation:

```bash
pytest tests/test_t2_audio_visual_latents.py tests/test_t2_training_runner.py
```

Result:

```text
9 passed
```

This validates mechanics and safety checks only. It does not establish converged GPT-2-quality training, multimodal grounding, hallucination reduction, policy behavior, or real audio/video model quality.

## Training Workflows

T1 text/code training docs:

- `docs/training/t1_text_code_training.md`
- `training/README.md`

T2 audio/visual latent training docs:

- `docs/training/preprocessing.md`
- `training/README.md`

Useful T2 commands:

```bash
python -m training.t2_prepare_latents --config configs/t2/prepare_audio_visual_generated_smoke.json
python -m training.t2_train_latents --config configs/t2/slwm_t2_tiny_smoke.json
python -m training.t2_train_latents --config configs/t2/slwm_124m_audio_visual_pilot.json --describe-only
python -m training.t2_train_latents --config configs/t2/slwm_700m_audio_visual_24gb_fitcheck.json --describe-only
```

The generated T2 fixture is for pipeline validation only and must not be used as evidence of real audio/video quality.

## Current research questions

- When present state is ambiguous, can synchronized evidence reveal a physical condition that changes the optimal action?
- Can a fixed-width shared state retain the short-timescale information needed for genuine audiovisual synchrony and intelligible speech?
- Can learned dynamics improve two-step terminal goal success with raw state first, then with decoded shared state?

Removing the camera while keeping proprioception had no reliable effect on the fresh 116.50M hybrid’s held simulator error. Removing the proposed action raised error above 25 pixels. G8 independently varied fresh simulator starts and fresh non-simulator examples with a matched dense Transformer. [Sensor and plan controls](experiments/multimodal/t7/continuation-alignment-control-20261110/metrics/input-controls.json) [Experimental decision and gates](experiments/multimodal/t7/continuation-alignment-control-20261110/next-steps.md)

## Scientific Guardrails

- Do not claim reduced hallucination without reporting unsupported-claim rate, contradiction rate, usefulness/accuracy, abstention/no-op rate, and calibration.
- Do not treat diagnostic probes as proof of understanding or grounded representations.
- Do not compare runs with different data, parameter, or compute budgets unless clearly labeled approximate.
- Do not merge sprint scopes unless the sprint explicitly requires it.
- No main-weight continual learning or online inference-time weight updates in the initial phase.

## Branch workflow

Active trainings use focused research branches. Completed studies enter `main` through pull requests after their controls, artifact integrity, tests and README summary have been checked.

## License

This project is licensed under the MIT License. See `LICENSE` for copyright holders and terms.

## Earlier real audiovisual pilots — 2026-11-04

[Findings and research decision](docs/research/findings.md) summarize 18 real-data
GPU runs, including a fresh-person residual follow-up. Saved checkpoints, full
metrics, source snapshots and diagnostic plots live under
`experiments/multimodal/t2/real-av-20261104/` and
`experiments/multimodal/t2/real-av-residual-20261104/`. The current recipe does
not pass the multimodal or visual-dynamics gates; larger or joint text/code
runs are not justified by these results.

The final validation suite passes **84 tests**. See
[the training workflow](training/README.md#real-recorded-audiovisual-pilots)
for reproduction and checkpoint inference.

## Motion-aware architecture and model outputs

The new physical-time recipe learns real audio/video timing alignment without a
pretrained neural encoder. Compare 4.74M and 15.23M models, matched baselines,
ablations and input-corruption controls in the
[research report](experiments/multimodal/t2/motion-alignment-20261104/report.md).
Inspect the [actual conditional output gallery](experiments/multimodal/t2/motion-alignment-20261104/exploration/gallery.md)
for audio→mouth, video→audio, codec/reconstruction/null controls, and bounded
recognition probes. The current source passes **96 tests**.

[Standalone inference](docs/training/motion_alignment.md) accepts an audio or
video file with a saved checkpoint. Outputs are diagnostic predictions; larger
parameter count has not resolved generation quality.
