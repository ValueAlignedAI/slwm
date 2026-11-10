# SLWM

SLWM tests whether a shared predictive state can carry useful knowledge between text, audio, video, numeric observations and actions. The current comparison gives a recurrent–attention hybrid and a dense GPT-2-style Transformer the same data, parameter count and 128-dimensional shared state. The original SLWM-124M target and larger fit checks are documented separately. [Architecture](docs/model_spec.md)

## Verified studies

### G7 shared state, speech and control

On 14 additional speakers, a timing head picked the correct camera among four alternatives in 43–44% of trials with the hybrid and 40–43% with dense GPT-2. The same head applied to raw sensor features reached 65–69%. All alternatives used identical clocks. Moving encoded positions along with images had inflated an earlier alignment score. Supplying the true acoustic units still yielded 85% word error for acted speech and 99% for audiobooks. [Clock-controlled confirmation](experiments/multimodal/t7/continuation-alignment-control-20261110/physical-time.md) · [Speech synthesis controls](experiments/multimodal/t7/continuation-alignment-control-20261110/alignment-and-control.md)

On 512 new Reacher starts, both architectures retained the initial goal cue after it disappeared. Raw observations with the common physical prior gave the lowest supervised action regret. None of four RL policies reliably improved final goal success. Exact two-step planning reached 45–48% in two diagnostic cohorts, and the learned transition lost that advantage. [Decisions and independent transfer](experiments/multimodal/t7/continuation-alignment-control-20261110/transfer-and-planning.md)

The registered scaling comparison continues both sizes and both architectures from their exact G6 optimizer states. It separates additional exposure, independent examples and elapsed TPU time. The two terminal architectures also generated 43.69-second clips from audio, video and text. [Study design](experiments/multimodal/t7/continuation-alignment-control-20261110/protocol.md) · [Inference previews](experiments/multimodal/t7/continuation-alignment-control-20261110/examples/README.md)

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

- With simulator training trajectories held fixed, do additional speech, text and audiovisual observations improve decisions on a new physical task?
- Can a fixed-width shared state retain the short-timescale information needed for genuine audiovisual synchrony and intelligible speech?
- Can learned dynamics improve two-step terminal goal success with raw state first, then with decoded shared state?

Removing the camera while keeping proprioception had no reliable effect on the fresh 116.50M hybrid’s held simulator error. Removing the proposed action raised error above 25 pixels. The next comparison crosses fresh simulator starts with fresh examples from the other modalities and retains a matched dense Transformer. [Sensor and plan controls](experiments/multimodal/t7/continuation-alignment-control-20261110/metrics/input-controls.json) [Experimental decision and gates](experiments/multimodal/t7/continuation-alignment-control-20261110/next-steps.md)

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
