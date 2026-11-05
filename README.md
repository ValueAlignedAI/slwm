# SLWM

SLWM is a research project for **Signal-Latent World Models**: models that process shared latent signal fields for text/code, audio, and visual/video instead of treating tokens as the only modeling substrate.

The original anchor remains the GPT-2-small-scale `SLWM-124M` comparison target, but the repository now tracks multiple scale profiles. Current configs include near-124M strict/core comparison runs and a 700M+ fit-check profile for later large-run training. Every result must report its parameter accounting mode, data budget, and claim limits.

The project is evidence-first: no capability claim is accepted without registered experiments, baselines, ablations, and failure-case analysis.

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

Recorded active-sensing and persistent-memory experiments are in progress.
Physical robotics/action control remains future work.

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

## Current Sprint Status

| Sprint | Status | Main artifacts | Gate status |
|---|---|---|---|
| R0 — Hypotheses and falsification | Complete as research specification | `docs/research/hypotheses.md`, `docs/research/risks_and_assumptions.md`, `docs/experiments/experiment_registry.md` | Claims are measurable; hypotheses remain untested until registered experiments run. |
| R1 — Literature-to-design mapping | Complete as research/design specification | `docs/research/literature_map.md`, `docs/research/design_decisions.md` | Design choices are traceable to sources and R0 hypotheses; no empirical success claim is made. |
| I0 — Repo skeleton and contracts | Complete by local validation | `docs/model_spec.md`, `docs/data_contract.md`, module stubs, config/registry utilities, tests | Shape tests pass; dummy adapter → latent field → processor → head → policy path works. |
| I1 — Baselines | Implemented locally | `docs/evaluation/baselines.md`, baseline modules, baseline configs, tiny smoke registry entries | GPT-2-style and vanilla multimodal baselines overfit tiny batches and log metrics; no SLWM quality claim is made. |
| T1 — Text baseline and scale | Four124M/125M continuations complete at151M tokens/run | `experiments/text/t1/extended-throughput-20261105/report.md` | Test NLL improves3.64–3.67% at equal tokens, at1.47–1.49x continuation time. Small timer-matched validation gains; severe undertraining and repetitive samples remain. |
| T2 — Speech content and video timing | Ten new runs pass fresh-speaker content, timing rejection and spectral-video gates | `experiments/multimodal/t2/content-sync-20261105/report_v2.md`, `docs/training/content_and_scale.md` | 749 new GRID clips/eight people and unseen complete sentence combinations. Word readout +9.7–12.4pp; video gap-spectrum error −11–12%; ±240ms admission below 1%. Practical speech-recognition gate fails. Restricted grammar; 7.71M deployed video weights. |
| T2-B1 / P1-S1 — Broader shared fields and continuous sensing | Training; final scores pending | `docs/training/operational_world.md` |24M main fields, paired Transformers,100.6h broader speech, long media and direct-sensor policy controls. No new capability gate is yet claimed. |

## Latest completed research — 2026-11-05

The new [151M-token text comparison](experiments/text/t1/extended-throughput-20261105/report.md)
is complete: SLWM test NLL is3.64–3.67% lower than the matched GPT-2-style model,
at1.47–1.49x continuation training time. Equal total timer-budget validation
gains are only0.10–1.04%, with checkpoint-cadence and timer limits reported.
All fixed raw continuations remain available and still show substantial
repetition. These124/125M language weights are separate from the24M audiovisual
fields and remain severely undertrained at roughly1.2 tokens per parameter.

The broader speech/shared-field matrix and continuous sensing study are still
training. Their [protocols and ordinary-file workflow](docs/training/operational_world.md)
cover100.6h LibriSpeech, paired AMI recordings, independent fresh people,
40.96s audio/video and longer text-conditioned media, plus matched Transformer
and direct-sensor decision controls. Final B1/P1 scores have not been read.
Read the [next research questions](docs/research/operational_next_questions_20261105.md)
for the data, representation and physical-action requirements. No new speech,
world-model or motor-control success is claimed while these jobs are in progress.

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
163 passed
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

## Next Training Focus

The next training workstream is **T1-T2 refinement, dataset gathering, T3 implementation, and large-run training**.

Immediate priorities:

1. Refine T1 and T2 configs, registry outputs, and matched baseline/ablation coverage.
2. Gather and pin external text/code and audio/video datasets with licenses, split hashes, leakage checks, and feature-extractor versions.
3. Implement T3 joint SLWM training for mixed-modality latent batches after T1/T2 inputs are reproducible.
4. Run larger 124M and 700M+ training profiles only after dataset cards, baselines, checkpoints, and claim language are registered.

## Scientific Guardrails

- Do not claim reduced hallucination without reporting unsupported-claim rate, contradiction rate, usefulness/accuracy, abstention/no-op rate, and calibration.
- Do not treat diagnostic probes as proof of understanding or grounded representations.
- Do not compare runs with different data, parameter, or compute budgets unless clearly labeled approximate.
- Do not merge sprint scopes unless the sprint explicitly requires it.
- No main-weight continual learning or online inference-time weight updates in the initial phase.

## Branch Workflow

Use focused feature branches and merge through pull requests. `main` is the accepted project trunk for completed, reviewed work; integration branches may still be used for larger multi-sprint staging.

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
