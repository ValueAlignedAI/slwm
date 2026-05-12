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

Optional sensor, robotics, action, and persistent-memory experiments are future-phase only.

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
| T1 — Text/code baseline training | Full-stack mechanics implemented; larger corpus work remains | `training/t1_prepare_text_code.py`, `training/t1_torch_text.py`, `configs/t1/*`, `scripts/train/*`, `docs/training/t1_text_code_training.md` | Tiny and limited GPT-2-BPE runs are registered; full guardrail claims require matched GPT-2 and SLWM runs on pinned corpora. |
| T2 — Audio/visual latent training | Mechanics implemented; external data run remains | `data/audio_visual_latents.py`, `training/t2_prepare_latents.py`, `training/t2_train_latents.py`, `configs/t2/*`, `docs/training/preprocessing.md` | Generated-fixture smoke run passes mechanics only; external curated audio/video corpora and matched baselines are required before hypothesis support. |

## Validation

Current local validation evidence:

```bash
pytest
```

Latest result from the T1/T2 foundation branch:

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
