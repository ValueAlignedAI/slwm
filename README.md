# SLWM-124M

SLWM-124M is a research project for a GPT-2-scale **Signal-Latent World Model**. The core question is whether a model that processes shared latent signal fields can learn reusable multimodal representations for text/code, audio, and visual/video signals without treating tokens as the only modeling substrate.

The project is evidence-first: no capability claim is accepted without registered experiments, baselines, ablations, and failure-case analysis.

## Core idea

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

The initial GPT-2-scale target is approximately 124M parameters, with strict and core-only parameter accounting tracked separately in future experiments.

## Required modalities

Current required modalities are:

- `text_code` — English text and code at the edge codec/decoder.
- `audio` — speech and general audio latent/features.
- `visual_video` — image/video patch, tube, or latent streams.
- `noop` — valid committed policy behavior.

Optional sensor, robotics, action, and persistent-memory experiments are future-phase only.

## Current sprint status

| Sprint | Status | Main artifacts | Gate status |
|---|---|---|---|
| R0 — Hypotheses and falsification | Complete as research specification | `hypotheses.md`, `risks_and_assumptions.md`, `experiment_registry.md` | Claims are measurable; all hypotheses remain untested until registered experiments run. |
| R1 — Literature-to-design mapping | Complete as research/design specification | `literature_map.md`, `design_decisions.md` | Design choices are traceable to sources and R0 hypotheses; no empirical success claim is made. |
| I0 — Repo skeleton and contracts | Complete by local validation | `docs/model_spec.md`, `docs/data_contract.md`, module stubs, config/registry utilities, tests | Shape tests pass; dummy adapter → latent field → processor → head → policy path works. |
| I1 — Baselines | Implemented locally | `baselines.md`, baseline modules, baseline configs, tiny smoke registry entries | GPT-2-style and vanilla multimodal baselines overfit tiny batches and log metrics; no SLWM quality claim is made. |

## I0 implementation scope

Sprint I0 intentionally implements only repository structure and shape contracts. It does **not** implement real training, real datasets, learned codecs, GPT-2 baselines, spectral/SSM/long-conv blocks, or model-quality evaluation.

Implemented skeleton areas:

```text
configs/
data/
docs/
evals/
exploration/
models/
  adapters/
  baselines/
  heads/
  policy/
  processor/
training/
tests/
utils/
```

Key I0 modules:

- `TextSignalAdapter`, `AudioSignalAdapter`, `VisualSignalAdapter`
- `LatentSignalField`
- `SignalWorldProcessor`
- `LatentPredictionHead`, `ReconstructionHead`, `UncertaintyHead`
- `TextDecoderHead`, `AudioDecoderHead`, `VisualDecoderHead`, `NoOpHead`
- `PolicyCommitGate`
- dependency-free `TensorSpec` shape carriers
- JSON config loader and experiment registry writer

## Validation

Current local I0 validation evidence:

```bash
python -m pytest
```

Result:

```text
26 passed
```

Coverage validation over the implemented I0 packages:

```bash
python -m pytest --cov=data --cov=models --cov=utils --cov-report=term-missing
```

Result:

```text
26 passed
TOTAL 413 stmts, 0 miss, 100% coverage
```

## Scientific guardrails

- Do not claim reduced hallucination without reporting unsupported-claim rate, contradiction rate, usefulness/accuracy, abstention/no-op rate, and calibration.
- Do not treat diagnostic probes as proof of understanding or grounded representations.
- Do not compare runs with different data, parameter, or compute budgets unless clearly labeled approximate.
- Do not merge sprint scopes: baselines, real SLWM architecture, training, evaluation, and exploration have separate gates.
- No main-weight continual learning or online inference-time weight updates in the initial phase.

## Sprint I1 baseline scope

Sprint I1 implements required baselines before SLWM novelty:

1. GPT-2-small-style decoder-only text/code baseline.
2. Vanilla multimodal Transformer baseline.
3. Null/random baselines for probes.

See `baselines.md` for exact parameter counts, configs, context lengths, tokenizer/codec choices, smoke metrics, and registry paths. Minimal Perceiver-style latent bottleneck remains pending/future because the accepted I1 gate only requires one text baseline and one multimodal/latent baseline to tiny-train and log metrics.

## Next allowed implementation stage

The next implementation sprint after I1 is **I2 — SLWM core**, but only after I1 validation artifacts are accepted. I2 scope is adapters and processor core; it must still avoid policy/exploration claims.

## Branch workflow

The project branch policy is:

```text
research
implementation
training
evaluation
exploration
integration
```

Use one branch per sprint and merge completed sprint work through a PR into `integration`. Promote from `integration` to `main` only after the integrated state is accepted.

## License

This project is licensed under the MIT License. See `LICENSE` for copyright holders and terms.
