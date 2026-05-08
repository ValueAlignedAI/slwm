# SLWM-124M Training Runners

This directory contains dependency-light training runners for completed/pilot
sprints. Each runner is config-driven and writes registry artifacts before any
result can be used as evidence.

---

# Sprint T0 Training — Synthetic Signal Pretraining

Sprint T0 is limited to controlled synthetic signal tasks. It does **not** use text/code, audio, visual/video, or external multimodal datasets.

## Scope

Implemented tasks:

- sine mixtures,
- chirp extrapolation,
- phase shift detection,
- noisy periodic denoising,
- missing-span reconstruction,
- long-horizon extrapolation.

Implemented comparisons:

- `slwm`: direct latent SLWM signal predictor with spectral mixer enabled,
- `slwm_no_spectral`: same predictor with spectral mixer disabled,
- `vanilla_transformer`: continuous latent Transformer regression baseline,
- `random_signal`: random continuous signal control,
- `noop_signal`: copy-input no-op control.

## Canonical tensor contract

All T0 data and models operate on:

```python
input_latents: FloatTensor[B,T,D]
target_latents: FloatTensor[B,T,D]
input_mask: BoolTensor[B,T]
loss_mask: BoolTensor[B,T]
```

`input_mask` marks observed context positions. `loss_mask` marks target positions included in the objective. Missing-span reconstruction uses `input_mask=false` on hidden spans and `loss_mask=true` on those hidden spans.

## Metrics

The runner records the Sprint T0 required metrics per task and model:

- MSE,
- spectral magnitude loss,
- phase error,
- frequency recovery error,
- throughput in samples/second,
- stability (`nan_or_inf`, `loss_explosion`, gradient-norm summaries).

## Running the tiny config

```bash
python -m training.t0_synthetic_pretrain --config configs/t0/synthetic_tiny.json
```

Artifacts are written under `experiments/synthetic/t0/<experiment_id>/`:

- `metrics.json`,
- `registry.json`,
- `comparison_table.csv`,
- `prediction_preview_<task>.csv`,
- `prediction_preview_<task>.svg`,
- `failure_report.md` if the stop condition is triggered.

## Stop condition

If SLWM does not beat the vanilla Transformer baseline on any controlled synthetic signal task, the T0 runner writes a failure report and marks the registry entry as failed. Later text/code/audio/video training must not proceed from that result.

## Claims allowed

T0 results can support only controlled synthetic signal metric claims. They do not support claims about multimodal grounding, hallucination reduction, policy behavior, text/code performance, or audio/video understanding.

---

# Sprint T1 Training — Text/Code Baseline Training

Sprint T1 is limited to text/code data. It does **not** use audio or visual/video
data.

## Scope

Implemented dependency-light pilot paths:

- `gpt2_baseline`: GPT-2-style decoder-only next-token baseline,
- `slwm_text_only`: text adapter → shared latent field → SLWM processor → text LM head,
- `slwm_text_only_no_spectral`: same as above with spectral mixer disabled.

## Running the tiny pilot configs

```bash
python -m training.t1_text_baseline --config configs/t1/gpt2_text_tiny_pilot.json
python -m training.t1_text_baseline --config configs/t1/slwm_text_tiny_pilot.json
python -m training.t1_text_baseline --config configs/t1/slwm_text_no_spectral_tiny_pilot.json
```

Artifacts are written under `experiments/text/t1/<experiment_id>/`:

- `registry.json`,
- `metrics.json`,
- `samples.json`,
- `report.md`,
- `checkpoint.npz`,
- copied `config.json`.

## Claim limits

The tiny T1 pilot uses a dependency-free byte fallback tokenizer and inline
project-authored records for pipeline validation. Full T1 text/code evidence
requires the documented GPT-2 BPE tokenizer and prepared external text/code
datasets described in `docs/t1_text_code_training.md` and
`configs/t1/text_code_full_dataset_plan.json`.
