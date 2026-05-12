# SLWM Training Runners

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

Implemented full-stack mechanics path:

- `training.t1_prepare_text_code`: prepares GPT-2-BPE text/code-only token streams from configured local/Hugging Face sources and writes manifest/split hashes under `artifacts/t1_text_code/`.
- `training.t1_torch_text`: trains GPT-2-size PyTorch/MPS GPT-2 and SLWM text-only variants from prepared corpora and writes T1 registry artifacts.

The full-stack path is still limited by the registered train-token budget; the current configs are 124M-scale mechanics/initial benchmark runs, not converged GPT-2 training.

## Running the GPT-2-BPE prepared-corpus path

```bash
python -m training.t1_prepare_text_code --config configs/t1/text_code_gpt2_bpe_larger_local_prepare.json
python -m training.t1_torch_text --config configs/t1/gpt2_text_torch_124m_larger_local.json
python -m training.t1_torch_text --config configs/t1/slwm_text_torch_124m_larger_local.json
python -m training.t1_torch_text --config configs/t1/slwm_text_no_spectral_torch_124m_larger_local.json
```

Artifacts are written under `experiments/text/t1/<experiment_id>/`:

- `registry.json`,
- `metrics.json`,
- `samples.json`,
- `report.md`,
- `checkpoint.pt`,
- copied `config.json`.

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
datasets described in `../docs/training/t1_text_code_training.md` and
`../configs/t1/text_code_full_dataset_plan.json`.

---

# Sprint T2 Training — Audio/Visual Latent Training

Sprint T2 is limited to audio and visual/video **latents**. It does not train
text generation, raw waveform/video generation, policy behavior, hallucination
metrics, or action/sensor modalities.

## Scope

Implemented paths:

- `training.t2_prepare_latents`: standardizes precomputed audio/visual latent
  records into NPZ files, split manifests, and a dataset card under
  `artifacts/t2_audio_visual/`.
- `training.t2_train_latents`: trains a PyTorch audio/visual latent model on
  continuation, missing-span reconstruction, and audio-video contrastive
  alignment.

Implemented controls and metrics:

- null/persistence latent MSE,
- random latent MSE,
- shuffled audio-video retrieval baseline,
- audio latent MSE and spectral/phase/frequency proxy metrics,
- visual/video latent error,
- audio-video retrieval R@1/R@5/R@10 and correspondence accuracy,
- throughput, memory, registry, config copy, report, and optional checkpoint.

## Running the generated smoke fixture

```bash
python -m training.t2_prepare_latents --config configs/t2/prepare_audio_visual_generated_smoke.json
python -m training.t2_train_latents --config configs/t2/slwm_t2_tiny_smoke.json --max-steps 2 --no-checkpoint
```

The generated fixture is for pipeline validation only and must not be used as
audio/video quality evidence.

See `../docs/training/preprocessing.md` for the T2 external dataset and feature
extraction plan.

## Inspecting larger model configs without starting training

```bash
python -m training.t2_train_latents --config configs/t2/slwm_124m_audio_visual_pilot.json --describe-only
python -m training.t2_train_latents --config configs/t2/slwm_700m_audio_visual_24gb_fitcheck.json --describe-only
```

The 700M+ config is a best-effort 24GB VRAM fit-check profile using bf16/fp16,
activation checkpointing, batch size 1, and gradient accumulation. It is not a
replacement for the 124M strict-comparison path.

## Claim limits

T2 can report only latent prediction, reconstruction, spectral proxy,
audio-video correspondence/retrieval, shuffled/null controls, throughput, and
memory. It cannot support text/code, policy, hallucination, raw generation, or
grounding claims.
