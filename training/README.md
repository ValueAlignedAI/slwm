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

## Real recorded audiovisual pilots

These are 4.47M-trainable-parameter T2 experiments on recorded CREMA-D speech
video and audio. The frozen pooled ImageNet ResNet18 has 11,176,512 deployed
parameters, counted separately. English sentence transcripts are metadata only;
there is no trained text/code or raw-media decoder.

Install the `t2-full` optional dependencies and `pytest`; `ffmpeg`/`ffprobe` must
be available. On the authorized Fedora machine, the GPU runtime is
`/home/db/venvs/vllm/bin/python` and the checkout is `/home/db/dev/slwm`.
The isolated `.research-venv` supplies test/plotting packages.

```bash
python -m training.t2_prepare_cremad --config configs/t2/cremad_prepare_pilot.json
python -m training.t2_real_av --config configs/t2/cremad_pilot_matrix.json
python -m training.t2_real_av_report --config configs/t2/cremad_pilot_matrix.json
python -m training.t2_prepare_cremad --config configs/t2/cremad_prepare_fresh_residual.json
python -m training.t2_real_av --config configs/t2/cremad_residual_matrix.json
python -m training.t2_real_av_report --config configs/t2/cremad_residual_matrix.json
```

On Fedora, `bash scripts/train/run_t2_real_av_fedora.sh` runs the first prepared
matrix and report. Set `SLWM_MATRIX=configs/t2/cremad_residual_matrix.json` for
the follow-up. Completed results are reused after configuration checks; prepared
corpora are reused after file-hash checks and never silently overwritten. Use a
new versioned directory and experiment IDs for a changed experiment.

Both cycles have `best.pt` checkpoints with optimizer/RNG/config/provenance,
`metrics.json`, `registry.json`, `curves.json`, `source_snapshot/`, report,
comparison tables, and plots. Exact frozen-run source is archived, including the
original accounting convention; `parameter_accounting.json` corrects the unused
classifier count without modifying hashed source cards. The final source also
adds artifact-integrity checks. All 18 checkpoints and compact prepared corpora
are retained locally; all original raw clips remain on Fedora.

Run a saved residual model on the supplied local/Fedora real-clip example:

```bash
python -m training.t2_real_av_infer \
  --checkpoint experiments/multimodal/t2/real-av-residual-20261104/EXP-T2-211/best.pt \
  --clip artifacts/t2_audio_visual/cremad_fresh_residual_v2/raw/1003_DFA_ANG_XX.flv \
  --output experiments/multimodal/t2/real-av-residual-20261104/example_prediction.npz
```

Outputs contain observed context and predicted future features, tagged internal
and uncommitted. They are not generated waveforms, video frames, or statements.
`--mode audio_only` and `--mode visual_only` support diagnostic missing-input
checks. `--device cpu` permits inference without CUDA.

Exploratory stronger controls are separate from frozen gates:

```bash
python -m evals.t2_feature_controls --config configs/t2/cremad_residual_matrix.json
```

The fixed-regularization linear weights are saved as reusable NPZ artifacts.
They were added after some test outputs were seen and must remain labeled
exploratory. Their inputs and retained persistence anchors are documented in
the metrics/report.

## Motion-aware alignment, scaling, and generated outputs

Use [the motion alignment workflow](../docs/training/motion_alignment.md) and
`configs/t2/motion_alignment_matrix.json`. The new 4.74M/15.23M models train
neural components from scratch on physical-time audio and mouth motion.
`bash scripts/train/run_motion_alignment_fedora.sh` reproduces training,
causal checks, frozen probes, the actual media gallery, and reporting.

The exploratory linear-head diagnosis is separate:

```bash
python -m exploration.motion_linear_decoders --config configs/t2/motion_alignment_matrix.json
python -m training.t2_motion_report --config configs/t2/motion_alignment_matrix.json
```

It freezes core weights and retains every original head/output and primary metric.
The optional `evals.motion_audio_intelligibility` uses a separate pretrained
recognizer only to score saved waveforms. It never supplies generated audio.

## T2-G1/G2 conditional generation

Fifteen additional GPU runs test 100 Hz sound, a single real appearance seed,
and a signal-preserving raw audio path. The raw residual audio edge passes its
two-seed reconstruction gate; frozen-core benefit, video-to-speech and visual
dynamics gates fail. Run `scripts/train/run_conditional_generation_fedora.sh`
and `scripts/train/run_conditional_residual_fedora.sh` from the repository.
See `docs/training/conditional_generation.md` for standalone real-file inference,
data/parameter accounting, the shuffled-control evaluation correction, and claim
limits. All original checkpoints, metrics and media remain retained.

## T2-R1/R2 signal restoration and content conservation

Thirteen further GPU runs mix real LibriSpeech audio with CREMA audiovisual
training. A 16,448-weight auxiliary observed-content readout fixes a measurable
decodability problem in the small shared field, passing two-seed content/output/
timing gates on a fresh harder test-other set. Correct video still does not
reliably improve repair beyond sound alone. See `docs/training/signal_restoration.md`
and the current findings for field-only probes, held-out split rules, actual
audio, ordinary-file inference and the limits of mel-packet restoration.

```bash
bash scripts/train/run_signal_restoration_fedora.sh
bash scripts/train/run_content_conservation_fedora.sh
```


# T2-V1 — true-gap video versus audio repair

Eleven registered RTX 3090 runs finish the controlled test: video reduces reused
CREMA gap-spectrum error ~9% over additional audio capacity in both seeds, but
fresh-speaker intervals and the broader speech gate fail. The full model has
7,990,547 weights, including a1,423,025-weight newly trained visual branch.
The old content expert and adapted audio anchor are frozen during fusion.

See [the training/inference guide](../docs/training/video_repair.md),
[frozen report](../experiments/multimodal/t2/video-repair-20261105/report.md),
[all fixed listening cases](../experiments/multimodal/t2/video-repair-20261105/listen.html)
and [packaged weights](../experiments/multimodal/t2/video-repair-20261105/bundle/README.md).

```bash
scripts/train/run_video_repair_fedora.sh data
scripts/train/run_video_repair_fedora.sh train
scripts/train/run_video_repair_fedora.sh evaluate
```

The completed evidence preserves the first failed rendering attempt. Its
terminal-padding correction has a separate evaluation revision:
`scripts/train/resume_video_repair_render_v2.sh`. The original evaluation source
guard deliberately rejects changed rendering sources; use this explicit revision
for the retained run. Training and primary feature scores are unchanged.
The full suite passes124 tests; audits verify448 WAVs and all11 frozen experts.

## Fresh-person speech content and GPT-2-scale text — 2026-11-05

The [completed study guide](../docs/training/content_and_scale.md) covers ten
small speech-content/timing runs, four separate124M/125M text runs, portable
checkpoints, original-file CPU inference, corpus/compute accounting and scoring
revision2. Current speech content, rejection and spectral-video gates pass;
practical speech quality remains mixed. The text runs are early16.8M-token
checkpoints from a prepared2.6B-token corpus, not converged pretraining.
