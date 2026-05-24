# SLWM T1/T2 Reproducible Training Runbook

This runbook captures the current reproducible path after the approximately 1 TB
T1/T2 data-preparation pass. It is intended for spinning off registered training
runs, not for making final model-quality claims.

## Current prepared data

### T1 text/code corpus

- Path: `artifacts/t1_text_code/gpt2_bpe_1b_v0_exact_dedup/`
- Dataset card: `artifacts/t1_text_code/gpt2_bpe_1b_v0_exact_dedup/dataset_card.json`
- Status: `prepared_exact_split_deduped`
- Tokenizer: GPT-2 BPE
- Tokens: `999,965,359` after exact cross-split duplicate filtering
- Splits:
  - train: `981,619,919`
  - validation: `9,115,247`
  - test: `9,230,193`
- Scope: T1 text/code only; no audio/visual data.
- Filter note: `50` exact text hashes that crossed split boundaries were removed
  from validation/test while preserving train-priority canonical rows. This does
  not replace near-dedup, PII/secrets, or eval-contamination scans.

### T2 external control latent corpus

- Path: `artifacts/t2_audio_visual/external_t2_control_v0/`
- Dataset card: `artifacts/t2_audio_visual/external_t2_control_v0/dataset_card.json`
- Status: `prepared`
- Samples: `1,200` (`1,000/100/100` train/validation/test)
- Scope: LibriSpeech audio + COCO image cross-source control pairs.
- Natural AV correspondence: `false`.

### T2 VGGSound available-clips latent corpus

- Path: `artifacts/t2_audio_visual/vggsound_5k_available_v0/`
- Dataset card: `artifacts/t2_audio_visual/vggsound_5k_available_v0/dataset_card.json`
- Status: `prepared`
- Media availability: `2,196` downloaded / `3,804` failed from `6,000` attempted rows.
- Prepared records: `2,194`.
- Splits: `2,183/5/6` train/validation/test.
- Scope: natural paired audio-video available subset; labels are metadata only.
- Caveat: validation/test are too small for evidence-bearing natural-AV claims.

## Environment setup

```bash
cd /home/db/dev/slwm
source .venv/bin/activate
set -a
source /data/slwm/env_defaults.sh
source .env.local  # local secret file; do not commit
set +a
```

## Data controls before training claims

Run the manifest-level controls:

```bash
scripts/train/spin_off_t1_t2_repro.sh data-controls
```

Output:

```text
/data/slwm/manifests/controls/t1_t2_manifest_controls_v0/
  t1_t2_manifest_controls.json
  t1_t2_manifest_controls.md
```

This closes only exact manifest-level accounting/split checks. It does **not**
close near-duplicate deduplication, PII/secrets scans over raw source text/code,
or benchmark/eval contamination checks.

## T1 reproducible 1B-token matched runs

These runs are an optional T1 guardrail path, not the critical path for the
current T2 latent-training work. If the immediate goal is T2 audio/visual latent
learning and token/byte-to-signal conversion, do not spend GPU time training a
GPT-2-style baseline from scratch first. A downloaded/pretrained GPT-2 reference
can be used later as a clearly labeled text-only comparison, while T2 remains
audio/visual-latent only.

Launch all three matched 124M-scale T1 configs sequentially:

```bash
scripts/train/spin_off_t1_t2_repro.sh t1-1b
```

Equivalent explicit commands:

```bash
python -m training.t1_torch_text --config configs/t1/gpt2_text_torch_124m_1b_v0.json
python -m training.t1_torch_text --config configs/t1/slwm_text_torch_124m_1b_v0.json
python -m training.t1_torch_text --config configs/t1/slwm_text_no_spectral_torch_124m_1b_v0.json
```

Expected registry outputs:

```text
experiments/text/t1/EXP-T1-411/  # GPT-2-style baseline
experiments/text/t1/EXP-T1-412/  # SLWM text-only
experiments/text/t1/EXP-T1-413/  # SLWM no-spectral ablation
```

These are long-running 1B-token-budget runs. They should be reviewed against the
same corpus manifest hash, tokenizer metadata, seed, optimizer family, sequence
length, train-token budget, and parameter accounting.

## T2 reproducible smoke checks

Run both available prepared-corpus smoke checks:

```bash
scripts/train/spin_off_t1_t2_repro.sh t2-smokes
```

Equivalent explicit commands:

```bash
python -m training.t2_train_latents \
  --config configs/t2/slwm_t2_external_control_smoke.json \
  --max-steps 2 --no-checkpoint

python -m training.t2_train_latents \
  --config configs/t2/slwm_t2_vggsound_available_smoke.json \
  --max-steps 2 --no-checkpoint
```

Expected registry outputs:

```text
experiments/multimodal/t2/EXP-T2-902/  # external control smoke
experiments/multimodal/t2/EXP-T2-903/  # VGGSound available smoke
```

These prove loader/training mechanics only.

## T2 124M VGGSound available-subset runs

The T2 configs are now tuned for RTX 3090 utilization with physical microbatch
`batch_size=256` and `gradient_accumulation_steps=1` (effective batch `256`). This
matters because audio-video contrastive alignment needs in-batch negatives;
`batch_size=1` is a mechanics-only configuration and produces degenerate
retrieval/correspondence metrics even if gradient accumulation is large.

Inspect first:

```bash
scripts/train/spin_off_t1_t2_repro.sh describe
```

Launch matched 124M T2 runs sequentially:

```bash
scripts/train/spin_off_t1_t2_repro.sh t2-vggsound-124m
```

Equivalent explicit commands:

```bash
python -m training.t2_train_latents --config configs/t2/slwm_124m_vggsound_available_v0.json
python -m training.t2_train_latents --config configs/t2/slwm_124m_vggsound_available_no_spectral_v0.json
```

Expected registry outputs:

```text
experiments/multimodal/t2/EXP-T2-111/  # SLWM AV latent
experiments/multimodal/t2/EXP-T2-112/  # no-spectral ablation
```

Because the VGGSound available validation/test splits are tiny, these runs are
useful for scaling and mechanics. Strong natural-AV conclusions require a larger
availability-balanced evaluation split.

## Monitoring launched runs

```bash
pgrep -af 't1_torch_text|t2_train_latents|spin_off_t1_t2_repro'
ls -lt /data/slwm/logs/repro_training
tail -f /data/slwm/logs/repro_training/<log-file>
df -h /home/db/dev/slwm /data/slwm
```

## Claim limits

Allowed now:

- T1 1B GPT-2-BPE corpus is prepared.
- T2 external-control and VGGSound available latent corpora are prepared.
- T2 smoke mechanics load audio/visual latent batches and record controls.

Not allowed until additional evidence exists:

- training-ready or model-quality claims;
- leakage-clean claims;
- PII/secrets-clean T1 claims;
- strong natural audio-video correspondence claims from the tiny VGGSound eval split;
- full 15 TB high-volume completion claims.
