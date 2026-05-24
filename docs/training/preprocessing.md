# SLWM Preprocessing — Sprint T2 Audio/Visual Latents

**Sprint:** T2 — Audio/visual latent training  
**Owner role:** Training  
**Status:** config-driven latent-corpus preparation and training mechanics are implemented. External high-quality corpus preparation is documented but not run in this change.

## 1. Scope and anti-drift boundary

T2 trains on **audio and visual/video latents only**. It supports:

- audio latent continuation;
- visual/video latent continuation;
- missing-span reconstruction;
- audio-video correspondence/alignment with shuffled controls.

T2 explicitly does **not** train text generation, raw waveform/video generation, policy/commit behavior, hallucination evaluation, or action/sensor modalities.

## 2. Applicable requirements

| Source | Requirement | T2 handling |
|---|---|---|
| `docs/process/sprint_playbook_prompts.md` T2 | Use compressed audio and visual/video latents; report prediction metrics. | `training/t2_prepare_latents.py` prepares latent corpora; `training/t2_train_latents.py` logs latent prediction, spectral proxy, visual latent, and correspondence metrics. |
| T2 KPI | Audio latent dataset loads. | `T2PreparedLatentDataset` loads `audio_features: FloatTensor[B,T,A]`. |
| T2 KPI | Visual/video latent dataset loads. | `T2PreparedLatentDataset` loads `visual_features: FloatTensor[B,T,V]`. |
| T2 KPI | Batch format matches data contract. | Batches preserve modality IDs `audio=2`, `visual_video=3`, masks, and source tags. |
| T2 KPI | Prediction loss decreases. | Short runs log `initial_validation.total_loss`, `validation.total_loss`, and `success_gate.prediction_loss_decreased`. |
| T2 KPI | Shuffled-modality baseline included. | Runner logs shuffled audio-video retrieval control metrics. |
| T2 KPI | Cross-modal alignment metric reported. | Runner logs retrieval R@1/R@5/R@10 and binary correspondence accuracy. |
| H-R0-1 / DD-R1-009 | Latent prediction is the primary objective. | MSE continuation and missing-span reconstruction over latent features. |
| H-R0-2 / DD-R1-018 | Shared latent alignment must beat null/shuffled controls before claims. | Null/random/shuffled control metrics are registered; full support requires external runs and ablations. |
| DD-R1-007 | Start audio with log-mel/lightweight latents. | Recommended external audio feature: frozen 80-bin log-mel or equivalent precomputed latent. |
| DD-R1-008 | Start visual/video with patch/tube latents. | Recommended external visual feature: frozen ViT/DINO/VideoMAE-style frame/tube latents. |

## 3. Latent corpus format

Prepared samples are stored as compressed NPZ files with:

```text
audio_features: FloatTensor[T_audio,A]
visual_features: FloatTensor[T_visual,V]
audio_mask: BoolTensor[T_audio]
visual_mask: BoolTensor[T_visual]
```

Each split has a JSONL manifest under:

```text
artifacts/t2_audio_visual/<corpus>/manifests/{train,validation,test}.jsonl
```

and a dataset card:

```text
artifacts/t2_audio_visual/<corpus>/dataset_card.json
```

The dataset card records split counts, feature specs, dataset/source notes, license notes, leakage checks, and a manifest SHA-256.

## 3.1 High-volume `/data` acquisition plan

The high-volume data planner shared with T1 is available at:

```bash
python -m training.t1_t2_high_volume_data \
  --config configs/data/t1_t2_high_volume_data_plan.json \
  --data-root /data/projects/slwm \
  --stage high_volume
```

The planner creates small manifests and a directory layout under `/data/projects/slwm`:

```text
/data/projects/slwm/
  manifests/t1_t2_high_volume/<plan_id>/
  t1_text_code/
  t2_signal_latents/
    raw_cache/
    manifests/
    audio_latents/
    visual_latents/
    av_pair_index/
    feature_extraction_logs/
```

It remains dry-run by default. High-volume media acquisition is blocked until
dataset-specific terms, credentials, required tools, checksums, and legal/ToS
decisions are recorded. This is intentional because T2 external sources include
YouTube-derived corpora and mixed-license audio. The planner therefore satisfies
only the acquisition-planning gate; it does **not** prove that external T2 data is
downloaded, deduplicated, feature-extracted, or ready for model-quality claims.

T2 source priorities in the current `/data` plan are:

1. LibriSpeech and FSD50K for audio controls;
2. AudioSet released features before raw YouTube acquisition;
3. COCO/UCF101 for visual/video controls, with captions/labels as metadata only;
4. VGGSound 5k available clips in the approximately 1 TB quality stage, then
   larger VGGSound available clips as the high-volume audio-video correspondence
   target after terms, legal/ToS review, and media availability are recorded;
5. Common Voice EN and Kinetics/MSR-VTT subsets only after terms/tooling review.

For T2, all caption/transcript text must remain metadata or alignment labels. It
must not become a text generation objective in this sprint.

## 4. Recommended high-quality dataset plan

### Primary paired audio-video corpus: VGGSound subset

- **Initial evidence-bearing local subset:** 5k train / 500 validation / 500 test clips after availability filtering.
- **Larger local subset:** 20k train / 2k validation / 2k test clips if storage and download stability allow.
- **Current `/data` quality-stage estimate:** about 45 GB for the initial
  available-clips subset, bringing the quality-stage T1/T2 plan to roughly
  1.0 TB before preprocessing overhead.
- **Why:** directly supports audio-video correspondence and shuffled-pair controls.
- **Caveats:** YouTube availability drift and source-media rights must be logged; dead/unavailable clips must be recorded in the manifest.

### `/data` staging commands and storage headroom

For the user-requested approximately 1 TB multimodal acquisition/preparation
target, use the quality stage:

```bash
python -m training.t1_t2_high_volume_data \
  --config configs/data/t1_t2_high_volume_data_plan.json \
  --data-root /data/slwm \
  --stage quality
```

The full high-volume dry-run remains available, but should be treated as a later
capacity-gated step:

```bash
python -m training.t1_t2_high_volume_data \
  --config configs/data/t1_t2_high_volume_data_plan.json \
  --data-root /data/slwm \
  --stage high_volume
```

Keep additional free space for decompression, retries, duplicate/leakage indexes,
token arrays, feature extraction intermediates, and checkpoints. Do not start a
stage if its raw estimate plus processing overhead would consume more than about
80-85% of available `/data` space.

### Audio-only controls

- **LibriSpeech train-clean-100:** stable, clean, license-clear English speech, useful for audio latent continuation sanity checks.
- **FSD50K 10k subset:** general audio events with downloadable metadata/audio; track per-clip license classes.

### Visual/video-only controls

- **UCF101 or Kinetics-mini:** manageable temporal visual prediction controls. Use only visual/video latents in T2; labels are metadata.

### Deferred in T2

- Raw AudioSet full/unbalanced media mirrors, Kinetics full, and any MSR-VTT/COCO
  caption-driven text objectives remain deferred. AudioSet released features and
  COCO images may be used as T2 controls only when labels/captions stay metadata
  and do not become text-generation targets.

## 5. Feature extraction recommendations

T2 currently expects precomputed arrays. External feature extraction should be done before training and recorded in the dataset card.

| Modality | First-pass feature | Notes |
|---|---|---|
| Audio | 16 kHz mono, 80-bin log-mel, fixed window/hop | Lightweight and reproducible. EnCodec-style latents are future comparison after log-mel path works. |
| Visual/video | Frozen frame or tube patch latents, fp16/fp32 arrays | Sample frames at fixed FPS; store compact per-frame/per-tube latent vectors. |
| Paired AV | Same clip start/end for audio and visual | Split before training; shuffled negatives drawn within the same validation split. |

Every external corpus must record feature extractor name/version, source split, sample count, clip duration policy, failed media count, and SHA-256 hashes.

## 6. Runnable commands

The current end-to-end T1/T2 data-prep and training spin-off commands are in
`docs/training/t1_t2_reproducible_training_runbook.md`. That runbook includes the
prepared external T2 control corpus, the VGGSound available-clips corpus, and the
claim limits for availability-biased natural-AV evaluation.

### Smoke latent corpus

```bash
python -m training.t2_prepare_latents --config configs/t2/prepare_audio_visual_generated_smoke.json
```

This writes a project-generated latent fixture. It is for **pipeline validation only**, not evidence of real audio/video quality.

### Tiny smoke training

```bash
python -m training.t2_train_latents --config configs/t2/slwm_t2_tiny_smoke.json --max-steps 2 --no-checkpoint
```

This verifies that the runner loads audio/visual latents, executes training briefly, logs prediction/correspondence metrics, and terminates.

### GPT-2-scale and 700M config inspection

```bash
python -m training.t2_train_latents --config configs/t2/slwm_124m_audio_visual_pilot.json --describe-only
python -m training.t2_train_latents --config configs/t2/slwm_700m_audio_visual_24gb_fitcheck.json --describe-only
```

`--describe-only` estimates parameter counts and checks config metadata without allocating the large model or starting training.

## 7. Model-size configs

| Config | Purpose | Notes |
|---|---|---|
| `configs/t2/slwm_124m_audio_visual_pilot.json` | GPT-2-scale T2 latent run | Intended for VGGSound 5k latent corpus; strict accounting is near 124M without text embeddings because T2 is AV latent-only. |
| `configs/t2/slwm_124m_audio_visual_no_spectral_pilot.json` | Required no-spectral ablation | Same planned corpus/optimizer/seed; only spectral mixer disabled. |
| `configs/t2/slwm_700m_audio_visual_24gb_fitcheck.json` | 700M+ scaling fit-check | Best-effort 24GB VRAM config: bf16/fp16, activation checkpointing, batch size 1, gradient accumulation. Not comparable to 124M claims. |

## 8. Acceptance and claim limits

T2 acceptance requires registered artifacts showing:

1. audio and visual latent splits load;
2. batch masks and modality IDs match the data contract;
3. physical microbatch size is at least `2` whenever audio-video contrastive
   alignment is enabled, because gradient accumulation does not create in-batch
   negatives;
4. prediction loss decreases in a short run;
5. shuffled-pair and null/random controls are logged;
6. cross-modal alignment metrics are reported;
7. configs and checkpoints are saved when not running a no-checkpoint smoke.

Even if those pass, T2 supports only latent-prediction/correspondence mechanics and metrics. It does not support claims about text/code quality, hallucination reduction, policy behavior, or external-output grounding.
