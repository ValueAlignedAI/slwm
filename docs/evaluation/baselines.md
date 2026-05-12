# SLWM Sprint I1 — Baselines

**Sprint:** I1 — Baselines  
**Status:** implemented baseline smoke paths; no SLWM novelty modules; no model-quality claim.  
**Gate:** at least one text baseline and one multimodal/latent baseline can tiny-train, overfit a fixed tiny batch, and write an experiment registry entry.

## 1. Applicable requirements

| Source ID | Requirement | I1 status |
|---|---|---|
| Sprint I1 | Implement baselines only before SLWM novelty. | Met for GPT-2-style text and vanilla multimodal Transformer baselines. |
| Sprint I1 | GPT-2-small-style decoder-only baseline. | Implemented in `models/baselines/gpt2_decoder.py`. |
| Sprint I1 / DD-R1-003 | Vanilla multimodal Transformer baseline. | Implemented in `models/baselines/vanilla_multimodal_transformer.py`. |
| Sprint I1 / H-R0-2 | Null/random controls for probes/baselines. | Implemented in `models/baselines/null_random.py`. |
| DD-R1-001 / G-R0-1 | GPT-2 baseline before text/code comparison. | Baseline config and tiny smoke registry entry exist; no SLWM comparison yet. |
| DD-R1-002 | Use BPE first for text/code comparability. | Full GPT config uses GPT-2 BPE vocabulary size; tiny smoke uses deterministic integer surrogate IDs. |
| DD-R1-020 | Register every experiment before using it as evidence. | `EXP-I1-001` and `EXP-I1-002` registry files were written by `training/baseline_smoke.py`. |

Perceiver-style latent baseline remains **not implemented** in this sprint pass. It is allowed only “if feasible”; the accepted I1 gate is satisfied by the text and vanilla multimodal/latent smoke baselines below.

## 2. Implemented files

| Path | Purpose |
|---|---|
| `models/baselines/numpy_nn.py` | Small NumPy Transformer primitives and AdamW for deterministic smoke training. |
| `models/baselines/gpt2_decoder.py` | GPT-2-style causal decoder-only language-model baseline. |
| `models/baselines/vanilla_multimodal_transformer.py` | Serialized text/audio/visual vanilla Transformer baseline. |
| `models/baselines/null_random.py` | Uniform, random, and shuffled-target controls. |
| `models/baselines/parameter_count.py` | Exact formula parameter accounting. |
| `training/baseline_smoke.py` | Config-driven tiny overfit smoke runner and registry writer. |
| `configs/baselines/gpt2_small_style.json` | Full GPT-2-small-style reference config. |
| `configs/baselines/gpt2_tiny_smoke.json` | Tiny GPT-2-style overfit smoke config. |
| `configs/baselines/vanilla_multimodal_124m_style.json` | 124M-style vanilla multimodal reference config. |
| `configs/baselines/vanilla_multimodal_tiny_smoke.json` | Tiny vanilla multimodal overfit smoke config. |

## 3. Model configs and parameter counts

### 3.1 GPT-2-small-style decoder baseline

Config path: `configs/baselines/gpt2_small_style.json`

| Field | Value |
|---|---:|
| Variant | `gpt2_baseline` |
| Context length | 1024 |
| Vocabulary / tokenizer | GPT-2 BPE, `vocab_size=50257` |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| MLP size | 3072 |
| Weight tying | token embedding tied to LM head; no separate LM-head matrix |
| Parameter accounting mode | strict |

Exact formula count:

| Component | Parameters |
|---|---:|
| Token embeddings | 38,597,376 |
| Position embeddings | 786,432 |
| Transformer blocks | 85,054,464 |
| Final layer norm | 1,536 |
| Separate LM head | 0 |
| **Total** | **124,439,808** |

### 3.2 GPT-2 tiny overfit smoke config

Config path: `configs/baselines/gpt2_tiny_smoke.json`

| Field | Value |
|---|---:|
| Context length | 8 |
| Vocabulary / tokenizer | tiny GPT-2-BPE surrogate integer IDs, `vocab_size=32` |
| Layers | 1 |
| Hidden size | 16 |
| Attention heads | 2 |
| MLP size | 64 |
| Batch / sequence | 4 × 8 |
| Steps | 180 |
| Parameters | 3,952 |

### 3.3 Vanilla multimodal Transformer reference baseline

Config path: `configs/baselines/vanilla_multimodal_124m_style.json`

| Field | Value |
|---|---:|
| Variant | `vanilla_multimodal_transformer` |
| Context length | 1024 serialized positions |
| Text codec | GPT-2 BPE, `text_vocab_size=50257` |
| Audio codec/features | provided continuous audio latents or log-mel-like features, `audio_feature_dim=80` |
| Visual codec/features | provided visual patch/video latents, `visual_feature_dim=256` |
| Output target vocabulary | 1024 discrete/latent classes |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| MLP size | 3072 |
| Parameter accounting mode | strict |

Exact formula count:

| Component | Parameters |
|---|---:|
| Text embeddings | 38,597,376 |
| Position embeddings | 786,432 |
| Modality embeddings | 2,304 |
| Audio projection | 62,208 |
| Visual projection | 197,376 |
| Transformer blocks | 85,054,464 |
| Final layer norm | 1,536 |
| Output head | 787,456 |
| **Total** | **125,489,152** |

This is approximately 1.20M parameters above GPT-2-small-style strict count and within the documented 124M ±5% project tolerance.

### 3.4 Vanilla multimodal tiny overfit smoke config

Config path: `configs/baselines/vanilla_multimodal_tiny_smoke.json`

| Field | Value |
|---|---:|
| Context length | 9 serialized positions: 3 text + 3 audio + 3 visual |
| Text codec | tiny GPT-2-BPE surrogate integer IDs, `text_vocab_size=32` |
| Audio features | deterministic continuous smoke latents, `audio_feature_dim=5` |
| Visual features | deterministic continuous smoke latents, `visual_feature_dim=6` |
| Target vocabulary | 24 |
| Layers | 1 |
| Hidden size | 16 |
| Attention heads | 2 |
| MLP size | 64 |
| Batch / sequence | 4 × 9 |
| Steps | 220 |
| Parameters | 4,632 |

## 4. Smoke-test metrics logged

The following metrics were generated by:

```bash
python -m training.baseline_smoke --config configs/baselines/gpt2_tiny_smoke.json
python -m training.baseline_smoke --config configs/baselines/vanilla_multimodal_tiny_smoke.json
```

### EXP-I1-001 — GPT-2-style tiny overfit

Registry: `experiments/baselines/EXP-I1-001.json`  
Metrics: `experiments/baselines/EXP-I1-001.metrics.json`

| Metric | Value |
|---|---:|
| Initial loss | 3.4723506989445987 |
| Final loss | 0.00027337018293628065 |
| Loss drop | 99.99212723003413% |
| Final perplexity | 1.00027340755197 |
| NaN/Inf observed | false |
| Loss explosion observed | false |
| Trainable parameters | 3,952 |
| Context length | 8 |
| Text tokenizer/codec | `tiny_gpt2_bpe_surrogate_ids` |

### EXP-I1-002 — Vanilla multimodal Transformer tiny overfit

Registry: `experiments/baselines/EXP-I1-002.json`  
Metrics: `experiments/baselines/EXP-I1-002.metrics.json`

| Metric | Value |
|---|---:|
| Initial loss | 3.743400626375035 |
| Final loss | 0.00006488073409118063 |
| Loss drop | 99.99826679694303% |
| Final perplexity | 1.0000648828388916 |
| Uniform null loss | 3.178053830347946 |
| Random null loss | 3.360125061858165 |
| Shuffled-target loss | 13.19818574140111 |
| NaN/Inf observed | false |
| Loss explosion observed | false |
| Trainable parameters | 4,632 |
| Context length | 9 |
| Text codec | `tiny_gpt2_bpe_surrogate_ids` |
| Audio codec/features | `provided_continuous_audio_latents_for_smoke` |
| Visual codec/features | `provided_continuous_visual_latents_for_smoke` |

## 5. Scope exclusions and limitations

- No SLWM novelty modules were implemented: no spectral mixer, no LongConv/SSM, no SLWM processor, no learned policy gate, and no uncertainty/source head.
- No external dataset was loaded. Smoke data is deterministic synthetic/in-memory data for overfit validation only.
- No text/code quality, multimodal grounding, hallucination, or policy claim is made from these smoke runs.
- Full 124M-style configs are parameter-accounting/config artifacts only; this sprint did not run large training.
- Perceiver-style latent baseline remains future/pending.

## 6. Acceptance status

Sprint I1 baseline implementation gate is met for the required text and vanilla multimodal/latent paths:

1. GPT-2-style baseline runs forward/backward and overfits a tiny batch.
2. Vanilla multimodal Transformer baseline runs forward/backward and overfits a tiny serialized multimodal batch.
3. Exact parameter counts are logged for full reference and tiny smoke configs.
4. Tokenizer/codec choices, context lengths, and configs are recorded.
5. Both smoke runs wrote registry and metrics artifacts.
