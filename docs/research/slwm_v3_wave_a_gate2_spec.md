# SLWM-v3 Wave A — Gate 2 Connected Data Contract, Causal Codec Ceilings & Dual-Track Accounting Specification (Rev 2)

**Status:** Gate 2 Evidence Pack (Rev 2 — resolves all 5 Gate 2 peer-review blocking flaws and all non-blocking advisories; strictly `dev`/`val`-only codec audits with zero sealed `test`/`vault` data inspected)  
**Branch:** `research/slwm-v3-selective-memory`  
**Verified Code, Unit Tests & JSON Receipts:**
- Data Contract, Causal Timestamps, Pixel-Observable Renderer & Collision-Free Interleaved Streams: [v3_data_contract.py](../../data/v3_data_contract.py)
- Gate 2 Unit Tests: [test_v3_gate2_data_contract.py](../../tests/test_v3_gate2_data_contract.py) (`5/5` passed on CPU)
- Connected-Component Split Ledger, Complete Exclusion Manifest & Cross-Corpus Overlap Audit: [gate2_split_ledger_receipt.json](../../reports/v3/gate2_split_ledger_receipt.json) ([v3_gate2_split_ledger.py](../../scripts/data/v3_gate2_split_ledger.py))
- Strictly-Dev Causal Codec Ceilings, Streaming Equivalence, `98 s` Counterfactual Leak Audit & Dual-Track Accounting: [gate2_codec_ceilings_receipt.json](../../reports/v3/gate2_codec_ceilings_receipt.json) ([v3_gate2_codec_ceilings.py](../../scripts/data/v3_gate2_codec_ceilings.py))

---

## 1. Connected-Component Partition Coverage & Cross-Corpus Disjointness

All `8` Wave A corpora are verified present (`expected_corpora_all_present = True`), partitioned by their connected-component identity keys with deterministic SHA-256 split digests (`split_manifest_sha256` recorded for every corpus), and backed by complete non-truncated exclusion ledgers ([gate2_split_ledger_receipt.json](../../reports/v3/gate2_split_ledger_receipt.json)):

| Corpus | Modalities Present (Zero Fabrication) | Connected Split Key | Partition Coverage (`train / dev / test` or `vault`) | Overlap Check | Exclusion & Auxiliary Split Policy |
|---|---|---|---|---|---|
| `SpeechCommands v0.02` | Audio (`16 kHz`), Text label | `speaker_id` (+ disjoint `_background_noise_`) | `2,112 / 256 / 250` speakers (`84,843 / 9,981 / 11,005` wavs); bg noise `4 / 1 / 1` disjoint clips | `0` speaker & `0` bg overlap (`zero_overlap=True`) | Disjoint ambient beds: `train` (`doing_the_dishes`, `dude_miaowing`, `exercise_bike`, `pink_noise`), `dev` (`running_tap`), `test` (`white_noise`) |
| `HoloAssist` (local `-av.mp4`) | Video (`896×504`), Audio (`48 kHz`), Timed text | `session_performer_prefix` (casefold; **not instructor-disjoint**) | `457 / 86 / 129` recs (`196 / 39 / 47` prefixes; `31.18 / 6.14 / 9.02 h = 46.34 h`) | `0` overlap (`zero_overlap=True`) | All `48` duplicate file copies listed completely; `0` unmatched recordings |
| `Physical xArm6 v2` | Video (`64×64`), Paired speech audio, Text, Action | `connected_scene_group` & speaker | `1,745 / 351 / 284` eps (`763 / 151 / 135` unique scene clusters verified in receipt) | `15/15` checks `== 0` (`zero_overlap=True`) | `22,412 / 24,792` upstream rows logged by reason + `wave8a_episode_dispositions.jsonl.gz` SHA-256 |
| `GRID` (`grid_content_v1`) | Video (mouth), Audio (`16 kHz`), Aligned text | `speaker_id` | `12,704 / 395 / 278 / 749` (`16 train / 4 val / 3 cal / 8 test`) | `0` overlap across primary splits (`zero_overlap=True`) | Dynamically computed absent speakers `GRID-21, 33, 34` + `90` short clips + SHA-256; auxiliary `train_person_holdout` & `test_seen` excluded from speaker-disjoint eval |
| `GreatestHits` (`vis-data-256`) | Paired Video (`_denoised.mp4`), Impact Audio (`96 kHz`) | `recording_date` (`YYYY-MM-DD`) | `512 / 232 / 233` clips (`16 / 6 / 6` dates; `977/977` verified paired video+audio) | `0` overlap (`zero_overlap=True`) | `0` dropped clips |
| `FineART` (ALOHA) | Video (`cam_high`), Subtask text, `14-DoF` action | `cam_high_video_file_shard` | `116 / 37 / 27` shards (`180` `.mp4` files under `cam_high`; `2` data parquets, `1` episode-meta parquet, verified `14-DoF` action) | `0` overlap (`zero_overlap=True`) | Single camera view (`cam_high`) scoped explicitly; `0` dropped shards |
| `LibriSpeech 360` | Audio (`16 kHz`), Text | `speaker_id` (train-only, disjoint from held speakers) | `665` eligible speakers / `75,139` utterances (SHA-256 over sorted `665` speakers + card SHA-256) | `zero_overlap=True` | `21,678` existing-train overlap + `7,197` held-speaker utts excluded; `libri360-eligible.jsonl.gz` SHA-256 logged |
| `FineWeb-Edu v1` | English text | `parquet_shard_disjoint` | `3 train / 1 dev / 1 test` parquet shards (`000..002 / 003 / 004`) | `0` overlap (`zero_overlap=True`) | `0` dropped shards |

### Cross-Corpus Connected Speaker Purity (`Physical xArm6 v2` × `SpeechCommands v0.02` × `GRID`)
Because `Physical xArm6 v2` pairs robot demonstrations with spoken direction words (`SpeechCommands v0.02` speakers) and spoken color words (`GRID` speakers), Gate 2 verifies all `9/9` cross-corpus cross-split intersections:
- **`xArm6[train]`** (`902` dir speakers, `GRID 1..16`) has **`0`** speakers in `SpeechCommands[dev]`, **`0`** in `SpeechCommands[test]`, and **`0`** in `GRID[val/cal/test]`.
- **`xArm6[dev]`** (`156` dir speakers, `GRID 17..20`) has **`0`** speakers in `SpeechCommands[train/test]` and **`0`** in `GRID[train/cal/test]`.
- **`xArm6[vault]`** (`135` dir speakers, `GRID 33, 34`) has **`0`** speakers in `SpeechCommands[train/dev]` and **`0`** in `GRID[train/val/cal/test]`.
- **`cross_corpus_zero_leak_verified = True`**, **`all_connected_splits_zero_overlap = True`**.

---

## 2. Hardened Causal Alignment, Collision-Free Overwrite Schedules & Pixel-Observable Renderer

Verified in [v3_data_contract.py](../../data/v3_data_contract.py) and `5/5` passing unit tests in [test_v3_gate2_data_contract.py](../../tests/test_v3_gate2_data_contract.py):

1. **Strict Causal Alignment & Finite Guards (`causal_align_past_index`, `CausalPacketTimestamps`):** Rejects `NaN`/`Inf` and unsorted arrays, selects the latest source timestamp strictly `<= query_ts` (`-1` pre-start), and enforces `capture_time_s <= last_sample_time_s <= availability_time_s` for 4-frame `Wan 2.1 VAE` video groups (`25 fps`) and `80 ms` `Kyutai Mimi` audio packets (`12.5 Hz`).
2. **Collision-Free Interleaved `Real-Speech-Overwrite` Schedules Exceeding BOTH `sliding_window_gqa_eqbyte` AND `Kyutai Mimi`'s `79.68 s` Stacked RF (`N = 1,200` each):**
   - **Zero speech collisions:** Enforces `>= 1.15 s` minimum separation across all spoken events (`pre`, `init1`, `mid1`, `init2`, `mid2`, `corr`, `post`), eliminating `+3 dB` double-talk energy spikes (`min_inter_event_gap_s >= 1.1516 s`, `zero_speech_collisions_verified = True`).
   - **Indistinguishable `0`-correction catch trials (`~25%`):** Places a same-domain distractor at `t_corr_s` (`catch_distractor_at_corr_s`) so catch and overwrite episodes share identical inter-word timing distributions (eliminating any silence hole).
   - **Constant-power randomized background tiling:** `tile_background_with_crossfade` applies a randomized circular phase shift via `rng` across all durations (including `106 s–193 s` streams) and equal-power $\sin/\cos$ crossfades ($w_{\text{in}}^2 + w_{\text{out}}^2 = 1.0$).
   - **Implemented cluster key (`assign_episode_primary_speaker_id`):** Anchors `corr_speaker_id` on overwrite episodes and `init1_speaker_id` on `0`-correction catch trials.
   - **Registered beyond-eqbyte & beyond-Mimi-RF horizons:**
     - **Discovery (`beyond-eqbyte + beyond-Mimi-RF`, `L = 12, W_eq = 64`, RF `= 756` ticks; Mimi RF `= 996` ticks / `79.68 s`):** `total = 106 s`, `min_delay = 1,154 > max(756, 996)`, `min_init_to_corr = 72 > 64`, `min_gap = 1.1516 s`.
     - **Confirmation (`beyond-eqparam + beyond-Mimi-RF`, `L = 16, W = 64`, RF `= 1,008` ticks):** `total = 106 s`, `min_delay = 1,153 > 1,008`, `min_init_to_corr = 72 > 64`, `min_gap = 1.1525 s`.
     - **Confirmation (`beyond-eqbyte`, `L = 16, W_eq = 128`, RF `= 2,032` ticks):** `total = 193 s`, `min_delay = 2,207 > 2,032`, `min_init_to_corr = 136 > 128`, `min_gap = 1.1532 s`.
3. **Pixel-Observable Conveyor Renderer (`render_v3_dynamics_world_rgb`):** Renders neutral charcoal/silver-gray directional chevrons inside `|x| < 0.42` scrolling at `v_belt = 0.055 * sign(drift_y)` with apex leading in `sign(drift_y)`, maintaining high chroma contrast against the vivid cyan puck (`> 180` blue-minus-red separation).

---

## 3. Small-Shard Causal Codec Causality, Streaming Equivalence & Ceilings (Strictly `dev`-Only)

All benchmarks ran strictly over `dev`/`val` slices (`0` sealed `test`/`vault` items touched; [gate2_codec_ceilings_receipt.json](../../reports/v3/gate2_codec_ceilings_receipt.json)):

### 3.1 Audio Edge: `Kyutai Mimi 32-RVQ` (`12.5 Hz`, `4.4 kbps`) — Streaming Verified & `98 s` Counterfactual Leak Audited

- **Raw-PCM suffix invariance (`t_cut = 2.0 s`):** `mimi_32rvq_25frames_exact = True` (`diff_frac = 0.0`); `encodec_150frames_exact = True` (`diff_frac = 0.0`).
- **Native `use_streaming=True` online == offline equivalence (`24 s` = `600` Transformer frames `> 250` sliding window):**
  - `mimi_24s_600tf_online_vs_offline_exact = True` (`diff_frac = 0.0`).
  - Live `encoder_past_key_values` sequence length across 6 chunks: **`[100, 200, 249, 249, 249, 249]`** (`mimi_sliding_window_o1_capped_at_249 = True`), proving strict $\mathcal{O}(1)$ bounded attention memory.
- **Empirical `98 s` real-speech counterfactual code-space leak audit (`SpeechCommands val` `"left"` vs. `"right"` at `[10 s, 11 s]` over `dev` background `running_tap.wav`):**
  - `Mimi`'s 8-layer causal Transformer (`sliding_window = 250` @ `25 Hz`) has a theoretical stacked causal RF of `8 × 249 / 25 = 79.68 s` (`996` ticks @ `12.5 Hz`).
  - Observed `RVQ-32` / `RVQ-0 (semantic)` code mismatch decay after `t_corr_end`: `+0 s: 79.25% / 20.0%` → `+15 s: 35.88% / 4.0%` → `+38 s: 4.38% / 4.0%` → `+50 s: 0.25% / 4.0%` → **`+65 s: 0.00% / 0.00%`** → **`+81 s (> 79.68 s RF): 0.00% / 0.00%`** (`mimi_counterfactual_zero_leak_beyond_79p68s_rf = True`).
  - **Scientific consequence:** Had Discovery remained at `38 s` delay, `4.4%` frozen-codec leakage would have confounded counterfactual post-cue invariance. Locking Discovery & Confirmation `Real-Speech-Overwrite` delays strictly **`>= 1,153` ticks (`>= 92.2 s > 79.68 s`)** guarantees **bit-exact `0.000000` counterfactual code invariance** at the query window while simultaneously evicting `sliding_window_gqa_eqbyte`.

| Audio Edge Codec | Packet Rate | Speech Keyword Ceiling (`SC val n=40`, `whisper-tiny.en`, no `"tree"` alias) | Speech `SI-SNR` / `MR-STFT` | Stratified Impact Onset Corr (`GH dev n=11`, all `6` dev dates, `20 ms`) | Impact Peak Timing Error (`median / mean`) |
|---|---:|---:|---:|---:|---:|
| Raw `16/24 kHz` PCM (upper bound) | — | `97.5%` (`39/40`) | `— / 0.000` | `1.0000` | `0.0 / 0.0 ms` |
| **`Kyutai Mimi 32-RVQ` (`4.4 kbps`, Selected)** | **`12.5 Hz`** | **`95.0%` (`38/40`)** | **`+7.62 dB / 0.481`** | **`0.9976`** | **`0.0 / 92.7 ms`** |
| `Kyutai Mimi 8-RVQ` (`1.1 kbps`) | `12.5 Hz` | `72.5%` (`29/40`) | `-2.53 dB / 0.522` | `0.9917` | `0.0 / 298.2 ms` |
| `EnCodec 24 kHz 6 kbps` | `75.0 Hz` | `85.0%` (`34/40`) | `+6.16 dB / 0.606` | `0.9976` | `0.0 / 50.9 ms` |
| `10 Hz` Non-Overlapping Hann STFT + GL | `10.0 Hz` | `10.0%` (`4/40`) | `-26.36 dB / 1.075` | `0.2395` | `40.0 / 305.5 ms` |

### 3.2 Video Edge: `Wan 2.1 Causal VAE` (`192×320`, `25 fps`, `0%` Black Bars, `3×5 = 15` Isotropic `8×8` Patches)

- **Unified resolution & aspect policy:** Scaling `896×504` (`16:9`) egocentric video to `341×192` with a minimal `6%` horizontal center crop to **`192×320`** yields **`0.0%` black letterbox bars**, preserves `94%` horizontal / `100%` vertical egocentric FOV, and produces an exact `24×40×16` latent grid partitioned into **`3 × 5 = 15` isotropic `8×8` linear spatial patches** (`1,024` dims/patch) → **`93.75` video pos/s + `12.5` Mimi pos/s = `106.25` paired AV pos/s (`382,500` pos/hr)**. *(Note: under `106.25` paired AV pos/s, `W = 32` spans `0.30 s` and `L*(W-1) = 372` spans `3.50 s`, updating Gate 1's `10 Hz` reference span while preserving `HoloAssist-DC`'s scoping as a secondary ecological check.)*
- **Causality, per-chunk `quant_conv` streaming equivalence & exact $\mathcal{O}(1)$ cache constancy (`9` frames = `1+4+4`):**
  - `prefix_latent_0_1_max_abs_diff_fp32 = 0.0` (`suffix_latent_2_max_abs_diff_fp32 = 1.9409`).
  - `online_per_chunk_quant_conv_vs_offline_max_abs_diff = 0.0` (`online_equals_offline = True`).
  - `feat_cache_bytes_chunk1_fp32 = 345,047,040 == feat_cache_bytes_chunk2_fp32 = 345,047,040` (`exact_o1_constant = True`; **`164.531 MiB` bf16 / `329.062 MiB` fp32**).
- **Strictly `dev`-only round-trip ceilings using true `11×11` Gaussian-windowed local SSIM (`sigma = 1.5`):**
  - `HoloAssist dev` (`R043-13July-BigPrinter`, `R055-13July-Navvis`, `R090-27July-Circuit-Breaker`, `25 fps`, `9` frames, `0%` black bars, `fp32`): **`27.44 dB` PSNR**, **`0.8873` local `11×11` SSIM**, `0.05245` edge RMSE (`bf16` latent quantization check on `R043`: `30.31 dB` PSNR, `0.9446` local SSIM; full native `bf16` encoder execution will be verified on native TPU `v6e-1` BF16 hardware in Phase 1).
  - `Physical xArm6 dev` (`dev_video`, `n = 4`, `5` frames): **`39.68 dB` PSNR**, **`0.9814` local `11×11` SSIM**, **`frame 0` (`1×` temporal) color-block XY drift `= 0.00451`**, **`frames 1..4` (`4×` temporal compression) color-block XY drift `= 0.00698`** normalized table units (>2× tighter than the `0.015` RMSE non-regression gate).
  - `V3 Dynamics` (`n = 8` distinct worlds, `5` frames): **`28.97 dB` PSNR**, **`0.9640` local `11×11` SSIM**, **`conveyor_drift_direction_preserved_acc = 1.000` (`8/8`)**, **`puck XY drift = 0.00819`**, **`end-effector XY drift = 0.00148`**.

### 3.3 Text/Code Edge: Lossless `ByT5` `256`-Byte UTF-8 Interface
- Audited across `150` real `HoloAssist dev` transcripts + `50` Python lines (`6,951` UTF-8 bytes): **`100.0%` exact lossless round-trip**, `0.0%` OOV, `264` total IDs (`0..255` UTF-8 + `8` controls), `0` frozen params, `0` streaming state bytes.

---

## 4. Exact Dual-Track Cost & Provenance Accounting (`AGENTS.md §4.1`, Exact Binary Units)

All parameter partitions are verified exact (`encoder + decoder == total` asserted programmatically for both `Wan 2.1 VAE` and `Kyutai Mimi`), and streaming state footprints are measured directly from live cache tensors and reported in **both `bf16` and `fp32`**:

| Component | Accounting Track | Trainable Params | Frozen Params | Live Streaming State (`bf16` / `fp32`) | Storage (`per AV hr`, binary) | License & Provenance |
|---|---|---:|---:|---:|---:|---|
| `Wan 2.1 Causal VAE` Encoder + `quant_conv` (`192×320`) | Strict System (frozen encoder) | `0` | `53,596,928` | **`164.531 MiB` / `329.062 MiB`** (`feat_cache`, exact $\mathcal{O}(1)$) | `0.644 GiB` (`bf16`) | `apache-2.0`; open Wan 2.1 pretraining (benchmark overlap `unknown`) |
| `Wan 2.1 Causal VAE` Decoder + `post_quant_conv` (eval/gen only) | Strict System (frozen decoder) | `0` | `73,295,603` | — (diagnostic/gen only) | — | `apache-2.0` |
| `Kyutai Mimi` Encoder + `encoder_transformer` + `downsample` + `quantizer` (`32-RVQ`) | Strict System (frozen encoder) | `0` | `39,391,520` | **`3.913 MiB` / `7.825 MiB`** (`2,039,808` KV + `11,526` conv elems) | `2.75 MiB` (`int16`) | `cc-by-4.0`; Kyutai Moshi 7M-hr audio pretraining (overlap `unknown`) |
| `Kyutai Mimi` `upsample` + `decoder_transformer` + Decoder (eval/gen only) | Strict System (frozen decoder) | `0` | `39,917,089` | — (diagnostic/gen only) | — | `cc-by-4.0` |
| `ByT5` UTF-8 Byte Edge | Both | `0` | `0` | `0.000 MiB / 0.000 MiB` | `< 0.50 MiB` | Lossless deterministic UTF-8 |
| **Total Frozen Inference-Active Encoders** | **Strict System Edge Encoders** | **`0`** | **`92,988,448`** | **`168.444 MiB` / `336.887 MiB`** (strict $\mathcal{O}(1)$) | **`0.647 GiB/hr`** | Controlled via Discovery `2×2` `Backbone × Representation` cross |
| **Total Frozen Encoders + Decoders (Exact Sum)** | **Strict System Full Codecs** | **`0`** | **`206,201,140`** | — | — | `92,988,448 + 113,212,692 == 206,201,140` (`exact=True`) |
| **`SLWM-v3` Discovery Core (`L=12, W=32`)** | **Core Comparison** | **`~40.5M`** (verified Gate 3) | **`0`** | **`0.375 MiB`** (`192 KiB` fp32 matrix + `192 KiB` bf16 ring KV) | — | Trained from scratch on Gate 2 splits |
| **`SLWM-v3` Confirmation Core (`L=16, W=64`)** | **Core Comparison** | **`~130M`** (verified Gate 3) | **`0`** | **`2.001 MiB`** (`1.0 MiB` fp32 matrix + `1.0 MiB` bf16 ring KV) | — | Trained from scratch on Gate 2 splits |

**`Backbone × Representation` Control Guarantee:** At Discovery scale, both `SLWM-v3 Routed` and `SlidingWindow-GQA` will be evaluated across **both** `Frozen Causal Codec Edge` (`92.99M` frozen encoder params, `168.44 MiB` bf16 / `336.89 MiB` fp32 streaming state) **and** `Train-Only Scratch Edge` (`0` frozen params, `0` upstream pretraining) so representation gains are strictly separated from sequence-core memory gains.
