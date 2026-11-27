# `SLWM-v4` Gate 1 — Hypothesis, Physical Observability, Closed-Loop Oracle Ceilings & `160 ms` Live Heartbeat Specification

**Status:** Gate 1 Hardening & Pre-Registration Pack (ready for independent Gate 1 peer review prior to Gate 2 data engineering)  
**Branch:** `research/slwm-v4-humanoid-world-model`  
**Verified Receipts & Unit Tests:**
- Hardened Anti-Knockoff Matched-Perturbation Closed-Loop MuJoCo ALOHA Oracle Ceilings (`AlohaTransferCube-v0` + `AlohaInsertion-v0`, `N = 32` seeds × `5` arms each + `8`-seed RGB counterfactual audits): [gate1_closed_loop_robotics_receipt.json](../../reports/v4/gate1_closed_loop_robotics_receipt.json) ([v4_gate1_closed_loop_oracle.py](../../evals/v4_gate1_closed_loop_oracle.py))
- Strictly Causal `160 ms` Interleaved Heartbeat Clock Lock, Non-Aliased `52`/`68` Humanoid Schema, Seam-Free AV Desync & Programmatically Derived Two-Tier Single-Pass Corpus Receipt: [gate1_heartbeat_and_corpus_receipt.json](../../reports/v4/gate1_heartbeat_and_corpus_receipt.json) ([v4_heartbeat_contract.py](../../data/v4_heartbeat_contract.py))
- Contract Unit Tests (`5/5 passed`): [test_v4_gate1_contract.py](../../tests/unit/test_v4_gate1_contract.py)

---

## 1. Scientific Motivation & Falsifiable Research Questions (One Primary Question Per Wave)

`SLWM-v3` demonstrated that bounded-memory ($\mathcal{O}(1)$) recurrent latent cores outperform equal-byte `Sliding-Window GQA` on language modeling, physical impact `Video -> Audio`, robotics next-video Huber, and `167 s` selective overwrite, but exposed six structural bottlenecks blocking humanoid world-model capability:
1. Small multimodal shards looped `4–5×` while `Text & Code` consumed `85.3–86.5%` of training positions.
2. Non-interleaved `Video -> Audio` serialization broke sub-second acoustic onset synchronization (`env_r ≈ 0`) and prevented live simultaneous `Audio + Video` streaming.
3. Deterministic Huber video regression collapsed under-determined `audio2video`/`text2video` into chromatic fog (`12.1–12.7 dB`) and blurred gripper contacts, while flat independent `8-RVQ` softmax suffered `47–100%` `RVQ-0` repetition collapse.
4. Per-layer dual-branch GQA+Memory and `M_anchor` state doubling made `slwm_v3_routed` `2.1–3.0×` slower and `1.5–1.67×` heavier in state bytes than `single_bank_gated_deltanet` without a decisive win at `703M`.
5. `703M` (`0.28` pos/param) and `1.072B` (`0.18` pos/param) were severely under-exposed (`0.001` robotics pos/param at `1.072B`).
6. Teacher-forced Huber/PSNR lacked closed-loop robotics success evaluation, SOTA AV synchrony metrics (`DeSync`, `FAD`, `FVD`, `20 ms Onset F1`), and `2+ minute` live rollout stability tests.

### Pre-Registered Primary Research Questions (Zero Sprint Drift)

- **Wave V4-S3 (`45M` Core Architecture & Objective Discovery Sweep, `600M` single-pass pos = `13.3` pos/param on Tier 1 `438.88 GB` `fp16` persistent pool, `max_epochs = 0.9728 <= 1.00`, `75.0%` multimodal signal):**
  > Under a strictly causal `160 ms` live interleaved AV+Motor+Text clock, shared generative heads across all compared backbones, and `<= 1.0` epoch single-pass exposure, which single-factor architectural and objective changes—isolated one at a time in Section 5—maximize physical cross-modal synchrony, `2+ min` live rollout stability, and closed-loop robotics utility at matched core parameters and streaming state bytes?
- **Wave V4-S4 (`150M` Full Humanoid World Model Confirmation, `3.0B` single-pass pos = `20.0` Chinchilla pos/param on Tier 2 streaming/`int8` pool, `max_epochs = 1.0000`, `71.99%` multimodal signal, `n = 3` paired seeds):**
  > Does the locked `SLWM-v4` winner outperform matched `3:1 Single-Bank Gated DeltaNet`, equal-byte `Sliding-Window GQA`, and `SLWM-v3` (all with identical generative heads) simultaneously on (1) live synchronized `Audio <-> Video` (`DeSync`, `FAD`, `20 ms Onset F1`), (2) `2+ minute` (`>= 120 s` / `750` heartbeats) sustainable live AV+Action streaming (`FVD`/`LPIPS`/`DINOv2` drift and `> 30 s` object-permanence re-ID at `< 160 ms` per-heartbeat latency on A100/H100), and (3) closed-loop robotics (`AlohaTransferCube-v0` & `AlohaInsertion-v0` policy/MPC task success % and `DROID` success-prediction AUROC / policy ranking `Kendall's τ`), without regressing on language/code (`WikiText-103`, `LAMBADA`) or calibrated 5-action `PolicyCommitGate` (`ECE < 0.05` with non-degenerate accuracy & abstention)?
- **Wave V4-S5 (`700M -> 1.3B` Compute-Optimal Scale-Up, `>= 15B–26B` single-pass pos via distributed multi-TB streaming shards):**
  > Authorized **only** after Wave V4-S4 passes Gate 4. Explicitly falsifiable under the quantitative retirement margin in Section 5.

---

## 2. Strictly Causal `160 ms` Live Interleaved Physical Heartbeat & Non-Aliased `52`/`68` Humanoid Contract

Certified by [test_v4_gate1_contract.py](../../tests/unit/test_v4_gate1_contract.py) (`5/5` passed) and [gate1_heartbeat_and_corpus_receipt.json](../../reports/v4/gate1_heartbeat_and_corpus_receipt.json):

- **Strict Physical Availability Timestamp Chain (`capture_time_s <= last_sample_time_s <= wall_time_s == availability_time_s`):**
  Matches `data/v3_data_contract.py`'s Wan 2.1 `1 + 4k` and Kyutai Mimi end-of-window availability semantics with **zero `np.maximum.accumulate` warping** (raw non-monotonicity raises `AssertionError` loudly):
  - **Bootstrap Frame 0 (`heartbeat_id = -1`, optional Wan 2.1 `1 + 4k` anchor):** `15` patches of RGB frame `0` at `capture = last = avail = 0.00 s`.
  - **Inside every `160 ms` (`6.25 Hz`) heartbeat $h$ spanning $(t_0, t_0 + 0.16]$ ($t_0 = 0.16 h$):**
    1. $t_0 + 0.04\text{ s}$: `MOT[4h+0]` (`dt_s = 0.04 s`; sees strictly $t \le t_0$, zero visual/acoustic future lookahead).
    2. $t_0 + 0.08\text{ s}$: `MOT[4h+1]` (`dt_s = 0.04 s`), then `AUD[2h+0]` (Mimi packet covering $[t_0, t_0+0.08]$, `dt_s = 0.00 s`).
    3. $t_0 + 0.12\text{ s}$: `MOT[4h+2]` (`dt_s = 0.04 s`; sees `AUD[2h+0]`, never future `AUD[2h+1]` or `VID[h]`).
    4. $t_0 + 0.16\text{ s}$: `MOT[4h+3]` (`dt_s = 0.04 s`, emitted strictly **before** `VID[h]` and `AUD[2h+1]` so no motor tick ever peeks at its own visual/acoustic consequence).
    5. $t_0 + 0.16\text{ s}$: `VID[h]` (`15` Wan 2.1 `8×8×16` patches at `192×320` covering frames `[4h+1..4h+4]`, `capture = t_0 + 0.04`, `avail = t_0 + 0.16`, `dt_s = 0.00 s`).
    6. $t_0 + 0.16\text{ s}$: `AUD[2h+1]` (Mimi `12.5 Hz`, `K = 16` RVQ codebooks stored covering $[t_0+0.08, t_0+0.16]$; causally attends to `VID[h]` and `MOT[0..3]` for `0 ms`-lag impact audio synthesis).
    7. $t_0 + 0.16\text{ s}$: `TXT[h]` (lossless UTF-8 bytes, `264` vocab, whose physical arrival time $\le t_0 + 0.16$, `dt_s = 0.00 s`, preserving exact `dt_s = 0.04 s` into heartbeat $h+1$).
- **Non-Aliased `52`-DoF Action / `68`-Dim Proprio+Force Schema (`validate_disjoint_embodiment_slices()`):**
  Disjoint contiguous slices eliminate both truncation of real `45`-DoF Unitree G1 WBT+Inspire / `40`-DoF Unitree H1 humanoids and cross-embodiment channel aliasing:
  - `ACTION_SLICES (52)`: `left_arm[0:7]`, `right_arm[7:14]`, `left_hand_or_gripper[14:21]`, `right_hand_or_gripper[21:28]`, `left_leg[28:34]`, `right_leg[34:40]`, `waist[40:43]`, `mobile_base_twist[43:47]`, `head_and_aux[47:52]`.
  - `PROPRIO_SLICES (68)`: `joint_state[0:52]` (1:1 mirror), `left_wrist_force_torque[52:58]`, `right_wrist_force_torque[58:64]`, `torso_imu_base_orient[64:68]`.
  - Both `action_mask` and `proprio_mask` are validated strictly binary `{0.0, 1.0}` tensors passed **both** as motor-encoder input conditioning (disambiguating active `0.0` joint angle from absent DoF) and as per-channel loss masks.
- **Zero-Shortcut Seam-Free Same-Clip AV Desync Pair (`make_same_clip_temporal_desync_pair`):**
  Replaces circular `np.roll` (which introduces an audio-only splice seam) with **strictly interior contiguous window cropping** (`video[s0:s0+L]` paired with `audio[s0+Δ:s0+Δ+L]`, verified `np.all(np.diff(src_ticks) == 1)`) while strictly stripping `MOD_TEXT` (zero audio-text shortcut) and holding `reliability == 1.0`, `valid_mask`, `wall_time_s`, and `dt_s` identical. Gate 2/3 will verify an audio-only classifier scores at chance (`AUROC ≈ 0.50`) on clean vs. desync pairs.
- **`2+ Minute` Sustainable Live Streaming Contract:** `120.0 s` = exactly `750` heartbeats = `3,000` video frames + `1,500` audio ticks (+ `3,000` motor ticks) = `12,750` AV packets (`15,750` AV+Motor packets).

---

## 3. Real-Physics Closed-Loop Robotics Observability & Matched-Perturbation Oracle Ceilings (`ALOHA-Sim` MuJoCo)

[v4_gate1_closed_loop_oracle.py](../../evals/v4_gate1_closed_loop_oracle.py) evaluates **BOTH `AlohaTransferCube-v0` AND `AlohaInsertion-v0`** across `N = 32` unseen randomized MuJoCo seeds (`25 Hz` control ticks, `6.4 s` horizon) across **5 decoupled arms** using one shared controller helper, strict anti-knockoff physical stage gates (`>= 15` consecutive ticks = `0.60 s` sustained mid-air), and **unconfounded matched-perturbation actuation drift** (`0.15 rad` applied strictly to the 12 radian arm joints clipped to `model.jnt_range`; grippers `[6, 13]` untouched) ([gate1_closed_loop_robotics_receipt.json](../../reports/v4/gate1_closed_loop_robotics_receipt.json)):

### 3.1 `AlohaTransferCube-v0` (Anti-Knockoff 3-Stage Analytical IK Ceiling, `Kendall's τ = 1.0000`, `env_gate1_pass = True`)

| Decoupled Control Arm (`N = 32` Unseen Seeds, `25 Hz`) | Stage 1: Touch (`z>=1.5cm`) | **Stage 2: Sustained Mid-Air Lift (`>=0.6s, z>=14cm`)** | **Stage 3: Sustained Goal Transport (`>=0.6s, z>=16cm, <=5.5cm`)** | **3-Stage Physical Score** | Min Pinch Dist (`m`) | Mean Max Box `z` (`m`) |
|---|---:|---:|---:|---:|---:|---:|
| **`Oracle-Perc × Oracle-Dyn Closed-Loop` (`0.00 rad` drift)** | **`100.0%`** | **`100.0%`** | **`96.88%`** | **`0.9896`** | **`0.0072`** | **`0.2449`** |
| **`Oracle-Perc × Oracle-Dyn Closed-Loop` (`0.15 rad` matched drift)** | **`100.0%`** | **`93.75%`** | **`93.75%`** | **`0.9583`** | **`0.0079`** | **`0.2306`** |
| `Noisy-Perc (3 cm) × Oracle-Dyn Closed-Loop` (`0.00 rad` drift) | `93.75%` | `31.25%` | `31.25%` | `0.5208` | `0.0431` | `0.1181` |
| `Oracle-Perc × Open-Loop Nominal` (`0.15 rad` matched drift) | `59.38%` | `9.38%` | `3.12%` | `0.2396` | `0.0640` | `0.0705` |
| `Hold-Pose Null Baseline` | `0.00%` | `0.00%` | `0.00%` | `0.0000` | `0.3010` | `0.0500` |

- **Matched-Perturbation Closed-Loop Feedback Gain (`0.15 rad` arm-joint drift):** **`+0.7187`** (`0.9583` closed-loop vs. `0.2396` open-loop).
- **In-Distribution Counterfactual Audit (`N = 8`, `50` ticks, `192×320`):** `0.0000%` joint-limit violations, `0.5222 m` bimanual pinch divergence, `9.894 / 255` RGB L1.

### 3.2 `AlohaInsertion-v0` (Anti-Knockoff 3-Stage Dual-Arm Ceiling, `Kendall's τ = 1.0000`, `env_gate1_pass = True`)

| Decoupled Control Arm (`N = 32` Unseen Seeds, `25 Hz`) | Stage 1: Dual Touch (`peg>=0.6cm, soc>=1.5cm`) | **Stage 2: Sustained Dual Lift (`>=0.6s, both z>=14cm`)** | **Stage 3: Sustained Pre-Dock Align (`>=0.6s, <=6.5cm`)** | **3-Stage Physical Score** | Min Pinch Dist (`m`) | Mean Max Min-Obj `z` (`m`) |
|---|---:|---:|---:|---:|---:|---:|
| **`Oracle-Perc × Oracle-Dyn Closed-Loop` (`0.00 rad` drift)** | **`100.0%`** | **`96.88%`** | **`87.50%`** | **`0.9479`** | **`0.0325`** | **`0.2190`** |
| **`Oracle-Perc × Oracle-Dyn Closed-Loop` (`0.15 rad` matched drift)** | **`84.38%`** | **`31.25%`** | **`25.00%`** | **`0.4688`** | **`0.0336`** | **`0.1145`** |
| `Noisy-Perc (3 cm) × Oracle-Dyn Closed-Loop` (`0.00 rad` drift) | `53.12%` | `6.25%` | `3.12%` | `0.2083` | `0.0502` | `0.0617` |
| `Oracle-Perc × Open-Loop Nominal` (`0.15 rad` matched drift) | `25.00%` | `3.12%` | `3.12%` | `0.1042` | `0.0736` | `0.0563` |
| `Hold-Pose Null Baseline` | `0.00%` | `0.00%` | `0.00%` | `0.0000` | `0.2769` | `0.0500` |

- **Matched-Perturbation Closed-Loop Feedback Gain (`0.15 rad` arm-joint drift):** **`+0.3646`** (`0.4688` closed-loop vs. `0.1042` open-loop).
- **In-Distribution Counterfactual Audit (`N = 8`, `50` ticks, `192×320`):** `0.0000%` joint-limit violations, `0.5086 m` bimanual pinch divergence, `9.728 / 255` RGB L1.
- **Explicit Scope Honesty:** Analytical 3-stage IK ceilings score Stages 1–3 (`/ 3.0`); full Stage-4 mid-air handover (`TransferCube reward == 4`) and pin insertion (`Insertion reward == 4`) are deferred to learned-policy / human-demo evaluation (`lerobot/aloha_sim_*_human`). **`LIBERO` closed-loop evaluation is deferred** until a dedicated Gate 1 `LIBERO` oracle addendum is executed and is excluded from Gate 4 mandatory closed-loop floors.

---

## 4. Programmatically Derived Single-Pass (`<= 1.0` Epoch) Two-Tier Corpus Plan

Every packet count and uncompressed storage footprint below is **computed programmatically by exact formula** from `target_hours`, `modalities`, `text_bytes_per_sec`, and `SLWMv4HeartbeatSpec` (`22,500 hb/hr`; `fp16` video = `0.70335 GB/hr` raw + 5% shard overhead; `int8` affine video = `0.35910 GB/hr` + 5% overhead; Mimi `16-RVQ` = `0.00306 GB/hr`; `52`/`68` motor+proprio = `0.04644 GB/hr`), enforced by unit test against `verified_source_hours` from live HuggingFace API inspection ([gate1_heartbeat_and_corpus_receipt.json](../../reports/v4/gate1_heartbeat_and_corpus_receipt.json)):

### 4.1 Tier 1 Persistent `fp16` Pool (`V4-S3` `45M` Discovery Sweep, `600M` Single-Pass Pos = `13.3` Pos/Param, `438.88 GB` Total)

| Corpus Suite | Verified Open Source(s) & License | Verified Source Hrs | Tier 1 Hrs | Exact Pool Pkts (`M`) | `fp16` Size (`GB`) | `V4-S3 (600M)` Alloc (`M`) / Epochs / Share |
|---|---|---:|---:|---:|---:|---:|
| **Unitree G1/H1 Humanoid** | 12 curated `unitreerobotics/G1_*` + `lerobot/unitreeh1_fold_clothes` (`apache-2.0 / unitree-open`) | `108.67 h` | `60.0 h` | `26.08` | `47.24` | `25.20` / `0.9662` / `4.20%` |
| **DROID 1.0.1** | `lerobot/droid_1.0.1` (`95,658` eps, `15 fps`, `apache-2.0`) | `511.67 h` | `115.0 h` | `49.99` | `90.54` | `45.00` / `0.9002` / `7.50%` |
| **RH20T Contact AV+Force**\* | `tonyfanggalaxies/RH20T` (`cc-by-4.0`; `audio_mixed/*.wav` + `force_torque.npy` verified in `cfg1..7`) | `>= 50.0 h` | `45.0 h` | `21.59` | `35.57` | `21.00` / `0.9728` / `3.50%` |
| **AgiBot + ALOHA-Sim** | `agibot-world/AgiBotWorld2026` (`cc-by-nc-sa-4.0`, `30 fps`), `aloha_sim_*` (`apache-2.0`, `50 fps`) | `>= 1,000 h` | `75.0 h` | `32.60` | `59.05` | `28.80` / `0.8834` / `4.80%` |
| **Full VGGSound Sync** | `Loie/VGGSound` (`199,467` clips, `25 fps`, `cc-by-4.0`) | `554.10 h` | `150.0 h` | `58.99` | `111.26` | `55.20` / `0.9357` / `9.20%` |
| **EPIC-KITCHENS-100 Ego AV** | `a1raman/epic_kitchens_100` (`470` videos, `cc-by-nc-4.0`) + official GitHub `epic-kitchens-100-annotations` | `65.00 h` | `55.0 h` | `22.23` | `40.80` | `21.60` / `0.9719` / `3.60%` |
| **HoloAssist + GreatestHits** | `SitongGong/HoloAssist` (`cc-by-4.0`) + MIT CSAIL `GreatestHits` (`~8.1 h`, verified in `v3`) | `174.00 h` | `55.0 h` | `22.23` | `40.80` | `21.60` / `0.9719` / `3.60%` |
| **WavCaps + LibriHeavy** | `cvssp/WavCaps`, `mythicinfinity/libriheavy` (`cc-by-4.0`, `14 B/s` aligned text) | `51,000 h` | `4,000.0 h` | `381.60` | `13.07` | `231.60` / `0.6069` / `38.60%` |
| **FineWeb-Edu + Stack v2** | `HuggingFaceFW/fineweb-edu`, `HuggingFaceTB/smollm-corpus` (`odc-by / apache-2.0`) | — | — | `500.00` | `0.55` | `150.00` / `0.3000` / `25.00%` |
| **TIER 1 TOTAL (`V4-S3`)** | **9 Verified Suites (`555 h` video/AV/robotics + `4,000 h` audio)** | — | **`4,555 h`** | **`1,115.31`** | **`438.88`** | **`600.00` / `max_ep = 0.9728` / `75.0%` signal** |

*\*Empirical `RH20T` Audio Verification:* HTTP byte-range (`0..3 MiB`) streaming tar-header inspection of `tonyfanggalaxies/RH20T` (`RH20T_lowdim/RH20T_cfg1..cfg7.tar.gz`) empirically verified `audio_mixed/audio_<unix_ms>.wav` (`1.34–3.36 MB` uncompressed PCM WAV) alongside `transformed/force_torque.npy` inside **all seven configs (`cfg1..cfg7`)** (logged in `gate1_heartbeat_and_corpus_receipt.json`). Full episode-level SNR, sample-rate, and non-silence coverage across extracted shards will be audited in Gate 2 (any silent/corrupt audio episode gets `audio_valid = 0` without fabrication).

### 4.2 Tier 2 Streaming / Quantized Pool (`V4-S4` `150M` Confirmation, `3.0B` Single-Pass Pos = `20.0` Chinchilla Pos/Param)

Scaling to `3.0B` single-pass packets (`<= 1.00` epoch, `71.99%` multimodal signal) requires **`2,100.0 h` of video/AV/robotics** (`100 h` Unitree G1/H1 + `450 h` DROID + `50 h` RH20T + `750 h` AgiBotWorld2026/ALOHA + `520 h` VGGSound + `65 h` EPIC-KITCHENS + `165 h` HoloAssist/GH) + **`14,500.0 h` of WavCaps/LibriHeavy** + **`1,200M` text/code bytes** (`3,469.85M` total unique packets; `max_ep = 1.0000`, `18.90%` robotics, `9.58%` AV-sync/ego, `43.51%` speech/audio, `28.01%` text/code). Exact uncompressed `fp16` storage is **`1,668.03 GB` (`1.67 TB`)**, which explicitly exceeds a `500 GB` local SSD; Tier 2 will therefore execute either via (a) rotating `~400 GB` streaming shard windows on cloud SSD/GCS, or (b) Gate-2-certified per-patch affine `int8` Wan 2.1 latent shards (**`908.95 GB`** total, authorized only if Gate 2 verifies `> 50 dB` latent PSNR and `< 0.5%` downstream NLL delta vs. `fp16`). Tier 3 (`V4-S5` `700M/1.3B`, `15B–26B` single-pass pos) requires multi-TB distributed object-store streaming and is gated on `V4-S4` Gate 4 PASS.

### 4.3 Multi-FPS Physical Resampling Policy (Gate 2 Enforced)

- **`50 fps` (`ALOHA-Sim`, `Unitree H1`, `EPIC-KITCHENS-100`):** Exact `2:1` integer decimation to `25.0 fps` video & `25.0 Hz` kinematics (zero phase jitter).
- **`30 fps` (`Unitree G1`, `AgiBotWorld2026`, `HoloAssist`):** Exact causal nearest-past frame sampling to `25.0 fps` (`<= 16.7 ms` phase bound) + linear/SLERP kinematics interpolation to `25.0 Hz`.
- **`25 fps` (`VGGSound`, `RH20T` timestamp-locked):** Native / PTS-locked `25.0 fps` audio-video synchronization.
- **`15 fps` (`DROID 1.0.1`):** Causal timestamp resampling to `25.0 fps` with explicit duplicate-frame tracking (`target_valid_mask = 0` on repeated video groups, or native-rate causal timestamps audited in Gate 2).
- **Explicit Exclusion of Low-FPS Corpora:** `BridgeData V2` (`5 fps`) is **excluded** from `25 fps` Wan 2.1 4-frame temporal video compression (eliminating `5×` static-frame collapse), and `LIBERO` (`10 fps`) is held out of `25 fps` video dynamics.

*Non-Commercial License Provenance (`CC-BY-NC`):* `agibot-world/AgiBotWorld2026` (`cc-by-nc-sa-4.0`) and `a1raman/epic_kitchens_100` (`cc-by-nc-4.0`) are strictly non-commercial research corpora and will be explicitly tagged in all Track 2 provenance tables and checkpoint cards.

---

## 5. Single-Factor `V4-S3` Ablation Matrix, Pre-Registered Promotion Rule & Dual-Track System Accounting

### 5.1 Dual-Track Accounting & Shared-Head Parity (`AGENTS.md §4.1`)

1. **Track 1 (`Core Comparison` — Strict Shared-Head Parity):** Every compared backbone (`SLWM-v4` variants, `3:1 Single-Bank Gated DeltaNet`, equal-byte `SW-GQA`, and `SLWM-v3` reference) is matched in trainable core parameters (`45M` in `V4-S3`, `150M` in `V4-S4`) and attaches **byte-for-byte identical** generative heads (`Causal Flow-Matching Video Head` ~`30M`, `Moshi Depth-Transformer Audio Head` ~`15M`, `Unified Masked 52/68 Motor Head`, `Calibrated 5-Action PolicyCommitGate`) and identical frozen codecs/single-pass exposures so generative head gains are never misattributed to the sequence backbone.
2. **Track 2 (`Strict System Cost`):** Reports trainable core + generative heads + policy head + frozen **Kyutai Mimi (`79.31M` total: `39.39M` encoder `3.91 MiB bf16` state + `39.92M` decoder streaming state newly measured in Gate 2 covering 8-layer sliding-window `decoder_transformer` KV cache & SEANet causal transposed-conv cache)** + frozen **Wan 2.1 Causal VAE (`126.89M` total: `53.59M` encoder `164.53 MiB bf16` cache + `73.30M` decoder 3D causal conv cache newly measured in Gate 2)** in both `bf16` and `fp32` (`MiB`), alongside per-heartbeat latency, FLOPs, peak HBM, and upstream pretraining provenance.
3. **Gate 2 Pre-Training Codec Benchmarks:** Gate 2 will directly measure (a) **Kyutai Mimi `16-RVQ` Whisper WER ceiling** (not extrapolated from `v3`'s `32-RVQ` `95%`), (b) both frozen **Wan 2.1 Decoder** and **Kyutai Mimi Decoder** streaming cache bytes (`MiB`), and (c) empirical per-heartbeat latency breakdown on the target live inference accelerator: **single NVIDIA A100-80GB / H100-80GB GPU (`bf16`)**, targeting `< 160 ms` total wall-clock per heartbeat (`<= 25 ms` Wan 2.1 enc+dec, `<= 15 ms` Mimi enc+dec, `<= 55 ms` core + Flow-Video `4–8` NFE + Depth-Audio, leaving `>= 65 ms` headroom).

### 5.2 Single-Factor `V4-S3` Discovery Matrix (`45M` Core, `600M` Single-Pass Pos, Fixed Anchor `A0`)

**Fixed Anchor (`A0_3to1_single_bank`):** `45M` core, `3:1` macro hybrid (`3` Gated DeltaNet blocks : `1` Sliding-Window GQA + Attention-Sink block), short causal `conv1d (k=4)`, Householder `beta ∈ (0, 2)`, SiLU `o_gate`, head rank `d_k = 64`, dense SwiGLU FFN, attached to shared Flow-Video + Depth-Audio + Motor + PolicyCommitGate heads (`JEPA-Sync = 0`, `Self-Forcing = 0`).

| Run ID | Single Factor Changed Relative to Parent | Scientific Isolation Target (`AGENTS.md §9` Coverage) |
|---|---|---|
| `B0_v3_ref` | `SLWM-v3` dual-branch per-layer + `M_anchor` (`45M` core) | Prior-generation SLWM reference |
| `B1_sw_gqa` | Pure equal-state-byte `SW-GQA + Attention Sink` (`45M` core) | Strong contemporary GQA attention baseline |
| **`A0_3to1_single_bank`** | **Fixed `V4-S3` Anchor** (`3:1` Hybrid Single-Bank Gated DeltaNet) | Fast, minimal-state recurrent baseline |
| `A1_5to1_single_bank` | Macro ratio `3:1 -> 5:1` ONLY (vs. `A0`) | Macro hybrid ratio sensitivity |
| `A2_1to1_single_bank` | Macro ratio `3:1 -> 1:1` ONLY (vs. `A0`) | Macro hybrid ratio sensitivity |
| `C1_no_conv1d` | Remove short causal `conv1d` ONLY (vs. `A0`) | Local temporal mixing ablation (`§9` no-conv) |
| `C2_beta_0_1` | Restrict Householder `beta ∈ (0, 2) -> (0, 1)` ONLY (vs. `A0`) | Negative-eigenvalue oscillatory dynamics ablation |
| `C3_no_ogate` | Remove SiLU `o_gate` ONLY (vs. `A0`) | Output-gating stability ablation |
| `C4_rank128` | Expand head rank `d_k = 64 -> 128` at matched params ONLY (vs. `A0`) | State capacity vs. head count trade-off |
| `C5_routed_2bank` | Zero-`M_anchor` pooled-surprise `B=2` routed SLWM (`1.0×` state bytes) ONLY | Multi-timescale routing isolation (zero state penalty) |
| `C6_routed_4bank` | Zero-`M_anchor` pooled-surprise `B=4` routed SLWM ONLY | Multi-timescale bank count scaling |
| `C7_ms_moe_ffn` | Dense SwiGLU -> Modality-Specific `MS-MoE` (matched active params) ONLY | Cross-modal FFN interference isolation |
| `O1_jepa_sync` | Add Causal Windowed Cross-Modal JEPA/SigLIP Sync ONLY (vs. best core) | Sub-second AV synchrony objective isolation |
| `H1_huber_video_abl` | Swap Flow-Matching Video Head -> Deterministic Huber ONLY | Proves generative Flow-Video cures fog (`FVD`/`LPIPS`) |
| `H2_flat_rvq_abl` | Swap Moshi Depth-Transformer Audio -> Flat `16-RVQ` Softmax ONLY | Proves Depth-Audio cures `RVQ-0` repeat collapse |
| `H3_self_forcing` | Add `Diffusion Forcing` history noise + short `Self-Forcing` rollout ONLY | Proves `2+ min` exposure-bias anti-drift |
| `H4_no_policy_gate` | Remove `PolicyCommitGate` / `ABSTAIN_SILENCE` head ONLY | `AGENTS.md §9` mandatory policy/no-op ablation |

### 5.3 Pre-Registered Quantitative Promotion & Falsification Rule

Define the equal-weighted 4-pillar normalized composite score (standardized across runs within the sweep):
$$Z_{\text{v4}} = 0.25\,Z_{\text{sync}}(\text{DeSync}\!\downarrow,\,20\text{ms Onset F1}\!\uparrow,\,\text{FAD}\!\downarrow) + 0.25\,Z_{\text{2min}}(60\text{–}120\text{s FVD/DINOv2 drift}\!\downarrow,\,>30\text{s Re-ID}\!\uparrow) + 0.25\,Z_{\text{rob}}(\text{ALOHA S2/S3}\!\uparrow,\,\tau\!\uparrow,\,\text{DROID AUROC}\!\uparrow) + 0.25\,Z_{\text{lang}}(\text{WT103 NLL}\!\downarrow,\,\text{LAMBADA}\!\uparrow).$$

1. **Single-Factor Promotion Criterion (`V4-S3`):** A single-factor change is promoted over its parent iff (a) it causes **zero regression `> 1.0%` relative** (or `> 0.015` absolute on bounded `[0, 1]` rates / `ECE`) on any of the four protected floors (`WikiText-103` NLL, `ALOHA` Stage 2/3, `DeSync`/`20 ms Onset F1`, calibrated `PolicyCommitGate` `ECE` & clean accuracy), AND (b) it achieves **`ΔZ_v4 >= +0.15 SD`** (or `>= +5%` relative win on its targeted pillar).
2. **Explicit Falsification / Retirement Margin (`V4-S3` & `V4-S4`):** If routed multi-timescale `SLWM-v4` (`C5`/`C6`) fails to beat **`3:1 Single-Bank Gated DeltaNet` (`A0`), equal-byte `SW-GQA` (`B1`), AND `SLWM-v3` (`B0`)** by **`>= +2.0%` relative on at least 2 of the 4 pillars** (with 95% hierarchical + scene-cluster CI excluding `0` in `V4-S4`) at matched core params and `<= 1.10×` streaming state bytes / `<= 1.15×` step latency, we will **report that negative result honestly, retire routed multi-timescale memory, and promote the simpler, faster winning core**.

---

## 6. Pre-Registered Gate 4 Capability Floors & `AGENTS.md §13.2` Showcase Protocol (`V4-S4` `150M`)

1. **Closed-Loop Robotics & World-Model Utility:**
   - Certified `ALOHA-Sim` (`AlohaTransferCube-v0` & `AlohaInsertion-v0`, `N >= 32` unseen seeds): `SLWM-v4` closed-loop policy/MPC achieves **`>= 50%`** sustained mid-air lift (`Stage 2`) and outperforms equal-byte `SW-GQA`; world-model policy ranking achieves **`Kendall's τ >= 0.70`** against true simulator success across checkpoints; `DROID` held-out building success-prediction AUROC **`>= 0.75`**.
2. **Synchronized Audio-Video (`Audio <-> Video`):**
   - `GreatestHits` & `RH20T` (contingent on Gate 2 `RH20T` non-silence audit) contact `Video -> Audio`: **`20 ms` / `50 ms` Impact Onset F1** decisively above interior-cropped desync null (`p < 0.01`), `RVQ-0` adjacent repeat rate **`<= 55%`** (curing `v3`'s `89.8–100%` collapse), and `DeSync` (`SynchFormer`, sec ↓) + `FAD` ↓ + `CLAP`/`ImageBind` ↑ reported directly alongside `MMAudio` (`157M` SOTA reference) and matched `3:1 Gated DeltaNet` / `SW-GQA`.
   - `Audio -> Video` & `Action -> Video`: Causal Flow-Matching video head decisively beats deterministic Huber (`H1_huber_video_abl`) on **`FVD` ↓** and **`LPIPS` ↓**.
3. **`2+ Minute` (`>= 120 s` / `750` Heartbeats) Sustainable Live Streaming:**
   - Flat $\mathcal{O}(1)$ VRAM (`<= 1%` growth from `t = 10 s` to `t = 120 s`), live throughput **`< 160 ms` per heartbeat** on single NVIDIA A100-80GB / H100-80GB (`bf16`), bounded `FVD`/`DINOv2` drift across `[0–30 s, 30–60 s, 60–120 s]` on `EPIC-KITCHENS-100` & `ALOHA`, and measurable **Object Permanence Re-ID** after `> 30 s` camera pan / occlusion (beating `SW-GQA` window forgetting).
4. **Calibrated 5-Action `PolicyCommitGate` & Usefulness/Hallucination Completeness (`AGENTS.md §8`):**
   - Exercises all 5 actions (`SPEAK_TEXT`, `EMIT_AUDIO`, `GENERATE_VIDEO`, `ACT_MOTOR`, `ABSTAIN_SILENCE`) with **`ECE < 0.05`**, **clean grounded task accuracy `>= 85%`** of the unconstrained forced-decode baseline, **clean abstention rate `<= 10%`**, and **blind-desync / OOD abstention rate `>= 70%`** (preventing degenerate abstention collapse), plus non-regression vs. `SLWM-v3` on `WikiText-103` & `LAMBADA`.
5. **Complete `AGENTS.md §13.2` Qualitative Showcase & Cross-Modal Causal Protocol:**
   - **Dev vs. Sealed Test Separation:** Fixed `dev` gallery during iteration; generate sealed `test` gallery **once** with strictly frozen weights (`no` online updates) and fixed decoding seeds (`no` best-of-$N$ regeneration or cherry-picking).
   - **Pre-Locked Common Manifest & Quantiles:** Lock common cross-model episode IDs (`showcase_manifest_common_v4.json`) in Gate 2 prior to training, plus deterministic post-evaluation quantile selection (`P05` failure, `P25`, `P50` median, `P75`, `P95`) ranked on the median seed across video, audio, robotics, and `2+ min` live rollouts.
   - **Side-by-Side Provenance & Epistemic Labels:** Render synchronized comparisons (`Ground Truth | Trivial Copy/Retrieval Baseline | Matched Baselines | SLWM-v4`) with explicit per-channel epistemic labels (`observed`, `reconstructed`, `predicted`, `imagined`, `abstained`) distinguishing forced diagnostic decoding from `PolicyCommitGate`-routed output.
   - **Causal Conditioning & Non-Inferiority Checks:** Run condition-shuffling (`shifted/mismatched audio/video`) and valid in-distribution action-intervention tests proving causal conditioning rather than unconditional prior hallucination, plus non-inferiority gates confirming generative detail heads do not degrade closed-loop robotics control, overwrite recall, or `PolicyCommitGate` calibration.

---

## 7. Gate 2 Realized Data-Contract Addendum (Pre-Training Reconciliation)

To keep the pre-registered Gate 1 plan (`§4`) strictly aligned with the verified Gate 2 public `valueai-org/slwm-v4-*` shards prior to any `V4-S3` training run, the following audited Gate 2 refinements are formally locked:

1. **Certified Per-Patch Affine `int8` Tier-1 Pool & `V4-S3` Single-Pass Budget (`450M` Positions):**
   - Frozen `Wan 2.1 Causal VAE` per-patch affine `int8` quantization was directly certified in Gate 2 at **`61.44 dB` latent PSNR**, **`54.03 dB` decoded RGB PSNR**, and **`0.99957` `11×11` SSIM** vs. `fp16` ([gate2_codec_state_and_parity_receipt.json](../../reports/v4/gate2_codec_state_and_parity_receipt.json)), exceeding the Gate 1 `> 50 dB` authorization bar by `+11.44 dB`.
   - `V4-S3` (`45M` core architecture discovery sweep) therefore trains on a **`450M`-position single-pass budget** (`10.0` pos/param, strictly `<= 0.98` epoch on every individual corpus via water-filling: **`75.00%` multimodal corpus share** [`23.88%` video/motor/AV + `51.12%` AudioCaps/LibriHeavy audio+text] / `25.00%` text+code; exact realized sensory non-text packet share = **`47.95%`** [`19.36%` video + `25.68%` audio + `2.92%` motor] vs `52.05%` text bytes including aligned transcripts/captions, closely matching Gate 1's `49.14%` target) over the verified public Tier-1 pool (**`303.55 h`** video/AV/robotics + **`3,129.89 h`** AudioCaps/LibriHeavy speech = **`3,433.45 h`** total + `500M` FineWeb-Edu/GitHub Clean Code bytes). `V4-S4` (`150M` confirmation, `3.0B` single-pass pos) remains unchanged.
2. **Strict Zero-Lookahead (`0.000 ms`) Kinematics & Video Resampling:**
   - Replaced linear/SLERP inter-frame kinematics interpolation in `§4.3` with strictly causal **zero-order hold (`src_ts <= tgt_ts`, `0.000 ms` future lookahead)** aligned identically to `video_src_frame_indices` across `15 / 30 / 50 fps` robotics corpora, and enforced `fps=25:round=down` in FFmpeg for continuous video streams. *(Tier-1 disclosure: all 5 robotics corpora and native `25 fps` `VGGSound` are strictly `0.000 ms`; Tier-1 `EPIC-KITCHENS-100` [`59.94 fps`] and `HoloAssist` [`~24.46 fps`] were encoded prior to adding `:round=down` and retain a `<= 8.34 ms` / `<= 20.4 ms` nearest-frame video-resampling residual, whereas Tier-2 `V4-S4` re-encoding enforces `fps=25:round=down` globally.)*
   - All LeRobot v3.0 robotics corpora (`DROID 1.0.1`, `12/13` `Unitree G1/H1` repos encoded within the `40 h` cap [`G1_Dex1_Wipe_Table` holds `0` eps under Tier-1 cap], and `100/100` `ALOHA-Sim`) are indexed strictly via `meta/episodes/chunk-*/file-*.parquet` video file IDs and exact `from_timestamp` offsets with single-stage frame-exact seeking.
3. **Empirically Verified Upstream Modality Corrections:**
   - **`RH20T`:** Upstream `RH20T_lowdim` provides joint state `q[t]` and 6D wrist force/torque without separate commanded-action logs or natural-language task descriptions; accordingly, `RH20T` encodes causal 1-step kinematic targets (`proprio = q[t]`, `action = q[t+1]` with `target_valid_mask = 0` on the final boundary tick, eliminating the `action == proprio` identity shortcut) and omits raw directory names from `MOD_TEXT`.
   - **`AgiBotWorld2026 PhysicalReasoning`:** Upstream `succeed` is empirically `100% False` (`0/3,058` positive); `episode_success` is therefore recorded as `-1` (unlabeled) rather than degenerate constant zeros. Verified dual-wrist 6D wrenches are mapped from `state[:, 2:8]` (left) and `state[:, 8:14]` (right) (`state[:, 2:14] == state[:, 14:26]` bit-identically).
   - **`Unitree H1` (`lerobot/unitreeh1_fold_clothes`):** All `40/40` action channels map losslessly without aliasing (`5` L-leg, `5` R-leg, `1` waist, `4` L-arm + `2` L-wrist, `4` R-arm + `2` R-wrist, `5` aux, `6` L-Inspire fingers, `6` R-Inspire fingers), while `observation.state[19]` activates strictly the `19` measured body joints in `proprio_mask` (`0` fabricated proprio dims).
4. **Small-Cluster Evaluation Disclosure & Stratified Split Fractions:**
   - `AgiBotWorld2026 + ALOHA-Sim` uses `68/17/15` (`19/4/5` AgiBot `task::scene` clusters + `15/2/3` ALOHA blocks -> `34 train / 6 dev / 8 test` clusters) and `Unitree G1/H1` uses `80/10/10` (realized **`218 train / 29 dev / 13 test`** session blocks across `12/13` repos covering `WBT`, `Inspire`, `Brainco`, `Dex3`, `Dex1`, and `H1`) so every robotics eval split carries `>= 6` disjoint clusters. *(Per-embodiment split skew disclosed for Gate 4: `unitreeh1_fold_clothes` has `35 train / 0 dev / 2 test`, `G1_Dex3_ToastedBread_Dataset` has `357 train / 60 dev / 0 test`, `G1_Dex1_Arrange_Surgical_Kits` has `100 train / 0 dev / 0 test`.)*
   - `GreatestHits` preserves exact official MIT CSAIL recording-date splits (`16 train / 6 dev / 6 test` true `gh::{YYYY-MM-DD}` clusters across all `977` clips).
   - `EPIC-KITCHENS-100` (`37` total participants -> realized **`32 train / 2 dev / 3 test`** participants) and `HoloAssist` (realized **`140 train / 8 dev / 3 test`** session clusters) have inherently small held-out cluster counts; Gate 4 statistical reporting will explicitly flag their limited cluster-level degrees of freedom and present hierarchical + cluster-bootstrap CIs alongside per-cluster breakdowns.
   - Thirty-nine upstream cross-split duplicate recordings detected by our non-vacuous SHA-256 audit (`2` YouTube re-uploads in `VGGSound`, `36` upstream-duplicated recordings at shifted episode indices in `Unitree G1`, and `1` YouTube re-upload `ac_9073` in `AudioCaps`) were atomically excised from `dev`/`test` shards prior to training and permanently logged in [gate2_eval_exclusion_manifest.json](../../reports/v4/gate2_eval_exclusion_manifest.json).
5. **Public License-Pure Dataset Distribution (`valueai-org/slwm-v4-*`):**
   - All 11 corpora and frozen eval benchmarks are published publicly under `valueai-org/slwm-v4-*` strictly separated by upstream license (`Apache-2.0`, `CC-BY-4.0`, `CDLA-Permissive-2.0`, `ODC-BY`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`, and MIT CSAIL Academic Non-Commercial), with `1:1` `.npz` <-> `_meta.json` provenance pairing and full upstream citations.

