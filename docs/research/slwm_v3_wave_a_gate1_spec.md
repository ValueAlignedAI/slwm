# SLWM-v3 Wave A — Gate 1 Hypothesis, Observability & Environment Hardening Specification (Rev 2)

**Status:** Pre-Registration & Hardened Gate 1 Evidence Pack (`0 CU` spend; resolves all 7 Gate 1 peer-review blocking flaws prior to Gate 2)  
**Branch:** `research/slwm-v3-selective-memory`  
**Verified Receipts & Tests:**
- Decoupled 9-Arm Control Matrix (`N = 1,024`) & Counterfactual Certifier: [gate1_control_receipt.json](../../reports/v3/gate1_control_receipt.json) ([v3_gate1_oracle_and_observability.py](../../evals/v3_gate1_oracle_and_observability.py), [test_v3_gate1_observability.py](../../tests/test_v3_gate1_observability.py) — `4/4 passed`)
- Real-Data Connected-Component & Inventory Audit: [gate1_real_data_receipt.json](../../reports/v3/gate1_real_data_receipt.json) ([v3_gate1_real_data_audit.py](../../scripts/data/v3_gate1_real_data_audit.py))
- Scored 20-Case Blind Observability Audit: [gate1_holo_blind_sheet.json](../../reports/v3/gate1_holo_blind_sheet.json), [gate1_holo_blind_key.json](../../reports/v3/gate1_holo_blind_key.json), [gate1_holo_blind_scores.json](../../reports/v3/gate1_holo_blind_scores.json)
- Real-Audio Same-Domain Overwrite Prototype & Anti-Splice Audit: [gate1_speech_prototype_receipt.json](../../reports/v3/gate1_speech_prototype_receipt.json) ([v3_gate1_speech_overwrite_prototype.py](../../scripts/data/v3_gate1_speech_overwrite_prototype.py))

---

## 1. Wave A Falsifiable Hypothesis & Pre-Registered Decision Rule

### Primary Research Question (Wave A — Selective Content-Routed Multi-Timescale Memory)

> Given **identical causal multimodal observations** and **fixed streaming state bytes**, does a general-purpose content-routed multi-timescale matrix memory (with learned bank allocation, a learned no-write option, a direct zero-lag sensory residual path, and **zero hardcoded slot indices, reserved token-ID ranges, or task opcodes**) retain the **latest valid value** after a multimodal correction while **preserving an unrelated concurrent fact**, across unseen speakers, scenes, and cue modalities — *without degrading immediate grounded scene perception or closed-loop control*?

### Pre-Registered Wave A Decision Rule (Frozen Prior to Training; `n = 5` Paired Seeds)

All confirmation inferences run **`n = 5` paired seeds** (`5^5 = 3,125` seed bootstrap draws, eliminating the `n = 3` worst-seed collapse) at `~124–150M` trainable core scale (after `30–60M` `5`-seed discovery). `SLWM-v3` **supports** the Wave A hypothesis if and only if **all three criteria** pass simultaneously:

1. **Criterion 1 — Primary Powered Delayed Selective-Overwrite Superiority (`Real-Speech-Overwrite`, `N = 1,200` test episodes across `250` unseen `SpeechCommands v0.02` test speakers):**
   - Achieves **≥ +5.0 pp** paired joint accuracy gain (requiring **both** the latest valid corrected attribute AND the preserved untouched concurrent attribute to be correct) against **every** registered bounded-memory comparator:
     - `SlidingWindow-GQA` (both **equal-parameter** and **equal-state-byte** variants),
     - `Mamba-2` and `DiagonalSSM`,
     - `Single-Bank GatedDeltaNet` (equal matrix bytes, single timescale),
     - `-LearnedRouting` (equal matrix count & bytes, tied timescales / fixed routing),
     - **Mandatory `Bounded Explicit Multi-Entry KV Memory Controller`** (byte-matched capacity $K_{\text{KV}}$ to `SLWM-v3`'s matrix memory state, explicit LRU/FIFO write & eviction policy; `Last-Command` is reported separately as a trivial sanity baseline),
   - With crossed **`(Seed × speaker_id)` `95%` bootstrap CI lower bound strictly `> 0.0 pp`**.
   - **Ecological generalization check (`HoloAssist-DC`):** Reported alongside with crossed `(Seed × session_prefix)` CIs; explicitly scoped as a secondary ecological check (not powered alone for a 5 pp claim at local `test n = 66 / 17`).
2. **Criterion 2 — Simultaneous Non-Regression (Explicit Reference Arms & Attainable Power):**
   - Paired contrast taken explicitly against **(a) the `dev`-selected strongest non-routed bounded baseline** AND **(b) the equal-byte `-LearnedRouting` ablation**:
     - **Closed-Loop Control (`N = 1,024` independent unseen worlds, `world_id` cluster):** `95%` paired CI lower bound **`≥ −2.0 pp`** on sustained 16-tick attainment (at `N = 1,024, p_disc = 0.10`, attainable null 95% CI half-width is **`±1.94 pp < 2.0 pp`**).
     - **Physical `xArm6` Immediate Grounded Scene Perception (`135` fixed `scene_cluster_id`s, `284` vault episodes):** (i) primary continuous target-block Euclidean localization RMSE `95%` paired CI upper bound **`≤ +0.015`** normalized table units, AND (ii) `180°`-flip paired dual-localization accuracy `95%` paired CI lower bound **`≥ −5.0 pp`** (calibrated honestly to the `135`-cluster null 95% CI half-width of **`±3.88 pp`**).
3. **Criterion 3 — Surprise-vs-Corruption & Distractor Resistance:**
   - Evaluated via **paired clean vs. stress streams** (`stream_a_clean` vs. `stream_a_distractor` matched-norm unrelated binding AND `stream_a_corrupt` moderate `3.5-sigma` sensory burst at randomized ticks): joint accuracy drop **`≤ 5.0 pp`** relative to the paired clean stream.

### Mandatory Single-Factor Causal Ablations (Isolating All Three Core Mechanisms)
Before confirmation, discovery scale (`30–60M`, `n = 5` seeds) isolates each mechanism independently:
1. **`-LearnedRouting`** (tied timescales / fixed uniform bank routing at exact equal state bytes),
2. **`-NoWriteOption`** (forced content write on every present packet, removing the explicit no-write channel),
3. **`-ZeroLagResidual`** (removing the direct zero-lag sensory residual bypass),
4. **`Backbone × Representation` cross** (`{SLWM-v3, SlidingWindow-GQA}` × `{Train-Only Scratch Edge, Frozen Causal Codec Edge}`).

If the **Bounded Explicit Multi-Entry KV Controller** or **tied-timescale `-LearnedRouting`** matches learned routing at equal state bytes on independently verified delayed corrections, we will report that negative result honestly and retire the learned-routing hypothesis before scaling.

---

## 2. Sample-Level Panels, Permitted History, Observability & Crossed-Cluster Power

### Panel 1 — Powered Real-Audio Selective Overwrite (`Real-Speech-Overwrite`, `SpeechCommands v0.02`)

- **Why `SpeechCommands v0.02` Replaces `GRID` as the Primary Powered Panel:**
  - Exact power audit ([gate1_control_receipt.json](../../reports/v3/gate1_control_receipt.json)) proved that `GRID`'s `8` test speakers (`m = 128` at `N = 1,024`) yield `DEFF = 13.70`, `N_eff = 74.7`, and `MDE = 16.2 pp` (severely underpowered!), AND `GRID`'s 6-word fixed sentence grammar contradicts single-attribute overwrite while splicing studio `GRID` with audiobook `LibriSpeech` introduces a domain shortcut.
  - **`SpeechCommands v0.02` eliminates all three flaws** ([gate1_real_data_receipt.json](../../reports/v3/gate1_real_data_receipt.json)):
    - **`2,112` train (`84,843` wavs) / `256` val (`9,981` wavs) / `250` test speakers (`11,005` wavs)** with **`zero_speaker_overlap_verified = True` (`0 / 0 / 0`)**.
    - **Fact Family 1 (`Direction`, 4 classes):** `{left, right, up, down}` (`396–425` test wavs across `157–168` test speakers each).
    - **Fact Family 2 (`Digit`, 4 classes):** `{zero, one, two, three}` (`399–424` test wavs across `156–170` test speakers each).
    - **Same-Domain Non-Correction Distractors:** `{yes, no, happy, house, marvin, sheila, cat, dog, bird, tree}` (`2,417` test wavs from the **exact same `250` speakers and `16 kHz` microphone distribution**; `50%` matched-speaker with the correction utterance!).
- **Empirical Prototype & Anti-Splice Verification ([gate1_speech_prototype_receipt.json](../../reports/v3/gate1_speech_prototype_receipt.json), `n = 64` test episodes):**
  - `199` unique unseen test speakers drawn in 64 episodes; randomized arrival order (`57.8%` digit-first) and randomized target attribute (`60.9%` digit-updated).
  - Overwrite occurs **after** initial fact exits single-layer local window (`min_init_to_corr = 47 > W = 32` Mimi `12.5 Hz` ticks), and query occurs **strictly beyond** the stacked sliding-window receptive field (`min_delay_after_corr = 460 > L*(W-1) = 372`).
  - Bit-exact counterfactual audio invariance: **`post_corr_pcm_max_abs_diff = 0.0`** across every `16 kHz` PCM sample after `t_corr_end` (while `corr_window_pcm_min_abs_diff = 0.4725`).
  - **Anti-splice acoustic audit** (RMS-matched `0.08`, `15 ms` cosine crossfade over real continuous `_background_noise_` bed): 1D low-level acoustic AUROC between correction windows and same-domain distractor windows is **`rms = 0.558, peak = 0.497, zcr = 0.440, spectral_centroid = 0.397`** (`max_dev_from_0.5 = 0.103 < 0.15` — `gate1_speech_prototype_pass = True`).
- **Exact Crossed Power (`N = 1,200` test episodes, `K_speakers = 250`):**
  - `m_speaker = 4.80`; at conservative `ICC = 0.10`, `DEFF = 1.38 → N_eff = 869.6 > 783` (`MDE = 4.75 pp < 5.0 pp` at `80%` power, `α = 0.05`). (Even at `N = 1,024`, `N_eff = 781.9`, `MDE = 5.01 pp`; locking `N = 1,200` provides comfortable headroom.)
  - `GRID` (`31` active speakers; `GRID-21, 33, 34` excluded) is retained solely as a secondary audiovisual lip-sync / speech non-regression check.

### Panel 2 — Unscripted Real Egocentric Delayed Corrections (`HoloAssist-DC`, Ecological Check)

- **Inventory & Honest Split Granularity ([gate1_real_data_receipt.json](../../reports/v3/gate1_real_data_receipt.json)):**
  - `672` local `-av.mp4` recordings (`46.34 h`; `48` duplicate file copies deduped, `0` unmatched files excluded; `131.47 h` available in full index).
  - Partitioned by case-normalized `session/performer prefix` (`282` prefixes: `196` train / `39` dev / `47` test; **`zero_session_prefix_split_overlap = True`**). **Explicit caveat:** instructors recur across sessions in HoloAssist, so splits are **session/performer-prefix disjoint, NOT instructor-disjoint**.
- **Strictly Causal Extraction & Non-Tautological Baselines:**
  - Enforces **zero intervening conversation turns** between `c_end` and `na_start`, skips purely preparatory micro-verbs (`approach`, `inspect`, `look_at`, `touch`), applies strict **word-boundary regexes** (`\b...\b`), and defines off-window preserved facts strictly over **causal history `<= query_time_s`** using full action-interval overlap (`a.end >= t_q - 10` and `a.start <= t_q`) — zero future `na_start` conditioning.
  - Yields `329` delayed ($\ge 1\text{ s}$) substantive corrections (`225` train / `38` dev / `66` test) and `82` component-explicit corrections (`52` train / `13` dev / `17` test).
  - **Non-tautological trivial baselines:**
    - All delayed ($\ge 1\text{ s}$, `n = 329` / `104` eval): persistence `verb = 17.0%, noun = 44.7%, pair = 10.6%`; train same-task majority `next_noun = 23.1%`, `off_window_noun = 18.0%`.
    - Component-explicit subset (`n = 82` / `30` eval): persistence `verb = 20.7%, noun = 39.0%, pair = 14.6%`; train same-task majority `next_noun = 10.0%`, `off_window_noun = 21.1%`.
  - **Honest window-eviction scope:** `t_q = c_end + min(1.0s, 0.5*delay)` (`<= 10` ticks at `10 Hz`) lies inside `W = 32`, and off-window gaps (`20–60 s`) straddle `L*(W-1) = 372`; `HoloAssist-DC` tests realistic unscripted short/medium-delay ecological grounding, **not** structural eviction.
- **Scored 20-Case Independent Blind Observability Audit ([gate1_holo_blind_scores.json](../../reports/v3/gate1_holo_blind_scores.json)):**
  - `20` distinct `dev`/`train` recordings exported to [gate1_holo_blind_sheet.json](../../reports/v3/gate1_holo_blind_sheet.json) (permitted causal history + 4 within-recording matched noun choices; ground truth segregated in [gate1_holo_blind_key.json](../../reports/v3/gate1_holo_blind_key.json)).
  - Scored strictly blind (text-only causal history, zero key access, zero video frames) by context-isolated subagent `a9653a67`:
    - **Overall 4-way blind accuracy:** **`15 / 20` (`75.0%`)** vs. `25.0%` chance and `10.0%` train task-majority prior!
    - **Calibration by blind rating:** `clearly_observable`: **`9/10` (`90.0%`)**; `partially_ambiguous`: **`6/9` (`66.7%`)**; `unobservable_from_text_alone`: **`0/1` (`0.0%`)** — confirming strong genuine observability even before visual frames are supplied.

### Panel 3 — Decoupled 9-Arm Control Matrix (`N = 1,024`) & Randomized Counterfactual Certifier

#### A. Decoupled Perception × Dynamics × Rollout Matrix (`16×16` Unsealed Grid, `N = 1,024`, `T = 180`, `H = 14`)

Both `Rollout` and `Static-Geodesic No-Rollout` share the exact same unsealed `16×16` occupancy grid, exact same `K = 38` two-stage `(wp1, wp2)` candidates, and identical `(-2.4*v - 0.8*d)` weights. Perception noise (`σ = 0.04`) and dynamics misspecification (`Nominal-Dyn`: frozen walls, `mass = 1, damp = 0.72, drift = 0, bumper = 0`) are **strictly decoupled**. Every cell of all 9 arms is reported from [gate1_control_receipt.json](../../reports/v3/gate1_control_receipt.json):

| Controller Arm (`N = 1,024` Unseen Worlds, 256/family) | Closest (`tail16`) | **Sustained Hold (`all 16`)** | `static_wall` (hold) | `dual_chicane` (hold) | `conveyor_drift` (hold) | `moving_bumper` (hold) |
|---|---:|---:|---:|---:|---:|---:|
| **`Oracle-Perc × Oracle-Dyn Rollout (H=14)`** | **`87.5%`** | **`87.5%`** | `100.0%` | `50.0%` | **`100.0%`** | **`100.0%`** |
| `Oracle-Perc × Nominal-Dyn Rollout (H=14)` | `55.7%` | `55.1%` | `100.0%` | `50.0%` | `0.0%` | `70.3%` |
| `Oracle-Perc × Static-Geodesic (No-Rollout)` | `37.5%` | `37.5%` | **`100.0%`** | `50.0%` | `0.0%` | `0.0%` |
| `Oracle-Perc × H=1 Greedy (No-Rollout)` | `37.9%` | `37.1%` | `71.1%` | `30.1%` | `0.0%` | `47.3%` |
| `Exact-State Reactive` | `33.0%` | `33.0%` | `96.9%` | `0.0%` | `0.0%` | `35.2%` |
| **`Noisy-Perc (σ=0.04) × Oracle-Dyn Rollout`** | **`56.8%`** | **`53.6%`** | `88.7%` | `52.7%` | `0.0%` | **`73.0%`** |
| `Noisy-Perc (σ=0.04) × Nominal-Dyn Rollout` | `42.0%` | `39.6%` | `88.7%` | `52.7%` | `0.0%` | `16.8%` |
| `Noisy-Perc (σ=0.04) × Static-Geodesic` | `38.5%` | `37.2%` | `100.0%` | `48.8%` | `0.0%` | `0.0%` |
| `Noisy-State Reactive` | `30.6%` | `29.9%` | `91.0%` | `1.2%` | `0.0%` | `27.3%` |

- **Key Decoupled Paired Contrasts (`2,000` bootstrap resamples, `N = 1,024`):**
  - **Fairness verification (`static_off_center_wall`):** `Oracle-Rollout = 100.0%` vs `Static-Geodesic = 100.0%` (zero static-geometry handicap).
  - **Multi-step rollout value (`Oracle-Rollout` vs `Static-Geodesic`):** **`+50.00 pp`** (`95% CI [+46.97, +53.03]`) sustained hold.
  - **Dynamics-model accuracy value (`Oracle-Dyn` vs `Nominal-Dyn`, both `H=14`):** **`+32.42 pp`** (`95% CI [+29.39, +35.35]`) sustained hold (`+100.0 pp` on `conveyor_drift`, `+29.7 pp` on `moving_bumper`).
  - **Planning value (`Oracle-Rollout` vs `Exact-State Reactive`):** **`+54.49 pp`** (`95% CI [+51.37, +57.52]`) sustained hold.
  - **Rollout advantage preserved under noisy perception (`Noisy-Oracle-Dyn` vs `Noisy-Static-Geodesic`):** **`+16.41 pp`** (`95% CI [+13.38, +19.73]`) sustained hold (`+14.1 pp` over `Noisy-Nominal-Dyn`).
  - **Clean perception monotonicity (`Oracle-Perc` vs `Noisy-Perc`, true dynamics):** **`+33.89 pp`** (`95% CI [+30.27, +37.40]`).

#### B. Shortcut-Free Randomized Counterfactual Certifier (Discovery `L=12, W=32` & Confirmation `L=16, W=64`)

Verified in [gate1_control_receipt.json](../../reports/v3/gate1_control_receipt.json) and `4/4` passing unit tests in [test_v3_gate1_observability.py](../../tests/test_v3_gate1_observability.py):
- **Dynamic scale-parameterized beyond-stack guard:**
  - Discovery (`L=12, W=32`, `L(W-1)=372`): `total_ticks = 522`, `min_delay = 442 > 372`, `min_init_to_corr_gap = 38 > 32`.
  - Confirmation (`L=16, W=64`, `L(W-1)=1008`): `total_ticks = 1190`, `min_delay = 1078 > 1008`, `min_init_to_corr_gap = 70 > 64`.
- **Zero clock/text/bf16/outlier shortcuts:** Randomized `t_first`, `t_second`, `t_corr` (`19` unique values/batch), `t_impulse` (`32` unique values/batch), random arrival order (`40.6%` F1-first), random update target (`43.8%` F1), random wall-clock offset `U(0, 50s)`, `100%` nonzero background text traffic, unit-norm `R^8` codebook (`SNR ~ 6.7` over `σ = 0.15`), paired `stream_a_clean` / `stream_a_distractor` / `stream_a_corrupt` (`3.5-sigma` burst), and **`max_abs_diff = 0.0` across all 7 post-cue tensors**.

---

## 3. Joint-Modality Support Ledger (Zero Fabricated Modalities)

| Corpus | Video | Audio | Text / Label | True Action | Verified Local Volume | Connected Split Key & Overlap Check |
|---|:---:|:---:|:---:|:---:|---|---|
| `SpeechCommands v0.02` | No | Yes (`16 kHz`) | Yes (35 words) | No | `84,843 / 9,981 / 11,005` wavs | `speaker_id` (`2,112 / 256 / 250`; `0` overlap) |
| `HoloAssist` (local `-av.mp4`) | Yes (`896×504`) | Yes (`48 kHz`) | Yes (timed turns) | No | `672` recs (`46.34 h`; `131.47 h` index) | `session_prefix` (`196 / 39 / 47`; `0` overlap; *not* instructor-disjoint) |
| `Physical xArm6` (`wave8a_v2`) | Yes (`64×64`) | Yes (paired) | Yes | Yes | `1,745 / 351 / 284` eps | `scene_cluster_id` (`763 / 151 / 135`; `15/15` zero overlap) |
| `GRID` (`grid_content_v1`) | Yes (face) | Yes (`16 kHz`) | Yes (aligned) | No | `14,126` utts (`31` active; `21,33,34` excl.) | `speaker_id` (`16 / 4 / 3 / 8`; `0` overlap) |
| `FineART` (ALOHA) | Yes (`>=480p`) | No | Yes (subtasks) | Yes (`14-DoF`) | `180` `.mp4` clips | `episode_index` |
| `LibriSpeech 360` | No | Yes (`16 kHz`) | Yes | No | `75,139` `.flac` files | `speaker_id` |
| `FineWeb-Edu v1` | No | No | Yes (English) | No | `11.0 GB` | document shard |
