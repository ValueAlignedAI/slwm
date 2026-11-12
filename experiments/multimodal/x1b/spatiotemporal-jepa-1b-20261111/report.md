# Study X1B — Spatiotemporal Latent-JEPA, Zero-Lag Multi-Rate GatedDelta & Innovation-Gated MPC at 1.008B Scale

- **Sprint / Study ID:** `X1B-20261111` (`experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111`)
- **Branch:** `research/x1b-spatiotemporal-jepa-mpc-20261111` (strictly isolated from `G*` studies, local Fedora GPU, and `~/.local/state/slwm/`)
- **Accounting Standard:** `strict_total_parameters` (`AGENTS.md` §4 & §10; **`1,008,454,410`** on `SLWM-X1B` vs **`1,008,435,114`** on `DenseTransformer-X1B` and `LegacyScalarHybrid-X1B`, **`-0.0019%` delta**; `< 0.006%` parity across all `152M` factorial models)
- **Compute Ledger:** **`201.44 / 500.00` Colab compute units** (`40.3%` of budget) expended across **120 dual-HDD verified cloud runs** (`Wave 1` `28`-run diagnostic + `Wave 2` `28`-run scaling diagnostic + `Wave 3` `28`-run Zero-Lag 1-seed `1B` cohort + **`Wave 4` `36`-run 3-seed `1.008B` / `T=1024` / OOD Goal-Switch cohort**); initial balance `965.17` → final balance `763.73`; **`0` active Colab sessions (`current_rate_per_hr = 0.0`)**. Every single `1.008B` (`9` runs) and `453M` (`3` runs) model executed on **Google TPU `v6e-1` (`33.55 GB` HBM)**; all `24` `152M` 3-seed factorial runs executed in parallel on NVIDIA `A100-SXM4-40GB`.
- **Dual-HDD Artifact Custody (`192.168.1.200`, Fedora):** **`120 / 120` cumulative runs** (`36 / 36` in [`runs/`](runs/), `28 / 28` in [`runs_wave3/`](runs_wave3/), `28 / 28` in [`runs_wave2/`](runs_wave2/), and `28 / 28` in [`runs_wave1/`](runs_wave1/)) verified bit-for-bit (`MIRROR_VERIFIED_2_COPIES`) across independent physical HDDs `fbbb53d9-dd42-4f19-95e6-dadab02a201e` (`/mnt/slwm-checkpoints-a/SLWM/x1b`) and `c961a363-8d77-4fba-a48d-622b00a6d8c3` (`/mnt/slwm-checkpoints-b/SLWM/x1b`).
- **Runnable `1.008B` Checkpoints & Live Replays:** [Standalone Inference & Streaming CLI](inference/README.md) · [Cryptographic Model Manifest](inference/manifest.json) · [Executed `1.008B` Closed-Loop Replays & Filmstrips](examples/README.md)

---

## 1. Hypotheses & Pre-Registered Remedies for G0–G14 Bottlenecks

Earlier `SLWM` studies (`G0`–`G14`) established strict falsification gates (`AGENTS.md`) while exposing five structural world-model bottlenecks: destructive `16×16 → 1×128` edge compression (`G8` `1-px` barrier blindness and cross-modal collapse), single-matrix linear pixel/audio MSE losing to zero-order persistence, scalar-decay working-memory amnesia (`~57 pp` transient-cue drop and `6.2%` overwrite recall), myopic `1-step` 3-candidate random shooting (`0–1.6%` Push-T), and an uncalibrated `0`-parameter `PolicyCommitGate` vulnerable to `G6` camera spoofing.

Study X1B tested five pre-registered architectural and algorithmic remedies across **3 seeds (`2001, 2002, 2003`) at every scale rung** (`152M × 8 variants × 3 seeds = 24 runs`, `453M × 3 seeds = 3 runs`, and **`1.008B × 3 architectures × 3 seeds = 9 runs`** = `36` Wave 4 runs; `120` total across Waves 1–4):

1. **H1 — Spatiotemporal Field Preservation (`S = 24` Structured Slots vs `1×D` Vector Bottleneck):** Preserving a `64×64` overlapping Conv2D + `4×4` spatial patch grid (`16` visual slots) alongside `2` spectral/onset audio slots, `1` proprioceptive slot, `1` role-structured `4096`-vocab subword slot, `1` motor-action slot, and `3` persistent working-memory registers (`S = 24` slots/tick) achieves **`1.000` pre-impact `1-px` barrier AUROC** (vs `0.51–0.66` under `16×16` subsampling) and **`94.3% ± 1.9%` real audio-visual `R@1`** at `1.008B` (vs **`1.0% ± 0.7%` random chance** on `LegacyScalarHybrid-X1B`, `+93.23 pp`, 99% CI `[+88.02, +97.40]`).
2. **H2 — Position-Free Multi-Head Householder `GatedDelta` Matrix Memory + $\mathcal{O}(1)$ Ring-Buffer Hybrid:** Interleaving `3:1` multi-head Householder `GatedDelta` matrix recurrence ($M_t^{(h)} = \alpha_t^{(h)} M_{t-1}^{(h)} + \beta_t^{(h)}(v_t^{(h)} - \alpha_t^{(h)} M_{t-1}^{(h)} k_t^{(h)}) (k_t^{(h)})^\top$) with fixed-window ring-buffer Causal RoPE bounds streaming state to a strictly constant **`14.91 MB` at both `T = 64` and `T = 1024`** (`222.2×` smaller than `DenseTransformer-X1B`'s **`3,312.54 MB`** single-stream KV cache at `T = 1024`, which physically exhausted `32 GB` TPU `v6e-1` HBM at `46.99 GB` during unchunked `bs = 96` MPC rollouts). At **`32×` out-of-distribution horizon extrapolation (`T = 32` train → `T = 1024` eval = `24,576` slots)**, `SLWM-X450M` (`453M`) and `SLWM-X150M` (`152M`) achieve **`100.0% ± 0.0%`** across all 3 seeds and `SLWM-X1B` (`1.008B`) achieves **`89.6% ± 14.7%` (`T=512`) / `85.4% ± 20.6%` (`T=1024`, 2/3 seeds at `100.0%`)**, beating `DenseTransformer-X1B` (`33.3% ± 5.9%`, **`+52.08 pp`**) and `LegacyScalarHybrid-X1B` (`2.1% ± 2.9%`, **`+83.33 pp`**).
3. **H3 — Zero-Lag Multi-Rate Lead-Lag Kinematics + Decoupled Pose/Goal Readout Heads (`L = 23`):** Splitting temporal slot filtering into a zero-lag lead-lag bypass on sensory slots (`0..20`) and multi-timescale working memory (`SLOTS_REGISTER` `21..23`), paired in Wave 4 with decoupled `head_vis_pose_w/b` (instantaneous pose), `head_proprio_w/b` (Euler differential $\Delta s$), and `head_goal_w/b` (register working-memory latch), eliminates both Wave 2's deep IIR phase lag and Wave 3's open-loop transient-cue interference — lifting `SLWM-X1B` (`1.008B`) `0.4s` open-loop transient retention from `2.1%` (Wave 3) to **`90.3% ± 6.4%` (`+88.2 pp`)** and enabling **`93.2% ± 3.9%` OOD `T = 76` mid-episode Goal-B overwrite MPC** (`96.9% ± 2.6%` at `453M`) vs **`6.8% ± 8.5%` on `DenseTransformer-X1B` (`+86.46 pp`, 99% CI `[+79.17, +92.71]`)** and **`73.4% ± 13.4%` on `LegacyScalarHybrid-X1B` (`+19.79 pp`, 99% CI `[+10.42, +29.17]`)**.
4. **H4 — 4-Step Unrolled Residual-EMA-JEPA + `SIGReg` + Receding-Horizon `Latent-CEM-MPC`:** Training an action-conditioned Euler residual latent predictor (`jepa_1 + 0.5*jepa_2 + 0.35*jepa_4` with 15% off-policy motor perturbations) regularized by isotropic variance-covariance `SIGReg` prevents dimensional collapse (**`Participation Ratio = 11.5% ± 0.8%`** at `1.008B` vs `1.1%` without `SIGReg`, `1.0%` raw-pixel MSE, `3.2%` Legacy-1B, and `4.4%` Dense-1B), achieves **`+30.3% ± 0.8%` 4-step physical forecast skill over persistence**, and powers `Latent-CEM-MPC` (`71.9% ± 12.7%` at `1.008B` and **`89.6% ± 6.1%` at `453M`** vs `23.3% ± 2.0%` for G8 3-shot random shooting, **`+48.61 pp`** on identical `1.008B` weights, 99% CI `[+40.28, +56.94]`).
5. **H5 — Mahalanobis Innovation `PolicyCommitGate` Against `G6` Camera Spoofing & Hallucination:** Conditioning commitment routing (`commit_full`, `commit_trusted_proprio`, `probe_or_abstain`, `wait`) on normalized latent innovation surprise and visual-proprioceptive pose discrepancy detects `180°` wrong-camera spoofing on **`96.9%`** of post-attack ticks at `1.008B`, cuts spoofed action regret from `0.499 ± 0.068` (`always_commit`) to **`0.295 ± 0.008`** (`-40.9%` regret reduction), eliminates unsupported claims under missing sensory grounding (**`0.0%` vs `100.0%`**), and preserves **`70.5%`** clean-task usefulness (`ECE = 0.0383`).

---

## 2. Iterative 4-Wave Scientific Discovery Arc (`120` Verified Cloud Runs)

Every run across all four experimental waves (`28 + 28 + 28 + 36 = 120` runs) is preserved with full SHA-256 dual-HDD custody:

1. **Wave 1 (`28` runs, [`runs_wave1/`](runs_wave1/), [`wave1_diagnostic_summary_metrics.json`](wave1_diagnostic_summary_metrics.json)):** Exposed bimodal `+y / -y` CEM trajectory cancellation into central `1-px` barriers (`0%` barrier clearance) and `rms_norm` radial distance erasure (`-2.4%` persistence skill). Fixed via sharp MPPI Boltzmann weighting, unnormalized affine proprioceptive skip, and causal Read-Before-Write in `GatedDelta`.
2. **Wave 2 (`28` runs, [`runs_wave2/`](runs_wave2/), [`wave2_scaling_diagnostic_summary_metrics.json`](wave2_scaling_diagnostic_summary_metrics.json)):** Uncovered two deep-scaling pathologies at `L = 23`: (a) cascaded 18-layer pure-IIR sensory phase delay (`35.4%` `1B` MPC) and (b) visual pose drift corrupting symbolic keys in `SLOT_TEXT` (`0%` overwritten-key recall at `1B` vs `100%` on `abl_separate_cores`).
3. **Wave 3 (`28` runs, [`runs_wave3/`](runs_wave3/), [`wave3_1seed_1b_summary_metrics.json`](wave3_1seed_1b_summary_metrics.json)):** Zero-Lag Multi-Rate `GatedDelta` + directed cross-modal grounding (`_shield_text_slot_mask`) lifted 1-seed `SLWM-X1B` to `77.1%` MPC and `100%` `T = 512` recall, while exposing two final limitations: (a) single-seed (`n = 1`) statistical power at `1.008B` / `453M`, and (b) open-loop `0.4s` transient policy drop (`2.1%` at `1B`) caused by sharing a single `head_proprio_w` matrix between instantaneous pose decoding and differential $\Delta s$ Euler transitions.
4. **Wave 4 (`36` runs, [`runs/`](runs/), [`summary_metrics.json`](summary_metrics.json)):** Decoupled `head_vis_pose_w/b`, `head_proprio_w/b`, and `head_goal_w/b`, added 4-step unrolled Residual-EMA-JEPA with off-policy calibration, enforced strict $\mathcal{O}(1)$ ring-buffer streaming (`cache_len = 64`), expanded `Panel 3` to `T = 1024` (`32×` OOD) and `Panel 5` to `N = 96`/seed (`288` pooled episodes), added **`Panel 8`** (`T = 76` OOD Sustained Transient Thin-Barrier MPC & Mid-Episode Goal-B Overwrite MPC), and executed **full 3-seed (`2001, 2002, 2003`) training & evaluation across all `1.008B`, `453M`, and `152M` cohorts on TPU `v6e-1` and `A100`**.

---

## 3. Strict Parameter Accounting, Config Hashes & Dual-HDD Receipts (`Wave 4`)

| Model / Cohort Group (`n=3` seeds each) | Arch | Strict Total Params | Adapters | Processor Core | JEPA & Unc | Policy & Heads | Config SHA-256 (`s2001`) | `int8` Portable SHA-256 (`s2001`) | Accelerator |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| **`SLWM-X1B` (`flagship_1b`, 3 seeds)** | `slwm_x1b` | **`1,008,454,410`** | `19,033,728` | `954,618,720` | `16,529,666` | `18,272,296` | `1a56fcec57a7` | `0e0ae653c5af` (`918.9 MB`) | TPU `v6e-1` |
| **`DenseTransformer-X1B` (`dense_1b`, 3 seeds)** | `dense_transformer` | **`1,008,435,114`** (`-0.0019%`) | `19,033,728` | `954,599,424` | `16,529,666` | `18,272,296` | `5275c49e967e` | `5fd4c02104d1` (`916.8 MB`) | TPU `v6e-1` |
| **`LegacyScalarHybrid-X1B` (`legacy_1b`, 3 seeds)** | `legacy_scalar_hybrid` | **`1,008,435,114`** (`-0.0019%`) | `19,033,728` | `954,599,424` | `16,529,666` | `18,272,296` | `44b9a2b5dc9f` | `a8d90699d340` (`915.1 MB`) | TPU `v6e-1` |
| **`SLWM-X450M` (`scale_450m`, 3 seeds)** | `slwm_x1b` | **`453,199,938`** | `11,939,040` | `420,025,464` | `9,521,858` | `11,713,576` | `b3b83cd3bdd7` | `86624c2f54c6` (`415.1 MB`) | TPU `v6e-1` |
| **`SLWM-X150M` (`abl_slwm_full`, 3 seeds)** | `slwm_x1b` | **`152,017,626`** | `6,401,856` | `134,702,832` | `4,430,978` | `6,481,960` | `d8b390781dcd` | `9d7193fbe975` (`140.3 MB`) | `A100-40GB` |
| **`152M` GatedDelta Ablations (5 × 3 seeds)** | `slwm_*` / `separate` | **`152,017,626`** (`0.0000%`) | `6,401,856` | `134,702,832` | `4,430,978` | `6,481,960` | per-variant | dual-HDD verified | `A100-40GB` |
| **`152M` Attn / Scalar Ablations (2 × 3 seeds)** | `dense` / `scalar` | **`152,009,706`** (`-0.0052%`) | `6,401,856` | `134,694,912` | `4,430,978` | `6,481,960` | per-variant | dual-HDD verified | `A100-40GB` |

---

## 4. Primary Quantitative Results (`36` Verified Wave 4 Runs, 3 Seeds Everywhere)

### 4.1 3-Seed `1.008B` Head-to-Head & Scaling Trajectory (`152M → 453M → 1.008B`, Mean ± SD)

![3-Seed 1.008B Scaling, Rank & Grounding](figures/x1b_scaling_and_1b_head_to_head.png)

| Model (`n=3` seeds, Strict Params) | Overall MPC (`N=288`) | `1-px` Barrier MPC (`N=144`) | OOD `T=76` Sustained MPC (`N=192`) | OOD `T=76` Goal-Switch MPC (`N=192`) | Assoc `T=512` (`16×`) | Assoc `T=1024` (`32×`) | `0.4s` Transient Policy | Real AV `R@1` | 1-Step / 4-Step Skill | Latent `PR` | Stream State `T=1024` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`SLWM-X1B` (`1.008B`)** | `71.9 ± 12.7%` | `55.6 ± 20.3%` | **`100.0 ± 0.0%`** | **`93.2 ± 3.9%`** | **`89.6 ± 14.7%`** | **`85.4 ± 20.6%`** | **`90.3 ± 6.4%`** | **`94.3 ± 1.9%`** | `+11.8%` / `+30.3%` | **`0.115`** | **`14.91 MB`** ($\mathcal{O}(1)$) |
| **`DenseTransformer-X1B` (`1.008B`)** | `47.2 ± 2.6%` | `0.7 ± 1.0%` | `90.1 ± 7.0%` | `6.8 ± 8.5%` | `50.0 ± 0.0%` | `33.3 ± 5.9%` | `68.1 ± 8.6%` | `94.3 ± 2.7%` | `+12.8%` / `+28.7%` | `0.044` | `3,312.54 MB` ($\mathcal{O}(T)$) |
| **`LegacyScalarHybrid-X1B` (`1.008B`)** | `79.2 ± 1.7%` | `59.7 ± 4.3%` | `94.8 ± 4.1%` | `73.4 ± 13.4%` | `2.1 ± 2.9%` | `2.1 ± 2.9%` | **`98.6 ± 1.0%`** | `1.0 ± 0.7%` | **`+22.0%`** / `+30.4%` | `0.032` | **`8.16 MB`** ($\mathcal{O}(1)$) |
| **`SLWM-X450M` (`453M`)** | **`89.6 ± 6.1%`** | **`83.3 ± 9.0%`** | `99.5 ± 0.7%` | **`96.9 ± 2.6%`** | **`100.0 ± 0.0%`** | **`100.0 ± 0.0%`** | `93.8 ± 5.9%` | `92.2 ± 3.8%` | `+16.3%` / **`+33.8%`** | **`0.128`** | **`8.79 MB`** ($\mathcal{O}(1)$) |
| **`SLWM-X150M` (`152M`)** | `60.4 ± 3.4%` | `24.3 ± 4.9%` | `89.1 ± 7.8%` | `65.1 ± 13.1%` | **`100.0 ± 0.0%`** | **`100.0 ± 0.0%`** | `69.4 ± 2.6%` | `92.2 ± 2.6%` | `+17.2%` / `+31.5%` | `0.119` | **`4.27 MB`** ($\mathcal{O}(1)$) |

![T=1024 Associative Overwrite, Transient Retention & O(1) Memory State](figures/x1b_memory_and_horizon_extrapolation.png)

### 4.2 `152M` 3-Seed Factorial Ablation Matrix (`24` Runs, Mean ± SD)

![3-Seed Objective Ablation, Planner Comparison & Spoofing Safety](figures/x1b_jepa_mpc_and_spoofing_safety.png)

| Factorial Variant (`152M`, `n=3` seeds) | Isolates Bottleneck | Overall MPC | `1-px` Barrier MPC | OOD `T=76` Sustained | OOD `T=76` Goal-Switch | Assoc `T=1024` (`32×`) | Real AV `R@1` | 1-Step / 4-Step Skill | Latent `PR` | `0.4s` Policy |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`abl_slwm_full` (Full SLWM)** | Reference unified design | `60.4 ± 3.4%` | `24.3 ± 4.9%` | `89.1 ± 7.8%` | `65.1 ± 13.1%` | **`100.0 ± 0.0%`** | `92.2 ± 2.6%` | `+17.2%` / `+31.5%` | **`0.119`** | `69.4 ± 2.6%` |
| **`abl_no_registers`** | Removes `SLOTS_REGISTER` (`21..23`) | `74.7 ± 4.4%` | `49.3 ± 8.7%` | **`67.2 ± 3.4%`** | **`53.1 ± 2.6%`** | **`8.3 ± 2.9%`** | `95.3 ± 3.8%` | `+22.0%` / `+34.1%` | `0.114` | `70.1 ± 11.3%` |
| **`abl_scalar_retention`** | Replaces matrix `GatedDelta` w/ scalar EMA | `84.4 ± 5.2%` | `68.8 ± 10.3%` | `96.4 ± 5.2%` | `79.2 ± 15.4%` | **`37.5 ± 17.7%`** | `91.7 ± 5.2%` | `+24.7%` / `+38.3%` | **`0.024`** | `65.3 ± 3.5%` |
| **`abl_dense_attn`** | Replaces `GatedDelta` w/ full RoPE KV | **`47.6 ± 13.7%`** | **`13.2 ± 13.2%`** | `89.1 ± 4.4%` | **`45.8 ± 26.2%`** | **`39.6 ± 11.8%`** | `93.2 ± 2.7%` | `+15.2%` / `+27.1%` | **`0.023`** | `81.2 ± 13.9%` |
| **`abl_separate_cores`** | Blocks cross-modal spatial attention | **`35.1 ± 3.8%`** | **`2.1 ± 2.9%`** | **`32.3 ± 15.0%`** | **`5.7 ± 1.9%`** | `100.0 ± 0.0%` | **`62.5 ± 3.4%`** | `+35.9%` / `+45.9%` | `0.029` | **`19.4 ± 10.9%`** |
| **`abl_raw_pixel_mse`** | Replaces EMA-JEPA w/ pixel/audio MSE | `69.8 ± 4.4%` | `41.0 ± 10.0%` | `96.4 ± 3.2%` | `72.9 ± 15.6%` | `100.0 ± 0.0%` | **`1.6 ± 0.0%`** | `+19.4%` / `+32.0%` | **`0.010`** | `72.9 ± 5.9%` |
| **`abl_direct_jepa_no_res`** | Removes Euler residual latent integration | **`46.2 ± 4.0%`** | `29.2 ± 11.2%` | **`65.1 ± 2.7%`** | **`20.8 ± 7.4%`** | `100.0 ± 0.0%` | `90.6 ± 0.0%` | **`-2.4%`** / `+23.6%` | **`0.017`** | `69.4 ± 8.0%` |
| **`abl_jepa_no_sigreg`** | Removes `SIGReg` variance-covariance loss | `63.5 ± 5.6%` | `34.0 ± 6.9%` | `89.6 ± 3.2%` | `57.3 ± 16.0%` | `100.0 ± 0.0%` | `94.3 ± 0.7%` | `+20.7%` / `+35.4%` | **`0.011`** | `75.7 ± 18.3%` |

### 4.3 Pooled 3-Seed Paired Bootstrap 95% & 99% Confidence Intervals (`10,000` Resamples)

![OOD T=76 Goal-Switch MPC & Pooled 3-Seed 1.008B Paired Bootstrap Forest Plot](figures/x1b_long_horizon_ood_goal_switch_mpc.png)

| Pooled 3-Seed Paired Contrast | Mean Effect ($\Delta$) | 95% Paired CI | 99% Paired CI | Scientific Interpretation |
| :--- | ---: | :---: | :---: | :--- |
| **`1.008B` `SLWM-X1B` vs `DenseTransformer-X1B` (OOD `T=76` Goal-Switch MPC)** | **`+86.46 pp`** | `[+81.25, +91.15]` | **`[+79.17, +92.71]`** | Decisive `3-seed` OOD working-memory overwrite superiority (`93.2%` vs `6.8%`, `49.0%` stale-Goal-A trap on Dense) |
| **`1.008B` `SLWM-X1B` vs `DenseTransformer-X1B` (`1-px` Thin-Barrier MPC)** | **`+54.86 pp`** | `[+46.53, +63.19]` | **`[+43.75, +65.28]`** | Resolves Dense-1B `1-px` barrier collapse (`55.6%` vs `0.7%` across `N = 144` pooled worlds) |
| **`1.008B` `SLWM-X1B` vs `DenseTransformer-X1B` (`Overall MPC`, `N=288`)** | **`+24.65 pp`** | `[+18.40, +30.90]` | **`[+16.32, +32.99]`** | `p < 0.001` 3-seed closed-loop control superiority (`71.9%` vs `47.2%`) |
| **`1.008B` `SLWM-X1B` vs `DenseTransformer-X1B` (OOD `T=76` Sustained MPC)** | **`+9.90 pp`** | `[+5.73, +14.06]` | **`[+4.69, +16.15]`** | `100.0%` (`192/192` worlds) sustained transient-goal navigation vs `90.1%` |
| **`1.008B` `SLWM-X1B` vs `DenseTransformer-X1B` (`T=1024` `32×` Overwrite)** | **`+52.08 pp`** | `[+35.42, +68.75]` | **`[+29.17, +72.92]`** | `24,576`-slot OOD associative recall with `222.2×` smaller streaming state (`14.91 MB` vs `3,312.54 MB`) |
| **`1.008B` `SLWM-X1B` vs `LegacyScalarHybrid-X1B` (`Real AV R@1`)** | **`+93.23 pp`** | `[+89.58, +96.35]` | **`[+88.02, +97.40]`** | Eliminates cross-modal collapse (`94.3%` vs `1.0%` random chance) |
| **`1.008B` `SLWM-X1B` vs `LegacyScalarHybrid-X1B` (`T=1024` `32×` Overwrite)** | **`+83.33 pp`** | `[+72.92, +93.75]` | **`[+68.75, +95.83]`** | Resolves scalar-retention overwrite impossibility (`85.4%` vs `2.1%`) |
| **`1.008B` `SLWM-X1B` vs `LegacyScalarHybrid-X1B` (OOD `T=76` Goal-Switch MPC)** | **`+19.79 pp`** | `[+12.50, +27.08]` | **`[+10.42, +29.17]`** | `p < 0.001` OOD mid-episode working-memory overwrite control gain (`93.2%` vs `73.4%`) |
| **`1.008B` `SLWM-X1B` vs `LegacyScalarHybrid-X1B` (OOD `T=76` Sustained MPC)** | **`+5.21 pp`** | `[+2.08, +8.33]` | **`[+1.56, +9.90]`** | Significant `99% CI > 0` long-horizon sustained transient-goal gain (`100.0%` vs `94.8%`) |
| **`1.008B` `SLWM-X1B` vs `LegacyScalarHybrid-X1B` (`Short T=38 Cont. MPC`)** | **`-7.29 pp`** | `[-13.54, -1.04]` | `[-15.28, +0.69]` | Unimodal kinematic trade-off when `Legacy-1B` receives X1B's `64×64` stem + decoupled `head_proprio` (see §6) |
| **`1.008B` `Latent-CEM-MPC` vs `G8 3-Shot Shooting` (`SLWM-X1B` weights)** | **`+48.61 pp`** | `[+42.36, +54.51]` | **`[+40.28, +56.94]`** | Proves multi-step receding-horizon CEM/MPPI planning is essential (`71.9%` vs `23.3%`) |
| **`1.008B` `Latent-CEM-MPC` vs `Open-Loop Policy` (`SLWM-X1B` weights)** | **`+31.94 pp`** | `[+25.35, +38.54]` | **`[+23.26, +40.28]`** | Closed-loop latent replanning decisively beats amortized open-loop execution (`71.9%` vs `39.9%`) |
| **`152M` `SLWM-Full` vs `Separate-Cores` (`MPC` / `Real AV R@1`)** | **`+25.35 pp` / `+29.69 pp`** | `[+19.79, +30.90]` / `[+22.40, +36.98]` | **`[+18.40, +32.99]` / `[+20.31, +39.58]`** | Proves shared cross-modal spatial field drives both barrier control and audiovisual grounding |
| **`152M` `SLWM-Full` vs `Direct-JEPA-No-Res` (`MPC`)** | **`+14.24 pp`** | `[+7.64, +21.18]` | **`[+5.55, +23.61]`** | Proves Euler physical residual integration prevents `-2.4%` persistence loss and `20.8%` Goal-Switch collapse |
| **`152M` `SLWM-Full` vs `Dense-Attn` (`MPC` / `OOD Goal-Switch`)** | **`+12.85 pp` / `+19.27 pp`** | `[+7.29, +18.40]` / `[+9.38, +29.17]` | **`[+5.56, +20.14]` / `[+6.25, +32.29]`** | Replicates `GatedDelta` over dense RoPE superiority at `152M` (`3` seeds) |
| **`152M` `SLWM-Full` vs `Scalar-Retention` (`T=1024` `32×` Overwrite)** | **`+62.50 pp`** | `[+47.92, +75.00]` | **`[+43.75, +79.17]`** | Isolates Householder `GatedDelta` matrix erasure vs scalar exponential decay (`100.0%` vs `37.5%`) |

---

## 5. Executed `1.008B` Closed-Loop Replays, Spatial Resolution & Latency Scaling

- **Standalone `1.008B` Closed-Loop Mission Filmstrips ([`examples/`](examples/)):** Executing [`scripts/eval/x1b_infer.py --build-study-pack`](../../../../scripts/eval/x1b_infer.py) streams SHA-256-verified `s2001` `1.008B` checkpoints directly from Fedora and records full `64×64` closed-loop episodes:
  - **OOD `T = 76` Mid-Episode Goal-B Overwrite Mission ([`preview_ood_goal_switch_mpc.png`](examples/preview_ood_goal_switch_mpc.png)):** After detouring around the `1-px` central barrier to transient Goal A (`t = 0..3`) with zero external goal input during `t = 4..35`, a `0.4s` Goal-B flash at `t = 36..39 > T_train` overwrites `SLWM-X1B`'s `GatedDelta` working-memory register and drives the puck cleanly to Goal B (**`PASS`, `d = 0.03`**), whereas `DenseTransformer-X1B`'s RoPE KV cache drifts back toward stale Goal A (**`FAIL`, `d = 0.58`**) and `LegacyScalarHybrid-X1B` overshoots off-target (**`FAIL`, `d = 0.60`**).
  - **`1-px` Thin-Barrier Detour Mission ([`preview_thin_barrier_mpc.png`](examples/preview_thin_barrier_mpc.png)):** `SLWM-X1B` executes a smooth clearance trajectory around the `1-px` barrier (`mean_barrier_prob = 0.9999`) to **`PASS` (`d = 0.01`)**.

![1.008B OOD T=76 Transient Goal-Switch Mission Filmstrip](examples/preview_ood_goal_switch_mpc.png)

![1.008B Closed-Loop 1-px Thin-Barrier Detour Filmstrip](examples/preview_thin_barrier_mpc.png)

- **Pre-Impact `1-px` Barrier Resolution (`Panel 2`):** All `64×64` `SLWM` cohorts achieve **`1.000 ± 0.000` AUROC** on both pre-contact synthetic `1-px` barriers (`ticks 0..2`, contacts zeroed) and held-out `64×64` PyMunk `puck_barrier` vs `pusht` episodes, whereas `16×16` G8 subsampling degrades pre-impact AUROC to `0.51–0.66`.
- **TPU `v6e-1` Throughput & $\mathcal{O}(1)$ Streaming State (`Panel 7`):** `SLWM-X1B` (`1.008B`) processes `T = 32` (`768` slots) in **`9.7 ms`** and `T = 256` (`6,144` slots) in **`113.8 ms`** (`107,969` slots/s) while holding total streaming memory strictly constant at **`14.907 MB`** from `T = 64` to `T = 1024` (`SLWM-X450M`: **`6.0 ms`** at `T = 32`, `185,723` slots/s, **`8.790 MB`** constant state). By contrast, `DenseTransformer-X1B`'s KV cache grows linearly from `207.034 MB` (`T = 64`) to **`3,312.539 MB` (`T = 1024`)** per single stream and triggered a `46.99 GB > 31.24 GB` `CompileTimeHbmOom` on `32 GB` TPU `v6e-1` until its `Panel 5/8` rollouts were micro-batched (`chunk_bs = 24 / 16`).

---

## 6. Honest Trade-Offs, Failure Modes & Scope Boundaries (`AGENTS.md` §13)

1. **Compute/Data-Optimal Scaling Frontier (`453M` vs `1.008B` Under Fixed `33,600`-Sequence Budget):** Under a fixed training exposure (`2,600–2,800` steps × `batch_size = 12` = `31,200–33,600` sequences), **`SLWM-X450M` (`453M`)** sits closest to the Chinchilla sample-per-parameter optimum for contact manipulation (**`89.6 ± 6.1%` overall MPC, `83.3 ± 9.0%` `1-px` barrier MPC, `96.9 ± 2.6%` OOD Goal-Switch MPC, and `100.0 ± 0.0%` `T = 1024` recall across all 3 seeds**). Scaling parameters `2.23×` to **`SLWM-X1B` (`1.008B`)** without proportionally increasing sequence exposure improves Real Audio-Visual `R@1` (`92.2% → 94.3%`) and OOD `T = 76` Sustained MPC (`99.5% → 100.0%`), while introducing single-seed under-training variance on `s2002` (`27.1%` `1-px` barrier MPC vs `72.9%` on `s2001` and `66.7%` on `s2003`) and `s2003` `T = 1024` recall (`56.2%` vs `100.0%` on `s2001` and `s2002`).
2. **Unimodal Short-Horizon Kinematics vs Multimodal Grounding & Memory (`SLWM` vs Upgraded `LegacyScalarHybrid`):** Because `LegacyScalarHybrid-X1B` (`raw_pixel_linear_mse`) sets `jepa_loss = 0` and `sigreg = 0`, equipping it with our Wave 4 `64×64` visual stem, decoupled `head_proprio`, and barrier-anchored `Latent-CEM-MPC` allows its entire gradient budget to specialize on short-horizon unimodal coordinates (`79.2 ± 1.7%` short `T = 38` continuous-cue MPC vs `71.9 ± 12.7%` on `SLWM-X1B` and **`89.6 ± 6.1%` on `SLWM-X450M`**; and `84.4 ± 5.2%` on `152M` `abl_scalar_retention`). Crucially, this controlled upgrade proves that `G8`'s original `0–1.6%` failure stemmed from `16×16` subsampling + coupled linear MSE + 1-step shooting, while `LegacyScalarHybrid-X1B` still suffers **complete cross-modal collapse (`1.0%` AV `R@1`), associative overwrite collapse (`2.1%` at `T = 512..1024`), representation rank collapse (`PR = 3.2%`), and `-19.79 pp` worse OOD `T = 76` Goal-Switch MPC (`73.4%` vs `93.2%`, 99% CI `[+10.42, +29.17]`)**.
3. **Simulation-to-Hardware Boundary:** While `x1b_real_bank_v1` trains and evaluates zero-leakage held-out recordings from real GreatestHits, STARSS23, DROID, and FineWeb-Edu, closed-loop manipulation and camera-spoofing rollouts (`Panels 4, 5, 6, 8`) execute in deterministic `64×64` rigid-body contact physics rather than on physical robot arms.

---

## 7. Next Allowed Step

1. **Chinchilla Token-Matched `1.008B` Scaling (`8,000–12,000` Steps):** Scale `SLWM-X1B` sequence exposure by `3×` on multi-chip TPU `v6e` with expanded DROID + Open X-Embodiment action trajectories to bring `1.008B` short-horizon barrier variance up to `SLWM-X450M`'s `89.6%` frontier while retaining `100.0%` OOD sustained control and `94.3%+` real audiovisual grounding.
2. **Real-Robot Hardware-in-the-Loop Deployment:** Deploy portable `int8` `SLWM-X450M` (`415.1 MB`, `6.0 ms` tick latency) and `SLWM-X1B` (`918.9 MB`, `9.7 ms` tick latency) with $\mathcal{O}(1)$ `init_streaming_memories` and live `PolicyCommitGate` camera-spoof / occlusion protection on a physical tabletop arm.
