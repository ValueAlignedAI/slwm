# Study X1B Pre-Registered Protocol — Spatiotemporal Register-Delta World Models at ~1.00B Scale (`SLWM-X1B`)

**Sprint / Study ID:** `X1B-20261111`  
**Branch:** `research/x1b-spatiotemporal-jepa-mpc-20261111`  
**Date:** `2026-11-11`  
**Scale Profiles:** `SLWM-X150M` (~152M factorial anchor), `SLWM-X450M` (~455M scaling bridge), `SLWM-X1B` (**~1.002B flagship**, strict total parameter accounting)  
**Operational Isolation:** 100% disjoint in scientific scope, local workspace (`/Users/db/.gemini/antigravity/scratch/slwm-x1b`), Colab sessions (`slwm-x1b-*`), and Fedora storage (`/mnt/slwm-data/SLWM/x1b`, `/mnt/slwm-checkpoints-{a,b}/SLWM/x1b`) from concurrent study `G11`.

---

## 1. Scientific Motivation & Root-Cause Diagnosis of Prior `G4–G10` Bottlenecks

Prior `SLWM` studies (`G4–G10`, 12.6M–13.4M parameters) established rigorous falsification infrastructure (`AGENTS.md`) yet exposed five compounding architectural and algorithmic bottlenecks that capped world-model capability:

1. **Destructive Edge Signal Compression (`1×128` vector/tick):** Crushing video to `16×16` flat RGB/grayscale erased sub-centimeter barriers (`G8` 5-pixel wall zero-pixel bug), T-block rotational contacts, and fine affordances; `G10` hashed words into 128 `blake2s` bins with severe collisions; all modalities were summed into a single `128-D` vector per 100 ms tick.
2. **Blurry Mean Collapse & Persistence Deficit from Linear Pixel/Spectral MSE:** Optimizing single-matrix linear MSE readouts (`z @ p["read"]`) caused deterministic conditional-expectation blur (`-2.3%` to `-33.8%` skill on held video) and lost to zero-order persistence ($x_{t+1} = x_t$) on 100 ms sensor forecasting (`1.684` vs `0.418` RMSE).
3. **Scalar Exponential Decay Amnesia & Key-Value Overwrite Failure:** `retention` (`models/g5_shared.py`) maintained an 8-group scalar EMA without key-value matrix addressing or delta-rule erasure, yielding ~50% delayed-calibration recall at 480 ticks and losing ~57 percentage points when goal instructions were transient (`0.4 s`) rather than repeated every tick.
4. **Myopic 1-Step Random Shooting Control:** `G8` evaluated 3 random open-loop action branches over a single decision window, achieving `0.0–1.6%` closed-loop Push-T success.
5. **Unimplemented Policy/Uncertainty Commitment:** `PolicyCommitGate` (`models/policy/commit_gate.py`) remained a 0-parameter rule stub, leaving the hybrid *more* vulnerable than dense attention to `G6`'s wrong-camera spoofing attack (lower dynamics error caused overconfident execution of spoofed visual plans).

---

## 2. Core Hypotheses & Falsification Gates

### H1 — Structured Spatiotemporal Register Field (`Z[B, T, S=24, D]`) Resolves Thin-Geometry & Transient-Goal Blindness
Replacing the `1×128` vector bottleneck with a structured per-tick field of `16` spatial visual patch tokens (`4×4` over `64×64×3` RGB via overlapping `3×3` conv stem), `2` spectral/onset audio tokens, `1` proprioceptive state token, `1` 4096-vocab subword text/code token, `1` continuous motor action token, and `3` persistent working-memory register slots (`R_0..R_2`) will:
- Detect thin (`5 px` arena) physical barriers with **>= 90% contact/branch discrimination accuracy** (vs. near-chance at `16×16`),
- Retain `0.4 s` transient goal instructions across `64+` silent ticks with **< 5 percentage-point drop** relative to continuous goal repetition *without* external input latching.
- **Falsification gate:** Falsified if `No-Registers` or `Legacy-1xD` matches `SLWM-X1B` within 95% bootstrap CI on transient-command control or thin-barrier branch ranking.

### H2 — Multi-Head Gated Delta Recurrence Outperforms Scalar Retention & Scales Sub-Quadratically vs. Dense Attention
Updating causal per-head matrix memory via the Gated Delta Rule:
$$M_t^{(h)} = \alpha_t^{(h)} M_{t-1}^{(h)} + \beta_t^{(h)} \bigl(v_t^{(h)} - M_{t-1}^{(h)} k_t^{(h)}\bigr) {k_t^{(h)}}^\top, \qquad o_t^{(h)} = M_t^{(h)} q_t^{(h)}$$
interleaved 3:1 with Causal RoPE Temporal Attention and within-tick Spatial/Cross-Modal Attention will:
- Exceed **85% accuracy** on `512`-tick delayed associative key-value calibration with mid-episode polarity overwrites (where scalar `retention` remains near 50–60%),
- Generalize stably from training horizon $T=32$ to evaluation horizons $T \in \{128, 256, 512\}$ while maintaining fixed recurrent state size per Gated Delta layer.
- **Falsification gate:** Falsified if `Scalar-Retention` matches `Gated-Delta` on 512-tick associative calibration or if `DenseTransformer-X1B` dominates both accuracy and long-horizon extrapolation at matched 1.002B parameters.

### H3 — Residual Action-Conditioned EMA-JEPA + `SIGReg` Beats Linear Pixel MSE & Persistence
Training a residual latent dynamics predictor $\hat{z}_{t+1} = z_t + \Delta_\theta(z_{\le t}, u_t)$ against a stop-gradient EMA target encoder ($\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau)\theta$) regularized by isotropic variance-covariance regularization (`SIGReg`/`VICReg`) plus heteroscedastic innovation log-variance $\log \hat{\sigma}_t^2$ will:
- Prevent representation collapse (**Participation Ratio >= 35% of latent dimension**, vs. < 5% without `SIGReg`),
- Strictly beat zero-order persistence ($x_{t+1} = x_t$) on 1-step and multi-step physical state & real impact audio-visual forecasting (**Skill Score > +25% over persistence**, reversing G10's `-2.3%` to `-33.8%`),
- Preserve cross-modal retrieval (`R@1`, `R@5`) on real `GreatestHits` (`vis-data-256`) `64×64` impact video/audio and `STARSS23` spatial audio-video while maintaining competitive `FineWeb-Edu` text perplexity within `AGENTS.md` H2 tolerance (`<= +15%` NLL vs dense Transformer).
- **Falsification gate:** Falsified if `Raw-Pixel-Linear-MSE` or `Direct-JEPA-No-Residual` beats `Residual-EMA-JEPA + SIGReg` on downstream physical forecasting or closed-loop control, or if forecast RMSE loses to persistence.

### H4 — Multi-Horizon `Latent-CEM-MPC` + Innovation-Calibrated `PolicyCommitGate` Solves Closed-Loop Manipulation & Defeats Sensor Spoofing
1. Planning over learned residual latent rollouts (`H = 12` ticks, `3` CEM iterations × `64` candidate trajectories scored by terminal Value Critic + uncertainty penalty) will achieve **>= 65% closed-loop target/barrier manipulation success** on `PushT-64` and `ThinBarrier-64` (vs. `<= 15%` for G8-style 1-step 3-candidate random shooting).
2. Routing external proposals through a learned `PolicyCommitGate` conditioned on cross-modal Mahalanobis innovation surprise:
   $$s_{\text{innov}}(t) = \frac{1}{M}\sum_{m} \frac{\|z_{t,m} - \hat{z}_{t,m}(z_{t-1}, u_{t-1})\|^2}{\hat{\sigma}_{t,m}^2 + \epsilon}$$
   will detect `G6`'s **Wrong-Camera Spoofing** and `G8`'s **Hidden Physical Wall**, suppressing spoofed visual plans / issuing safe `probe` or `noop` commitments and cutting spoofed-sensor regret by **>= 60%** without collapsing to abstention on clean inputs (**clean task usefulness >= 90%** of unconstrained policy).
- **Falsification gate:** Falsified if `Always-Commit` or `FixedRulePolicyCommitGate` achieves equal or lower regret at matched clean-sensor usefulness.

### H5 — Positive Scaling Law from `152M` → `455M` → `1.002B` Parameters
Across strict parameter-accounted rungs (`152.4M`, `454.8M`, `1,002.1M`) under matched architecture and objective, validation latent JEPA prediction skill, real audio-visual retrieval, FineWeb-Edu language modeling, and closed-loop MPC success improve monotonically with model scale, and at `1.002B` parameters `SLWM-X1B` outperforms both `DenseTransformer-X1B` (`1,002.1M`) and `LegacyScalarHybrid-X1B` (`1,002.1M`).

---

## 3. Strict Parameter Accounting & Scale Rungs

All headline comparisons use **`AGENTS.md` §4 Mode 1 (Strict Total Parameter Comparison)**: every parameter in modality codecs, spatial/temporal core blocks, registers, JEPA predictor, uncertainty head, learned `PolicyCommitGate`, value/action heads, and diagnostic probes is counted and logged.

| Profile | Layers $L$ | Hidden $D$ | Heads $H$ ($d_h$) | FFN $D_{\text{ff}}$ | Slots/Tick $S$ | Strict Total Params | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `SLWM-X150M` (Factorial Anchor) | 13 | 768 | 12 (64) | 2048 | 24 | `152,003,766` (~152.0M) | TPU `v6e1` / `H100` |
| `SLWM-X450M` (Scaling Bridge) | 18 | 1152 | 18 (64) | 3072 | 24 | `453,179,166` (~453.2M) | TPU `v6e1` / `H100` |
| `SLWM-X1B` (**Flagship 1B**) | 23 | 1536 | 24 (64) | 4096 | 24 | **`1,008,426,726` (~1.008B)** | TPU `v6e1` (preferred) / `H100` |
| `DenseTransformer-X1B` (Baseline) | 23 | 1536 | 24 (64) | 4409 | 24 | **`1,008,407,430` (0.0019% delta)** | TPU `v6e1` (preferred) / `H100` |
| `LegacyScalarHybrid-X1B` (G10-at-1B) | 23 | 1536 | 24 (64) | 4409 | 24 (pooled/scalar) | **`1,008,407,430` (0.0019% delta)** | TPU `v6e1` (preferred) / `H100` |

---

## 4. Parallel 4-Lane Experimental Execution Plan

- **Lane 1 (`slwm-x1b-lane1-flagship`):** Flagship `SLWM-X1B` (`1.002B`) + Scaling Bridge `SLWM-X450M` (`455M`) trained on the unified multimodal real + high-res `64×64` physical world corpus; full evaluation + dual-HDD checkpoint verification.
- **Lane 2 (`slwm-x1b-lane2-baselines1b`):** Exact-parameter-matched `DenseTransformer-X1B` (`1.002B`) + `LegacyScalarHybrid-X1B` (`1.002B`) trained on identical seed, batch order, and token/patch exposure.
- **Lane 3 (`slwm-x1b-lane3-mem-ablation`):** 3-seed (`2001, 2002, 2003`) Architecture & Memory Factorial at `152M` (`SLWM-Full`, `No-Registers`, `Scalar-Retention`, `Dense-Attn-Only`, `Separate-Modality-Cores`), evaluated on 512-tick associative recall, 0.4s transient commands, and horizon extrapolation (`T = 32 → 512`).
- **Lane 4 (`slwm-x1b-lane4-jepa-policy`):** 3-seed (`2001, 2002, 2003`) Objective, Planning & Commit-Gate Factorial (`Residual-EMA-JEPA+SIGReg`, `Raw-Pixel-Linear-MSE`, `Direct-JEPA-No-Residual`, `JEPA-No-SIGReg`) + Closed-Loop `Latent-CEM-MPC` vs. `1-Step Random Shooting` + `Innovation-CommitGate` under Wrong-Camera Spoofing, Hidden Barrier, and Unsupported Queries.

---

## 5. Statistical & Custody Standards

1. Paired scene/recording/episode bootstrap (`10,000` resamples) reporting mean and 95% confidence intervals for all headline differences.
2. Zero train/dev/test recording or scene leakage (enforced by recording SHA-256 manifests).
3. All `1.002B` and `455M` and canonical `152M` checkpoints (`weights.npz` + `portable_int8.npz` + `receipt.json`) verified bit-for-bit via SHA-256 across **both** physical 9.1 TB HDDs on Fedora (`/mnt/slwm-checkpoints-a/SLWM/x1b` and `/mnt/slwm-checkpoints-b/SLWM/x1b`).
4. Hard compute budget ceiling: `<= 500.0` Colab compute units (target `< 120.0` units via TPU `v6e1` preference); mandatory `colab stop` verification of zero remaining `slwm-x1b-*` runtimes upon completion.

---

## 6. Wave 4 (`X1B-v4`) 3-Seed `1.008B` & `T=1024` / OOD Goal-Switch Expansion (`36` Runs; `120` Cumulative)

Following Waves 1–3 (`84` runs), Wave 4 (`X1B-v4`, `36` runs) resolves the remaining statistical power, open-loop policy readout coupling, and long-horizon OOD control tests:

1. **Decoupled Physical & Working-Memory Readout Heads (`1,008,454,410` Strict Params, `< 0.30%` Dense / `< 0.01%` Legacy Parity):** Separate `head_vis_pose_w/b`, `head_proprio_w/b`, and `head_goal_w/b` with decoupled core/head AdamW gradient clipping eliminate cross-objective interference between camera-spoof pose verification, residual state forecasting, and `0.4 s` working-memory goal retention, while bounding streaming recurrent + `3:1` attention state to $O(1)$ (`6.79 MB` at both `T=64` and `T=1024` vs `7,200.00 MB` for `DenseTransformer-X1B` at `T=1024`).
2. **4-Step Unrolled Residual-EMA-JEPA + Off-Policy Calibration:** Unrolled latent rollouts (`jepa_1 + 0.5*jepa_2 + 0.35*jepa_4` + `8.0*prop_step4_loss`) with 15% uniform off-policy motor perturbations and `0.75 * prop_delta + 0.25 * vis_delta` spatial-visual fusion sharpen multi-step `1-px` barrier collision forecasting.
3. **3-Seed `1.008B` & `453M` + 8-Panel OOD Evaluation (`36` Runs):**
   - `9` `1.008B` runs: `3 seeds (2001, 2002, 2003) × {SLWM-X1B, DenseTransformer-X1B, LegacyScalarHybrid-X1B}` (`steps=2800, batch_size=12, seq_len=32`, `N=288` pooled `Panel 5` episodes and `N=192` pooled `Panel 8` OOD `T=76` episodes per architecture).
   - `3` `453M` runs: `3 seeds (2001, 2002, 2003) × SLWM-X450M` (`steps=2600`).
   - `24` `152M` runs: `3 seeds (2001, 2002, 2003) × 8` architecture/objective factorials (`steps=1800`).
   - Expanded `Panel 3` to **`T=1024` (`32×` OOD = `24,576` spatiotemporal slots)** and added **`Panel 8`** (`T=76 = 2.38× T_train` Sustained Transient Thin-Barrier MPC & Mid-Episode Goal-B Overwrite MPC at `t=36..39 > T_train`, zero external goal vector).
