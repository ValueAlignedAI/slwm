# SLWM-v3 Wave A — Gate 3 Pre-Run Architecture, Causality, Isolation & $\mu$P Specification (Rev 4)

**Status:** Gate 3 Pre-Run Engineering & Mathematical Audit Pack (Rev 4; pure-JAX, XLA-native, verified on CPU prior to accelerator execution)  
**Branch:** `research/slwm-v3-selective-memory`  
**Verified Code, Unit Tests & JSON Receipts:**
- Architecture Spec, Dual-Track State Counter & Exact Matched Comparator Factory: [spec.py](../../models/v3/spec.py)
- Strictly Causal $\mathcal{O}(1)$ Memory, Safe Multi-Head KV & Ring-Buffer GQA Primitives: [memory.py](../../models/v3/memory.py)
- Shared World Core, Balanced Edge Adapters, Chunked Remat, $\mu$P LR Multipliers & WD Mask: [core.py](../../models/v3/core.py)
- Single Multi-Horizon Frozen-Target Predictive Objective & `stop_gradient` Probe Isolation: [objectives.py](../../models/v3/objectives.py)
- Non-Vacuous Gate 3 Unit Test Suite (`17/17` passing): [test_v3_gate3_causality_and_mup.py](../../tests/test_v3_gate3_causality_and_mup.py)
- Exact Comparator Parity, Real-AdamW $\mu$P/CompleteP & Zero-Crutch Receipt: [gate3_architecture_and_mup_receipt.json](../../reports/v3/gate3_architecture_and_mup_receipt.json) ([v3_gate3_audit_receipt.py](../../evals/v3_gate3_audit_receipt.py))
- Accelerator Training & 4-Phase Validation Pilot Suite: [v3_pilot_train.py](../../scripts/train/v3_pilot_train.py), [v3_tpu_pilot_suite.py](../../scripts/train/v3_tpu_pilot_suite.py), [v3_colab_pilot_launcher.py](../../scripts/train/v3_colab_pilot_launcher.py)

---

## 1. General-Purpose Architecture Contract (Zero Hardcoded Task Crutches)

The shared world core processes a uniform stream of causal multimodal packets (`modality_id ∈ {0:video, 1:audio, 2:text_byte, 3:proprio_action, 4:signal_probe}`, physical elapsed `dt_s ≥ 0`, `valid_mask ∈ {0, 1}`, `reliability ∈ [0, 1]`) with **zero hardcoded slot indices, reserved token-ID ranges, task IDs, or overwrite opcodes** (verified statically across all 5 modules in `models/v3/`, with both `models_v3_sha256` and full `pilot_bundle_sha256` bound in the Gate 3 receipt). Edge projections are balanced to a common initial activation RMS (`~0.5`) across all 5 modalities (`max/min < 2.0`), and only the active audio path is allocated per `representation_mode` (`edge_audio_rvq_*` for `frozen_codec_edge`, `edge_audio_cont_w` for `scratch_edge`; zero dead edge parameters).

### 1.1 Content-Routed Multi-Timescale Matrix Memory ([`step_matrix_memory`](../../models/v3/memory.py#L304-L418))

Each layer maintains a strictly bounded $\mathcal{O}(1)$ `float32` matrix bank state $M_{t-1}^{(b,h)} \in \mathbb{R}^{d_v \times d_k}$ across $B = 4$ banks initialized at separated physical timescales $\tau_b \in \{0.25, 2.0, 16.0, 128.0\}\text{ s}$:

1. **Physical Decay:**
   $$\widetilde{M}_t^{(b,h)} = \alpha_{bt}^{(h)} M_{t-1}^{(b,h)}, \qquad \alpha_{bt}^{(h)} = \exp\!\left(-g_{bt}^{(h)}\,\frac{\Delta t_t}{\tau_b}\right), \qquad g_{bt}^{(h)} = \operatorname{softplus}(W_g x_t + b_g) > 0.$$
2. **Causal Read-Before-Write Retrieval Residual:**
   Using [`safe_l2_normalize`](../../models/v3/memory.py#L73-L79) (JAX double-`where`: exact unit norm $\|u\|_2 = 1$ whenever $\|x\|_2^2 > \epsilon = 10^{-8}$, returning $x$ unchanged with exact identity Jacobian $1.0 \cdot I$ below the $\epsilon$ cutoff) in `float32`, the model reads $\widetilde{M}_t$ *before* modifying state to compute $\hat{x}_{\text{prior}, t}$ and prior retrieval residual $e_t = \sqrt{\frac{1}{d}\|\operatorname{RMSNorm}(x_t) - \hat{x}_{\text{prior}, t}\|_2^2 + \epsilon}$.
3. **Learned Content Router + Explicit No-Write Channel:**
   A two-layer SiLU MLP (`router_hid = max(32, d // 4)`, scaling proportionally with $d$ across $d \in \{128, 256, 512, 768\}$) over $[\operatorname{RMSNorm}(x_t), e_{\text{mod}}, \log(1+\Delta t_t), \text{reliability}_t, \log(1+e_t)]$ outputs $[r_{1,t}, \dots, r_{B,t}, r_{\text{nowrite}, t}] \in \Delta^B$. Missing packets (`valid_mask_t = 0`) multiply $r_{b,t}$ by $0$, executing a **bit-exact zero content write** ($M_t = \widetilde{M}_t$, and $M_t = M_{t-1}$ when $\Delta t_t = 0$).
4. **Normalized-Key Rank-1 Delta Overwrite (`float32`) & Exact Stability Conditions:**
   With $\|k_t^{(h)}\|_2 \le 1$ (`1.0` when $\|k_{\text{raw}}\|_2^2 > 10^{-8}$) and $\beta_t^{(h)} = \sigma(W_\beta x_t + b_\beta) \in (0, 1)$:
   $$M_t^{(b,h)} = \widetilde{M}_t^{(b,h)} + r_{bt}\,\beta_t^{(h)}\bigl(v_t^{(h)} - \widetilde{M}_t^{(b,h)} k_t^{(h)}\bigr)(k_t^{(h)})^\top.$$
   Because $\eta = r_{bt}\beta_t^{(h)} \in [0, 1]$ and $\|k_t^{(h)}\|_2 \le 1$, the rank-1 update operator $(I - \eta k k^\top)$ has eigenvalues in $[0, 1]$ (with eigenvalue $1$ in directions orthogonal to $k$). **Exact uniform boundedness condition:** under **uniformly bounded forgetting** $\alpha_{bt}^{(h)} \le \bar{\alpha} < 1$ and bounded values $\|v_t\|_2 \le V_{\max}$, the Frobenius norm $\|M_t^{(b,h)}\|_F$ stays uniformly bounded as $T \to \infty$; conversely, if $\alpha_{bt}^{(h)} = 1$ (e.g., $\Delta t_t = 0$) or $\alpha_t \to 1$ slowly alongside orthogonal unit keys and non-square-summable write energy ($\sum_t \eta_t^2 v_t^2 = \infty$), additive variance accumulates without uniform bound ($\|m_t\|_2^2 = \alpha_t^2 \|m_{t-1}\|_2^2 + \eta_t^2 v_t^2$).
5. **Local Ring-Buffer GQA, Chunked Rematerialization & Extra Zero-Lag Bypass:**
   Each layer pairs the memory branch with a strictly causal $\mathcal{O}(1)$ sliding-window GQA ring buffer (`[B, W, H_kv, d_head]` in `bfloat16`) and CompleteP $\alpha = 1.0$ depth residual branch scaling $L_{\text{base}} / L$ ([`SLWMv3Spec.depth_residual_scale`](../../models/v3/spec.py#L135-L138), `arXiv:2505.01618`, Table 1; using standard fixed Adam $\epsilon = 10^{-8}$ rather than full width/depth $\epsilon$ rescaling). Long-sequence scans (`scan_core`) use outer chunked `jax.checkpoint` rematerialization (`remat_chunk_size = 64`), cutting saved-carry HBM by $64\times$ while matching `step_core`. After the $L$-layer stack, a learned gated bypass $Z_t = \operatorname{RMSNorm}(h_t^{(L)}) + \sigma(\gamma_{z0}) \odot W_{z0} x_{\text{raw}, t}$ augments the standard residual stream.

### 1.2 Fair Single-Factor Ablation & Comparator Contracts

- **`-LearnedRouting` ([`slwm_v3_no_routing`](../../models/v3/spec.py#L325-L330)):** Holds physical timescales $\tau_b$, bank decay gates $g_b$, learned read routing `w_read_gate`, AND the identical `(nb + 1)`-logit softmax no-write probability `r_nowrite` (nominal `0.20` no-write / `0.80` write mass at zero logits; exact `+0.000%` parameter parity!) strictly identical to `slwm_v3_routed`, replacing ONLY content-routed bank write selection with a **content-independent round-robin one-hot selector** `r_banks = one_hot(step_idx % nb) * (1 - r_nowrite)`. This preserves exact parameter count, exact no-write salience calibration, exact aggregate write budget $\sum_b r_b = 1 - r_{\text{nowrite}}$, and the full $[0, 1]$ single-bank write ceiling with zero extra state bytes, isolating content-dependent bank routing alone. *(Note: supersedes Gate 1's preliminary tied-timescale sketch to eliminate the multi-mechanism confound; `single_bank_gated_deltanet` separately covers the single-timescale non-routed baseline at equal matrix bytes.)*
- **`-NoWriteOption` (`slwm_v3_no_nowrite`):** Removes the explicit `r_nowrite` simplex channel while retaining $\beta_t \in (0, 1)$; all training and evaluation logs report `mean_eff_write_eta` ($r_b \beta$), `mean_eff_decay`, and `mean_state_delta_norm` alongside `mean_nowrite`.
- **`-ZeroLagResidual` (`slwm_v3_no_zerolag`):** Isolates the marginal value of the extra gated sensory bypass projection on top of the standard residual stream.
- **`single_bank_gated_deltanet`:** Core single-bank ($B = 1$) gated-delta matrix recurrence with $2\times(d_k, d_v)$ (exact equal matrix bytes), single timescale $\tau = 4.0\text{ s}$, and **no** surprise/no-write router MLP or `w_prior_proj` *(explicitly scoped to the gated-delta recurrence without the full published block's short convolutions or output gate)*.
- **`diag_ssm`:** Diagonal scalar-decay SSM comparator (`dt_pos = max(dt_s, 0.0)`, zero spurious decay at $\Delta t = 0$); distinct from a structured-semiseparable Mamba-2 block.
- **`explicit_bounded_kv` ([`step_explicit_kv`](../../models/v3/memory.py#L517-L645)):** Competitive differentiable multi-head bounded LRU/content-overwrite KV controller (`[B, H_kv=4, K_slots, D_kv]`), with matched 2-layer SiLU write-salience MLP, differentiable match gate `p_match = sigmoid(10*(best_sim - thresh))`, monotonic logical-tick + physical-time LRU aging (`dt_pos + 1e-3 * valid_mask`, preventing `dt=0` read-refresh tie eviction), clean empty/evicted replacement (zero victim-value bleed), and exact zero-write key preservation (`active_ow = kv_valid & (match_w > 1e-7)`).

---

## 2. Single Coherent Predictive Objective & Gradient Isolation

- **Wave A Predictive Loss ([`masked_multi_horizon_predictive_loss`](../../models/v3/objectives.py#L31-L112)):** One Huber objective predicting fixed, train-normalized future packet features at horizons $\Delta \in \{1, 4, 8\}$ ticks (zero EMA moving targets in the validation ruler). Action conditioning enters strictly via `has_true_action ∈ {0, 1}` paired with a 2-way `action_avail_emb`, separating executed zero actions from missing actions in both the core and predictive head. Packed sequences compute episode IDs from `reset_mask` so supervision never crosses episode boundaries.
- **Strictly Decoupled Readouts ([`apply_decoupled_probe`](../../models/v3/objectives.py#L127-L131)):** Every downstream diagnostic head applies `jax.lax.stop_gradient(Z_t)` on an isolated parameter tree, verified non-vacuously with non-zero `probe_w2` to yield `|grad(probe_w1)| > 1e-5`, `|grad(probe_w2)| > 1e-5`, `max(|grad_core|) == 0.0`, and bit-exact core weight invariance.

---

## 3. Verified Gate 3 Invariants (`17/17` Non-Vacuous Unit Tests Passing)

All invariants below are enforced by [test_v3_gate3_causality_and_mup.py](../../tests/test_v3_gate3_causality_and_mup.py):
1. **Raw-Input Suffix Invariance, Online == Offline & Missing-Packet Invariance (ALL 8 backbones):** Prefix `max(|Z_base[:12] - Z_alt[:12]|) == 0.0`, suffix response `> 1e-4`; `step_core == scan_core` to `< 1e-3` (`bf16` ring KV) and `< 1e-5` (`fp32` outputs & final state); missing packets (`valid_mask = 0, dt_s = 0`) leave persistent state bit-exact unchanged across all 8 backbones.
2. **Chunked Rematerialization Equivalence:** Outer chunked `jax.checkpoint` (`remat_chunk_size = 8` on non-multiple $T = 25$) matches unchunked `scan_core` to `< 1e-5` forward and `< 1e-4` full trunk gradients.
3. **Zero Streaming State Growth & Non-Vacuous Boundary/Action/Probe Isolation:** Constant state bytes from $T = 1$ to $T = 256$; bit-exact reset at $t = 12$; positive no-reset control (`> 1.0` diff) vs `0.0` diff with `reset_mask`; non-zero `pred_head_w` action causality (`[:8] == 0.0`, `[8:] > 1e-4`); non-zero `probe_w2` isolation (`|grad(probe_w1)| > 1e-5`, `|grad_core| == 0.0`).
4. **Fair Ablations & Controlled Explicit KV Eviction/Overwrite Verification:** Confirmed `-LearnedRouting` matches `slwm_v3_routed` bit-exactly in parameter count, `r_nowrite` (nominal `0.20` at zero logits), and aggregate write mass `sum_b r_b == 1 - r_nowrite` (nominal `0.80`) while cycling deterministically round-robin; confirmed `explicit_bounded_kv` cleanly replaces the exact oldest victim slot on orthogonal novel input (zero victim-value bleed; non-victim values within `< 1e-2` soft-match tail), smoothly overwrites matched keys (`>94%`) without evicting the oldest slot, preserves zero-write valid keys bit-exact, protects `dt = 0` read-refreshed slots from tie eviction, and sustains finite `T = 40` gradients with `|grad(thresh)| > 1e-8`.
5. **Phase 3 Observable Sensory Echo Oracle, Phase 2 Strict Verdict Fixtures & Launcher Guardrails:** Confirmed `100.0%` analytic causal oracle accuracy across clean/distractor/background-calibrated `3.5-sigma` streams with real single-tick observable future sensory echoes on `pkts["probe_feat"]` (`targs` derived purely from `probe_feat`, `0%` delayed-value copy shortcut, `post_corr_pre_echo_max_abs_diff == 0.0`); confirmed Phase 2 verdict rejects any-scale boundary, non-finite, unrounded `0.0150004 > 0.015` regret, and cross-grid width/depth LR mismatch fixtures; confirmed `is_tpu_v6e_device_kind` accepts both `TPU v6e` and `TPU v6 lite`; confirmed file-backed watchdog handles `> 128 KiB` stdout without pipe deadlock; confirmed pre-phase projected reserve check (`41.90 -> 42.05 >= 42.00`), strict `verified_stop_session` (rejecting `stop rc=1`, empty `sessions` output, and `[?]`, recovering from timeout), TOCTOU-free single-read bundle verification, and strict reusable Phase 2 receipt locks.

---

## 4. Exact Base-Shape $\mu$P, CompleteP Residual Scaling & Masked Independent Weight Decay

Under exact base-shape $\mu$P ($m_d = d / d_{\text{base}}$):
- **Initialization & LR Multipliers:** Every leaf has LR multiplier `1.0` at $d = d_{\text{base}}$ across all 8 backbones (exact SP identity at base width); width-scaling 2D hidden/readout weights (including `router_w1`/`router_w2` whose hidden dimension `max(32, d // 4)` scales proportionally across $d \in \{128, 256, 512, 768\}$) receive `1 / m_d`; fixed-fan-in projections (`mem.w_o`, `mem.w_prior_proj`, `kv.w_o`), inputs, and scalars receive `1.0`.
- **Masked Width-Invariant Independent Weight Decay:** `wd = 0.01` is applied *after* per-leaf LR scaling (`delta_p = -(base_lr * m) * adam(g) - (base_lr * wd) * wd_m * p`) and strictly masked via [`build_weight_decay_mask`](../../models/v3/core.py#L497-L509) so `log_tau`, `log_a`, `norm_w`, `rel_pos_bias`, `gate_z0`, `thresh`, and all 1D biases receive `0.0` weight decay.

Production masked `optax` AdamW 5-step coordinate audits on 100%-observable masked batches ([gate3_architecture_and_mup_receipt.json](../../reports/v3/gate3_architecture_and_mup_receipt.json)) confirm:
- **Width $\mu$P (`L = 4`, `d ∈ {128, 256, 512}`):**
  - `init_z_rms`: `1.00175 / 1.00055 / 1.00107` (**max/min ratio `1.0012`**).
  - `step5_delta_z_rms`: **$\mu$P `0.83708 / 0.81470 / 0.81450` (ratio `1.0277`; loss `0.27718 / 0.27700 / 0.27741`)** vs. uniform-LR SP `0.83708 / 0.94506 / 1.00649` (ratio `1.2024`; loss `0.27718 / 0.19822 / 0.10920`).
  - `mean_mem_out_rms`: `0.01173 / 0.01282 / 0.01389`; `mean_key_norm`: `1.000000`.
- **Separate Depth CompleteP Residual Scaling (`d = 128`, `L ∈ {4, 6, 8}`, $\alpha = 1.0$ `L_base / L`, fixed Adam $\epsilon = 10^{-8}$):**
  - `step5_delta_z_rms`: **`0.83708 / 0.82764 / 0.80574` (ratio `1.0389`; loss `0.27718 / 0.27743 / 0.27793`)**.
- **All 8 Wave A Backbones (`T = 40`, 3 steps real AdamW):** `8/8` finite gradients and finite losses.

---

## 5. Dual-Track (`Core` + `Strict System`) Comparator Parity (`±0.65%` Params Across All 3 Scale Profiles)

Exact parameter counts (`jax.eval_shape`, verified within strict **`±1.0%`** tolerance in receipt, unit tests, and pilot suite) and runtime-verified persistent streaming state footprints (`fp32` memory + `bf16` ring KV, combined with Gate 2 Rev 2 frozen encoder totals `92,988,448` params / `168.444 MiB` bf16) are recorded in [gate3_architecture_and_mup_receipt.json](../../reports/v3/gate3_architecture_and_mup_receipt.json):

| Scale Profile | Anchor (`slwm_v3_routed`) Core Params | All 9 Arms Param $\Delta$ Range | Anchor Core State (`bf16` ring + `fp32` mem) | Strict System Total Enc Params | Strict System Streaming State (`bf16`) |
|---|---:|---:|---:|---:|---:|
| **`pilot_d256`** (`L=12, W=32`, RF `372` / `W_eq=64` RF `756`) | `12,964,156` | `-0.507%` to `+0.554%` | `0.3754 MiB` (`393,648 B`) | `105,952,604` | `168.819 MiB` |
| **`discovery_45m`** (`L=12, W=32`, RF `372` / `W_eq=64` RF `756`) | `40,474,940` | `-0.649%` to `+0.358%` | `0.3754 MiB` (`393,648 B`) | `133,463,388` | `168.819 MiB` |
| **`confirmation_135m`** (`L=16, W=64`, RF `1,008` / `W_eq=128` RF `2,032`) | `130,158,864` | `-0.454%` to `+0.223%` (all `129.57M–130.45M`) | `2.0010 MiB` (`2,098,240 B`) | `223,147,312` | `170.445 MiB` |

Every byte-matched comparator (`slwm_v3_no_*`, `single_bank_gated_deltanet`, `sliding_window_gqa_eqbyte`, `diag_ssm`, `explicit_bounded_kv`) matches total core streaming state bytes within **`-0.781%` to `+0.975%`** across all three scale profiles.
