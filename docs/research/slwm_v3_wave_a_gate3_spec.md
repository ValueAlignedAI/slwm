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

### 1.1 Head-Routed Multi-Timescale Selective Matrix Memory ([`step_matrix_memory`](../../models/v3/memory.py#L317-L426))

Each layer maintains a strictly bounded $\mathcal{O}(1)$ `float32` matrix bank state $M_{t-1}^{(b,h)} \in \mathbb{R}^{d_v \times d_k}$ across $B = 4$ dedicated timescale banks (`n_mem_heads = 1` per bank, total `H_tot = 4`, full independent rank `d_k = d_v = 32` at `pilot_d256`/`discovery_45m` and `64` at `confirmation_135m` — matching `single_bank_gated_deltanet` bit-for-bit in both state bytes and per-head key/value rank!) initialized at separated physical timescales $\tau_b \in \{1.0, 8.0, 64.0, 512.0\}\text{ s}$:

1. **Physical Decay:**
   $$\widetilde{M}_t^{(b,h)} = \alpha_{bt}^{(h)} M_{t-1}^{(b,h)}, \qquad \alpha_{bt}^{(h)} = \exp\!\left(-g_{bt}^{(h)}\,\frac{\Delta t_t}{\tau_b}\right), \qquad g_{bt}^{(h)} = \operatorname{softplus}(W_g x_t + b_g) > 0.$$
2. **Causal Read-Before-Write Retrieval Residual & Bounded Per-Head Output RMSNorm:**
   Using smooth $C^\infty$ [`safe_l2_normalize`](../../models/v3/memory.py#L75-L79) ($u = x / \sqrt{\|x\|_2^2 + 10^{-6}}$, strictly $\|u\|_2 < 1$ everywhere, exact $0$ at $x = 0$, zero step discontinuity) in `float32`, the model reads $\widetilde{M}_t$ across all independent timescale heads with explicit `jax.lax.Precision.HIGHEST` (preventing silent TPU `bf16` mantissa truncation on `fp32` memory states) *before* modifying state, applies bounded per-head RMSNorm (`o_norm_w`, $\epsilon_{\text{norm}} = 10^{-2}$, exact `0` at $M = 0$, zero weight decay), projects to $\hat{x}_{\text{prior}, t}$, and computes prior retrieval residual $e_t = \sqrt{\frac{1}{d}\|\operatorname{RMSNorm}(x_t) - \hat{x}_{\text{prior}, t}\|_2^2 + \epsilon}$.
3. **Direct-Skip + 2-Layer Surprise Router & Per-Bank Explicit No-Write Channel:**
   Over $[\operatorname{RMSNorm}(x_t), e_{\text{mod}}, \log(1+\Delta t_t), \text{reliability}_t, \log(1+e_t)]$, a direct linear skip (`router_skip`) plus two-layer SiLU MLP (`router_hid = max(32, d // 4)`) outputs per-timescale-head write routing gates $\rho_{bt}^{(h)} \in (0, 1)$ and per-bank explicit no-write probabilities $r_{\text{nowrite}, bt} \in (0, 1)$, yielding $\eta_{bt}^{(h)} = \rho_{bt}^{(h)}(1 - r_{\text{nowrite}, bt})\,\text{valid}_t \in [0, 1]$. Missing packets (`valid_mask_t = 0`) execute a **bit-exact zero content write** ($M_t = \widetilde{M}_t$, and $M_t = M_{t-1}$ when $\Delta t_t = 0$).
4. **Normalized-Key Rank-1 Delta Overwrite (`float32` `Precision.HIGHEST`), Concatenated Readout & Exact Stability Conditions:**
   With smooth $\|k_t^{(b,h)}\|_2 < 1$ and $\eta_{bt}^{(h)} \in [0, 1]$:
   $$M_t^{(b,h)} = \widetilde{M}_t^{(b,h)} + \eta_{bt}^{(h)}\bigl(v_t^{(b,h)} - \widetilde{M}_t^{(b,h)} k_t^{(b,h)}\bigr)(k_t^{(b,h)})^\top.$$
   For fixed gates, $(I - \eta k k^\top)$ has eigenvalues in $[0, 1]$; because $\eta_t$ reads $M_{t-1}$ through the surprise residual, Phase 4 additionally audits pre-`o_norm_w` Frobenius norms $\|M_t\|_F$ and perturbation growth on long zero-time ($\Delta t = 0$) conflicting-key streams. Post-write readouts across all $B$ timescale heads are concatenated through bounded per-head `o_norm_w` before `w_o`. **Exact uniform boundedness condition:** under **uniformly bounded forgetting** $\alpha_{bt}^{(h)} \le \bar{\alpha} < 1$ and bounded values $\|v_t\|_2 \le V_{\max}$, $\|M_t^{(b,h)}\|_F$ stays uniformly bounded as $T \to \infty$.
5. **Local Ring-Buffer GQA, Chunked Rematerialization & Extra Zero-Lag Bypass:**
   Each layer pairs the memory branch with a strictly causal $\mathcal{O}(1)$ sliding-window GQA ring buffer (`[B, W, H_kv, d_head]` in `bfloat16`) and CompleteP $\alpha = 1.0$ normalized depth residual branch scaling `0.25 * (L_base / L)` (`3.0 / L` at `L_base = 12`; [`SLWMv3Spec.depth_residual_scale`](../../models/v3/spec.py#L135-L138), `arXiv:2505.01618`, Table 1; using standard fixed Adam $\epsilon = 10^{-8}$). Long-sequence scans (`scan_core`) use outer chunked `jax.checkpoint` rematerialization (`remat_chunk_size = 64`), cutting saved-carry HBM by $64\times$ while matching `step_core`. After the $L$-layer stack, a learned gated bypass $Z_t = \operatorname{RMSNorm}(h_t^{(L)}) + \sigma(\gamma_{z0}) \odot W_{z0} x_{\text{raw}, t}$ augments the standard residual stream.

### 1.2 Fair 2×2 Factorial & Single-Factor Ablation Contracts (12 Arms Total)

- **2×2 Factorial Cross (`Timescale Prior × Controller`):** Crosses `(Spread τ=(1,8,64,512)s vs Tied τ=64s)` × `(Surprise Router + Per-Bank NoWrite vs Linear DeltaNet w_beta)` across `slwm_v3_routed`, `slwm_v3_tied_tau`, `single_bank_spread_tau`, and `single_bank_gated_deltanet` — all four holding `H_tot = 4` independent per-head decay gates, full-rank `(d_k, d_v)`, `o_norm_w`, and exact matrix state bytes identical.
- **`-LearnedRouting` (`slwm_v3_no_routing`):** Holds parameters (`+0.000%`), $\tau_b$, $g_b$, `r_nowrite`, and aggregate write rate $\bar{\eta}_t = \frac{1}{B}\sum_b \eta_{bt}$ identical to `slwm_v3_routed`, replacing only per-bank content routing with uniform bank-averaged writes.
- **`-NoWriteOption` (`slwm_v3_no_nowrite`) & Preregistered `ForcedWrite` Control (`slwm_v3_forced_write`):** `-NoWriteOption` removes explicit `r_nowrite_b`; `slwm_v3_forced_write` additionally enforces $\eta_{bt} \ge 0.5$ on every valid tick, preventing silent write suppression via the routing sigmoid alone.
- **`-ZeroLagResidual` (`slwm_v3_no_zerolag`):** Isolates the marginal value of the gated sensory bypass projection.
- **`diag_ssm` & `explicit_bounded_kv`:** Diagonal scalar-decay SSM comparator (`dt_pos = max(dt_s, 0.0)`) and competitive differentiable multi-head bounded LRU/content-overwrite KV controller (`Precision.HIGHEST` on `fp32` slot reads/keys).

---

## 2. Single Coherent Predictive Objective & Gradient Isolation

- **Wave A Predictive Loss ([`masked_multi_horizon_predictive_loss`](../../models/v3/objectives.py#L31-L112)):** One Huber objective predicting fixed, train-normalized future packet features at horizons $\Delta \in \{1, 4, 8\}$ ticks (zero EMA moving targets in the validation ruler). Action conditioning enters strictly via `has_true_action ∈ {0, 1}` paired with a 2-way `action_avail_emb`, separating executed zero actions from missing actions in both the core and predictive head. Packed sequences compute episode IDs from `reset_mask` so supervision never crosses episode boundaries.
- **Strictly Decoupled Readouts ([`apply_decoupled_probe`](../../models/v3/objectives.py#L127-L131)):** Every downstream diagnostic head applies `jax.lax.stop_gradient(Z_t)` on an isolated parameter tree, verified non-vacuously with non-zero `probe_w2` to yield `|grad(probe_w1)| > 1e-5`, `|grad(probe_w2)| > 1e-5`, `max(|grad_core|) == 0.0`, and bit-exact core weight invariance.

---

## 3. Verified Gate 3 Invariants (`20/20` Non-Vacuous Unit Tests Passing)

All invariants below are enforced by [test_v3_gate3_causality_and_mup.py](../../tests/test_v3_gate3_causality_and_mup.py):
1. **Raw-Input Suffix Invariance, Online == Offline & Missing-Packet Invariance (ALL 11 backbones):** Prefix `max(|Z_base[:12] - Z_alt[:12]|) == 0.0`, suffix response `> 1e-4`; `step_core == scan_core` to `< 1e-3` (`bf16` ring KV) and `< 1e-5` (`fp32` outputs & final state); missing packets (`valid_mask = 0, dt_s = 0`) leave persistent state bit-exact unchanged across all 11 backbones.
2. **Chunked Rematerialization Equivalence:** Outer chunked `jax.checkpoint` (`remat_chunk_size = 8` on non-multiple $T = 25$) matches unchunked `scan_core` to `< 1e-5` forward and `< 1e-4` full trunk gradients.
3. **Zero Streaming State Growth & Non-Vacuous Boundary/Action/Probe Isolation:** Constant state bytes from $T = 1$ to $T = 256$; bit-exact reset at $t = 12$; positive no-reset control (`> 1.0` diff) vs `0.0` diff with `reset_mask`; non-zero `pred_head_w` action causality (`[:8] == 0.0`, `[8:] > 1e-4`); non-zero `probe_w2` isolation (`|grad(probe_w1)| > 1e-5`, `|grad_core| == 0.0`).
4. **Fair Ablations, Smooth Contraction, `fan_in` $\mu$P & Explicit KV Verification:** Confirmed `-LearnedRouting` matches `slwm_v3_routed` bit-exactly in parameter count, `r_nowrite` (`0.119`), and aggregate write rate `mean_b(eta)` (`max - min < 1e-6`), while `slwm_v3_routed` exhibits positive bank specialization (`> 1e-3`) and full `[0, 1]` single-bank overwrite (`> 0.99`); confirmed `slwm_v3_forced_write` enforces `eff_write_eta >= 0.5`; confirmed `slwm_v3_tied_tau` matches `slwm_v3_routed` to 3 scalar `log_tau` params; confirmed smooth `safe_l2_normalize` contraction with zero jump at `1e-4`; confirmed exact `fan_in` $\mu$P LR mult `0.5` on `mem.w_o` and `w_prior_proj` at `confirmation_135m` (`dk=dv=64`); confirmed `explicit_bounded_kv` clean eviction, smooth overwrite (`>94%`), zero-write preservation, `dt=0` LRU tie protection, and finite `T=40` gradients with `|grad(thresh)| > 1e-8`.
5. **Paired OOD Manifests, `hard_distractor` Oracle, Additive Speech RF & Launcher Guardrails:** Confirmed `100.0%` causal oracle accuracy across `clean`, `distractor`, `corrupt`, same-key/same-modality low-reliability `hard_distractor`, and unseen `ood_burst`, plus `100%` bit-exact episode & `t_corr` pairing between `clean[:n]` and `ood_2x_delay`; confirmed additive `Mimi + GQA RF` schedule guarantee (`min_delay > L*(W-1) + 1020`); confirmed strict Phase 2 verdict fixtures, `TPU v6e / v6 lite` guard, non-deadlocking `>128 KiB` file-backed watchdog, projected spend reserve, strict `verified_stop_session`, and TOCTOU-free bundle/receipt locks.

---

## 4. Exact Base-Shape $\mu$P (`width × depth × head-rank`), CompleteP & Masked WD

Under exact base-shape $\mu$P ($m_d = d / d_{\text{base}}$):
- **Initialization & LR Multipliers:** Every leaf has LR multiplier `1.0` at base geometry across all 11 backbones; width-scaling 2D hidden/readout weights receive `1 / m_d`; memory readout projections (`mem.w_o`, `mem.w_prior_proj`, `kv.w_o`) scale strictly as `mem_base_dim / fan_in` (`1.0` at `fan_in = 128`, `0.5` at `confirmation_135m` `fan_in = 256`, `0.25` at `700M–1B` `fan_in = 512`); inputs and scalars receive `1.0`.
- **Masked Width-Invariant Independent Weight Decay:** `wd = 0.01` is applied *after* per-leaf LR scaling and strictly masked via [`build_weight_decay_mask`](../../models/v3/core.py#L497-L509) so `log_tau`, `log_a`, `norm_w`, `o_norm_w`, `rel_pos_bias`, `gate_z0`, `thresh`, and all 1D biases receive `0.0` weight decay.

Production masked `optax` AdamW 5-step coordinate audits ([gate3_architecture_and_mup_receipt.json](../../reports/v3/gate3_architecture_and_mup_receipt.json)) confirm:
- **Width $\mu$P (`L = 4`, `d ∈ {128, 256, 512}`):** `init_z_rms` ratio **`1.0004`**; `step5_delta_z_rms` **$\mu$P ratio `1.0480` (loss ratio `1.0150`)** vs. uniform-LR SP `1.1046` (loss ratio `2.0103`).
- **Separate Depth CompleteP (`d = 128`, `L ∈ {4, 6, 8}`):** `step5_delta_z_rms` ratio **`1.0372`**.
- **Joint `Width × Depth × Head-Rank` (`d128 L4 dk32 → d256 L6 dk32 → d512 L8 dk64`):** `step5_delta_z_rms` `0.74234 / 0.70978 / 0.70705` (**ratio `1.0499`**; loss `0.16939 / 0.17096 / 0.17169`; **`100.0%` active gradient leaf fraction** after step-2 readout warmup, before weight decay).
- **All 11 Wave A Backbones (`T = 40`, 3 steps real AdamW):** `11/11` finite gradients and finite losses.

---

## 5. Dual-Track (`Core` + `Strict System`) Comparator Parity (`±0.62%` Params Across All 12 Arms × 3 Profiles)

Exact parameter counts (`jax.eval_shape`, verified within strict **`±1.0%`** tolerance) and runtime-verified persistent streaming state footprints (`fp32` memory + `bf16` ring KV, combined with Gate 2 frozen encoder totals `92,988,448` params / `168.444 MiB` bf16) are recorded in [gate3_architecture_and_mup_receipt.json](../../reports/v3/gate3_architecture_and_mup_receipt.json):

| Scale Profile | Anchor (`slwm_v3_routed`) Core Params | All 12 Arms Param $\Delta$ Range | Anchor Core State (`bf16` ring + `fp32` mem) | Strict System Total Enc Params | Strict System Streaming State (`bf16`) |
|---|---:|---:|---:|---:|---:|
| **`pilot_d256`** (`L=12, W=32`, RF `372` / `W_eq=64` RF `756`) | `13,914,976` | `-0.473%` to `+0.278%` | `0.3754 MiB` (`393,648 B`) | `106,903,424` | `168.819 MiB` |
| **`discovery_45m`** (`L=12, W=32`, RF `372` / `W_eq=64` RF `756`) | `42,374,240` | `-0.620%` to `+0.298%` | `0.3754 MiB` (`393,648 B`) | `135,362,688` | `168.819 MiB` |
| **`confirmation_135m`** (`L=16, W=64`, RF `1,008` / `W_eq=128` RF `2,032`) | `137,890,112` | `-0.428%` to `+0.184%` (all `137.3M–138.1M`) | `2.0010 MiB` (`2,098,240 B`) | `230,878,560` | `170.445 MiB` |

Every byte-matched comparator (`slwm_v3_no_*`, `single_bank_gated_deltanet`, `sliding_window_gqa_eqbyte`, `diag_ssm`, `explicit_bounded_kv`) matches total core streaming state bytes within **`-0.781%` to `+0.975%`** across all three scale profiles.
