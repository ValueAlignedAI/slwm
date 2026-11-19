"""SLWM-v3 Memory & Attention Primitives (Rev 2: Exact Base-Shape muP, Fair Ablations & Safe Multi-Head KV).

Implements ALL Wave A core sequence mechanisms sharing one strictly causal O(1) single-step contract:
  1. `ContentRoutedMultiTimescaleMemory`:
       - Read-before-write prior prediction error `e_t` computed strictly BEFORE updating `M_t`.
       - Learned content router over `B` banks + explicit `NO-WRITE` channel (zero task opcodes!).
       - Multi-timescale physical decay + normalized-key rank-1 delta overwrite in strict `float32`:
           M_tilde = exp(-g * dt / tau_b) * M_{t-1}
           M_t     = M_tilde + r_bt * beta_t * (v_t - M_tilde @ k_t) @ k_t^T
       - Fair `-LearnedRouting` (`use_learned_routing=False`): holds `tau_b`, `g_b`, and learned `read_w`
         strictly identical to `slwm_v3_routed`, replacing content-routed bank write allocation with
         content-independent round-robin one-hot allocation (`one_hot(step_idx % nb) * (1 - r_nowrite)`),
         preserving BOTH aggregate write budget `1 - r_nowrite` AND single-bank write ceiling `[0, 1]`!
       - Faithful `single_bank_gated_deltanet` (`use_surprise_router=False`, `n_banks=1`): core gated-delta
         single-bank matrix recurrence with NO surprise/no-write router MLP (without short conv / output gate).
  2. `DiagonalSSMMemory` (`diag_ssm` comparator):
       - Multi-channel diagonal scalar-decay SSM matched in bytes & params (`dt >= 0`, zero decay on `dt=0`).
  3. `ExplicitBoundedKVMemory` (`explicit_bounded_kv` comparator):
       - Differentiable multi-head bounded LRU/content-overwrite KV table (`[B, H_kv, K_slots, D_kv]`),
         matched 2-layer SiLU write-salience MLP, differentiable `thresh`, true LRU read-age refresh,
         clean victim replacement (no stale-victim blending), and JAX double-`where` safe unit norm
         (exact `1.0 * I` Jacobian at zero — zero gradient explosion across arbitrary horizons!).
  4. `RingBufferLocalGQA`:
       - Strictly causal O(1) sliding-window GQA (`bfloat16` ring KV state matching exact byte counter).
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from models.v3.spec import SLWMv3Spec

Array = jax.Array
PyTree = dict[str, Any]

SCALAR_LEAF_NAMES = frozenset(
    {
        "norm",
        "norm_w",
        "bias",
        "log_tau",
        "log_a",
        "rel_pos_bias",
        "gate_z0",
        "thresh",
        "write_bias",
        "g_bias",
        "beta_bias",
        "read_gate_bias",
        "router_b1",
        "router_b2",
        "dt_bias",
        "pred_head_bias",
    }
)


# ---------------------------------------------------------------------------
# Basic Primitives & Exact Base-Shape muP Initialization
# ---------------------------------------------------------------------------


def rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    """RMSNorm preserving input shape `[..., D]` in float32 accumulation."""
    xf = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv_rms).astype(x.dtype) * weight.astype(x.dtype)


def safe_l2_normalize(x: Array, eps: float = 1e-8) -> Array:
    """Strict float32 L2 unit-norm using JAX double-`where` so Jacobian at `x = 0` is `1.0 * I` (never `1000 * I`)."""
    xf = x.astype(jnp.float32)
    sq = jnp.sum(xf * xf, axis=-1, keepdims=True)
    active = sq > eps
    safe_sq = jnp.where(active, sq, 1.0)
    return jnp.where(active, xf * jax.lax.rsqrt(safe_sq), xf)


def init_mup_weight(
    key: Array,
    shape: tuple[int, ...],
    *,
    role: str,
    width_mult: float,
    base_std: float = 0.02,
    zero_init: bool = False,
) -> Array:
    """Initialize weight matrix under exact base-shape muP (`m_d = d_model / d_base`).

    At base width (`width_mult == 1.0`), init std equals `base_std` (exact SP identity).
    Roles:
      - `"input"`:        fixed-fan-in external input / embedding -> std = `base_std`; LR mult = `1.0`
      - `"hidden"`:       width-scaling hidden projection (`fan_in ~ d`) -> std = `base_std / sqrt(m_d)`; LR mult = `1 / m_d`
      - `"fixed_hidden"`: core projection whose `fan_in` does NOT scale with `d` (e.g. fixed `H_m*d_v -> d`) -> std = `base_std`; LR mult = `1.0`
      - `"readout"`:      output head (`fan_in = d`) -> `zero_init=True` (or `base_std / m_d`); LR mult = `1 / m_d`
      - `"scalar"`:       1D norm/bias/timescale parameters -> std = `base_std`; LR mult = `1.0`, weight decay = `0.0`
    """
    if zero_init:
        return jnp.zeros(shape, dtype=jnp.float32)
    m_d = max(float(width_mult), 1e-6)
    if role in ("input", "fixed_hidden", "scalar"):
        std = base_std
    elif role == "hidden":
        std = base_std / math.sqrt(m_d)
    elif role == "readout":
        std = base_std / m_d
    else:
        raise ValueError(f"Unknown muP role={role!r}")
    return jax.random.normal(key, shape, dtype=jnp.float32) * std


def classify_param_mup_role(path_parts: tuple[str, ...]) -> str:
    """Classify exact base-shape muP role from parameter tree path.

    Guarantees:
      - Every `"scalar"` / 1D bias / norm / timescale (`log_tau`, `log_a`, `router_b1`, `router_b2`, etc.)
        gets `lr_mult = 1.0` AND `weight_decay = 0.0`.
      - Every `"input"` and fixed-fan-in core projection (`mem.w_o`, `mem.w_prior_proj`, `kv.w_o`,
        `pred_head_action_w`, `pred_head_action_avail_emb`, `edge_*`) gets `lr_mult = 1.0`.
      - Every width-scaling hidden/readout 2D matrix (`fan_in` scales proportionally with `d_model`)
        gets `lr_mult = 1.0 / width_mult` (exact `1.0` at `d_model == d_base`!).
    """
    leaf = path_parts[-1] if path_parts else ""
    parent = path_parts[-2] if len(path_parts) >= 2 else ""
    if leaf in SCALAR_LEAF_NAMES or leaf.endswith("_bias") or leaf.endswith("_norm"):
        return "scalar"
    if any(
        p.startswith("edge_")
        or p in ("mod_emb", "pred_head_action_w", "pred_head_action_avail_emb")
        for p in path_parts
    ):
        return "input"
    # Fixed-fan-in memory output projections (nh*dv -> d and kv_h*kv_d -> d) do not scale fan_in with width `d`
    if parent in ("mem", "kv") and leaf in ("w_o", "w_prior_proj"):
        return "fixed_hidden"
    if leaf == "pred_head_w" or any(p.startswith("readout") or p.startswith("head_") for p in path_parts):
        return "readout"
    return "hidden"


# ---------------------------------------------------------------------------
# 1. Short-Window O(1) Causal Local GQA Ring Buffer
# ---------------------------------------------------------------------------


def init_ring_gqa_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    """Initialize parameters for one layer of O(1) sliding-window causal GQA."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d = spec.d_model
    hq = spec.n_attn_heads * spec.attn_head_dim
    hkv = spec.n_kv_heads * spec.attn_head_dim
    m_d = spec.width_mult
    return {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_q": init_mup_weight(k1, (d, hq), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_k": init_mup_weight(k2, (d, hkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_v": init_mup_weight(k3, (d, hkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_o": init_mup_weight(k4, (hq, d), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "rel_pos_bias": jnp.zeros((spec.n_attn_heads, spec.window_size), dtype=jnp.float32),
    }


def init_ring_gqa_state(spec: SLWMv3Spec, batch_size: int) -> PyTree:
    """Allocate fixed-size O(1) sliding-window GQA ring buffer matching exact `ring_kv_bytes_per_elem`."""
    kv_dtype = jnp.bfloat16 if spec.ring_kv_bytes_per_elem == 2 else jnp.float32
    return {
        "ring_k": jnp.zeros((batch_size, spec.window_size, spec.n_kv_heads, spec.attn_head_dim), dtype=kv_dtype),
        "ring_v": jnp.zeros((batch_size, spec.window_size, spec.n_kv_heads, spec.attn_head_dim), dtype=kv_dtype),
        "ring_valid": jnp.zeros((batch_size, spec.window_size), dtype=jnp.bool_),
        "step_idx": jnp.zeros((batch_size,), dtype=jnp.int32),
    }


def step_ring_gqa(
    params: PyTree,
    state: PyTree,
    x_t: Array,
    valid_mask_t: Array,
    spec: SLWMv3Spec,
) -> tuple[Array, PyTree]:
    """Single strictly-causal O(1) step of sliding-window local GQA."""
    b = x_t.shape[0]
    h_q = spec.n_attn_heads
    h_kv = spec.n_kv_heads
    d_h = spec.attn_head_dim
    groups = h_q // h_kv
    kv_dtype = state["ring_k"].dtype

    xn = rms_norm(x_t, params["norm_w"])
    q = jnp.matmul(xn, params["w_q"]).reshape(b, h_q, d_h)
    k_new = jnp.matmul(xn, params["w_k"]).reshape(b, h_kv, d_h).astype(kv_dtype)
    v_new = jnp.matmul(xn, params["w_v"]).reshape(b, h_kv, d_h).astype(kv_dtype)

    is_valid = valid_mask_t > 0.5
    cand_k = jnp.concatenate([state["ring_k"][:, 1:, :, :], k_new[:, None, :, :]], axis=1)
    cand_v = jnp.concatenate([state["ring_v"][:, 1:, :, :], v_new[:, None, :, :]], axis=1)
    cand_valid = jnp.concatenate([state["ring_valid"][:, 1:], is_valid[:, None]], axis=1)

    m4 = is_valid[:, None, None, None]
    m2 = is_valid[:, None]
    ring_k = jnp.where(m4, cand_k, state["ring_k"])
    ring_v = jnp.where(m4, cand_v, state["ring_v"])
    ring_valid = jnp.where(m2, cand_valid, state["ring_valid"])
    step_idx = state["step_idx"] + is_valid.astype(jnp.int32)

    k_exp = jnp.repeat(ring_k.astype(jnp.float32), groups, axis=2)
    v_exp = jnp.repeat(ring_v.astype(jnp.float32), groups, axis=2)

    scale = 1.0 / math.sqrt(float(d_h))
    logits = jnp.einsum("bhd,bwhd->bhw", q, k_exp) * scale + params["rel_pos_bias"][None, :, :]
    has_any = jnp.any(ring_valid, axis=-1, keepdims=True)
    safe_valid = jnp.where(has_any, ring_valid, jnp.ones_like(ring_valid))
    logits = jnp.where(safe_valid[:, None, :], logits, -1e9)
    weights = jnp.where(has_any[:, :, None], jax.nn.softmax(logits, axis=-1), 0.0)

    attn_out = jnp.einsum("bhw,bwhd->bhd", weights, v_exp).reshape(b, h_q * d_h)
    out = jnp.matmul(attn_out, params["w_o"]) * valid_mask_t[:, None]

    new_state = {
        "ring_k": ring_k,
        "ring_v": ring_v,
        "ring_valid": ring_valid,
        "step_idx": step_idx,
    }
    return out, new_state


# ---------------------------------------------------------------------------
# 2. Content-Routed Multi-Timescale Matrix Memory & Faithful Gated DeltaNet
# ---------------------------------------------------------------------------


def init_matrix_memory_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    """Initialize parameters for Content-Routed Multi-Timescale Matrix Memory or faithful Gated DeltaNet.

    Omits dead parameters cleanly:
      - `-LearnedRouting`: holds `tau_b`, `g_b`, `w_read_gate`, AND the `(nb + 1)`-logit no-write head identical
        to `slwm_v3_routed`, changing ONLY bank selection to content-independent round-robin!
      - `single_bank_gated_deltanet` (`use_surprise_router=False`): omits surprise router MLP & `w_prior_proj`.
    """
    keys = jax.random.split(key, 12)
    d = spec.d_model
    m_d = spec.width_mult
    nb = spec.n_banks
    nh = spec.n_mem_heads
    dk = spec.mem_key_dim
    dv = spec.mem_val_dim
    mod_d = 16
    # Scale fixed-fan-in projections `w_o` and `w_prior_proj` by 1 / sqrt(nh * dv / 64) for stable RMS across d_v
    out_std = spec.init_std / math.sqrt(max(float(nh * dv) / 64.0, 1.0))

    if spec.tie_bank_timescales:
        mid_tau = float(math.exp(sum(math.log(t) for t in spec.init_timescales_s) / nb))
        log_tau_init = jnp.asarray([math.log(mid_tau)], dtype=jnp.float32)  # Single shared timescale!
        n_g_banks = 1  # Shared decay gate across banks!
    else:
        log_tau_init = jnp.asarray([math.log(t) for t in spec.init_timescales_s], dtype=jnp.float32)
        n_g_banks = nb

    params: PyTree = {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_q": init_mup_weight(keys[1], (d, nh * dk), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_k": init_mup_weight(keys[2], (d, nh * dk), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_v": init_mup_weight(keys[3], (d, nh * dv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_o": init_mup_weight(keys[4], (nh * dv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std),
        "w_g": init_mup_weight(keys[5], (d, n_g_banks * nh), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "g_bias": jnp.full((n_g_banks, nh), -0.5, dtype=jnp.float32),
        "w_beta": init_mup_weight(keys[6], (d, nh), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "beta_bias": jnp.full((nh,), 0.5, dtype=jnp.float32),
        "log_tau": log_tau_init,
    }

    if spec.use_surprise_router:
        router_in = d + mod_d + 3  # [RMSNorm(x_t), mod_emb, log1p(dt_s), reliability, log1p(prior_err)]
        router_hid = max(32, d // 4)  # Scales strictly proportionally with d across d in {128, 256, 512, 768}!
        # Use IDENTICAL (nb + 1)-logit softmax parameterization in BOTH `slwm_v3_routed` and `slwm_v3_no_routing`
        # so the no-write probability `r_nowrite` (0.20 at init) and total write budget `1 - r_nowrite` (0.80 at init)
        # are BIT-EXACT identical across both arms, isolating ONLY bank selection!
        n_router_out = nb + 1 if spec.use_nowrite_option else nb
        params["mod_emb"] = init_mup_weight(keys[0], (spec.n_modalities, mod_d), role="input", width_mult=m_d, base_std=0.25)
        params["w_prior_proj"] = init_mup_weight(keys[10], (nh * dv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std)
        params["router_w1"] = init_mup_weight(keys[8], (router_in, router_hid), role="hidden", width_mult=m_d, base_std=spec.init_std)
        params["router_b1"] = jnp.zeros((router_hid,), dtype=jnp.float32)
        params["router_w2"] = init_mup_weight(keys[9], (router_hid, n_router_out), role="hidden", width_mult=m_d, base_std=spec.init_std)
        params["router_b2"] = jnp.zeros((n_router_out,), dtype=jnp.float32)
        if nb > 1:
            # Hold read routing fixed across `slwm_v3_routed` vs `slwm_v3_no_routing` for strict
            # single-factor isolation of content-routed bank write allocation!
            params["w_read_gate"] = init_mup_weight(keys[7], (d + mod_d + 1, nb), role="hidden", width_mult=m_d, base_std=spec.init_std)
            params["read_gate_bias"] = jnp.zeros((nb,), dtype=jnp.float32)
    return params


def init_matrix_memory_state(spec: SLWMv3Spec, batch_size: int) -> PyTree:
    """Allocate strict float32 O(1) matrix memory bank state `M: float32[B, n_banks, H_mem, d_v, d_k]`."""
    return {
        "M": jnp.zeros(
            (batch_size, spec.n_banks, spec.n_mem_heads, spec.mem_val_dim, spec.mem_key_dim),
            dtype=jnp.float32,
        )
    }


def step_matrix_memory(
    params: PyTree,
    state: PyTree,
    x_t: Array,
    modality_id_t: Array,
    dt_s_t: Array,
    valid_mask_t: Array,
    reliability_t: Array,
    spec: SLWMv3Spec,
    step_idx_t: Array | None = None,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Single strictly-causal Read-Before-Write step of Multi-Timescale Matrix Memory or Gated DeltaNet."""
    b = x_t.shape[0]
    nb = spec.n_banks
    nh = spec.n_mem_heads
    dk = spec.mem_key_dim
    dv = spec.mem_val_dim

    xn = rms_norm(x_t, params["norm_w"])
    dt_pos = jnp.maximum(dt_s_t.astype(jnp.float32), 0.0)
    log_dt = jnp.log1p(dt_pos)[:, None]

    # 1. Physical decay (strictly tied across banks when `tie_bank_timescales=True`!)
    M_prev = state["M"].astype(jnp.float32)
    n_g_banks = 1 if spec.tie_bank_timescales else nb
    g_raw = jnp.matmul(xn, params["w_g"]).reshape(b, n_g_banks, nh) + params["g_bias"][None, :, :]
    g = jax.nn.softplus(g_raw)
    if spec.tie_bank_timescales:
        g = jnp.broadcast_to(g, (b, nb, nh))
        tau = jnp.broadcast_to(jnp.exp(jnp.clip(params["log_tau"], -4.0, 8.0)), (nb,))
    else:
        tau = jnp.exp(jnp.clip(params["log_tau"], -4.0, 8.0))
    decay = jnp.exp(-g * (dt_pos[:, None, None] / (tau[None, :, None] + 1e-6)))
    M_tilde = M_prev * decay[:, :, :, None, None]

    q = safe_l2_normalize(jnp.matmul(xn, params["w_q"]).reshape(b, nh, dk))
    k = safe_l2_normalize(jnp.matmul(xn, params["w_k"]).reshape(b, nh, dk))
    v = jnp.matmul(xn, params["w_v"]).reshape(b, nh, dv).astype(jnp.float32)
    beta = jax.nn.sigmoid(jnp.matmul(xn, params["w_beta"]) + params["beta_bias"][None, :])  # [B, nh]

    if not spec.use_surprise_router:
        # Faithful Single-Bank Gated DeltaNet (NO surprise/no-write router MLP!)
        read_w = jnp.full((b, nb), 1.0 / float(nb), dtype=jnp.float32)
        r_banks = jnp.ones((b, nb), dtype=jnp.float32)
        r_nowrite = jnp.zeros((b,), dtype=jnp.float32)
        prior_err = jnp.zeros((b,), dtype=jnp.float32)
    else:
        mod_idx = jnp.clip(modality_id_t, 0, spec.n_modalities - 1)
        mod_e = params["mod_emb"][mod_idx]

        # 2. READ-BEFORE-WRITE: compute causal prior readout & retrieval residual BEFORE writing!
        prior_bank_y = jnp.einsum("bnhvk,bhk->bnhv", M_tilde, q)
        if nb > 1:
            read_in = jnp.concatenate([xn, mod_e, log_dt], axis=-1)
            read_w = jax.nn.softmax(jnp.matmul(read_in, params["w_read_gate"]) + params["read_gate_bias"], axis=-1)
        else:
            read_w = jnp.full((b, nb), 1.0 / float(nb), dtype=jnp.float32)
        prior_y = jnp.einsum("bn,bnhv->bhv", read_w, prior_bank_y).reshape(b, nh * dv)
        x_prior_hat = jnp.matmul(prior_y, params["w_prior_proj"])
        prior_err = jnp.sqrt(jnp.mean((xn - x_prior_hat) ** 2, axis=-1) + 1e-6)

        # 3. Learned content router + optional explicit NO-WRITE channel
        router_feat = jnp.concatenate(
            [xn, mod_e, log_dt, reliability_t[:, None].astype(jnp.float32), jnp.log1p(prior_err)[:, None]],
            axis=-1,
        )
        rh = jax.nn.silu(jnp.matmul(router_feat, params["router_w1"]) + params["router_b1"])
        r_logits = jnp.matmul(rh, params["router_w2"]) + params["router_b2"]

        r_probs = jax.nn.softmax(r_logits, axis=-1)
        if spec.use_nowrite_option:
            r_nowrite = r_probs[:, nb]
            write_mass = 1.0 - r_nowrite  # Exact `sum(r_probs[:, :nb])` in BOTH routed and ablated arms!
        else:
            r_nowrite = jnp.zeros((b,), dtype=jnp.float32)
            write_mass = jnp.ones((b,), dtype=jnp.float32)

        if spec.use_learned_routing:
            r_banks = r_probs[:, :nb]
        else:
            # Strict single-factor `-LearnedRouting` ablation:
            # Uses the EXACT SAME `(nb + 1)`-logit no-write calculation (`r_nowrite`, `write_mass = 1 - r_nowrite`)
            # as `slwm_v3_routed`, replacing ONLY content-routed bank selection with content-independent round-robin!
            rr_idx = (
                jnp.mod(step_idx_t.astype(jnp.int32), nb)
                if step_idx_t is not None
                else jnp.zeros((b,), dtype=jnp.int32)
            )
            r_banks = jax.nn.one_hot(rr_idx, nb, dtype=jnp.float32) * write_mass[:, None]

    # Missing packets perform BIT-EXACT ZERO CONTENT WRITE!
    r_banks_eff = r_banks * valid_mask_t[:, None].astype(jnp.float32)

    # 4. Normalized-key rank-1 delta overwrite in strict float32
    retrieved_k = jnp.einsum("bnhvk,bhk->bnhv", M_tilde, k)
    delta_v = v[:, None, :, :] - retrieved_k
    eta = r_banks_eff[:, :, None] * beta[:, None, :]  # [B, nb, nh] in [0, 1]
    dM = jnp.einsum("bnh,bnhv,bhk->bnhvk", eta, delta_v, k)
    M_next = M_tilde + dM

    # 5. Post-write readout using causal query q_t
    post_bank_y = jnp.einsum("bnhvk,bhk->bnhv", M_next, q)
    post_y = jnp.einsum("bn,bnhv->bhv", read_w, post_bank_y).reshape(b, nh * dv)
    out = jnp.matmul(post_y, params["w_o"]) * valid_mask_t[:, None]

    diags = {
        "r_banks": r_banks_eff,
        "r_nowrite": r_nowrite,
        "eff_write_eta": jnp.mean(eta, axis=(1, 2)),
        "eff_decay": jnp.mean(decay, axis=(1, 2)),
        "state_delta_norm": jnp.sqrt(jnp.mean(dM * dM, axis=(1, 2, 3, 4)) + 1e-12),
        "mem_out_rms": jnp.sqrt(jnp.mean(out * out, axis=-1) + 1e-12),
        "prior_error": prior_err,
        "key_norm": jnp.sqrt(jnp.sum(k * k, axis=-1)),
    }
    return out, {"M": M_next}, diags


# ---------------------------------------------------------------------------
# 3. Diagonal SSM Comparator (`diag_ssm`)
# ---------------------------------------------------------------------------


def init_diag_ssm_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    keys = jax.random.split(key, 6)
    d = spec.d_model
    n = spec.ssm_state_dim
    m_d = spec.width_mult
    return {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_b": init_mup_weight(keys[0], (d, n), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_c": init_mup_weight(keys[1], (d, n), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_dt": init_mup_weight(keys[2], (d, d), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "dt_bias": jnp.full((d,), -1.5, dtype=jnp.float32),
        "log_a": jnp.linspace(-2.0, 0.5, n, dtype=jnp.float32),
        "w_o": init_mup_weight(keys[3], (d, d), role="hidden", width_mult=m_d, base_std=spec.init_std),
    }


def init_diag_ssm_state(spec: SLWMv3Spec, batch_size: int) -> PyTree:
    return {"S": jnp.zeros((batch_size, spec.d_model, spec.ssm_state_dim), dtype=jnp.float32)}


def step_diag_ssm(
    params: PyTree,
    state: PyTree,
    x_t: Array,
    dt_s_t: Array,
    valid_mask_t: Array,
    spec: SLWMv3Spec,
) -> tuple[Array, PyTree, dict[str, Array]]:
    del spec
    xn = rms_norm(x_t, params["norm_w"])
    # Strict dt >= 0.0 so dt_s = 0 performs ZERO spurious decay!
    dt_pos = jnp.maximum(dt_s_t.astype(jnp.float32), 0.0)[:, None]
    step_g = jax.nn.softplus(jnp.matmul(xn, params["w_dt"]) + params["dt_bias"]) * dt_pos
    a_rate = jnp.exp(jnp.clip(params["log_a"], -4.0, 4.0))
    decay = jnp.exp(-step_g[:, :, None] * a_rate[None, None, :])

    b_vec = safe_l2_normalize(jnp.matmul(xn, params["w_b"]))
    c_vec = safe_l2_normalize(jnp.matmul(xn, params["w_c"]))

    S_prev = state["S"].astype(jnp.float32)
    S_decayed = S_prev * decay
    write_scale = jnp.where(dt_pos[:, :, None] > 0.0, 1.0 - decay, 0.0)
    S_next = S_decayed + (valid_mask_t[:, None, None] * write_scale * (xn[:, :, None] * b_vec[:, None, :]))
    y = jnp.einsum("bdn,bn->bd", S_next, c_vec)
    out = jnp.matmul(y, params["w_o"]) * valid_mask_t[:, None]
    return out, {"S": S_next}, {"prior_error": jnp.zeros((x_t.shape[0],), dtype=jnp.float32)}


# ---------------------------------------------------------------------------
# 4. Competitive Differentiable Multi-Head Bounded LRU/Overwrite KV Controller
# ---------------------------------------------------------------------------


def init_explicit_kv_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    """Initialize byte- & param-matched differentiable multi-head bounded LRU/overwrite KV controller."""
    keys = jax.random.split(key, 10)
    d = spec.d_model
    h_e = spec.explicit_kv_heads
    dkv = spec.explicit_kv_dim
    m_d = spec.width_mult
    mod_d = 16
    router_in = d + mod_d + 3
    router_hid = max(32, d // 4)
    out_std = spec.init_std / math.sqrt(max(float(h_e * dkv) / 64.0, 1.0))

    return {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "mod_emb": init_mup_weight(keys[0], (spec.n_modalities, mod_d), role="input", width_mult=m_d, base_std=0.25),
        "w_q": init_mup_weight(keys[1], (d, h_e * dkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_k": init_mup_weight(keys[2], (d, h_e * dkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_v": init_mup_weight(keys[3], (d, h_e * dkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_o": init_mup_weight(keys[4], (h_e * dkv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std),
        # Matched 2-layer SiLU write-salience MLP (exact parity with SLWM-v3's salience MLP!)
        "router_w1": init_mup_weight(keys[5], (router_in, router_hid), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "router_b1": jnp.zeros((router_hid,), dtype=jnp.float32),
        "router_w2": init_mup_weight(keys[6], (router_hid, h_e), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "router_b2": jnp.full((h_e,), -0.5, dtype=jnp.float32),
        # Learnable differentiable content-match cosine threshold (per head)
        "thresh": jnp.full((h_e,), 0.70, dtype=jnp.float32),
    }


def init_explicit_kv_state(spec: SLWMv3Spec, batch_size: int) -> PyTree:
    """Allocate fixed-size multi-head bounded KV table `[B, H_kv, K_slots, D_kv]` matched in bytes."""
    h_e = spec.explicit_kv_heads
    k_s = spec.explicit_kv_slots
    dkv = spec.explicit_kv_dim
    return {
        "kv_keys": jnp.zeros((batch_size, h_e, k_s, dkv), dtype=jnp.float32),
        "kv_vals": jnp.zeros((batch_size, h_e, k_s, dkv), dtype=jnp.float32),
        "kv_age": jnp.full((batch_size, h_e, k_s), 1e6, dtype=jnp.float32),
        "kv_valid": jnp.zeros((batch_size, h_e, k_s), dtype=jnp.bool_),
    }


def step_explicit_kv(
    params: PyTree,
    state: PyTree,
    x_t: Array,
    modality_id_t: Array,
    dt_s_t: Array,
    valid_mask_t: Array,
    reliability_t: Array,
    spec: SLWMv3Spec,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Differentiable multi-head bounded LRU/content-overwrite KV controller (zero gradient blowup!).

    Guarantees:
      - Unselected slots are preserved bit-exact in BOTH forward and backward passes (no `l2_normalize(0)`!).
      - Differentiable match-vs-LRU gate (`p_match = sigmoid(10*(best_sim - thresh))`) trains `thresh` cleanly.
      - True LRU bookkeeping: `kv_age` refreshes on BOTH writes AND attended reads (`q_w > 0.25`).
      - Clean empty/eviction replacement (never blends victim's stale value into new entry!).
    """
    b = x_t.shape[0]
    h_e = spec.explicit_kv_heads
    k_s = spec.explicit_kv_slots
    dkv = spec.explicit_kv_dim

    xn = rms_norm(x_t, params["norm_w"])
    mod_e = params["mod_emb"][jnp.clip(modality_id_t, 0, spec.n_modalities - 1)]
    dt_pos = jnp.maximum(dt_s_t.astype(jnp.float32), 0.0)

    q = safe_l2_normalize(jnp.matmul(xn, params["w_q"]).reshape(b, h_e, dkv))  # [B, H, D]
    k = safe_l2_normalize(jnp.matmul(xn, params["w_k"]).reshape(b, h_e, dkv))  # [B, H, D]
    v = jnp.matmul(xn, params["w_v"]).reshape(b, h_e, dkv).astype(jnp.float32)  # [B, H, D]

    kv_keys = state["kv_keys"].astype(jnp.float32)  # [B, H, S, D]
    kv_vals = state["kv_vals"].astype(jnp.float32)
    # Monotonic age increment: elapsed physical time PLUS small logical tick (1e-3) on valid packets so
    # even at dt_s=0 a read-refreshed slot (age=0.0) stays strictly younger than every non-refreshed slot!
    # When valid_mask_t=0 and dt_s_t=0, age increment is exact 0.0 (bit-exact missing-packet invariance).
    age_inc = dt_pos + 1e-3 * valid_mask_t.astype(jnp.float32)
    kv_age = state["kv_age"].astype(jnp.float32) + age_inc[:, None, None]
    kv_valid = state["kv_valid"]  # bool[B, H, S]

    # 1. Read-before-write cosine attention over valid entries
    q_sim = jnp.einsum("bhd,bhsd->bhs", q, kv_keys) * 8.0
    has_any = jnp.any(kv_valid, axis=-1, keepdims=True)  # [B, H, 1]
    safe_valid = jnp.where(has_any, kv_valid, jnp.ones_like(kv_valid))
    q_w = jnp.where(has_any, jax.nn.softmax(jnp.where(safe_valid, q_sim, -1e9), axis=-1), 0.0)
    prior_v = jnp.einsum("bhs,bhsd->bhd", q_w, kv_vals)
    prior_err = jnp.sqrt(jnp.mean((v - prior_v) ** 2, axis=(1, 2)) + 1e-6)  # [B]

    # True LRU age refresh on attended read hits (strictly when packet is valid!)
    read_hit = kv_valid & (q_w > 0.25) & (valid_mask_t[:, None, None] > 0.5)
    kv_age = jnp.where(read_hit, 0.0, kv_age)

    # 2. Matched 2-layer SiLU write-salience MLP (per-head write probability in [0, 1])
    w_feat = jnp.concatenate(
        [xn, mod_e, jnp.log1p(dt_pos)[:, None], reliability_t[:, None], jnp.log1p(prior_err)[:, None]],
        axis=-1,
    )
    wh = jax.nn.silu(jnp.matmul(w_feat, params["router_w1"]) + params["router_b1"])
    write_prob = jax.nn.sigmoid(jnp.matmul(wh, params["router_w2"]) + params["router_b2"]) * valid_mask_t[:, None]  # [B, H]

    # 3. Differentiable content-match overwrite vs. clean LRU/empty-slot insertion!
    k_sim = jnp.einsum("bhd,bhsd->bhs", k, kv_keys)  # [B, H, S]
    k_sim_valid = jnp.where(kv_valid, k_sim, -2.0)
    best_match_sim = jnp.max(k_sim_valid, axis=-1)  # [B, H]
    # Differentiable gate `p_match` cleanly passes gradients to BOTH `k_sim` AND `params["thresh"]`!
    p_match = jnp.where(
        has_any[..., 0],
        jax.nn.sigmoid(10.0 * (best_match_sim - params["thresh"][None, :])),
        0.0,
    )  # [B, H]

    # Soft content-match overwrite weights strictly over existing VALID slots (0 on all empty slots!)
    match_soft = jnp.where(
        has_any,
        jax.nn.softmax(jnp.where(safe_valid, k_sim * 10.0, -1e9), axis=-1),
        0.0,
    )  # [B, H, S]
    match_w = (write_prob * p_match)[:, :, None] * match_soft  # [B, H, S]

    # Clean LRU/empty slot target (unused slots have age=1e6 so empty slots always fill before evicting!)
    lru_idx = jnp.argmax(kv_age, axis=-1)  # [B, H]
    lru_onehot = jax.nn.one_hot(lru_idx, k_s, dtype=jnp.float32)  # [B, H, S]
    replace_w = (write_prob * (1.0 - p_match))[:, :, None] * lru_onehot  # [B, H, S]

    # Clean update preserving unselected slots EXACTLY (zero gradient or norm distortion on empty slots!):
    # - On matched valid slots: smooth content-overwrite update with safe unit-norm (never evaluates norm(0)!)
    # - On LRU/empty replacement slot: clean insertion weighted by `replace_w` without victim value bleed when empty/evicted!
    is_lru_slot = lru_onehot > 0.5
    do_replace = is_lru_slot & ((write_prob * (1.0 - p_match))[:, :, None] > 0.15) & (valid_mask_t[:, None, None] > 0.5)

    # Smooth overwrite candidate ONLY for occupied slots with strictly positive match weight (> 1e-7),
    # preserving untouched valid keys bit-exactly in BOTH forward and backward passes!
    active_ow = kv_valid & (match_w > 1e-7)
    ow_k_raw = (1.0 - match_w[..., None]) * kv_keys + match_w[..., None] * k[:, :, None, :]
    ow_k = jnp.where(active_ow[..., None], safe_l2_normalize(ow_k_raw), kv_keys)
    ow_v = jnp.where(active_ow[..., None], (1.0 - match_w[..., None]) * kv_vals + match_w[..., None] * v[:, :, None, :], kv_vals)

    # Clean replacement candidate for empty/evicted slot (passes gradients cleanly through `replace_w` and `k, v`!)
    rep_scale = replace_w[..., None] / jnp.maximum(jax.lax.stop_gradient(replace_w[..., None]), 1e-4)
    new_keys = jnp.where(do_replace[..., None], safe_l2_normalize(k[:, :, None, :] * rep_scale), ow_k)
    new_vals = jnp.where(do_replace[..., None], v[:, :, None, :] * rep_scale, ow_v)

    updated_any = do_replace | (kv_valid & (match_w > 0.10))
    new_age = jnp.where(updated_any, 0.0, kv_age)
    new_valid = kv_valid | do_replace

    # Missing packets (`valid_mask_t == 0`) preserve keys/vals/valid BIT-EXACT (zero normalization rounding!)
    is_valid_pkt = (valid_mask_t > 0.5)[:, None, None]  # [B, 1, 1]
    new_keys = jnp.where(is_valid_pkt[..., None], new_keys, state["kv_keys"].astype(jnp.float32))
    new_vals = jnp.where(is_valid_pkt[..., None], new_vals, state["kv_vals"].astype(jnp.float32))
    new_age = jnp.where(is_valid_pkt, new_age, kv_age)
    new_valid = jnp.where(is_valid_pkt, new_valid, kv_valid)

    # 4. Post-write readout
    post_sim = jnp.einsum("bhd,bhsd->bhs", q, new_keys) * 8.0
    post_has_any = jnp.any(new_valid, axis=-1, keepdims=True)
    post_safe = jnp.where(post_has_any, new_valid, jnp.ones_like(new_valid))
    post_w = jnp.where(post_has_any, jax.nn.softmax(jnp.where(post_safe, post_sim, -1e9), axis=-1), 0.0)
    post_v = jnp.einsum("bhs,bhsd->bhd", post_w, new_vals).reshape(b, h_e * dkv)
    out = jnp.matmul(post_v, params["w_o"]) * valid_mask_t[:, None]

    new_state = {
        "kv_keys": new_keys,
        "kv_vals": new_vals,
        "kv_age": new_age,
        "kv_valid": new_valid,
    }
    return out, new_state, {"r_nowrite": 1.0 - jnp.mean(write_prob, axis=-1), "prior_error": prior_err}
