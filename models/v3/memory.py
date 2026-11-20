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
        "o_norm_w",
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
        "router_meta_b1",
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


def safe_l2_normalize(x: Array, eps: float = 1e-6) -> Array:
    """Smooth C^infty float32 L2 contraction (`||u||_2 < 1` everywhere, exact `0` at `x = 0`, zero step jump)."""
    xf = x.astype(jnp.float32)
    sq = jnp.sum(xf * xf, axis=-1, keepdims=True)
    return xf * jax.lax.rsqrt(sq + eps)


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
        or p.startswith("router_meta_")
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
    """Initialize Head-Routed Multi-Timescale Selective Matrix Memory or faithful Single-Bank Gated DeltaNet.

    Each timescale bank `b in 0..n_banks-1` owns `n_mem_heads` dedicated full-rank `(dk, dv)` heads
    (`h_tot = n_banks * n_mem_heads`) with independent `(q, k, v)` projections, bounded per-head output
    RMSNorm (`o_norm_w`, zero weight decay), and concatenated multi-timescale readout.
    Omits dead parameters strictly:
      - `use_surprise_router=True` (`slwm_v3_*`): allocates surprise router (`router_skip` + 2-layer SiLU MLP,
        `w_prior_proj`, `mod_emb`), omitting unused `w_beta`/`beta_bias`.
      - `use_surprise_router=False` (`single_bank_gated_deltanet`): allocates linear `w_beta`/`beta_bias`
        and identical `(h_tot, dk, dv)` + `o_norm_w`, omitting surprise router & `w_prior_proj`.
    """
    keys = jax.random.split(key, 16)
    d = spec.d_model
    m_d = spec.width_mult
    nb = spec.n_banks
    nh = spec.n_mem_heads
    h_tot = nb * nh
    dk = spec.mem_key_dim
    dv = spec.mem_val_dim
    mod_d = 16
    out_std = 0.25 / math.sqrt(float(h_tot * dv))

    if spec.tie_bank_timescales or nb == 1:
        mid_tau = float(math.exp(sum(math.log(t) for t in spec.init_timescales_s) / len(spec.init_timescales_s)))
        log_tau_init = jnp.asarray([math.log(mid_tau)], dtype=jnp.float32)
    else:
        log_tau_init = jnp.asarray([math.log(t) for t in spec.init_timescales_s], dtype=jnp.float32)

    params: PyTree = {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_q": init_mup_weight(keys[1], (d, h_tot * dk), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_k": init_mup_weight(keys[2], (d, h_tot * dk), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_v": init_mup_weight(keys[3], (d, h_tot * dv), role="hidden", width_mult=m_d, base_std=1.0 / math.sqrt(float(spec.d_base))),
        "w_o": init_mup_weight(keys[4], (h_tot * dv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std),
        "o_norm_w": jnp.ones((h_tot, dv), dtype=jnp.float32),
        # Always allocate 1 decay gate per total head `(nb, nh)` so 2x2 factorial cells match exact gate expressivity & param count!
        "w_g": init_mup_weight(keys[5], (d, h_tot), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "g_bias": jnp.full((nb, nh), -1.5, dtype=jnp.float32),
        "log_tau": log_tau_init,
    }

    if not spec.use_surprise_router:
        params["w_beta"] = init_mup_weight(keys[6], (d, h_tot), role="hidden", width_mult=m_d, base_std=spec.init_std)
        params["beta_bias"] = jnp.full((h_tot,), 1.0, dtype=jnp.float32)
    else:
        # Exact muP two-branch router:
        #   1. Content branch on width-scaling `xn: [B, d]` (`role="hidden"`, `lr_mult = 1 / m_d`, 100% hoistable!)
        #   2. Width-invariant epistemic/meta branch on fixed-dim `[mod_emb(16), log1p(dt_s), reliability, log1p(prior_err)]`
        #      (`meta_in = 19`, `meta_hid = 32`, `role="input"`, `lr_mult = 1.0` at ALL widths `128 -> 1920` so
        #      epistemic signals are NEVER diluted O(1/d) under `||xn|| = sqrt(d)`!)
        meta_in = mod_d + 3
        meta_hid = 32
        router_hid = max(32, d // 4)
        router_hid_base = max(32, spec.d_base // 4)
        # Per-head timescale routing logits `h_tot` + per-bank explicit no-write logits `nb` (when enabled)
        n_router_out = h_tot + (nb if spec.use_nowrite_option else 0)
        params["mod_emb"] = init_mup_weight(keys[0], (spec.n_modalities, mod_d), role="input", width_mult=m_d, base_std=0.25)
        params["w_prior_proj"] = init_mup_weight(keys[10], (h_tot * dv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std)
        params["router_skip"] = init_mup_weight(
            keys[11], (d, n_router_out), role="hidden", width_mult=m_d, base_std=0.25 / math.sqrt(float(spec.d_base))
        )
        params["router_w1"] = init_mup_weight(
            keys[8], (d, router_hid), role="hidden", width_mult=m_d, base_std=1.0 / math.sqrt(float(spec.d_base))
        )
        params["router_b1"] = jnp.zeros((router_hid,), dtype=jnp.float32)
        params["router_w2"] = init_mup_weight(
            keys[9], (router_hid, n_router_out), role="hidden", width_mult=m_d, base_std=0.25 / math.sqrt(float(router_hid_base))
        )
        params["router_meta_skip"] = init_mup_weight(
            keys[12], (meta_in, n_router_out), role="input", width_mult=m_d, base_std=0.25 / math.sqrt(float(meta_in))
        )
        params["router_meta_w1"] = init_mup_weight(
            keys[13], (meta_in, meta_hid), role="input", width_mult=m_d, base_std=1.0 / math.sqrt(float(meta_in))
        )
        params["router_meta_b1"] = jnp.zeros((meta_hid,), dtype=jnp.float32)
        params["router_meta_w2"] = init_mup_weight(
            keys[14], (meta_hid, n_router_out), role="input", width_mult=m_d, base_std=0.25 / math.sqrt(float(meta_hid))
        )
        b2_head = jnp.full((h_tot,), 0.8, dtype=jnp.float32)
        if spec.use_nowrite_option:
            b2_nw = jnp.full((nb,), -2.0, dtype=jnp.float32)  # Initial r_nowrite = sigmoid(-2.0) = 0.119
            params["router_b2"] = jnp.concatenate([b2_head, b2_nw], axis=0)
        else:
            params["router_b2"] = b2_head
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
    """Single strictly-causal Read-Before-Write step of Head-Routed Multi-Timescale Matrix Memory or Gated DeltaNet."""
    del step_idx_t
    b = x_t.shape[0]
    nb = spec.n_banks
    nh = spec.n_mem_heads
    h_tot = nb * nh
    dk = spec.mem_key_dim
    dv = spec.mem_val_dim

    xn = rms_norm(x_t, params["norm_w"])
    dt_pos = jnp.maximum(dt_s_t.astype(jnp.float32), 0.0)
    log_dt = jnp.log1p(dt_pos)[:, None]

    # 1. Multi-timescale physical decay (or single shared timescale when `tie_bank_timescales` or `nb == 1`)
    M_prev = state["M"].astype(jnp.float32)
    g_raw = jnp.matmul(xn, params["w_g"]).reshape(b, nb, nh) + params["g_bias"][None, :, :]
    g = jax.nn.softplus(g_raw)
    if spec.tie_bank_timescales or nb == 1:
        tau = jnp.broadcast_to(jnp.exp(jnp.clip(params["log_tau"], -4.0, 8.0)), (nb,))
    else:
        tau = jnp.exp(jnp.clip(params["log_tau"], -4.0, 8.0))
    decay = jnp.exp(-g * (dt_pos[:, None, None] / (tau[None, :, None] + 1e-6)))
    M_tilde = M_prev * decay[:, :, :, None, None]

    # Independent full-rank (q, k, v) per (bank, head)
    q = safe_l2_normalize(jnp.matmul(xn, params["w_q"]).reshape(b, nb, nh, dk))
    k = safe_l2_normalize(jnp.matmul(xn, params["w_k"]).reshape(b, nb, nh, dk))
    v = jnp.matmul(xn, params["w_v"]).reshape(b, nb, nh, dv).astype(jnp.float32)

    def _soft_head_norm(y_bnhv: Array) -> Array:
        # Bounded per-head RMSNorm (eps=1e-2): exact 0 at M=0, smooth Jacobian, O(1) readout once populated
        y_flat = y_bnhv.reshape(b, h_tot, dv)
        inv_r = jax.lax.rsqrt(jnp.mean(y_flat * y_flat, axis=-1, keepdims=True) + 1e-2)
        return ((y_flat * inv_r) * params["o_norm_w"][None, :, :]).reshape(b, h_tot * dv)

    valid_f = valid_mask_t.astype(jnp.float32)
    fp32_prec = jax.lax.Precision.HIGHEST

    if not spec.use_surprise_router:
        # Faithful Gated DeltaNet (identical h_tot, dk, dv, o_norm; NO surprise/no-write router!)
        beta = jax.nn.sigmoid(jnp.matmul(xn, params["w_beta"]) + params["beta_bias"][None, :]).reshape(b, nb, nh)
        eta = beta * valid_f[:, None, None]
        r_banks_eff = jnp.mean(eta, axis=-1)
        r_nowrite = jnp.zeros((b,), dtype=jnp.float32)
        prior_err = jnp.zeros((b,), dtype=jnp.float32)
    else:
        mod_idx = jnp.clip(modality_id_t, 0, spec.n_modalities - 1)
        mod_e = params["mod_emb"][mod_idx]

        # 2. READ-BEFORE-WRITE: strict fp32 causal prior readout & retrieval residual strictly BEFORE writing!
        prior_y_bnhv = jnp.einsum("bnhvk,bnhk->bnhv", M_tilde, q, precision=fp32_prec)
        prior_y = _soft_head_norm(prior_y_bnhv)
        x_prior_hat = jnp.matmul(prior_y, params["w_prior_proj"])
        prior_err = jnp.sqrt(jnp.mean((xn - x_prior_hat) ** 2, axis=-1) + 1e-6)

        # 3. Exact muP two-branch router (content branch on `xn` + width-invariant epistemic/meta branch)
        c_h = jax.nn.silu(jnp.matmul(xn, params["router_w1"]) + params["router_b1"])
        content_logits = (
            jnp.matmul(xn, params["router_skip"]) + jnp.matmul(c_h, params["router_w2"]) + params["router_b2"]
        )
        meta_feat = jnp.concatenate(
            [mod_e, log_dt, reliability_t[:, None].astype(jnp.float32), jnp.log1p(prior_err)[:, None]],
            axis=-1,
        )
        m_h = jax.nn.silu(jnp.matmul(meta_feat, params["router_meta_w1"]) + params["router_meta_b1"])
        meta_logits = jnp.matmul(meta_feat, params["router_meta_skip"]) + jnp.matmul(m_h, params["router_meta_w2"])
        r_logits = content_logits + meta_logits

        route_raw = jax.nn.sigmoid(r_logits[:, :h_tot].reshape(b, nb, nh))
        if spec.use_nowrite_option:
            r_nowrite_b = jax.nn.sigmoid(r_logits[:, h_tot:])  # [B, nb] per-bank explicit no-write!
            keep_write_b = 1.0 - r_nowrite_b
            r_nowrite = jnp.mean(r_nowrite_b, axis=-1)
        else:
            keep_write_b = jnp.ones((b, nb), dtype=jnp.float32)
            r_nowrite = jnp.zeros((b,), dtype=jnp.float32)

        raw_eta = route_raw * keep_write_b[:, :, None]  # Strictly in (0, 1) -> exact spectral contraction!
        if spec.force_write_floor > 0.0:
            fl = float(spec.force_write_floor)
            raw_eta = fl + (1.0 - fl) * raw_eta
        if spec.use_learned_routing:
            eta = raw_eta * valid_f[:, None, None]
        else:
            # Strict single-factor `-LearnedRouting`: holds parameters (+0.000%), `tau_b`, `g_b`, `r_nowrite`,
            # and aggregate write mass IDENTICAL to `slwm_v3_routed`, replacing ONLY per-bank content routing
            # with uniform bank-averaged write allocation (zero content-to-timescale specialization)!
            shared = jnp.mean(raw_eta, axis=1, keepdims=True)  # [B, 1, nh]
            eta = jnp.broadcast_to(shared, (b, nb, nh)) * valid_f[:, None, None]
        r_banks_eff = jnp.mean(eta, axis=-1)

    # 4. Normalized-key rank-1 delta overwrite in STRICT TPU-exact float32 (`eta in [0, 1]`, `||k||_2 < 1`)
    retrieved_k = jnp.einsum("bnhvk,bnhk->bnhv", M_tilde, k, precision=fp32_prec)
    delta_v = v - retrieved_k
    dM = jnp.einsum("bnh,bnhv,bnhk->bnhvk", eta, delta_v, k, precision=fp32_prec)
    M_next = M_tilde + dM

    # 5. Concatenated full-strength post-write readout across all timescale heads (strict fp32 state read!)
    post_y_bnhv = jnp.einsum("bnhvk,bnhk->bnhv", M_next, q, precision=fp32_prec)
    post_y = _soft_head_norm(post_y_bnhv)
    out = jnp.matmul(post_y, params["w_o"]) * valid_f[:, None]

    mem_frob_per_head = jnp.sqrt(jnp.sum(M_next * M_next, axis=(3, 4)) + 1e-12)  # [B, nb, nh], pre-o_norm_w!
    diags = {
        "r_banks": r_banks_eff,
        "r_nowrite": r_nowrite,
        "eff_write_eta": jnp.mean(eta, axis=(1, 2)),
        "eff_decay": jnp.mean(decay, axis=(1, 2)),
        "state_delta_norm": jnp.sqrt(jnp.mean(dM * dM, axis=(1, 2, 3, 4)) + 1e-12),
        "mem_frob_norm": jnp.mean(mem_frob_per_head, axis=(1, 2)),
        "mem_frob_norm_max": jnp.max(mem_frob_per_head, axis=(1, 2)),
        "mem_out_rms": jnp.sqrt(jnp.mean(out * out, axis=-1) + 1e-12),
        "prior_error": prior_err,
        "key_norm": jnp.sqrt(jnp.sum(k * k, axis=-1)).reshape(b, h_tot),
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
    keys = jax.random.split(key, 14)
    d = spec.d_model
    h_e = spec.explicit_kv_heads
    dkv = spec.explicit_kv_dim
    m_d = spec.width_mult
    mod_d = 16
    meta_in = mod_d + 3
    meta_hid = 32
    router_hid = max(32, d // 4)
    router_hid_base = max(32, spec.d_base // 4)
    out_std = spec.init_std / math.sqrt(max(float(h_e * dkv) / 64.0, 1.0))

    return {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "mod_emb": init_mup_weight(keys[0], (spec.n_modalities, mod_d), role="input", width_mult=m_d, base_std=0.25),
        "w_q": init_mup_weight(keys[1], (d, h_e * dkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_k": init_mup_weight(keys[2], (d, h_e * dkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_v": init_mup_weight(keys[3], (d, h_e * dkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_o": init_mup_weight(keys[4], (h_e * dkv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std),
        # Exact muP two-branch write-salience router (100% parity with SLWM-v3's content + epistemic branches!)
        "router_skip": init_mup_weight(
            keys[5], (d, h_e), role="hidden", width_mult=m_d, base_std=0.25 / math.sqrt(float(spec.d_base))
        ),
        "router_w1": init_mup_weight(
            keys[6], (d, router_hid), role="hidden", width_mult=m_d, base_std=1.0 / math.sqrt(float(spec.d_base))
        ),
        "router_b1": jnp.zeros((router_hid,), dtype=jnp.float32),
        "router_w2": init_mup_weight(
            keys[7], (router_hid, h_e), role="hidden", width_mult=m_d, base_std=0.25 / math.sqrt(float(router_hid_base))
        ),
        "router_meta_skip": init_mup_weight(
            keys[8], (meta_in, h_e), role="input", width_mult=m_d, base_std=0.25 / math.sqrt(float(meta_in))
        ),
        "router_meta_w1": init_mup_weight(
            keys[9], (meta_in, meta_hid), role="input", width_mult=m_d, base_std=1.0 / math.sqrt(float(meta_in))
        ),
        "router_meta_b1": jnp.zeros((meta_hid,), dtype=jnp.float32),
        "router_meta_w2": init_mup_weight(
            keys[10], (meta_hid, h_e), role="input", width_mult=m_d, base_std=0.25 / math.sqrt(float(meta_hid))
        ),
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

    # 1. Read-before-write cosine attention over valid entries (strict TPU-exact fp32 state math!)
    fp32_prec = jax.lax.Precision.HIGHEST
    q_sim = jnp.einsum("bhd,bhsd->bhs", q, kv_keys, precision=fp32_prec) * 8.0
    has_any = jnp.any(kv_valid, axis=-1, keepdims=True)  # [B, H, 1]
    safe_valid = jnp.where(has_any, kv_valid, jnp.ones_like(kv_valid))
    q_w = jnp.where(has_any, jax.nn.softmax(jnp.where(safe_valid, q_sim, -1e9), axis=-1), 0.0)
    prior_v = jnp.einsum("bhs,bhsd->bhd", q_w, kv_vals, precision=fp32_prec)
    prior_err = jnp.sqrt(jnp.mean((v - prior_v) ** 2, axis=(1, 2)) + 1e-6)  # [B]

    # True LRU age refresh on attended read hits (strictly when packet is valid!)
    read_hit = kv_valid & (q_w > 0.25) & (valid_mask_t[:, None, None] > 0.5)
    kv_age = jnp.where(read_hit, 0.0, kv_age)

    # 2. Exact muP two-branch write-salience router (per-head write probability in [0, 1])
    c_h = jax.nn.silu(jnp.matmul(xn, params["router_w1"]) + params["router_b1"])
    content_logits = jnp.matmul(xn, params["router_skip"]) + jnp.matmul(c_h, params["router_w2"]) + params["router_b2"]
    meta_feat = jnp.concatenate(
        [mod_e, jnp.log1p(dt_pos)[:, None], reliability_t[:, None].astype(jnp.float32), jnp.log1p(prior_err)[:, None]],
        axis=-1,
    )
    m_h = jax.nn.silu(jnp.matmul(meta_feat, params["router_meta_w1"]) + params["router_meta_b1"])
    meta_logits = jnp.matmul(meta_feat, params["router_meta_skip"]) + jnp.matmul(m_h, params["router_meta_w2"])
    write_prob = jax.nn.sigmoid(content_logits + meta_logits) * valid_mask_t[:, None]  # [B, H]

    # 3. Differentiable content-match overwrite vs. clean LRU/empty-slot insertion!
    k_sim = jnp.einsum("bhd,bhsd->bhs", k, kv_keys, precision=fp32_prec)  # [B, H, S]
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

    # 4. Post-write readout (strict TPU-exact fp32 state math!)
    post_sim = jnp.einsum("bhd,bhsd->bhs", q, new_keys, precision=fp32_prec) * 8.0
    post_has_any = jnp.any(new_valid, axis=-1, keepdims=True)
    post_safe = jnp.where(post_has_any, new_valid, jnp.ones_like(new_valid))
    post_w = jnp.where(post_has_any, jax.nn.softmax(jnp.where(post_safe, post_sim, -1e9), axis=-1), 0.0)
    post_v = jnp.einsum("bhs,bhsd->bhd", post_w, new_vals, precision=fp32_prec).reshape(b, h_e * dkv)
    out = jnp.matmul(post_v, params["w_o"]) * valid_mask_t[:, None]

    new_state = {
        "kv_keys": new_keys,
        "kv_vals": new_vals,
        "kv_age": new_age,
        "kv_valid": new_valid,
    }
    return out, new_state, {"r_nowrite": 1.0 - jnp.mean(write_prob, axis=-1), "prior_error": prior_err}
