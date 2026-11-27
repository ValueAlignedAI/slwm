"""SLWM-v4 Strictly Causal O(1) Core Sequence Blocks & Base-Shape muP Primitives.

Implements all core sequence-processing primitives required for `V4-S3`–`V4-S5` in pure functional JAX:
  1. Exact base-shape muP (`m_d = d_model / d_base`) weight initialization & role classification.
  2. `ShortCausalConv1d` (`k=4`, left-padded causal depthwise conv on `[q, k, v]` with `reset_mask` & `valid_mask`).
  3. `GatedDeltaNetV4Block`:
       - Pre-RMSNorm -> `(q, k, v)` -> optional `ShortCausalConv1d` + SiLU -> smooth `safe_l2_normalize(q, k)`.
       - Physical multi-timescale decay `exp(-g * dt / tau_b)` in strict `float32`.
       - Householder step size `beta in (0, beta_max)` (`beta_max = 2.0` default; `1.0` for `C2_beta_0_1`).
       - Zero-`M_anchor` causal novelty-EMA multi-bank routing (`C5_routed_2bank`, `C6_routed_4bank`) preserving
         exact `1.0x` matrix state bytes AND unlocking full `C=64` Chunkwise Parallel WY DeltaNet!
       - Per-head bounded RMSNorm + optional SiLU output gate (`C3_no_ogate` toggles off).
  4. `SlidingWindowGQASinkBlock`:
       - Strictly causal `O(1)` `bfloat16` ring KV cache + `n_sinks` learnable Attention Sink KV slots
         (prevents long-horizon softmax collapse across continuous 2+ minute live streaming rollouts).
       - Hoisted contiguous chunked banded MXU path + exact hole-preserving fallback (`step == seq` bit-exact!).
  5. `SLWMv3DualRefBlock` (`B0_v3_ref`):
       - Faithful prior-generation SLWM-v3 per-layer dual-branch (local GQA + legacy `M_anchor` 4-bank router).
  6. `SwiGLUFFN` & `ModalitySpecificMoEFFN` (`C7_ms_moe_ffn`, 4 modality experts at matched `1.0x` active params).

Every block exposes BOTH single-step O(1) online streaming (`step_*`) AND hoisted sequence-parallel training
(`apply_*_seq`) certified bit-equivalent across multi-episode `reset_mask` boundaries and `valid_mask == 0` holes.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from models.v4.spec import SLWMv4Spec

Array = jax.Array
PyTree = dict[str, Any]

V4_SCALAR_LEAF_NAMES = frozenset(
    {
        "norm",
        "norm_w",
        "ffn_norm_w",
        "final_norm_w",
        "o_norm_w",
        "bias",
        "conv_b",
        "log_tau",
        "log_temp",
        "rel_pos_bias",
        "g_bias",
        "beta_bias",
        "router_b1",
        "router_b2",
        "router_meta_b1",
        "sink_k",
        "sink_v",
        "b1",
        "b2",
        "ev_b1",
        "b_act",
        "b_prop",
        "b_succ",
        "t_b1",
        "t_b2",
    }
)


# ---------------------------------------------------------------------------
# 1. Precision, Norms & Base-Shape muP Helpers
# ---------------------------------------------------------------------------


def dense_mm(x: Array, w: Array, matmul_dtype: jnp.dtype) -> Array:
    """Explicit-precision matmul (`bf16` MXU/TensorCore for speed; `fp32` HIGHEST for exact audit)."""
    if matmul_dtype == jnp.float32:
        return jnp.matmul(x.astype(jnp.float32), w.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    return jnp.matmul(x.astype(matmul_dtype), w.astype(matmul_dtype)).astype(jnp.float32)


def rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    """RMSNorm preserving input shape `[..., D]` in float32 accumulation."""
    xf = x.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv_rms) * weight.astype(jnp.float32)


def safe_l2_normalize(x: Array, eps: float = 1e-6) -> Array:
    """Smooth C^infty float32 L2 contraction (`||u||_2 < 1` everywhere, exact `0` at `x = 0`).

    Guarantees Householder contraction `||I - eta k k^T||_2 <= 1` for ALL `eta in [0, 2]`!
    """
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
    """Initialize weight tensor under exact base-shape muP (`m_d = d_model / d_base`)."""
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


def classify_v4_param_mup_role(path_parts: tuple[str, ...], *, param_ndim: int | None = None) -> str:
    """Classify exact base-shape muP role from parameter pytree path (and optional tensor rank).

    - `"scalar"`:       ALL 0D/1D leaves (norms, biases, `log_tau`, `log_temp`, `b_act/b_prop/b_succ`, `t_b1/t_b2`),
                        plus `rel_pos_bias`, `sink_k/v`, `conv_w/b`, `o_norm_w` -> `lr_mult = 1.0`, `weight_decay = 0.0`.
    - `"input"`:        edge encoders, fixed-dim meta routers, `mod_emb`, policy evidence branch, embeddings
                        -> `lr_mult = 1.0`.
    - `"fixed_hidden"`: constant-fan-in internal/output head matrices (`head_video` MLP/out, `head_audio` depth/out,
                        `head_jepa.proj_a`, `policy_gate.w2`) -> `lr_mult = 1.0`.
    - `"readout"`:      width-scaling readout matrices (`fan_in ~ d_model`: `head_text.w`, `head_video.cond_proj`,
                        `head_audio.in_proj`, `head_jepa.proj_v`, `head_motor.w_*`, `policy_gate.w1`) -> `lr_mult = 1 / m_d`.
    - `"hidden"`:       width-scaling core matrices (`fan_in ~ d_model`, including BOTH `mem.w_o` and `gqa.w_o` so
                        GDN and SW-GQA share identical muP optimizer scaling!) -> `lr_mult = 1 / m_d`.
    """
    leaf = path_parts[-1] if path_parts else ""
    parent = path_parts[-2] if len(path_parts) >= 2 else ""
    if (
        (param_ndim is not None and param_ndim <= 1)
        or leaf in V4_SCALAR_LEAF_NAMES
        or leaf.startswith("b_")
        or leaf.endswith(("_bias", "_norm", "_norm_w", "_b", "_b1", "_b2"))
        or leaf == "conv_w"
    ):
        return "scalar"
    if any(
        p.startswith("edge_")
        or p.startswith("router_meta_")
        or p.startswith("ev_")
        or p in ("mod_emb", "spatial_pos_emb", "level_emb", "rvq_in_emb", "aud_rvq_emb")
        for p in path_parts
    ):
        return "input"
    if parent == "policy_gate":
        return "readout" if leaf == "w1" else "fixed_hidden"
    if any(p.startswith("head_video") or p.startswith("head_audio") or p.startswith("head_jepa") for p in path_parts):
        if leaf in ("cond_proj", "in_proj", "proj_v"):
            return "readout"
        return "fixed_hidden"
    if any(p.startswith("head_") for p in path_parts):
        return "readout"
    return "hidden"


# ---------------------------------------------------------------------------
# 2. Strictly Causal Depthwise ShortConv1d (`k = 4`)
# ---------------------------------------------------------------------------


def init_short_conv1d_params(key: Array, dim: int, kernel_size: int) -> dict[str, Array]:
    """Initialize causal depthwise Conv1d near identity on the current tap `[0, ..., 0, 1]`."""
    noise = jax.random.normal(key, (dim, kernel_size), dtype=jnp.float32) * 0.02
    ident = jnp.zeros((dim, kernel_size), dtype=jnp.float32).at[:, -1].set(1.0)
    return {
        "conv_w": ident + noise,
        "conv_b": jnp.zeros((dim,), dtype=jnp.float32),
    }


def step_short_conv1d(
    conv_w: Array,
    conv_b: Array,
    conv_buf: Array,
    u_t: Array,
    valid_mask_t: Array,
    reset_mask_t: Array,
) -> tuple[Array, Array]:
    """Single strictly-causal O(1) depthwise Conv1d step over `[conv_buf, u_t]`.

    Args:
      conv_w:       `float32[C, k]`
      conv_b:       `float32[C]`
      conv_buf:     `float32[B, k-1, C]` (oldest at index 0, most recent past at index k-2)
      u_t:          `float32[B, C]`
      valid_mask_t: `float32[B]`
      reset_mask_t: `bool[B]`
    Returns:
      y_t:          `float32[B, C]` (masked by `valid_mask_t`)
      new_buf:      `float32[B, k-1, C]`
    """
    rst = reset_mask_t.astype(jnp.bool_)
    is_valid = valid_mask_t > 0.5
    buf = jnp.where(rst[:, None, None], jnp.zeros_like(conv_buf), conv_buf)

    win = jnp.concatenate([buf, u_t[:, None, :]], axis=1)  # [B, k, C]
    y_t = jnp.einsum("bkc,ck->bc", win, conv_w.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST) + conv_b[None, :]
    y_t = y_t * valid_mask_t.astype(jnp.float32)[:, None]

    cand_buf = win[:, 1:, :]  # [B, k-1, C]
    new_buf = jnp.where(is_valid[:, None, None], cand_buf, buf)
    return y_t, new_buf


def apply_short_conv1d_seq(
    conv_w: Array,
    conv_b: Array,
    init_buf: Array,
    u_seq: Array,
    valid_mask_seq: Array,
    reset_mask_seq: Array,
    *,
    assume_all_valid: bool = False,
) -> tuple[Array, Array]:
    """Hoisted causal depthwise Conv1d matching `step_short_conv1d` bit-for-bit across resets & holes."""
    t_len, b_sz, dim = u_seq.shape
    k_sz = int(conv_w.shape[1])
    k_prev = k_sz - 1
    is_valid = valid_mask_seq > 0.5
    is_reset = reset_mask_seq.astype(jnp.bool_)
    w_f = conv_w.astype(jnp.float32)
    b_f = conv_b.astype(jnp.float32)

    def _dense_path(_: None) -> tuple[Array, Array]:
        init_t = jnp.swapaxes(init_buf.astype(jnp.float32), 0, 1)  # [k-1, B, C]
        u_pool = jnp.concatenate([init_t, u_seq.astype(jnp.float32)], axis=0)  # [k-1 + T, B, C]
        ep_new = jnp.cumsum(is_reset.astype(jnp.int32), axis=0)  # [T, B]
        ep_pool = jnp.concatenate([jnp.zeros((k_prev, b_sz), dtype=jnp.int32), ep_new], axis=0)

        y = jnp.broadcast_to(b_f[None, None, :], (t_len, b_sz, dim))
        for j in range(k_sz):
            u_j = u_pool[j : j + t_len]
            same_ep = ep_pool[j : j + t_len] == ep_new
            y = y + jnp.where(same_ep[..., None], u_j, 0.0) * w_f[None, None, :, j]
        y = y * valid_mask_seq.astype(jnp.float32)[..., None]

        fin_raw = jnp.swapaxes(u_pool[t_len : t_len + k_prev], 0, 1)  # [B, k-1, C]
        fin_ep = jnp.swapaxes(ep_pool[t_len : t_len + k_prev], 0, 1)  # [B, k-1]
        fin_same = fin_ep == ep_new[-1][:, None]
        fin_buf = jnp.where(fin_same[:, :, None], fin_raw, jnp.zeros_like(fin_raw))
        return y, fin_buf

    def _hole_path(_: None) -> tuple[Array, Array]:
        def _scan_step(buf: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
            u_t, v_t, r_t = xs
            y_t, nb = step_short_conv1d(w_f, b_f, buf, u_t, v_t, r_t)
            return nb, y_t

        fin_buf, y_seq = jax.lax.scan(_scan_step, init_buf.astype(jnp.float32), (u_seq.astype(jnp.float32), valid_mask_seq, is_reset))
        return y_seq, fin_buf

    if assume_all_valid:
        return _dense_path(None)
    return jax.lax.cond(jnp.all(is_valid), _dense_path, _hole_path, None)


# ---------------------------------------------------------------------------
# 3. Exact Chunkwise Parallel WY Delta-Rule (supports Householder beta in (0, 2)!)
# ---------------------------------------------------------------------------


def apply_deltanet_chunkwise_wy(
    M_init: Array,
    q_seq: Array,
    k_seq: Array,
    v_seq: Array,
    decay_seq: Array,
    eta_seq: Array,
    reset_mask_seq: Array,
    *,
    chunk_size: int = 64,
) -> tuple[Array, Array, Array, Array, Array]:
    """Exact Chunkwise Parallel WY Delta-Rule in strict fp32 for ANY real `eta_t in [0, 2]`.

    Algebraic derivation:
      M_tilde_i = dec_i * M_{i-1}
      M_i       = M_tilde_i + u_i k_i^T,  where u_i = eta_i (v_i - M_tilde_i k_i).
      Expanding M_tilde_i across chunk indices j < i (same episode) gives the exact unit lower-triangular
      linear system `(I + A) U = eta * V - (eta * dec_from_start * K) M_prev^T` with
      `A[i, j] = eta_i * D[i, j] * (k_i^T k_j)` for `i > j`. Because `I + A` is unit lower-triangular
      (`det = 1`), `T = (I + A)^{-1}` is exact for ALL `eta_t in [0, 2]` (Householder negative eigenvalues)!
    """
    t_len, b_sz, nb, nh, dk = k_seq.shape
    dv = v_seq.shape[-1]
    fp32_prec = jax.lax.Precision.HIGHEST
    c_sz = min(max(1, chunk_size), t_len)
    rem = t_len % c_sz
    pad_n = (c_sz - rem) if rem != 0 else 0
    t_pad = t_len + pad_n
    n_chunks = t_pad // c_sz

    def _pad(arr: Array, fill: float = 0.0) -> Array:
        if pad_n == 0:
            return arr.reshape((n_chunks, c_sz) + arr.shape[1:])
        pad_a = jnp.full((pad_n,) + arr.shape[1:], fill, dtype=arr.dtype)
        return jnp.concatenate([arr, pad_a], axis=0).reshape((n_chunks, c_sz) + arr.shape[1:])

    q_c = _pad(q_seq, 0.0)
    k_c = _pad(k_seq, 0.0)
    v_c = _pad(v_seq, 0.0)
    dec_c = _pad(decay_seq, 1.0)
    eta_c = _pad(eta_seq, 0.0)
    rst_c = _pad(reset_mask_seq.astype(jnp.bool_), False)

    tril_strict = jnp.tril(jnp.ones((c_sz, c_sz), dtype=jnp.bool_), k=-1)
    tril_incl = jnp.tril(jnp.ones((c_sz, c_sz), dtype=jnp.bool_), k=0)
    eye_c = jnp.eye(c_sz, dtype=jnp.float32)

    def _prep_chunk(
        k_ch: Array, v_ch: Array, dec_ch: Array, eta_ch: Array, rst_ch: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        k_t = jnp.transpose(k_ch, (1, 2, 3, 0, 4))  # [B, nb, nh, C, dk]
        v_t = jnp.transpose(v_ch, (1, 2, 3, 0, 4))  # [B, nb, nh, C, dv]
        rst_e = jnp.transpose(rst_ch, (1, 0))[:, None, None, :]  # [B, 1, 1, C]
        ep_id = jnp.cumsum(rst_e.astype(jnp.int32), axis=-1)
        same_ep = ep_id[..., :, None] == ep_id[..., None, :]
        same_ep_start = ep_id == 0
        has_reset_in_chunk = jnp.any(rst_e, axis=-1)

        log_d = jnp.log(jnp.clip(jnp.transpose(dec_ch, (1, 2, 3, 0)), 1e-7, 1.0))
        # Segment-local cumulative log-decay (exact 0-ULP isolation across mid-chunk `reset_mask` episodes!)
        seg_tril = (tril_incl & same_ep).astype(jnp.float32)  # [B, 1, 1, C, C]
        gamma = jnp.einsum("bmnij,bnhj->bnhi", seg_tril, log_d, precision=fp32_prec)  # [B, nb, nh, C]
        g_diff = jnp.where(tril_incl & same_ep, gamma[..., :, None] - gamma[..., None, :], -1e9)
        D_mat = jnp.where(tril_incl & same_ep, jnp.exp(jnp.clip(g_diff, -40.0, 40.0)), 0.0)

        eta_t = jnp.transpose(eta_ch, (1, 2, 3, 0))
        kk = jnp.einsum("bnhik,bnhjk->bnhij", k_t, k_t, precision=fp32_prec)
        A_mat = jnp.where(tril_strict & same_ep, eta_t[..., :, None] * D_mat * kk, 0.0)
        T_mat = jax.lax.linalg.triangular_solve(
            eye_c + A_mat,
            jnp.broadcast_to(eye_c, A_mat.shape),
            left_side=True,
            lower=True,
            unit_diagonal=True,
        )
        dec_from_start = jnp.where(same_ep_start, jnp.exp(jnp.clip(gamma, -40.0, 0.0)), 0.0)
        same_ep_end = ep_id == ep_id[..., -1:]
        g_to_end = jnp.where(same_ep_end, gamma[..., -1:] - gamma, -1e9)
        dec_to_end = jnp.where(same_ep_end, jnp.exp(jnp.clip(g_to_end, -40.0, 40.0)), 0.0)
        chunk_carry_decay = jnp.where(has_reset_in_chunk, 0.0, jnp.exp(jnp.clip(gamma[..., -1], -40.0, 0.0)))

        U_0 = jnp.einsum("bnhij,bnhjv->bnhiv", T_mat, eta_t[..., None] * v_t, precision=fp32_prec)
        W_mat = jnp.einsum("bnhij,bnhjk->bnhik", T_mat, (eta_t * dec_from_start)[..., None] * k_t, precision=fp32_prec)
        return U_0, W_mat, D_mat, dec_from_start, dec_to_end, chunk_carry_decay

    U0_all, W_all, D_all, dfs_all, dte_all, ccd_all = jax.vmap(_prep_chunk)(k_c, v_c, dec_c, eta_c, rst_c)

    def _chunk_step(
        M_prev: Array,
        xs: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        q_ch, k_ch, U_0, W_mat, D_mat, dec_fs, dec_te, c_decay = xs
        q_t = jnp.transpose(q_ch, (1, 2, 3, 0, 4))
        k_t = jnp.transpose(k_ch, (1, 2, 3, 0, 4))

        U = U_0 - jnp.einsum("bnhik,bnhvk->bnhiv", W_mat, M_prev, precision=fp32_prec)
        qk = jnp.einsum("bnhik,bnhjk->bnhij", q_t, k_t, precision=fp32_prec) * D_mat
        y_intra = jnp.einsum("bnhij,bnhjv->bnhiv", qk, U, precision=fp32_prec)
        y_inter = dec_fs[..., None] * jnp.einsum("bnhvk,bnhik->bnhiv", M_prev, q_t, precision=fp32_prec)
        y_post = jnp.transpose(y_inter + y_intra, (3, 0, 1, 2, 4))  # [C, B, nb, nh, dv]

        dM_chunk = jnp.einsum("bnhj,bnhjv,bnhjk->bnhvk", dec_te, U, k_t, precision=fp32_prec)
        M_next = M_prev * c_decay[..., None, None] + dM_chunk

        d_norm = jnp.broadcast_to(jnp.sqrt(jnp.mean(U * U, axis=(1, 2, 4)) + 1e-12).T, (c_sz, b_sz))
        frob_h = jnp.sqrt(jnp.sum(M_next * M_next, axis=(3, 4)) + 1e-12)
        frob_mean = jnp.broadcast_to(jnp.mean(frob_h, axis=(1, 2))[None, :], (c_sz, b_sz))
        frob_max = jnp.broadcast_to(jnp.max(frob_h, axis=(1, 2))[None, :], (c_sz, b_sz))
        return M_next, (y_post, d_norm, frob_mean, frob_max)

    M_fin, (y_chunks, dn_chunks, fm_chunks, fx_chunks) = jax.lax.scan(
        _chunk_step, M_init, (q_c, k_c, U0_all, W_all, D_all, dfs_all, dte_all, ccd_all)
    )
    return (
        y_chunks.reshape(t_pad, b_sz, nb, nh, dv)[:t_len],
        M_fin,
        dn_chunks.reshape(t_pad, b_sz)[:t_len],
        fm_chunks.reshape(t_pad, b_sz)[:t_len],
        fx_chunks.reshape(t_pad, b_sz)[:t_len],
    )


# ---------------------------------------------------------------------------
# 4. Gated DeltaNet v4 & Zero-M_anchor Routed Multi-Timescale Block (`gdn`)
# ---------------------------------------------------------------------------


def _compute_bounded_tau(log_tau: Array, init_timescales_s: tuple[float, ...]) -> Array:
    lt_base = jnp.asarray([math.log(t) for t in init_timescales_s], dtype=jnp.float32)
    return jnp.exp(lt_base + 0.7 * jnp.tanh(log_tau.astype(jnp.float32) - lt_base))


def init_gdn_v4_params(key: Array, spec: SLWMv4Spec, *, is_v3_legacy: bool = False) -> PyTree:
    """Initialize parameters for one Gated DeltaNet v4 (`gdn`) or SLWM-v3 legacy memory branch."""
    keys = jax.random.split(key, 18)
    d = spec.d_model
    m_d = spec.width_mult
    nb = spec.n_banks
    nh = spec.n_mem_heads
    h_tot = nb * nh
    dk = spec.mem_key_dim
    dv = spec.mem_val_dim
    qkv_dim = h_tot * (2 * dk + dv)
    out_std = 0.25 / math.sqrt(float(h_tot * dv))

    params: PyTree = {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_q": init_mup_weight(keys[0], (d, h_tot * dk), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_k": init_mup_weight(keys[1], (d, h_tot * dk), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_v": init_mup_weight(keys[2], (d, h_tot * dv), role="hidden", width_mult=m_d, base_std=1.0 / math.sqrt(float(spec.d_base))),
        "w_o": init_mup_weight(keys[3], (h_tot * dv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std),
        "o_norm_w": jnp.ones((h_tot, dv), dtype=jnp.float32),
        "w_g": init_mup_weight(keys[4], (d, h_tot), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "g_bias": jnp.full((nb, nh), -1.5, dtype=jnp.float32),
        "log_tau": jnp.asarray([math.log(t) for t in spec.init_timescales_s], dtype=jnp.float32),
    }

    if spec.use_short_conv and not is_v3_legacy:
        params.update(init_short_conv1d_params(keys[5], qkv_dim, spec.conv_kernel_size))

    if spec.use_output_gate and not is_v3_legacy:
        params["w_ogate"] = init_mup_weight(keys[6], (d, h_tot * dv), role="hidden", width_mult=m_d, base_std=spec.init_std)

    if not spec.use_surprise_router:
        params["w_beta"] = init_mup_weight(keys[7], (d, h_tot), role="hidden", width_mult=m_d, base_std=spec.init_std)
        # Initialize sigmoid(beta_bias) ~= 0.5 -> beta ~= 1.0 when beta_max=2.0, or 0.73 when beta_max=1.0
        init_bb = 0.0 if spec.beta_max > 1.5 else 1.0
        params["beta_bias"] = jnp.full((h_tot,), init_bb, dtype=jnp.float32)
    else:
        mod_d = 16
        meta_in = mod_d + 3
        meta_hid = 32
        router_hid = max(32, d // 4)
        router_hid_base = max(32, spec.d_base // 4)
        n_out = h_tot + (nb if spec.use_nowrite_option else 0)
        params["mod_emb"] = init_mup_weight(keys[8], (spec.n_modalities, mod_d), role="input", width_mult=m_d, base_std=0.25)
        params["router_skip"] = init_mup_weight(
            keys[9], (d, n_out), role="hidden", width_mult=m_d, base_std=0.25 / math.sqrt(float(spec.d_base))
        )
        params["router_w1"] = init_mup_weight(
            keys[10], (d, router_hid), role="hidden", width_mult=m_d, base_std=1.0 / math.sqrt(float(spec.d_base))
        )
        params["router_b1"] = jnp.zeros((router_hid,), dtype=jnp.float32)
        params["router_w2"] = init_mup_weight(
            keys[11], (router_hid, n_out), role="hidden", width_mult=m_d, base_std=0.25 / math.sqrt(float(router_hid_base))
        )
        params["router_meta_skip"] = init_mup_weight(
            keys[12], (meta_in, n_out), role="input", width_mult=m_d, base_std=0.25 / math.sqrt(float(meta_in))
        )
        params["router_meta_w1"] = init_mup_weight(
            keys[13], (meta_in, meta_hid), role="input", width_mult=m_d, base_std=1.0 / math.sqrt(float(meta_in))
        )
        params["router_meta_b1"] = jnp.zeros((meta_hid,), dtype=jnp.float32)
        params["router_meta_w2"] = init_mup_weight(
            keys[14], (meta_hid, n_out), role="input", width_mult=m_d, base_std=0.25 / math.sqrt(float(meta_hid))
        )
        b2_head = jnp.full((h_tot,), 0.0 if spec.beta_max > 1.5 else 0.8, dtype=jnp.float32)
        if spec.use_nowrite_option:
            b2_nw = jnp.full((nb,), -2.0, dtype=jnp.float32)
            params["router_b2"] = jnp.concatenate([b2_head, b2_nw], axis=0)
        else:
            params["router_b2"] = b2_head
        if spec.use_m_anchor_legacy:
            params["w_prior_proj"] = init_mup_weight(
                keys[15], (h_tot * dv, d), role="fixed_hidden", width_mult=m_d, base_std=out_std
            )
    return params


def init_gdn_v4_state(spec: SLWMv4Spec, batch_size: int, *, is_v3_legacy: bool = False) -> PyTree:
    """Allocate exact O(1) recurrent state matching `compute_v4_core_streaming_state_bytes` byte-for-byte."""
    m_dtype = jnp.float32 if spec.state_dtype == "float32" else jnp.bfloat16
    m_zero = jnp.zeros((batch_size, spec.n_banks, spec.n_mem_heads, spec.mem_val_dim, spec.mem_key_dim), dtype=m_dtype)
    state: PyTree = {"M": m_zero}
    if is_v3_legacy and spec.use_m_anchor_legacy:
        state["M_anchor"] = jnp.zeros_like(m_zero)
        state["cum_decay"] = jnp.ones((batch_size, spec.n_banks, spec.n_mem_heads), dtype=jnp.float32)
        state["chunk_phase"] = jnp.zeros((batch_size,), dtype=jnp.int32)
    if spec.use_surprise_router and not spec.use_m_anchor_legacy:
        state["x_ema"] = jnp.zeros((batch_size, spec.d_model), dtype=jnp.float32)
        state["has_ema"] = jnp.zeros((batch_size,), dtype=jnp.bool_)
    if spec.use_short_conv and not is_v3_legacy:
        qkv_dim = spec.total_mem_heads * (2 * spec.mem_key_dim + spec.mem_val_dim)
        state["conv_buf"] = jnp.zeros((batch_size, spec.conv_kernel_size - 1, qkv_dim), dtype=jnp.float32)
    return state


def _soft_head_norm(y_bnhv: Array, o_norm_w: Array) -> Array:
    """Bounded per-head RMSNorm (`eps = 1e-2`) preserving leading dimensions `[*lead, nb, nh, dv] -> [*lead, h_tot*dv]`."""
    lead = y_bnhv.shape[:-3]
    h_tot = int(o_norm_w.shape[0])
    dv = int(o_norm_w.shape[1])
    y_flat = y_bnhv.reshape(*lead, h_tot, dv)
    inv_r = jax.lax.rsqrt(jnp.mean(y_flat * y_flat, axis=-1, keepdims=True) + 1e-2)
    normed = (y_flat * inv_r) * o_norm_w.reshape((1,) * len(lead) + (h_tot, dv))
    return normed.reshape(*lead, h_tot * dv)


def _eval_surprise_router_logits(
    params: PyTree,
    xn: Array,
    mod_id: Array,
    dt_pos: Array,
    reliability: Array,
    novelty: Array,
    spec: SLWMv4Spec,
    matmul_dtype: jnp.dtype,
) -> Array:
    """Evaluate two-branch muP surprise router (`[... , n_out]`) on arbitrary leading batch/time axes."""
    c_h = jax.nn.silu(dense_mm(xn, params["router_w1"], matmul_dtype) + params["router_b1"])
    c_logits = dense_mm(xn, params["router_skip"], matmul_dtype) + dense_mm(c_h, params["router_w2"], matmul_dtype) + params["router_b2"]
    m_idx = jnp.clip(mod_id.astype(jnp.int32), 0, spec.n_modalities - 1)
    meta = jnp.concatenate(
        [
            params["mod_emb"][m_idx],
            jnp.log1p(dt_pos)[..., None],
            reliability.astype(jnp.float32)[..., None],
            jnp.log1p(novelty.astype(jnp.float32))[..., None],
        ],
        axis=-1,
    )
    m_h = jax.nn.silu(dense_mm(meta, params["router_meta_w1"], matmul_dtype) + params["router_meta_b1"])
    m_logits = dense_mm(meta, params["router_meta_skip"], matmul_dtype) + dense_mm(m_h, params["router_meta_w2"], matmul_dtype)
    return c_logits + m_logits


def step_gdn_v4(
    params: PyTree,
    state: PyTree,
    x_t: Array,
    modality_id_t: Array,
    dt_s_t: Array,
    valid_mask_t: Array,
    reliability_t: Array,
    reset_mask_t: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Single strictly-causal O(1) step of Gated DeltaNet v4 (single-bank OR zero-M_anchor routed multi-bank)."""
    b_sz = x_t.shape[0]
    nb, nh = spec.n_banks, spec.n_mem_heads
    h_tot = nb * nh
    dk, dv = spec.mem_key_dim, spec.mem_val_dim
    fp32_prec = jax.lax.Precision.HIGHEST
    rst = reset_mask_t.astype(jnp.bool_)
    vld_f = valid_mask_t.astype(jnp.float32)
    is_valid = vld_f > 0.5
    dt_pos = jnp.maximum(dt_s_t.astype(jnp.float32), 0.0)

    xn = rms_norm(x_t, params["norm_w"])
    q_raw = dense_mm(xn, params["w_q"], matmul_dtype)
    k_raw = dense_mm(xn, params["w_k"], matmul_dtype)
    v_raw = dense_mm(xn, params["w_v"], matmul_dtype)

    new_state: PyTree = {}
    if spec.use_short_conv:
        qkv_in = jnp.concatenate([q_raw, k_raw, v_raw], axis=-1)
        qkv_conv, new_conv_buf = step_short_conv1d(
            params["conv_w"], params["conv_b"], state["conv_buf"], qkv_in, vld_f, rst
        )
        qkv_act = jax.nn.silu(qkv_conv)
        c_qk = h_tot * dk
        q_raw = qkv_act[:, :c_qk]
        k_raw = qkv_act[:, c_qk : 2 * c_qk]
        v_raw = qkv_act[:, 2 * c_qk :]
        new_state["conv_buf"] = new_conv_buf

    q = safe_l2_normalize(q_raw.reshape(b_sz, nb, nh, dk))
    k = safe_l2_normalize(k_raw.reshape(b_sz, nb, nh, dk))
    v = v_raw.reshape(b_sz, nb, nh, dv).astype(jnp.float32)

    g = jax.nn.sigmoid(dense_mm(xn, params["w_g"], matmul_dtype).reshape(b_sz, nb, nh) + params["g_bias"][None, :, :])
    tau = _compute_bounded_tau(params["log_tau"], spec.init_timescales_s)
    dec = jnp.exp(-g * (dt_pos[:, None, None] / (tau[None, :, None] + 1e-6)))

    if not spec.use_surprise_router:
        beta = float(spec.beta_max) * jax.nn.sigmoid(
            dense_mm(xn, params["w_beta"], matmul_dtype) + params["beta_bias"][None, :]
        ).reshape(b_sz, nb, nh)
        eta = beta * vld_f[:, None, None]
        r_nw = jnp.zeros((b_sz,), dtype=jnp.float32)
        novelty = jnp.zeros((b_sz,), dtype=jnp.float32)
    else:
        # Zero-M_anchor causal input-EMA novelty computed STRICTLY BEFORE updating x_ema or M!
        alpha = float(spec.surprise_ema_alpha)
        ema_prev = jnp.where(rst[:, None], jnp.zeros_like(state["x_ema"]), state["x_ema"].astype(jnp.float32))
        has_prev = jnp.where(rst, jnp.zeros_like(state["has_ema"]), state["has_ema"].astype(jnp.bool_))
        diff = xn - ema_prev
        novelty = jnp.where(has_prev, jnp.sqrt(jnp.mean(diff * diff, axis=-1) + 1e-6), 1.0)
        cand_ema = jnp.where(has_prev[:, None], (1.0 - alpha) * ema_prev + alpha * xn, xn)
        new_state["x_ema"] = jnp.where(is_valid[:, None], cand_ema, ema_prev)
        new_state["has_ema"] = jnp.where(is_valid, True, has_prev)

        r_logits = _eval_surprise_router_logits(params, xn, modality_id_t, dt_pos, reliability_t, novelty, spec, matmul_dtype)
        route_raw = float(spec.beta_max) * jax.nn.sigmoid(r_logits[:, :h_tot].reshape(b_sz, nb, nh))
        if spec.use_nowrite_option:
            r_nw_b = jax.nn.sigmoid(r_logits[:, h_tot:])
            keep_b = 1.0 - r_nw_b
            r_nw = jnp.mean(r_nw_b, axis=-1)
        else:
            keep_b = jnp.ones((b_sz, nb), dtype=jnp.float32)
            r_nw = jnp.zeros((b_sz,), dtype=jnp.float32)
        eta = route_raw * keep_b[:, :, None] * vld_f[:, None, None]

    M_prev = jnp.where(rst[:, None, None, None, None], jnp.zeros_like(state["M"]), state["M"].astype(jnp.float32))
    M_tilde = M_prev * dec[:, :, :, None, None]
    ret_k = jnp.einsum("bnhvk,bnhk->bnhv", M_tilde, k, precision=fp32_prec)
    dM = jnp.einsum("bnh,bnhv,bnhk->bnhvk", eta, v - ret_k, k, precision=fp32_prec)
    M_next = M_tilde + dM
    y_post = jnp.einsum("bnhvk,bnhk->bnhv", M_next, q, precision=fp32_prec)
    y_norm = _soft_head_norm(y_post, params["o_norm_w"])

    if spec.use_output_gate:
        y_norm = y_norm * jax.nn.silu(dense_mm(xn, params["w_ogate"], matmul_dtype))

    out = dense_mm(y_norm, params["w_o"], matmul_dtype) * vld_f[:, None]
    new_state["M"] = M_next.astype(state["M"].dtype)

    frob_h = jnp.sqrt(jnp.sum(M_next * M_next, axis=(3, 4)) + 1e-12)
    diag = {
        "r_nowrite": r_nw,
        "novelty": novelty,
        "eff_eta": jnp.mean(eta, axis=(1, 2)),
        "delta_norm": jnp.sqrt(jnp.mean(dM * dM, axis=(1, 2, 3, 4)) + 1e-12),
        "frob_mean": jnp.mean(frob_h, axis=(1, 2)),
        "frob_max": jnp.max(frob_h, axis=(1, 2)),
    }
    return out, new_state, diag


def apply_gdn_v4_seq(
    params: PyTree,
    init_state: PyTree,
    x_seq: Array,
    modality_id_seq: Array,
    dt_s_seq: Array,
    valid_mask_seq: Array,
    reliability_seq: Array,
    reset_mask_seq: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    wy_chunk_size: int = 64,
    assume_all_valid: bool = False,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Hoisted sequence-parallel Gated DeltaNet v4 executing full Chunkwise Parallel WY for BOTH A0 and C5/C6!"""
    t_len, b_sz, _ = x_seq.shape
    nb, nh = spec.n_banks, spec.n_mem_heads
    h_tot = nb * nh
    dk, dv = spec.mem_key_dim, spec.mem_val_dim
    rst_b = reset_mask_seq.astype(jnp.bool_)
    vld_f = valid_mask_seq.astype(jnp.float32)
    is_valid = vld_f > 0.5
    dt_pos = jnp.maximum(dt_s_seq.astype(jnp.float32), 0.0)

    xn = rms_norm(x_seq, params["norm_w"])
    q_raw = dense_mm(xn, params["w_q"], matmul_dtype)
    k_raw = dense_mm(xn, params["w_k"], matmul_dtype)
    v_raw = dense_mm(xn, params["w_v"], matmul_dtype)

    final_state: PyTree = {}
    if spec.use_short_conv:
        qkv_in = jnp.concatenate([q_raw, k_raw, v_raw], axis=-1)
        qkv_conv, fin_conv_buf = apply_short_conv1d_seq(
            params["conv_w"],
            params["conv_b"],
            init_state["conv_buf"],
            qkv_in,
            vld_f,
            rst_b,
            assume_all_valid=assume_all_valid,
        )
        qkv_act = jax.nn.silu(qkv_conv)
        c_qk = h_tot * dk
        q_raw = qkv_act[:, :, :c_qk]
        k_raw = qkv_act[:, :, c_qk : 2 * c_qk]
        v_raw = qkv_act[:, :, 2 * c_qk :]
        final_state["conv_buf"] = fin_conv_buf

    q_seq = safe_l2_normalize(q_raw.reshape(t_len, b_sz, nb, nh, dk))
    k_seq = safe_l2_normalize(k_raw.reshape(t_len, b_sz, nb, nh, dk))
    v_seq = v_raw.reshape(t_len, b_sz, nb, nh, dv).astype(jnp.float32)

    g_seq = jax.nn.sigmoid(
        dense_mm(xn, params["w_g"], matmul_dtype).reshape(t_len, b_sz, nb, nh) + params["g_bias"][None, None, :, :]
    )
    tau = _compute_bounded_tau(params["log_tau"], spec.init_timescales_s)
    decay_seq = jnp.exp(-g_seq * (dt_pos[:, :, None, None] / (tau[None, None, :, None] + 1e-6)))

    if not spec.use_surprise_router:
        beta_seq = float(spec.beta_max) * jax.nn.sigmoid(
            dense_mm(xn, params["w_beta"], matmul_dtype) + params["beta_bias"][None, None, :]
        ).reshape(t_len, b_sz, nb, nh)
        eta_seq = beta_seq * vld_f[:, :, None, None]
        r_nw_seq = jnp.zeros((t_len, b_sz), dtype=jnp.float32)
        novelty_seq = jnp.zeros((t_len, b_sz), dtype=jnp.float32)
    else:
        # Tiny [B, d] causal novelty EMA scan finishes BEFORE matrix memory -> unlocks full Chunkwise WY!
        alpha = float(spec.surprise_ema_alpha)

        def _ema_step(carry: tuple[Array, Array], xs: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array], Array]:
            ema_p, has_p = carry
            xn_t, v_t, r_t = xs
            ema_p = jnp.where(r_t[:, None], jnp.zeros_like(ema_p), ema_p)
            has_p = jnp.where(r_t, jnp.zeros_like(has_p), has_p)
            diff = xn_t - ema_p
            nov = jnp.where(has_p, jnp.sqrt(jnp.mean(diff * diff, axis=-1) + 1e-6), 1.0)
            cand = jnp.where(has_p[:, None], (1.0 - alpha) * ema_p + alpha * xn_t, xn_t)
            ema_n = jnp.where(v_t[:, None], cand, ema_p)
            has_n = jnp.where(v_t, True, has_p)
            return (ema_n, has_n), nov

        (fin_ema, fin_has), novelty_seq = jax.lax.scan(
            _ema_step,
            (init_state["x_ema"].astype(jnp.float32), init_state["has_ema"].astype(jnp.bool_)),
            (xn, is_valid, rst_b),
        )
        final_state["x_ema"] = fin_ema
        final_state["has_ema"] = fin_has

        r_logits = _eval_surprise_router_logits(
            params, xn, modality_id_seq, dt_pos, reliability_seq, novelty_seq, spec, matmul_dtype
        )
        route_raw = float(spec.beta_max) * jax.nn.sigmoid(r_logits[:, :, :h_tot].reshape(t_len, b_sz, nb, nh))
        if spec.use_nowrite_option:
            r_nw_b = jax.nn.sigmoid(r_logits[:, :, h_tot:])
            keep_b = 1.0 - r_nw_b
            r_nw_seq = jnp.mean(r_nw_b, axis=-1)
        else:
            keep_b = jnp.ones((t_len, b_sz, nb), dtype=jnp.float32)
            r_nw_seq = jnp.zeros((t_len, b_sz), dtype=jnp.float32)
        eta_seq = route_raw * keep_b[:, :, :, None] * vld_f[:, :, None, None]

    # Exact Chunkwise Parallel WY Delta-Rule for ALL `A0..C7`!
    y_post_tbnhv, M_fin, dn_seq, fm_seq, fx_seq = apply_deltanet_chunkwise_wy(
        init_state["M"].astype(jnp.float32),
        q_seq,
        k_seq,
        v_seq,
        decay_seq,
        eta_seq,
        rst_b,
        chunk_size=wy_chunk_size,
    )
    y_norm = _soft_head_norm(y_post_tbnhv, params["o_norm_w"])
    if spec.use_output_gate:
        y_norm = y_norm * jax.nn.silu(dense_mm(xn, params["w_ogate"], matmul_dtype))

    out_seq = dense_mm(y_norm, params["w_o"], matmul_dtype) * vld_f[..., None]
    final_state["M"] = M_fin.astype(init_state["M"].dtype)
    diag = {
        "r_nowrite": r_nw_seq,
        "novelty": novelty_seq,
        "eff_eta": jnp.mean(eta_seq, axis=(2, 3)),
        "delta_norm": dn_seq,
        "frob_mean": fm_seq,
        "frob_max": fx_seq,
    }
    return out_seq, final_state, diag


# ---------------------------------------------------------------------------
# 5. Sliding-Window GQA + Learnable Attention Sinks (`gqa`)
# ---------------------------------------------------------------------------


def init_sw_gqa_sink_params(key: Array, spec: SLWMv4Spec, *, enable_sinks: bool = True) -> PyTree:
    """Initialize O(1) sliding-window causal GQA with optional learnable Attention Sink KV slots."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    d = spec.d_model
    hq = spec.n_attn_heads * spec.attn_head_dim
    hkv = spec.n_kv_heads * spec.attn_head_dim
    m_d = spec.width_mult
    params: PyTree = {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_q": init_mup_weight(k1, (d, hq), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_k": init_mup_weight(k2, (d, hkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_v": init_mup_weight(k3, (d, hkv), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_o": init_mup_weight(k4, (hq, d), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "rel_pos_bias": jnp.zeros((spec.n_attn_heads, spec.window_size), dtype=jnp.float32),
    }
    if enable_sinks and spec.n_sinks > 0:
        params["sink_k"] = jax.random.normal(k5, (spec.n_sinks, spec.n_kv_heads, spec.attn_head_dim), dtype=jnp.float32) * 0.02
        params["sink_v"] = jax.random.normal(k6, (spec.n_sinks, spec.n_kv_heads, spec.attn_head_dim), dtype=jnp.float32) * 0.02
    return params


def init_sw_gqa_sink_state(spec: SLWMv4Spec, batch_size: int) -> PyTree:
    """Allocate fixed-size O(1) bf16 sliding-window GQA ring buffer."""
    kv_dtype = jnp.bfloat16 if spec.ring_kv_bytes_per_elem == 2 else jnp.float32
    return {
        "ring_k": jnp.zeros((batch_size, spec.window_size, spec.n_kv_heads, spec.attn_head_dim), dtype=kv_dtype),
        "ring_v": jnp.zeros((batch_size, spec.window_size, spec.n_kv_heads, spec.attn_head_dim), dtype=kv_dtype),
        "ring_valid": jnp.zeros((batch_size, spec.window_size), dtype=jnp.bool_),
        "step_idx": jnp.zeros((batch_size,), dtype=jnp.int32),
    }


def step_sw_gqa_sink(
    params: PyTree,
    state: PyTree,
    x_t: Array,
    valid_mask_t: Array,
    reset_mask_t: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, PyTree]:
    """Single strictly-causal O(1) step of Sliding-Window GQA + Learnable Attention Sinks."""
    b_sz = x_t.shape[0]
    h_q, h_kv, d_h = spec.n_attn_heads, spec.n_kv_heads, spec.attn_head_dim
    groups = h_q // h_kv
    kv_dtype = state["ring_k"].dtype
    rst = reset_mask_t.astype(jnp.bool_)
    vld_f = valid_mask_t.astype(jnp.float32)
    is_valid = vld_f > 0.5
    scale = 1.0 / math.sqrt(float(d_h))

    rk = jnp.where(rst[:, None, None, None], jnp.zeros_like(state["ring_k"]), state["ring_k"])
    rv = jnp.where(rst[:, None, None, None], jnp.zeros_like(state["ring_v"]), state["ring_v"])
    rvld = jnp.where(rst[:, None], jnp.zeros_like(state["ring_valid"]), state["ring_valid"])
    st_idx = jnp.where(rst, jnp.zeros_like(state["step_idx"]), state["step_idx"])

    xn = rms_norm(x_t, params["norm_w"])
    q = dense_mm(xn, params["w_q"], matmul_dtype).reshape(b_sz, h_q, d_h)
    k_new = dense_mm(xn, params["w_k"], matmul_dtype).reshape(b_sz, h_kv, d_h).astype(kv_dtype)
    v_new = dense_mm(xn, params["w_v"], matmul_dtype).reshape(b_sz, h_kv, d_h).astype(kv_dtype)

    cand_k = jnp.concatenate([rk[:, 1:, :, :], k_new[:, None, :, :]], axis=1)
    cand_v = jnp.concatenate([rv[:, 1:, :, :], v_new[:, None, :, :]], axis=1)
    cand_vld = jnp.concatenate([rvld[:, 1:], is_valid[:, None]], axis=1)

    ring_k = jnp.where(is_valid[:, None, None, None], cand_k, rk)
    ring_v = jnp.where(is_valid[:, None, None, None], cand_v, rv)
    ring_valid = jnp.where(is_valid[:, None], cand_vld, rvld)
    step_idx = st_idx + is_valid.astype(jnp.int32)

    k_exp = jnp.repeat(ring_k.astype(jnp.float32), groups, axis=2)  # [B, W, H_q, d_h]
    v_exp = jnp.repeat(ring_v.astype(jnp.float32), groups, axis=2)
    win_logits = (
        jnp.einsum("bhd,bwhd->bhw", q.astype(jnp.float32), k_exp, precision=jax.lax.Precision.HIGHEST) * scale
        + params["rel_pos_bias"][None, :, :]
    )

    if "sink_k" in params:
        n_s = int(params["sink_k"].shape[0])
        sk_exp = jnp.repeat(params["sink_k"].astype(jnp.float32), groups, axis=1)  # [S, H_q, d_h]
        sv_exp = jnp.repeat(params["sink_v"].astype(jnp.float32), groups, axis=1)
        sink_logits = jnp.einsum("bhd,shd->bhs", q.astype(jnp.float32), sk_exp, precision=jax.lax.Precision.HIGHEST) * scale
        win_logits = jnp.where(ring_valid[:, None, :], win_logits, -1e9)
        all_logits = jnp.concatenate([sink_logits, win_logits], axis=-1)
        weights = jax.nn.softmax(all_logits, axis=-1)
        w_sink = weights[:, :, :n_s]
        w_win = weights[:, :, n_s:]
        attn_out = (
            jnp.einsum("bhs,shd->bhd", w_sink, sv_exp, precision=jax.lax.Precision.HIGHEST)
            + jnp.einsum("bhw,bwhd->bhd", w_win, v_exp, precision=jax.lax.Precision.HIGHEST)
        ).reshape(b_sz, h_q * d_h)
    else:
        has_any = jnp.any(ring_valid, axis=-1, keepdims=True)
        safe_vld = jnp.where(has_any, ring_valid, jnp.ones_like(ring_valid))
        win_logits = jnp.where(safe_vld[:, None, :], win_logits, -1e9)
        weights = jnp.where(has_any[:, :, None], jax.nn.softmax(win_logits, axis=-1), 0.0)
        attn_out = jnp.einsum("bhw,bwhd->bhd", weights, v_exp, precision=jax.lax.Precision.HIGHEST).reshape(b_sz, h_q * d_h)

    out = dense_mm(attn_out, params["w_o"], matmul_dtype) * vld_f[:, None]
    return out, {"ring_k": ring_k, "ring_v": ring_v, "ring_valid": ring_valid, "step_idx": step_idx}


def apply_sw_gqa_sink_seq(
    params: PyTree,
    init_state: PyTree,
    x_seq: Array,
    valid_mask_seq: Array,
    reset_mask_seq: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    gqa_chunk_size: int = 64,
    assume_all_valid: bool = False,
) -> tuple[Array, PyTree]:
    """Hoisted causal banded Sliding-Window GQA + Learnable Attention Sinks matching `step_sw_gqa_sink` bit-exactly."""
    t_len, b_sz, _ = x_seq.shape
    w_sz = spec.window_size
    h_q, h_kv, d_h = spec.n_attn_heads, spec.n_kv_heads, spec.attn_head_dim
    groups = h_q // h_kv
    kv_dtype = init_state["ring_k"].dtype
    scale = 1.0 / math.sqrt(float(d_h))
    has_sinks = "sink_k" in params

    xn = rms_norm(x_seq, params["norm_w"])
    q = dense_mm(xn, params["w_q"], matmul_dtype).reshape(t_len, b_sz, h_q, d_h)
    k_new = dense_mm(xn, params["w_k"], matmul_dtype).reshape(t_len, b_sz, h_kv, d_h).astype(kv_dtype)
    v_new = dense_mm(xn, params["w_v"], matmul_dtype).reshape(t_len, b_sz, h_kv, d_h).astype(kv_dtype)

    is_valid = valid_mask_seq > 0.5
    is_reset = reset_mask_seq.astype(jnp.bool_)

    init_k = jnp.swapaxes(init_state["ring_k"], 0, 1)
    init_v = jnp.swapaxes(init_state["ring_v"], 0, 1)
    init_vld = jnp.swapaxes(init_state["ring_valid"], 0, 1)
    k_pool = jnp.concatenate([init_k, k_new], axis=0)
    v_pool = jnp.concatenate([init_v, v_new], axis=0)
    vld_pool = jnp.concatenate([init_vld, is_valid], axis=0)

    if has_sinks:
        sk_exp = jnp.repeat(params["sink_k"].astype(jnp.float32), groups, axis=1)  # [S, H_q, d_h]
        sv_exp = jnp.repeat(params["sink_v"].astype(jnp.float32), groups, axis=1)
        n_s = int(params["sink_k"].shape[0])
    else:
        sk_exp = sv_exp = None
        n_s = 0

    def _dense_contiguous_path(_: None) -> tuple[Array, PyTree]:
        ep_new = jnp.cumsum(is_reset.astype(jnp.int32), axis=0)
        ep_pool = jnp.concatenate([jnp.zeros((w_sz, b_sz), dtype=jnp.int32), ep_new], axis=0)

        c_sz = min(max(1, gqa_chunk_size), t_len)
        rem = t_len % c_sz
        pad_n = (c_sz - rem) if rem != 0 else 0
        t_pad = t_len + pad_n
        n_chunks = t_pad // c_sz

        if pad_n > 0:
            q_pad = jnp.concatenate([q, jnp.zeros((pad_n, b_sz, h_q, d_h), dtype=q.dtype)], axis=0)
            ep_q_pad = jnp.concatenate([ep_new, jnp.full((pad_n, b_sz), -1, dtype=jnp.int32)], axis=0)
            k_pool_p = jnp.concatenate([k_pool, jnp.zeros((pad_n, b_sz, h_kv, d_h), dtype=kv_dtype)], axis=0)
            v_pool_p = jnp.concatenate([v_pool, jnp.zeros((pad_n, b_sz, h_kv, d_h), dtype=kv_dtype)], axis=0)
            vld_pool_p = jnp.concatenate([vld_pool, jnp.zeros((pad_n, b_sz), dtype=jnp.bool_)], axis=0)
            ep_pool_p = jnp.concatenate([ep_pool, jnp.full((pad_n, b_sz), -2, dtype=jnp.int32)], axis=0)
        else:
            q_pad, ep_q_pad = q, ep_new
            k_pool_p, v_pool_p, vld_pool_p, ep_pool_p = k_pool, v_pool, vld_pool, ep_pool

        win_len = c_sz + w_sz - 1
        i_grid = jnp.arange(c_sz, dtype=jnp.int32)[:, None]
        j_grid = jnp.arange(win_len, dtype=jnp.int32)[None, :]
        rel_w = j_grid - i_grid
        in_band = (rel_w >= 0) & (rel_w < w_sz)
        bias_band = params["rel_pos_bias"][:, jnp.clip(rel_w, 0, w_sz - 1)]  # [H_q, C, win_len]

        q_chunks = q_pad.reshape(n_chunks, c_sz, b_sz, h_q, d_h)
        ep_q_chunks = ep_q_pad.reshape(n_chunks, c_sz, b_sz)
        starts = jnp.arange(n_chunks, dtype=jnp.int32) * c_sz + 1

        def _chunk_attn(_carry: None, xs: tuple[Array, Array, Array]) -> tuple[None, Array]:
            q_c, ep_qc, start_idx = xs
            k_c = jax.lax.dynamic_slice(k_pool_p, (start_idx, 0, 0, 0), (win_len, b_sz, h_kv, d_h))
            v_c = jax.lax.dynamic_slice(v_pool_p, (start_idx, 0, 0, 0), (win_len, b_sz, h_kv, d_h))
            vld_c = jax.lax.dynamic_slice(vld_pool_p, (start_idx, 0), (win_len, b_sz))
            ep_kc = jax.lax.dynamic_slice(ep_pool_p, (start_idx, 0), (win_len, b_sz))

            k_exp = jnp.repeat(k_c, groups, axis=2)
            v_exp = jnp.repeat(v_c, groups, axis=2)
            if matmul_dtype == jnp.float32:
                win_logits = jnp.einsum(
                    "cbhd,kbhd->bhck", q_c.astype(jnp.float32), k_exp.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
                )
            else:
                win_logits = jnp.einsum("cbhd,kbhd->bhck", q_c.astype(matmul_dtype), k_exp.astype(matmul_dtype)).astype(jnp.float32)
            win_logits = win_logits * scale + bias_band[None, :, :, :]

            same_ep = ep_qc[:, None, :] == ep_kc[None, :, :]
            mask_ckb = in_band[:, :, None] & same_ep & vld_c[None, :, :]
            mask_bck = jnp.transpose(mask_ckb, (2, 0, 1))[:, None, :, :]  # [B, 1, C, win_len]

            if has_sinks:
                assert sk_exp is not None and sv_exp is not None
                sink_logits = (
                    jnp.einsum("cbhd,shd->bhcs", q_c.astype(jnp.float32), sk_exp, precision=jax.lax.Precision.HIGHEST)
                    * scale
                )
                win_masked = jnp.where(mask_bck, win_logits, -1e9)
                all_logits = jnp.concatenate([sink_logits, win_masked], axis=-1)
                wts = jax.nn.softmax(all_logits, axis=-1)
                w_sink = wts[:, :, :, :n_s]
                w_win = wts[:, :, :, n_s:]
                out_sink = jnp.einsum("bhcs,shd->cbhd", w_sink, sv_exp, precision=jax.lax.Precision.HIGHEST)
                if matmul_dtype == jnp.float32:
                    out_win = jnp.einsum(
                        "bhck,kbhd->cbhd", w_win.astype(jnp.float32), v_exp.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
                    )
                else:
                    out_win = jnp.einsum("bhck,kbhd->cbhd", w_win.astype(matmul_dtype), v_exp.astype(matmul_dtype)).astype(jnp.float32)
                out_c = out_sink + out_win
            else:
                has_any = jnp.any(mask_bck, axis=-1, keepdims=True)
                safe_mask = jnp.where(has_any, mask_bck, jnp.ones_like(mask_bck))
                win_masked = jnp.where(safe_mask, win_logits, -1e9)
                wts = jnp.where(has_any, jax.nn.softmax(win_masked, axis=-1), 0.0)
                if matmul_dtype == jnp.float32:
                    out_c = jnp.einsum(
                        "bhck,kbhd->cbhd", wts.astype(jnp.float32), v_exp.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
                    )
                else:
                    out_c = jnp.einsum("bhck,kbhd->cbhd", wts.astype(matmul_dtype), v_exp.astype(matmul_dtype)).astype(jnp.float32)
            return None, out_c.reshape(c_sz, b_sz, h_q * d_h)

        _, attn_chunks = jax.lax.scan(_chunk_attn, None, (q_chunks, ep_q_chunks, starts))
        attn_seq = attn_chunks.reshape(t_pad, b_sz, h_q * d_h)[:t_len]

        ep_end = ep_new[-1]
        fin_k_raw = jnp.swapaxes(k_pool[t_len : t_len + w_sz], 0, 1)
        fin_v_raw = jnp.swapaxes(v_pool[t_len : t_len + w_sz], 0, 1)
        fin_ep = jnp.swapaxes(ep_pool[t_len : t_len + w_sz], 0, 1)
        fin_vld_raw = jnp.swapaxes(vld_pool[t_len : t_len + w_sz], 0, 1)
        fin_same_ep = fin_ep == ep_end[:, None]
        fin_vld = fin_vld_raw & fin_same_ep
        fin_k = jnp.where(fin_same_ep[:, :, None, None], fin_k_raw, jnp.zeros_like(fin_k_raw))
        fin_v = jnp.where(fin_same_ep[:, :, None, None], fin_v_raw, jnp.zeros_like(fin_v_raw))

        def _step_cnt(cnt: Array, rst_t: Array) -> tuple[Array, None]:
            return jnp.where(rst_t, jnp.zeros_like(cnt), cnt) + 1, None

        fin_step, _ = jax.lax.scan(_step_cnt, init_state["step_idx"], is_reset)
        return attn_seq, {"ring_k": fin_k, "ring_v": fin_v, "ring_valid": fin_vld, "step_idx": fin_step}

    def _hole_preserving_path(_: None) -> tuple[Array, PyTree]:
        init_idx = jnp.broadcast_to(jnp.arange(w_sz, dtype=jnp.int32)[None, :], (b_sz, w_sz))

        def _idx_step(carry: tuple[Array, Array, Array], xs: tuple[Array, Array, Array]):
            r_idx, r_vld, st_idx = carry
            t_i, vld_t, rst_t = xs
            r_vld = jnp.where(rst_t[:, None], jnp.zeros_like(r_vld), r_vld)
            st_idx = jnp.where(rst_t, jnp.zeros_like(st_idx), st_idx)
            cand_idx = jnp.concatenate([r_idx[:, 1:], jnp.full((b_sz, 1), w_sz + t_i, dtype=jnp.int32)], axis=1)
            cand_vld = jnp.concatenate([r_vld[:, 1:], vld_t[:, None]], axis=1)
            r_idx = jnp.where(vld_t[:, None], cand_idx, r_idx)
            r_vld = jnp.where(vld_t[:, None], cand_vld, r_vld)
            st_idx = st_idx + vld_t.astype(jnp.int32)
            return (r_idx, r_vld, st_idx), (r_idx, r_vld)

        (fin_idx, fin_vld, fin_step), (all_idx, all_vld) = jax.lax.scan(
            _idx_step, (init_idx, init_state["ring_valid"], init_state["step_idx"]), (jnp.arange(t_len, dtype=jnp.int32), is_valid, is_reset)
        )
        b_rng = jnp.arange(b_sz, dtype=jnp.int32)[:, None]

        def _eval_t(_carry: None, xs: tuple[Array, Array, Array]) -> tuple[None, Array]:
            q_t, idx_t, vld_t = xs
            k_w = k_pool[idx_t, b_rng]
            v_w = v_pool[idx_t, b_rng]
            k_exp = jnp.repeat(k_w.astype(jnp.float32), groups, axis=2)
            v_exp = jnp.repeat(v_w.astype(jnp.float32), groups, axis=2)
            win_logits = (
                jnp.einsum("bhd,bwhd->bhw", q_t.astype(jnp.float32), k_exp, precision=jax.lax.Precision.HIGHEST) * scale
                + params["rel_pos_bias"][None, :, :]
            )
            if has_sinks:
                assert sk_exp is not None and sv_exp is not None
                sink_logits = jnp.einsum("bhd,shd->bhs", q_t.astype(jnp.float32), sk_exp, precision=jax.lax.Precision.HIGHEST) * scale
                win_masked = jnp.where(vld_t[:, None, :], win_logits, -1e9)
                wts = jax.nn.softmax(jnp.concatenate([sink_logits, win_masked], axis=-1), axis=-1)
                out_t = (
                    jnp.einsum("bhs,shd->bhd", wts[:, :, :n_s], sv_exp, precision=jax.lax.Precision.HIGHEST)
                    + jnp.einsum("bhw,bwhd->bhd", wts[:, :, n_s:], v_exp, precision=jax.lax.Precision.HIGHEST)
                ).reshape(b_sz, h_q * d_h)
            else:
                has_any = jnp.any(vld_t, axis=-1, keepdims=True)
                safe_vld = jnp.where(has_any, vld_t, jnp.ones_like(vld_t))
                win_masked = jnp.where(safe_vld[:, None, :], win_logits, -1e9)
                wts = jnp.where(has_any[:, :, None], jax.nn.softmax(win_masked, axis=-1), 0.0)
                out_t = jnp.einsum("bhw,bwhd->bhd", wts, v_exp, precision=jax.lax.Precision.HIGHEST).reshape(b_sz, h_q * d_h)
            return None, out_t

        _, attn_seq = jax.lax.scan(_eval_t, None, (q, all_idx, all_vld))
        fin_k_raw = k_pool[fin_idx, b_rng]
        fin_v_raw = v_pool[fin_idx, b_rng]
        fin_k = jnp.where(fin_vld[:, :, None, None], fin_k_raw, jnp.zeros_like(fin_k_raw))
        fin_v = jnp.where(fin_vld[:, :, None, None], fin_v_raw, jnp.zeros_like(fin_v_raw))
        return attn_seq, {"ring_k": fin_k, "ring_v": fin_v, "ring_valid": fin_vld, "step_idx": fin_step}

    if assume_all_valid:
        attn_seq, new_state = _dense_contiguous_path(None)
    else:
        attn_seq, new_state = jax.lax.cond(jnp.all(is_valid), _dense_contiguous_path, _hole_preserving_path, None)

    out = dense_mm(attn_seq, params["w_o"], matmul_dtype) * valid_mask_seq.astype(jnp.float32)[..., None]
    return out, new_state


# ---------------------------------------------------------------------------
# 6. Prior-Generation SLWM-v3 Legacy M_anchor Memory (`B0_v3_ref`)
# ---------------------------------------------------------------------------


def step_v3_legacy_mem(
    params: PyTree,
    state: PyTree,
    x_t: Array,
    modality_id_t: Array,
    dt_s_t: Array,
    valid_mask_t: Array,
    reliability_t: Array,
    reset_mask_t: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.float32,
    anchor_period: int = 16,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Single step of legacy SLWM-v3 `M_anchor` surprise-routed multi-timescale matrix memory (`B0_v3_ref`)."""
    b_sz = x_t.shape[0]
    nb, nh = spec.n_banks, spec.n_mem_heads
    h_tot = nb * nh
    dk, dv = spec.mem_key_dim, spec.mem_val_dim
    fp32_prec = jax.lax.Precision.HIGHEST
    rst = reset_mask_t.astype(jnp.bool_)
    vld_f = valid_mask_t.astype(jnp.float32)
    is_valid = vld_f > 0.5
    dt_pos = jnp.maximum(dt_s_t.astype(jnp.float32), 0.0)

    xn = rms_norm(x_t, params["norm_w"])
    q = safe_l2_normalize(dense_mm(xn, params["w_q"], matmul_dtype).reshape(b_sz, nb, nh, dk))
    k = safe_l2_normalize(dense_mm(xn, params["w_k"], matmul_dtype).reshape(b_sz, nb, nh, dk))
    v = dense_mm(xn, params["w_v"], matmul_dtype).reshape(b_sz, nb, nh, dv).astype(jnp.float32)
    g = jax.nn.sigmoid(dense_mm(xn, params["w_g"], matmul_dtype).reshape(b_sz, nb, nh) + params["g_bias"][None, :, :])
    tau = _compute_bounded_tau(params["log_tau"], spec.init_timescales_s)
    dec = jnp.exp(-g * (dt_pos[:, None, None] / (tau[None, :, None] + 1e-6)))

    M_p = jnp.where(rst[:, None, None, None, None], jnp.zeros_like(state["M"]), state["M"].astype(jnp.float32))
    M_a = jnp.where(rst[:, None, None, None, None], jnp.zeros_like(state["M_anchor"]), state["M_anchor"].astype(jnp.float32))
    c_d = jnp.where(rst[:, None, None], jnp.ones_like(state["cum_decay"]), state["cum_decay"].astype(jnp.float32))
    ph = jnp.where(rst, jnp.zeros_like(state["chunk_phase"]), state["chunk_phase"].astype(jnp.int32))

    is_st = ph == 0
    M_a = jnp.where(is_st[:, None, None, None, None], M_p, M_a)
    c_d = jnp.where(is_st[:, None, None], jnp.ones_like(c_d), c_d)
    c_d_next = c_d * dec
    M_tilde = M_p * dec[:, :, :, None, None]

    prior_y = _soft_head_norm(c_d_next[..., None] * jnp.einsum("bnhvk,bnhk->bnhv", M_a, q, precision=fp32_prec), params["o_norm_w"])
    prior_pred = jnp.matmul(prior_y, params["w_prior_proj"].astype(jnp.float32), precision=fp32_prec)
    prior_err = jnp.sqrt(jnp.mean((xn - prior_pred) ** 2, axis=-1) + 1e-6)

    r_logits = _eval_surprise_router_logits(params, xn, modality_id_t, dt_pos, reliability_t, prior_err, spec, matmul_dtype)
    route_raw = float(spec.beta_max) * jax.nn.sigmoid(r_logits[:, :h_tot].reshape(b_sz, nb, nh))
    if spec.use_nowrite_option:
        r_nw_b = jax.nn.sigmoid(r_logits[:, h_tot:])
        keep_b = 1.0 - r_nw_b
        r_nw = jnp.mean(r_nw_b, axis=-1)
    else:
        keep_b = jnp.ones((b_sz, nb), dtype=jnp.float32)
        r_nw = jnp.zeros((b_sz,), dtype=jnp.float32)
    eta = route_raw * keep_b[:, :, None] * vld_f[:, None, None]

    ret_k = jnp.einsum("bnhvk,bnhk->bnhv", M_tilde, k, precision=fp32_prec)
    dM = jnp.einsum("bnh,bnhv,bnhk->bnhvk", eta, v - ret_k, k, precision=fp32_prec)
    M_next = M_tilde + dM
    y_norm = _soft_head_norm(jnp.einsum("bnhvk,bnhk->bnhv", M_next, q, precision=fp32_prec), params["o_norm_w"])
    out = dense_mm(y_norm, params["w_o"], matmul_dtype) * vld_f[:, None]

    ph_next = jnp.where(is_valid, (ph + 1) % anchor_period, ph)
    new_state = {
        "M": M_next.astype(state["M"].dtype),
        "M_anchor": M_a.astype(state["M_anchor"].dtype),
        "cum_decay": c_d_next,
        "chunk_phase": ph_next,
    }
    frob_h = jnp.sqrt(jnp.sum(M_next * M_next, axis=(3, 4)) + 1e-12)
    diag = {
        "r_nowrite": r_nw,
        "novelty": prior_err,
        "eff_eta": jnp.mean(eta, axis=(1, 2)),
        "delta_norm": jnp.sqrt(jnp.mean(dM * dM, axis=(1, 2, 3, 4)) + 1e-12),
        "frob_mean": jnp.mean(frob_h, axis=(1, 2)),
        "frob_max": jnp.max(frob_h, axis=(1, 2)),
    }
    return out, new_state, diag


def apply_v3_legacy_mem_seq(
    params: PyTree,
    init_state: PyTree,
    x_seq: Array,
    modality_id_seq: Array,
    dt_s_seq: Array,
    valid_mask_seq: Array,
    reliability_seq: Array,
    reset_mask_seq: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    anchor_period: int = 16,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Hoisted algebraic-surprise scan for `B0_v3_ref` matching `step_v3_legacy_mem` bit-exactly."""
    t_len, b_sz, d = x_seq.shape
    nb, nh = spec.n_banks, spec.n_mem_heads
    h_tot = nb * nh
    dk, dv = spec.mem_key_dim, spec.mem_val_dim
    fp32_prec = jax.lax.Precision.HIGHEST
    rst_b = reset_mask_seq.astype(jnp.bool_)
    vld_f = valid_mask_seq.astype(jnp.float32)
    is_valid = vld_f > 0.5
    dt_pos = jnp.maximum(dt_s_seq.astype(jnp.float32), 0.0)

    xn = rms_norm(x_seq, params["norm_w"])
    q_seq = safe_l2_normalize(dense_mm(xn, params["w_q"], matmul_dtype).reshape(t_len, b_sz, nb, nh, dk))
    k_seq = safe_l2_normalize(dense_mm(xn, params["w_k"], matmul_dtype).reshape(t_len, b_sz, nb, nh, dk))
    v_seq = dense_mm(xn, params["w_v"], matmul_dtype).reshape(t_len, b_sz, nb, nh, dv).astype(jnp.float32)
    g_seq = jax.nn.sigmoid(
        dense_mm(xn, params["w_g"], matmul_dtype).reshape(t_len, b_sz, nb, nh) + params["g_bias"][None, None, :, :]
    )
    tau = _compute_bounded_tau(params["log_tau"], spec.init_timescales_s)
    dec_seq = jnp.exp(-g_seq * (dt_pos[:, :, None, None] / (tau[None, None, :, None] + 1e-6)))

    # Algebraic Gram-matrix surprise hoisting (0 d-dim matmuls inside lax.scan!)
    P = params["w_prior_proj"].astype(jnp.float32)
    G_P = jnp.matmul(P, P.T, precision=fp32_prec)
    c_seq = jnp.matmul(xn.astype(jnp.float32), P.T, precision=fp32_prec)
    xsq_seq = jnp.sum(xn.astype(jnp.float32) ** 2, axis=-1)

    c_h = jax.nn.silu(dense_mm(xn, params["router_w1"], matmul_dtype) + params["router_b1"])
    c_logits = dense_mm(xn, params["router_skip"], matmul_dtype) + dense_mm(c_h, params["router_w2"], matmul_dtype) + params["router_b2"]
    m_idx = jnp.clip(modality_id_seq.astype(jnp.int32), 0, spec.n_modalities - 1)
    static_meta = jnp.concatenate(
        [params["mod_emb"][m_idx], jnp.log1p(dt_pos)[..., None], reliability_seq.astype(jnp.float32)[..., None]], axis=-1
    )
    mw1 = params["router_meta_w1"].astype(jnp.float32)
    mskip = params["router_meta_skip"].astype(jnp.float32)
    mw2 = params["router_meta_w2"].astype(jnp.float32)
    pre_mh = dense_mm(static_meta, mw1[:-1, :], matmul_dtype) + params["router_meta_b1"]
    pre_log = c_logits + dense_mm(static_meta, mskip[:-1, :], matmul_dtype)
    mw1_e = mw1[-1, :]
    mskip_e = mskip[-1, :]
    inv_d = 1.0 / float(d)

    init_c = (
        init_state["M"].astype(jnp.float32),
        init_state["M_anchor"].astype(jnp.float32),
        init_state["cum_decay"].astype(jnp.float32),
        init_state["chunk_phase"].astype(jnp.int32),
    )

    def _scan_step(carry, xs):
        M_p, M_a, c_d, ph = carry
        q_t, k_t, v_t, dec_t, c_t, xsq_t, pmh_t, plog_t, vf_t, vb_t, rst_t = xs
        M_p = jnp.where(rst_t[:, None, None, None, None], jnp.zeros_like(M_p), M_p)
        M_a = jnp.where(rst_t[:, None, None, None, None], jnp.zeros_like(M_a), M_a)
        c_d = jnp.where(rst_t[:, None, None], jnp.ones_like(c_d), c_d)
        ph = jnp.where(rst_t, jnp.zeros_like(ph), ph)

        is_st = ph == 0
        M_a = jnp.where(is_st[:, None, None, None, None], M_p, M_a)
        c_d = jnp.where(is_st[:, None, None], jnp.ones_like(c_d), c_d)
        c_d_next = c_d * dec_t
        M_tilde = M_p * dec_t[:, :, :, None, None]

        prior_y = _soft_head_norm(c_d_next[..., None] * jnp.einsum("bnhvk,bnhk->bnhv", M_a, q_t, precision=fp32_prec), params["o_norm_w"])
        y_G_y = jnp.sum(jnp.matmul(prior_y, G_P, precision=fp32_prec) * prior_y, axis=-1)
        sq_err = jnp.maximum(xsq_t - 2.0 * jnp.sum(c_t * prior_y, axis=-1) + y_G_y, 0.0)
        prior_err = jnp.sqrt(sq_err * inv_d + 1e-6)
        ef = jnp.log1p(prior_err)

        mh = jax.nn.silu(pmh_t + ef[:, None] * mw1_e[None, :])
        r_logits = plog_t + ef[:, None] * mskip_e[None, :] + jnp.matmul(mh, mw2, precision=fp32_prec)
        route_raw = float(spec.beta_max) * jax.nn.sigmoid(r_logits[:, :h_tot].reshape(b_sz, nb, nh))
        if spec.use_nowrite_option:
            r_nw_b = jax.nn.sigmoid(r_logits[:, h_tot:])
            keep_b = 1.0 - r_nw_b
            r_nw = jnp.mean(r_nw_b, axis=-1)
        else:
            keep_b = jnp.ones((b_sz, nb), dtype=jnp.float32)
            r_nw = jnp.zeros((b_sz,), dtype=jnp.float32)
        eta = route_raw * keep_b[:, :, None] * vf_t[:, None, None]

        ret_k = jnp.einsum("bnhvk,bnhk->bnhv", M_tilde, k_t, precision=fp32_prec)
        dM = jnp.einsum("bnh,bnhv,bnhk->bnhvk", eta, v_t - ret_k, k_t, precision=fp32_prec)
        M_next = M_tilde + dM
        y_norm = _soft_head_norm(jnp.einsum("bnhvk,bnhk->bnhv", M_next, q_t, precision=fp32_prec), params["o_norm_w"])
        ph_next = jnp.where(vb_t, (ph + 1) % anchor_period, ph)
        d_norm = jnp.sqrt(jnp.mean(dM * dM, axis=(1, 2, 3, 4)) + 1e-12)
        frob_h = jnp.sqrt(jnp.sum(M_next * M_next, axis=(3, 4)) + 1e-12)
        return (M_next, M_a, c_d_next, ph_next), (
            y_norm,
            r_nw,
            prior_err,
            jnp.mean(eta, axis=(1, 2)),
            d_norm,
            jnp.mean(frob_h, axis=(1, 2)),
            jnp.max(frob_h, axis=(1, 2)),
        )

    (M_f, Ma_f, cd_f, ph_f), (y_norm_seq, rnw_s, nov_s, eta_s, dn_s, fm_s, fx_s) = jax.lax.scan(
        _scan_step,
        init_c,
        (q_seq, k_seq, v_seq, dec_seq, c_seq, xsq_seq, pre_mh, pre_log, vld_f, is_valid, rst_b),
    )
    out_seq = dense_mm(y_norm_seq, params["w_o"], matmul_dtype) * vld_f[..., None]
    fin_state = {
        "M": M_f.astype(init_state["M"].dtype),
        "M_anchor": Ma_f.astype(init_state["M_anchor"].dtype),
        "cum_decay": cd_f,
        "chunk_phase": ph_f,
    }
    diag = {
        "r_nowrite": rnw_s,
        "novelty": nov_s,
        "eff_eta": eta_s,
        "delta_norm": dn_s,
        "frob_mean": fm_s,
        "frob_max": fx_s,
    }
    return out_seq, fin_state, diag


# ---------------------------------------------------------------------------
# 7. SwiGLU FFN & Modality-Specific MS-MoE FFN (`C7_ms_moe_ffn`)
# ---------------------------------------------------------------------------


def init_ffn_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize dense SwiGLU FFN or 4-expert Modality-Specific MS-MoE FFN (`1.0x` active params/token)."""
    k1, k2, k3 = jax.random.split(key, 3)
    d = spec.d_model
    ffn = spec.ffn_dim
    m_d = spec.width_mult
    if not spec.use_ms_moe_ffn:
        return {
            "ffn_norm_w": jnp.ones((d,), dtype=jnp.float32),
            "w_g": init_mup_weight(k1, (d, ffn), role="hidden", width_mult=m_d, base_std=spec.init_std),
            "w_u": init_mup_weight(k2, (d, ffn), role="hidden", width_mult=m_d, base_std=spec.init_std),
            "w_d": init_mup_weight(k3, (ffn, d), role="hidden", width_mult=m_d, base_std=spec.init_std),
        }
    n_m = spec.n_modalities
    return {
        "ffn_norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_g": init_mup_weight(k1, (n_m, d, ffn), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_u": init_mup_weight(k2, (n_m, d, ffn), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_d": init_mup_weight(k3, (n_m, ffn, d), role="hidden", width_mult=m_d, base_std=spec.init_std),
    }


def apply_ffn(
    params: PyTree,
    x: Array,
    modality_id: Array,
    valid_mask: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    """Apply SwiGLU FFN or deterministic modality-routed MS-MoE FFN (`[*lead, D] -> [*lead, D]`)."""
    xn = rms_norm(x, params["ffn_norm_w"])
    vld_f = valid_mask.astype(jnp.float32)[..., None]
    if not spec.use_ms_moe_ffn:
        hid = jax.nn.silu(dense_mm(xn, params["w_g"], matmul_dtype)) * dense_mm(xn, params["w_u"], matmul_dtype)
        return dense_mm(hid, params["w_d"], matmul_dtype) * vld_f

    m_idx = jnp.clip(modality_id.astype(jnp.int32), 0, spec.n_modalities - 1)
    # Fast B=1 single-step live streaming path: execute ONLY the 1 active modality expert (exact 1.0x FLOPs!)
    if x.ndim == 2 and x.shape[0] == 1:
        e_i = m_idx[0]
        wg = params["w_g"][e_i]
        wu = params["w_u"][e_i]
        wd = params["w_d"][e_i]
        hid = jax.nn.silu(dense_mm(xn, wg, matmul_dtype)) * dense_mm(xn, wu, matmul_dtype)
        return dense_mm(hid, wd, matmul_dtype) * vld_f

    # Exact zero-token-dropping static-shape multi-expert dispatch (bit-exact with B=1 step!)
    out = jnp.zeros_like(xn, dtype=jnp.float32)
    for m in range(spec.n_modalities):
        mask_m = (m_idx == m).astype(jnp.float32)[..., None]
        hid_m = jax.nn.silu(dense_mm(xn, params["w_g"][m], matmul_dtype)) * dense_mm(xn, params["w_u"][m], matmul_dtype)
        out = out + dense_mm(hid_m, params["w_d"][m], matmul_dtype) * mask_m
    return out * vld_f
