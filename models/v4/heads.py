"""SLWM-v4 Shared Generative, Predictive, Synchrony & Frozen-Core Policy Heads.

Every head operates identically on top of the shared latent world field `Z_seq: float32[T, B, D]`
produced by ANY compared SLWM-v4 backbone (`A0..A2`, `B0..B1`, `C1..C7`), ensuring exact Dual-Track
Core Comparison parity (`AGENTS.md §4.1`):
  1. `head_video`:
       - `"flow"` (default): Rectified Flow Matching velocity MLP with AdaLN-Zero / FiLM timestep +
         spatial-patch conditioning (`tau ~ logit_normal(0, 1)`, straight-line path `x_tau = (1-tau)x_0 + tau x_1`,
         exact velocity MSE loss; Euler/Midpoint `4-6` NFE live ODE sampler).
       - `"huber"` (`H1_huber_video_abl`): matched residual MLP with direct deterministic Huber prediction.
  2. `head_audio`:
       - `"depth_transformer"` (default): Moshi-style strictly causal intra-tick 2-layer Depth Transformer
         across all `K = 16` Kyutai Mimi RVQ levels (`k = 0..15` within the SAME tick `t+1`, zero future-tick
         lookahead, `2.0x` semantic `RVQ-0` weighting, chunked `@jax.checkpoint` loss + AR sampler).
       - `"flat_rvq"` (`H2_flat_rvq_abl`): conditionally independent flat 16-RVQ softmax head.
  3. `head_motor`:
       - Unified non-aliased `52`-DoF Action + `68`-Dim Proprioception/Force + optional `episode_success` BCE.
       - Strictly per-channel masked by binary `action_mask [..., 52]` and `proprio_mask [..., 68]` (`0.0` grad on
         inactive DoFs!) and `episode_success >= 0` (`0.0` grad on unlabeled `-1` episodes!).
  4. `head_text`:
       - Lossless `264`-vocab UTF-8 next-byte cross-entropy -> exact BPB.
  5. `head_jepa` (`O1_jepa_sync`, optional):
       - Strictly causal predictive AV-JEPA InfoNCE contrasting `z_seq[t]` at the final video patch of heartbeat `h`
         (`sp == 14`) against a dedicated local embedding of `audio_codes[t+1]` (`AUD[2h+1]`, same `heartbeat_id`,
         zero residual identity shortcut, zero future-heartbeat lookahead).
  6. `policy_gate` (disabled ONLY in `H4_no_policy_gate`):
       - Strictly wraps `Z_t` in `jax.lax.stop_gradient(Z_t)` (exact `0.0` gradient into world core), combining
         width-invariant epistemic uncertainty features with learnable temperature scaling over the 5 committed
         actions `(SPEAK_TEXT, EMIT_AUDIO, GENERATE_VIDEO, ACT_MOTOR, ABSTAIN_SILENCE)`.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import optax

from data.v4_heartbeat_contract import MOD_AUDIO, MOD_MOTOR, MOD_TEXT, MOD_VIDEO
from models.v4.blocks import dense_mm, init_mup_weight, rms_norm
from models.v4.spec import SLWMv4Spec

Array = jax.Array
PyTree = dict[str, Any]

COMMIT_ACTIONS = (
    "SPEAK_TEXT",
    "EMIT_AUDIO",
    "GENERATE_VIDEO",
    "ACT_MOTOR",
    "ABSTAIN_SILENCE",
)


# ---------------------------------------------------------------------------
# 1. Video Head: Rectified Flow Matching (`flow`) vs Deterministic Huber (`huber`)
# ---------------------------------------------------------------------------


def _sinusoidal_time_emb(tau: Array, dim: int = 64) -> Array:
    """Sinusoidal embedding for continuous flow time `tau in [0, 1]` -> `[..., dim]`."""
    half = dim // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1))
    ang = tau.astype(jnp.float32)[..., None] * (2.0 * math.pi) * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def init_video_head_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize Rectified Flow Matching (`flow`) or matched Huber (`huber`) video patch head."""
    keys = jax.random.split(key, 12 + 5 * spec.video_flow_layers)
    d = spec.d_model
    fd = spec.video_flow_dim
    pd = spec.video_patch_dim
    m_d = spec.width_mult
    ff_h = fd * 2

    blocks = []
    ki = 6
    for _ in range(spec.video_flow_layers):
        blk: PyTree = {
            "norm_w": jnp.ones((fd,), dtype=jnp.float32),
            "w_g": init_mup_weight(keys[ki], (fd, ff_h), role="fixed_hidden", width_mult=1.0, base_std=0.5 / math.sqrt(float(fd))),
            "w_u": init_mup_weight(keys[ki + 1], (fd, ff_h), role="fixed_hidden", width_mult=1.0, base_std=0.5 / math.sqrt(float(fd))),
            "w_d": init_mup_weight(keys[ki + 2], (ff_h, fd), role="fixed_hidden", width_mult=1.0, zero_init=True),
        }
        if spec.video_head_type == "flow":
            # AdaLN-Zero modulation (scale, shift, residual gate) from `(cond + t_emb)`
            blk["ada_w"] = init_mup_weight(keys[ki + 3], (fd, 3 * fd), role="fixed_hidden", width_mult=1.0, zero_init=True)
            blk["ada_bias"] = jnp.zeros((3 * fd,), dtype=jnp.float32)
        blocks.append(blk)
        ki += 5

    params: PyTree = {
        "z_norm_w": jnp.ones((d,), dtype=jnp.float32),
        "cond_proj": init_mup_weight(keys[0], (d, fd), role="readout", width_mult=m_d, base_std=0.5 / math.sqrt(float(spec.d_base))),
        "spatial_pos_emb": init_mup_weight(keys[1], (spec.video_spatial_packets, fd), role="input", width_mult=1.0, base_std=0.05),
        "blocks": blocks,
        "out_norm_w": jnp.ones((fd,), dtype=jnp.float32),
        "w_out": init_mup_weight(keys[2], (fd, pd), role="fixed_hidden", width_mult=1.0, zero_init=True),
        "out_bias": jnp.zeros((pd,), dtype=jnp.float32),
    }
    if spec.video_head_type == "flow":
        params["x_in_w"] = init_mup_weight(keys[3], (pd, fd), role="input", width_mult=1.0, base_std=0.5 / math.sqrt(float(pd)))
        params["x_in_bias"] = jnp.zeros((fd,), dtype=jnp.float32)
        params["t_w1"] = init_mup_weight(keys[4], (64, fd), role="input", width_mult=1.0, base_std=0.5 / math.sqrt(64.0))
        params["t_b1"] = jnp.zeros((fd,), dtype=jnp.float32)
        params["t_w2"] = init_mup_weight(keys[5], (fd, fd), role="fixed_hidden", width_mult=1.0, base_std=0.5 / math.sqrt(float(fd)))
        params["t_b2"] = jnp.zeros((fd,), dtype=jnp.float32)
    return params


def apply_video_flow_velocity(
    head_params: PyTree,
    z_cond: Array,
    sp_pos: Array,
    x_tau: Array,
    tau: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    """Predict Rectified Flow velocity `v_pred: [*lead, 1024]` at noisy patch `x_tau` and time `tau`."""
    zn = rms_norm(z_cond, head_params["z_norm_w"])
    sp_idx = jnp.clip(sp_pos.astype(jnp.int32), 0, spec.video_spatial_packets - 1)
    cond = dense_mm(zn, head_params["cond_proj"], matmul_dtype) + head_params["spatial_pos_emb"][sp_idx]

    t_feat = _sinusoidal_time_emb(tau, dim=64)
    t_h = jax.nn.silu(dense_mm(t_feat, head_params["t_w1"], matmul_dtype) + head_params["t_b1"])
    t_emb = dense_mm(t_h, head_params["t_w2"], matmul_dtype) + head_params["t_b2"]
    c_mod = jax.nn.silu(cond + t_emb)

    h = dense_mm(x_tau.astype(jnp.float32), head_params["x_in_w"], matmul_dtype) + head_params["x_in_bias"] + cond
    fd = int(head_params["out_norm_w"].shape[0])
    for blk in head_params["blocks"]:
        ada = dense_mm(c_mod, blk["ada_w"], matmul_dtype) + blk["ada_bias"]
        shift, scale, gate = ada[..., :fd], ada[..., fd : 2 * fd], ada[..., 2 * fd :]
        hn = rms_norm(h, blk["norm_w"]) * (1.0 + scale) + shift
        ff = jax.nn.silu(dense_mm(hn, blk["w_g"], matmul_dtype)) * dense_mm(hn, blk["w_u"], matmul_dtype)
        h = h + (1.0 + gate) * dense_mm(ff, blk["w_d"], matmul_dtype)

    h_out = rms_norm(h, head_params["out_norm_w"])
    return dense_mm(h_out, head_params["w_out"], matmul_dtype) + head_params["out_bias"]


def apply_video_huber_pred(
    head_params: PyTree,
    z_cond: Array,
    sp_pos: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    """Direct deterministic video patch prediction (`H1_huber_video_abl`) -> `[*lead, 1024]`."""
    zn = rms_norm(z_cond, head_params["z_norm_w"])
    sp_idx = jnp.clip(sp_pos.astype(jnp.int32), 0, spec.video_spatial_packets - 1)
    h = dense_mm(zn, head_params["cond_proj"], matmul_dtype) + head_params["spatial_pos_emb"][sp_idx]
    for blk in head_params["blocks"]:
        hn = rms_norm(h, blk["norm_w"])
        ff = jax.nn.silu(dense_mm(hn, blk["w_g"], matmul_dtype)) * dense_mm(hn, blk["w_u"], matmul_dtype)
        h = h + dense_mm(ff, blk["w_d"], matmul_dtype)
    h_out = rms_norm(h, head_params["out_norm_w"])
    return dense_mm(h_out, head_params["w_out"], matmul_dtype) + head_params["out_bias"]


def sample_video_patch(
    head_params: PyTree,
    z_t: Array,
    sp_pos: Array,
    rng_key: Array,
    spec: SLWMv4Spec,
    *,
    nfe: int | None = None,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, Array]:
    """Generate one video patch `[*lead, 1024]` using EXACTLY `nfe` velocity function evaluations.

    - Even `nfe >= 2` (e.g. `4` or `6`): executes `nfe // 2` 2nd-order Midpoint RK2 steps (`2` evals/step = EXACT `nfe` NFEs).
    - Odd `nfe` (e.g. `1`, `3`, `5`): executes `nfe` 1st-order Euler steps (`1` eval/step = EXACT `nfe` NFEs).
    Returns `(patch_pred, mean_vel_norm)`.
    """
    if spec.video_head_type == "huber":
        pred = apply_video_huber_pred(head_params, z_t, sp_pos, spec, matmul_dtype=matmul_dtype)
        return pred, jnp.sqrt(jnp.mean(pred * pred, axis=-1) + 1e-6)

    nfe_eff = max(1, int(spec.video_flow_nfe if nfe is None else nfe))
    lead_shape = z_t.shape[:-1]
    x = jax.random.normal(rng_key, lead_shape + (spec.video_patch_dim,), dtype=jnp.float32)

    if nfe_eff >= 2 and (nfe_eff % 2 == 0):
        rk2_steps = nfe_eff // 2
        dt = 1.0 / float(rk2_steps)

        def _rk2_step(carry: tuple[Array, Array], step_i: Array) -> tuple[tuple[Array, Array], None]:
            x_cur, vn_acc = carry
            tau_i = jnp.full(lead_shape, step_i.astype(jnp.float32) * dt, dtype=jnp.float32)
            v1 = apply_video_flow_velocity(head_params, z_t, sp_pos, x_cur, tau_i, spec, matmul_dtype=matmul_dtype)
            x_mid = x_cur + 0.5 * dt * v1
            tau_mid = jnp.minimum(tau_i + 0.5 * dt, 1.0)
            v_mid = apply_video_flow_velocity(head_params, z_t, sp_pos, x_mid, tau_mid, spec, matmul_dtype=matmul_dtype)
            x_nxt = x_cur + dt * v_mid
            vn = jnp.sqrt(jnp.mean(v_mid * v_mid, axis=-1) + 1e-6)
            return (x_nxt, vn_acc + vn * dt), None

        (x_fin, vel_norm), _ = jax.lax.scan(
            _rk2_step, (x, jnp.zeros(lead_shape, dtype=jnp.float32)), jnp.arange(rk2_steps, dtype=jnp.int32)
        )
        return x_fin, vel_norm

    dt = 1.0 / float(nfe_eff)

    def _euler_step(carry: tuple[Array, Array], step_i: Array) -> tuple[tuple[Array, Array], None]:
        x_cur, vn_acc = carry
        tau_i = jnp.full(lead_shape, step_i.astype(jnp.float32) * dt, dtype=jnp.float32)
        v1 = apply_video_flow_velocity(head_params, z_t, sp_pos, x_cur, tau_i, spec, matmul_dtype=matmul_dtype)
        x_nxt = x_cur + dt * v1
        vn = jnp.sqrt(jnp.mean(v1 * v1, axis=-1) + 1e-6)
        return (x_nxt, vn_acc + vn * dt), None

    (x_fin, vel_norm), _ = jax.lax.scan(
        _euler_step, (x, jnp.zeros(lead_shape, dtype=jnp.float32)), jnp.arange(nfe_eff, dtype=jnp.int32)
    )
    return x_fin, vel_norm


# ---------------------------------------------------------------------------
# 2. Audio Head: Moshi Causal Depth Transformer (`depth_transformer`) vs `flat_rvq`
# ---------------------------------------------------------------------------


def init_audio_head_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize 16-RVQ Moshi Depth Transformer (`depth_transformer`) or `flat_rvq` (`H2`) head."""
    k_rvq = spec.audio_codebooks
    vocab = spec.audio_vocab_size
    d = spec.d_model
    dd = spec.audio_depth_dim
    m_d = spec.width_mult
    keys = jax.random.split(key, 10 + 8 * spec.audio_depth_layers)

    params: PyTree = {
        "z_norm_w": jnp.ones((d,), dtype=jnp.float32),
        "in_proj": init_mup_weight(keys[0], (d, dd), role="readout", width_mult=m_d, base_std=0.5 / math.sqrt(float(spec.d_base))),
        "out_norm_w": jnp.ones((dd,), dtype=jnp.float32),
        "w_out": init_mup_weight(keys[1], (k_rvq, dd, vocab), role="fixed_hidden", width_mult=1.0, zero_init=True),
        "out_bias": jnp.zeros((k_rvq, vocab), dtype=jnp.float32),
    }

    if spec.audio_head_type == "depth_transformer":
        params["level_emb"] = init_mup_weight(keys[2], (k_rvq, dd), role="input", width_mult=1.0, base_std=0.05)
        params["rvq_in_emb"] = init_mup_weight(keys[3], (max(1, k_rvq - 1), vocab, dd), role="input", width_mult=1.0, base_std=0.05)
        layers = []
        ki = 4
        ff_h = dd * 2
        for _ in range(spec.audio_depth_layers):
            layers.append(
                {
                    "attn_norm_w": jnp.ones((dd,), dtype=jnp.float32),
                    "w_qkv": init_mup_weight(keys[ki], (dd, 3 * dd), role="fixed_hidden", width_mult=1.0, base_std=0.5 / math.sqrt(float(dd))),
                    "w_o": init_mup_weight(keys[ki + 1], (dd, dd), role="fixed_hidden", width_mult=1.0, base_std=0.25 / math.sqrt(float(dd))),
                    "ffn_norm_w": jnp.ones((dd,), dtype=jnp.float32),
                    "w_g": init_mup_weight(keys[ki + 2], (dd, ff_h), role="fixed_hidden", width_mult=1.0, base_std=0.5 / math.sqrt(float(dd))),
                    "w_u": init_mup_weight(keys[ki + 3], (dd, ff_h), role="fixed_hidden", width_mult=1.0, base_std=0.5 / math.sqrt(float(dd))),
                    "w_d": init_mup_weight(keys[ki + 4], (ff_h, dd), role="fixed_hidden", width_mult=1.0, zero_init=True),
                }
            )
            ki += 6
        params["layers"] = layers
    return params


def _apply_depth_transformer_hidden(
    head_params: PyTree,
    h_in: Array,
    spec: SLWMv4Spec,
    matmul_dtype: jnp.dtype,
) -> Array:
    """Strictly causal Transformer over `K = 16` intra-tick RVQ levels: `[N, K, dd] -> [N, K, dd]`."""
    n_tok, k_rvq, dd = h_in.shape
    nh = spec.audio_depth_heads
    dh = dd // nh
    scale = 1.0 / math.sqrt(float(dh))
    causal_mask = jnp.tril(jnp.ones((k_rvq, k_rvq), dtype=jnp.bool_))  # [K, K]

    h = h_in
    for lyr in head_params["layers"]:
        hn = rms_norm(h, lyr["attn_norm_w"])
        qkv = dense_mm(hn, lyr["w_qkv"], matmul_dtype).reshape(n_tok, k_rvq, 3, nh, dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [N, K, H, dh]
        logits = jnp.einsum("nkhd,njhd->nhkj", q.astype(jnp.float32), k.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST) * scale
        logits = jnp.where(causal_mask[None, None, :, :], logits, -1e9)
        wts = jax.nn.softmax(logits, axis=-1)
        attn = jnp.einsum("nhkj,njhd->nkhd", wts, v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST).reshape(n_tok, k_rvq, dd)
        h = h + dense_mm(attn, lyr["w_o"], matmul_dtype)

        fn = rms_norm(h, lyr["ffn_norm_w"])
        ff = jax.nn.silu(dense_mm(fn, lyr["w_g"], matmul_dtype)) * dense_mm(fn, lyr["w_u"], matmul_dtype)
        h = h + dense_mm(ff, lyr["w_d"], matmul_dtype)

    return rms_norm(h, head_params["out_norm_w"])  # [N, K, dd]


def _apply_depth_transformer_core(
    head_params: PyTree,
    h_in: Array,
    spec: SLWMv4Spec,
    matmul_dtype: jnp.dtype,
) -> Array:
    """Strictly causal Transformer over `K = 16` intra-tick RVQ levels: `[N, K, dd] -> [N, K, vocab]`."""
    h_out = _apply_depth_transformer_hidden(head_params, h_in, spec, matmul_dtype)
    if matmul_dtype == jnp.float32:
        logits = jnp.einsum("nkd,kdv->nkv", h_out, head_params["w_out"].astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    else:
        logits = jnp.einsum("nkd,kdv->nkv", h_out.astype(matmul_dtype), head_params["w_out"].astype(matmul_dtype)).astype(jnp.float32)
    return logits + head_params["out_bias"][None, :, :]


def apply_audio_head_teacher_forced(
    head_params: PyTree,
    z_cond: Array,
    target_codes: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    """Compute strictly intra-tick causal logits `[..., K_rvq, vocab]` for `target_codes [..., K_rvq]`."""
    lead = z_cond.shape[:-1]
    n_tok = 1
    for s in lead:
        n_tok *= int(s)
    k_rvq = spec.audio_codebooks
    dd = spec.audio_depth_dim

    z_flat = z_cond.reshape(n_tok, spec.d_model)
    z_proj = dense_mm(rms_norm(z_flat, head_params["z_norm_w"]), head_params["in_proj"], matmul_dtype)  # [N, dd]

    if spec.audio_head_type == "flat_rvq":
        h_out = rms_norm(z_proj, head_params["out_norm_w"])
        if matmul_dtype == jnp.float32:
            logits = jnp.einsum("nd,kdv->nkv", h_out, head_params["w_out"].astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
        else:
            logits = jnp.einsum("nd,kdv->nkv", h_out.astype(matmul_dtype), head_params["w_out"].astype(matmul_dtype)).astype(jnp.float32)
        logits = logits + head_params["out_bias"][None, :, :]
        return logits.reshape(*lead, k_rvq, spec.audio_vocab_size)

    # Depth Transformer: level 0 receives `z_proj + level_emb[0]`; level k >= 1 receives `z_proj + rvq_in_emb[k-1, code_{k-1}] + level_emb[k]`
    codes_flat = jnp.clip(target_codes.reshape(n_tok, k_rvq).astype(jnp.int32), 0, spec.audio_vocab_size - 1)
    if k_rvq > 1:
        prev_idx = jnp.arange(k_rvq - 1, dtype=jnp.int32)[None, :]
        prev_emb = head_params["rvq_in_emb"][prev_idx, codes_flat[:, :-1]]  # [N, K-1, dd]
        tok_emb = jnp.concatenate([jnp.zeros((n_tok, 1, dd), dtype=jnp.float32), prev_emb], axis=1)
    else:
        tok_emb = jnp.zeros((n_tok, 1, dd), dtype=jnp.float32)

    h_in = z_proj[:, None, :] + tok_emb + head_params["level_emb"][None, :, :]
    logits = _apply_depth_transformer_core(head_params, h_in, spec, matmul_dtype)
    return logits.reshape(*lead, k_rvq, spec.audio_vocab_size)


def compute_audio_rvq0_entropy(
    head_params: PyTree,
    z_cond: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    """Strictly causal semantic `RVQ-0` entropy `[*lead]` depending ONLY on `z_cond` (zero future codes!)."""
    lead = z_cond.shape[:-1]
    n_tok = 1
    for s in lead:
        n_tok *= int(s)
    dd = spec.audio_depth_dim
    z_flat = z_cond.reshape(n_tok, spec.d_model)
    z_proj = dense_mm(rms_norm(z_flat, head_params["z_norm_w"]), head_params["in_proj"], matmul_dtype)  # [N, dd]

    if spec.audio_head_type == "flat_rvq":
        h_out = rms_norm(z_proj, head_params["out_norm_w"])
        lg0 = dense_mm(h_out, head_params["w_out"][0], matmul_dtype) + head_params["out_bias"][0][None, :]
    else:
        # Level k=0 sees ONLY `z_proj + level_emb[0]` (causal length-1 prefix, identical to k=0 in K=16 Depth Transformer!)
        h = (z_proj + head_params["level_emb"][0][None, :])[:, None, :]  # [N, 1, dd]
        nh = spec.audio_depth_heads
        dh = dd // nh
        for lyr in head_params["layers"]:
            hn = rms_norm(h, lyr["attn_norm_w"])
            qkv = dense_mm(hn, lyr["w_qkv"], matmul_dtype).reshape(n_tok, 1, 3, nh, dh)
            # For length-1 self-attention, softmax weight is identically 1.0 -> attn == v!
            v = qkv[:, :, 2].reshape(n_tok, 1, dd)
            h = h + dense_mm(v, lyr["w_o"], matmul_dtype)
            fn = rms_norm(h, lyr["ffn_norm_w"])
            ff = jax.nn.silu(dense_mm(fn, lyr["w_g"], matmul_dtype)) * dense_mm(fn, lyr["w_u"], matmul_dtype)
            h = h + dense_mm(ff, lyr["w_d"], matmul_dtype)
        h_out = rms_norm(h[:, 0, :], head_params["out_norm_w"])
        lg0 = dense_mm(h_out, head_params["w_out"][0], matmul_dtype) + head_params["out_bias"][0][None, :]

    p0 = jax.nn.softmax(lg0, axis=-1)
    ent0 = -jnp.sum(p0 * jnp.log(jnp.clip(p0, 1e-9, 1.0)), axis=-1)
    return ent0.reshape(lead)


def sample_audio_codes(
    head_params: PyTree,
    z_t: Array,
    rng_key: Array,
    spec: SLWMv4Spec,
    *,
    temperature: float = 0.8,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, Array]:
    """Sample all `K = 16` Mimi RVQ codes `[B, 16]` strictly causally (`k = 0..15`) + return semantic `rvq0_entropy [B]`."""
    b_sz = z_t.shape[0]
    k_rvq = spec.audio_codebooks
    dd = spec.audio_depth_dim
    temp = max(float(temperature), 1e-4)

    if spec.audio_head_type == "flat_rvq":
        dummy_codes = jnp.zeros((b_sz, k_rvq), dtype=jnp.int32)
        logits = apply_audio_head_teacher_forced(head_params, z_t, dummy_codes, spec, matmul_dtype=matmul_dtype)
        p0 = jax.nn.softmax(logits[:, 0, :], axis=-1)
        ent0 = -jnp.sum(p0 * jnp.log(jnp.clip(p0, 1e-9, 1.0)), axis=-1)
        keys = jax.random.split(rng_key, k_rvq)
        codes = jnp.stack([jax.random.categorical(keys[k], logits[:, k, :] / temp, axis=-1) for k in range(k_rvq)], axis=-1)
        return codes.astype(jnp.int32), ent0

    z_proj = dense_mm(rms_norm(z_t, head_params["z_norm_w"]), head_params["in_proj"], matmul_dtype)  # [B, dd]
    init_codes = jnp.zeros((b_sz, k_rvq), dtype=jnp.int32)
    keys = jax.random.split(rng_key, k_rvq)

    def _ar_level(carry: tuple[Array, Array], k_i: Array) -> tuple[tuple[Array, Array], None]:
        codes_cur, ent0_cur = carry
        # Build prefix embeddings up to current level (static length K=16 with causal mask -> fast XLA!)
        if k_rvq > 1:
            prev_idx = jnp.arange(k_rvq - 1, dtype=jnp.int32)[None, :]
            prev_emb = head_params["rvq_in_emb"][prev_idx, codes_cur[:, :-1]]
            tok_emb = jnp.concatenate([jnp.zeros((b_sz, 1, dd), dtype=jnp.float32), prev_emb], axis=1)
        else:
            tok_emb = jnp.zeros((b_sz, 1, dd), dtype=jnp.float32)
        h_in = z_proj[:, None, :] + tok_emb + head_params["level_emb"][None, :, :]
        h_out = _apply_depth_transformer_hidden(head_params, h_in, spec, matmul_dtype)  # [B, K, dd]
        # Project ONLY active codebook `k_i` (16x less vocab projection bandwidth vs projecting all 16 codebooks!)
        h_k = h_out[:, k_i, :]  # [B, dd]
        lg_k = dense_mm(h_k, head_params["w_out"][k_i], matmul_dtype) + head_params["out_bias"][k_i][None, :]
        c_k = jax.random.categorical(keys[k_i], lg_k / temp, axis=-1).astype(jnp.int32)
        codes_nxt = codes_cur.at[:, k_i].set(c_k)
        p_k = jax.nn.softmax(lg_k, axis=-1)
        e_k = -jnp.sum(p_k * jnp.log(jnp.clip(p_k, 1e-9, 1.0)), axis=-1)
        ent0_nxt = jnp.where(k_i == 0, e_k, ent0_cur)
        return (codes_nxt, ent0_nxt), None

    (codes_fin, rvq0_ent), _ = jax.lax.scan(
        _ar_level, (init_codes, jnp.zeros((b_sz,), dtype=jnp.float32)), jnp.arange(k_rvq, dtype=jnp.int32)
    )
    return codes_fin, rvq0_ent


# ---------------------------------------------------------------------------
# 3. Unified Motor Head (52-DoF Action + 68-Dim Proprio + Episode Success)
# ---------------------------------------------------------------------------


def init_motor_head_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize unified humanoid motor head (52-DoF action + 68-dim proprio/force + success logit)."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    d = spec.d_model
    m_d = spec.width_mult
    hid = max(128, d)
    return {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_g": init_mup_weight(k1, (d, hid), role="hidden", width_mult=m_d, base_std=0.5 / math.sqrt(float(spec.d_base))),
        "w_u": init_mup_weight(k2, (d, hid), role="hidden", width_mult=m_d, base_std=0.5 / math.sqrt(float(spec.d_base))),
        "w_act": init_mup_weight(k3, (hid, spec.action_dim), role="readout", width_mult=m_d, zero_init=True),
        "b_act": jnp.zeros((spec.action_dim,), dtype=jnp.float32),
        "w_prop": init_mup_weight(k4, (hid, spec.proprio_dim), role="readout", width_mult=m_d, zero_init=True),
        "b_prop": jnp.zeros((spec.proprio_dim,), dtype=jnp.float32),
        "w_succ": init_mup_weight(k5, (hid, 1), role="readout", width_mult=m_d, zero_init=True),
        "b_succ": jnp.zeros((1,), dtype=jnp.float32),
    }


def apply_motor_head(
    head_params: PyTree,
    z: Array,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, Array]:
    """Predict `(action_pred [..., 52], proprio_pred [..., 68], success_logit [...])`."""
    zn = rms_norm(z, head_params["norm_w"])
    hid = jax.nn.silu(dense_mm(zn, head_params["w_g"], matmul_dtype)) * dense_mm(zn, head_params["w_u"], matmul_dtype)
    return {
        "action_pred": dense_mm(hid, head_params["w_act"], matmul_dtype) + head_params["b_act"],
        "proprio_pred": dense_mm(hid, head_params["w_prop"], matmul_dtype) + head_params["b_prop"],
        "success_logit": (dense_mm(hid, head_params["w_succ"], matmul_dtype) + head_params["b_succ"])[..., 0],
    }


# ---------------------------------------------------------------------------
# 4. Text Byte Head (264 UTF-8 Vocab)
# ---------------------------------------------------------------------------


def init_text_head_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    return {
        "norm_w": jnp.ones((spec.d_model,), dtype=jnp.float32),
        "w": init_mup_weight(key, (spec.d_model, spec.text_vocab_size), role="readout", width_mult=spec.width_mult, zero_init=True),
        "bias": jnp.zeros((spec.text_vocab_size,), dtype=jnp.float32),
    }


def apply_text_head(head_params: PyTree, z: Array, *, matmul_dtype: jnp.dtype = jnp.bfloat16) -> Array:
    zn = rms_norm(z, head_params["norm_w"])
    return dense_mm(zn, head_params["w"], matmul_dtype) + head_params["bias"]


# ---------------------------------------------------------------------------
# 5. Strictly Causal Predictive AV-JEPA Synchrony Head (`O1_jepa_sync`)
# ---------------------------------------------------------------------------


def init_jepa_sync_head_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize strictly causal AV-JEPA synchrony head (zero residual identity shortcut, zero future lookahead)."""
    k1, k2, k3 = jax.random.split(key, 3)
    d = spec.d_model
    jp = spec.jepa_proj_dim
    m_d = spec.width_mult
    rvq_emb_d = 16
    return {
        "v_norm_w": jnp.ones((d,), dtype=jnp.float32),
        "proj_v": init_mup_weight(k1, (d, jp), role="readout", width_mult=m_d, base_std=0.5 / math.sqrt(float(spec.d_base))),
        "aud_rvq_emb": init_mup_weight(k2, (spec.audio_codebooks, spec.audio_vocab_size, rvq_emb_d), role="input", width_mult=1.0, base_std=0.05),
        "proj_a": init_mup_weight(k3, (spec.audio_codebooks * rvq_emb_d, jp), role="fixed_hidden", width_mult=1.0, base_std=0.5 / math.sqrt(float(spec.audio_codebooks * rvq_emb_d))),
        "log_temp": jnp.asarray(math.log(0.07), dtype=jnp.float32),
    }


def compute_causal_jepa_sync_loss(
    jepa_params: PyTree,
    z_src: Array,
    batch: dict[str, Array],
    same_ep: Array,
    targ_vld: Array,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, Array]:
    """Compute strictly causal AV-JEPA symmetric InfoNCE over synchronized `(VID_last[h] -> AUD_1[h])` pairs.

    Zero identity shortcut (audio side encodes ONLY `audio_codes[t+1]`) and zero future-heartbeat lookahead
    (requires `heartbeat_id[t] == heartbeat_id[t+1]`, `modality_id[t] == MOD_VIDEO`, `spatial_pos[t] == 14`,
    and `modality_id[t+1] == MOD_AUDIO`).
    """
    tm1, b_sz, d = z_src.shape
    mod_cur = batch["modality_id"][:-1].astype(jnp.int32)
    mod_nxt = batch["modality_id"][1:].astype(jnp.int32)
    sp_cur = batch["spatial_pos"][:-1].astype(jnp.int32)
    hb_cur = batch["heartbeat_id"][:-1].astype(jnp.int32)
    hb_nxt = batch["heartbeat_id"][1:].astype(jnp.int32)

    is_pair = (
        (targ_vld > 0.5)
        & (same_ep > 0.5)
        & (mod_cur == MOD_VIDEO)
        & (sp_cur == (spec.video_spatial_packets - 1))
        & (mod_nxt == MOD_AUDIO)
        & (hb_cur == hb_nxt)
    )
    n_found = jnp.sum(is_pair)
    k_max = max(8, min(512, (tm1 // max(1, spec.video_spatial_packets) + 1) * b_sz))
    t_idx, b_idx = jnp.nonzero(is_pair, size=k_max, fill_value=0)
    pair_mask = jnp.arange(k_max, dtype=jnp.int32) < n_found  # [K]

    z_v = z_src[t_idx, b_idx]  # [K, D]
    a_codes = jnp.clip(batch["audio_codes"][1:][t_idx, b_idx, : spec.audio_codebooks].astype(jnp.int32), 0, spec.audio_vocab_size - 1)

    zv_proj = dense_mm(rms_norm(z_v, jepa_params["v_norm_w"]), jepa_params["proj_v"], matmul_dtype)
    zv_norm = zv_proj * jax.lax.rsqrt(jnp.sum(zv_proj * zv_proj, axis=-1, keepdims=True) + 1e-6)

    cb_idx = jnp.arange(spec.audio_codebooks, dtype=jnp.int32)[None, :]
    a_flat = jepa_params["aud_rvq_emb"][cb_idx, a_codes].reshape(k_max, -1)
    za_proj = dense_mm(a_flat, jepa_params["proj_a"], matmul_dtype)
    za_norm = za_proj * jax.lax.rsqrt(jnp.sum(za_proj * za_proj, axis=-1, keepdims=True) + 1e-6)

    temp = jnp.exp(jnp.clip(jepa_params["log_temp"], math.log(0.02), math.log(1.0)))
    logits = jnp.matmul(zv_norm, za_norm.T, precision=jax.lax.Precision.HIGHEST) / temp  # [K, K]
    # Mask invalid padded columns strictly to -1e9
    col_mask = pair_mask[None, :]
    row_mask = pair_mask.astype(jnp.float32)
    labels = jnp.arange(k_max, dtype=jnp.int32)

    logits_v2a = jnp.where(col_mask, logits, -1e9)
    logits_a2v = jnp.where(col_mask, logits.T, -1e9)
    # Keep diagonal valid even for padded rows so softmax never NaNs before masking by `row_mask`
    eye_b = jnp.eye(k_max, dtype=jnp.bool_)
    logits_v2a = jnp.where(eye_b & (~pair_mask[:, None]), 0.0, logits_v2a)
    logits_a2v = jnp.where(eye_b & (~pair_mask[:, None]), 0.0, logits_a2v)

    ce_v2a = optax.softmax_cross_entropy_with_integer_labels(logits_v2a, labels)
    ce_a2v = optax.softmax_cross_entropy_with_integer_labels(logits_a2v, labels)
    denom = jnp.maximum(jnp.sum(row_mask), 1.0)
    loss = 0.5 * (jnp.sum(ce_v2a * row_mask) + jnp.sum(ce_a2v * row_mask)) / denom
    loss = jnp.where(n_found >= 2, loss, 0.0)
    return loss, n_found.astype(jnp.float32)


# ---------------------------------------------------------------------------
# 6. Frozen-Core Calibrated PolicyCommitGate (Strictly `stop_gradient(Z_t)`)
# ---------------------------------------------------------------------------


def init_policy_commit_gate_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize 5-action PolicyCommitGate with strict `stop_gradient(Z_t)` & learnable calibration temperature."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    d = spec.d_model
    hid = spec.policy_hidden_dim
    n_ev = 5  # [text_entropy, aud_rvq0_entropy, z_rms, reliability, mod_avail_frac]
    ev_hid = 32
    n_act = len(COMMIT_ACTIONS)
    return {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w1": init_mup_weight(k1, (d, hid), role="hidden", width_mult=1.0, base_std=1.0 / math.sqrt(float(d))),
        "b1": jnp.zeros((hid,), dtype=jnp.float32),
        "w2": init_mup_weight(k2, (hid, n_act), role="readout", width_mult=1.0, zero_init=True),
        "ev_skip": init_mup_weight(k3, (n_ev, n_act), role="input", width_mult=1.0, base_std=0.25 / math.sqrt(float(n_ev))),
        "ev_w1": init_mup_weight(k4, (n_ev, ev_hid), role="input", width_mult=1.0, base_std=1.0 / math.sqrt(float(n_ev))),
        "ev_b1": jnp.zeros((ev_hid,), dtype=jnp.float32),
        "ev_w2": init_mup_weight(k5, (ev_hid, n_act), role="input", width_mult=1.0, base_std=0.25 / math.sqrt(float(ev_hid))),
        "b2": jnp.zeros((n_act,), dtype=jnp.float32),
        "log_temp": jnp.asarray(0.0, dtype=jnp.float32),
    }


def apply_policy_commit_gate(
    gate_params: PyTree,
    z_t: Array,
    evidence_feats: Array,
    *,
    matmul_dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, Array]:
    """Select calibrated committed output action strictly on `jax.lax.stop_gradient(z_t)` (zero core grad!)."""
    z_det = jax.lax.stop_gradient(z_t.astype(jnp.float32))
    ev_det = jax.lax.stop_gradient(evidence_feats.astype(jnp.float32))
    zn = rms_norm(z_det, gate_params["norm_w"])
    h = jax.nn.silu(dense_mm(zn, gate_params["w1"], matmul_dtype) + gate_params["b1"])
    ev_h = jax.nn.silu(dense_mm(ev_det, gate_params["ev_w1"], matmul_dtype) + gate_params["ev_b1"])
    raw_logits = (
        dense_mm(h, gate_params["w2"], matmul_dtype)
        + dense_mm(ev_det, gate_params["ev_skip"], matmul_dtype)
        + dense_mm(ev_h, gate_params["ev_w2"], matmul_dtype)
        + gate_params["b2"]
    )
    temp = jnp.exp(jnp.clip(gate_params["log_temp"].astype(jnp.float32), -2.0, 2.0))
    scaled_logits = raw_logits / temp
    probs = jax.nn.softmax(scaled_logits, axis=-1)
    return scaled_logits, probs


# ---------------------------------------------------------------------------
# 7. Unified Heads Factory & Strictly Causal Multimodal Training Loss
# ---------------------------------------------------------------------------


def init_v4_heads_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize all active SLWM-v4 shared generative, predictive, synchrony & policy heads."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    heads: PyTree = {
        "head_video": init_video_head_params(k1, spec),
        "head_audio": init_audio_head_params(k2, spec),
        "head_motor": init_motor_head_params(k3, spec),
        "head_text": init_text_head_params(k4, spec),
    }
    if spec.use_jepa_sync:
        heads["head_jepa"] = init_jepa_sync_head_params(k5, spec)
    if spec.use_policy_commit_gate:
        heads["policy_gate"] = init_policy_commit_gate_params(k6, spec)
    return heads


def compute_v4_heads_loss(
    heads_params: PyTree,
    z_seq: Array,
    batch: dict[str, Array],
    spec: SLWMv4Spec,
    rng_key: Array,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    loss_weights: dict[str, float] | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute strictly causal next-packet multimodal loss across ALL active SLWM-v4 heads.

    Strict isolation & masking invariants:
      - Zero cross-episode leakage (`same_ep = ep[t] == ep[t+1]`).
      - Zero invalid-source or invalid-target supervision (`src_vld = valid_mask[t]`, `targ_vld = target_valid_mask[t+1]`).
      - Exact `0.0` gradient on inactive `action_mask == 0`, inactive `proprio_mask == 0`, and unlabeled `episode_success == -1`.
      - Exact `0.0` gradient from `policy_gate` into `z_seq` / world core (`jax.lax.stop_gradient`).
    """
    lw = loss_weights or {"text": 1.0, "audio": 1.0, "video": 1.0, "motor": 1.0, "success": 0.25, "jepa": 0.15, "policy": 0.10}
    t_len, b_sz, _ = z_seq.shape
    tm1 = t_len - 1
    k_vid_t, k_vid_n = jax.random.split(rng_key, 2)

    rst = batch.get("reset_mask", jnp.zeros((t_len, b_sz), dtype=jnp.bool_)).astype(jnp.bool_)
    ep_id = jnp.cumsum(rst.astype(jnp.int32), axis=0)
    same_ep = (ep_id[:-1] == ep_id[1:]).astype(jnp.float32)  # [T-1, B]
    src_vld = batch["valid_mask"][:-1].astype(jnp.float32)
    targ_vld = batch.get("target_valid_mask", batch["valid_mask"]).astype(jnp.float32)[1:] * src_vld * same_ep

    z_src = z_seq[:-1]  # [T-1, B, D]
    next_mod = batch["modality_id"][1:].astype(jnp.int32)  # [T-1, B]

    # 1. Text / Code Next-Byte CE (`modality_id == 2`)
    m_txt = targ_vld * (next_mod == MOD_TEXT).astype(jnp.float32)
    txt_logits = apply_text_head(heads_params["head_text"], z_src, matmul_dtype=matmul_dtype)
    txt_targets = jnp.clip(batch["text_byte"][1:].astype(jnp.int32), 0, spec.text_vocab_size - 1)
    txt_ce_per = optax.softmax_cross_entropy_with_integer_labels(txt_logits, txt_targets)
    txt_cnt = jnp.sum(m_txt)
    txt_loss = jnp.sum(txt_ce_per * m_txt) / jnp.maximum(txt_cnt, 1.0)
    txt_has = (txt_cnt > 0.5).astype(jnp.float32)

    # 2. Audio 16-RVQ Depth Transformer / Flat RVQ (`modality_id == 1`, chunked `@jax.checkpoint`!)
    m_aud = targ_vld * (next_mod == MOD_AUDIO).astype(jnp.float32)
    aud_cnt = jnp.sum(m_aud)
    aud_has = (aud_cnt > 0.5).astype(jnp.float32)
    k_rvq = spec.audio_codebooks
    aud_targets = jnp.clip(batch["audio_codes"][1:, :, :k_rvq].astype(jnp.int32), 0, spec.audio_vocab_size - 1)
    cb_weights = jnp.ones((k_rvq,), dtype=jnp.float32).at[0].set(2.0)
    cb_weights = cb_weights / jnp.sum(cb_weights)

    def _chunked_audio_loss(_: None) -> tuple[Array, Array]:
        c_sz = min(64, max(1, tm1))
        rem = tm1 % c_sz
        pad_n = (c_sz - rem) if rem != 0 else 0
        n_ch = (tm1 + pad_n) // c_sz
        if pad_n > 0:
            zp = jnp.concatenate([z_src, jnp.zeros((pad_n, b_sz, spec.d_model), dtype=z_src.dtype)], axis=0)
            tp = jnp.concatenate([aud_targets, jnp.zeros((pad_n, b_sz, k_rvq), dtype=jnp.int32)], axis=0)
            mp = jnp.concatenate([m_aud, jnp.zeros((pad_n, b_sz), dtype=jnp.float32)], axis=0)
        else:
            zp, tp, mp = z_src, aud_targets, m_aud
        zc = zp.reshape(n_ch, c_sz, b_sz, spec.d_model)
        tc = tp.reshape(n_ch, c_sz, b_sz, k_rvq)
        mc = mp.reshape(n_ch, c_sz, b_sz)

        @jax.checkpoint
        def _aud_chunk(acc: tuple[Array, Array], xs: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array], None]:
            acc_all, acc_0 = acc
            z_i, t_i, m_i = xs
            lg = apply_audio_head_teacher_forced(heads_params["head_audio"], z_i, t_i, spec, matmul_dtype=matmul_dtype)
            ce_k = optax.softmax_cross_entropy_with_integer_labels(lg, t_i)  # [C, B, K]
            ce_w = jnp.sum(ce_k * cb_weights[None, None, :], axis=-1)  # [C, B]
            return (acc_all + jnp.sum(ce_w * m_i), acc_0 + jnp.sum(ce_k[:, :, 0] * m_i)), None

        (tot_all, tot_0), _ = jax.lax.scan(
            _aud_chunk, (jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)), (zc, tc, mc)
        )
        denom = jnp.maximum(aud_cnt, 1.0)
        return tot_all / denom, tot_0 / denom

    aud_loss, aud_rvq0_loss = jax.lax.cond(
        aud_cnt > 0.5,
        _chunked_audio_loss,
        lambda _: (jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)),
        None,
    )

    # 3. Video Patch Head: Rectified Flow Matching (`flow`) OR Huber (`huber`), chunked `@jax.checkpoint`!
    m_vid = targ_vld * (next_mod == MOD_VIDEO).astype(jnp.float32)
    vid_cnt = jnp.sum(m_vid)
    vid_has = (vid_cnt > 0.5).astype(jnp.float32)
    vid_targ = jax.lax.stop_gradient(batch["video_feat"][1:].astype(jnp.float32))
    sp_next = batch["spatial_pos"][1:].astype(jnp.int32)

    def _chunked_video_loss(_: None) -> Array:
        c_sz = min(128, max(1, tm1))
        rem = tm1 % c_sz
        pad_n = (c_sz - rem) if rem != 0 else 0
        n_ch = (tm1 + pad_n) // c_sz
        tau_all = jax.nn.sigmoid(jax.random.normal(k_vid_t, (tm1, b_sz), dtype=jnp.float32))
        x0_all = jax.random.normal(k_vid_n, (tm1, b_sz, spec.video_patch_dim), dtype=jnp.float32)

        if pad_n > 0:
            zp = jnp.concatenate([z_src, jnp.zeros((pad_n, b_sz, spec.d_model), dtype=z_src.dtype)], axis=0)
            vp = jnp.concatenate([vid_targ, jnp.zeros((pad_n, b_sz, spec.video_patch_dim), dtype=jnp.float32)], axis=0)
            sp_p = jnp.concatenate([sp_next, jnp.zeros((pad_n, b_sz), dtype=jnp.int32)], axis=0)
            mp = jnp.concatenate([m_vid, jnp.zeros((pad_n, b_sz), dtype=jnp.float32)], axis=0)
            taup = jnp.concatenate([tau_all, jnp.zeros((pad_n, b_sz), dtype=jnp.float32)], axis=0)
            x0p = jnp.concatenate([x0_all, jnp.zeros((pad_n, b_sz, spec.video_patch_dim), dtype=jnp.float32)], axis=0)
        else:
            zp, vp, sp_p, mp, taup, x0p = z_src, vid_targ, sp_next, m_vid, tau_all, x0_all

        zc = zp.reshape(n_ch, c_sz, b_sz, spec.d_model)
        vc = vp.reshape(n_ch, c_sz, b_sz, spec.video_patch_dim)
        spc = sp_p.reshape(n_ch, c_sz, b_sz)
        mc = mp.reshape(n_ch, c_sz, b_sz)
        tauc = taup.reshape(n_ch, c_sz, b_sz)
        x0c = x0p.reshape(n_ch, c_sz, b_sz, spec.video_patch_dim)

        @jax.checkpoint
        def _vid_chunk(acc: Array, xs: tuple[Array, Array, Array, Array, Array, Array]) -> tuple[Array, None]:
            z_i, v_i, sp_i, m_i, tau_i, x0_i = xs
            if spec.video_head_type == "flow":
                x_tau = (1.0 - tau_i[..., None]) * x0_i + tau_i[..., None] * v_i
                v_targ = v_i - x0_i
                v_pred = apply_video_flow_velocity(heads_params["head_video"], z_i, sp_i, x_tau, tau_i, spec, matmul_dtype=matmul_dtype)
                err = jnp.mean((v_pred - v_targ) ** 2, axis=-1)
            else:
                v_pred = apply_video_huber_pred(heads_params["head_video"], z_i, sp_i, spec, matmul_dtype=matmul_dtype)
                diff = v_pred - v_i
                ad = jnp.abs(diff)
                err = jnp.mean(jnp.where(ad <= 1.0, 0.5 * diff * diff, ad - 0.5), axis=-1)
            return acc + jnp.sum(err * m_i), None

        tot_v, _ = jax.lax.scan(_vid_chunk, jnp.asarray(0.0, dtype=jnp.float32), (zc, vc, spc, mc, tauc, x0c))
        return tot_v / jnp.maximum(vid_cnt, 1.0)

    vid_loss = jax.lax.cond(vid_cnt > 0.5, _chunked_video_loss, lambda _: jnp.asarray(0.0, dtype=jnp.float32), None)

    # 4. Unified Motor Head (52-DoF Action + 68-Dim Proprio + Episode Success, STRICTLY PER-CHANNEL MASKED!)
    m_mot = targ_vld * (next_mod == MOD_MOTOR).astype(jnp.float32)
    mot_out = apply_motor_head(heads_params["head_motor"], z_src, matmul_dtype=matmul_dtype)
    act_targ = jax.lax.stop_gradient(batch["action_feat"][1:].astype(jnp.float32))
    prop_targ = jax.lax.stop_gradient(batch["proprio_feat"][1:].astype(jnp.float32))
    act_mask = batch["action_mask"][1:].astype(jnp.float32)  # [T-1, B, 52]
    prop_mask = batch["proprio_mask"][1:].astype(jnp.float32)  # [T-1, B, 68]

    # Per-channel masked Huber -> EXACT 0.0 gradient on inactive channels (`mask == 0`)!
    ad_a = jnp.abs(mot_out["action_pred"] - act_targ)
    hub_a = jnp.where(ad_a <= 1.0, 0.5 * ad_a * ad_a, ad_a - 0.5) * act_mask
    a_ch_cnt = jnp.sum(act_mask, axis=-1)  # [T-1, B]
    m_act_tok = m_mot * (a_ch_cnt > 0.5).astype(jnp.float32)
    act_per_tok = jnp.sum(hub_a, axis=-1) / jnp.maximum(a_ch_cnt, 1.0)
    act_cnt = jnp.sum(m_act_tok)
    act_loss = jnp.sum(act_per_tok * m_act_tok) / jnp.maximum(act_cnt, 1.0)

    ad_p = jnp.abs(mot_out["proprio_pred"] - prop_targ)
    hub_p = jnp.where(ad_p <= 1.0, 0.5 * ad_p * ad_p, ad_p - 0.5) * prop_mask
    p_ch_cnt = jnp.sum(prop_mask, axis=-1)
    m_prop_tok = m_mot * (p_ch_cnt > 0.5).astype(jnp.float32)
    prop_per_tok = jnp.sum(hub_p, axis=-1) / jnp.maximum(p_ch_cnt, 1.0)
    prop_cnt = jnp.sum(m_prop_tok)
    prop_loss = jnp.sum(prop_per_tok * m_prop_tok) / jnp.maximum(prop_cnt, 1.0)

    mot_has = ((act_cnt + prop_cnt) > 0.5).astype(jnp.float32)
    mot_denom = (act_cnt > 0.5).astype(jnp.float32) + (prop_cnt > 0.5).astype(jnp.float32)
    mot_loss = (
        act_loss * (act_cnt > 0.5).astype(jnp.float32) + prop_loss * (prop_cnt > 0.5).astype(jnp.float32)
    ) / jnp.maximum(mot_denom, 1.0)

    # Optional episode_success BCE strictly where `episode_success >= 0` (`-1` -> exact 0.0 grad!)
    ep_succ = batch.get("episode_success", jnp.full((t_len, b_sz), -1, dtype=jnp.int32))[1:].astype(jnp.int32)
    m_succ = m_mot * (ep_succ >= 0).astype(jnp.float32)
    succ_cnt = jnp.sum(m_succ)
    succ_has = (succ_cnt > 0.5).astype(jnp.float32)
    succ_y = jnp.clip(ep_succ.astype(jnp.float32), 0.0, 1.0)
    succ_bce = optax.sigmoid_binary_cross_entropy(mot_out["success_logit"], succ_y)
    succ_loss = jnp.sum(succ_bce * m_succ) / jnp.maximum(succ_cnt, 1.0)

    # 5. Optional Strictly Causal AV-JEPA Synchrony (`O1_jepa_sync`)
    if spec.use_jepa_sync and "head_jepa" in heads_params:
        jepa_loss, jepa_cnt = compute_causal_jepa_sync_loss(
            heads_params["head_jepa"], z_src, batch, same_ep, targ_vld, spec, matmul_dtype=matmul_dtype
        )
        jepa_has = (jepa_cnt >= 2.0).astype(jnp.float32)
    else:
        jepa_loss = jnp.asarray(0.0, dtype=jnp.float32)
        jepa_cnt = jnp.asarray(0.0, dtype=jnp.float32)
        jepa_has = jnp.asarray(0.0, dtype=jnp.float32)

    # 6. Frozen-Core PolicyCommitGate Aux Supervision (STRICTLY `stop_gradient(z_src)` -> ZERO core grad!)
    if spec.use_policy_commit_gate and "policy_gate" in heads_params:
        # Target commit action derived strictly from next valid modality (or ABSTAIN_SILENCE=4 when target_valid==0!)
        # Map MOD_TEXT(2)->0(SPEAK_TEXT), MOD_AUDIO(1)->1(EMIT_AUDIO), MOD_VIDEO(0)->2(GENERATE_VIDEO), MOD_MOTOR(3)->3(ACT_MOTOR)
        mod_to_act = jnp.asarray([2, 1, 0, 3], dtype=jnp.int32)
        pol_targ = jnp.where(
            batch.get("target_valid_mask", batch["valid_mask"])[1:].astype(jnp.float32) > 0.5,
            mod_to_act[jnp.clip(next_mod, 0, 3)],
            4,  # ABSTAIN_SILENCE
        )
        m_pol = src_vld * same_ep
        pol_cnt = jnp.sum(m_pol)
        pol_has = (pol_cnt > 0.5).astype(jnp.float32)

        txt_p = jax.nn.softmax(jax.lax.stop_gradient(txt_logits), axis=-1)
        txt_ent = -jnp.sum(txt_p * jnp.log(jnp.clip(txt_p, 1e-9, 1.0)), axis=-1)
        aud_ent0 = jax.lax.stop_gradient(
            compute_audio_rvq0_entropy(heads_params["head_audio"], jax.lax.stop_gradient(z_src), spec, matmul_dtype=matmul_dtype)
        )
        rel_cur = batch["reliability"][:-1].astype(jnp.float32)
        mod_avail = (batch.get("target_valid_mask", batch["valid_mask"])[:-1].astype(jnp.float32) > 0.5).astype(jnp.float32)
        ev_feats = jnp.stack(
            [
                txt_ent / math.log(float(spec.text_vocab_size)),
                aud_ent0 / math.log(float(spec.audio_vocab_size)),
                jnp.sqrt(jnp.mean(jax.lax.stop_gradient(z_src) ** 2, axis=-1) + 1e-6),
                rel_cur,
                mod_avail,
            ],
            axis=-1,
        )
        pol_logits, _ = apply_policy_commit_gate(heads_params["policy_gate"], z_src, ev_feats, matmul_dtype=matmul_dtype)
        pol_ce = optax.softmax_cross_entropy_with_integer_labels(pol_logits, pol_targ)
        pol_loss = jnp.sum(pol_ce * m_pol) / jnp.maximum(pol_cnt, 1.0)
    else:
        pol_loss = jnp.asarray(0.0, dtype=jnp.float32)
        pol_has = jnp.asarray(0.0, dtype=jnp.float32)

    w_sum = (
        lw["text"] * txt_has
        + lw["audio"] * aud_has
        + lw["video"] * vid_has
        + lw["motor"] * mot_has
    )
    core_loss = (
        lw["text"] * txt_loss * txt_has
        + lw["audio"] * aud_loss * aud_has
        + lw["video"] * vid_loss * vid_has
        + lw["motor"] * mot_loss * mot_has
    ) / jnp.maximum(w_sum, 1.0)

    total_loss = (
        core_loss
        + lw.get("success", 0.25) * succ_loss * succ_has
        + lw.get("jepa", 0.15) * jepa_loss * jepa_has
        + lw.get("policy", 0.10) * pol_loss * pol_has
    )

    metrics = {
        "total_loss": total_loss,
        "core_loss": core_loss,
        "text_ce_nat": txt_loss,
        "text_bpb": txt_loss / jnp.log(2.0),
        "audio_rvq_ce_nat": aud_loss,
        "audio_rvq0_ce_nat": aud_rvq0_loss,
        "video_loss": vid_loss,
        "action_huber": act_loss,
        "proprio_huber": prop_loss,
        "motor_loss": mot_loss,
        "success_bce": succ_loss,
        "jepa_sync_loss": jepa_loss,
        "policy_ce_nat": pol_loss,
        "n_text_targets": txt_cnt,
        "n_audio_targets": aud_cnt,
        "n_video_targets": vid_cnt,
        "n_action_targets": act_cnt,
        "n_proprio_targets": prop_cnt,
        "n_success_targets": succ_cnt,
        "n_jepa_pairs": jepa_cnt,
    }
    return total_loss, metrics
