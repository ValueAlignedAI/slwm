"""SLWM-v4 Complete World Model: Edge Encoders, Causal O(1) Core, Self-Forcing & Live 2+ Min Rollout Engine.

Unifies:
  1. Strictly Causal Multimodal Edge Encoder (`embed_v4_packets_seq` / `embed_v4_packet_step`):
       - Wan 2.1 video patches (`1024`-dim + spatial pos `0..14` + optional Diffusion Forcing noise level `sigma_h`).
       - Kyutai Mimi 16-RVQ audio codes (`16 x 2048`).
       - Lossless UTF-8 bytes (`264` vocab).
       - Non-aliased `52`-DoF Action + `68`-Dim Proprio/Force with explicit binary `action_mask` & `proprio_mask`
         concatenated into the input features (disambiguating active `0.0` joint angle from absent DoF per Gate 1 §2).
  2. Separable SLWM-v4 Core Processor (`apply_v4_core_seq` / `step_v4_core`):
       - Supports ALL 17 pre-registered `V4-S3` architectures (`hybrid_gdn_gqa`, `sw_gqa_only`, `slwm_v3_dual_ref`),
         exact CompleteP `alpha = 1.0` depth residual scaling, and layer-wise `@jax.checkpoint`.
  3. Full Training Loss (`compute_v4_training_loss`):
       - Combines all active heads + optional `O1_jepa_sync` + optional `H3_self_forcing` (causal per-heartbeat
         Diffusion Forcing noise + short strictly-causal closed-loop self-fed video rollout consistency).
  4. Dual-Track Accounting (`summarize_v4_accounting`) & Base-Shape muP Optax Optimizer (`make_v4_mup_optimizer`).
  5. Flat-O(1) Live Streaming Heartbeat Rollout Driver (`run_v4_live_heartbeat_rollout`):
       - Runs continuous `2+ minute` (`750+` heartbeats) simultaneous `Video + Audio + Motor + Text` streaming
         strictly via `step_v4_core` with `0.0%` state growth and explicit `AGENTS.md §13.2` epistemic labels
         (`observed`, `predicted`, `imagined`, `abstained`).
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import optax

from data.v4_heartbeat_contract import MOD_AUDIO, MOD_MOTOR, MOD_TEXT, MOD_VIDEO
from models.v4.blocks import (
    apply_ffn,
    apply_gdn_v4_seq,
    apply_sw_gqa_sink_seq,
    apply_v3_legacy_mem_seq,
    classify_v4_param_mup_role,
    dense_mm,
    init_ffn_params,
    init_gdn_v4_params,
    init_gdn_v4_state,
    init_mup_weight,
    init_sw_gqa_sink_params,
    init_sw_gqa_sink_state,
    rms_norm,
    step_gdn_v4,
    step_sw_gqa_sink,
    step_v3_legacy_mem,
)
from models.v4.heads import (
    COMMIT_ACTIONS,
    apply_motor_head,
    apply_policy_commit_gate,
    apply_text_head,
    apply_video_flow_velocity,
    apply_video_huber_pred,
    compute_audio_rvq0_entropy,
    compute_v4_heads_loss,
    init_v4_heads_params,
    sample_audio_codes,
    sample_video_patch,
)
from models.v4.spec import (
    SLWMv4Spec,
    compute_v4_core_streaming_state_bytes,
    count_state_tree_bytes,
    count_tree_params,
    estimate_v4_active_core_param_count,
    estimate_v4_core_param_count,
)

Array = jax.Array
PyTree = dict[str, Any]

FROZEN_CODEC_PARAMS_WAN21 = 126_892_531  # Exact Wan 2.1 Causal VAE params (Gate 2 receipt)
FROZEN_CODEC_PARAMS_MIMI = 79_308_609    # Exact Kyutai Mimi 16-RVQ params (Gate 2 receipt)
FROZEN_CODEC_STATE_BYTES_BIDIR_BF16 = 463_073_280 + 8_182_408  # Exact 449.42 MiB (Wan 2.1 bf16 + Mimi KV=249 bf16)


# ---------------------------------------------------------------------------
# 1. Strictly Causal Multimodal Edge Encoders
# ---------------------------------------------------------------------------


def init_v4_edge_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize shared multimodal edge adapters (identical across all compared backbones)."""
    keys = jax.random.split(key, 12)
    d = spec.d_model
    m_d = spec.width_mult
    rvq_emb_d = 32
    act_in_d = 2 * spec.action_dim     # [action * mask, mask] -> 104
    prop_in_d = 2 * spec.proprio_dim   # [proprio * mask, mask] -> 136

    return {
        "edge_video_w": init_mup_weight(keys[0], (spec.video_patch_dim, d), role="input", width_mult=m_d, base_std=0.5 / math.sqrt(float(spec.video_patch_dim))),
        "edge_video_bias": jnp.zeros((d,), dtype=jnp.float32),
        "edge_spatial_pos_emb": init_mup_weight(keys[1], (spec.video_spatial_packets, d), role="input", width_mult=m_d, base_std=0.05),
        "edge_audio_rvq_emb": init_mup_weight(keys[2], (spec.audio_codebooks, spec.audio_vocab_size, rvq_emb_d), role="input", width_mult=m_d, base_std=0.05),
        "edge_audio_rvq_w": init_mup_weight(keys[3], (spec.audio_codebooks * rvq_emb_d, d), role="input", width_mult=m_d, base_std=0.5 / math.sqrt(float(spec.audio_codebooks * rvq_emb_d))),
        "edge_audio_bias": jnp.zeros((d,), dtype=jnp.float32),
        "edge_text_emb": init_mup_weight(keys[4], (spec.text_vocab_size, d), role="input", width_mult=m_d, base_std=0.05),
        "edge_action_w": init_mup_weight(keys[5], (act_in_d, d), role="input", width_mult=m_d, base_std=0.5 / math.sqrt(float(act_in_d))),
        "edge_proprio_w": init_mup_weight(keys[6], (prop_in_d, d), role="input", width_mult=m_d, base_std=0.5 / math.sqrt(float(prop_in_d))),
        "edge_motor_bias": jnp.zeros((d,), dtype=jnp.float32),
        "edge_mod_emb": init_mup_weight(keys[7], (spec.n_modalities, d), role="input", width_mult=m_d, base_std=0.05),
        # Meta features: [log1p(dt_s), reliability, video_noise_sigma]
        "edge_meta_w": init_mup_weight(keys[8], (3, d), role="input", width_mult=m_d, base_std=0.05),
    }


def _embed_v4_common(
    edge_params: PyTree,
    packets: PyTree,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype,
) -> Array:
    """Rank-polymorphic (`[B, ...]` or `[T, B, ...]`) strictly causal multimodal packet embedding."""
    mod_id = jnp.clip(packets["modality_id"].astype(jnp.int32), 0, spec.n_modalities - 1)
    lead = mod_id.shape
    valid = packets["valid_mask"].astype(jnp.float32)
    dt_s = jnp.maximum(packets["dt_s"].astype(jnp.float32), 0.0)
    rel = packets["reliability"].astype(jnp.float32)
    vid_sig = packets.get("video_noise_sigma", jnp.zeros(lead, dtype=jnp.float32)).astype(jnp.float32)

    # 0. Video patch + spatial position
    v_feat = packets.get("video_feat", jnp.zeros(lead + (spec.video_patch_dim,), dtype=jnp.float32))
    sp_pos = jnp.clip(packets.get("spatial_pos", jnp.zeros(lead, dtype=jnp.int32)).astype(jnp.int32), 0, spec.video_spatial_packets - 1)
    x_vid = dense_mm(v_feat, edge_params["edge_video_w"], matmul_dtype) + edge_params["edge_video_bias"] + edge_params["edge_spatial_pos_emb"][sp_pos]

    # 1. Audio 16-RVQ codes
    codes = jnp.clip(
        packets.get("audio_codes", jnp.zeros(lead + (spec.audio_codebooks,), dtype=jnp.int32)).astype(jnp.int32),
        0,
        spec.audio_vocab_size - 1,
    )
    cb_idx = jnp.arange(spec.audio_codebooks, dtype=jnp.int32).reshape((1,) * len(lead) + (spec.audio_codebooks,))
    rvq_e = edge_params["edge_audio_rvq_emb"][cb_idx, codes].reshape(lead + (-1,))
    x_aud = dense_mm(rvq_e, edge_params["edge_audio_rvq_w"], matmul_dtype) + edge_params["edge_audio_bias"]

    # 2. Text UTF-8 byte
    t_byte = jnp.clip(packets.get("text_byte", jnp.zeros(lead, dtype=jnp.int32)).astype(jnp.int32), 0, spec.text_vocab_size - 1)
    x_txt = edge_params["edge_text_emb"][t_byte]

    # 3. Non-aliased 52-DoF Action + 68-Dim Proprio WITH explicit binary masks concatenated!
    act_f = packets.get("action_feat", jnp.zeros(lead + (spec.action_dim,), dtype=jnp.float32)).astype(jnp.float32)
    act_m = packets.get("action_mask", jnp.zeros(lead + (spec.action_dim,), dtype=jnp.float32)).astype(jnp.float32)
    prop_f = packets.get("proprio_feat", jnp.zeros(lead + (spec.proprio_dim,), dtype=jnp.float32)).astype(jnp.float32)
    prop_m = packets.get("proprio_mask", jnp.zeros(lead + (spec.proprio_dim,), dtype=jnp.float32)).astype(jnp.float32)
    act_in = jnp.concatenate([act_f * act_m, act_m], axis=-1)
    prop_in = jnp.concatenate([prop_f * prop_m, prop_m], axis=-1)
    x_mot = (
        dense_mm(act_in, edge_params["edge_action_w"], matmul_dtype)
        + dense_mm(prop_in, edge_params["edge_proprio_w"], matmul_dtype)
        + edge_params["edge_motor_bias"]
    )

    stacked = jnp.stack([x_vid, x_aud, x_txt, x_mot], axis=-2)  # [*lead, 4, D]
    one_hot = jax.nn.one_hot(mod_id, spec.n_modalities, dtype=jnp.float32)[..., None]  # [*lead, 4, 1]
    x_sel = jnp.sum(stacked * one_hot, axis=-2)

    meta_f = jnp.stack([jnp.log1p(dt_s), rel, vid_sig], axis=-1)
    x_out = x_sel + edge_params["edge_mod_emb"][mod_id] + dense_mm(meta_f, edge_params["edge_meta_w"], matmul_dtype)
    return (x_out * valid[..., None]).astype(jnp.float32)


def embed_v4_packets_seq(
    edge_params: PyTree,
    packets_seq: PyTree,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    """Vectorized sequence-parallel multimodal edge encoder `[T, B, ...] -> float32[T, B, D]`."""
    return _embed_v4_common(edge_params, packets_seq, spec, matmul_dtype=matmul_dtype)


def embed_v4_packet_step(
    edge_params: PyTree,
    packet_t: PyTree,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Single-step live streaming edge encoder `[B, ...] -> float32[B, D]` (bit-exact with seq!)."""
    return _embed_v4_common(edge_params, packet_t, spec, matmul_dtype=matmul_dtype)


# ---------------------------------------------------------------------------
# 2. SLWM-v4 Core Processor Stack & Full Model Params/State
# ---------------------------------------------------------------------------


def init_v4_core_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize ONLY the SLWM-v4 core sequence backbone matching `estimate_v4_core_param_count` exactly."""
    layer_keys = jax.random.split(key, spec.n_layers)
    layers = []
    for i, lt in enumerate(spec.layer_types()):
        k_mix, k_aux, k_ffn = jax.random.split(layer_keys[i], 3)
        if lt == "gdn":
            layers.append(
                {
                    "mem": init_gdn_v4_params(k_mix, spec, is_v3_legacy=False),
                    "ffn": init_ffn_params(k_ffn, spec),
                }
            )
        elif lt == "gqa":
            layers.append(
                {
                    "gqa": init_sw_gqa_sink_params(k_mix, spec, enable_sinks=True),
                    "ffn": init_ffn_params(k_ffn, spec),
                }
            )
        elif lt == "v3_dual":
            layers.append(
                {
                    "gqa": init_sw_gqa_sink_params(k_mix, spec, enable_sinks=False),
                    "mem": init_gdn_v4_params(k_aux, spec, is_v3_legacy=True),
                    "ffn": init_ffn_params(k_ffn, spec),
                }
            )
        else:
            raise ValueError(f"Unknown layer type={lt!r}")
    return {
        "layers": layers,
        "final_norm_w": jnp.ones((spec.d_model,), dtype=jnp.float32),
    }


def init_v4_core_state(spec: SLWMv4Spec, batch_size: int = 1) -> PyTree:
    """Allocate exact O(1) persistent streaming state matching `compute_v4_core_streaming_state_bytes` byte-for-byte."""
    layer_states = []
    for lt in spec.layer_types():
        if lt == "gdn":
            layer_states.append({"mem": init_gdn_v4_state(spec, batch_size, is_v3_legacy=False)})
        elif lt == "gqa":
            layer_states.append({"gqa": init_sw_gqa_sink_state(spec, batch_size)})
        elif lt == "v3_dual":
            layer_states.append(
                {
                    "gqa": init_sw_gqa_sink_state(spec, batch_size),
                    "mem": init_gdn_v4_state(spec, batch_size, is_v3_legacy=True),
                }
            )
        else:
            raise ValueError(f"Unknown layer type={lt!r}")
    return {"layers": layer_states}


def init_v4_model_params(key: Array, spec: SLWMv4Spec) -> PyTree:
    """Initialize complete SLWM-v4 model (`edge`, `core`, `heads`) with strictly separable parameter groups."""
    k_edge, k_core, k_heads = jax.random.split(key, 3)
    return {
        "edge": init_v4_edge_params(k_edge, spec),
        "core": init_v4_core_params(k_core, spec),
        "heads": init_v4_heads_params(k_heads, spec),
    }


def step_v4_core(
    params: PyTree,
    state: PyTree,
    packet_t: PyTree,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Single strictly-causal O(1) live streaming step: `packet_t[B, ...] -> (z_t[B, D], new_state, diag)`."""
    x = embed_v4_packet_step(params["edge"], packet_t, spec, matmul_dtype=matmul_dtype)
    mod_t = packet_t["modality_id"].astype(jnp.int32)
    dt_t = packet_t["dt_s"].astype(jnp.float32)
    vld_t = packet_t["valid_mask"].astype(jnp.float32)
    rel_t = packet_t["reliability"].astype(jnp.float32)
    rst_t = packet_t.get("reset_mask", jnp.zeros_like(vld_t, dtype=jnp.bool_)).astype(jnp.bool_)

    scale = spec.depth_residual_scale
    new_layers = []
    mem_diags: list[dict[str, Array]] = []
    core_p = params["core"]

    for i, lt in enumerate(spec.layer_types()):
        lp = core_p["layers"][i]
        ls = state["layers"][i]
        if lt == "gdn":
            m_out, m_st, m_d = step_gdn_v4(lp["mem"], ls["mem"], x, mod_t, dt_t, vld_t, rel_t, rst_t, spec, matmul_dtype=matmul_dtype)
            x = x + scale * m_out
            x = x + scale * apply_ffn(lp["ffn"], x, mod_t, vld_t, spec, matmul_dtype=matmul_dtype)
            new_layers.append({"mem": m_st})
            mem_diags.append(m_d)
        elif lt == "gqa":
            g_out, g_st = step_sw_gqa_sink(lp["gqa"], ls["gqa"], x, vld_t, rst_t, spec, matmul_dtype=matmul_dtype)
            x = x + scale * g_out
            x = x + scale * apply_ffn(lp["ffn"], x, mod_t, vld_t, spec, matmul_dtype=matmul_dtype)
            new_layers.append({"gqa": g_st})
        elif lt == "v3_dual":
            g_out, g_st = step_sw_gqa_sink(lp["gqa"], ls["gqa"], x, vld_t, rst_t, spec, matmul_dtype=matmul_dtype)
            m_out, m_st, m_d = step_v3_legacy_mem(lp["mem"], ls["mem"], x, mod_t, dt_t, vld_t, rel_t, rst_t, spec, matmul_dtype=matmul_dtype)
            x = x + scale * (g_out + m_out)
            x = x + scale * apply_ffn(lp["ffn"], x, mod_t, vld_t, spec, matmul_dtype=matmul_dtype)
            new_layers.append({"gqa": g_st, "mem": m_st})
            mem_diags.append(m_d)

    z_t = rms_norm(x, core_p["final_norm_w"]) * vld_t[:, None]
    agg_diag = _aggregate_mem_diags(mem_diags, vld_t.shape)
    return z_t, {"layers": new_layers}, agg_diag


def apply_v4_core_seq(
    params: PyTree,
    init_state: PyTree,
    packets_seq: PyTree,
    spec: SLWMv4Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    wy_chunk_size: int = 64,
    gqa_chunk_size: int = 64,
    use_remat: bool = True,
    assume_all_valid: bool = False,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Hoisted sequence-parallel SLWM-v4 forward pass (`[T, B, ...] -> (z_seq[T, B, D], fin_state, diag)`).

    Certified bit-equivalent to iterating `step_v4_core` over `t = 0..T-1` across resets & `valid_mask == 0` holes.
    """
    x = embed_v4_packets_seq(params["edge"], packets_seq, spec, matmul_dtype=matmul_dtype)
    mod_s = packets_seq["modality_id"].astype(jnp.int32)
    dt_s = packets_seq["dt_s"].astype(jnp.float32)
    vld_s = packets_seq["valid_mask"].astype(jnp.float32)
    rel_s = packets_seq["reliability"].astype(jnp.float32)
    rst_s = packets_seq.get("reset_mask", jnp.zeros_like(vld_s, dtype=jnp.bool_)).astype(jnp.bool_)

    scale = spec.depth_residual_scale
    new_layers = []
    mem_diags: list[dict[str, Array]] = []
    core_p = params["core"]

    for i, lt in enumerate(spec.layer_types()):
        lp = core_p["layers"][i]
        ls = init_state["layers"][i]

        def _make_layer_fn(layer_type: str):
            def _layer_fn(l_params: PyTree, l_state: PyTree, x_in: Array):
                if layer_type == "gdn":
                    m_out, m_st, m_d = apply_gdn_v4_seq(
                        l_params["mem"],
                        l_state["mem"],
                        x_in,
                        mod_s,
                        dt_s,
                        vld_s,
                        rel_s,
                        rst_s,
                        spec,
                        matmul_dtype=matmul_dtype,
                        wy_chunk_size=wy_chunk_size,
                        assume_all_valid=assume_all_valid,
                    )
                    x_mid = x_in + scale * m_out
                    x_out = x_mid + scale * apply_ffn(l_params["ffn"], x_mid, mod_s, vld_s, spec, matmul_dtype=matmul_dtype)
                    return x_out, {"mem": m_st}, m_d
                elif layer_type == "gqa":
                    g_out, g_st = apply_sw_gqa_sink_seq(
                        l_params["gqa"],
                        l_state["gqa"],
                        x_in,
                        vld_s,
                        rst_s,
                        spec,
                        matmul_dtype=matmul_dtype,
                        gqa_chunk_size=gqa_chunk_size,
                        assume_all_valid=assume_all_valid,
                    )
                    x_mid = x_in + scale * g_out
                    x_out = x_mid + scale * apply_ffn(l_params["ffn"], x_mid, mod_s, vld_s, spec, matmul_dtype=matmul_dtype)
                    return x_out, {"gqa": g_st}, None
                else:
                    g_out, g_st = apply_sw_gqa_sink_seq(
                        l_params["gqa"],
                        l_state["gqa"],
                        x_in,
                        vld_s,
                        rst_s,
                        spec,
                        matmul_dtype=matmul_dtype,
                        gqa_chunk_size=gqa_chunk_size,
                        assume_all_valid=assume_all_valid,
                    )
                    m_out, m_st, m_d = apply_v3_legacy_mem_seq(
                        l_params["mem"],
                        l_state["mem"],
                        x_in,
                        mod_s,
                        dt_s,
                        vld_s,
                        rel_s,
                        rst_s,
                        spec,
                        matmul_dtype=matmul_dtype,
                    )
                    x_mid = x_in + scale * (g_out + m_out)
                    x_out = x_mid + scale * apply_ffn(l_params["ffn"], x_mid, mod_s, vld_s, spec, matmul_dtype=matmul_dtype)
                    return x_out, {"gqa": g_st, "mem": m_st}, m_d

            return jax.checkpoint(_layer_fn) if use_remat else _layer_fn

        layer_fn = _make_layer_fn(lt)
        x, l_st_new, m_d = layer_fn(lp, ls, x)
        new_layers.append(l_st_new)
        if m_d is not None:
            mem_diags.append(m_d)

    z_seq = rms_norm(x, core_p["final_norm_w"]) * vld_s[..., None]
    agg_diag = _aggregate_mem_diags(mem_diags, vld_s.shape)
    return z_seq, {"layers": new_layers}, agg_diag


def _aggregate_mem_diags(diags: list[dict[str, Array]], lead_shape: tuple[int, ...]) -> dict[str, Array]:
    if not diags:
        z = jnp.zeros(lead_shape, dtype=jnp.float32)
        return {"r_nowrite": z, "novelty": z, "eff_eta": z, "delta_norm": z, "frob_mean": z, "frob_max": z}
    out: dict[str, Array] = {}
    for k in ("r_nowrite", "novelty", "eff_eta", "delta_norm", "frob_mean"):
        out[k] = jnp.mean(jnp.stack([d[k] for d in diags], axis=0), axis=0)
    out["frob_max"] = jnp.max(jnp.stack([d["frob_max"] for d in diags], axis=0), axis=0)
    return out


# ---------------------------------------------------------------------------
# 3. Complete Training Loss (Incl. Diffusion Forcing & Self-Forcing `H3`)
# ---------------------------------------------------------------------------


def _apply_causal_diffusion_forcing_inputs(
    batch: dict[str, Array],
    spec: SLWMv4Spec,
    rng_key: Array,
) -> dict[str, Array]:
    """Corrupt ONLY input `video_feat` with strictly causal per-heartbeat noise `sigma_h` (clean targets preserved!)."""
    k_sig, k_eps = jax.random.split(rng_key, 2)
    t_len, b_sz = batch["modality_id"].shape
    mod_id = batch["modality_id"].astype(jnp.int32)
    sp_pos = batch["spatial_pos"].astype(jnp.int32)
    is_vid = mod_id == MOD_VIDEO
    # Sample a fresh noise level `sigma_h` strictly at heartbeat video start (`sp_pos == 0`) or reset, then forward-fill causally!
    raw_sig = jax.random.uniform(k_sig, (t_len, b_sz), minval=0.0, maxval=float(spec.diffusion_forcing_max_sigma), dtype=jnp.float32)
    is_new_hb = (sp_pos == 0) | batch["reset_mask"].astype(jnp.bool_)

    def _ffill(prev_sig: Array, xs: tuple[Array, Array]) -> tuple[Array, Array]:
        sig_t, new_t = xs
        cur = jnp.where(new_t, sig_t, prev_sig)
        return cur, cur

    _, sig_hb = jax.lax.scan(_ffill, jnp.zeros((b_sz,), dtype=jnp.float32), (raw_sig, is_new_hb))
    sig_vid = jnp.where(is_vid, sig_hb, 0.0)
    eps = jax.random.normal(k_eps, batch["video_feat"].shape, dtype=jnp.float32)
    v_clean = batch["video_feat"].astype(jnp.float32)
    v_noisy = jnp.where(is_vid[..., None], (1.0 - sig_vid[..., None]) * v_clean + sig_vid[..., None] * eps, v_clean)

    in_batch = dict(batch)
    in_batch["video_feat"] = v_noisy
    in_batch["video_noise_sigma"] = sig_vid
    return in_batch


def compute_v4_training_loss(
    params: PyTree,
    batch: dict[str, Array],
    spec: SLWMv4Spec,
    rng_key: Array,
    *,
    init_state: PyTree | None = None,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    wy_chunk_size: int = 64,
    gqa_chunk_size: int = 64,
    use_remat: bool = True,
    assume_all_valid: bool = False,
    loss_weights: dict[str, float] | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute complete SLWM-v4 causal training loss across all modalities + optional `H3` Self-Forcing."""
    t_len, b_sz = batch["modality_id"].shape
    state0 = init_v4_core_state(spec, b_sz) if init_state is None else init_state
    k_df, k_heads, k_sf_n, k_sf_h = jax.random.split(rng_key, 4)

    # 1. Optional causal Diffusion Forcing noise on INPUT video history (targets in `batch` remain 100% clean!)
    in_packets = _apply_causal_diffusion_forcing_inputs(batch, spec, k_df) if spec.use_diffusion_forcing else batch

    z_seq, _, core_diag = apply_v4_core_seq(
        params,
        state0,
        in_packets,
        spec,
        matmul_dtype=matmul_dtype,
        wy_chunk_size=wy_chunk_size,
        gqa_chunk_size=gqa_chunk_size,
        use_remat=use_remat,
        assume_all_valid=assume_all_valid,
    )
    total_loss, metrics = compute_v4_heads_loss(
        params["heads"],
        z_seq,
        batch,
        spec,
        k_heads,
        matmul_dtype=matmul_dtype,
        loss_weights=loss_weights,
    )

    # 2. Optional `H3_self_forcing`: short strictly-causal closed-loop self-fed video rollout consistency!
    sf_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if spec.use_self_forcing and t_len >= 4:
        sf_len = min(64, t_len)
        sub_batch = {k: v[:sf_len] for k, v in batch.items()}
        z_sub_src = jax.lax.stop_gradient(z_seq[: sf_len - 1])
        sp_sub_nxt = sub_batch["spatial_pos"][1:].astype(jnp.int32)
        if spec.video_head_type == "flow":
            x0_sf = jax.random.normal(k_sf_n, (sf_len - 1, b_sz, spec.video_patch_dim), dtype=jnp.float32)
            tau0 = jnp.zeros((sf_len - 1, b_sz), dtype=jnp.float32)
            v0 = apply_video_flow_velocity(params["heads"]["head_video"], z_sub_src, sp_sub_nxt, x0_sf, tau0, spec, matmul_dtype=matmul_dtype)
            pred_v_nxt = jax.lax.stop_gradient(x0_sf + v0)
        else:
            pred_v_nxt = jax.lax.stop_gradient(
                apply_video_huber_pred(params["heads"]["head_video"], z_sub_src, sp_sub_nxt, spec, matmul_dtype=matmul_dtype)
            )
        rst_sub = sub_batch["reset_mask"].astype(jnp.bool_)
        ep_sub = jnp.cumsum(rst_sub.astype(jnp.int32), axis=0)
        can_self_feed = (
            (ep_sub[:-1] == ep_sub[1:])
            & (sub_batch["valid_mask"][:-1].astype(jnp.float32) > 0.5)
            & (sub_batch["modality_id"][1:].astype(jnp.int32) == MOD_VIDEO)
        )
        sf_in = dict(sub_batch)
        v_in_sf = sub_batch["video_feat"].astype(jnp.float32)
        v_in_sf = v_in_sf.at[1:].set(jnp.where(can_self_feed[..., None], pred_v_nxt, v_in_sf[1:]))
        sf_in["video_feat"] = v_in_sf

        z_sf, _, _ = apply_v4_core_seq(
            params,
            state0,
            sf_in,
            spec,
            matmul_dtype=matmul_dtype,
            wy_chunk_size=min(wy_chunk_size, sf_len),
            gqa_chunk_size=min(gqa_chunk_size, sf_len),
            use_remat=use_remat,
            assume_all_valid=assume_all_valid,
        )
        _, sf_metrics = compute_v4_heads_loss(
            params["heads"],
            z_sf,
            sub_batch,
            spec,
            k_sf_h,
            matmul_dtype=matmul_dtype,
            loss_weights=loss_weights,
        )
        sf_has = (sf_metrics["n_video_targets"] > 0.5).astype(jnp.float32)
        sf_loss = sf_metrics["video_loss"] * sf_has
        total_loss = total_loss + 0.15 * sf_loss

    metrics["self_forcing_loss"] = sf_loss
    metrics["total_loss"] = total_loss
    metrics["mem_frob_max"] = jnp.max(core_diag["frob_max"])
    metrics["mem_eff_eta"] = jnp.mean(core_diag["eff_eta"])
    metrics["mem_r_nowrite"] = jnp.mean(core_diag["r_nowrite"])
    return total_loss, metrics


# ---------------------------------------------------------------------------
# 4. Dual-Track Accounting & Base-Shape muP Optimizer
# ---------------------------------------------------------------------------


def summarize_v4_accounting(spec: SLWMv4Spec, params: PyTree | None = None, state: PyTree | None = None) -> dict[str, Any]:
    """Produce complete Dual-Track (`Core Comparison` + `Strict System Cost`) accounting dictionary (`AGENTS.md §4.1`)."""
    est_core_total = estimate_v4_core_param_count(spec)
    est_core_active = estimate_v4_active_core_param_count(spec)
    est_state_b = compute_v4_core_streaming_state_bytes(spec, batch_size=1)

    act_core = count_tree_params(params["core"]) if params is not None else est_core_total
    act_edge = count_tree_params(params["edge"]) if params is not None else None
    act_heads = count_tree_params(params["heads"]) if params is not None else None
    act_total_trainable = count_tree_params(params) if params is not None else None
    act_state_b = count_state_tree_bytes(state) if state is not None else est_state_b["total_core_state_bytes"]
    frozen_total = FROZEN_CODEC_PARAMS_WAN21 + FROZEN_CODEC_PARAMS_MIMI

    return {
        "name": spec.name,
        "backbone_family": spec.backbone_family,
        "core_params_analytic": est_core_total,
        "core_params_active_analytic": est_core_active,
        "core_params_actual": act_core,
        "edge_params_actual": act_edge,
        "heads_params_actual": act_heads,
        "total_trainable_params": act_total_trainable,
        "frozen_codec_params_wan21": FROZEN_CODEC_PARAMS_WAN21,
        "frozen_codec_params_mimi": FROZEN_CODEC_PARAMS_MIMI,
        "frozen_codec_params_total": frozen_total,
        "strict_system_total_params": (act_total_trainable + frozen_total) if act_total_trainable is not None else None,
        "core_state_bytes_b1_analytic": est_state_b["total_core_state_bytes"],
        "core_state_bytes_b1_actual": act_state_b,
        "core_state_mib_b1": round(act_state_b / (1024.0**2), 4),
        "strict_system_state_mib_b1": round((act_state_b + FROZEN_CODEC_STATE_BYTES_BIDIR_BF16) / (1024.0**2), 4),
        "state_breakdown_bytes_b1": est_state_b,
    }


def make_v4_mup_optimizer(
    params: PyTree,
    spec: SLWMv4Spec,
    *,
    peak_lr: float = 1e-3,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    weight_decay: float = 0.05,
    grad_clip_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Create exact base-shape muP AdamW optimizer (`wd = 0` on scalars, `1 / m_d` LR mult on width-scaling matrices)."""
    m_d = max(float(spec.width_mult), 1e-6)
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=peak_lr * 0.01,
        peak_value=peak_lr,
        warmup_steps=max(1, int(warmup_steps)),
        decay_steps=max(2, int(total_steps)),
        end_value=peak_lr * 0.1,
    )

    def _label_tree(tree: Any, prefix: tuple[str, ...] = ()) -> Any:
        if isinstance(tree, dict):
            return {k: _label_tree(v, prefix + (str(k),)) for k, v in tree.items()}
        if isinstance(tree, list):
            return [_label_tree(v, prefix + (str(i),)) for i, v in enumerate(tree)]
        if isinstance(tree, tuple):
            return tuple(_label_tree(v, prefix + (str(i),)) for i, v in enumerate(tree))
        return classify_v4_param_mup_role(prefix, param_ndim=int(tree.ndim) if hasattr(tree, "ndim") else None)

    labels = _label_tree(params)

    def _make_adamw(lr_mult: float, wd: float) -> optax.GradientTransformation:
        return optax.adamw(
            learning_rate=lambda step: lr_sched(step) * lr_mult,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=wd,
        )

    transforms = {
        "scalar": _make_adamw(1.0, 0.0),
        "input": _make_adamw(1.0, weight_decay),
        "fixed_hidden": _make_adamw(1.0, weight_decay),
        "hidden": _make_adamw(1.0 / m_d, weight_decay),
        "readout": _make_adamw(1.0 / m_d, weight_decay),
    }
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.multi_transform(transforms, labels),
    )


# ---------------------------------------------------------------------------
# 5. Live 2+ Minute Flat-O(1) Streaming Heartbeat Rollout Driver
# ---------------------------------------------------------------------------


def step_v4_live_heartbeat(
    params: PyTree,
    state: PyTree,
    z_prev: Array,
    rng_key: Array,
    action_mask: Array,
    proprio_mask: Array,
    is_first_heartbeat: Array,
    spec: SLWMv4Spec,
    *,
    video_flow_nfe: int | None = None,
    audio_temp: float = 0.8,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[Array, PyTree, dict[str, Array]]:
    """Single XLA-fusable 160 ms live autonomous humanoid heartbeat (`4` motor + `2` audio 16-RVQ + `15` Flow-Video + Policy).

    Zero Python host dispatch between the 21 causal packets inside a heartbeat; maintains strictly flat O(1) state!
    """
    b_sz = z_prev.shape[0]
    zero_v = jnp.zeros((b_sz, spec.video_patch_dim), dtype=jnp.float32)
    zero_a = jnp.zeros((b_sz, spec.audio_codebooks), dtype=jnp.int32)
    zero_act = jnp.zeros((b_sz, spec.action_dim), dtype=jnp.float32)
    zero_prop = jnp.zeros((b_sz, spec.proprio_dim), dtype=jnp.float32)
    ones_vld = jnp.ones((b_sz,), dtype=jnp.float32)
    zero_rst = jnp.zeros((b_sz,), dtype=jnp.bool_)
    first_rst = jnp.broadcast_to(is_first_heartbeat.astype(jnp.bool_), (b_sz,))
    zero_sp = jnp.zeros((b_sz,), dtype=jnp.int32)
    zero_tb = jnp.zeros((b_sz,), dtype=jnp.int32)
    act_m = action_mask.reshape(b_sz, spec.action_dim).astype(jnp.float32)
    prop_m = proprio_mask.reshape(b_sz, spec.proprio_dim).astype(jnp.float32)

    def _pkt(mod_val: int, dt_val: Array, rst_val: Array, **kw: Array) -> PyTree:
        p: PyTree = {
            "modality_id": jnp.full((b_sz,), mod_val, dtype=jnp.int32),
            "dt_s": jnp.broadcast_to(dt_val.astype(jnp.float32), (b_sz,)),
            "valid_mask": ones_vld,
            "reliability": ones_vld,
            "reset_mask": rst_val,
            "spatial_pos": zero_sp,
            "text_byte": zero_tb,
            "video_feat": zero_v,
            "audio_codes": zero_a,
            "action_feat": zero_act,
            "proprio_feat": zero_prop,
            "action_mask": act_m,
            "proprio_mask": prop_m,
        }
        p.update(kw)
        return p

    k_a0, k_a1, k_v = jax.random.split(rng_key, 3)
    k_vs = jax.random.split(k_v, spec.video_spatial_packets)
    dt04 = jnp.asarray(0.04, dtype=jnp.float32)
    dt00 = jnp.asarray(0.0, dtype=jnp.float32)

    z = z_prev
    st = state
    acts = []
    props = []

    # 1. MOT[0] @ t0 + 0.04s
    m0 = apply_motor_head(params["heads"]["head_motor"], z, matmul_dtype=matmul_dtype)
    a0, p0 = m0["action_pred"] * act_m, m0["proprio_pred"] * prop_m
    acts.append(a0)
    props.append(p0)
    z, st, _ = step_v4_core(
        params, st, _pkt(MOD_MOTOR, jnp.where(is_first_heartbeat, dt00, dt04), first_rst, action_feat=a0, proprio_feat=p0), spec, matmul_dtype=matmul_dtype
    )

    # 2. MOT[1] @ t0 + 0.08s
    m1 = apply_motor_head(params["heads"]["head_motor"], z, matmul_dtype=matmul_dtype)
    a1, p1 = m1["action_pred"] * act_m, m1["proprio_pred"] * prop_m
    acts.append(a1)
    props.append(p1)
    z, st, _ = step_v4_core(params, st, _pkt(MOD_MOTOR, dt04, zero_rst, action_feat=a1, proprio_feat=p1), spec, matmul_dtype=matmul_dtype)

    # 3. AUD[0] @ t0 + 0.08s
    aud0, ent0 = sample_audio_codes(params["heads"]["head_audio"], z, k_a0, spec, temperature=audio_temp, matmul_dtype=matmul_dtype)
    z, st, _ = step_v4_core(params, st, _pkt(MOD_AUDIO, dt00, zero_rst, audio_codes=aud0), spec, matmul_dtype=matmul_dtype)

    # 4. MOT[2] @ t0 + 0.12s
    m2 = apply_motor_head(params["heads"]["head_motor"], z, matmul_dtype=matmul_dtype)
    a2, p2 = m2["action_pred"] * act_m, m2["proprio_pred"] * prop_m
    acts.append(a2)
    props.append(p2)
    z, st, _ = step_v4_core(params, st, _pkt(MOD_MOTOR, dt04, zero_rst, action_feat=a2, proprio_feat=p2), spec, matmul_dtype=matmul_dtype)

    # 5. MOT[3] @ t0 + 0.16s (strictly BEFORE VID[h] & AUD[1])
    m3 = apply_motor_head(params["heads"]["head_motor"], z, matmul_dtype=matmul_dtype)
    a3, p3 = m3["action_pred"] * act_m, m3["proprio_pred"] * prop_m
    acts.append(a3)
    props.append(p3)
    z, st, _ = step_v4_core(params, st, _pkt(MOD_MOTOR, dt04, zero_rst, action_feat=a3, proprio_feat=p3), spec, matmul_dtype=matmul_dtype)

    # 6. VID[h]: 15 Wan 2.1 patches @ t0 + 0.16s via `jax.lax.scan` (zero Python unroll bloat!)
    def _vid_patch_step(carry: tuple[Array, PyTree], xs: tuple[Array, Array]) -> tuple[tuple[Array, PyTree], Array]:
        zc, stc = carry
        sp_i, kp_i = xs
        sp_b = jnp.full((b_sz,), sp_i, dtype=jnp.int32)
        vp, _ = sample_video_patch(params["heads"]["head_video"], zc, sp_b, kp_i, spec, nfe=video_flow_nfe, matmul_dtype=matmul_dtype)
        zn, stn, _ = step_v4_core(params, stc, _pkt(MOD_VIDEO, dt00, zero_rst, video_feat=vp, spatial_pos=sp_b), spec, matmul_dtype=matmul_dtype)
        return (zn, stn), vp

    (z, st), vid_patches_15 = jax.lax.scan(
        _vid_patch_step, (z, st), (jnp.arange(spec.video_spatial_packets, dtype=jnp.int32), k_vs)
    )  # [15, B, 1024]

    # 7. AUD[1] @ t0 + 0.16s (0-lag contact audio conditioned on VID[h]!)
    aud1, ent1 = sample_audio_codes(params["heads"]["head_audio"], z, k_a1, spec, temperature=audio_temp, matmul_dtype=matmul_dtype)
    z, st, _ = step_v4_core(params, st, _pkt(MOD_AUDIO, dt00, zero_rst, audio_codes=aud1), spec, matmul_dtype=matmul_dtype)

    # 8. PolicyCommitGate (strictly stop_gradient; exact post-step train/inference 5-feature parity!)
    if spec.use_policy_commit_gate and "policy_gate" in params["heads"]:
        txt_lg = apply_text_head(params["heads"]["head_text"], z, matmul_dtype=matmul_dtype)
        txt_p = jax.nn.softmax(txt_lg, axis=-1)
        txt_ent = -jnp.sum(txt_p * jnp.log(jnp.clip(txt_p, 1e-9, 1.0)), axis=-1)
        aud_ent_post = compute_audio_rvq0_entropy(params["heads"]["head_audio"], z, spec, matmul_dtype=matmul_dtype)
        z_rms = jnp.sqrt(jnp.mean(z * z, axis=-1) + 1e-6)
        ev = jnp.stack(
            [
                txt_ent / math.log(float(spec.text_vocab_size)),
                aud_ent_post / math.log(float(spec.audio_vocab_size)),
                z_rms,
                ones_vld,
                ones_vld,
            ],
            axis=-1,
        )
        _, pol_probs = apply_policy_commit_gate(params["heads"]["policy_gate"], z, ev, matmul_dtype=jnp.float32)
    else:
        pol_probs = jnp.full((b_sz, len(COMMIT_ACTIONS)), 1.0 / len(COMMIT_ACTIONS), dtype=jnp.float32)

    out = {
        "video_patches": jnp.swapaxes(vid_patches_15, 0, 1),  # [B, 15, 1024]
        "audio_codes": jnp.stack([aud0, aud1], axis=1),       # [B, 2, 16]
        "action_seq": jnp.stack(acts, axis=1),                # [B, 4, 52]
        "proprio_seq": jnp.stack(props, axis=1),              # [B, 4, 68]
        "policy_probs": pol_probs,                            # [B, 5]
        "rvq0_ent": 0.5 * (ent0 + ent1),
    }
    return z, st, out


def run_v4_live_heartbeat_rollout(
    params: PyTree,
    spec: SLWMv4Spec,
    rng_key: Array,
    *,
    n_heartbeats: int = 750,
    generate_video: bool = True,
    generate_audio: bool = True,
    generate_motor: bool = True,
    obs_video: Array | None = None,
    obs_audio: Array | None = None,
    obs_action: Array | None = None,
    obs_proprio: Array | None = None,
    action_mask: Array | None = None,
    proprio_mask: Array | None = None,
    video_flow_nfe: int | None = None,
    audio_temp: float = 0.8,
    use_policy_gate: bool = True,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, Any]:
    """Execute continuous `n_heartbeats` (`750` = `120.0 s`) live streaming rollout strictly in O(1) core state.

    Follows the exact Gate-1/Gate-2 `160 ms` causal heartbeat order:
      `MOT[0] (0.04s) -> MOT[1] (0.08s) -> AUD[0] (0.08s) -> MOT[2] (0.12s) -> MOT[3] (0.16s) -> VID[0..14] (0.16s) -> AUD[1] (0.16s)`
    with frozen model weights and explicit `AGENTS.md §13.2` epistemic provenance labels
    (`observed`, `predicted`, `imagined`, `abstained`).
    """
    b_sz = 1
    state = init_v4_core_state(spec, batch_size=b_sz)
    init_state_bytes = count_state_tree_bytes(state)

    act_m = jnp.ones((b_sz, spec.action_dim), dtype=jnp.float32) if action_mask is None else action_mask.reshape(b_sz, spec.action_dim).astype(jnp.float32)
    prop_m = jnp.ones((b_sz, spec.proprio_dim), dtype=jnp.float32) if proprio_mask is None else proprio_mask.reshape(b_sz, spec.proprio_dim).astype(jnp.float32)

    @jax.jit
    def _step_core_jit(st: PyTree, pkt: PyTree) -> tuple[Array, PyTree, dict[str, Array]]:
        return step_v4_core(params, st, pkt, spec, matmul_dtype=matmul_dtype)

    @jax.jit
    def _sample_vid_jit(z: Array, sp: Array, k: Array) -> tuple[Array, Array]:
        return sample_video_patch(params["heads"]["head_video"], z, sp, k, spec, nfe=video_flow_nfe, matmul_dtype=matmul_dtype)

    @jax.jit
    def _sample_aud_jit(z: Array, k: Array) -> tuple[Array, Array]:
        return sample_audio_codes(params["heads"]["head_audio"], z, k, spec, temperature=audio_temp, matmul_dtype=matmul_dtype)

    @jax.jit
    def _pred_mot_jit(z: Array) -> dict[str, Array]:
        return apply_motor_head(params["heads"]["head_motor"], z, matmul_dtype=matmul_dtype)

    @jax.jit
    def _eval_policy_from_z_jit(z: Array) -> tuple[Array, Array]:
        txt_lg = apply_text_head(params["heads"]["head_text"], z, matmul_dtype=matmul_dtype)
        txt_p = jax.nn.softmax(txt_lg, axis=-1)
        txt_ent = -jnp.sum(txt_p * jnp.log(jnp.clip(txt_p, 1e-9, 1.0)), axis=-1)
        aud_ent0 = compute_audio_rvq0_entropy(params["heads"]["head_audio"], z, spec, matmul_dtype=matmul_dtype)
        z_rms = jnp.sqrt(jnp.mean(z * z, axis=-1) + 1e-6)
        ev = jnp.stack(
            [
                txt_ent / math.log(float(spec.text_vocab_size)),
                aud_ent0 / math.log(float(spec.audio_vocab_size)),
                z_rms,
                jnp.ones((z.shape[0],), dtype=jnp.float32),
                jnp.ones((z.shape[0],), dtype=jnp.float32),
            ],
            axis=-1,
        )
        return apply_policy_commit_gate(params["heads"]["policy_gate"], z, ev, matmul_dtype=jnp.float32)

    zero_v = jnp.zeros((b_sz, spec.video_patch_dim), dtype=jnp.float32)
    zero_a = jnp.zeros((b_sz, spec.audio_codebooks), dtype=jnp.int32)
    zero_act = jnp.zeros((b_sz, spec.action_dim), dtype=jnp.float32)
    zero_prop = jnp.zeros((b_sz, spec.proprio_dim), dtype=jnp.float32)
    ones_vld = jnp.ones((b_sz,), dtype=jnp.float32)
    zero_rst = jnp.zeros((b_sz,), dtype=jnp.bool_)
    one_rst = jnp.ones((b_sz,), dtype=jnp.bool_)
    zero_sp = jnp.zeros((b_sz,), dtype=jnp.int32)
    zero_tb = jnp.zeros((b_sz,), dtype=jnp.int32)

    def _make_pkt(mod_val: int, dt_val: float, is_first: bool, **kwargs: Array) -> PyTree:
        pkt: PyTree = {
            "modality_id": jnp.full((b_sz,), mod_val, dtype=jnp.int32),
            "dt_s": jnp.full((b_sz,), dt_val, dtype=jnp.float32),
            "valid_mask": ones_vld,
            "reliability": ones_vld,
            "reset_mask": one_rst if is_first else zero_rst,
            "spatial_pos": zero_sp,
            "text_byte": zero_tb,
            "video_feat": zero_v,
            "audio_codes": zero_a,
            "action_feat": zero_act,
            "proprio_feat": zero_prop,
            "action_mask": act_m,
            "proprio_mask": prop_m,
        }
        pkt.update(kwargs)
        return pkt

    n_obs_v = int(obs_video.shape[0]) if obs_video is not None else 0
    n_obs_a = int(obs_audio.shape[0]) if obs_audio is not None else 0
    n_obs_m = int(obs_action.shape[0]) if obs_action is not None else 0

    out_vid = []
    out_aud = []
    out_act = []
    out_prop = []
    epistemic_vid: list[str] = []
    epistemic_aud: list[str] = []
    epistemic_mot: list[str] = []
    policy_actions: list[str] = []
    state_bytes_trace: list[int] = [init_state_bytes]

    z_cur = jnp.zeros((b_sz, spec.d_model), dtype=jnp.float32)
    first_pkt = True
    cur_key = rng_key
    last_rvq0_ent = jnp.full((b_sz,), 0.5, dtype=jnp.float32)
    last_vid_vn = jnp.full((b_sz,), 0.5, dtype=jnp.float32)

    for h in range(n_heartbeats):
        # 4 motor ticks + 2 audio ticks + 15 video patches per 160 ms heartbeat
        for m_off, dt_m in ((0, 0.04), (1, 0.04), (2, 0.04), (3, 0.04)):
            m_idx = h * 4 + m_off
            if generate_motor:
                if obs_action is not None and m_idx < n_obs_m:
                    a_vec = obs_action[m_idx : m_idx + 1]
                    p_vec = obs_proprio[m_idx : m_idx + 1] if obs_proprio is not None else zero_prop
                    epistemic_mot.append("observed")
                else:
                    m_pred = _pred_mot_jit(z_cur)
                    a_vec = m_pred["action_pred"] * act_m
                    p_vec = m_pred["proprio_pred"] * prop_m
                    epistemic_mot.append("predicted" if (h < max(n_obs_v, n_obs_a // 2)) else "imagined")
                out_act.append(a_vec[0])
                out_prop.append(p_vec[0])
                z_cur, state, _ = _step_core_jit(
                    state, _make_pkt(MOD_MOTOR, 0.0 if first_pkt else dt_m, first_pkt, action_feat=a_vec, proprio_feat=p_vec)
                )
                first_pkt = False

            # AUD[2h + 0] right after MOT[1] at t0 + 0.08 s
            if m_off == 1 and generate_audio:
                a_idx = h * 2
                if obs_audio is not None and a_idx < n_obs_a:
                    a_codes = obs_audio[a_idx : a_idx + 1].astype(jnp.int32)
                    epistemic_aud.append("observed")
                else:
                    cur_key, ka = jax.random.split(cur_key)
                    a_codes, last_rvq0_ent = _sample_aud_jit(z_cur, ka)
                    epistemic_aud.append("predicted" if (h < n_obs_v) else "imagined")
                out_aud.append(a_codes[0])
                dt_a0 = 0.0 if generate_motor else (0.0 if first_pkt else 0.08)
                z_cur, state, _ = _step_core_jit(state, _make_pkt(MOD_AUDIO, dt_a0, first_pkt, audio_codes=a_codes))
                first_pkt = False

        # VID[h]: 15 Wan 2.1 spatial patches at t0 + 0.16 s
        if generate_video:
            vid_patches_h = []
            if obs_video is not None and h < n_obs_v:
                epistemic_vid.append("observed")
                for p in range(spec.video_spatial_packets):
                    vp = obs_video[h, p : p + 1].astype(jnp.float32)
                    vid_patches_h.append(vp[0])
                    dt_v = 0.0 if (generate_motor or p > 0) else (0.0 if first_pkt else 0.08)
                    z_cur, state, _ = _step_core_jit(
                        state, _make_pkt(MOD_VIDEO, dt_v, first_pkt, video_feat=vp, spatial_pos=jnp.full((b_sz,), p, dtype=jnp.int32))
                    )
                    first_pkt = False
            else:
                epistemic_vid.append("predicted" if (obs_audio is not None and (h * 2 + 1) < n_obs_a) else "imagined")
                for p in range(spec.video_spatial_packets):
                    cur_key, kv = jax.random.split(cur_key)
                    sp_arr = jnp.full((b_sz,), p, dtype=jnp.int32)
                    vp, last_vid_vn = _sample_vid_jit(z_cur, sp_arr, kv)
                    vid_patches_h.append(vp[0])
                    dt_v = 0.0 if (generate_motor or p > 0) else (0.0 if first_pkt else 0.08)
                    z_cur, state, _ = _step_core_jit(state, _make_pkt(MOD_VIDEO, dt_v, first_pkt, video_feat=vp, spatial_pos=sp_arr))
                    first_pkt = False
            out_vid.append(jnp.stack(vid_patches_h, axis=0))

        # AUD[2h + 1] immediately after VID[h] at t0 + 0.16 s (0-lag contact audio!)
        if generate_audio:
            a_idx = h * 2 + 1
            if obs_audio is not None and a_idx < n_obs_a:
                a_codes = obs_audio[a_idx : a_idx + 1].astype(jnp.int32)
                epistemic_aud.append("observed")
            else:
                cur_key, ka = jax.random.split(cur_key)
                a_codes, last_rvq0_ent = _sample_aud_jit(z_cur, ka)
                epistemic_aud.append("predicted" if (h < n_obs_v) else "imagined")
            out_aud.append(a_codes[0])
            dt_a1 = 0.0 if (generate_video or generate_motor) else (0.0 if first_pkt else 0.08)
            z_cur, state, _ = _step_core_jit(state, _make_pkt(MOD_AUDIO, dt_a1, first_pkt, audio_codes=a_codes))
            first_pkt = False

        # PolicyCommitGate decision at heartbeat boundary (exact post-step train/inference 5-feature parity!)
        if use_policy_gate and spec.use_policy_commit_gate and "policy_gate" in params["heads"]:
            _, probs = _eval_policy_from_z_jit(z_cur)
            act_i = int(jnp.argmax(probs[0]))
            act_name = COMMIT_ACTIONS[act_i]
            policy_actions.append(act_name)
            if act_name == "ABSTAIN_SILENCE" and epistemic_aud and epistemic_aud[-1] != "observed":
                epistemic_aud[-1] = "abstained"

        state_bytes_trace.append(count_state_tree_bytes(state))

    return {
        "n_heartbeats": n_heartbeats,
        "duration_sec": round(n_heartbeats * 0.16, 2),
        "video_groups": jnp.stack(out_vid, axis=0) if out_vid else None,
        "audio_codes": jnp.stack(out_aud, axis=0) if out_aud else None,
        "action_seq": jnp.stack(out_act, axis=0) if out_act else None,
        "proprio_seq": jnp.stack(out_prop, axis=0) if out_prop else None,
        "epistemic_labels": {
            "video": epistemic_vid,
            "audio": epistemic_aud,
            "motor": epistemic_mot,
        },
        "policy_actions": policy_actions,
        "init_state_bytes": init_state_bytes,
        "final_state_bytes": state_bytes_trace[-1],
        "state_bytes_flat_o1": all(b == init_state_bytes for b in state_bytes_trace),
    }
