"""SLWM-v3 Shared World Core (Rev 2: Chunked Remat, Balanced Edge RMS, Exact Base-Shape muP & Masked WD).

Architecture contract (`AGENTS.md §3` & `slwm_v3_wave_a_gate1_spec.md`):
  - Separable edge adapters (`video`, `audio`, `text_byte`, `proprio_action`, `signal_probe`) balanced to
    matched initial activation RMS (~0.5) across all 5 modalities, with ZERO dead edge params across
    `frozen_codec_edge` vs `scratch_edge` arms and ZERO hardcoded task opcodes.
  - Explicit 2-way `has_true_action` conditioning separating executed zero action from missing action.
  - Unified L-layer stack combining O(1) ring-buffer GQA, dispatched selective memory branch, and SwiGLU FFN
    with CompleteP `alpha=1.0` residual branch scaling (`L_base / L`).
  - Chunked `jax.checkpoint` rematerialization (`chunk_size = 64`) cutting long-stream carry HBM by 64x while
    preserving exact mathematical equivalence to `step_core`.
  - Exact base-shape muP LR multipliers (`1.0` on every leaf at `d == d_base`, `1 / m_d` on width-scaling
    hidden/readout 2D weights) and strict weight-decay mask excluding all 1D/scalar/timescale parameters.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from models.v3.memory import (
    classify_param_mup_role,
    init_diag_ssm_params,
    init_diag_ssm_state,
    init_explicit_kv_params,
    init_explicit_kv_state,
    init_matrix_memory_params,
    init_matrix_memory_state,
    init_mup_weight,
    init_ring_gqa_params,
    init_ring_gqa_state,
    rms_norm,
    step_diag_ssm,
    step_explicit_kv,
    step_matrix_memory,
    step_ring_gqa,
)
from models.v3.spec import MATRIX_MEMORY_BACKBONES, SLWMv3Spec

Array = jax.Array
PyTree = dict[str, Any]


# ---------------------------------------------------------------------------
# Edge Adapters (Balanced Init RMS Across All 5 Modalities; Zero Dead Params)
# ---------------------------------------------------------------------------


def init_edge_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    """Initialize modality-specific edge projections matched to ~0.5 initial RMS across all 5 modalities."""
    keys = jax.random.split(key, 12)
    d = spec.d_model
    m_d = spec.width_mult
    audio_sub_d = 32
    edge_rms = 0.5  # Common initial activation RMS for video, audio, text, proprio, and probe!

    params: PyTree = {
        "edge_mod_emb": init_mup_weight(keys[0], (spec.n_modalities, d), role="input", width_mult=m_d, base_std=0.1),
        "edge_spatial_pos_emb": init_mup_weight(
            keys[1], (spec.video_spatial_packets, d), role="input", width_mult=m_d, base_std=0.1
        ),
        "edge_video_w": init_mup_weight(
            keys[2], (spec.video_patch_dim, d), role="input", width_mult=m_d, base_std=edge_rms / math.sqrt(spec.video_patch_dim)
        ),
        "edge_video_bias": jnp.zeros((d,), dtype=jnp.float32),
        "edge_audio_bias": jnp.zeros((d,), dtype=jnp.float32),
        "edge_text_emb": init_mup_weight(keys[6], (spec.text_vocab_size, d), role="input", width_mult=m_d, base_std=edge_rms),
        "edge_proprio_w": init_mup_weight(
            keys[7], (spec.proprio_dim, d), role="input", width_mult=m_d, base_std=edge_rms / math.sqrt(spec.proprio_dim)
        ),
        "edge_action_w": init_mup_weight(
            keys[8], (spec.action_dim, d), role="input", width_mult=m_d, base_std=edge_rms / math.sqrt(spec.action_dim)
        ),
        "edge_action_avail_emb": init_mup_weight(keys[9], (2, d), role="input", width_mult=m_d, base_std=0.1),
        "edge_probe_w": init_mup_weight(
            keys[10], (spec.probe_dim, d), role="input", width_mult=m_d, base_std=edge_rms / math.sqrt(spec.probe_dim)
        ),
        "edge_probe_bias": jnp.zeros((d,), dtype=jnp.float32),
        "edge_meta_w": init_mup_weight(keys[11], (2, d), role="input", width_mult=m_d, base_std=0.05),
    }

    # Allocate ONLY the active audio edge path for the chosen representation_mode (zero dead parameters!)
    if spec.representation_mode == "frozen_codec_edge":
        params["edge_audio_rvq_emb"] = init_mup_weight(
            keys[3], (spec.audio_codebooks, spec.audio_vocab_size, audio_sub_d), role="input", width_mult=m_d, base_std=1.0
        )
        params["edge_audio_rvq_w"] = init_mup_weight(
            keys[4],
            (spec.audio_codebooks * audio_sub_d, d),
            role="input",
            width_mult=m_d,
            base_std=edge_rms / math.sqrt(spec.audio_codebooks * audio_sub_d),
        )
    else:
        params["edge_audio_cont_w"] = init_mup_weight(
            keys[5], (spec.audio_cont_dim, d), role="input", width_mult=m_d, base_std=edge_rms / math.sqrt(spec.audio_cont_dim)
        )
    return params


def embed_packet_step(edge_params: PyTree, packet_t: PyTree, spec: SLWMv3Spec) -> Array:
    """Embed one timestep's multimodal packet into shared latent vector `x_t: float32[B, D]`.

    Supported `modality_id` values:
      - `0`: video latent patch (`video_feat: [B, video_patch_dim]`, `spatial_pos: int32[B]` in `0..14`)
      - `1`: audio packet (`audio_codes: int32[B, 32]` on `frozen_codec_edge`, `audio_cont: [B, 64]` on `scratch_edge`)
      - `2`: lossless ByT5 UTF-8 byte (`text_byte: int32[B]` in `0..263`)
      - `3`: proprio/action state (`proprio_feat`, `action_feat`, `has_true_action`)
      - `4`: continuous signal / counterfactual probe (`probe_feat: [B, probe_dim]`)

    Missing packets (`valid_mask == 0`) map to exact zero vector.
    """
    mod_id = jnp.clip(packet_t["modality_id"].astype(jnp.int32), 0, spec.n_modalities - 1)
    b = mod_id.shape[0]
    valid = packet_t["valid_mask"].astype(jnp.float32)
    dt_s = jnp.maximum(packet_t["dt_s"].astype(jnp.float32), 0.0)
    rel = packet_t["reliability"].astype(jnp.float32)

    v_feat = packet_t.get("video_feat", jnp.zeros((b, spec.video_patch_dim), dtype=jnp.float32)).astype(jnp.float32)
    sp_pos = jnp.clip(
        packet_t.get("spatial_pos", jnp.zeros((b,), dtype=jnp.int32)).astype(jnp.int32),
        0,
        spec.video_spatial_packets - 1,
    )
    x_vid = jnp.matmul(v_feat, edge_params["edge_video_w"]) + edge_params["edge_video_bias"] + edge_params["edge_spatial_pos_emb"][sp_pos]

    if spec.representation_mode == "frozen_codec_edge":
        codes = jnp.clip(
            packet_t.get("audio_codes", jnp.zeros((b, spec.audio_codebooks), dtype=jnp.int32)).astype(jnp.int32),
            0,
            spec.audio_vocab_size - 1,
        )
        cb_idx = jnp.arange(spec.audio_codebooks, dtype=jnp.int32)[None, :]
        rvq_e = edge_params["edge_audio_rvq_emb"][cb_idx, codes]
        x_aud = jnp.matmul(rvq_e.reshape(b, -1), edge_params["edge_audio_rvq_w"]) + edge_params["edge_audio_bias"]
    else:
        a_cont = packet_t.get("audio_cont", jnp.zeros((b, spec.audio_cont_dim), dtype=jnp.float32)).astype(jnp.float32)
        x_aud = jnp.matmul(a_cont, edge_params["edge_audio_cont_w"]) + edge_params["edge_audio_bias"]

    t_byte = jnp.clip(packet_t.get("text_byte", jnp.zeros((b,), dtype=jnp.int32)).astype(jnp.int32), 0, spec.text_vocab_size - 1)
    x_txt = edge_params["edge_text_emb"][t_byte]

    p_feat = packet_t.get("proprio_feat", jnp.zeros((b, spec.proprio_dim), dtype=jnp.float32)).astype(jnp.float32)
    act_feat = packet_t.get("action_feat", jnp.zeros((b, spec.action_dim), dtype=jnp.float32)).astype(jnp.float32)
    has_act = packet_t.get("has_true_action", jnp.zeros((b,), dtype=jnp.float32)).astype(jnp.float32) > 0.5
    act_avail_e = edge_params["edge_action_avail_emb"][has_act.astype(jnp.int32)]
    x_prop = (
        jnp.matmul(p_feat, edge_params["edge_proprio_w"])
        + jnp.matmul(act_feat, edge_params["edge_action_w"]) * has_act.astype(jnp.float32)[:, None]
        + act_avail_e
    )

    pr_feat = packet_t.get("probe_feat", jnp.zeros((b, spec.probe_dim), dtype=jnp.float32)).astype(jnp.float32)
    x_prb = jnp.matmul(pr_feat, edge_params["edge_probe_w"]) + edge_params["edge_probe_bias"]

    stacked = jnp.stack([x_vid, x_aud, x_txt, x_prop, x_prb], axis=1)
    x_sel = stacked[jnp.arange(b), mod_id]

    meta_feat = jnp.stack([jnp.log1p(dt_s), rel], axis=-1)
    x_out = x_sel + edge_params["edge_mod_emb"][mod_id] + jnp.matmul(meta_feat, edge_params["edge_meta_w"])
    return (x_out * valid[:, None]).astype(jnp.float32)


# ---------------------------------------------------------------------------
# SwiGLU FFN & Unified SLWM-v3 Layer
# ---------------------------------------------------------------------------


def init_ffn_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    k1, k2, k3 = jax.random.split(key, 3)
    d = spec.d_model
    ff = spec.ffn_dim
    m_d = spec.width_mult
    return {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w_gate": init_mup_weight(k1, (d, ff), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_up": init_mup_weight(k2, (d, ff), role="hidden", width_mult=m_d, base_std=spec.init_std),
        "w_down": init_mup_weight(k3, (ff, d), role="hidden", width_mult=m_d, base_std=spec.init_std),
    }


def apply_ffn(params: PyTree, x_t: Array, valid_mask_t: Array) -> Array:
    xn = rms_norm(x_t, params["norm_w"])
    h = jax.nn.silu(jnp.matmul(xn, params["w_gate"])) * jnp.matmul(xn, params["w_up"])
    return jnp.matmul(h, params["w_down"]) * valid_mask_t[:, None]


def init_layer_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    k_attn, k_mem, k_ffn = jax.random.split(key, 3)
    layer: PyTree = {
        "attn": init_ring_gqa_params(k_attn, spec),
        "ffn": init_ffn_params(k_ffn, spec),
    }
    if spec.backbone_type in MATRIX_MEMORY_BACKBONES:
        layer["mem"] = init_matrix_memory_params(k_mem, spec)
    elif spec.backbone_type == "diag_ssm":
        layer["ssm"] = init_diag_ssm_params(k_mem, spec)
    elif spec.backbone_type == "explicit_bounded_kv":
        layer["kv"] = init_explicit_kv_params(k_mem, spec)
    elif spec.backbone_type == "sliding_window_gqa":
        pass
    else:
        raise ValueError(f"Unknown backbone_type={spec.backbone_type}")
    return layer


def init_layer_state(spec: SLWMv3Spec, batch_size: int) -> PyTree:
    st: PyTree = {"attn": init_ring_gqa_state(spec, batch_size)}
    if spec.backbone_type in MATRIX_MEMORY_BACKBONES:
        st["mem"] = init_matrix_memory_state(spec, batch_size)
    elif spec.backbone_type == "diag_ssm":
        st["ssm"] = init_diag_ssm_state(spec, batch_size)
    elif spec.backbone_type == "explicit_bounded_kv":
        st["kv"] = init_explicit_kv_state(spec, batch_size)
    return st


# ---------------------------------------------------------------------------
# Full SLWM-v3 Core Initialization, Reset, Single-Step & Chunked-Remat Scan
# ---------------------------------------------------------------------------


def init_slwm_v3_params(key: Array, spec: SLWMv3Spec) -> PyTree:
    """Initialize full SLWM-v3 core + single multi-horizon predictive head under exact base-shape muP."""
    keys = jax.random.split(key, spec.n_layers + 5)
    d = spec.d_model
    m_d = spec.width_mult
    n_h = len(spec.pred_horizons)

    layers = [init_layer_params(keys[i], spec) for i in range(spec.n_layers)]
    params: PyTree = {
        "edge": init_edge_params(keys[spec.n_layers], spec),
        "layers": layers,
        "final_norm_w": jnp.ones((d,), dtype=jnp.float32),
        "pred_head_action_w": init_mup_weight(
            keys[spec.n_layers + 1], (spec.action_dim, d), role="input", width_mult=m_d, base_std=0.5 / math.sqrt(spec.action_dim)
        ),
        "pred_head_action_avail_emb": init_mup_weight(
            keys[spec.n_layers + 2], (2, d), role="input", width_mult=m_d, base_std=0.1
        ),
        "pred_head_w": init_mup_weight(
            keys[spec.n_layers + 3],
            (d, n_h * spec.target_feat_dim),
            role="readout",
            width_mult=m_d,
            base_std=spec.init_std,
            zero_init=True,
        ),
        "pred_head_bias": jnp.zeros((n_h * spec.target_feat_dim,), dtype=jnp.float32),
    }
    if spec.use_zero_lag_residual:
        params["zero_lag_proj"] = init_mup_weight(
            keys[spec.n_layers + 4], (d, d), role="hidden", width_mult=m_d, base_std=spec.init_std
        )
        params["gate_z0"] = jnp.zeros((d,), dtype=jnp.float32)
    return params


def init_core_state(spec: SLWMv3Spec, batch_size: int) -> list[PyTree]:
    """Allocate fresh zeroed O(1) persistent streaming state across all L layers."""
    return [init_layer_state(spec, batch_size) for _ in range(spec.n_layers)]


def reset_core_state_rows(state: list[PyTree], reset_mask: Array, spec: SLWMv3Spec) -> list[PyTree]:
    """Reset persistent streaming state bit-exactly for batch rows where `reset_mask == True`."""
    b = reset_mask.shape[0]
    fresh = init_core_state(spec, b)
    rm = reset_mask.astype(jnp.bool_)

    def _select_leaf(f_arr: Array, cur_arr: Array) -> Array:
        shape = (b,) + (1,) * (cur_arr.ndim - 1)
        return jnp.where(rm.reshape(shape), f_arr, cur_arr)

    out_state = jax.tree_util.tree_map(_select_leaf, fresh, state)
    # For Chunk-Anchor Surprise WY (`surprise_chunk_size > 1`), `chunk_phase` is purely the global stream clock
    # `t mod C` (zero data from prior episodes; `M` and `M_anchor` ARE zeroed bit-exactly!), preserving bit-exact
    # equivalence with fixed-grid WY chunks across arbitrary mid-chunk episode resets!
    if spec.backbone_type in MATRIX_MEMORY_BACKBONES and spec.use_surprise_router and spec.surprise_chunk_size > 1:
        for li in range(spec.n_layers):
            out_state[li]["mem"]["chunk_phase"] = state[li]["mem"]["chunk_phase"]
    return out_state


def step_core(
    params: PyTree,
    state: list[PyTree],
    packet_t: PyTree,
    spec: SLWMv3Spec,
) -> tuple[Array, list[PyTree], dict[str, Array]]:
    """Execute ONE strictly causal streaming step: `(state_{t-1}, packet_t) -> (Z_t, state_t, diags)`."""
    if "reset_mask" in packet_t:
        state = reset_core_state_rows(state, packet_t["reset_mask"], spec)

    x_raw = embed_packet_step(params["edge"], packet_t, spec)
    mod_id = packet_t["modality_id"].astype(jnp.int32)
    dt_s = packet_t["dt_s"].astype(jnp.float32)
    valid_mask = packet_t["valid_mask"].astype(jnp.float32)
    reliability = packet_t["reliability"].astype(jnp.float32)

    alpha_l = spec.depth_residual_scale
    h = x_raw
    new_states: list[PyTree] = []
    nowrite_list: list[Array] = []
    eta_list: list[Array] = []
    decay_list: list[Array] = []
    delta_norm_list: list[Array] = []
    frob_norm_list: list[Array] = []
    frob_max_list: list[Array] = []
    mem_rms_list: list[Array] = []
    prior_err_list: list[Array] = []
    key_norm_list: list[Array] = []
    layer_rms_list: list[Array] = []

    for li in range(spec.n_layers):
        lp = params["layers"][li]
        ls = state[li]
        ns: PyTree = {}

        attn_out, ns["attn"] = step_ring_gqa(lp["attn"], ls["attn"], h, valid_mask, spec)
        h = h + alpha_l * attn_out

        if spec.backbone_type in MATRIX_MEMORY_BACKBONES:
            mem_out, ns["mem"], m_diag = step_matrix_memory(
                lp["mem"], ls["mem"], h, mod_id, dt_s, valid_mask, reliability, spec, step_idx_t=ls["attn"]["step_idx"]
            )
            h = h + alpha_l * mem_out
            nowrite_list.append(m_diag["r_nowrite"])
            eta_list.append(m_diag["eff_write_eta"])
            decay_list.append(m_diag["eff_decay"])
            delta_norm_list.append(m_diag["state_delta_norm"])
            frob_norm_list.append(m_diag["mem_frob_norm"])
            frob_max_list.append(m_diag["mem_frob_norm_max"])
            mem_rms_list.append(m_diag["mem_out_rms"])
            prior_err_list.append(m_diag["prior_error"])
            key_norm_list.append(m_diag["key_norm"])
        elif spec.backbone_type == "diag_ssm":
            ssm_out, ns["ssm"], m_diag = step_diag_ssm(lp["ssm"], ls["ssm"], h, dt_s, valid_mask, spec)
            h = h + alpha_l * ssm_out
            mem_rms_list.append(jnp.sqrt(jnp.mean(ssm_out * ssm_out, axis=-1) + 1e-12))
            prior_err_list.append(m_diag["prior_error"])
        elif spec.backbone_type == "explicit_bounded_kv":
            kv_out, ns["kv"], m_diag = step_explicit_kv(
                lp["kv"], ls["kv"], h, mod_id, dt_s, valid_mask, reliability, spec
            )
            h = h + alpha_l * kv_out
            nowrite_list.append(m_diag["r_nowrite"])
            mem_rms_list.append(jnp.sqrt(jnp.mean(kv_out * kv_out, axis=-1) + 1e-12))
            prior_err_list.append(m_diag["prior_error"])

        ffn_out = apply_ffn(lp["ffn"], h, valid_mask)
        h = h + alpha_l * ffn_out

        layer_rms_list.append(jnp.sqrt(jnp.mean(h * h, axis=-1) + 1e-8))
        new_states.append(ns)

    z_t = rms_norm(h, params["final_norm_w"])
    if spec.use_zero_lag_residual:
        z_t = z_t + jax.nn.sigmoid(params["gate_z0"])[None, :] * jnp.matmul(x_raw, params["zero_lag_proj"])

    b = x_raw.shape[0]
    diags = {
        "layer_rms": jnp.stack(layer_rms_list, axis=-1),
        "mean_nowrite": jnp.mean(jnp.stack(nowrite_list, axis=0), axis=0) if nowrite_list else jnp.zeros((b,), dtype=jnp.float32),
        "mean_eff_write_eta": jnp.mean(jnp.stack(eta_list, axis=0), axis=0) if eta_list else jnp.zeros((b,), dtype=jnp.float32),
        "mean_eff_decay": jnp.mean(jnp.stack(decay_list, axis=0), axis=0) if decay_list else jnp.ones((b,), dtype=jnp.float32),
        "mean_state_delta_norm": jnp.mean(jnp.stack(delta_norm_list, axis=0), axis=0)
        if delta_norm_list
        else jnp.zeros((b,), dtype=jnp.float32),
        "mean_mem_frob_norm": jnp.mean(jnp.stack(frob_norm_list, axis=0), axis=0)
        if frob_norm_list
        else jnp.zeros((b,), dtype=jnp.float32),
        "max_mem_frob_norm": jnp.max(jnp.stack(frob_max_list, axis=0), axis=0)
        if frob_max_list
        else jnp.zeros((b,), dtype=jnp.float32),
        "mean_mem_out_rms": jnp.mean(jnp.stack(mem_rms_list, axis=0), axis=0)
        if mem_rms_list
        else jnp.zeros((b,), dtype=jnp.float32),
        "mean_prior_error": jnp.mean(jnp.stack(prior_err_list, axis=0), axis=0) if prior_err_list else jnp.zeros((b,), dtype=jnp.float32),
        "mean_key_norm": jnp.mean(jnp.stack(key_norm_list, axis=0), axis=(0, 2)) if key_norm_list else jnp.ones((b,), dtype=jnp.float32),
    }
    return z_t, new_states, diags


def _scan_step(
    params: PyTree,
    state: list[PyTree],
    packet_t: PyTree,
    spec: SLWMv3Spec,
) -> tuple[list[PyTree], tuple[Array, dict[str, Array]]]:
    z_t, new_state, diags = step_core(params, state, packet_t, spec)
    return new_state, (z_t, diags)


def scan_core(
    params: PyTree,
    state: list[PyTree],
    packets_seq: PyTree,
    spec: SLWMv3Spec,
    *,
    remat_step: bool = False,
    remat_chunk_size: int = 64,
) -> tuple[Array, list[PyTree], dict[str, Array]]:
    """Run strictly causal sequence scan over `[T, B, ...]` packets using the exact `step_core` kernel.

    When `remat_step=True` and `T > remat_chunk_size`, uses outer chunked `jax.checkpoint` over chunks
    of `remat_chunk_size` steps, cutting saved reverse-mode carry HBM by `remat_chunk_size`x (e.g., 64x)
    while remaining mathematically identical to unchunked `lax.scan`.
    """
    t_len = int(packets_seq["valid_mask"].shape[0])
    if (not remat_step) or (t_len <= remat_chunk_size):
        step_fn = jax.checkpoint(lambda st, pkt: _scan_step(params, st, pkt, spec)) if remat_step else (
            lambda st, pkt: _scan_step(params, st, pkt, spec)
        )
        final_state, (z_seq, seq_diags) = jax.lax.scan(step_fn, state, packets_seq)
        return z_seq, final_state, seq_diags

    # Pad tail cleanly with `valid_mask = 0, dt_s = 0` (bit-exact zero state change!) to multiple of `remat_chunk_size`
    rem = t_len % remat_chunk_size
    pad_n = (remat_chunk_size - rem) if rem != 0 else 0
    n_chunks = (t_len + pad_n) // remat_chunk_size

    def _pad_and_chunk(k: str, arr: Array) -> Array:
        if pad_n > 0:
            pad_shape = (pad_n,) + arr.shape[1:]
            pad_val = jnp.zeros(pad_shape, dtype=arr.dtype)
            arr = jnp.concatenate([arr, pad_val], axis=0)
        return arr.reshape((n_chunks, remat_chunk_size) + arr.shape[1:])

    chunked_pkts = {k: _pad_and_chunk(k, v) for k, v in packets_seq.items()}

    @jax.checkpoint
    def _run_chunk(st: list[PyTree], pkt_chunk: PyTree) -> tuple[list[PyTree], tuple[Array, dict[str, Array]]]:
        return jax.lax.scan(lambda s, p: _scan_step(params, s, p, spec), st, pkt_chunk)

    final_state, (z_chunks, diag_chunks) = jax.lax.scan(_run_chunk, state, chunked_pkts)
    z_seq = z_chunks.reshape((n_chunks * remat_chunk_size,) + z_chunks.shape[2:])[:t_len]
    seq_diags = {
        k: v.reshape((n_chunks * remat_chunk_size,) + v.shape[2:])[:t_len]
        for k, v in diag_chunks.items()
    }
    return z_seq, final_state, seq_diags


def predict_future_latents(
    params: PyTree,
    z_seq: Array,
    planned_action_seq: Array,
    has_true_action_seq: Array,
    spec: SLWMv3Spec,
) -> Array:
    """Predict fixed train-normalized future packet features at `len(spec.pred_horizons)` horizons."""
    has_act = has_true_action_seq.astype(jnp.float32) > 0.5
    act_cond = (
        jnp.matmul(planned_action_seq.astype(jnp.float32), params["pred_head_action_w"]) * has_act.astype(jnp.float32)[..., None]
        + params["pred_head_action_avail_emb"][has_act.astype(jnp.int32)]
    )
    h = z_seq + act_cond
    raw = jnp.matmul(h, params["pred_head_w"]) + params["pred_head_bias"]
    return raw.reshape(z_seq.shape[0], z_seq.shape[1], len(spec.pred_horizons), spec.target_feat_dim)


# ---------------------------------------------------------------------------
# Exact Base-Shape muP Per-Leaf LR Multipliers & Strict Weight-Decay Mask
# ---------------------------------------------------------------------------


def _key_str(k: Any) -> str:
    if hasattr(k, "key"):
        return str(k.key)
    if hasattr(k, "idx"):
        return str(k.idx)
    return str(k)


def build_mup_lr_multipliers(params: PyTree, spec: SLWMv3Spec) -> PyTree:
    """Build exact per-leaf base-shape muP AdamW LR multiplier tree (`1.0` on EVERY leaf at base geometry).

    - `"input"`, `"scalar"`, or 1D leaves: `1.0`
    - `"fixed_hidden"` (`mem.w_o`, `mem.w_prior_proj`, `kv.w_o`): `spec.mem_base_dim / fan_in`
      (exact `1.0` at `pilot_d256`/`discovery_45m` `fan_in=128`, `0.5` at `confirmation_135m` `fan_in=256`,
      `0.25` at `700M–1B` `fan_in=512` — preserving O(1) coordinate updates when head rank scales!)
    - `"hidden"` and `"readout"` 2D width-scaling matrices: `1.0 / spec.width_mult` (`d_base / d_model`)
    """
    inv_m_d = 1.0 / max(float(spec.width_mult), 1e-6)
    base_mem_d = max(float(spec.mem_base_dim), 1.0)

    def _mult_for_leaf(path: tuple[Any, ...], arr: Array) -> float:
        parts = tuple(_key_str(k) for k in path)
        role = classify_param_mup_role(parts)
        if role in ("input", "scalar") or arr.ndim < 2:
            return 1.0
        if role == "fixed_hidden":
            return base_mem_d / max(float(arr.shape[0]), 1.0)
        return inv_m_d

    return jax.tree_util.tree_map_with_path(_mult_for_leaf, params)


def build_weight_decay_mask(params: PyTree) -> PyTree:
    """Return boolean pytree masking decoupled weight decay strictly to 2D+ non-scalar weight matrices.

    Guarantees ZERO weight decay on `log_tau`, `log_a`, `norm_w`, `rel_pos_bias`, `gate_z0`, `thresh`,
    and all 1D bias vectors (`g_bias`, `beta_bias`, `router_b1`, `router_b2`, etc.).
    """

    def _mask_for_leaf(path: tuple[Any, ...], arr: Array) -> bool:
        parts = tuple(_key_str(k) for k in path)
        role = classify_param_mup_role(parts)
        return bool(arr.ndim >= 2 and role != "scalar")

    return jax.tree_util.tree_map_with_path(_mask_for_leaf, params)
