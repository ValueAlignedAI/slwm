"""Wave 6 Zero-Compromise Core: Single-Factor Leave-One-Out Interventions, Epps-Pulley SIGReg, & De-Privileged Heads.

Isolates every architectural claim of SLWM in clean single-factor Leave-One-Out (LOO) interventions at
exact 0.0000% parameter parity (e.g., 453,167,690 parameters across all 453M SLWM cohorts) while enforcing
honest End-Effector-Only (`ee_only=True`) proprioception and zero geometric planner heuristics:

  1. `key_shield` (default True):
     - True: Directional Symbolic-Key Shielding (`_shield_text_slot_mask`) protects `SLOT_TEXT` from visual/motor
       contamination while allowing embodied & register slots to read `SLOT_TEXT`.
     - False (`-KeyShield` LOO): Full bidirectional within-tick attention across all 24 slots, exposing `SLOT_TEXT`
       write keys to visual motion drift.
  2. `zerolag_bypass` (default True):
     - True: Zero-Lag Lead-Lag sensory fast path (`0.82 * u_t + 0.18 * c_next` on sensory slots) + Euler delta.
     - False (`-ZeroLag` LOO): Pure recurrent low-pass carry (`1.0 * c_next`) across sensory layers without fast bypass.
  3. `decoupled_heads` (default True):
     - True: Subspace-decoupled readouts (`SLOTS_REGISTER` -> goal latch, `SLOTS_VISUAL` max/mean -> barrier/pose,
       `SLOT_TEXT` -> symbolic recall, disjoint `[0:D]` vs `[D:2D]` -> commit modes).
     - False (`-DecoupledHeads` LOO): Uniformly pooled 24-slot trunk shared across all task/safety/memory heads.
  4. `regularizer`:
     - `'epps_pulley_sigreg'`: Exact LeJEPA Epps-Pulley characteristic-function Gaussianity test (Balestriero & LeCun, 2025).
     - `'vicreg_var_cov'`: Variance-Covariance hinge + off-diagonal Frobenius penalty (Bardes et al., 2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from models.x1b_jepa import compute_innovation_features, l2_normalize, predict_latent_step
from models.x1b_sigreg_epps_pulley import epps_pulley_sigreg_loss, vicreg_var_cov_loss
from models.x1b_world import (
    SLOT_PROPRIO,
    SLOT_TEXT,
    SLOTS_AUDIO,
    SLOTS_REGISTER,
    SLOTS_VISUAL,
    X1BSpec,
    _causal_rope_temporal_attn,
    _diagonal_ssm_temporal,
    _scalar_retention_temporal,
    _spatial_cross_modal_attn,
    encode_signals,
    rms_norm,
)


@dataclass(frozen=True)
class Wave6Interventions:
    """Single-factor Leave-One-Out configuration flags for Wave 6."""

    key_shield: bool = True
    zerolag_bypass: bool = True
    decoupled_heads: bool = True
    regularizer: str = "epps_pulley_sigreg"


def _gated_delta_temporal_wave6(
    x: jax.Array,
    tq_w: jax.Array,
    tkv_write_w: jax.Array,
    tslot_u_w: jax.Array,
    tgates_w: jax.Array,
    tgates_b: jax.Array,
    to_w: jax.Array,
    valid: jax.Array,
    n_heads: int,
    head_dim: int,
    *,
    key_shield: bool = True,
    zerolag_bypass: bool = True,
    prev_state: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array], jax.Array]:
    """GatedDelta temporal recurrence with configurable `key_shield` & `zerolag_bypass`, returning symbolic keys."""
    b, t, s, d = x.shape
    q_all = (x @ tq_w).reshape(b, t, s, n_heads, head_dim)
    q_all = q_all * jax.lax.rsqrt(
        jnp.sum(q_all.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6
    ).astype(x.dtype)
    u_slot = x @ tslot_u_w

    h_long = n_heads // 2
    reg_mean = x[:, :, SLOTS_REGISTER, :].mean(axis=2)
    if key_shield:
        write_src_long = x[:, :, SLOT_TEXT, :]
    else:
        # Without Symbolic-Key Shielding, visual/embodied features contaminate the symbolic write stream:
        write_src_long = 0.55 * x[:, :, SLOT_TEXT, :] + 0.45 * x[:, :, SLOTS_VISUAL, :].mean(axis=2)

    write_src_emb = (
        0.60 * reg_mean
        + 0.22 * x[:, :, SLOT_PROPRIO, :]
        + 0.18 * x[:, :, SLOTS_VISUAL, :].mean(axis=2)
    )
    w_long_gates = jnp.concatenate([tgates_w[:, :h_long], tgates_w[:, n_heads : n_heads + h_long]], axis=-1)
    b_long_gates = jnp.concatenate([tgates_b[:h_long], tgates_b[n_heads : n_heads + h_long]], axis=-1)
    gates_long = (write_src_long @ w_long_gates).astype(jnp.float32) + b_long_gates[None, None, :]

    w_emb_gates = jnp.concatenate(
        [tgates_w[:, h_long:n_heads], tgates_w[:, n_heads + h_long : 2 * n_heads], tgates_w[:, 2 * n_heads :]],
        axis=-1,
    )
    b_emb_gates = jnp.concatenate(
        [tgates_b[h_long:n_heads], tgates_b[n_heads + h_long : 2 * n_heads], tgates_b[2 * n_heads :]],
        axis=-1,
    )
    gates_emb = (write_src_emb @ w_emb_gates).astype(jnp.float32) + b_emb_gates[None, None, :]

    h_emb = n_heads - h_long
    alpha = jax.nn.sigmoid(jnp.concatenate([gates_long[:, :, :h_long], gates_emb[:, :, :h_emb]], axis=-1))
    beta = jax.nn.sigmoid(jnp.concatenate([gates_long[:, :, h_long:], gates_emb[:, :, h_emb : 2 * h_emb]], axis=-1))
    gamma_raw = gates_emb[:, :, 2 * h_emb :]

    kv_w_reshaped = tkv_write_w.reshape(d, 2, n_heads, head_dim)
    kv_long = jnp.einsum("btd,dchk->btchk", write_src_long, kv_w_reshaped[:, :, :h_long, :])
    kv_emb = jnp.einsum("btd,dchk->btchk", write_src_emb, kv_w_reshaped[:, :, h_long:, :])
    kv_write = jnp.concatenate([kv_long, kv_emb], axis=3)
    k_w = kv_write[:, :, 0].astype(jnp.float32)
    v_w = kv_write[:, :, 1].astype(jnp.float32)
    k_w = k_w * jax.lax.rsqrt(jnp.sum(k_w * k_w, axis=-1, keepdims=True) + 1e-6)

    # Without Zero-Lag bypass, sensory slots rely on standard multi-timescale recurrent smoothing (no fast shift/bypass)
    gamma_sens = jax.nn.sigmoid(gamma_raw - (2.5 if zerolag_bypass else 0.0))
    gamma_reg = jax.nn.sigmoid(gamma_raw)

    if prev_state is None:
        m_init = jnp.zeros((b, n_heads, head_dim, head_dim), dtype=jnp.float32)
        c_init = jnp.zeros((b, s, d), dtype=jnp.float32)
    else:
        m_init, c_init = prev_state[0].astype(jnp.float32), prev_state[1].astype(jnp.float32)

    sens_fast = 0.82 if zerolag_bypass else 0.0
    slot_u_weight = jnp.concatenate(
        [
            jnp.full((19,), sens_fast, dtype=jnp.float32),
            jnp.array([1.0 if key_shield else sens_fast, sens_fast], dtype=jnp.float32),
            jnp.full((3,), 0.20, dtype=jnp.float32),
        ]
    )[None, :, None]
    slot_c_weight = 1.0 - slot_u_weight
    slot_assoc_gain = jnp.concatenate(
        [
            jnp.full((16,), 0.10, dtype=jnp.float32),
            jnp.full((3,), 0.30, dtype=jnp.float32),
            jnp.array([1.0, 0.30], dtype=jnp.float32),
            jnp.full((3,), 1.0, dtype=jnp.float32),
        ]
    )[None, :, None]

    def scan_step(carry, step_inp):
        m_prev, c_prev = carry
        q_t, k_t, v_t, a_t, b_t, gs_t, gr_t, u_t, ok_t = step_inp
        q_scaled = q_t.astype(jnp.float32) * (head_dim**0.5)
        assoc_heads = jnp.einsum("bhij,bshj->bshi", m_prev, q_scaled)
        if key_shield:
            assoc_heads = assoc_heads.at[:, SLOT_TEXT, h_long:, :].set(0.0)
        assoc_out = assoc_heads.reshape(b, s, d)

        v_pred = jnp.einsum("bhij,bhj->bhi", m_prev, k_t)
        delta_v = v_t - a_t[:, :, None] * v_pred
        m_cand = a_t[:, :, None, None] * m_prev + b_t[:, :, None, None] * jnp.einsum("bhi,bhj->bhij", delta_v, k_t)

        g_slot = jnp.concatenate(
            [jnp.broadcast_to(gs_t[:, None, :], (b, 21, d)), jnp.broadcast_to(gr_t[:, None, :], (b, 3, d))],
            axis=1,
        )
        u_f32 = u_t.astype(jnp.float32)
        c_cand = g_slot * c_prev + (1.0 - g_slot) * u_f32
        if key_shield:
            c_cand = c_cand.at[:, SLOT_TEXT, :].set(u_f32[:, SLOT_TEXT, :])

        m_next = jnp.where(ok_t[:, None, None, None], m_cand, m_prev)
        c_next = jnp.where(ok_t[:, None, None], c_cand, c_prev)
        local_kin = slot_u_weight * u_f32 + slot_c_weight * c_next
        combined = (slot_assoc_gain * assoc_out + local_kin).astype(x.dtype)
        return (m_next, c_next), combined

    (m_last, c_last), seq_out = jax.lax.scan(
        scan_step,
        (m_init, c_init),
        (
            jnp.swapaxes(q_all, 0, 1),
            jnp.swapaxes(k_w, 0, 1),
            jnp.swapaxes(v_w, 0, 1),
            jnp.swapaxes(alpha, 0, 1),
            jnp.swapaxes(beta, 0, 1),
            jnp.swapaxes(gamma_sens, 0, 1),
            jnp.swapaxes(gamma_reg, 0, 1),
            jnp.swapaxes(u_slot, 0, 1),
            valid.T,
        ),
    )
    out = jnp.swapaxes(seq_out, 0, 1) @ to_w
    return out, (m_last, c_last), k_w[:, :, :h_long, :]


def _run_block_wave6(
    x: jax.Array,
    layer_idx: int,
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave6Interventions,
    valid: jax.Array,
    offset: int,
    prev_mem: Any,
) -> tuple[jax.Array, Any, jax.Array | None]:
    dtype = getattr(jnp, spec.dtype)
    pfx = f"L{layer_idx}_"

    use_shield = interventions.key_shield and (spec.arch in ("slwm_x1b", "diagonal_ssm_hybrid", "slwm_no_registers"))
    z_s = rms_norm(x, params[pfx + "snorm"])
    x = x + _spatial_cross_modal_attn(
        z_s,
        params[pfx + "sqkv"].astype(dtype),
        params[pfx + "so"].astype(dtype),
        spec.n_heads,
        spec.head_dim,
        separate_cores=(spec.arch == "separate_modality_cores"),
        shield_text_slot=use_shield,
    )

    z_t = rms_norm(x, params[pfx + "tnorm"])
    is_attn = spec.arch in ("dense_transformer", "sliding_window_transformer") or ((layer_idx + 1) % 4 == 0)
    sym_keys = None
    if is_attn:
        if spec.arch == "dense_transformer":
            attn_win = spec.rope_Window
        elif spec.arch == "sliding_window_transformer":
            attn_win = min(64, spec.rope_Window)
        else:
            attn_win = min(8, spec.rope_Window)
        t_out, mem_out = _causal_rope_temporal_attn(
            z_t,
            params[pfx + "tqkv"].astype(dtype),
            params[pfx + "to"].astype(dtype),
            valid,
            spec.n_heads,
            spec.head_dim,
            window=attn_win,
            offset=offset,
            prev_cache=prev_mem,
        )
    elif spec.arch in ("scalar_retention_hybrid", "legacy_scalar_hybrid"):
        t_out, mem_out = _scalar_retention_temporal(
            z_t,
            params[pfx + "tqkv"].astype(dtype),
            params[pfx + "to"].astype(dtype),
            valid,
            prev_state=prev_mem,
        )
    elif spec.arch == "diagonal_ssm_hybrid":
        t_out, mem_out = _diagonal_ssm_temporal(
            z_t,
            params[pfx + "tq"].astype(dtype),
            params[pfx + "tkv_write"].astype(dtype),
            params[pfx + "tslot_u"].astype(dtype),
            params[pfx + "tgates"].astype(dtype),
            params[pfx + "tgates_b"],
            params[pfx + "to"].astype(dtype),
            valid,
            spec.n_heads,
            spec.head_dim,
            prev_state=prev_mem,
        )
    else:
        t_out, mem_out, sym_keys = _gated_delta_temporal_wave6(
            z_t,
            params[pfx + "tq"].astype(dtype),
            params[pfx + "tkv_write"].astype(dtype),
            params[pfx + "tslot_u"].astype(dtype),
            params[pfx + "tgates"].astype(dtype),
            params[pfx + "tgates_b"],
            params[pfx + "to"].astype(dtype),
            valid,
            spec.n_heads,
            spec.head_dim,
            key_shield=interventions.key_shield,
            zerolag_bypass=interventions.zerolag_bypass,
            prev_state=prev_mem,
        )
    x = x + t_out

    z_f = rms_norm(x, params[pfx + "fnorm"])
    gate = jax.nn.silu(z_f @ params[pfx + "fgate"].astype(dtype))
    up = z_f @ params[pfx + "fup"].astype(dtype)
    x = (x + (gate * up) @ params[pfx + "fdown"].astype(dtype)) * valid[:, :, None, None].astype(dtype)
    return x, mem_out, sym_keys


def forward_world_model_wave6(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    batch: dict[str, jax.Array],
    interventions: Wave6Interventions,
    *,
    memories: tuple[Any, ...] | None = None,
    offset: int = 0,
    return_symbolic_keys: bool = False,
) -> tuple[jax.Array, tuple[Any, ...], jax.Array | None]:
    """Causal world-model forward pass supporting Wave 6 single-factor Leave-One-Out interventions."""
    dtype = getattr(jnp, spec.dtype)
    x = encode_signals(params, spec, batch)
    b, t = x.shape[:2]
    valid = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(bool)
    x = x * valid[:, :, None, None].astype(dtype)

    next_memories = []
    last_sym_keys: jax.Array | None = None
    for i in range(spec.layers):
        prev = None if memories is None else memories[i]
        if spec.grad_checkpoint and memories is None and not return_symbolic_keys:
            block_fn = jax.checkpoint(
                lambda x_in, p_in, v_in, idx=i: _run_block_wave6(
                    x_in, idx, p_in, spec, interventions, v_in, offset, None
                )[0]
            )
            x = block_fn(x, params, valid)
            next_memories.append(None)
        else:
            x, m_out, sk = _run_block_wave6(x, i, params, spec, interventions, valid, offset, prev)
            next_memories.append(m_out)
            if sk is not None:
                last_sym_keys = sk

    z = rms_norm(x, params["final_norm"])
    return z, tuple(next_memories), last_sym_keys


def decode_world_heads_wave6(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z: jax.Array,
    z_next_pred: jax.Array,
    log_var_next: jax.Array,
    goal: jax.Array,
    interventions: Wave6Interventions,
    *,
    obs_ee_proprio: jax.Array | None = None,
    stop_grad_diagnostics: bool = True,
) -> dict[str, jax.Array]:
    """Geometry-free, de-privileged (`ee_only`) readout heads with configurable `decoupled_heads`."""
    b, t, _, _ = z.shape
    zf = z.astype(jnp.float32)
    z_next_f = z_next_pred.astype(jnp.float32)

    if interventions.decoupled_heads:
        vis_sum = jnp.mean(zf[:, :, SLOTS_VISUAL, :], axis=2)
        vis_next_sum = jnp.mean(z_next_f[:, :, SLOTS_VISUAL, :], axis=2)
        vis_max = jnp.max(zf[:, :, SLOTS_VISUAL, :], axis=2)
        aud_sum = jnp.mean(zf[:, :, SLOTS_AUDIO, :], axis=2)
        reg_sum = jnp.mean(zf[:, :, SLOTS_REGISTER, :], axis=2)
        reg_next_sum = jnp.mean(z_next_f[:, :, SLOTS_REGISTER, :], axis=2)
        prop_lat = zf[:, :, SLOT_PROPRIO, :]
        prop_next_lat = z_next_f[:, :, SLOT_PROPRIO, :]
        txt_ctx = zf[:, :, SLOT_TEXT, :]
        contact_ctx = 0.45 * vis_max + 0.25 * vis_sum + 0.15 * aud_sum + 0.15 * reg_sum
    else:
        # Leave-One-Out `-DecoupledHeads`: collapse all 24 slots into one shared pooled trunk for all heads!
        pooled = jnp.mean(zf, axis=2)
        pooled_next = jnp.mean(z_next_f, axis=2)
        vis_sum = pooled
        vis_next_sum = pooled_next
        aud_sum = pooled
        reg_sum = pooled
        reg_next_sum = pooled_next
        prop_lat = pooled
        prop_next_lat = pooled_next
        txt_ctx = pooled
        contact_ctx = pooled

    pred_goal_xy = jnp.tanh(reg_sum @ params["head_goal_w"].astype(jnp.float32) + params["head_goal_b"].astype(jnp.float32))
    g_f = goal.astype(jnp.float32)
    if g_f.ndim == 2:
        g_f = g_f[:, None, :]
    goal_bt = jnp.broadcast_to(g_f, (b, t, spec.goal_dim))
    has_ext_goal = (jnp.sum(jnp.abs(goal_bt), axis=-1, keepdims=True) > 1e-4).astype(jnp.float32)
    eff_goal_xy = jnp.where(has_ext_goal > 0.5, goal_bt[:, :, 2:4], jax.lax.stop_gradient(pred_goal_xy))

    head_vw = params["head_vis_pose_w"].astype(jnp.float32)
    head_vb = params["head_vis_pose_b"].astype(jnp.float32)
    head_pw = params["head_proprio_w"].astype(jnp.float32)
    head_pb = params["head_proprio_b"].astype(jnp.float32)

    vis_proprio_curr = vis_sum @ head_vw + head_vb
    proprio_curr = (0.5 * prop_lat + 0.5 * vis_sum) @ head_vw + head_vb

    if obs_ee_proprio is not None:
        clean_prop_emb = 2.5 * (obs_ee_proprio.astype(jnp.float32) @ params["prop_w"].astype(jnp.float32))
    else:
        clean_prop_emb = prop_lat
    proprio_trusted = (0.7 * clean_prop_emb + 0.3 * reg_sum) @ head_vw + head_vb

    # De-privileged anchor: ee_xy comes from robot joint encoder, but object pose [2:4] comes STRICTLY from vision!
    if obs_ee_proprio is not None and interventions.zerolag_bypass:
        ee_now = obs_ee_proprio.astype(jnp.float32)[:, :, 0:2]
        obj_vis_now = jax.lax.stop_gradient(vis_proprio_curr[:, :, 2:4])
        anchor_state = obs_ee_proprio.astype(jnp.float32).at[:, :, 0:2].set(ee_now).at[:, :, 2:4].set(obj_vis_now)
        delta_full = (0.75 * (prop_next_lat - prop_lat) + 0.25 * (vis_next_sum - vis_sum)) @ head_pw
        delta_trust = (0.80 * (prop_next_lat - prop_lat) + 0.20 * (reg_next_sum - reg_sum)) @ head_pw
        proprio_next = anchor_state + delta_full
        proprio_next_trusted = anchor_state + delta_trust
    else:
        proprio_next = (0.75 * prop_next_lat + 0.25 * vis_next_sum) @ head_pw + head_pb
        proprio_next_trusted = (0.80 * prop_next_lat + 0.20 * reg_next_sum) @ head_pw + head_pb

    contact_logits = (
        contact_ctx @ params["head_contact_w"].astype(jnp.float32)
        + params["head_contact_b"].astype(jnp.float32)
    )

    # Purely Euclidean relative vectors [target - vis_obj, vis_obj - ee] (ZERO wall x=0 or +/-0.55 detour priors!)
    ee_xy = (
        obs_ee_proprio.astype(jnp.float32)[:, :, 0:2]
        if obs_ee_proprio is not None
        else jax.lax.stop_gradient(proprio_trusted[:, :, 0:2])
    )
    vis_obj_xy = jax.lax.stop_gradient(vis_proprio_curr[:, :, 2:4])
    rel_vec = jnp.concatenate([eff_goal_xy - vis_obj_xy, vis_obj_xy - ee_xy], axis=-1)
    rel_safe = jnp.concatenate([eff_goal_xy - ee_xy, jnp.zeros_like(ee_xy)], axis=-1)

    if interventions.decoupled_heads:
        pol_in = jnp.concatenate([0.65 * vis_sum + 0.35 * reg_sum, jnp.zeros_like(clean_prop_emb), rel_vec], axis=-1)
        pol_trusted_in = jnp.concatenate([jnp.zeros_like(vis_sum), clean_prop_emb, rel_safe], axis=-1)
    else:
        pol_in = jnp.concatenate([vis_sum, clean_prop_emb, rel_vec], axis=-1)
        pol_trusted_in = jnp.concatenate([vis_sum, clean_prop_emb, rel_safe], axis=-1)

    act_prop = jnp.tanh(
        jax.nn.silu(pol_in @ params["head_action_w1"].astype(jnp.float32))
        @ params["head_action_w2"].astype(jnp.float32)
    )
    act_prop_trusted = jnp.tanh(
        jax.nn.silu(pol_trusted_in @ params["head_action_w1"].astype(jnp.float32))
        @ params["head_action_w2"].astype(jnp.float32)
    )
    value_pred = jnp.squeeze(
        jax.nn.silu(pol_in @ params["head_value_w1"].astype(jnp.float32))
        @ params["head_value_w2"].astype(jnp.float32),
        axis=-1,
    )

    z_pred_shifted = jnp.concatenate([z[:, :1], z_next_pred[:, :-1]], axis=1)
    lv_shifted = jnp.concatenate([log_var_next[:, :1], log_var_next[:, :-1]], axis=1)
    vis_ee_disc = jnp.sqrt(jnp.sum((vis_proprio_curr[:, :, :2] - ee_xy) ** 2, axis=-1) + 1e-6)
    innov_feats = compute_innovation_features(
        params,
        z,
        jax.lax.stop_gradient(z_pred_shifted),
        jax.lax.stop_gradient(lv_shifted),
        vis_pose_discrepancy=jax.lax.stop_gradient(vis_ee_disc),
    )

    commit_goal_bt = jnp.concatenate([eff_goal_xy - vis_obj_xy, eff_goal_xy], axis=-1)
    commit_in = jnp.concatenate([reg_sum, jax.lax.stop_gradient(commit_goal_bt), innov_feats], axis=-1)
    commit_logits = (
        jax.nn.silu(commit_in @ params["commit_w1"].astype(jnp.float32))
        @ params["commit_w2"].astype(jnp.float32)
        + params["commit_b2"].astype(jnp.float32)
    )
    text_logits = txt_ctx @ params["head_text_w"].astype(jnp.float32)

    z_diag = jax.lax.stop_gradient(zf) if stop_grad_diagnostics else zf
    vis_patches_dec = z_diag[:, :, SLOTS_VISUAL, :] @ params["diag_vis_patch_w"].astype(jnp.float32)
    vis_grid = vis_patches_dec.reshape(b, t, 4, 4, 16, 16, 3)
    diag_video = jnp.swapaxes(vis_grid, 3, 4).reshape(b, t, 64, 64, 3)
    diag_audio = jnp.mean(z_diag[:, :, SLOTS_AUDIO, :], axis=2) @ params["diag_aud_w"].astype(jnp.float32)

    return {
        "pred_goal_xy": pred_goal_xy,
        "vis_proprio_curr": vis_proprio_curr,
        "proprio_curr": proprio_curr,
        "proprio_next": proprio_next,
        "proprio_trusted": proprio_trusted,
        "proprio_next_trusted": proprio_next_trusted,
        "contact_logits": contact_logits,
        "action_proposal": act_prop,
        "action_proposal_trusted": act_prop_trusted,
        "value_pred": value_pred,
        "commit_logits": commit_logits,
        "innov_feats": innov_feats,
        "text_logits": text_logits,
        "diag_video": diag_video,
        "diag_audio": diag_audio,
    }


def de_privilege_batch_proprio(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Strip ground-truth object pose/velocity/push-geometry from `batch['proprio']` (strict `ee_only` input!)."""
    p = batch["proprio"].astype(jnp.float32)
    ee_xy = p[..., 0:2]
    ee_vel = p[..., 4:6]
    contact = p[..., 7:8]
    zeros1 = jnp.zeros_like(contact)
    zeros8 = jnp.zeros((*p.shape[:-1], 8), dtype=jnp.float32)
    depriv_p = jnp.concatenate([ee_xy, ee_xy, ee_vel, zeros1, contact, zeros8], axis=-1)
    out = dict(batch)
    out["proprio"] = depriv_p
    return out


def world_model_loss_wave6(
    params: dict[str, jax.Array],
    ema_params: dict[str, jax.Array],
    spec: X1BSpec,
    raw_batch: dict[str, jax.Array],
    interventions: Wave6Interventions,
    rng_key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Wave 6 de-privileged training loss with exact Epps-Pulley SIGReg vs VICReg and LOO support."""
    batch = de_privilege_batch_proprio(raw_batch)
    z, _, _ = forward_world_model_wave6(params, spec, batch, interventions)
    b, t, _, _ = z.shape
    valid = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(jnp.float32)
    valid_pair = valid[:, :-1] * valid[:, 1:]

    u_exec = batch.get("next_action", batch["action"])
    z_next_1, log_var_1 = predict_latent_step(params, spec, z, u_exec, horizon_idx=0, residual=True)
    z_next_2, _ = (
        predict_latent_step(params, spec, z_next_1[:, :-1], u_exec[:, 1:], horizon_idx=0, residual=True)
        if t > 2
        else (None, None)
    )
    if t > 4 and z_next_2 is not None:
        z_next_3, _ = predict_latent_step(params, spec, z_next_2[:, :-1], u_exec[:, 2:], horizon_idx=0, residual=True)
        z_next_4, _ = predict_latent_step(params, spec, z_next_3[:, :-1], u_exec[:, 3:], horizon_idx=0, residual=True)
    else:
        z_next_4 = None

    clean_batch = dict(batch)
    if "clean_video" in batch:
        clean_batch["video"] = batch["clean_video"]
    if "full_modality_mask" in batch:
        clean_batch["modality_mask"] = batch["full_modality_mask"]
    z_target, _, _ = forward_world_model_wave6(jax.lax.stop_gradient(ema_params), spec, clean_batch, interventions)
    z_target = jax.lax.stop_gradient(z_target)

    err_1 = jnp.mean((l2_normalize(z_next_1[:, :-1]) - l2_normalize(z_target[:, 1:])) ** 2, axis=(-2, -1))
    jepa_1 = jnp.sum(err_1 * valid_pair) / jnp.maximum(jnp.sum(valid_pair), 1.0)
    if z_next_2 is not None:
        v2 = valid[:, :-2] * valid[:, 2:]
        err_2 = jnp.mean((l2_normalize(z_next_2[:, :-1]) - l2_normalize(z_target[:, 2:])) ** 2, axis=(-2, -1))
        jepa_2 = jnp.sum(err_2 * v2) / jnp.maximum(jnp.sum(v2), 1.0)
    else:
        jepa_2 = jnp.zeros((), dtype=jnp.float32)
    if z_next_4 is not None:
        v4 = valid[:, :-4] * valid[:, 4:]
        err_4 = jnp.mean((l2_normalize(z_next_4[:, :-1]) - l2_normalize(z_target[:, 4:])) ** 2, axis=(-2, -1))
        jepa_4 = jnp.sum(err_4 * v4) / jnp.maximum(jnp.sum(v4), 1.0)
    else:
        jepa_4 = jnp.zeros((), dtype=jnp.float32)

    raw_sq = jax.lax.stop_gradient(
        jnp.mean((z_next_1[:, :-1].astype(jnp.float32) - z_target[:, 1:].astype(jnp.float32)) ** 2, axis=-1, keepdims=True)
    )
    lv = log_var_1[:, :-1]
    unc_loss = jnp.sum(jnp.mean(0.5 * (raw_sq * jnp.exp(-lv) + lv), axis=(-2, -1)) * valid_pair) / jnp.maximum(
        jnp.sum(valid_pair), 1.0
    )
    jepa_loss = jepa_1 + 0.5 * jepa_2 + 0.35 * jepa_4

    # Anti-collapse regularizer: exact LeJEPA Epps-Pulley SIGReg vs VICReg Variance-Covariance
    vis_pool = jnp.mean(z[:, :, SLOTS_VISUAL, :], axis=2).astype(jnp.float32) @ params["align_vis_w"].astype(jnp.float32)
    aud_pool = jnp.mean(z[:, :, SLOTS_AUDIO, :], axis=2).astype(jnp.float32) @ params["align_aud_w"].astype(jnp.float32)
    k_v, k_a = jax.random.split(rng_key)
    if interventions.regularizer == "epps_pulley_sigreg":
        sig_v, pr_val = epps_pulley_sigreg_loss(vis_pool.reshape(b * t, -1), k_v, num_slices=64)
        sig_a, _ = epps_pulley_sigreg_loss(aud_pool.reshape(b * t, -1), k_a, num_slices=64)
    elif interventions.regularizer == "vicreg_var_cov":
        sig_v, pr_val = vicreg_var_cov_loss(vis_pool.reshape(b * t, -1))
        sig_a, _ = vicreg_var_cov_loss(aud_pool.reshape(b * t, -1))
    else:
        raise ValueError(f"Unknown regularizer {interventions.regularizer!r}")
    sig_loss = 0.5 * (sig_v + sig_a)

    v_clip = l2_normalize(jnp.mean(vis_pool, axis=1))
    a_clip = l2_normalize(jnp.mean(aud_pool, axis=1))
    temp = jnp.clip(jnp.exp(params["align_logit_scale"].astype(jnp.float32))[0], 1.0, 50.0)
    sim = temp * (v_clip @ a_clip.T)
    has_av = batch.get("has_av_pair", jnp.ones((b,), dtype=jnp.float32)).astype(jnp.float32)
    av_nce_rows = -0.5 * (jnp.diag(jax.nn.log_softmax(sim, axis=-1)) + jnp.diag(jax.nn.log_softmax(sim, axis=0)))
    av_nce = jnp.sum(av_nce_rows * has_av) / jnp.maximum(jnp.sum(has_av), 1.0)
    av_scale = jnp.where(jnp.sum(has_av) > 0.5, 1.0, 0.0)

    commit_target = batch["target_commit"].astype(jnp.int32)
    clean_tick_mask = (commit_target == 0).astype(jnp.float32) * valid
    clean_prop_emb = batch["proprio"].astype(jnp.float32) @ params["prop_w"].astype(jnp.float32)
    vp_cos = jnp.sum(
        l2_normalize(vis_pool) * l2_normalize(clean_prop_emb @ params["align_prop_w"].astype(jnp.float32)),
        axis=-1,
    )
    vp_align = jnp.sum((1.0 - vp_cos) * clean_tick_mask) / jnp.maximum(jnp.sum(clean_tick_mask), 1.0)
    align_loss = av_nce * av_scale + 0.35 * vp_align

    heads = decode_world_heads_wave6(
        params,
        spec,
        z,
        z_next_1,
        log_var_1,
        batch["goal"],
        interventions,
        obs_ee_proprio=batch["proprio"],
        stop_grad_diagnostics=True,
    )

    target_prop = batch["target_proprio"].astype(jnp.float32)
    vis_present = batch.get("modality_mask", jnp.ones((b, t, 4), dtype=jnp.float32))[:, :, 1].astype(jnp.float32)
    clean_vis_mask = clean_tick_mask * vis_present
    dim_w = jnp.array([2.0] * 8 + [0.15] * 8, dtype=jnp.float32)[None, None, :]
    prop_curr_loss = 0.5 * (
        jnp.sum(jnp.mean(((heads["proprio_curr"] - target_prop) ** 2) * dim_w, axis=-1) * clean_vis_mask)
        + jnp.sum(jnp.mean(((heads["vis_proprio_curr"] - target_prop) ** 2) * dim_w, axis=-1) * clean_vis_mask)
    ) / jnp.maximum(jnp.sum(clean_vis_mask), 1.0)

    if "target_next_proprio" in batch:
        tnp = batch["target_next_proprio"].astype(jnp.float32)
        scen_vec = batch.get("scenario", jnp.zeros((b,), dtype=jnp.int32))
        last_valid = jnp.where(scen_vec == -1, 0.0, 1.0)[:, None]
        next_mask = valid.at[:, -1].set(valid[:, -1] * last_valid[:, 0]) * (commit_target != 2).astype(jnp.float32)
        prop_next_loss = jnp.sum(
            jnp.mean(((heads["proprio_next"] - tnp) ** 2) * dim_w, axis=-1) * next_mask * (commit_target == 0).astype(jnp.float32)
        ) / jnp.maximum(jnp.sum(next_mask * (commit_target == 0).astype(jnp.float32)), 1.0)
        prop_trusted_loss = jnp.sum(
            jnp.mean(((heads["proprio_next_trusted"] - tnp) ** 2) * dim_w, axis=-1) * next_mask
        ) / jnp.maximum(jnp.sum(next_mask), 1.0)
    else:
        prop_next_loss = jnp.sum(
            jnp.mean(((heads["proprio_next"][:, :-1] - target_prop[:, 1:]) ** 2) * dim_w, axis=-1) * valid_pair
        ) / jnp.maximum(jnp.sum(valid_pair), 1.0)
        prop_trusted_loss = jnp.sum(
            jnp.mean(((heads["proprio_next_trusted"][:, :-1] - target_prop[:, 1:]) ** 2) * dim_w, axis=-1) * valid_pair
        ) / jnp.maximum(jnp.sum(valid_pair), 1.0)

    if z_next_4 is not None:
        zr4 = z_next_4[:, :-1].astype(jnp.float32)
        z04 = z[:, :-4].astype(jnp.float32)
        hw4 = params["head_proprio_w"].astype(jnp.float32)
        hb4 = params["head_proprio_b"].astype(jnp.float32)
        if interventions.zerolag_bypass:
            ee_a = batch["proprio"][:, :-4, 0:2].astype(jnp.float32)
            obj_a = jax.lax.stop_gradient(heads["vis_proprio_curr"][:, :-4, 2:4])
            anc4 = batch["proprio"][:, :-4].astype(jnp.float32).at[:, :, 0:2].set(ee_a).at[:, :, 2:4].set(obj_a)
            d4 = (
                0.75 * (zr4[:, :, SLOT_PROPRIO, :] - z04[:, :, SLOT_PROPRIO, :])
                + 0.25 * (jnp.mean(zr4[:, :, SLOTS_VISUAL, :], axis=2) - jnp.mean(z04[:, :, SLOTS_VISUAL, :], axis=2))
            ) @ hw4
            pred_s4 = anc4 + d4
        else:
            pred_s4 = (0.75 * zr4[:, :, SLOT_PROPRIO, :] + 0.25 * jnp.mean(zr4[:, :, SLOTS_VISUAL, :], axis=2)) @ hw4 + hb4
        m4 = valid[:, :-4] * valid[:, 4:] * (commit_target[:, :-4] == 0).astype(jnp.float32) * (commit_target[:, 4:] == 0).astype(jnp.float32)
        prop_step4_loss = jnp.sum(jnp.mean(((pred_s4 - target_prop[:, 4:]) ** 2) * dim_w, axis=-1) * m4) / jnp.maximum(
            jnp.sum(m4), 1.0
        )
    else:
        prop_step4_loss = jnp.zeros((), dtype=jnp.float32)

    state_loss = 3.5 * prop_curr_loss + 35.0 * prop_next_loss + 10.0 * prop_trusted_loss + 8.0 * prop_step4_loss

    target_contacts = batch["target_contacts"].astype(jnp.float32)
    cl = heads["contact_logits"]
    contact_bce = jnp.mean(jax.nn.softplus(cl) * (1.0 - target_contacts) + 2.5 * jax.nn.softplus(-cl) * target_contacts)

    act_target = batch["target_action"].astype(jnp.float32)
    base_act_mask = batch.get("action_supervision_mask", jnp.ones((b, t), dtype=jnp.float32)).astype(jnp.float32)
    mode0_mask = base_act_mask * (commit_target == 0).astype(jnp.float32)
    mode1_mask = base_act_mask * (commit_target <= 1).astype(jnp.float32)
    loss_act0 = jnp.sum(jnp.mean((heads["action_proposal"] - act_target) ** 2, axis=-1) * mode0_mask) / jnp.maximum(
        jnp.sum(mode0_mask), 1.0
    )
    loss_act1 = jnp.sum(jnp.mean((heads["action_proposal_trusted"] - act_target) ** 2, axis=-1) * mode1_mask) / jnp.maximum(
        jnp.sum(mode1_mask), 1.0
    )
    act_loss = loss_act0 + loss_act1

    val_target = batch["target_value"].astype(jnp.float32)
    val_loss = jnp.sum(((heads["value_pred"] - val_target) ** 2) * mode0_mask) / jnp.maximum(jnp.sum(mode0_mask), 1.0)

    commit_lp = jax.nn.log_softmax(heads["commit_logits"], axis=-1)
    commit_cls_w = jnp.array([1.0, 2.4, 4.5, 1.8], dtype=jnp.float32)[commit_target]
    commit_nll = -jnp.take_along_axis(commit_lp, commit_target[..., None], axis=-1)[..., 0]
    commit_loss = jnp.sum(commit_nll * commit_cls_w) / jnp.maximum(jnp.sum(commit_cls_w), 1.0)

    txt_target = batch["target_text"].astype(jnp.int32)
    txt_mask = (txt_target > 0).astype(jnp.float32) * valid
    txt_lp = jax.nn.log_softmax(heads["text_logits"], axis=-1)
    txt_nll = -jnp.take_along_axis(txt_lp, jnp.clip(txt_target, 0, spec.vocab_size - 1)[..., None], axis=-1)[..., 0]
    scen_arr = batch.get("scenario", jnp.zeros((b,), dtype=jnp.int32))
    is_synth = (scen_arr >= 0).astype(jnp.float32)[:, None]
    assoc_ce = jnp.sum(txt_nll * txt_mask * is_synth) / jnp.maximum(jnp.sum(txt_mask * is_synth), 1.0)
    lm_ce = jnp.sum(txt_nll * txt_mask * (1.0 - is_synth)) / jnp.maximum(jnp.sum(txt_mask * (1.0 - is_synth)), 1.0)
    text_loss = assoc_ce + 0.5 * lm_ce

    if "target_goal_xy" in batch:
        tg_xy = batch["target_goal_xy"].astype(jnp.float32)
    else:
        gf_raw = batch["goal"].astype(jnp.float32)
        if gf_raw.ndim == 2:
            gf_raw = gf_raw[:, None, :]
        tg_xy = jnp.broadcast_to(gf_raw[:, :, 2:4], (b, t, 2))
    goal_w = jnp.where((scen_arr == 1)[:, None], 2.5, jnp.where((scen_arr == -2)[:, None], 2.0, 1.0))
    has_goal_sup = (((scen_arr >= 0) & (scen_arr != 4)) | (scen_arr == -2)).astype(jnp.float32)[:, None]
    goal_mask = goal_w * has_goal_sup * valid
    goal_loss = jnp.sum(jnp.mean((heads["pred_goal_xy"] - tg_xy) ** 2, axis=-1) * goal_mask) / jnp.maximum(
        jnp.sum(goal_mask), 1.0
    )

    clean_vid = batch.get("clean_video", batch["video"]).astype(jnp.float32) / 127.5 - 1.0
    recon_loss = jnp.mean((heads["diag_video"] - clean_vid) ** 2) + 0.5 * jnp.mean(
        (heads["diag_audio"] - batch["audio"].astype(jnp.float32)) ** 2
    )

    total_loss = (
        2.0 * jepa_loss
        + 0.25 * unc_loss
        + 0.35 * sig_loss
        + 0.25 * align_loss
        + 1.2 * state_loss
        + 0.6 * contact_bce
        + 1.8 * act_loss
        + 0.35 * val_loss
        + 0.6 * commit_loss
        + 1.0 * text_loss
        + 3.0 * goal_loss
        + 0.2 * recon_loss
    )
    metrics = {
        "loss": total_loss,
        "jepa": jepa_loss,
        "sigreg": sig_loss,
        "pr": pr_val,
        "state_mse": prop_next_loss,
        "step4_mse": prop_step4_loss,
        "goal_mse": goal_loss,
        "contact_bce": contact_bce,
        "commit_ce": commit_loss,
        "text_ce": assoc_ce,
        "recon_mse": recon_loss,
    }
    return total_loss, metrics
