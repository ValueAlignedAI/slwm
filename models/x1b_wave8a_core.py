"""Wave 8A Joint Recall + Spatial Non-Regression & GQA Parity Core (`SLWM-X2`).

Eliminates the cross-task representational & gradient interference that caused `T = 1024` dynamic-visual
associative overwrite recall to collapse to `6.3%` in Wave 7 (`runs_wave7`) while preserving non-inferior
de-privileged spatial control, `T = 160` post-perturbation goal-switching, OOD corruption detection, and
spoken physical `xArm6` grounding from a SINGLE checkpoint at `0.0000%` parameter delta (`453,199,938`
strict parameters across all `SLWM-X2` and `DiagonalSSM` 453M cohorts, `< 0.01%` on `9:1` GQA):

  1. Disjoint Recurrent Head Partitioning (`H_sym ⊕ H_goal ⊕ H_phys`):
     Splits the `h_long = H // 2` long-horizon heads into:
       - `H_sym = [0 : h_sym]` (`6` heads at `453M`, `8` at `1B`): written by key-shielded `SLOT_TEXT`,
         read EXCLUSIVELY by `SLOT_TEXT` for discrete multi-item symbolic key-value bindings (`K = 1..16`).
       - `H_goal = [h_sym : h_long]` (`3` heads at `453M`, `4` at `1B`): written by key-shielded `SLOT_TEXT`
         with attenuated task backprop, read by `SLOTS_REGISTER` using causal query `l2_norm(0.35*q_reg + 0.65*q_txt)`
         so coordinate-free queries `[0, GOAL_KEY_ID, 0, 2]` retrieve latched 2D goals across `T = 160` without
         routing rank-2 goal regression through `SLOT_TEXT` or `H_sym`.
       - `H_phys = [h_long : H]`: embodied multi-timescale kinematics and contact state.
     Applied identically to `DiagonalSSM` (`_diagonal_ssm_temporal_wave8a`) so Cohort 1 vs Cohort 5 isolates
     outer-product matrix memory `M_t in R^{Dh x Dh}` vs diagonal vector state `s_t in R^{Dh}` with zero confounds.
  2. Top-Level `Z_mem ⊕ Z_task` Gradient Isolation (`fork_isolated_latent_streams` via `jax.custom_vjp`):
     Zeros all control, spatial, JEPA, and rank-2 goal-regression gradients on top `SLOT_TEXT`
     (`d L_task / d Z_top[SLOT_TEXT] = 0`) and decouples `decode_goal_latch_wave8a` (`0.85*reg_sum + 0.15*aud_sum`)
     from `SLOT_TEXT`, while preserving forward cross-modal grounding across all `L` layers.
  3. Token-Subspace Disjoint Text Loss (`world_model_loss_wave8a`):
     Separates discrete associative recall targets (`64 <= txt_target < 96`) from dense 2D goal-bin targets
     (`txt_target >= 3584`) by exact vocabulary subspace rather than scenario index, with `enable_assoc_loss`
     toggle for the Spatial-Control-Only ceiling anchor (Cohort 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from models.x1b_jepa import compute_innovation_features, l2_normalize, predict_latent_step
from models.x1b_sigreg_epps_pulley import epps_pulley_sigreg_loss, vicreg_var_cov_loss
from models.x1b_wave6_core import (
    Wave6Interventions,
    _gated_delta_temporal_wave6,
)
from models.x1b_wave7_core import (
    GOAL_KEY_ID,
    DIST_KEY_ID,
    GOAL_VAL_OFFSET,
    GOAL_NUM_BINS_2D,
    canonical_grid_centers_8x8,
    canonical_patch_centers_4x4,
    decode_spatial_hazard_grid_8x8,
    decode_spatial_vis_pose,
    encode_goal_query_tokens,
    encode_goal_tokens_2d,
    encode_pure_visual_grid_8x8,
    encode_pure_visual_patches,
    quantize_goal_2d_val_id,
    sample_hazard_grid_bilinear,
    spatial_grid_local_context,
)
from models.x1b_world import (
    SLOT_PROPRIO,
    SLOT_TEXT,
    SLOTS_AUDIO,
    SLOTS_REGISTER,
    SLOTS_VISUAL,
    TOTAL_SLOTS,
    X1BSpec,
    _causal_rope_temporal_attn,
    _diagonal_ssm_temporal,
    _scalar_retention_temporal,
    _separate_modality_mask,
    _shield_text_slot_mask,
    _spatial_grid_fourier,
    effective_kv_heads,
    encode_signals,
    rms_norm,
)


@dataclass(frozen=True)
class Wave8AInterventions:
    """Single-factor intervention & loss-toggle flags for Study X2 Wave 8A."""

    key_shield: bool = True
    zerolag_bypass: bool = True
    decoupled_heads: bool = True
    zmem_isolation: bool = True
    enable_assoc_loss: bool = True
    regularizer: str = "epps_pulley_sigreg"

    def as_wave6(self) -> Wave6Interventions:
        return Wave6Interventions(
            key_shield=self.key_shield,
            zerolag_bypass=self.zerolag_bypass,
            decoupled_heads=self.decoupled_heads,
            regularizer=self.regularizer,
        )


def _fork_isolated_streams_impl(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    return z, z


@jax.custom_vjp
def _fork_isolated_streams_vjp(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _fork_isolated_streams_impl(z)


def _fork_fwd(z: jax.Array) -> tuple[tuple[jax.Array, jax.Array], None]:
    return (z, z), None


def _fork_bwd(_res: None, grads: tuple[jax.Array, jax.Array]) -> tuple[jax.Array]:
    g_mem, g_task = grads
    # Strictly block rank-2 goal regression, JEPA, and control gradients from entering top-level SLOT_TEXT.
    g_task_shielded = g_task.at[:, :, SLOT_TEXT, :].set(jnp.zeros_like(g_task[:, :, SLOT_TEXT, :]))
    return (g_mem + g_task_shielded,)


_fork_isolated_streams_vjp.defvjp(_fork_fwd, _fork_bwd)


def fork_isolated_latent_streams(z: jax.Array, zmem_isolation: bool = True) -> tuple[jax.Array, jax.Array]:
    """Fork unified trunk field `Z` into `(Z_mem, Z_task)` with single-backward `SLOT_TEXT` gradient isolation."""
    if not zmem_isolation:
        return z, z
    return _fork_isolated_streams_vjp(z)


def _spatial_cross_modal_attn_wave8a(
    x: jax.Array,
    qkv_w: jax.Array,
    o_w: jax.Array,
    n_heads: int,
    head_dim: int,
    *,
    separate_cores: bool = False,
    shield_text_slot: bool = True,
    zmem_isolation: bool = True,
    layer_idx: int = 0,
) -> jax.Array:
    """Within-tick attention across S=24 slots with strict across-depth SLOT_TEXT gradient isolation."""
    b, t, s, d = x.shape
    if zmem_isolation and shield_text_slot and (not separate_cores):
        # Forward values are 100% identical to `x`. At layer 0 only, 0.20x trains disjoint non-associative
        # rows of `text_emb`; across layers 1..L-1, exact stop_gradient blocks all task gradients from
        # entering any trunk layer's SLOT_TEXT computation.
        x_txt = x[:, :, SLOT_TEXT : SLOT_TEXT + 1, :]
        if layer_idx == 0:
            x_txt_damped = (
                0.20 * x_txt.astype(jnp.float32) + 0.80 * jax.lax.stop_gradient(x_txt.astype(jnp.float32))
            ).astype(x.dtype)
        else:
            x_txt_damped = jax.lax.stop_gradient(x_txt)
        x_for_emb = jnp.concatenate([x[:, :, :SLOT_TEXT, :], x_txt_damped, x[:, :, SLOT_TEXT + 1 :, :]], axis=2)

        qkv_emb = (x_for_emb @ qkv_w).reshape(b * t, s, 3, n_heads, head_dim)
        q, k, v = qkv_emb[:, :, 0], qkv_emb[:, :, 1], qkv_emb[:, :, 2]
        scores = jnp.einsum("bshd,bthd->bhst", q, k).astype(jnp.float32) * (head_dim**-0.5)
        mask = _shield_text_slot_mask()[None, None, :, :]
        probs = jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1).astype(x.dtype)
        out_emb = jnp.einsum("bhst,bthd->bshd", probs, v).reshape(b, t, s, d)

        # For SLOT_TEXT attending strictly to itself (1-element softmax == 1.0), keep 100% unattenuated gradient!
        qkv_txt = (x_txt @ qkv_w).reshape(b, t, 1, 3, n_heads, head_dim)
        v_txt = qkv_txt[:, :, :, 2].reshape(b, t, 1, d)
        out_all = out_emb.at[:, :, SLOT_TEXT : SLOT_TEXT + 1, :].set(v_txt)
        return out_all @ o_w

    qkv = (x @ qkv_w).reshape(b * t, s, 3, n_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scores = jnp.einsum("bshd,bthd->bhst", q, k).astype(jnp.float32) * (head_dim**-0.5)
    if separate_cores:
        mask = _separate_modality_mask()[None, None, :, :]
        scores = jnp.where(mask, scores, -1e9)
    elif shield_text_slot:
        mask = _shield_text_slot_mask()[None, None, :, :]
        scores = jnp.where(mask, scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("bhst,bthd->bshd", probs, v).reshape(b, t, s, d)
    return out @ o_w


def _sym_head_split(n_heads: int) -> tuple[int, int]:
    """Return `(h_sym, h_long)` head boundaries (`H_sym = [0:h_sym]`, `H_goal = [h_sym:h_long]`, `H_phys = [h_long:H]`)."""
    h_long = max(2, n_heads // 2)
    h_sym = max(1, (2 * h_long) // 3)
    if h_sym >= h_long:
        h_sym = h_long - 1
    return h_sym, h_long


def _gated_delta_temporal_wave8a(
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
    zmem_isolation: bool = True,
    prev_state: tuple[jax.Array, jax.Array] | None = None,
    modality_mask: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array], jax.Array]:
    """GatedDelta matrix recurrence with `H_sym ⊕ H_goal ⊕ H_phys` isolation when `zmem_isolation=True`."""
    if not zmem_isolation:
        return _gated_delta_temporal_wave6(
            x,
            tq_w,
            tkv_write_w,
            tslot_u_w,
            tgates_w,
            tgates_b,
            to_w,
            valid,
            n_heads,
            head_dim,
            key_shield=key_shield,
            zerolag_bypass=zerolag_bypass,
            prev_state=prev_state,
            modality_mask=modality_mask,
        )

    b, t, s, d = x.shape
    h_sym, h_long = _sym_head_split(n_heads)
    h_goal = h_long - h_sym
    h_emb = n_heads - h_long

    aud_p = (
        jnp.ones((b, t, 1), dtype=jnp.float32)
        if modality_mask is None
        else modality_mask[:, :, 0:1].astype(jnp.float32)
    )
    txt_p = (
        jnp.ones((b, t, 1), dtype=jnp.float32)
        if modality_mask is None
        else modality_mask[:, :, 3:4].astype(jnp.float32)
    )

    tq_reshaped = tq_w.reshape(d, n_heads, head_dim)
    q_raw = jnp.einsum("btsd,dhk->btshk", x, tq_reshaped)
    q_all = q_raw * jax.lax.rsqrt(
        jnp.sum(q_raw.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6
    ).astype(x.dtype)

    # Allow SLOTS_REGISTER to query H_goal using blended register + symbolic query key when text is present,
    # and pure register query when text is masked (`audio-only` grounding), with zero gradient leak into SLOT_TEXT.
    txt_slot = x[:, :, SLOT_TEXT, :]
    txt_gq_src = jax.lax.stop_gradient(txt_slot)
    q_txt_g_raw = jnp.einsum("btd,dhk->bthk", txt_gq_src, tq_reshaped[:, h_sym:h_long, :]).astype(jnp.float32)
    q_txt_g = q_txt_g_raw * jax.lax.rsqrt(jnp.sum(q_txt_g_raw * q_txt_g_raw, axis=-1, keepdims=True) + 1e-6)
    q_reg_g = q_all[:, :, SLOTS_REGISTER, h_sym:h_long, :].astype(jnp.float32)
    q_blend = 0.35 * q_reg_g + 0.65 * q_txt_g[:, :, None, :, :]
    q_blend = q_blend * jax.lax.rsqrt(jnp.sum(q_blend * q_blend, axis=-1, keepdims=True) + 1e-6)
    q_use = jnp.where(txt_p[:, :, None, None, :] > 0.5, q_blend, q_reg_g)
    q_all = q_all.at[:, :, SLOTS_REGISTER, h_sym:h_long, :].set(q_use.astype(x.dtype))

    u_slot = x @ tslot_u_w
    reg_mean = x[:, :, SLOTS_REGISTER, :].mean(axis=2)
    txt_goal_src = jax.lax.stop_gradient(txt_slot)
    aud_goal_src = x[:, :, SLOTS_AUDIO, :].mean(axis=2)
    lang_goal_src = txt_p.astype(x.dtype) * txt_goal_src + (aud_p * (1.0 - txt_p)).astype(x.dtype) * aud_goal_src

    if key_shield:
        write_src_sym = txt_slot
        write_src_goal = lang_goal_src
    else:
        vis_m = x[:, :, SLOTS_VISUAL, :].mean(axis=2)
        write_src_sym = 0.55 * txt_slot + 0.45 * vis_m
        write_src_goal = 0.55 * lang_goal_src + 0.45 * vis_m

    write_src_emb = (
        0.60 * reg_mean
        + 0.22 * x[:, :, SLOT_PROPRIO, :]
        + 0.18 * x[:, :, SLOTS_VISUAL, :].mean(axis=2)
    )

    w_sym_gates = jnp.concatenate([tgates_w[:, :h_sym], tgates_w[:, n_heads : n_heads + h_sym]], axis=-1)
    b_sym_gates = jnp.concatenate([tgates_b[:h_sym], tgates_b[n_heads : n_heads + h_sym]], axis=-1)
    gates_sym = (write_src_sym @ w_sym_gates).astype(jnp.float32) + b_sym_gates[None, None, :]

    w_goal_gates = jnp.concatenate(
        [tgates_w[:, h_sym:h_long], tgates_w[:, n_heads + h_sym : n_heads + h_long]], axis=-1
    )
    b_goal_gates = jnp.concatenate(
        [tgates_b[h_sym:h_long], tgates_b[n_heads + h_sym : n_heads + h_long]], axis=-1
    )
    gates_goal = (write_src_goal @ w_goal_gates).astype(jnp.float32) + b_goal_gates[None, None, :]

    w_emb_gates = jnp.concatenate(
        [tgates_w[:, h_long:n_heads], tgates_w[:, n_heads + h_long : 2 * n_heads], tgates_w[:, 2 * n_heads :]],
        axis=-1,
    )
    b_emb_gates = jnp.concatenate(
        [tgates_b[h_long:n_heads], tgates_b[n_heads + h_long : 2 * n_heads], tgates_b[2 * n_heads :]],
        axis=-1,
    )
    gates_emb = (write_src_emb @ w_emb_gates).astype(jnp.float32) + b_emb_gates[None, None, :]

    alpha = jax.nn.sigmoid(
        jnp.concatenate([gates_sym[:, :, :h_sym], gates_goal[:, :, :h_goal], gates_emb[:, :, :h_emb]], axis=-1)
    )
    beta = jax.nn.sigmoid(
        jnp.concatenate([gates_sym[:, :, h_sym:], gates_goal[:, :, h_goal:], gates_emb[:, :, h_emb : 2 * h_emb]], axis=-1)
    )
    gamma_raw = gates_emb[:, :, 2 * h_emb :]

    kv_w_reshaped = tkv_write_w.reshape(d, 2, n_heads, head_dim)
    kv_sym = jnp.einsum("btd,dchk->btchk", write_src_sym, kv_w_reshaped[:, :, :h_sym, :])
    kv_goal = jnp.einsum("btd,dchk->btchk", write_src_goal, kv_w_reshaped[:, :, h_sym:h_long, :])
    kv_emb = jnp.einsum("btd,dchk->btchk", write_src_emb, kv_w_reshaped[:, :, h_long:, :])
    kv_write = jnp.concatenate([kv_sym, kv_goal, kv_emb], axis=3)
    k_w = kv_write[:, :, 0].astype(jnp.float32)
    v_w = kv_write[:, :, 1].astype(jnp.float32)
    k_w = k_w * jax.lax.rsqrt(jnp.sum(k_w * k_w, axis=-1, keepdims=True) + 1e-6)

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
            # Strict H_sym vs (H_goal, H_phys) read isolation:
            # SLOT_TEXT reads ONLY H_sym [0:h_sym]; embodied & register slots read ONLY [h_sym:n_heads].
            assoc_heads = assoc_heads.at[:, SLOT_TEXT, h_sym:, :].set(0.0)
            assoc_heads = assoc_heads.at[:, :SLOT_TEXT, :h_sym, :].set(0.0)
            assoc_heads = assoc_heads.at[:, SLOT_TEXT + 1 :, :h_sym, :].set(0.0)
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
    return out, (m_last, c_last), k_w[:, :, :h_sym, :]


def _diagonal_ssm_temporal_wave8a(
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
    zmem_isolation: bool = True,
    prev_state: tuple[jax.Array, jax.Array] | None = None,
    modality_mask: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Diagonal vector SSM recurrence with matched `H_sym ⊕ H_goal ⊕ H_phys` head isolation (`0.00%` param delta)."""
    if not zmem_isolation:
        return _diagonal_ssm_temporal(
            x, tq_w, tkv_write_w, tslot_u_w, tgates_w, tgates_b, to_w, valid, n_heads, head_dim, prev_state=prev_state
        )

    b, t, s, d = x.shape
    h_sym, h_long = _sym_head_split(n_heads)
    h_goal = h_long - h_sym
    h_emb = n_heads - h_long

    aud_p = (
        jnp.ones((b, t, 1), dtype=jnp.float32)
        if modality_mask is None
        else modality_mask[:, :, 0:1].astype(jnp.float32)
    )
    txt_p = (
        jnp.ones((b, t, 1), dtype=jnp.float32)
        if modality_mask is None
        else modality_mask[:, :, 3:4].astype(jnp.float32)
    )

    tq_reshaped = tq_w.reshape(d, n_heads, head_dim)
    q_raw = jnp.einsum("btsd,dhk->btshk", x, tq_reshaped)
    q_all = q_raw * jax.lax.rsqrt(
        jnp.sum(q_raw.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6
    ).astype(x.dtype)

    txt_slot = x[:, :, SLOT_TEXT, :]
    txt_gq_src = jax.lax.stop_gradient(txt_slot)
    q_txt_g_raw = jnp.einsum("btd,dhk->bthk", txt_gq_src, tq_reshaped[:, h_sym:h_long, :]).astype(jnp.float32)
    q_txt_g = q_txt_g_raw * jax.lax.rsqrt(jnp.sum(q_txt_g_raw * q_txt_g_raw, axis=-1, keepdims=True) + 1e-6)
    q_reg_g = q_all[:, :, SLOTS_REGISTER, h_sym:h_long, :].astype(jnp.float32)
    q_blend = 0.35 * q_reg_g + 0.65 * q_txt_g[:, :, None, :, :]
    q_blend = q_blend * jax.lax.rsqrt(jnp.sum(q_blend * q_blend, axis=-1, keepdims=True) + 1e-6)
    q_use = jnp.where(txt_p[:, :, None, None, :] > 0.5, q_blend, q_reg_g)
    q_all = q_all.at[:, :, SLOTS_REGISTER, h_sym:h_long, :].set(q_use.astype(x.dtype))

    u_slot = x @ tslot_u_w
    reg_mean = x[:, :, SLOTS_REGISTER, :].mean(axis=2)
    txt_goal_src = jax.lax.stop_gradient(txt_slot)
    aud_goal_src = x[:, :, SLOTS_AUDIO, :].mean(axis=2)
    lang_goal_src = txt_p.astype(x.dtype) * txt_goal_src + (aud_p * (1.0 - txt_p)).astype(x.dtype) * aud_goal_src

    if key_shield:
        write_src_sym = txt_slot
        write_src_goal = lang_goal_src
    else:
        vis_m = x[:, :, SLOTS_VISUAL, :].mean(axis=2)
        write_src_sym = 0.55 * txt_slot + 0.45 * vis_m
        write_src_goal = 0.55 * lang_goal_src + 0.45 * vis_m

    write_src_emb = (
        0.60 * reg_mean
        + 0.22 * x[:, :, SLOT_PROPRIO, :]
        + 0.18 * x[:, :, SLOTS_VISUAL, :].mean(axis=2)
    )

    w_sym_gates = jnp.concatenate([tgates_w[:, :h_sym], tgates_w[:, n_heads : n_heads + h_sym]], axis=-1)
    b_sym_gates = jnp.concatenate([tgates_b[:h_sym], tgates_b[n_heads : n_heads + h_sym]], axis=-1)
    gates_sym = (write_src_sym @ w_sym_gates).astype(jnp.float32) + b_sym_gates[None, None, :]

    w_goal_gates = jnp.concatenate(
        [tgates_w[:, h_sym:h_long], tgates_w[:, n_heads + h_sym : n_heads + h_long]], axis=-1
    )
    b_goal_gates = jnp.concatenate(
        [tgates_b[h_sym:h_long], tgates_b[n_heads + h_sym : n_heads + h_long]], axis=-1
    )
    gates_goal = (write_src_goal @ w_goal_gates).astype(jnp.float32) + b_goal_gates[None, None, :]

    w_emb_gates = jnp.concatenate(
        [tgates_w[:, h_long:n_heads], tgates_w[:, n_heads + h_long : 2 * n_heads], tgates_w[:, 2 * n_heads :]],
        axis=-1,
    )
    b_emb_gates = jnp.concatenate(
        [tgates_b[h_long:n_heads], tgates_b[n_heads + h_long : 2 * n_heads], tgates_b[2 * n_heads :]],
        axis=-1,
    )
    gates_emb = (write_src_emb @ w_emb_gates).astype(jnp.float32) + b_emb_gates[None, None, :]

    alpha = jax.nn.sigmoid(
        jnp.concatenate([gates_sym[:, :, :h_sym], gates_goal[:, :, :h_goal], gates_emb[:, :, :h_emb]], axis=-1)
    )
    beta = jax.nn.sigmoid(
        jnp.concatenate([gates_sym[:, :, h_sym:], gates_goal[:, :, h_goal:], gates_emb[:, :, h_emb : 2 * h_emb]], axis=-1)
    )
    gamma_raw = gates_emb[:, :, 2 * h_emb :]

    kv_w_reshaped = tkv_write_w.reshape(d, 2, n_heads, head_dim)
    kv_sym = jnp.einsum("btd,dchk->btchk", write_src_sym, kv_w_reshaped[:, :, :h_sym, :])
    kv_goal = jnp.einsum("btd,dchk->btchk", write_src_goal, kv_w_reshaped[:, :, h_sym:h_long, :])
    kv_emb = jnp.einsum("btd,dchk->btchk", write_src_emb, kv_w_reshaped[:, :, h_long:, :])
    kv_write = jnp.concatenate([kv_sym, kv_goal, kv_emb], axis=3)
    b_sel = jnp.tanh(kv_write[:, :, 0].astype(jnp.float32))
    v_w = kv_write[:, :, 1].astype(jnp.float32)

    gamma_sens = jax.nn.sigmoid(gamma_raw - (2.5 if zerolag_bypass else 0.0))
    gamma_reg = jax.nn.sigmoid(gamma_raw)

    if prev_state is None:
        s_init = jnp.zeros((b, n_heads, head_dim), dtype=jnp.float32)
        c_init = jnp.zeros((b, s, d), dtype=jnp.float32)
    else:
        s_init, c_init = prev_state[0].astype(jnp.float32), prev_state[1].astype(jnp.float32)

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
        s_prev, c_prev = carry
        q_t, bsel_t, v_t, a_t, b_t, gs_t, gr_t, u_t, ok_t = step_inp
        q_scaled = q_t.astype(jnp.float32) * (head_dim**0.5)
        assoc_heads = q_scaled * s_prev[:, None, :, :]
        if key_shield:
            assoc_heads = assoc_heads.at[:, SLOT_TEXT, h_sym:, :].set(0.0)
            assoc_heads = assoc_heads.at[:, :SLOT_TEXT, :h_sym, :].set(0.0)
            assoc_heads = assoc_heads.at[:, SLOT_TEXT + 1 :, :h_sym, :].set(0.0)
        assoc_out = assoc_heads.reshape(b, s, d)

        s_cand = a_t[:, :, None] * s_prev + b_t[:, :, None] * (bsel_t * v_t)
        g_slot = jnp.concatenate(
            [jnp.broadcast_to(gs_t[:, None, :], (b, 21, d)), jnp.broadcast_to(gr_t[:, None, :], (b, 3, d))],
            axis=1,
        )
        u_f32 = u_t.astype(jnp.float32)
        c_cand = g_slot * c_prev + (1.0 - g_slot) * u_f32
        if key_shield:
            c_cand = c_cand.at[:, SLOT_TEXT, :].set(u_f32[:, SLOT_TEXT, :])

        s_next = jnp.where(ok_t[:, None, None], s_cand, s_prev)
        c_next = jnp.where(ok_t[:, None, None], c_cand, c_prev)
        local_kin = slot_u_weight * u_f32 + slot_c_weight * c_next
        combined = (slot_assoc_gain * assoc_out + local_kin).astype(x.dtype)
        return (s_next, c_next), combined

    (s_last, c_last), seq_out = jax.lax.scan(
        scan_step,
        (s_init, c_init),
        (
            jnp.swapaxes(q_all, 0, 1),
            jnp.swapaxes(b_sel, 0, 1),
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
    return out, (s_last, c_last)


def _run_block_wave8a(
    x: jax.Array,
    layer_idx: int,
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    valid: jax.Array,
    offset: int,
    prev_mem: Any,
    modality_mask: jax.Array | None = None,
) -> tuple[jax.Array, Any, jax.Array | None]:
    dtype = getattr(jnp, spec.dtype)
    pfx = f"L{layer_idx}_"

    use_shield = interventions.key_shield and (
        spec.arch
        in (
            "slwm_x1b",
            "diagonal_ssm_hybrid",
            "slwm_no_registers",
            "sliding_window_transformer",
            "sliding_window_gqa_transformer",
        )
    )
    z_s = rms_norm(x, params[pfx + "snorm"])
    x = x + _spatial_cross_modal_attn_wave8a(
        z_s,
        params[pfx + "sqkv"].astype(dtype),
        params[pfx + "so"].astype(dtype),
        spec.n_heads,
        spec.head_dim,
        separate_cores=(spec.arch == "separate_modality_cores"),
        shield_text_slot=use_shield,
        zmem_isolation=interventions.zmem_isolation,
        layer_idx=layer_idx,
    )

    z_t = rms_norm(x, params[pfx + "tnorm"])
    is_attn = (
        spec.arch in ("dense_transformer", "sliding_window_transformer", "sliding_window_gqa_transformer")
        or ((layer_idx + 1) % 4 == 0)
    )
    sym_keys = None
    if is_attn:
        if spec.arch == "dense_transformer":
            attn_win = spec.rope_Window
        elif spec.arch in ("sliding_window_transformer", "sliding_window_gqa_transformer"):
            attn_win = min(64, spec.rope_Window)
        else:
            attn_win = min(8, spec.rope_Window)
        h_kv = effective_kv_heads(spec) if spec.arch == "sliding_window_gqa_transformer" else spec.n_heads
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
            n_kv_heads=h_kv,
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
        t_out, mem_out = _diagonal_ssm_temporal_wave8a(
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
            zmem_isolation=interventions.zmem_isolation,
            prev_state=prev_mem,
            modality_mask=modality_mask,
        )
    else:
        t_out, mem_out, sym_keys = _gated_delta_temporal_wave8a(
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
            zmem_isolation=interventions.zmem_isolation,
            prev_state=prev_mem,
            modality_mask=modality_mask,
        )
    x = x + t_out

    z_f = rms_norm(x, params[pfx + "fnorm"])
    gate = jax.nn.silu(z_f @ params[pfx + "fgate"].astype(dtype))
    up = z_f @ params[pfx + "fup"].astype(dtype)
    x = (x + (gate * up) @ params[pfx + "fdown"].astype(dtype)) * valid[:, :, None, None].astype(dtype)
    return x, mem_out, sym_keys


def forward_world_model_wave8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    batch: dict[str, jax.Array],
    interventions: Wave8AInterventions,
    *,
    memories: tuple[Any, ...] | None = None,
    offset: int = 0,
    return_symbolic_keys: bool = False,
) -> tuple[jax.Array, tuple[Any, ...], jax.Array | None]:
    """Causal world-model trunk forward pass with `Z_mem ⊕ Z_task` head isolation and GQA support."""
    dtype = getattr(jnp, spec.dtype)
    x = encode_signals(params, spec, batch)
    b, t = x.shape[:2]
    valid = batch.get("valid")
    valid = jnp.ones((b, t), dtype=bool) if valid is None else valid.astype(bool)
    mm_raw = batch.get("modality_mask")
    if mm_raw is None:
        mm_norm = jnp.ones((b, t, 4), dtype=jnp.float32)
    elif mm_raw.ndim == 2:
        mm_norm = jnp.broadcast_to(mm_raw[:, None, :].astype(jnp.float32), (b, t, 4))
    else:
        mm_norm = mm_raw.astype(jnp.float32)
    mm_norm = mm_norm * valid[:, :, None].astype(jnp.float32)

    x = x * valid[:, :, None, None].astype(dtype)
    next_memories = []
    last_sym_keys = None

    for i in range(spec.layers):
        prev = None if memories is None else memories[i]
        if spec.grad_checkpoint and memories is None and not return_symbolic_keys:
            block_fn = jax.checkpoint(
                lambda x_in, p_in, v_in, mm_in, idx=i: _run_block_wave8a(
                    x_in, idx, p_in, spec, interventions, v_in, offset, None, modality_mask=mm_in
                )[0]
            )
            x = block_fn(x, params, valid, mm_norm)
            next_memories.append(None)
        else:
            x, m_out, sk = _run_block_wave8a(
                x, i, params, spec, interventions, valid, offset, prev, modality_mask=mm_norm
            )
            next_memories.append(m_out)
            if sk is not None:
                last_sym_keys = sk

    z = rms_norm(x, params["final_norm"])
    if spec.arch == "legacy_scalar_hybrid":
        z = jnp.broadcast_to(z.mean(axis=2, keepdims=True), z.shape)
    return z, tuple(next_memories), last_sym_keys


def decode_goal_latch_wave8a(
    params: dict[str, jax.Array],
    z_task: jax.Array,
    interventions: Wave8AInterventions,
) -> jax.Array:
    """Decode persistent 2D target goal / robot displacement without rank-2 crushing `SLOT_TEXT`."""
    zf = z_task.astype(jnp.float32)
    if interventions.decoupled_heads:
        reg_sum = jnp.mean(zf[..., SLOTS_REGISTER, :], axis=-2)
        aud_sum = jnp.mean(zf[..., SLOTS_AUDIO, :], axis=-2)
        if interventions.zmem_isolation:
            # Read exclusively from persistent working-memory registers (fed by H_goal) + spoken audio;
            # zero direct SLOT_TEXT readout prevents rank-2 head_goal_w from collapsing symbolic recall.
            g_ctx = 0.85 * reg_sum + 0.15 * aud_sum
        else:
            txt_lat = zf[..., SLOT_TEXT, :]
            g_ctx = 0.45 * reg_sum + 0.40 * txt_lat + 0.15 * aud_sum
    else:
        g_ctx = jnp.mean(zf, axis=-2)
    return jnp.tanh(g_ctx @ params["head_goal_w"].astype(jnp.float32) + params["head_goal_b"].astype(jnp.float32))


def decode_target_block_xy_wave8a(
    params: dict[str, jax.Array],
    z_task: jax.Array,
    interventions: Wave8AInterventions,
    *,
    pure_v_8x8: jax.Array | None = None,
) -> jax.Array:
    """Decode 2D tabletop block coordinate `[-1, 1]^2` via cross-modal `q_lang x k_vis` spatial soft-argmax (`0` extra params).

    Uses reserved columns `8:16` of `head_proprio_w` (`[D, 16]`) and `head_proprio_b` (`[16]`):
      - `8:11`: shift-equivariant mean-centered visual cell appearance key projection (`[D, 3]`)
      - `11:14`: spoken-audio + working-memory register color query projection (`[D, 3]`)
      - `14:16`: attended visual cell sub-grid refinement (`[D, 2]`)
    Swapping the spoken/text color word shifts spatial attention across visible candidate blocks without any hardcoded
    blank-image clamp, and equivariantly tracks `180`-degree spatial flips of the tabletop scene.
    """
    zf = z_task.astype(jnp.float32)
    head_pw = params["head_proprio_w"].astype(jnp.float32)
    head_pb = params["head_proprio_b"].astype(jnp.float32)

    if not interventions.decoupled_heads:
        pooled = jnp.mean(zf, axis=-2)
        return jnp.tanh(pooled @ head_pw[:, 14:16] + head_pb[14:16])

    reg_sum = jnp.mean(zf[..., SLOTS_REGISTER, :], axis=-2)
    aud_sum = jnp.mean(zf[..., SLOTS_AUDIO, :], axis=-2)
    if interventions.zmem_isolation:
        lang_ctx = 0.85 * reg_sum + 0.15 * aud_sum
    else:
        txt_lat = zf[..., SLOT_TEXT, :]
        lang_ctx = 0.50 * reg_sum + 0.30 * txt_lat + 0.20 * aud_sum
    q_lang = lang_ctx @ head_pw[:, 11:14] + head_pb[11:14]

    d = zf.shape[-1]
    if pure_v_8x8 is not None:
        v_cells = pure_v_8x8.astype(jnp.float32).reshape(*pure_v_8x8.shape[:-3], 64, d)
        centers = canonical_grid_centers_8x8()
        fine_scale = 0.16
    else:
        pos = _spatial_grid_fourier(d, jnp.float32)
        v_cells = zf[..., SLOTS_VISUAL, :] - pos
        centers = canonical_patch_centers_4x4()
        fine_scale = 0.30

    v_centered = v_cells - jnp.mean(v_cells, axis=-2, keepdims=True)
    k_vis = v_centered @ head_pw[:, 8:11]
    logits = jnp.einsum("...sr,...r->...s", k_vis, q_lang)
    attn = jax.nn.softmax(logits, axis=-1)
    coarse_xy = jnp.einsum("...s,sc->...c", attn, centers)
    vis_att = jnp.einsum("...s,...sd->...d", attn, v_centered)
    fine_xy = fine_scale * jnp.tanh(vis_att @ head_pw[:, 14:16] + head_pb[14:16])
    return jnp.clip(coarse_xy + fine_xy, -0.96, 0.96)


def init_decoder_streaming_carry(spec: X1BSpec, batch_size: int) -> dict[str, jax.Array]:
    """Initialize causal streaming carry for `decode_world_heads_wave8a` across chunk boundaries."""
    dtype = getattr(jnp, spec.dtype)
    return {
        "has_prev_tick": jnp.zeros((batch_size,), dtype=bool),
        "modality_seen": jnp.zeros((batch_size, 4), dtype=bool),
        "prev_z_next_pred": jnp.zeros((batch_size, 1, spec.slots_per_tick, spec.d_model), dtype=dtype),
        "prev_log_var_next": jnp.zeros((batch_size, 1, spec.slots_per_tick, 1), dtype=jnp.float32),
    }


def extract_decoder_streaming_carry(heads: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Extract the updated streaming carry dictionary from `decode_world_heads_wave8a` outputs."""
    return {
        "has_prev_tick": heads["next_has_prev_tick"],
        "modality_seen": heads["next_modality_seen"],
        "prev_z_next_pred": heads["next_z_next_pred_vis_masked"],
        "prev_log_var_next": heads["next_log_var_next"],
    }


def decode_world_heads_wave8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_mem: jax.Array,
    z_task: jax.Array,
    z_next_pred: jax.Array,
    log_var_next: jax.Array,
    goal: jax.Array,
    interventions: Wave8AInterventions,
    *,
    obs_ee_proprio: jax.Array | None = None,
    obs_video: jax.Array | None = None,
    obs_modality_mask: jax.Array | None = None,
    obs_valid: jax.Array | None = None,
    decoder_carry: dict[str, jax.Array] | None = None,
    prev_modality_seen: jax.Array | None = None,
    prev_z_next_pred: jax.Array | None = None,
    prev_log_var_next: jax.Array | None = None,
    stop_grad_diagnostics: bool = True,
) -> dict[str, jax.Array]:
    """Decode symbolic recall from `z_mem` and spatial/control/safety heads from `z_task`.

    When `obs_modality_mask` is provided:
      - Instantaneous sensor bypasses (`obs_video`, `obs_ee_proprio`, `vis_ee_disc`) are gated by per-tick
        `vis_obs_m` / `prop_obs_m` (`0.0` when dropped at tick `t`).
      - Latent modality slot fallbacks (`z_vis_masked`, `z_task_vis_masked`, `z_aud_masked`, `prop_lat`) are gated
        by causal prefix visibility (`prev_modality_seen | (cumsum(obs_m) > 0.5)`), preventing ungrounded fallback
        on streams that have never observed that modality while preserving recurrent world-model belief across both
        full-sequence and multi-chunk streaming camera/sensor occlusions.
      - Innovation gate shift history (`prev_z_next_pred`, `prev_log_var_next`) is carried across chunk boundaries
        via `decoder_carry` / `extract_decoder_streaming_carry`.
    """
    b, t, _, _ = z_task.shape
    zf = z_task.astype(jnp.float32)
    zmf = z_mem.astype(jnp.float32)
    z_next_f = z_next_pred.astype(jnp.float32)
    w6_iv = interventions.as_wave6()

    valid_bool = jnp.ones((b, t), dtype=bool) if obs_valid is None else obs_valid.astype(bool)
    valid_f = valid_bool.astype(jnp.float32)

    if decoder_carry is not None and "has_prev_tick" in decoder_carry:
        has_prev = decoder_carry["has_prev_tick"].astype(bool)
    elif prev_z_next_pred is not None or prev_log_var_next is not None:
        has_prev = jnp.ones((b,), dtype=bool)
    else:
        has_prev = jnp.zeros((b,), dtype=bool)

    if prev_modality_seen is not None:
        prev_seen = prev_modality_seen.astype(bool)
    elif decoder_carry is not None and "modality_seen" in decoder_carry:
        prev_seen = decoder_carry["modality_seen"].astype(bool)
    else:
        prev_seen = jnp.zeros((b, 4), dtype=bool)

    if prev_z_next_pred is not None:
        prev_zn = prev_z_next_pred
    elif decoder_carry is not None and "prev_z_next_pred" in decoder_carry:
        prev_zn = decoder_carry["prev_z_next_pred"]
    else:
        prev_zn = jnp.zeros((b, 1, z_task.shape[2], z_task.shape[3]), dtype=z_task.dtype)

    if prev_log_var_next is not None:
        prev_lv = prev_log_var_next
    elif decoder_carry is not None and "prev_log_var_next" in decoder_carry:
        prev_lv = decoder_carry["prev_log_var_next"]
    else:
        prev_lv = jnp.zeros((b, 1, log_var_next.shape[2], log_var_next.shape[3]), dtype=log_var_next.dtype)

    if obs_modality_mask is not None:
        mm_f = obs_modality_mask.astype(jnp.float32) * valid_f[:, :, None]
        aud_obs_m = mm_f[:, :, 0]
        vis_obs_m = mm_f[:, :, 1]
        prop_obs_m = mm_f[:, :, 2]
        cum_all_bool = prev_seen[:, None, :] | (jnp.cumsum(mm_f, axis=1) > 0.5)
    else:
        aud_obs_m = valid_f
        vis_obs_m = valid_f
        prop_obs_m = valid_f
        cum_all_bool = prev_seen[:, None, :] | jnp.broadcast_to((jnp.cumsum(valid_f, axis=1) > 0.5)[:, :, None], (b, t, 4))

    cum_aud_seen = cum_all_bool[:, :, 0].astype(jnp.float32)
    cum_vis_seen = cum_all_bool[:, :, 1].astype(jnp.float32)
    cum_prop_seen = cum_all_bool[:, :, 2].astype(jnp.float32)
    cum_txt_seen = cum_all_bool[:, :, 3].astype(jnp.float32)
    next_modality_seen = cum_all_bool[:, -1, :]

    z_vis_masked = zf[:, :, SLOTS_VISUAL, :] * cum_vis_seen[:, :, None, None]
    z_next_vis_masked = z_next_f[:, :, SLOTS_VISUAL, :] * cum_vis_seen[:, :, None, None]
    z_aud_masked = zf[:, :, SLOTS_AUDIO, :] * cum_aud_seen[:, :, None, None]
    z_task_vis_masked = z_task.at[:, :, SLOTS_VISUAL, :].set(
        z_task[:, :, SLOTS_VISUAL, :] * cum_vis_seen[:, :, None, None].astype(z_task.dtype)
    )
    z_next_pred_vis_masked = z_next_pred.at[:, :, SLOTS_VISUAL, :].set(
        z_next_pred[:, :, SLOTS_VISUAL, :] * cum_vis_seen[:, :, None, None].astype(z_next_pred.dtype)
    )

    if interventions.decoupled_heads:
        vis_sum = jnp.mean(z_vis_masked, axis=2)
        vis_next_sum = jnp.mean(z_next_vis_masked, axis=2)
        vis_max = jnp.max(z_vis_masked, axis=2)
        aud_sum = jnp.mean(z_aud_masked, axis=2)
        reg_sum = jnp.mean(zf[:, :, SLOTS_REGISTER, :], axis=2)
        reg_next_sum = jnp.mean(z_next_f[:, :, SLOTS_REGISTER, :], axis=2)
        prop_lat = zf[:, :, SLOT_PROPRIO, :] * cum_prop_seen[:, :, None]
        prop_next_lat = z_next_f[:, :, SLOT_PROPRIO, :] * cum_prop_seen[:, :, None]
        txt_ctx = zmf[:, :, SLOT_TEXT, :]
        contact_ctx = 0.45 * vis_max + 0.25 * vis_sum + 0.15 * aud_sum + 0.15 * reg_sum
    else:
        zf_all_masked = (
            zf.at[:, :, SLOTS_VISUAL, :]
            .set(z_vis_masked)
            .at[:, :, SLOTS_AUDIO, :]
            .set(z_aud_masked)
            .at[:, :, SLOT_PROPRIO, :]
            .set(zf[:, :, SLOT_PROPRIO, :] * cum_prop_seen[:, :, None])
            .at[:, :, SLOT_TEXT, :]
            .set(zf[:, :, SLOT_TEXT, :] * cum_txt_seen[:, :, None])
        )
        z_next_all_masked = (
            z_next_f.at[:, :, SLOTS_VISUAL, :]
            .set(z_next_vis_masked)
            .at[:, :, SLOTS_AUDIO, :]
            .set(z_next_f[:, :, SLOTS_AUDIO, :] * cum_aud_seen[:, :, None, None])
            .at[:, :, SLOT_PROPRIO, :]
            .set(z_next_f[:, :, SLOT_PROPRIO, :] * cum_prop_seen[:, :, None])
            .at[:, :, SLOT_TEXT, :]
            .set(z_next_f[:, :, SLOT_TEXT, :] * cum_txt_seen[:, :, None])
        )
        pooled = jnp.mean(zf_all_masked, axis=2)
        pooled_next = jnp.mean(z_next_all_masked, axis=2)
        vis_sum = vis_max = aud_sum = reg_sum = prop_lat = contact_ctx = pooled
        txt_ctx = jnp.mean(zmf.at[:, :, SLOT_TEXT, :].set(zmf[:, :, SLOT_TEXT, :] * cum_txt_seen[:, :, None]), axis=2)
        vis_next_sum = reg_next_sum = prop_next_lat = pooled_next

    lat_ee_curr, lat_obj_curr, lat_proprio_curr = decode_spatial_vis_pose(
        params, z_task_vis_masked, w6_iv, pure_v_patches=None
    )
    vis_ee_next, vis_obj_next, _ = decode_spatial_vis_pose(
        params, z_next_pred_vis_masked, w6_iv, pure_v_patches=None
    )

    if obs_video is not None:
        obs_vid_eff = jnp.where(vis_obs_m[:, :, None, None, None] > 0.5, obs_video, jnp.zeros_like(obs_video))
        pure_vp_ti = encode_pure_visual_patches(params, spec, obs_vid_eff, include_pos=False)
        pure_v8 = encode_pure_visual_grid_8x8(params, spec, obs_vid_eff)
        pv_ee_curr, pv_obj_curr, pv_proprio_curr = decode_spatial_vis_pose(
            params, z_task_vis_masked, w6_iv, pure_v_patches=pure_vp_ti, pure_v_8x8=pure_v8
        )
        vis_ee_curr = jnp.where(vis_obs_m[:, :, None] > 0.5, pv_ee_curr, lat_ee_curr)
        vis_obj_curr = jnp.where(vis_obs_m[:, :, None] > 0.5, pv_obj_curr, lat_obj_curr)
        vis_proprio_curr = jnp.where(vis_obs_m[:, :, None] > 0.5, pv_proprio_curr, lat_proprio_curr)
    else:
        pure_vp_ti = None
        pure_v8 = None
        vis_ee_curr, vis_obj_curr, vis_proprio_curr = lat_ee_curr, lat_obj_curr, lat_proprio_curr

    if obs_ee_proprio is not None:
        obs_prop_eff = jnp.where(prop_obs_m[:, :, None] > 0.5, obs_ee_proprio.astype(jnp.float32), 0.0)
        ee_for_loc = jnp.where(
            prop_obs_m[:, :, None] > 0.5, obs_prop_eff[:, :, 0:2], jax.lax.stop_gradient(vis_ee_curr)
        )
    else:
        obs_prop_eff = None
        ee_for_loc = jax.lax.stop_gradient(vis_ee_curr)

    if interventions.decoupled_heads:
        loc_vis_lat = spatial_grid_local_context(
            z_vis_masked,
            jax.lax.stop_gradient(vis_obj_curr),
            ee_for_loc,
            pure_v_ti=None,
        )
        if pure_vp_ti is not None:
            loc_vis_pv = spatial_grid_local_context(
                z_vis_masked,
                jax.lax.stop_gradient(vis_obj_curr),
                ee_for_loc,
                pure_v_ti=pure_vp_ti,
            )
            loc_vis_ctx = jnp.where(vis_obs_m[:, :, None] > 0.5, loc_vis_pv, loc_vis_lat)
        else:
            loc_vis_ctx = loc_vis_lat
        contact_ctx = 0.72 * loc_vis_ctx + 0.14 * aud_sum + 0.14 * reg_sum

    pred_goal_xy = decode_goal_latch_wave8a(params, z_task, interventions)
    g_f = goal.astype(jnp.float32)
    if g_f.ndim == 2:
        g_f = g_f[:, None, :]
    goal_bt = jnp.broadcast_to(g_f, (b, t, spec.goal_dim))
    has_ext_goal = (jnp.sum(jnp.abs(goal_bt), axis=-1, keepdims=True) > 1e-4).astype(jnp.float32)
    eff_goal_xy = jnp.where(has_ext_goal > 0.5, goal_bt[:, :, 2:4], jax.lax.stop_gradient(pred_goal_xy))

    head_pw = params["head_proprio_w"].astype(jnp.float32)
    head_pb = params["head_proprio_b"].astype(jnp.float32)

    proprio_curr = vis_proprio_curr
    if obs_prop_eff is not None:
        raw_clean_prop = 2.5 * (obs_prop_eff @ params["prop_w"].astype(jnp.float32))
        clean_prop_emb = jnp.where(prop_obs_m[:, :, None] > 0.5, raw_clean_prop, prop_lat)
    else:
        clean_prop_emb = prop_lat
    proprio_trusted = (0.7 * clean_prop_emb + 0.3 * reg_sum) @ head_pw + head_pb

    lat_prop_next = (0.75 * prop_next_lat + 0.25 * vis_next_sum) @ head_pw + head_pb
    lat_prop_next_trusted = (0.80 * prop_next_lat + 0.20 * reg_next_sum) @ head_pw + head_pb
    if obs_prop_eff is not None and interventions.zerolag_bypass:
        ee_now = obs_prop_eff[:, :, 0:2]
        obj_vis_now = jax.lax.stop_gradient(vis_obj_curr)
        anchor_state = obs_prop_eff.at[:, :, 0:2].set(ee_now).at[:, :, 2:4].set(obj_vis_now)
        delta_full = (0.75 * (prop_next_lat - prop_lat) + 0.25 * (vis_next_sum - vis_sum)) @ head_pw
        delta_trust = (0.80 * (prop_next_lat - prop_lat) + 0.20 * (reg_next_sum - reg_sum)) @ head_pw
        raw_next = anchor_state + delta_full
        blended_obj_next = 0.65 * raw_next[:, :, 2:4] + 0.35 * vis_obj_next
        byp_next = raw_next.at[:, :, 2:4].set(blended_obj_next)
        byp_next_trusted = anchor_state + delta_trust
        proprio_next = jnp.where(prop_obs_m[:, :, None] > 0.5, byp_next, lat_prop_next)
        proprio_next_trusted = jnp.where(prop_obs_m[:, :, None] > 0.5, byp_next_trusted, lat_prop_next_trusted)
    else:
        proprio_next = lat_prop_next
        proprio_next_trusted = lat_prop_next_trusted

    contact_logits = (
        contact_ctx @ params["head_contact_w"].astype(jnp.float32)
        + params["head_contact_b"].astype(jnp.float32)
    )

    ee_xy = ee_for_loc
    vis_obj_xy = jax.lax.stop_gradient(vis_obj_curr)
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

    first_z_pred = jnp.where(
        has_prev[:, None, None, None],
        prev_zn.astype(z_task_vis_masked.dtype),
        z_task_vis_masked[:, :1],
    )
    first_lv = jnp.where(
        has_prev[:, None, None, None],
        prev_lv.astype(log_var_next.dtype),
        log_var_next[:, :1],
    )
    z_pred_shifted = jnp.concatenate([first_z_pred, z_next_pred_vis_masked[:, :-1]], axis=1)
    lv_shifted = jnp.concatenate([first_lv, log_var_next[:, :-1]], axis=1)
    raw_vis_ee_disc = jnp.sqrt(jnp.sum((vis_ee_curr - ee_xy) ** 2, axis=-1) + 1e-6)
    vis_ee_disc = raw_vis_ee_disc * vis_obs_m * prop_obs_m
    innov_feats = compute_innovation_features(
        params,
        z_task_vis_masked,
        jax.lax.stop_gradient(z_pred_shifted),
        jax.lax.stop_gradient(lv_shifted),
        vis_pose_discrepancy=jax.lax.stop_gradient(vis_ee_disc),
    )

    any_v = jnp.any(valid_bool, axis=1)
    t_arange = jnp.arange(t, dtype=jnp.int32)[None, :]
    last_v_idx = jnp.max(jnp.where(valid_bool, t_arange, 0), axis=1)
    b_arange = jnp.arange(b, dtype=jnp.int32)
    cand_zn = z_next_pred_vis_masked[b_arange, last_v_idx][:, None, :, :]
    cand_lv = log_var_next[b_arange, last_v_idx][:, None, :, :]
    next_z_pred_carry = jnp.where(any_v[:, None, None, None], cand_zn, prev_zn.astype(cand_zn.dtype))
    next_lv_carry = jnp.where(any_v[:, None, None, None], cand_lv, prev_lv.astype(cand_lv.dtype))
    next_has_prev_tick = has_prev | any_v

    commit_goal_bt = jnp.concatenate([eff_goal_xy - vis_obj_xy, eff_goal_xy], axis=-1)
    reg_gate_ctx = 0.08 * l2_normalize(jax.lax.stop_gradient(reg_sum))
    innov_gate_in = jax.lax.stop_gradient(innov_feats) * jnp.array(
        [1.8, 1.0, 1.8, 1.8, 4.5, 0.5], dtype=jnp.float32
    )
    commit_in = jnp.concatenate([reg_gate_ctx, 0.15 * jax.lax.stop_gradient(commit_goal_bt), innov_gate_in], axis=-1)
    commit_logits = (
        jax.nn.silu(commit_in @ params["commit_w1"].astype(jnp.float32))
        @ params["commit_w2"].astype(jnp.float32)
        + params["commit_b2"].astype(jnp.float32)
    )
    text_logits = txt_ctx @ params["head_text_w"].astype(jnp.float32)

    z_diag_vis = jax.lax.stop_gradient(z_vis_masked) if stop_grad_diagnostics else z_vis_masked
    z_diag_aud = jax.lax.stop_gradient(z_aud_masked) if stop_grad_diagnostics else z_aud_masked
    vis_patches_dec = z_diag_vis @ params["diag_vis_patch_w"].astype(jnp.float32)
    vis_grid = vis_patches_dec.reshape(b, t, 4, 4, 16, 16, 3)
    diag_video = jnp.swapaxes(vis_grid, 3, 4).reshape(b, t, 64, 64, 3)
    diag_audio = jnp.mean(z_diag_aud, axis=2) @ params["diag_aud_w"].astype(jnp.float32)

    haz_lat = decode_spatial_hazard_grid_8x8(params, z_task_vis_masked, w6_iv, pure_v_ti=None, pure_v_8x8=None)
    if pure_v8 is not None:
        haz_pv = decode_spatial_hazard_grid_8x8(
            params, z_task_vis_masked, w6_iv, pure_v_ti=pure_vp_ti, pure_v_8x8=pure_v8
        )
        hazard_logits_8x8 = jnp.where(vis_obs_m[:, :, None, None] > 0.5, haz_pv, haz_lat)
    else:
        hazard_logits_8x8 = haz_lat

    blk_lat = decode_target_block_xy_wave8a(params, z_task_vis_masked, interventions, pure_v_8x8=None)
    if pure_v8 is not None:
        blk_pv = decode_target_block_xy_wave8a(params, z_task_vis_masked, interventions, pure_v_8x8=pure_v8)
        pred_target_block_xy = jnp.where(vis_obs_m[:, :, None] > 0.5, blk_pv, blk_lat)
    else:
        pred_target_block_xy = blk_lat

    return {
        "pred_goal_xy": pred_goal_xy,
        "pred_target_block_xy": pred_target_block_xy,
        "vis_ee_curr": vis_ee_curr,
        "vis_obj_curr": vis_obj_curr,
        "lat_ee_curr": lat_ee_curr,
        "lat_obj_curr": lat_obj_curr,
        "vis_ee_next": vis_ee_next,
        "vis_obj_next": vis_obj_next,
        "vis_proprio_curr": vis_proprio_curr,
        "proprio_curr": proprio_curr,
        "proprio_next": proprio_next,
        "proprio_trusted": proprio_trusted,
        "proprio_next_trusted": proprio_next_trusted,
        "contact_logits": contact_logits,
        "hazard_logits_8x8": hazard_logits_8x8,
        "action_proposal": act_prop,
        "action_proposal_trusted": act_prop_trusted,
        "value_pred": value_pred,
        "commit_logits": commit_logits,
        "innov_feats": innov_feats,
        "text_logits": text_logits,
        "diag_video": diag_video,
        "diag_audio": diag_audio,
        "next_has_prev_tick": next_has_prev_tick,
        "next_modality_seen": next_modality_seen,
        "next_z_next_pred_vis_masked": next_z_pred_carry,
        "next_log_var_next": next_lv_carry,
    }


def convert_legacy_proprio_to_ee_only(p: jax.Array) -> jax.Array:
    """Convert legacy `state_to_proprio` (`[ee_xy, obj_xy, obj_vel, sin, cos, prev_u, prev_con, 0..0]`) to strict `ee_only`.

    Computes `ee_vel` strictly causally from `ee_xy` (`0` at `t=0`, `ee_xy[t] - ee_xy[t-1]` for `t >= 1`) and reads
    causal previous-tick impact contact from slot `10`, completely ignoring `2:10` (`obj_xy`, `obj_vel`, `obj_theta`).
    """
    pf = p.astype(jnp.float32)
    ee_xy = pf[..., 0:2]
    if pf.ndim >= 3 and pf.shape[-2] > 1:
        ee_vel = jnp.concatenate([jnp.zeros_like(ee_xy[..., :1, :]), ee_xy[..., 1:, :] - ee_xy[..., :-1, :]], axis=-2)
    else:
        ee_vel = jnp.zeros_like(ee_xy)
    contact = pf[..., 10:11]
    zeros1 = jnp.zeros_like(contact)
    zeros8 = jnp.zeros((*ee_xy.shape[:-1], 8), dtype=jnp.float32)
    return jnp.concatenate([ee_xy, ee_xy, ee_vel, zeros1, contact, zeros8], axis=-1)


def convert_legacy_target_proprio(p: jax.Array, prev_ee_xy: jax.Array | None = None) -> jax.Array:
    """Harmonize legacy `state_to_proprio` supervision target to `[ee_xy, obj_xy, ee_vel, obj_vel, 0..0]`."""
    pf = p.astype(jnp.float32)
    ee_xy = pf[..., 0:2]
    obj_xy = pf[..., 2:4]
    obj_vel = pf[..., 4:6]
    if prev_ee_xy is not None:
        ee_vel = ee_xy - prev_ee_xy.astype(jnp.float32)
    elif pf.ndim >= 3 and pf.shape[-2] > 1:
        ee_vel = jnp.concatenate([jnp.zeros_like(ee_xy[..., :1, :]), ee_xy[..., 1:, :] - ee_xy[..., :-1, :]], axis=-2)
    else:
        ee_vel = jnp.zeros_like(ee_xy)
    zeros8 = jnp.zeros((*ee_xy.shape[:-1], 8), dtype=jnp.float32)
    return jnp.concatenate([ee_xy, obj_xy, ee_vel, obj_vel, zeros8], axis=-1)


def de_privilege_proprio_wave8a(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Strip all privileged object pose/velocity/orientation from `batch['proprio']` across both schemas."""
    pf = batch["proprio"].astype(jnp.float32)
    leg_depriv = convert_legacy_proprio_to_ee_only(pf)
    ee_xy = pf[..., 0:2]
    ee_vel = pf[..., 4:6]
    contact = pf[..., 7:8]
    ee_depriv = jnp.concatenate(
        [ee_xy, ee_xy, ee_vel, jnp.zeros_like(contact), contact, jnp.zeros((*ee_xy.shape[:-1], 8), dtype=jnp.float32)],
        axis=-1,
    )
    out = dict(batch)
    if "ee_only_proprio" in batch:
        flag = batch["ee_only_proprio"].astype(bool)
        while flag.ndim < pf.ndim:
            flag = flag[..., None]
        out["proprio"] = jnp.where(flag, ee_depriv, leg_depriv)
        if "target_proprio" in batch:
            tp = batch["target_proprio"].astype(jnp.float32)
            out["target_proprio"] = jnp.where(flag, tp, convert_legacy_target_proprio(tp))
        if "target_next_proprio" in batch:
            tnp = batch["target_next_proprio"].astype(jnp.float32)
            out["target_next_proprio"] = jnp.where(
                flag, tnp, convert_legacy_target_proprio(tnp, prev_ee_xy=pf[..., 0:2])
            )
    else:
        out["proprio"] = leg_depriv
        if "target_proprio" in batch:
            out["target_proprio"] = convert_legacy_target_proprio(batch["target_proprio"])
        if "target_next_proprio" in batch:
            out["target_next_proprio"] = convert_legacy_target_proprio(
                batch["target_next_proprio"], prev_ee_xy=pf[..., 0:2]
            )
    if "modality_mask" in batch:
        prop_m = batch["modality_mask"].astype(jnp.float32)[..., 2:3]
        out["proprio"] = out["proprio"] * prop_m
    return out


def compute_disjoint_subspace_text_losses(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_mem: jax.Array,
    z_task: jax.Array,
    heads: dict[str, jax.Array],
    txt_target: jax.Array,
    txt_mask: jax.Array,
    scen_arr: jax.Array,
    interventions: Wave8AInterventions,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute `(assoc_ce, goal_sym_ce, lm_ce)` with strict subspace-restricted softmax when `zmem_isolation=True`."""
    is_assoc_tok = ((txt_target >= 64) & (txt_target < 96)).astype(jnp.float32)
    is_goal_sym_tok = ((txt_target >= 3584) | ((txt_target >= 4) & (txt_target < 16))).astype(jnp.float32)
    is_real_lm_tok = ((scen_arr < 0)[:, None] & (txt_target >= 96) & (txt_target < 3584)).astype(jnp.float32)

    if not interventions.zmem_isolation:
        txt_lp = jax.nn.log_softmax(heads["text_logits"], axis=-1)
        txt_nll = -jnp.take_along_axis(txt_lp, jnp.clip(txt_target, 0, spec.vocab_size - 1)[..., None], axis=-1)[..., 0]
        assoc_ce = jnp.sum(txt_nll * txt_mask * is_assoc_tok) / jnp.maximum(jnp.sum(txt_mask * is_assoc_tok), 1.0)
        goal_sym_ce = jnp.sum(txt_nll * txt_mask * is_goal_sym_tok) / jnp.maximum(jnp.sum(txt_mask * is_goal_sym_tok), 1.0)
        lm_ce = jnp.sum(txt_nll * txt_mask * is_real_lm_tok) / jnp.maximum(jnp.sum(txt_mask * is_real_lm_tok), 1.0)
        return assoc_ce, goal_sym_ce, lm_ce

    # 1. Associative binding recall [64:96]: strictly from z_mem[:, :, SLOT_TEXT, :] & head_text_w[:, 64:96]
    z_mem_txt = (
        z_mem[:, :, SLOT_TEXT, :].astype(jnp.float32)
        if interventions.decoupled_heads
        else jnp.mean(z_mem.astype(jnp.float32), axis=2)
    )
    w_assoc = params["head_text_w"][:, 64:96].astype(jnp.float32)
    assoc_lp = jax.nn.log_softmax(z_mem_txt @ w_assoc, axis=-1)
    assoc_idx = jnp.clip(txt_target - 64, 0, 31)
    assoc_nll = -jnp.take_along_axis(assoc_lp, assoc_idx[..., None], axis=-1)[..., 0]
    assoc_ce = jnp.sum(assoc_nll * txt_mask * is_assoc_tok) / jnp.maximum(jnp.sum(txt_mask * is_assoc_tok), 1.0)

    # 2. Goal-bin & LM tokens: strictly from z_task registers/audio with zero graph path to z_mem or head_text_w[:, 64:96]
    zf = z_task.astype(jnp.float32)
    task_ctx = 0.85 * jnp.mean(zf[:, :, SLOTS_REGISTER, :], axis=2) + 0.15 * jnp.mean(zf[:, :, SLOTS_AUDIO, :], axis=2)
    w_non_assoc = params["head_text_w"].astype(jnp.float32).at[:, 64:96].set(0.0)
    raw_task_logits = task_ctx @ w_non_assoc
    v_idx = jnp.arange(spec.vocab_size, dtype=jnp.int32)
    goal_col_mask = ((v_idx >= 4) & (v_idx < 16)) | (v_idx >= 3584)
    lm_col_mask = (v_idx >= 96) & (v_idx < 3584)

    cl_targ = jnp.clip(txt_target, 0, spec.vocab_size - 1)[..., None]
    goal_lp = jax.nn.log_softmax(jnp.where(goal_col_mask[None, None, :], raw_task_logits, -1e9), axis=-1)
    lm_lp = jax.nn.log_softmax(jnp.where(lm_col_mask[None, None, :], raw_task_logits, -1e9), axis=-1)
    goal_nll = -jnp.take_along_axis(goal_lp, cl_targ, axis=-1)[..., 0]
    lm_nll = -jnp.take_along_axis(lm_lp, cl_targ, axis=-1)[..., 0]

    goal_sym_ce = jnp.sum(goal_nll * txt_mask * is_goal_sym_tok) / jnp.maximum(jnp.sum(txt_mask * is_goal_sym_tok), 1.0)
    lm_ce = jnp.sum(lm_nll * txt_mask * is_real_lm_tok) / jnp.maximum(jnp.sum(txt_mask * is_real_lm_tok), 1.0)
    return assoc_ce, goal_sym_ce, lm_ce


def world_model_loss_wave8a(
    params: dict[str, jax.Array],
    ema_params: dict[str, jax.Array],
    spec: X1BSpec,
    raw_batch: dict[str, jax.Array],
    interventions: Wave8AInterventions,
    rng_key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Wave 8A joint recall + spatial control loss with `Z_mem ⊕ Z_task` isolation and subspace-restricted CE."""
    batch = de_privilege_proprio_wave8a(raw_batch)
    b, t = batch["video"].shape[:2]
    mm_obs = batch.get("modality_mask", jnp.ones((b, t, 4), dtype=jnp.float32)).astype(jnp.float32)
    mm_full = batch.get("full_modality_mask", mm_obs).astype(jnp.float32)
    aud_present = mm_obs[:, :, 0]
    vis_present = mm_obs[:, :, 1]
    prop_present = mm_obs[:, :, 2]
    txt_present = mm_obs[:, :, 3]

    # Enforce strict zero-input sanitization on all unavailable modalities before any forward or auxiliary head path.
    batch = {
        **batch,
        "audio": jnp.where(aud_present[:, :, None] > 0.5, batch["audio"], 0.0),
        "video": jnp.where(vis_present[:, :, None, None, None] > 0.5, batch["video"], jnp.zeros_like(batch["video"])),
        "proprio": jnp.where(prop_present[:, :, None] > 0.5, batch["proprio"], 0.0),
        "text": jnp.where(txt_present[:, :, None] > 0.5, batch["text"], 0),
        "modality_mask": mm_obs,
    }
    z_raw, _, _ = forward_world_model_wave8a(params, spec, batch, interventions)
    z_mem, z_task = fork_isolated_latent_streams(z_raw, interventions.zmem_isolation)

    valid = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(jnp.float32)
    valid_pair = valid[:, :-1] * valid[:, 1:]
    w6_iv = interventions.as_wave6()
    k_v, k_a, k_flip, k_cmd = jax.random.split(rng_key, 4)

    u_exec = batch.get("next_action", batch["action"])
    z_next_1, log_var_1 = predict_latent_step(params, spec, z_task, u_exec, horizon_idx=0, residual=True)
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

    raw_clean_vid = raw_batch.get("clean_video", raw_batch["video"])
    raw_clean_prop = de_privilege_proprio_wave8a(raw_batch)["proprio"]
    clean_batch = {
        **batch,
        "audio": jnp.where(mm_full[:, :, 0:1] > 0.5, raw_batch["audio"], 0.0),
        "video": jnp.where(mm_full[:, :, 1:2, None, None] > 0.5, raw_clean_vid, jnp.zeros_like(raw_clean_vid)),
        "proprio": jnp.where(mm_full[:, :, 2:3] > 0.5, raw_clean_prop, 0.0),
        "text": jnp.where(mm_full[:, :, 3:4] > 0.5, raw_batch["text"], 0),
        "modality_mask": mm_full,
    }
    z_target, _, _ = forward_world_model_wave8a(jax.lax.stop_gradient(ema_params), spec, clean_batch, interventions)
    z_target = jax.lax.stop_gradient(z_target)

    # Exclude SLOT_TEXT from action-conditioned JEPA & uncertainty prediction error when zmem_isolation=True.
    if interventions.zmem_isolation:
        emb_idx = jnp.array([i for i in range(TOTAL_SLOTS) if i != SLOT_TEXT], dtype=jnp.int32)
        zn1_j = z_next_1[:, :-1, emb_idx, :]
        zt1_j = z_target[:, 1:, emb_idx, :]
        raw_sq = jax.lax.stop_gradient(
            jnp.mean((zn1_j.astype(jnp.float32) - zt1_j.astype(jnp.float32)) ** 2, axis=-1, keepdims=True)
        )
        lv = log_var_1[:, :-1, emb_idx, :]
    else:
        zn1_j = z_next_1[:, :-1]
        zt1_j = z_target[:, 1:]
        raw_sq = jax.lax.stop_gradient(
            jnp.mean((z_next_1[:, :-1].astype(jnp.float32) - z_target[:, 1:].astype(jnp.float32)) ** 2, axis=-1, keepdims=True)
        )
        lv = log_var_1[:, :-1]

    commit_target = batch["target_commit"].astype(jnp.int32)
    clean_tick_mask = (commit_target == 0).astype(jnp.float32) * valid
    clean_pair = clean_tick_mask[:, :-1] * clean_tick_mask[:, 1:]

    err_1 = jnp.mean((l2_normalize(zn1_j) - l2_normalize(zt1_j)) ** 2, axis=(-2, -1))
    jepa_1 = jnp.sum(err_1 * clean_pair) / jnp.maximum(jnp.sum(clean_pair), 1.0)
    if z_next_2 is not None:
        v2 = clean_tick_mask[:, :-2] * clean_tick_mask[:, 2:]
        if interventions.zmem_isolation:
            err_2 = jnp.mean(
                (l2_normalize(z_next_2[:, :-1, emb_idx, :]) - l2_normalize(z_target[:, 2:, emb_idx, :])) ** 2,
                axis=(-2, -1),
            )
        else:
            err_2 = jnp.mean((l2_normalize(z_next_2[:, :-1]) - l2_normalize(z_target[:, 2:])) ** 2, axis=(-2, -1))
        jepa_2 = jnp.sum(err_2 * v2) / jnp.maximum(jnp.sum(v2), 1.0)
    else:
        jepa_2 = jnp.zeros((), dtype=jnp.float32)
    if z_next_4 is not None:
        v4 = clean_tick_mask[:, :-4] * clean_tick_mask[:, 4:]
        if interventions.zmem_isolation:
            err_4 = jnp.mean(
                (l2_normalize(z_next_4[:, :-1, emb_idx, :]) - l2_normalize(z_target[:, 4:, emb_idx, :])) ** 2,
                axis=(-2, -1),
            )
        else:
            err_4 = jnp.mean((l2_normalize(z_next_4[:, :-1]) - l2_normalize(z_target[:, 4:])) ** 2, axis=(-2, -1))
        jepa_4 = jnp.sum(err_4 * v4) / jnp.maximum(jnp.sum(v4), 1.0)
    else:
        jepa_4 = jnp.zeros((), dtype=jnp.float32)

    unc_loss = jnp.sum(jnp.mean(0.5 * (raw_sq * jnp.exp(-lv) + lv), axis=(-2, -1)) * clean_pair) / jnp.maximum(
        jnp.sum(clean_pair), 1.0
    )
    jepa_loss = jepa_1 + 0.5 * jepa_2 + 0.35 * jepa_4

    vis_pool = jnp.mean(z_task[:, :, SLOTS_VISUAL, :], axis=2).astype(jnp.float32) @ params["align_vis_w"].astype(jnp.float32)
    aud_pool = jnp.mean(z_task[:, :, SLOTS_AUDIO, :], axis=2).astype(jnp.float32) @ params["align_aud_w"].astype(jnp.float32)
    vis_valid = clean_tick_mask * vis_present
    aud_valid = valid * aud_present
    vis_sig_mask = vis_valid.reshape(b * t)
    aud_sig_mask = aud_valid.reshape(b * t)
    vis_clean_in = jnp.where(vis_valid[:, :, None] > 0.5, vis_pool, 0.0)
    aud_clean_in = jnp.where(aud_valid[:, :, None] > 0.5, aud_pool, 0.0)
    if interventions.regularizer == "epps_pulley_sigreg":
        sig_v, pr_val = epps_pulley_sigreg_loss(vis_clean_in.reshape(b * t, -1), k_v, num_slices=64, mask=vis_sig_mask)
        sig_a, _ = epps_pulley_sigreg_loss(aud_clean_in.reshape(b * t, -1), k_a, num_slices=64, mask=aud_sig_mask)
    else:
        sig_v, pr_val = vicreg_var_cov_loss(vis_clean_in.reshape(b * t, -1), mask=vis_sig_mask)
        sig_a, _ = vicreg_var_cov_loss(aud_clean_in.reshape(b * t, -1), mask=aud_sig_mask)
    sig_loss = 0.5 * (sig_v + sig_a)

    v_denom = jnp.maximum(jnp.sum(vis_valid, axis=1, keepdims=True), 1.0)
    a_denom = jnp.maximum(jnp.sum(aud_valid, axis=1, keepdims=True), 1.0)
    v_clip = l2_normalize(jnp.sum(vis_pool * vis_valid[:, :, None], axis=1) / v_denom)
    a_clip = l2_normalize(jnp.sum(aud_pool * aud_valid[:, :, None], axis=1) / a_denom)
    temp = jnp.clip(jnp.exp(params["align_logit_scale"].astype(jnp.float32))[0], 1.0, 50.0)
    sim = temp * (v_clip @ a_clip.T)
    has_av_raw = batch.get("has_av_pair", jnp.ones((b,), dtype=jnp.float32)).astype(jnp.float32)
    has_av = (
        has_av_raw
        * (jnp.max(vis_valid, axis=1) > 0.5).astype(jnp.float32)
        * (jnp.max(aud_valid, axis=1) > 0.5).astype(jnp.float32)
    )
    # Mask BOTH anchor rows and candidate columns so unpaired (has_av == 0) items never enter softmax denominators.
    pair_mat = (has_av[:, None] > 0.5) & (has_av[None, :] > 0.5)
    eye_b = jnp.eye(b, dtype=jnp.bool_)
    nce_mask = jnp.where(has_av[:, None] > 0.5, pair_mat, eye_b)
    lp_va = jax.nn.log_softmax(jnp.where(nce_mask, sim, -1.0e9), axis=-1)
    lp_av = jax.nn.log_softmax(jnp.where(nce_mask, sim.T, -1.0e9), axis=-1)
    av_nce_rows = -0.5 * (jnp.diag(lp_va) + jnp.diag(lp_av))
    n_av_pairs = jnp.sum(has_av)
    av_scale = jnp.where(n_av_pairs > 1.5, 1.0, 0.0)
    av_nce = jnp.sum(av_nce_rows * has_av) / jnp.maximum(n_av_pairs, 1.0)

    vp_mask = clean_tick_mask * vis_present * prop_present
    clean_prop_emb = batch["proprio"].astype(jnp.float32) @ params["prop_w"].astype(jnp.float32)
    vp_cos = jnp.sum(
        l2_normalize(vis_pool) * l2_normalize(clean_prop_emb @ params["align_prop_w"].astype(jnp.float32)),
        axis=-1,
    )
    vp_align = jnp.sum((1.0 - vp_cos) * vp_mask) / jnp.maximum(jnp.sum(vp_mask), 1.0)
    align_loss = av_nce * av_scale + 0.35 * vp_align

    heads = decode_world_heads_wave8a(
        params,
        spec,
        z_mem,
        z_task,
        z_next_1,
        log_var_1,
        batch["goal"],
        interventions,
        obs_ee_proprio=batch["proprio"],
        obs_video=batch["video"],
        obs_modality_mask=mm_obs,
        stop_grad_diagnostics=True,
    )

    target_prop = batch["target_proprio"].astype(jnp.float32)
    clean_vis_mask = clean_tick_mask * vis_present

    scen_arr = batch.get("scenario", jnp.zeros((b,), dtype=jnp.int32))
    is_arena_sim = ((scen_arr >= 0) & (scen_arr != 4)).astype(jnp.float32)[:, None]
    arena_vis_mask = clean_vis_mask * is_arena_sim
    denom_av = jnp.maximum(jnp.sum(arena_vis_mask), 1.0)
    vis_ee_mse = (
        jnp.sum(jnp.mean((heads["vis_ee_curr"] - target_prop[:, :, 0:2]) ** 2, axis=-1) * arena_vis_mask) / denom_av
        + 0.5 * jnp.sum(jnp.mean((heads["lat_ee_curr"] - target_prop[:, :, 0:2]) ** 2, axis=-1) * arena_vis_mask) / denom_av
    )
    vis_obj_mse = (
        jnp.sum(jnp.mean((heads["vis_obj_curr"] - target_prop[:, :, 2:4]) ** 2, axis=-1) * arena_vis_mask) / denom_av
        + 0.5 * jnp.sum(jnp.mean((heads["lat_obj_curr"] - target_prop[:, :, 2:4]) ** 2, axis=-1) * arena_vis_mask) / denom_av
    )

    # Columns 8:16 of head_proprio_w are reserved for zero-extra-param tabletop block localization (decode_target_block_xy_wave8a).
    dim_w = jnp.array([2.0] * 8 + [0.0] * 8, dtype=jnp.float32)[None, None, :]
    if "target_next_proprio" in batch:
        tnp = batch["target_next_proprio"].astype(jnp.float32)
        has_explicit_final_next = (scen_arr >= 0).astype(jnp.float32)[:, None]
        valid_next = jnp.concatenate([valid[:, 1:], valid[:, -1:] * has_explicit_final_next], axis=1)
        next_mask = valid * valid_next * prop_present * (commit_target != 2).astype(jnp.float32)
        clean_next_m = next_mask * (commit_target == 0).astype(jnp.float32)
        prop_next_loss = jnp.sum(
            jnp.mean(((heads["proprio_next"] - tnp) ** 2) * dim_w, axis=-1) * clean_next_m
        ) / jnp.maximum(jnp.sum(clean_next_m), 1.0)
        vis_obj_next_mse = jnp.sum(
            jnp.mean((heads["vis_obj_next"] - tnp[:, :, 2:4]) ** 2, axis=-1) * clean_next_m * is_arena_sim
        ) / jnp.maximum(jnp.sum(clean_next_m * is_arena_sim), 1.0)
        prop_trusted_loss = jnp.sum(
            jnp.mean(((heads["proprio_next_trusted"] - tnp) ** 2) * dim_w, axis=-1) * next_mask
        ) / jnp.maximum(jnp.sum(next_mask), 1.0)
    else:
        vp_pair = valid_pair * prop_present[:, :-1] * prop_present[:, 1:]
        prop_next_loss = jnp.sum(
            jnp.mean(((heads["proprio_next"][:, :-1] - target_prop[:, 1:]) ** 2) * dim_w, axis=-1) * vp_pair
        ) / jnp.maximum(jnp.sum(vp_pair), 1.0)
        vis_obj_next_mse = jnp.zeros((), dtype=jnp.float32)
        prop_trusted_loss = prop_next_loss

    if z_next_4 is not None:
        zr4 = z_next_4[:, :-1].astype(jnp.float32)
        z04 = z_task[:, :-4].astype(jnp.float32)
        hw4 = params["head_proprio_w"].astype(jnp.float32)
        hb4 = params["head_proprio_b"].astype(jnp.float32)
        _, vis_obj_4, _ = decode_spatial_vis_pose(params, zr4, w6_iv)
        if interventions.zerolag_bypass:
            ee_a = batch["proprio"][:, :-4, 0:2].astype(jnp.float32)
            obj_a = jax.lax.stop_gradient(heads["vis_obj_curr"][:, :-4])
            anc4 = batch["proprio"][:, :-4].astype(jnp.float32).at[:, :, 0:2].set(ee_a).at[:, :, 2:4].set(obj_a)
            d4 = (
                0.75 * (zr4[:, :, SLOT_PROPRIO, :] - z04[:, :, SLOT_PROPRIO, :])
                + 0.25 * (jnp.mean(zr4[:, :, SLOTS_VISUAL, :], axis=2) - jnp.mean(z04[:, :, SLOTS_VISUAL, :], axis=2))
            ) @ hw4
            raw_s4 = anc4 + d4
            pred_s4 = raw_s4.at[:, :, 2:4].set(0.60 * raw_s4[:, :, 2:4] + 0.40 * vis_obj_4)
        else:
            pred_s4 = (0.75 * zr4[:, :, SLOT_PROPRIO, :] + 0.25 * jnp.mean(zr4[:, :, SLOTS_VISUAL, :], axis=2)) @ hw4 + hb4
        m4 = (
            valid[:, :-4]
            * valid[:, 4:]
            * prop_present[:, :-4]
            * prop_present[:, 4:]
            * (commit_target[:, :-4] == 0).astype(jnp.float32)
            * (commit_target[:, 4:] == 0).astype(jnp.float32)
        )
        prop_step4_loss = jnp.sum(jnp.mean(((pred_s4 - target_prop[:, 4:]) ** 2) * dim_w, axis=-1) * m4) / jnp.maximum(
            jnp.sum(m4), 1.0
        )
    else:
        prop_step4_loss = jnp.zeros((), dtype=jnp.float32)

    spatial_pose_loss = 26.0 * (1.5 * vis_ee_mse + 1.5 * vis_obj_mse + 0.8 * vis_obj_next_mse)
    state_loss = spatial_pose_loss + 32.0 * prop_next_loss + 8.0 * prop_trusted_loss + 10.0 * prop_step4_loss

    raw_tc = batch["target_contacts"].astype(jnp.float32)
    is_legacy_synth = ((scen_arr >= 0) & (scen_arr != 5) & (scen_arr != 4)).astype(jnp.float32)[:, None]
    near_central_bar = (
        ((jnp.abs(target_prop[:, :, 2]) < 0.16) & (jnp.abs(target_prop[:, :, 3]) < 0.39))
        | ((jnp.abs(target_prop[:, :, 0]) < 0.13) & (jnp.abs(target_prop[:, :, 1]) < 0.39))
    ).astype(jnp.float32)
    loc_haz = jnp.where(is_legacy_synth > 0.5, raw_tc[:, :, 1] * near_central_bar, raw_tc[:, :, 1])
    target_contacts = jnp.stack([raw_tc[:, :, 0], loc_haz], axis=-1)
    cl = heads["contact_logits"]
    c_elem = jnp.mean(
        jax.nn.softplus(cl) * (1.0 - target_contacts) + 3.5 * jax.nn.softplus(-cl) * target_contacts,
        axis=-1,
    )
    contact_mask = valid * prop_present
    base_contact_bce = jnp.sum(c_elem * contact_mask) / jnp.maximum(jnp.sum(contact_mask), 1.0)

    c8 = jnp.linspace(-0.875, 0.875, 8, dtype=jnp.float32)
    yy8, xx8 = jnp.meshgrid(c8, c8, indexing="ij")
    if "in_domain_bar_hx" in raw_batch:
        bhx = raw_batch["in_domain_bar_hx"].astype(jnp.float32)
        bhy = raw_batch["in_domain_bar_hy"].astype(jnp.float32)
    else:
        has_leg_bar = (is_legacy_synth[:, 0] > 0.5) & (raw_tc[:, 0, 1] > 0.5)
        bhx = jnp.where(has_leg_bar, 0.025, 0.0)
        bhy = jnp.where(has_leg_bar, 0.38, 0.0)
    targ_haz_8x8 = (
        (bhx[:, None, None] > 1e-4)
        & (jnp.abs(xx8[None, :, :]) <= (bhx[:, None, None] + 0.115))
        & (jnp.abs(yy8[None, :, :]) <= (bhy[:, None, None] + 0.095))
    ).astype(jnp.float32)[:, None, :, :]
    hl8 = heads["hazard_logits_8x8"]
    ee_x = target_prop[:, :, 0][:, :, None, None]
    ee_y = target_prop[:, :, 1][:, :, None, None]
    ob_x = target_prop[:, :, 2][:, :, None, None]
    ob_y = target_prop[:, :, 3][:, :, None, None]
    d2_ee8 = (xx8[None, None, :, :] - ee_x) ** 2 + (yy8[None, None, :, :] - ee_y) ** 2
    d2_ob8 = (xx8[None, None, :, :] - ob_x) ** 2 + (yy8[None, None, :, :] - ob_y) ** 2
    near_ee_obj = ((d2_ee8 < 0.24**2) | (d2_ob8 < 0.24**2)).astype(jnp.float32) * (1.0 - targ_haz_8x8)
    neg_cell_w = (1.0 + 5.5 * near_ee_obj) * (1.0 - targ_haz_8x8)
    cell_bce = jax.nn.softplus(hl8) * neg_cell_w + 8.5 * jax.nn.softplus(-hl8) * targ_haz_8x8
    spatial_hazard_bce = jnp.sum(jnp.mean(cell_bce, axis=(-2, -1)) * arena_vis_mask) / denom_av
    contact_bce = base_contact_bce + 2.5 * spatial_hazard_bce

    act_target = batch["target_action"].astype(jnp.float32)
    base_act_mask = (
        batch.get("action_supervision_mask", jnp.ones((b, t), dtype=jnp.float32)).astype(jnp.float32)
        * valid
        * prop_present
    )
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
    commit_sup_mask = ((scen_arr >= 0)[:, None]).astype(jnp.float32) * valid
    commit_cls_w = jnp.array([1.0, 4.0, 4.5, 1.8], dtype=jnp.float32)[commit_target] * commit_sup_mask
    commit_nll = -jnp.take_along_axis(commit_lp, commit_target[..., None], axis=-1)[..., 0]
    commit_loss = jnp.sum(commit_nll * commit_cls_w) / jnp.maximum(jnp.sum(commit_cls_w), 1.0)

    txt_target = batch["target_text"].astype(jnp.int32)
    txt_mask = (txt_target > 0).astype(jnp.float32) * valid * txt_present
    assoc_ce, goal_sym_ce, lm_ce = compute_disjoint_subspace_text_losses(
        params, spec, z_mem, z_task, heads, txt_target, txt_mask, scen_arr, interventions
    )
    assoc_w = 5.0 if interventions.enable_assoc_loss else 0.0
    text_loss = assoc_w * assoc_ce + 0.35 * goal_sym_ce + 0.5 * lm_ce

    if "target_goal_xy" in batch:
        tg_xy = batch["target_goal_xy"].astype(jnp.float32)
    else:
        gf_raw = batch["goal"].astype(jnp.float32)
        if gf_raw.ndim == 2:
            gf_raw = gf_raw[:, None, :]
        tg_xy = jnp.broadcast_to(gf_raw[:, :, 2:4], (b, t, 2))

    goal_w = jnp.where(
        (scen_arr == 5)[:, None],
        6.0,
        jnp.where((scen_arr == 1)[:, None], 3.5, jnp.where((scen_arr == -2)[:, None], 3.0, 1.0)),
    )
    has_goal_sup = (((scen_arr >= 0) & (scen_arr != 4)) | (scen_arr == -2)).astype(jnp.float32)[:, None]
    aud_only_rob_pre_dir = ((scen_arr == -2)[:, None]) & (txt_present < 0.5) & (jnp.arange(t)[None, :] < 28)
    goal_mask = goal_w * has_goal_sup * valid * (~aud_only_rob_pre_dir).astype(jnp.float32)
    goal_loss = jnp.sum(jnp.mean((heads["pred_goal_xy"] - tg_xy) ** 2, axis=-1) * goal_mask) / jnp.maximum(
        jnp.sum(goal_mask), 1.0
    )

    wrong_col_aud_src: jax.Array | None = None
    wrong_col_txt_src: jax.Array | None = None
    targ_blk_xy_t: jax.Array | None = None
    dist_blk_xy_t: jax.Array | None = None

    if "rob_audio_32" in batch:
        b_r = int(batch["rob_audio_32"].shape[0])
        t_init = int(batch["rob_audio_32"].shape[1])
        if "rob_modality_mask" in batch:
            init_mm_base = batch["rob_modality_mask"][:, :t_init].astype(jnp.float32)
        elif b_r == b and t_init <= t:
            init_mm_base = mm_obs[:, :t_init]
        else:
            init_mm_base = jnp.ones((b_r, t_init, 4), dtype=jnp.float32)
        rob_aud_m = init_mm_base[:, :, 0:1] > 0.5
        rob_vis_m0 = init_mm_base[:, :1, 1:2, None, None] > 0.5
        rob_prop_m0 = init_mm_base[:, :1, 2:3] > 0.5
        rob_txt_m = init_mm_base[:, :, 3:4] > 0.5
        aud_src = jnp.where(rob_aud_m, batch["rob_audio_32"], 0.0)
        wrong_aud_src = jnp.where(rob_aud_m, batch["rob_wrong_audio_32"], 0.0)
        txt_src = jnp.where(rob_txt_m, batch["rob_text_32"], 0)
        wrong_txt_src = jnp.where(rob_txt_m, batch["rob_wrong_text_32"], 0)
        vid0_san = jnp.where(rob_vis_m0, batch["rob_vid0"], jnp.zeros_like(batch["rob_vid0"]))
        prop0_san = jnp.where(rob_prop_m0, batch["rob_prop0"], 0.0)
        vid_init = jnp.broadcast_to(vid0_san, (b_r, t_init, *vid0_san.shape[2:]))
        prop_init = jnp.broadcast_to(prop0_san, (b_r, t_init, prop0_san.shape[-1]))
        end_xy_t = batch["rob_end_xy"].astype(jnp.float32)
        is_rob_ep = (jnp.max(init_mm_base[:, :1, 1], axis=1) > 0.5).astype(jnp.float32)
        if (
            "rob_wrong_col_audio_32" in batch
            and "rob_wrong_col_text_32" in batch
            and "rob_target_block_xy" in batch
            and "rob_distractor_block_xy" in batch
        ):
            wrong_col_aud_src = jnp.where(rob_aud_m, batch["rob_wrong_col_audio_32"], 0.0)
            wrong_col_txt_src = jnp.where(rob_txt_m, batch["rob_wrong_col_text_32"], 0)
            targ_blk_xy_t = batch["rob_target_block_xy"].astype(jnp.float32)
            dist_blk_xy_t = batch["rob_distractor_block_xy"].astype(jnp.float32)
        run_init_disc = True
    elif "wrong_text" in batch and t >= 4:
        if "wrong_audio" not in batch:
            raise KeyError("Wave 8A grounded robot loss requires explicit 'wrong_audio' counterfactuals (no roll fallback).")
        b_r = b
        t_init = min(t, 32)
        init_mm_base = mm_obs[:, :t_init]
        aud_src = batch["audio"][:, :t_init]
        wrong_aud_src = jnp.where(aud_present[:, :t_init, None] > 0.5, batch["wrong_audio"][:, :t_init], 0.0)
        txt_src = batch["text"][:, :t_init]
        wrong_txt_src = jnp.where(txt_present[:, :t_init, None] > 0.5, batch["wrong_text"][:, :t_init], 0)
        vid_init = jnp.broadcast_to(batch["video"][:, :1], (b_r, t_init, *batch["video"].shape[2:]))
        prop_init = jnp.broadcast_to(batch["proprio"][:, :1], (b_r, t_init, batch["proprio"].shape[-1]))
        end_xy_t = jnp.sum(tg_xy * valid[:, :, None], axis=1) / jnp.maximum(jnp.sum(valid[:, :, None], axis=1), 1.0)
        is_rob_ep = (scen_arr == -2).astype(jnp.float32) * (jnp.max(vis_present[:, :1], axis=1) > 0.5).astype(jnp.float32)
        if (
            "wrong_color_audio" in batch
            and "wrong_color_text" in batch
            and "target_block_xy" in batch
            and "distractor_block_xy" in batch
        ):
            wrong_col_aud_src = jnp.where(aud_present[:, :t_init, None] > 0.5, batch["wrong_color_audio"][:, :t_init], 0.0)
            wrong_col_txt_src = jnp.where(txt_present[:, :t_init, None] > 0.5, batch["wrong_color_text"][:, :t_init], 0)
            targ_blk_xy_t = batch["target_block_xy"].astype(jnp.float32)
            dist_blk_xy_t = batch["distractor_block_xy"].astype(jnp.float32)
        run_init_disc = True
    else:
        init_mm_base = None
        run_init_disc = False

    if run_init_disc and init_mm_base is not None:
        b2 = 2 * b_r
        zero_act_init = jnp.zeros((b2, t_init, spec.action_dim), dtype=jnp.float32)
        zero_goal_init = jnp.zeros((b2, t_init, spec.goal_dim), dtype=jnp.float32)

        # Evaluate [audio_only (half 0), 70% joint / 30% text_only (half 1)] on EVERY step so H_goal + aud_w
        # receive dense spoken waveform counterfactual gradients alongside joint/symbolic grounding.
        cmd_u = jax.random.uniform(k_cmd, (b_r,))
        txt_only_1 = cmd_u < 0.30
        keep_aud = jnp.concatenate(
            [jnp.ones((b_r,), dtype=jnp.float32), (~txt_only_1).astype(jnp.float32)], axis=0
        )
        keep_txt = jnp.concatenate(
            [jnp.zeros((b_r,), dtype=jnp.int32), jnp.ones((b_r,), dtype=jnp.int32)], axis=0
        )
        init_mm_2 = jnp.concatenate([init_mm_base, init_mm_base], axis=0)
        mm_init = (
            init_mm_2
            .at[:, :, 0]
            .set(init_mm_2[:, :, 0] * keep_aud[:, None])
            .at[:, :, 3]
            .set(init_mm_2[:, :, 3] * keep_txt.astype(jnp.float32)[:, None])
        )
        val_init = jnp.ones((b2, t_init), dtype=bool)
        vid_2 = jnp.concatenate([vid_init, vid_init], axis=0)
        prop_2 = jnp.concatenate([prop_init, prop_init], axis=0)
        aud_2 = jnp.concatenate([aud_src, aud_src], axis=0)
        wrong_aud_2 = jnp.concatenate([wrong_aud_src, wrong_aud_src], axis=0)
        txt_2 = jnp.concatenate([txt_src, txt_src], axis=0)
        wrong_txt_2 = jnp.concatenate([wrong_txt_src, wrong_txt_src], axis=0)
        end_xy_2 = jnp.concatenate([end_xy_t, end_xy_t], axis=0)
        is_rob_2 = jnp.concatenate([is_rob_ep, is_rob_ep], axis=0)

        # 50% random 180-degree spatial flip on frame 0 so E[target_block_xy | color] == [0, 0] (eliminates static color priors).
        if targ_blk_xy_t is not None and dist_blk_xy_t is not None:
            flip_m = jax.random.bernoulli(k_flip, 0.5, (b2,))
            vid_init_use = jnp.where(flip_m[:, None, None, None, None], jnp.flip(vid_2, axis=(2, 3)), vid_2)
            flip_sign = jnp.where(flip_m[:, None], -1.0, 1.0)
            targ_blk_eff = jnp.concatenate([targ_blk_xy_t, targ_blk_xy_t], axis=0) * flip_sign
            dist_blk_eff = jnp.concatenate([dist_blk_xy_t, dist_blk_xy_t], axis=0) * flip_sign
        else:
            vid_init_use = vid_2
            targ_blk_eff = None
            dist_blk_eff = None

        b_init_tr = {
            "video": vid_init_use,
            "audio": aud_2 * keep_aud[:, None, None],
            "proprio": prop_2,
            "text": txt_2 * keep_txt[:, None, None],
            "action": zero_act_init,
            "goal": zero_goal_init,
            "modality_mask": mm_init,
            "valid": val_init,
        }
        b_init_wr = {
            **b_init_tr,
            "text": wrong_txt_2 * keep_txt[:, None, None],
            "audio": wrong_aud_2 * keep_aud[:, None, None],
        }
        z_itr_raw, _, _ = forward_world_model_wave8a(params, spec, b_init_tr, interventions)
        z_iwr_raw, _, _ = forward_world_model_wave8a(params, spec, b_init_wr, interventions)
        _, z_itr_task = fork_isolated_latent_streams(z_itr_raw, interventions.zmem_isolation)
        _, z_iwr_task = fork_isolated_latent_streams(z_iwr_raw, interventions.zmem_isolation)
        pg_tr = decode_goal_latch_wave8a(params, z_itr_task, interventions)[:, -1]
        pg_wr = decode_goal_latch_wave8a(params, z_iwr_task, interventions)[:, -1]

        has_lang_init = (
            (jnp.max(mm_init[:, :, 0], axis=1) > 0.5) | (jnp.max(mm_init[:, :, 3], axis=1) > 0.5)
        ).astype(jnp.float32)
        is_rob_2 = is_rob_2 * has_lang_init

        d_tr = jnp.mean((pg_tr - end_xy_2) ** 2, axis=-1)
        d_wr = jnp.mean((pg_wr - end_xy_2) ** 2, axis=-1)
        rob_denom = jnp.maximum(jnp.sum(is_rob_2), 1.0)
        init_goal_mse = jnp.sum(d_tr * is_rob_2) / rob_denom
        rank_margin = jnp.sum(jax.nn.relu(0.14 + d_tr - d_wr) * is_rob_2) / rob_denom
        goal_loss = goal_loss + 2.2 * init_goal_mse

        if (
            wrong_col_aud_src is not None
            and wrong_col_txt_src is not None
            and targ_blk_eff is not None
            and dist_blk_eff is not None
        ):
            wrong_col_aud_2 = jnp.concatenate([wrong_col_aud_src, wrong_col_aud_src], axis=0)
            wrong_col_txt_2 = jnp.concatenate([wrong_col_txt_src, wrong_col_txt_src], axis=0)
            b_init_wcol = {
                **b_init_tr,
                "text": wrong_col_txt_2 * keep_txt[:, None, None],
                "audio": wrong_col_aud_2 * keep_aud[:, None, None],
            }
            z_iwc_raw, _, _ = forward_world_model_wave8a(params, spec, b_init_wcol, interventions)
            _, z_iwc_task = fork_isolated_latent_streams(z_iwc_raw, interventions.zmem_isolation)
            vis_last_m = mm_init[:, -1:, 1]
            vid_last_eff = jnp.where(
                vis_last_m[:, :, None, None, None] > 0.5, vid_init_use[:, -1:], jnp.zeros_like(vid_init_use[:, -1:])
            )
            pv8_last = encode_pure_visual_grid_8x8(params, spec, vid_last_eff)
            pb_tr_pv = decode_target_block_xy_wave8a(
                params, z_itr_task[:, -1:], interventions, pure_v_8x8=pv8_last
            )[:, 0]
            pb_tr_lat = decode_target_block_xy_wave8a(
                params, z_itr_task[:, -1:], interventions, pure_v_8x8=None
            )[:, 0]
            pb_wc_pv = decode_target_block_xy_wave8a(
                params, z_iwc_task[:, -1:], interventions, pure_v_8x8=pv8_last
            )[:, 0]
            pb_wc_lat = decode_target_block_xy_wave8a(
                params, z_iwc_task[:, -1:], interventions, pure_v_8x8=None
            )[:, 0]
            pb_tr = jnp.where(vis_last_m[:, 0, None] > 0.5, pb_tr_pv, pb_tr_lat)
            pb_wc = jnp.where(vis_last_m[:, 0, None] > 0.5, pb_wc_pv, pb_wc_lat)
            blk_d_tr_targ = jnp.mean((pb_tr - targ_blk_eff) ** 2, axis=-1)
            blk_d_tr_dist = jnp.mean((pb_tr - dist_blk_eff) ** 2, axis=-1)
            blk_d_wc_dist = jnp.mean((pb_wc - dist_blk_eff) ** 2, axis=-1)
            blk_d_wc_targ = jnp.mean((pb_wc - targ_blk_eff) ** 2, axis=-1)
            blk_loc_mse = 0.5 * (jnp.sum((blk_d_tr_targ + blk_d_wc_dist) * is_rob_2) / rob_denom)
            blk_margin = 0.5 * (
                jnp.sum(
                    (jax.nn.relu(0.10 + blk_d_tr_targ - blk_d_tr_dist) + jax.nn.relu(0.10 + blk_d_wc_dist - blk_d_wc_targ))
                    * is_rob_2
                )
                / rob_denom
            )
            goal_loss = goal_loss + 2.8 * blk_loc_mse
            rank_margin = rank_margin + blk_margin
    else:
        rank_margin = jnp.zeros((), dtype=jnp.float32)

    clean_vid = clean_batch["video"].astype(jnp.float32) / 127.5 - 1.0
    rec_v = jnp.mean((heads["diag_video"] - clean_vid) ** 2, axis=(-3, -2, -1))
    rec_a = jnp.mean((heads["diag_audio"] - batch["audio"].astype(jnp.float32)) ** 2, axis=-1)
    recon_v_loss = jnp.sum(rec_v * vis_valid) / jnp.maximum(jnp.sum(vis_valid), 1.0)
    recon_a_loss = jnp.sum(rec_a * aud_valid) / jnp.maximum(jnp.sum(aud_valid), 1.0)
    recon_loss = recon_v_loss + 0.5 * recon_a_loss

    total_loss = (
        2.0 * jepa_loss
        + 0.25 * unc_loss
        + 0.28 * sig_loss
        + 0.25 * align_loss
        + 1.2 * state_loss
        + 0.8 * contact_bce
        + 1.8 * act_loss
        + 0.35 * val_loss
        + 0.7 * commit_loss
        + 1.2 * text_loss
        + 6.0 * goal_loss
        + 12.0 * rank_margin
        + 0.25 * recon_loss
    )
    metrics = {
        "loss": total_loss,
        "jepa": jepa_loss,
        "sigreg": sig_loss,
        "pr": pr_val,
        "vis_ee_mse": vis_ee_mse,
        "vis_obj_mse": vis_obj_mse,
        "state_mse": prop_next_loss,
        "step4_mse": prop_step4_loss,
        "goal_mse": goal_loss,
        "rank_margin": rank_margin,
        "contact_bce": contact_bce,
        "commit_ce": commit_loss,
        "text_ce": assoc_ce,
        "goal_sym_ce": goal_sym_ce,
        "lm_ce": lm_ce,
    }
    return total_loss, metrics
