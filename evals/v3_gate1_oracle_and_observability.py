"""SLWM-v3 Gate 1: Observability, Decoupled 2x2 Oracle Control Matrix, & Shortcut-Free Counterfactual Certifier.

Implements all Gate 1 pre-training verification requirements (AGENTS.md §12 + Gate 1 peer-review fixes):
  1. Decoupled Oracle-Perception x Oracle-Dynamics Closed-Loop Control Matrix (`16x16` unsealed occupancy grid):
     - Evaluates identical isotropic K=38 two-stage candidates `(wp1, wp2)` and matched objective terms
       across:
         * `oracle_perc_oracle_dyn_rollout`        (true state, true mass/damping, H=14 rollout)
         * `oracle_perc_nominal_dyn_rollout`       (true state, nominal mass=1.0/damp=0.72, H=14 rollout)
         * `oracle_perc_static_geodesic_no_rollout`(true state, fair 16x16 2-stage static geodesic, 0 physics steps)
         * `oracle_perc_no_rollout_h1`             (true state, H=1 greedy rollout)
         * `noisy_perc_oracle_dyn_rollout`         (noisy visual pose σ=0.04, true mass/damping, H=14 rollout)
         * `noisy_perc_nominal_dyn_rollout`        (noisy visual pose σ=0.04, nominal dynamics, H=14 rollout)
         * `noisy_perc_static_geodesic_no_rollout` (noisy visual pose σ=0.04, fair 16x16 2-stage static geodesic)
         * `exact_state_reactive`                  (true state straight push + hold)
         * `noisy_state_reactive`                  (noisy pose σ=0.04 straight push + hold)
     - Perception noise and dynamics mismatch are strictly unconfounded.
     - Both `oracle_dyn_rollout` and `static_geodesic_no_rollout` share the exact same `16x16` occupancy grid,
       exact same `wp1 + wp2` candidate evaluation, and identical `(-2.4*v - 0.8*d)` scoring weights.
     - Reports BOTH closest-approach (`min(dist[-16:]) < 0.24`) AND sustained attainment (`max(dist[-16:]) < 0.24`)
       for every arm and every family with zero omitted cells.
  2. Shortcut-Free Randomized Delayed-Correction & Concurrent-Fact Preservation Certifier:
     - Zero hardcoded slot indices, zero reserved token-ID ranges, zero opcodes, zero fixed event ticks.
     - Randomizes arrival order of F1 vs F2, randomizes `t_first`, `t_second`, `t_corr` (with `t_corr - t_first > W`
       so the overwritten fact has already exited the local window before overwrite), randomizes `t_impulse`,
       and adds random episode-level wall-clock offset.
     - Populates background `text_bytes` with random non-active vocabulary traffic (no `text != 0` trigger).
     - Encodes continuous audio/visual symbols via unit-norm codebook vectors (no `bf16` scalar ULP floor).
     - Emits paired clean, in-distribution distractor-binding, and moderate `3.5-sigma` sensory-burst streams
       alongside bit-exact counterfactual stream B (`max_abs_diff == 0.0` across all 7 post-cue tensors).
     - Parameterizes `total_ticks` dynamically to strictly exceed `L * (W - 1)` for any comparator `(L, W)`.
  3. Exact Paired Statistical Power & Crossed-Cluster Design-Effect Calculator.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_unseen_worlds import _box_sdf_push_out

V3_FAM_STATIC_WALL = 0
V3_FAM_DUAL_CHICANE = 1
V3_FAM_CONVEYOR_DRIFT = 2
V3_FAM_MOVING_BUMPER_GATE = 3


def init_v3_dynamics_control_worlds(rng_key: jax.Array, batch_size: int = 256) -> dict[str, jax.Array]:
    """Initialize 4 physically observable planar control families for Gate 1 decoupled oracle matrix.

    Families (balanced 25% each):
      0 `static_off_center_wall`: Static baseline verifying fair 16x16 static-geodesic parity (~100%).
      1 `dual_chicane`: Static staggered S-curve barriers blocking straight-line reactive control.
      2 `conveyor_drift_channel`: Heavy low-friction puck (mass=1.8, damping=0.95) crossing a central
        lateral drift channel (|x| < 0.42, steady cross-acceleration ±0.0085/tick, calm landing zone
        |x| >= 0.42) with narrow off-center walls — requires upstream lead & anticipatory rollout.
      3 `moving_bumper_gate`: Oscillating barrier at x=0 (y_gap(t) = 0.44 * sin(0.11 * t + phi))
        whose moving walls act as elastic bumpers kicking contacting pucks back (-0.12 * sign(x)),
        preventing blind wall-hugging and requiring forward timing rollout.
    """
    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)
    layout_id = jnp.arange(batch_size, dtype=jnp.int32) % 4
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (batch_size,)), 1.0, -1.0)
    gy_sign = jnp.where(jax.random.bernoulli(k2, 0.5, (batch_size,)), 1.0, -1.0)

    obj_xy = jnp.stack([-0.64 * side, 0.20 * gy_sign], axis=-1)
    ee_xy = jnp.clip(obj_xy + jnp.stack([-0.15 * side, jnp.zeros((batch_size,))], axis=-1), -0.88, 0.88)
    targ_a = jnp.stack([0.62 * side, -0.22 * gy_sign], axis=-1)

    shift_x = jnp.where(jax.random.bernoulli(k3, 0.5, (batch_size,)), 0.18, -0.18)
    gap_y = jax.random.uniform(k4, (batch_size,), minval=-0.24, maxval=0.24)
    phi0 = jax.random.uniform(k5, (batch_size,), minval=0.0, maxval=2.0 * np.pi)
    drift_sign = jnp.where(jax.random.bernoulli(jax.random.fold_in(k5, 1), 0.5, (batch_size,)), 1.0, -1.0)

    # Family 0: Static off-center wall
    f0_cx = jnp.stack([shift_x, shift_x, jnp.zeros((batch_size,)), jnp.zeros((batch_size,))], axis=-1)
    f0_cy0 = jnp.stack([gap_y + 0.62, gap_y - 0.62, jnp.zeros((batch_size,)), jnp.zeros((batch_size,))], axis=-1)
    f0_hx = jnp.stack([jnp.full((batch_size,), 0.03), jnp.full((batch_size,), 0.03), jnp.zeros((batch_size,)), jnp.zeros((batch_size,))], axis=-1)
    f0_hy = jnp.stack([jnp.full((batch_size,), 0.40), jnp.full((batch_size,), 0.40), jnp.zeros((batch_size,)), jnp.zeros((batch_size,))], axis=-1)
    f0_amp = jnp.zeros((batch_size, 4), dtype=jnp.float32)
    f0_omg = jnp.zeros((batch_size, 4), dtype=jnp.float32)
    f0_phi = jnp.zeros((batch_size, 4), dtype=jnp.float32)

    # Family 1: Static dual staggered chicane (blocks reactive straight-line push completely)
    f1_cx = jnp.broadcast_to(jnp.array([-0.22, 0.22, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f1_cy0 = jnp.broadcast_to(jnp.array([0.32, -0.32, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f1_hx = jnp.broadcast_to(jnp.array([0.025, 0.025, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f1_hy = jnp.broadcast_to(jnp.array([0.52, 0.52, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f1_amp, f1_omg, f1_phi = f0_amp, f0_omg, f0_phi

    # Family 2: Conveyor drift channel (central crosswind |x|<0.42 + narrow gate walls at x=±0.28)
    f2_cx = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f2_cy0 = jnp.broadcast_to(jnp.array([0.66, -0.66, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f2_hx = jnp.broadcast_to(jnp.array([0.03, 0.03, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f2_hy = jnp.broadcast_to(jnp.array([0.36, 0.36, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f2_amp, f2_omg, f2_phi = f0_amp, f0_omg, f0_phi

    # Family 3: Moving bumper gate at x=0 (smoothly oscillating gap y_gap(t) = 0.44*sin(0.11*t+phi), gap half-width=0.22)
    f3_cx = jnp.zeros((batch_size, 4), dtype=jnp.float32)
    f3_cy0 = jnp.broadcast_to(jnp.array([0.74, -0.74, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f3_hx = jnp.broadcast_to(jnp.array([0.035, 0.035, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f3_hy = jnp.broadcast_to(jnp.array([0.52, 0.52, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f3_amp = jnp.broadcast_to(jnp.array([0.44, 0.44, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f3_omg = jnp.broadcast_to(jnp.array([0.11, 0.11, 0.0, 0.0], dtype=jnp.float32), (batch_size, 4))
    f3_phi = jnp.stack([phi0, phi0, jnp.zeros((batch_size,)), jnp.zeros((batch_size,))], axis=-1)

    def _sel(a0, a1, a2, a3):
        return jnp.select(
            [layout_id[:, None] == 0, layout_id[:, None] == 1, layout_id[:, None] == 2],
            [a0, a1, a2],
            default=a3,
        )

    box_cx = _sel(f0_cx, f1_cx, f2_cx, f3_cx)
    box_cy0 = _sel(f0_cy0, f1_cy0, f2_cy0, f3_cy0)
    box_hx = _sel(f0_hx, f1_hx, f2_hx, f3_hx)
    box_hy = _sel(f0_hy, f1_hy, f2_hy, f3_hy)
    box_amp = _sel(f0_amp, f1_amp, f2_amp, f3_amp)
    box_omg = _sel(f0_omg, f1_omg, f2_omg, f3_omg)
    box_phi = _sel(f0_phi, f1_phi, f2_phi, f3_phi)

    is_drift = layout_id == V3_FAM_CONVEYOR_DRIFT
    is_bumper = layout_id == V3_FAM_MOVING_BUMPER_GATE
    drift_xy = jnp.stack([jnp.zeros((batch_size,)), jnp.where(is_drift, 0.0065 * drift_sign, 0.0)], axis=-1)
    t_zero = jnp.zeros((batch_size,), dtype=jnp.float32)
    box_cy = box_cy0 + box_amp * jnp.sin(box_omg * t_zero[:, None] + box_phi)

    return {
        "layout_id": layout_id,
        "t_step": t_zero,
        "ee_xy": ee_xy,
        "ee_vel": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "obj_xy": obj_xy,
        "obj_vel": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "targ_a": targ_a,
        "box_cx": box_cx,
        "box_cy": box_cy,
        "box_cy0": box_cy0,
        "box_hx": box_hx,
        "box_hy": box_hy,
        "box_amp": box_amp,
        "box_omg": box_omg,
        "box_phi": box_phi,
        "mass_scale": jnp.where(is_drift, 1.8, 1.0),
        "damping": jnp.where(is_drift, 0.945, 0.72),
        "drift_xy": drift_xy,
        "bumper_kick": jnp.where(is_bumper, 0.14, 0.0),
    }


def step_v3_dynamics_world(state: dict[str, jax.Array], action: jax.Array) -> tuple[dict[str, jax.Array], jax.Array]:
    """Advance rigid-body tabletop dynamics with central conveyor drift channel & moving bumper gates."""
    t_next = state["t_step"] + 1.0
    box_cy_next = state["box_cy0"] + state["box_amp"] * jnp.sin(state["box_omg"] * t_next[:, None] + state["box_phi"])

    u_xy = jnp.clip(action[:, :2], -1.0, 1.0)
    ee_vel = 0.55 * state["ee_vel"] + 0.45 * (0.11 * u_xy)
    ee_next = jnp.clip(state["ee_xy"] + ee_vel, -0.90, 0.90)
    ee_next, ee_vel, ee_wall = _box_sdf_push_out(
        ee_next, ee_vel, state["box_cx"], box_cy_next, state["box_hx"], state["box_hy"], radius=0.045
    )

    obj_xy = state["obj_xy"]
    obj_vel = state["obj_vel"]
    diff = obj_xy - ee_next
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1, keepdims=True) + 1e-8)
    min_sep = 0.125
    in_contact = dist < min_sep
    normal = diff / dist
    impulse = jnp.where(in_contact, (min_sep - dist) * normal * (1.15 / state["mass_scale"][:, None]), 0.0)

    # Central conveyor drift acts strictly inside the central channel |x| < 0.42 (calm landing zone |x| >= 0.42)
    in_channel = (jnp.abs(obj_xy[:, 0:1]) < 0.42).astype(jnp.float32)
    obj_vel_next = state["damping"][:, None] * obj_vel + impulse + in_channel * state["drift_xy"]
    obj_next = jnp.clip(obj_xy + obj_vel_next, -0.90, 0.90)
    obj_next, obj_vel_next, obj_wall = _box_sdf_push_out(
        obj_next, obj_vel_next, state["box_cx"], box_cy_next, state["box_hx"], state["box_hy"], radius=0.075
    )

    # Active moving bumper kick on moving_bumper_gate: colliding with the moving barrier kicks the puck away along x!
    kick_mag = state["bumper_kick"][:, None]
    kick_dir_x = jnp.where(obj_xy[:, 0:1] >= 0.0, 1.0, -1.0)
    kick_vec = jnp.concatenate([kick_dir_x * kick_mag, jnp.zeros_like(kick_mag)], axis=-1)
    hit_bumper = (obj_wall & (state["bumper_kick"] > 1e-4))[:, None]
    obj_vel_next = jnp.where(hit_bumper, obj_vel_next + kick_vec, obj_vel_next)
    obj_next = jnp.where(hit_bumper, jnp.clip(obj_next + kick_vec, -0.90, 0.90), obj_next)

    contact_scalar = (jnp.squeeze(in_contact, axis=-1) | ee_wall | obj_wall).astype(jnp.float32)
    return {
        **state,
        "t_step": t_next,
        "box_cy": box_cy_next,
        "ee_xy": ee_next,
        "ee_vel": ee_vel,
        "obj_xy": obj_next,
        "obj_vel": obj_vel_next,
    }, contact_scalar


# ---------------------------------------------------------------------------
# 1. Exact Paired Statistical Power & Crossed-Cluster Design-Effect Calculator
# ---------------------------------------------------------------------------


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (Acklam rational approximation, error < 1e-9)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137128996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
            (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (
            ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / (
        (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0
    )


def compute_paired_power_requirements(
    effect_pp: float = 5.0,
    discordant_rate: float = 0.25,
    alpha: float = 0.05,
    power: float = 0.80,
    cluster_size_m: float = 1.0,
    icc: float = 0.0,
) -> dict[str, float]:
    """Compute required independent and cluster-adjusted paired sample counts for McNemar / paired test."""
    delta = float(effect_pp) / 100.0
    if delta <= 0.0:
        raise ValueError("effect_pp must be strictly positive")
    if not (0.0 < discordant_rate <= 1.0):
        raise ValueError("discordant_rate must be in (0, 1]")
    if delta > discordant_rate:
        raise ValueError("delta cannot exceed discordant_rate")

    z_alpha = _norm_ppf(1.0 - alpha / 2.0)
    z_power = _norm_ppf(power)
    num = (z_alpha * math.sqrt(discordant_rate) + z_power * math.sqrt(max(discordant_rate - delta * delta, 1e-9))) ** 2
    n_indep = math.ceil(num / (delta * delta))
    deff = 1.0 + max(0.0, cluster_size_m - 1.0) * max(0.0, icc)
    n_clustered = math.ceil(n_indep * deff)
    return {
        "effect_pp": float(effect_pp),
        "discordant_rate": float(discordant_rate),
        "alpha": float(alpha),
        "power": float(power),
        "n_independent_pairs": int(n_indep),
        "cluster_size_m": float(cluster_size_m),
        "icc": float(icc),
        "design_effect_deff": float(deff),
        "n_clustered_pairs": int(n_clustered),
    }


def compute_crossed_design_power(
    n_episodes: int,
    n_clusters_a: int,
    icc_a: float,
    n_clusters_b: int = 0,
    icc_b: float = 0.0,
    discordant_rate: float = 0.25,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float]:
    """Compute exact effective sample size N_eff, 95% CI half-width (at delta=0), and MDE for crossed clusters.

    Under additive crossed random effects (cluster factor A with m_a = N/K_a and optional factor B
    with m_b = N/K_b), DEFF = 1 + (m_a - 1)*icc_a + (m_b - 1)*icc_b, N_eff = N / DEFF.
    """
    m_a = float(n_episodes) / float(max(1, n_clusters_a))
    deff = 1.0 + max(0.0, m_a - 1.0) * max(0.0, icc_a)
    m_b = 0.0
    if n_clusters_b > 0 and icc_b > 0.0:
        m_b = float(n_episodes) / float(n_clusters_b)
        deff += max(0.0, m_b - 1.0) * icc_b
    n_eff = float(n_episodes) / deff
    z_a = _norm_ppf(1.0 - alpha / 2.0)
    z_p = _norm_ppf(power)
    se_null_pp = 100.0 * math.sqrt(discordant_rate / max(n_eff, 1e-9))
    ci95_half_width_pp = z_a * se_null_pp
    mde_pp = (z_a + z_p) * se_null_pp
    return {
        "n_episodes": int(n_episodes),
        "n_clusters_a": int(n_clusters_a),
        "m_a": round(m_a, 3),
        "icc_a": float(icc_a),
        "n_clusters_b": int(n_clusters_b),
        "m_b": round(m_b, 3),
        "icc_b": float(icc_b),
        "deff": round(deff, 3),
        "n_eff": round(n_eff, 1),
        "discordant_rate": float(discordant_rate),
        "null_ci95_half_width_pp": round(ci95_half_width_pp, 2),
        "mde_80pct_pp": round(mde_pp, 2),
    }


def stacked_sliding_window_receptive_field(num_layers: int, window_size: int) -> int:
    """Exact maximum relay span L * (W - 1) of an L-layer causal sliding-window transformer."""
    if num_layers < 1 or window_size < 1:
        raise ValueError("num_layers and window_size must be >= 1")
    return int(num_layers * (window_size - 1))


# ---------------------------------------------------------------------------
# 2. Fair, Decoupled 2x2 Oracle-Perception x Oracle-Dynamics Matrix (16x16 grid)
# ---------------------------------------------------------------------------


def _push_action_toward_waypoint(
    ee_xy: jax.Array,
    obj_xy: jax.Array,
    obj_vel: jax.Array,
    waypoint_xy: jax.Array,
    final_goal_xy: jax.Array,
    *,
    is_hold_mode: jax.Array | None = None,
    push_gain: float = 0.72,
) -> jax.Array:
    """Pure geometric contact-push primitive (identical across ALL planners & controls)."""
    del obj_vel
    to_wp = waypoint_xy - obj_xy
    d_wp = jnp.sqrt(jnp.sum(to_wp * to_wp, axis=-1, keepdims=True) + 1e-8)
    dir_wp = to_wp / d_wp

    to_fg = final_goal_xy - obj_xy
    d_fg = jnp.sqrt(jnp.sum(to_fg * to_fg, axis=-1, keepdims=True) + 1e-8)

    ee_to_obj = obj_xy - ee_xy
    d_eo = jnp.sqrt(jnp.sum(ee_to_obj * ee_to_obj, axis=-1, keepdims=True) + 1e-8)
    align = jnp.sum((ee_to_obj / d_eo) * dir_wp, axis=-1, keepdims=True)

    perp = jnp.stack([-dir_wp[..., 1], dir_wp[..., 0]], axis=-1)
    side_sign = jnp.where(jnp.sum((ee_xy - obj_xy) * perp, axis=-1, keepdims=True) >= 0.0, 1.0, -1.0)
    behind_pt = obj_xy - 0.145 * dir_wp + jnp.where(align < 0.25, 0.18 * side_sign * perp, 0.0)
    push_pt = obj_xy - 0.055 * dir_wp

    ready_to_push = (align > 0.55) & (d_eo < 0.21)
    target_ee = jnp.where(ready_to_push, push_pt, behind_pt)

    in_goal_hold = d_fg < 0.16
    safe_park = jnp.clip(obj_xy - 0.25 * dir_wp, -0.88, 0.88)
    use_hold = (in_goal_hold | is_hold_mode[..., None]) if is_hold_mode is not None else in_goal_hold
    target_ee = jnp.where(use_hold, safe_park, target_ee)

    gain = jnp.where(ready_to_push & (~use_hold), 3.0 * push_gain, 4.0)
    u_xy = jnp.clip((target_ee - ee_xy) * gain, -1.0, 1.0)
    u_xy = jnp.where(use_hold & (d_eo > 0.19), 0.0, u_xy)
    pad = jnp.zeros(u_xy.shape[:-1] + (2,), dtype=jnp.float32)
    return jnp.concatenate([u_xy, pad], axis=-1)


def _propose_isotropic_two_stage_candidates(
    obj_xy: jax.Array,
    goal_xy: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate K=38 isotropic 2-stage subgoal sequences (wp1, wp2, is_hold) with ZERO wall priors."""
    b = obj_xy.shape[0]
    to_g = goal_xy - obj_xy
    d_g = jnp.sqrt(jnp.sum(to_g * to_g, axis=-1, keepdims=True) + 1e-8)
    fwd = to_g / d_g
    perp = jnp.stack([-fwd[:, 1], fwd[:, 0]], axis=-1)

    c0_w1 = goal_xy[:, None, :]
    c0_w2 = goal_xy[:, None, :]
    c1_w1 = obj_xy[:, None, :]
    c1_w2 = obj_xy[:, None, :]

    # 16 radial 1-segment pushes (r=0.40 — finely resolves narrow chicanes & clutter for both static and rollout!)
    ang = jnp.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=jnp.float32)
    dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)[None, :, :]
    rad_wp = jnp.clip(obj_xy[:, None, :] + 0.40 * dirs, -0.82, 0.82)

    lat_offsets = jnp.array([-0.68, -0.48, -0.32, -0.16, 0.16, 0.32, 0.48, 0.68], dtype=jnp.float32)
    fwd_offsets = jnp.array([-0.14, 0.14], dtype=jnp.float32)
    lat_grid, fwd_grid = jnp.meshgrid(lat_offsets, fwd_offsets, indexing="ij")
    lat_f = lat_grid.reshape(1, 16, 1)
    fwd_f = fwd_grid.reshape(1, 16, 1)
    det_w1 = jnp.clip(obj_xy[:, None, :] + lat_f * perp[:, None, :] + fwd_f * fwd[:, None, :], -0.82, 0.82)
    det_w2 = jnp.clip(det_w1 + 0.45 * fwd[:, None, :], -0.82, 0.82)

    y_corridors = jnp.array([-0.72, -0.42, 0.42, 0.72], dtype=jnp.float32)[None, :, None]
    x_mid = jnp.broadcast_to(0.5 * (obj_xy[:, 0:1] + goal_xy[:, 0:1])[:, None, :], (b, 4, 1))
    s_w1 = jnp.concatenate([jnp.broadcast_to(obj_xy[:, None, 0:1], (b, 4, 1)), jnp.broadcast_to(y_corridors, (b, 4, 1))], axis=-1)
    s_w2 = jnp.concatenate([x_mid, jnp.broadcast_to(y_corridors, (b, 4, 1))], axis=-1)

    wp1 = jnp.concatenate([c0_w1, c1_w1, rad_wp, det_w1, s_w1], axis=1)
    wp2 = jnp.concatenate([c0_w2, c1_w2, rad_wp, det_w2, s_w2], axis=1)
    k_tot = wp1.shape[1]
    is_hold = jnp.zeros((b, k_tot), dtype=bool).at[:, 1].set(True)
    return wp1, wp2, is_hold


def _occ_grid_16x16_from_boxes(st: dict[str, jax.Array]) -> jax.Array:
    """16x16 unsealed obstacle occupancy grid from perceived box geometry (cell width 0.125; cleanly preserves x=0 corridor)."""
    c16 = jnp.linspace(-0.9375, 0.9375, 16, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(c16, c16, indexing="ij")
    dx = jnp.abs(xx[None, :, :, None] - st["box_cx"][:, None, None, :])
    dy = jnp.abs(yy[None, :, :, None] - st["box_cy"][:, None, None, :])
    hit = (st["box_hx"][:, None, None, :] > 1e-4) & (dx <= (st["box_hx"][:, None, None, :] + 0.085)) & (
        dy <= (st["box_hy"][:, None, None, :] + 0.085)
    )
    return jnp.any(hit, axis=-1).astype(jnp.float32)


def _geodesic_value_grid_16x16(occ_16: jax.Array, goal_xy: jax.Array, n_iters: int = 36) -> jax.Array:
    """Differentiable 8-connected shortest-path potential on 16x16 unsealed occupancy grid [B, 16, 16]."""
    c16 = jnp.linspace(-0.9375, 0.9375, 16, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(c16, c16, indexing="ij")
    d_goal = jnp.sqrt((xx[None] - goal_xy[:, 0:1, None]) ** 2 + (yy[None] - goal_xy[:, 1:2, None]) ** 2 + 1e-6)
    step_c = 0.125
    diag_c = step_c * math.sqrt(2.0)
    cell_pen = 6.0 * occ_16
    v = jnp.where(d_goal < 0.18, d_goal, 20.0)

    def _relax(v_cur, _):
        vp = jnp.pad(v_cur, ((0, 0), (1, 1), (1, 1)), constant_values=20.0)
        nbrs = [
            vp[:, 0:16, 1:17] + step_c,
            vp[:, 2:18, 1:17] + step_c,
            vp[:, 1:17, 0:16] + step_c,
            vp[:, 1:17, 2:18] + step_c,
            vp[:, 0:16, 0:16] + diag_c,
            vp[:, 0:16, 2:18] + diag_c,
            vp[:, 2:18, 0:16] + diag_c,
            vp[:, 2:18, 2:18] + diag_c,
        ]
        best_n = jnp.minimum(v_cur, jnp.min(jnp.stack(nbrs, axis=0), axis=0) + cell_pen)
        return jnp.where(d_goal < 0.18, d_goal, best_n), None

    v_fin, _ = jax.lax.scan(_relax, v, None, length=n_iters)
    return v_fin


def _bilinear_sample_grid(grid: jax.Array, xy: jax.Array) -> jax.Array:
    """Bilinearly sample [B, G, G] grid (centers on [-1+1/G, 1-1/G]) at xy [B, K, 2]."""
    g = grid.shape[1]
    step = 2.0 / float(g)
    origin = -1.0 + 0.5 * step
    gx = jnp.clip((xy[..., 0] - origin) / step, 0.0, float(g - 1) - 1e-4)
    gy = jnp.clip((xy[..., 1] - origin) / step, 0.0, float(g - 1) - 1e-4)
    x0 = jnp.floor(gx).astype(jnp.int32)
    y0 = jnp.floor(gy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, g - 1)
    y1 = jnp.minimum(y0 + 1, g - 1)
    wx = gx - x0.astype(jnp.float32)
    wy = gy - y0.astype(jnp.float32)
    b_idx = jnp.arange(grid.shape[0])[:, None]
    v00 = grid[b_idx, y0, x0]
    v01 = grid[b_idx, y0, x1]
    v10 = grid[b_idx, y1, x0]
    v11 = grid[b_idx, y1, x1]
    return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)


def _plan_shared_rollout_mpc(
    perceived_st: dict[str, jax.Array],
    goal_xy: jax.Array,
    prev_wp1: jax.Array,
    has_prev_wp: jax.Array | bool,
    *,
    horizon: int = 14,
    use_rollout: bool = True,
    push_gain: float = 0.72,
) -> tuple[jax.Array, jax.Array]:
    """Dynamics-rollout receding-horizon MPC sharing exact 16x16 occupancy grid, candidates, and weights with static geodesic."""
    b = perceived_st["ee_xy"].shape[0]
    wp1, wp2, is_hold = _propose_isotropic_two_stage_candidates(perceived_st["obj_xy"], goal_xy)
    k_tot = wp1.shape[1]
    eff_h = int(horizon) if use_rollout else 1
    half_h = max(1, eff_h // 2)

    occ_16 = _occ_grid_16x16_from_boxes(perceived_st)
    v_grid = _geodesic_value_grid_16x16(occ_16, goal_xy)

    st_bk = {k: jnp.repeat(v, k_tot, axis=0) for k, v in perceived_st.items() if k != "layout_id"}
    w1_bk = wp1.reshape(b * k_tot, 2)
    w2_bk = wp2.reshape(b * k_tot, 2)
    hold_bk = is_hold.reshape(b * k_tot)
    fg_bk = jnp.repeat(goal_xy, k_tot, axis=0)

    def _step(carry, t_idx):
        cst, acc_pen = carry
        active_wp = jnp.where(t_idx < half_h, w1_bk, w2_bk)
        u_bk = _push_action_toward_waypoint(
            cst["ee_xy"], cst["obj_xy"], cst["obj_vel"], active_wp, fg_bk, is_hold_mode=hold_bk, push_gain=push_gain
        )
        nxt, _ = step_v3_dynamics_world(cst, u_bk)
        # Collision penalty evaluated strictly on the rolled-out 16x16 occupancy grid at tick t+k (shared across K candidates)!
        nxt_occ_16 = _occ_grid_16x16_from_boxes({k: v[::k_tot] for k, v in nxt.items() if k.startswith("box_")})
        obj_occ = _bilinear_sample_grid(nxt_occ_16, nxt["obj_xy"].reshape(b, k_tot, 2)).reshape(b * k_tot)
        ee_occ = _bilinear_sample_grid(nxt_occ_16, nxt["ee_xy"].reshape(b, k_tot, 2)).reshape(b * k_tot)
        d_t = jnp.sqrt(jnp.sum((nxt["obj_xy"] - fg_bk) ** 2, axis=-1) + 1e-8)
        speed_t = jnp.sqrt(jnp.sum(nxt["obj_vel"] ** 2, axis=-1) + 1e-8)
        overshoot_pen = jnp.where(d_t < 0.24, 1.8 * speed_t, 0.0)
        return (nxt, acc_pen + 3.0 * obj_occ + 0.6 * ee_occ + overshoot_pen), None

    (fin_st, total_pen), _ = jax.lax.scan(
        _step,
        (st_bk, jnp.zeros((b * k_tot,), dtype=jnp.float32)),
        jnp.arange(eff_h, dtype=jnp.int32),
    )

    fin_obj = fin_st["obj_xy"].reshape(b, k_tot, 2)
    fin_d = jnp.sqrt(jnp.sum((fin_obj - goal_xy[:, None, :]) ** 2, axis=-1) + 1e-8)
    # Evaluate terminal geodesic value on the FUTURE 16x16 occupancy grid at t + eff_h!
    fin_occ_16 = _occ_grid_16x16_from_boxes({k: v[::k_tot] for k, v in fin_st.items() if k.startswith("box_")})
    fin_v_grid = _geodesic_value_grid_16x16(fin_occ_16, goal_xy)
    fin_v = _bilinear_sample_grid(jnp.minimum(fin_v_grid, 12.0), fin_obj)
    fin_speed = jnp.sqrt(jnp.sum(fin_st["obj_vel"].reshape(b, k_tot, 2) ** 2, axis=-1) + 1e-8)
    pen = total_pen.reshape(b, k_tot)

    ret = -2.4 * fin_v - 0.8 * fin_d - 1.2 * (pen / float(eff_h)) - jnp.where(fin_d < 0.24, 2.5 * fin_speed, 0.0)

    has_p = jnp.broadcast_to(jnp.asarray(has_prev_wp, dtype=bool), (b,))
    pwp = jnp.where(has_p[:, None], prev_wp1, goal_xy)
    hyst = jnp.where(has_p[:, None], -0.05 * jnp.sqrt(jnp.sum((wp1 - pwp[:, None, :]) ** 2, axis=-1) + 1e-6), 0.0)
    ret = ret + hyst

    best_idx = jnp.argmax(ret, axis=-1)
    best_w1 = wp1[jnp.arange(b), best_idx]
    best_hold = is_hold[jnp.arange(b), best_idx]
    u0 = _push_action_toward_waypoint(
        perceived_st["ee_xy"],
        perceived_st["obj_xy"],
        perceived_st["obj_vel"],
        best_w1,
        goal_xy,
        is_hold_mode=best_hold,
        push_gain=push_gain,
    )
    return u0, best_w1


def _plan_static_geodesic_no_rollout(
    perceived_st: dict[str, jax.Array],
    goal_xy: jax.Array,
    prev_wp1: jax.Array,
    has_prev_wp: jax.Array | bool,
    *,
    push_gain: float = 0.72,
) -> tuple[jax.Array, jax.Array]:
    """Fair non-rollout comparator: identical 16x16 grid, identical K=38 `(wp1, wp2)` candidates, and matched weights.

    Evaluates BOTH stage-1 (`obj -> wp1`) and stage-2 (`wp1 -> wp2`) segments on the 16x16 occupancy
    grid plus terminal `-2.4 * v_grid(wp2) - 0.8 * d(wp2, goal)` with ZERO forward dynamics simulation.
    """
    b = perceived_st["ee_xy"].shape[0]
    wp1, wp2, is_hold = _propose_isotropic_two_stage_candidates(perceived_st["obj_xy"], goal_xy)
    occ_16 = _occ_grid_16x16_from_boxes(perceived_st)
    v_grid = _geodesic_value_grid_16x16(occ_16, goal_xy)

    obj = perceived_st["obj_xy"][:, None, :]
    # Check collision along BOTH segments (obj -> wp1 and wp1 -> wp2) at 4 sample points each
    alphas = jnp.array([0.25, 0.50, 0.75, 1.00], dtype=jnp.float32)[None, None, :, None]
    seg1 = obj[:, :, None, :] * (1.0 - alphas) + wp1[:, :, None, :] * alphas  # [B, K, 4, 2]
    seg2 = wp1[:, :, None, :] * (1.0 - alphas) + wp2[:, :, None, :] * alphas
    occ_s1 = jnp.max(_bilinear_sample_grid(occ_16, seg1.reshape(b, -1, 2)).reshape(b, wp1.shape[1], 4), axis=-1)
    occ_s2 = jnp.max(_bilinear_sample_grid(occ_16, seg2.reshape(b, -1, 2)).reshape(b, wp1.shape[1], 4), axis=-1)
    edge_occ = jnp.maximum(occ_s1, occ_s2)

    fin_v = _bilinear_sample_grid(jnp.minimum(v_grid, 12.0), wp2)
    fin_d = jnp.sqrt(jnp.sum((wp2 - goal_xy[:, None, :]) ** 2, axis=-1) + 1e-8)
    d_goal_now = jnp.sqrt(jnp.sum((perceived_st["obj_xy"] - goal_xy) ** 2, axis=-1, keepdims=True) + 1e-8)
    hold_pen = jnp.where(is_hold & (d_goal_now >= 0.16), 20.0, 0.0)

    ret = -2.4 * fin_v - 0.8 * fin_d - 12.0 * edge_occ - hold_pen

    has_p = jnp.broadcast_to(jnp.asarray(has_prev_wp, dtype=bool), (b,))
    pwp = jnp.where(has_p[:, None], prev_wp1, goal_xy)
    hyst = jnp.where(has_p[:, None], -0.05 * jnp.sqrt(jnp.sum((wp1 - pwp[:, None, :]) ** 2, axis=-1) + 1e-6), 0.0)
    ret = ret + hyst

    best_idx = jnp.argmax(ret, axis=-1)
    best_w1 = wp1[jnp.arange(b), best_idx]
    best_hold = is_hold[jnp.arange(b), best_idx]
    u0 = _push_action_toward_waypoint(
        perceived_st["ee_xy"],
        perceived_st["obj_xy"],
        perceived_st["obj_vel"],
        best_w1,
        goal_xy,
        is_hold_mode=best_hold,
        push_gain=push_gain,
    )
    return u0, best_w1


def run_gate1_2x2_oracle_control_matrix(
    *,
    seed: int = 31001,
    n_worlds: int = 256,
    horizon_ticks: int = 180,
    rollout_horizon: int = 14,
    perception_noise_std: float = 0.04,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Run the decoupled Perception x Dynamics x Rollout control matrix across all 9 arms with zero confounds."""
    init_state = init_v3_dynamics_control_worlds(jax.random.key(seed), batch_size=n_worlds)
    goal_xy = init_state["targ_a"]

    @functools.partial(jax.jit, static_argnums=(0, 1, 2))
    def _eval_arm(noisy_perception: bool, nominal_dynamics: bool, mode: str):
        def body(carry, t_idx):
            st, prev_wp, prev_obs_obj, rk = carry
            rk, k_obj, k_box = jax.random.split(rk, 3)

            if noisy_perception:
                obs_obj = jnp.clip(
                    st["obj_xy"] + perception_noise_std * jax.random.normal(k_obj, st["obj_xy"].shape),
                    -0.92,
                    0.92,
                )
                obs_obj_vel = jnp.where(t_idx > 0, jnp.clip(obs_obj - prev_obs_obj, -0.18, 0.18), 0.0)
                obs_bcx = st["box_cx"] + 0.5 * perception_noise_std * jax.random.normal(k_box, st["box_cx"].shape)
                obs_bcy0 = st["box_cy0"] + 0.5 * perception_noise_std * jax.random.normal(
                    jax.random.fold_in(k_box, 1), st["box_cy0"].shape
                )
                obs_bcy = obs_bcy0 + st["box_amp"] * jnp.sin(st["box_omg"] * st["t_step"][:, None] + st["box_phi"])
                perc_st = {
                    **st,
                    "obj_xy": obs_obj,
                    "obj_vel": obs_obj_vel,
                    "box_cx": obs_bcx,
                    "box_cy0": obs_bcy0,
                    "box_cy": obs_bcy,
                }
            else:
                obs_obj = st["obj_xy"]
                perc_st = dict(st)

            if nominal_dynamics:
                # Misspecified quasistatic/nominal dynamics: freezes moving walls & assumes mass=1, damp=0.72, drift=0, bumper=0
                perc_st = {
                    **perc_st,
                    "box_cy0": perc_st["box_cy"],
                    "box_amp": jnp.zeros_like(st["box_amp"]),
                    "box_omg": jnp.zeros_like(st["box_omg"]),
                    "mass_scale": jnp.ones_like(st["mass_scale"]),
                    "damping": jnp.full_like(st["damping"], 0.72),
                    "drift_xy": jnp.zeros_like(st["drift_xy"]),
                    "bumper_kick": jnp.zeros_like(st["bumper_kick"]),
                }

            if mode == "reactive":
                u_t = _push_action_toward_waypoint(
                    perc_st["ee_xy"], perc_st["obj_xy"], perc_st["obj_vel"], goal_xy, goal_xy
                )
                next_wp = goal_xy
            elif mode == "static_geodesic":
                u_t, next_wp = _plan_static_geodesic_no_rollout(perc_st, goal_xy, prev_wp, t_idx > 0)
            else:
                u_t, next_wp = _plan_shared_rollout_mpc(
                    perc_st,
                    goal_xy,
                    prev_wp,
                    t_idx > 0,
                    horizon=rollout_horizon,
                    use_rollout=(mode == "rollout"),
                )

            st_next, _ = step_v3_dynamics_world(st, u_t)
            dist_g = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_xy) ** 2, axis=-1) + 1e-8)
            return (st_next, next_wp, obs_obj, rk), dist_g

        (st_fin, _, _, _), dist_hist = jax.lax.scan(
            body,
            (init_state, goal_xy, init_state["obj_xy"], jax.random.key(seed + 99)),
            jnp.arange(horizon_ticks, dtype=jnp.int32),
        )
        tail_w = 16
        closest_tail = (jnp.min(dist_hist[-tail_w:], axis=0) < 0.24).astype(jnp.float32)
        sustained_hold = (jnp.max(dist_hist[-tail_w:], axis=0) < 0.24).astype(jnp.float32)
        any_reach = (jnp.min(dist_hist, axis=0) < 0.24).astype(jnp.float32)
        final_dist = dist_hist[-1]
        return closest_tail, sustained_hold, any_reach, final_dist, st_fin["layout_id"]

    arm_specs = {
        "oracle_perc_oracle_dyn_rollout": (False, False, "rollout"),
        "oracle_perc_nominal_dyn_rollout": (False, True, "rollout"),
        "oracle_perc_static_geodesic_no_rollout": (False, False, "static_geodesic"),
        "oracle_perc_no_rollout_h1": (False, False, "h1"),
        "noisy_perc_oracle_dyn_rollout": (True, False, "rollout"),
        "noisy_perc_nominal_dyn_rollout": (True, True, "rollout"),
        "noisy_perc_static_geodesic_no_rollout": (True, False, "static_geodesic"),
        "exact_state_reactive": (False, False, "reactive"),
        "noisy_state_reactive": (True, False, "reactive"),
    }

    fam_map = {
        V3_FAM_STATIC_WALL: "static_off_center_wall",
        V3_FAM_DUAL_CHICANE: "dual_chicane",
        V3_FAM_CONVEYOR_DRIFT: "conveyor_drift_channel",
        V3_FAM_MOVING_BUMPER_GATE: "moving_bumper_gate",
    }

    arms: dict[str, dict[str, Any]] = {}
    raw_vecs: dict[str, dict[str, np.ndarray]] = {}
    layout_id_np = None

    for name, (noisy, nom_dyn, mode) in arm_specs.items():
        c_t, h_t, a_t, fd_t, lid_t = jax.device_get(_eval_arm(noisy, nom_dyn, mode))
        c_np = np.asarray(c_t, dtype=np.float64)
        h_np = np.asarray(h_t, dtype=np.float64)
        a_np = np.asarray(a_t, dtype=np.float64)
        fd_np = np.asarray(fd_t, dtype=np.float64)
        layout_id_np = np.asarray(lid_t, dtype=np.int32)
        raw_vecs[name] = {"closest": c_np, "hold": h_np, "any": a_np}

        by_fam = {}
        for code, fname in fam_map.items():
            m = layout_id_np == code
            by_fam[fname] = {
                "closest_approach_tail16": float(np.mean(c_np[m])),
                "sustained_hold_tail16": float(np.mean(h_np[m])),
                "any_tick_reach": float(np.mean(a_np[m])),
                "mean_final_dist": float(np.mean(fd_np[m])),
            }
        arms[name] = {
            "closest_approach_tail16": float(np.mean(c_np)),
            "sustained_hold_tail16": float(np.mean(h_np)),
            "any_tick_reach": float(np.mean(a_np)),
            "mean_final_dist": float(np.mean(fd_np)),
            "by_family": by_fam,
        }

    rng = np.random.default_rng(seed + 777)

    def _paired_ci_pp(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        diff = (a - b) * 100.0
        idx = rng.integers(0, diff.size, size=(n_bootstrap, diff.size))
        means = np.mean(diff[idx], axis=1)
        return {
            "delta_pp": float(np.mean(diff)),
            "ci95_low_pp": float(np.percentile(means, 2.5)),
            "ci95_high_pp": float(np.percentile(means, 97.5)),
        }

    contrasts = {
        "oracle_rollout_vs_static_geodesic_closest": _paired_ci_pp(
            raw_vecs["oracle_perc_oracle_dyn_rollout"]["closest"],
            raw_vecs["oracle_perc_static_geodesic_no_rollout"]["closest"],
        ),
        "oracle_rollout_vs_static_geodesic_hold": _paired_ci_pp(
            raw_vecs["oracle_perc_oracle_dyn_rollout"]["hold"],
            raw_vecs["oracle_perc_static_geodesic_no_rollout"]["hold"],
        ),
        "oracle_rollout_vs_nominal_dyn_hold": _paired_ci_pp(
            raw_vecs["oracle_perc_oracle_dyn_rollout"]["hold"],
            raw_vecs["oracle_perc_nominal_dyn_rollout"]["hold"],
        ),
        "oracle_nominal_rollout_vs_static_geodesic_hold": _paired_ci_pp(
            raw_vecs["oracle_perc_nominal_dyn_rollout"]["hold"],
            raw_vecs["oracle_perc_static_geodesic_no_rollout"]["hold"],
        ),
        "oracle_rollout_vs_exact_reactive_closest": _paired_ci_pp(
            raw_vecs["oracle_perc_oracle_dyn_rollout"]["closest"],
            raw_vecs["exact_state_reactive"]["closest"],
        ),
        "oracle_rollout_vs_exact_reactive_hold": _paired_ci_pp(
            raw_vecs["oracle_perc_oracle_dyn_rollout"]["hold"],
            raw_vecs["exact_state_reactive"]["hold"],
        ),
        "noisy_oracle_rollout_vs_noisy_static_geodesic_closest": _paired_ci_pp(
            raw_vecs["noisy_perc_oracle_dyn_rollout"]["closest"],
            raw_vecs["noisy_perc_static_geodesic_no_rollout"]["closest"],
        ),
        "noisy_oracle_rollout_vs_noisy_static_geodesic_hold": _paired_ci_pp(
            raw_vecs["noisy_perc_oracle_dyn_rollout"]["hold"],
            raw_vecs["noisy_perc_static_geodesic_no_rollout"]["hold"],
        ),
        "noisy_nominal_rollout_vs_noisy_static_geodesic_hold": _paired_ci_pp(
            raw_vecs["noisy_perc_nominal_dyn_rollout"]["hold"],
            raw_vecs["noisy_perc_static_geodesic_no_rollout"]["hold"],
        ),
        "oracle_perc_vs_noisy_perc_true_dyn_hold": _paired_ci_pp(
            raw_vecs["oracle_perc_oracle_dyn_rollout"]["hold"],
            raw_vecs["noisy_perc_oracle_dyn_rollout"]["hold"],
        ),
    }

    gate1_control_pass = bool(
        arms["oracle_perc_oracle_dyn_rollout"]["closest_approach_tail16"] >= 0.85
        and arms["oracle_perc_oracle_dyn_rollout"]["sustained_hold_tail16"] >= 0.80
        and contrasts["oracle_rollout_vs_static_geodesic_hold"]["ci95_low_pp"] > 0.0
        and contrasts["oracle_rollout_vs_exact_reactive_closest"]["ci95_low_pp"] > 10.0
        and contrasts["oracle_rollout_vs_exact_reactive_hold"]["ci95_low_pp"] > 10.0
    )

    return {
        "n_worlds": int(n_worlds),
        "horizon_ticks": int(horizon_ticks),
        "rollout_horizon": int(rollout_horizon),
        "occupancy_grid_res": 16,
        "perception_noise_std": float(perception_noise_std),
        "arms": arms,
        "paired_contrasts": contrasts,
        "gate1_control_pass": gate1_control_pass,
    }


# ---------------------------------------------------------------------------
# 3. Shortcut-Free Randomized Delayed-Correction & Concurrent-Fact Certifier
# ---------------------------------------------------------------------------


def _build_unit_norm_codebook(vocab_size: int = 256, dim: int = 8, seed: int = 17) -> np.ndarray:
    """Deterministic unit-norm continuous codebook E[vocab_size, dim] cleanly representable in bfloat16."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 1.0, size=(vocab_size, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)


def generate_paired_delayed_correction_streams(
    *,
    seed: int = 31002,
    batch_size: int = 32,
    comparator_layers: int = 12,
    comparator_window: int = 32,
    extra_margin_ticks: int = 40,
    vocab_size: int = 256,
    bg_noise_std: float = 0.15,
) -> dict[str, Any]:
    """Generate shortcut-free, fully randomized counterfactual paired streams for Wave A selective overwrite.

    Eliminates every fixed-tick, zero-text, bf16-scalar, and 17-sigma-outlier shortcut:
      - Randomizes arrival order of F1 vs F2 per episode (50/50).
      - Randomizes `t_first`, `t_second`, `t_corr` with `t_corr - t_first > comparator_window` so the
        overwritten fact has ALREADY exited the single-layer local sliding window before correction.
      - Sets `total_ticks` dynamically so `min_delay_after_corr > comparator_layers * (comparator_window - 1)`.
      - Adds random per-episode wall-clock start offset `t0_offset ~ Uniform(0, 50s)`.
      - Populates background `text_bytes` with random non-active vocabulary traffic (zero `text != 0` shortcut).
      - Encodes audio/visual cues via unit-norm `R^8` codebook vectors (SNR ~ 6.7 over `σ=0.15` noise;
        zero `bf16` scalar quantization floor).
      - Emits paired `stream_a_clean`, `stream_a_distractor` (matched-norm unrelated `(k_d, v_d)` event at
        random `t_impulse`), `stream_a_corrupt` (moderate `3.5-sigma` sensory burst at random `t_impulse`),
        and `stream_b_clean` (counterfactual correction value), certifying `max_abs_diff == 0.0` across
        ALL 7 post-cue tensors (`t > t_corr[bi]`) for every episode.
    """
    receptive_field = stacked_sliding_window_receptive_field(comparator_layers, comparator_window)
    max_t_corr = 24 + 28 + comparator_window + 24
    total_ticks = max_t_corr + receptive_field + int(extra_margin_ticks) + 2

    rng = np.random.default_rng(seed)
    codebook = _build_unit_norm_codebook(vocab_size=vocab_size, dim=8, seed=seed + 5)

    keys = np.zeros((batch_size, 2), dtype=np.int32)
    init_vals = np.zeros((batch_size, 2), dtype=np.int32)
    corr_val_a = np.zeros((batch_size,), dtype=np.int32)
    corr_val_b = np.zeros((batch_size,), dtype=np.int32)
    dist_key = np.zeros((batch_size,), dtype=np.int32)
    dist_val = np.zeros((batch_size,), dtype=np.int32)

    init_order = rng.integers(0, 2, size=(batch_size,), dtype=np.int32)  # 0: F0 first; 1: F1 first
    which_updated = rng.integers(0, 2, size=(batch_size,), dtype=np.int32)
    cue_modality = rng.integers(0, 3, size=(batch_size, 4), dtype=np.int32)  # [init1, init2, corr, distractor]

    t_first = rng.integers(4, 24, size=(batch_size,), dtype=np.int32)
    t_second = t_first + rng.integers(8, 28, size=(batch_size,), dtype=np.int32)
    # Guarantee t_corr - t_first > comparator_window (fact already left single-layer local window before overwrite!)
    corr_lo = np.maximum(t_second + 10, t_first + comparator_window + 6)
    corr_hi = np.minimum(corr_lo + 18, max_t_corr)
    t_corr = np.array([int(rng.integers(lo, hi + 1)) for lo, hi in zip(corr_lo, corr_hi)], dtype=np.int32)
    t_impulse = np.array(
        [int(rng.integers(tc + 24, total_ticks - 24)) for tc in t_corr], dtype=np.int32
    )

    for bi in range(batch_size):
        sym = rng.choice(np.arange(1, vocab_size, dtype=np.int32), size=8, replace=False)
        keys[bi] = sym[:2]
        init_vals[bi] = sym[2:4]
        corr_val_a[bi] = sym[4]
        corr_val_b[bi] = sym[5]
        dist_key[bi] = sym[6]
        dist_val[bi] = sym[7]

    # Non-zero in-distribution background traffic across ALL modalities (excluding episode's active keys!)
    base_video = rng.normal(0.0, bg_noise_std, size=(batch_size, total_ticks, 16)).astype(np.float32)
    base_audio = rng.normal(0.0, bg_noise_std, size=(batch_size, total_ticks, 16)).astype(np.float32)
    base_text = np.zeros((batch_size, total_ticks, 4), dtype=np.int32)
    for bi in range(batch_size):
        forbidden = {int(keys[bi, 0]), int(keys[bi, 1]), int(dist_key[bi])}
        pool = np.array([x for x in range(1, vocab_size) if x not in forbidden], dtype=np.int32)
        base_text[bi] = rng.choice(pool, size=(total_ticks, 4))
    base_proprio = rng.normal(0.0, 0.25, size=(batch_size, total_ticks, 4)).astype(np.float32)
    base_action = rng.normal(0.0, 0.20, size=(batch_size, total_ticks, 2)).astype(np.float32)
    base_mask = np.ones((batch_size, total_ticks, 4), dtype=np.float32)
    t0_offsets = rng.uniform(0.0, 50.0, size=(batch_size, 1)).astype(np.float32)
    timestamps = (t0_offsets + np.arange(total_ticks, dtype=np.float32)[None, :] * 0.1).astype(np.float32)

    def _encode_kv(vid: np.ndarray, aud: np.ndarray, txt: np.ndarray, t_pos: int, bi: int, k_id: int, v_id: int, mod_id: int) -> None:
        if mod_id == 0:
            txt[bi, t_pos, 0] = k_id
            txt[bi, t_pos, 1] = v_id
        elif mod_id == 1:
            aud[bi, t_pos, :8] += codebook[k_id]
            aud[bi, t_pos, 8:] += codebook[v_id]
        else:
            vid[bi, t_pos, :8] += codebook[k_id]
            vid[bi, t_pos, 8:] += codebook[v_id]

    def _build_stream(corr_vals: np.ndarray, impulse_mode: str) -> dict[str, np.ndarray]:
        vid = base_video.copy()
        aud = base_audio.copy()
        txt = base_text.copy()
        for bi in range(batch_size):
            f_a = int(init_order[bi])
            f_b = 1 - f_a
            _encode_kv(vid, aud, txt, int(t_first[bi]), bi, int(keys[bi, f_a]), int(init_vals[bi, f_a]), int(cue_modality[bi, 0]))
            _encode_kv(vid, aud, txt, int(t_second[bi]), bi, int(keys[bi, f_b]), int(init_vals[bi, f_b]), int(cue_modality[bi, 1]))
            upd = int(which_updated[bi])
            _encode_kv(vid, aud, txt, int(t_corr[bi]), bi, int(keys[bi, upd]), int(corr_vals[bi]), int(cue_modality[bi, 2]))
            if impulse_mode == "distractor_binding":
                _encode_kv(vid, aud, txt, int(t_impulse[bi]), bi, int(dist_key[bi]), int(dist_val[bi]), int(cue_modality[bi, 3]))
            elif impulse_mode == "moderate_3p5sigma_burst":
                # Moderate 3.5-sigma sensory corruption burst (not a trivial 17-sigma LayerNorm outlier!)
                burst_rng = np.random.default_rng(seed + 1000 + bi)
                vid[bi, int(t_impulse[bi])] += burst_rng.normal(0.0, 3.5 * bg_noise_std, size=(16,)).astype(np.float32)
                aud[bi, int(t_impulse[bi])] += burst_rng.normal(0.0, 3.5 * bg_noise_std, size=(16,)).astype(np.float32)
        return {
            "video": vid,
            "audio": aud,
            "text_bytes": txt,
            "proprio": base_proprio.copy(),
            "action": base_action.copy(),
            "modality_mask": base_mask.copy(),
            "timestamps": timestamps.copy(),
        }

    stream_a_clean = _build_stream(corr_val_a, "clean")
    stream_b_clean = _build_stream(corr_val_b, "clean")
    stream_a_distractor = _build_stream(corr_val_a, "distractor_binding")
    stream_a_corrupt = _build_stream(corr_val_a, "moderate_3p5sigma_burst")

    # Per-episode exact post-cue tensor equality check for t > t_corr[bi]
    post_cue_max_abs_diff = {k: 0.0 for k in stream_a_clean}
    cue_tick_max_abs_diff = {k: 0.0 for k in ("video", "audio", "text_bytes")}
    for bi in range(batch_size):
        tc = int(t_corr[bi])
        for k in stream_a_clean:
            d = float(np.max(np.abs(stream_a_clean[k][bi, tc + 1 :].astype(np.float64) - stream_b_clean[k][bi, tc + 1 :].astype(np.float64))))
            post_cue_max_abs_diff[k] = max(post_cue_max_abs_diff[k], d)
        for k in ("video", "audio", "text_bytes"):
            dc = float(np.max(np.abs(stream_a_clean[k][bi, tc].astype(np.float64) - stream_b_clean[k][bi, tc].astype(np.float64))))
            cue_tick_max_abs_diff[k] = max(cue_tick_max_abs_diff[k], dc)

    min_delay = int(np.min((total_ticks - 1) - t_corr))
    min_init_to_corr_gap = int(np.min(t_corr - t_first))
    all_post_cue_zero = all(v == 0.0 for v in post_cue_max_abs_diff.values())
    has_cue_diff = all(v > 0.0 for v in cue_tick_max_abs_diff.values())

    return {
        "stream_a_clean": stream_a_clean,
        "stream_b_clean": stream_b_clean,
        "stream_a_distractor": stream_a_distractor,
        "stream_a_corrupt": stream_a_corrupt,
        "keys": keys,
        "init_vals": init_vals,
        "init_order": init_order,
        "which_updated": which_updated,
        "t_first": t_first,
        "t_second": t_second,
        "t_corr": t_corr,
        "t_impulse": t_impulse,
        "target_updated_val_a": corr_val_a,
        "target_updated_val_b": corr_val_b,
        "target_preserved_val": np.where(which_updated == 0, init_vals[:, 1], init_vals[:, 0]),
        "audit": {
            "total_ticks": int(total_ticks),
            "comparator_layers": int(comparator_layers),
            "comparator_window": int(comparator_window),
            "stacked_receptive_field": int(receptive_field),
            "min_delay_after_corr": min_delay,
            "strictly_beyond_receptive_field": bool(min_delay > receptive_field),
            "min_init_to_corr_gap": min_init_to_corr_gap,
            "overwrite_after_local_window_eviction": bool(min_init_to_corr_gap > comparator_window),
            "t_corr_unique_count": int(len(np.unique(t_corr))),
            "t_impulse_unique_count": int(len(np.unique(t_impulse))),
            "init_order_frac_f1_first": float(np.mean(init_order == 1)),
            "update_target_balance_frac_fact1": float(np.mean(which_updated == 1)),
            "bg_text_nonzero_fraction": float(np.mean(base_text != 0)),
            "post_cue_max_abs_diff_by_tensor": post_cue_max_abs_diff,
            "cue_tick_max_abs_diff_by_modality": cue_tick_max_abs_diff,
            "all_post_cue_tensors_bit_exact_equal": bool(all_post_cue_zero),
            "counterfactual_cue_differs_all_modalities": bool(has_cue_diff),
            "emits_clean_and_stress_paired_controls": True,
            "gate1_counterfactual_pass": bool(
                all_post_cue_zero
                and has_cue_diff
                and (min_delay > receptive_field)
                and (min_init_to_corr_gap > comparator_window)
                and (len(np.unique(t_corr)) > 1)
                and (len(np.unique(t_impulse)) > 1)
            ),
        },
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run SLWM-v3 Gate 1 Decoupled Oracle Matrix & Counterfactual Audit.")
    ap.add_argument("--n-worlds", type=int, default=1024)
    ap.add_argument("--horizon-ticks", type=int, default=180)
    ap.add_argument("--rollout-horizon", type=int, default=14)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    ctrl = run_gate1_2x2_oracle_control_matrix(
        n_worlds=args.n_worlds,
        horizon_ticks=args.horizon_ticks,
        rollout_horizon=args.rollout_horizon,
    )
    cf_disc = generate_paired_delayed_correction_streams(comparator_layers=12, comparator_window=32)
    cf_conf = generate_paired_delayed_correction_streams(comparator_layers=16, comparator_window=64)

    pwr_indep = compute_paired_power_requirements(effect_pp=5.0, discordant_rate=0.25, alpha=0.05, power=0.80)
    pwr_sc = compute_crossed_design_power(n_episodes=1024, n_clusters_a=250, icc_a=0.10, discordant_rate=0.25)
    pwr_grid_only = compute_crossed_design_power(n_episodes=1024, n_clusters_a=8, icc_a=0.10, discordant_rate=0.25)
    pwr_ctrl_1024 = compute_crossed_design_power(n_episodes=1024, n_clusters_a=1024, icc_a=0.0, discordant_rate=0.10)
    pwr_xarm_135 = compute_crossed_design_power(n_episodes=284, n_clusters_a=135, icc_a=0.10, discordant_rate=0.10)

    report = {
        "gate": "SLWM-v3 Gate 1 — Decoupled Oracle Matrix, Shortcut-Free Counterfactuals & Crossed Power",
        "control_matrix": ctrl,
        "counterfactual_discovery_L12_W32": cf_disc["audit"],
        "counterfactual_confirmation_L16_W64": cf_conf["audit"],
        "statistical_power": {
            "independent_pairs_5pp_80pct": pwr_indep,
            "speechcommands_250spk_1024ep_icc010": pwr_sc,
            "grid_8spk_1024ep_icc010_underpowered_ref": pwr_grid_only,
            "control_1024_worlds_pdisc010_noninferiority": pwr_ctrl_1024,
            "xarm6_135clusters_284ep_pdisc010_noninferiority": pwr_xarm_135,
        },
        "overall_gate1_synthetic_and_control_pass": bool(
            ctrl["gate1_control_pass"]
            and cf_disc["audit"]["gate1_counterfactual_pass"]
            and cf_conf["audit"]["gate1_counterfactual_pass"]
        ),
    }

    txt = json.dumps(report, indent=2)
    print(txt)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(txt + "\n")


if __name__ == "__main__":
    main()
