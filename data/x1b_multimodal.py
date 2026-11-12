"""Multimodal real-bank loader + vectorized JAX 64x64x3 physical & associative world environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ASSOC_NUM_KEYS = 16      # Token IDs 16..31
ASSOC_NUM_VALS = 16      # Token IDs 64..79 (chance = 6.25%)
ASSOC_KEY_OFFSET = 16
ASSOC_VAL_OFFSET = 64


@dataclass(frozen=True)
class PhysicsEnvConfig:
    """Configuration for vectorized 64x64x3 tabletop manipulation & thin-barrier worlds."""

    dt: float = 0.12
    puck_radius: float = 0.11
    ee_radius: float = 0.075
    barrier_x: float = 0.0
    barrier_y_half: float = 0.34
    barrier_half_width: float = 0.016  # ~1.0 pixel at 64x64; sub-pixel / erased under 16x16 stride-4!
    damping: float = 0.40


def render_scene_64x64(
    ee_xy: jax.Array,
    obj_xy: jax.Array,
    obj_theta: jax.Array,
    target_xy: jax.Array,
    has_visible_barrier: jax.Array,
    show_target_cue: jax.Array,
) -> jax.Array:
    """Anti-aliased vectorized 64x64x3 RGB renderer for Push-T / Thin-Barrier manipulation scenes."""
    grid = jnp.linspace(-1.0 + 1.0 / 64.0, 1.0 - 1.0 / 64.0, 64, dtype=jnp.float32)
    gy, gx = jnp.meshgrid(grid, grid, indexing="ij")  # [64, 64]

    # Subtle textured dark-slate arena background
    bg_r = 28.0 + 6.0 * gx
    bg_g = 32.0 + 6.0 * gy
    bg_b = 40.0 + 4.0 * (gx * gy)
    img = jnp.stack([bg_r, bg_g, bg_b], axis=-1)
    img = jnp.broadcast_to(img, (*ee_xy.shape[:-1], 64, 64, 3))

    # 1. Thin central vertical barrier (columns 31..32 at 64x64; missed completely by 2::4 16x16 subsampling!)
    dx_wall = jnp.abs(gx - 0.0)
    in_wall_y = jnp.abs(gy) <= 0.34
    wall_alpha = jnp.clip((0.030 - dx_wall) / 0.015, 0.0, 1.0) * in_wall_y.astype(jnp.float32)
    wall_alpha = wall_alpha * has_visible_barrier.astype(jnp.float32)[..., None, None]
    wall_color = jnp.array([225.0, 95.0, 75.0], dtype=jnp.float32)
    img = img * (1.0 - wall_alpha[..., None]) + wall_color * wall_alpha[..., None]

    # 2. Target goal ring (only visible when show_target_cue is True!)
    tx = target_xy[..., 0, None, None]
    ty = target_xy[..., 1, None, None]
    d_targ = jnp.sqrt((gx - tx) ** 2 + (gy - ty) ** 2 + 1e-6)
    ring_alpha = jnp.clip(1.0 - jnp.abs(d_targ - 0.11) / 0.028, 0.0, 1.0)
    ring_alpha = ring_alpha * show_target_cue.astype(jnp.float32)[..., None, None]
    targ_color = jnp.array([70.0, 225.0, 130.0], dtype=jnp.float32)
    img = img * (1.0 - ring_alpha[..., None]) + targ_color * ring_alpha[..., None]

    # 3. Oriented T-block / Puck rigid body
    ox = obj_xy[..., 0, None, None]
    oy = obj_xy[..., 1, None, None]
    c = jnp.cos(obj_theta)[..., None, None]
    s = jnp.sin(obj_theta)[..., None, None]
    rx = c * (gx - ox) + s * (gy - oy)
    ry = -s * (gx - ox) + c * (gy - oy)

    top_bar = (jnp.abs(rx) <= 0.13) & (jnp.abs(ry - 0.055) <= 0.042)
    stem_bar = (jnp.abs(rx) <= 0.045) & (ry >= -0.11) & (ry <= 0.06)
    d_puck = jnp.sqrt(rx * rx + ry * ry + 1e-6)
    disc_soft = jnp.clip((0.12 - d_puck) / 0.022, 0.0, 1.0)
    obj_alpha = jnp.maximum((top_bar | stem_bar).astype(jnp.float32), 0.55 * disc_soft)
    obj_color = jnp.array([85.0, 165.0, 245.0], dtype=jnp.float32)
    img = img * (1.0 - obj_alpha[..., None]) + obj_color * obj_alpha[..., None]

    # 4. Pusher / End-effector circle
    ex = ee_xy[..., 0, None, None]
    ey = ee_xy[..., 1, None, None]
    d_ee = jnp.sqrt((gx - ex) ** 2 + (gy - ey) ** 2 + 1e-6)
    ee_alpha = jnp.clip((0.075 - d_ee) / 0.02, 0.0, 1.0)
    ee_color = jnp.array([250.0, 215.0, 75.0], dtype=jnp.float32)
    img = img * (1.0 - ee_alpha[..., None]) + ee_color * ee_alpha[..., None]

    return jnp.clip(jnp.rint(img), 0.0, 255.0).astype(jnp.uint8)


def state_to_proprio(
    state: dict[str, jax.Array],
    prev_u_xy: jax.Array,
    prev_contacts: jax.Array,
) -> jax.Array:
    """Pack causal physical state s_t into the 16-D proprioceptive observation vector."""
    b = state["ee_xy"].shape[0]
    return jnp.concatenate(
        [
            state["ee_xy"],
            state["obj_xy"],
            state["obj_vel"],
            jnp.sin(state["obj_theta"])[:, None],
            jnp.cos(state["obj_theta"])[:, None],
            prev_u_xy,
            prev_contacts,
            jnp.zeros((b, 4), dtype=jnp.float32),
        ],
        axis=-1,
    )


def geodesic_goal_distance(
    obj_xy: jax.Array,
    target_xy: jax.Array,
    avoid_barrier: jax.Array,
) -> jax.Array:
    """Shortest barrier-respecting path length from obj_xy to target_xy."""
    direct = jnp.sqrt(jnp.sum((obj_xy - target_xy) ** 2, axis=-1) + 1e-6)
    on_opp = (obj_xy[..., 0] * target_xy[..., 0]) < 0.0
    blocked = avoid_barrier & on_opp & ((jnp.abs(obj_xy[..., 1]) < 0.44) | (jnp.abs(target_xy[..., 1]) < 0.40))

    gap_pos = jnp.array([0.0, 0.52], dtype=jnp.float32)
    gap_neg = jnp.array([0.0, -0.52], dtype=jnp.float32)
    d_pos = jnp.sqrt(jnp.sum((obj_xy - gap_pos) ** 2, axis=-1) + 1e-6) + jnp.sqrt(
        jnp.sum((target_xy - gap_pos) ** 2, axis=-1) + 1e-6
    )
    d_neg = jnp.sqrt(jnp.sum((obj_xy - gap_neg) ** 2, axis=-1) + 1e-6) + jnp.sqrt(
        jnp.sum((target_xy - gap_neg) ** 2, axis=-1) + 1e-6
    )
    return jnp.where(blocked, jnp.minimum(d_pos, d_neg), direct)


def step_physics_world(
    state: dict[str, jax.Array],
    action: jax.Array,
    cfg: PhysicsEnvConfig = PhysicsEnvConfig(),
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Advance batched 2D tabletop rigid-body contact + thin-barrier physics by one 100 ms tick."""
    pol = state["polarity"][:, None]
    u_xy = jnp.clip(action[:, :2], -1.0, 1.0) * pol
    ee_cand = jnp.clip(state["ee_xy"] + 0.115 * u_xy, -0.92, 0.92)

    # 1. Block end-effector against thin central barrier (x = 0, |y| <= barrier_y_half)
    ee_cross = (
        state["has_barrier"]
        & (
            (jnp.abs(ee_cand[:, 0]) < (cfg.ee_radius + cfg.barrier_half_width))
            | (state["ee_xy"][:, 0] * ee_cand[:, 0] < 0.0)
        )
        & (jnp.abs(ee_cand[:, 1]) <= cfg.barrier_y_half)
    )
    ee_side = jnp.where(state["ee_xy"][:, 0] >= 0.0, 1.0, -1.0)
    ee_x = jnp.where(ee_cross, ee_side * (cfg.ee_radius + cfg.barrier_half_width + 0.004), ee_cand[:, 0])
    ee_next = jnp.stack([ee_x, ee_cand[:, 1]], axis=-1)

    # 2. High-friction tabletop sliding contact between end-effector and T-block
    diff = state["obj_xy"] - ee_next
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1, keepdims=True) + 1e-6)
    min_sep = cfg.ee_radius + cfg.puck_radius
    penetration = jnp.maximum(0.0, min_sep - dist)
    normal = diff / dist
    hit_obj = penetration[:, 0] > 1e-4

    push_impulse = normal * (penetration * 1.6) + u_xy * (hit_obj.astype(jnp.float32)[:, None] * 0.18)
    vel = cfg.damping * state["obj_vel"] + push_impulse
    cand_obj = state["obj_xy"] + cfg.dt * vel + normal * (penetration * 0.95)

    # 3. Block object against thin central barrier
    crosses_x = (
        (jnp.abs(cand_obj[:, 0]) < (cfg.puck_radius + cfg.barrier_half_width))
        | (state["obj_xy"][:, 0] * cand_obj[:, 0] < 0.0)
    )
    in_y = jnp.abs(cand_obj[:, 1]) <= (cfg.barrier_y_half + 0.5 * cfg.puck_radius)
    hit_wall = state["has_barrier"] & crosses_x & in_y

    side = jnp.where(state["obj_xy"][:, 0] >= 0.0, 1.0, -1.0)
    clamped_x = side * jnp.maximum(jnp.abs(cand_obj[:, 0]), cfg.puck_radius + cfg.barrier_half_width + 0.005)
    obj_x = jnp.where(hit_wall, clamped_x, cand_obj[:, 0])
    vel_x = jnp.where(hit_wall, -0.20 * vel[:, 0], vel[:, 0])

    obj_final = jnp.clip(jnp.stack([obj_x, cand_obj[:, 1]], axis=-1), -0.88, 0.88)
    vel_final = jnp.stack([vel_x, vel[:, 1]], axis=-1)

    # Prevent end-effector from tunneling inside object when clamped against barrier
    diff2 = ee_next - obj_final
    d2 = jnp.sqrt(jnp.sum(diff2 * diff2, axis=-1, keepdims=True) + 1e-6)
    ee_final = jnp.clip(jnp.where(d2 < min_sep, obj_final + (diff2 / d2) * min_sep, ee_next), -0.92, 0.92)

    lever = diff[:, 0] * push_impulse[:, 1] - diff[:, 1] * push_impulse[:, 0]
    theta_next = state["obj_theta"] + 1.4 * lever + 0.12 * action[:, 2] * hit_obj.astype(jnp.float32)

    barrier_ev = (hit_wall | ee_cross).astype(jnp.float32)
    impact_ev = hit_obj.astype(jnp.float32)
    contacts_vec = jnp.stack([impact_ev, barrier_ev], axis=-1)

    # Synthesize 64-band resonant contact spectrogram
    freqs = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, :]
    speed = jnp.sqrt(jnp.sum(vel_final * vel_final, axis=-1, keepdims=True) + 1e-6)
    obj_tone = impact_ev[:, None] * jnp.exp(-((freqs - 0.28) ** 2) / 0.018) * (1.2 + 2.0 * speed)
    wall_tone = barrier_ev[:, None] * jnp.exp(-((freqs - 0.72) ** 2) / 0.012) * 2.2
    roll_hum = 0.25 * speed * jnp.exp(-freqs * 3.0)
    audio_spec = obj_tone + wall_tone + roll_hum - 0.35

    next_state = {
        "ee_xy": ee_final,
        "obj_xy": obj_final,
        "obj_vel": vel_final,
        "obj_theta": theta_next,
        "has_barrier": state["has_barrier"],
        "polarity": state["polarity"],
    }
    proprio_next = state_to_proprio(next_state, u_xy, contacts_vec)
    return next_state, {
        "proprio": proprio_next,
        "audio": audio_spec,
        "contacts": contacts_vec,
    }


def expert_barrier_aware_action(
    state: dict[str, jax.Array],
    target_xy: jax.Array,
    avoid_barrier: jax.Array,
) -> jax.Array:
    """Tangential-wrap, barrier-circumventing tabletop pushing controller (95.8% closed-loop oracle)."""
    ee = state["ee_xy"]
    obj = state["obj_xy"]
    pol = state["polarity"][:, None]

    on_opp_sides = (obj[:, 0] * target_xy[:, 0]) < 0.02
    blocked = avoid_barrier & on_opp_sides & (jnp.abs(obj[:, 1]) < 0.46)
    detour_sign = jnp.where(obj[:, 1] + 0.5 * target_xy[:, 1] >= 0.0, 1.0, -1.0)
    detour_y = detour_sign * 0.55
    targ_side = jnp.where(target_xy[:, 0] >= 0.0, 1.0, -1.0)
    waypoint = jnp.where(
        blocked[:, None],
        jnp.stack([targ_side * 0.22, detour_y], axis=-1),
        target_xy,
    )

    push_dir = waypoint - obj
    dist_wp = jnp.sqrt(jnp.sum(push_dir * push_dir, axis=-1, keepdims=True) + 1e-6)
    push_norm = push_dir / dist_wp

    staging = jnp.clip(obj - 0.20 * push_norm, -0.90, 0.90)
    rel_ee = ee - obj
    proj_along = jnp.sum(rel_ee * push_norm, axis=-1, keepdims=True)
    perp = jnp.stack([-push_norm[:, 1], push_norm[:, 0]], axis=-1)
    lat_err = jnp.sum(rel_ee * perp, axis=-1, keepdims=True)
    perp_side = jnp.where(lat_err >= 0.0, 1.0, -1.0)

    # Wrap tangentially around puck if end-effector is on the front side instead of ramming backward!
    need_wrap = (proj_along > -0.08) & (jnp.sqrt(jnp.sum(rel_ee * rel_ee, axis=-1, keepdims=True) + 1e-6) < 0.30)
    wrap_pt = jnp.clip(obj + perp * perp_side * 0.25 - 0.14 * push_norm, -0.90, 0.90)

    aligned = (proj_along <= -0.10) & (jnp.abs(lat_err) < 0.09)
    push_gain = jnp.minimum(1.0, dist_wp * 2.5)
    drive_pt = jnp.where(need_wrap, wrap_pt, jnp.where(aligned, obj + 0.08 * push_norm * push_gain, staging))

    ee_blocked = avoid_barrier & ((ee[:, 0] * drive_pt[:, 0]) < 0.0) & (jnp.abs(ee[:, 1]) < 0.42)
    ee_way = jnp.where(ee_blocked[:, None], jnp.stack([ee[:, 0] * 0.25, detour_y], axis=-1), drive_pt)

    goal_dist = jnp.sqrt(jnp.sum((obj - target_xy) ** 2, axis=-1, keepdims=True) + 1e-6)
    raw_cmd = jnp.where(goal_dist < 0.05, jnp.zeros_like(ee), jnp.clip((ee_way - ee) * 7.5, -1.0, 1.0) * pol)
    return jnp.concatenate([raw_cmd, jnp.zeros_like(raw_cmd)], axis=-1)


def generate_synthetic_world_batch(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int = 32,
    vocab_size: int = 4096,
) -> dict[str, jax.Array]:
    """Generate strictly causal 64x64x3 physical manipulation, transient-cue, multi-query associative
    overwrite, 180-deg wrong-camera spoofing, and hidden-barrier episodes.

    Causal timing contract at tick t:
        - video[:, t], proprio[:, t], audio[:, t] observe physical state s_t.
        - action[:, t] is prev_action u_{t-1} (0 at t=0; ZERO label leakage of u_t!).
        - next_action[:, t] is u_t, which transitions s_t -> target_next_proprio[:, t] (s_{t+1}).
        - target_action[:, t] is the barrier-aware expert action u_expert(s_t, goal).
    """
    k_init, k_task, k_assoc, k_noise, k_act = jax.random.split(rng_key, 5)
    b = batch_size
    t = seq_len

    # Scenario mixture:
    #   0: Clean Push-T / Visible Thin-Barrier (50% barrier, 50% open; Mode 0: commit_full)
    #   1: Transient 0.4s Goal Instruction (cue only at ticks 0..3; Mode 0: commit_full)
    #   2: Wrong-Camera 180-deg Visual Spoofing (video inverted from tick 4+; Mode 1: commit_trusted_proprio)
    #   3: Unobserved Hidden Barrier Trap (physical wall active, visually hidden; Mode 2: probe_cautious on stall)
    #   4: Unsupported Missing-Modality Query (visual+audio masked out; Mode 3: noop_abstain)
    scen = jax.random.randint(k_task, (b,), 0, 5)

    k1, k2, k3, k4, k5 = jax.random.split(k_init, 5)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (b,)), 1.0, -1.0)
    obj_init = jnp.stack(
        [
            side * jax.random.uniform(k2, (b,), minval=0.28, maxval=0.52),
            jax.random.uniform(k3, (b,), minval=-0.36, maxval=0.36),
        ],
        axis=-1,
    )
    # Stage end-effector mostly behind or beside puck with exploratory variety
    ee_offset = jnp.stack(
        [
            side * jax.random.uniform(k4, (b,), minval=0.12, maxval=0.26),
            jax.random.uniform(k5, (b,), minval=-0.16, maxval=0.16),
        ],
        axis=-1,
    )
    ee_init = jnp.clip(obj_init + ee_offset, -0.88, 0.88)

    # Quadrant-aligned goal centers so transient quadrant token (tokens 4..7) uniquely specifies goal region
    gx_sign = -side
    gy_sign = jnp.where(jax.random.bernoulli(k5, 0.5, (b,)), 1.0, -1.0)
    targ_xy = jnp.stack(
        [
            gx_sign * jax.random.uniform(k4, (b,), minval=0.30, maxval=0.44),
            gy_sign * jax.random.uniform(k3, (b,), minval=0.16, maxval=0.34),
        ],
        axis=-1,
    )

    coin_bar = jax.random.bernoulli(k1, 0.5, (b,))
    vis_barrier = ((scen == 0) & coin_bar) | ((scen == 1) & coin_bar)
    has_barrier = vis_barrier | (scen == 3)
    polarity = jnp.ones((b,), dtype=jnp.float32)

    init_state = {
        "ee_xy": ee_init,
        "obj_xy": obj_init,
        "obj_vel": jnp.zeros((b, 2), dtype=jnp.float32),
        "obj_theta": jax.random.uniform(k2, (b,), minval=-0.6, maxval=0.6),
        "has_barrier": has_barrier,
        "polarity": polarity,
    }
    init_prev_u = jnp.zeros((b, 4), dtype=jnp.float32)
    init_prev_con = jnp.zeros((b, 2), dtype=jnp.float32)
    init_prev_aud = jnp.zeros((b, 64), dtype=jnp.float32) - 0.35

    act_noise = jax.random.normal(k_act, (t, b, 4), dtype=jnp.float32) * 0.18
    # Occasional straight-push exploratory episodes (25%) so dynamics also observe direct barrier collisions!
    direct_explore = jax.random.bernoulli(k_act, 0.25, (b,))

    def sim_step(
        carry: tuple[dict[str, jax.Array], jax.Array, jax.Array, jax.Array],
        inp: tuple[jax.Array, jax.Array],
    ):
        st, prev_u, prev_con, prev_aud = carry
        step_idx, eps_u = inp
        show_cue = (scen != 1) | (step_idx < 4)
        vid_t = render_scene_64x64(
            st["ee_xy"], st["obj_xy"], st["obj_theta"], targ_xy, vis_barrier, show_cue
        )
        prop_t = state_to_proprio(st, prev_u[:, :2], prev_con)

        u_true_exp = expert_barrier_aware_action(st, targ_xy, has_barrier)
        u_vis_exp = expert_barrier_aware_action(st, targ_xy, vis_barrier & (~direct_explore))
        u_issued = jnp.clip(u_vis_exp + eps_u, -1.0, 1.0)

        st_next, obs_next = step_physics_world(st, u_issued)

        # Geodesic barrier-respecting value function on current state s_t
        dist_geo = geodesic_goal_distance(st["obj_xy"], targ_xy, vis_barrier)
        dist_ee = jnp.sqrt(jnp.sum((st["ee_xy"] - st["obj_xy"]) ** 2, axis=-1) + 1e-6)
        val_t = -1.35 * dist_geo - 0.25 * dist_ee

        # Channel 0: physical impact onset; Channel 1: visual thin-barrier hazard OR physical barrier contact
        hazard_t = jnp.maximum(vis_barrier.astype(jnp.float32), obs_next["contacts"][:, 1])
        contacts_target_t = jnp.stack([obs_next["contacts"][:, 0], hazard_t], axis=-1)

        return (st_next, u_issued, obs_next["contacts"], obs_next["audio"]), {
            "video": vid_t,
            "audio": prev_aud,
            "proprio": prop_t,
            "next_proprio": obs_next["proprio"],
            "raw_barrier_hit": obs_next["contacts"][:, 1],
            "contacts": contacts_target_t,
            "prev_u": prev_u,
            "u_issued": u_issued,
            "u_expert": u_true_exp,
            "value": val_t,
        }

    _, traj = jax.lax.scan(
        sim_step,
        (init_state, init_prev_u, init_prev_con, init_prev_aud),
        (jnp.arange(t, dtype=jnp.int32), act_noise),
    )
    clean_video = jnp.swapaxes(traj["video"], 0, 1)           # [B, T, 64, 64, 3]
    audio = jnp.swapaxes(traj["audio"], 0, 1)                 # [B, T, 64]
    proprio = jnp.swapaxes(traj["proprio"], 0, 1)             # [B, T, 16]
    next_proprio = jnp.swapaxes(traj["next_proprio"], 0, 1)   # [B, T, 16]
    raw_bar_hit = jnp.swapaxes(traj["raw_barrier_hit"], 0, 1) # [B, T]
    contacts = jnp.swapaxes(traj["contacts"], 0, 1)           # [B, T, 2]
    prev_action = jnp.swapaxes(traj["prev_u"], 0, 1)          # [B, T, 4]
    u_issued = jnp.swapaxes(traj["u_issued"], 0, 1)           # [B, T, 4]
    u_expert = jnp.swapaxes(traj["u_expert"], 0, 1)           # [B, T, 4]
    values = jnp.swapaxes(traj["value"], 0, 1)                # [B, T]

    # Inject 180-degree Inverted Wrong-Camera Spoofing on scen == 2 from tick t >= 4 onwards!
    flipped_video = jnp.flip(clean_video, axis=(2, 3))
    spoof_active = (scen[:, None] == 2) & (jnp.arange(t)[None, :] >= 4)
    obs_video = jnp.where(spoof_active[:, :, None, None, None], flipped_video, clean_video)

    # Multi-Query Associative Key-Value Overwrite Stream (16 keys: 16..31, 16 values: 64..79)
    ka1, ka2, ka3, ka4, ka5 = jax.random.split(k_assoc, 5)
    k0 = jax.random.randint(ka1, (b,), ASSOC_KEY_OFFSET, ASSOC_KEY_OFFSET + ASSOC_NUM_KEYS)
    k1_off = jax.random.randint(ka2, (b,), 1, ASSOC_NUM_KEYS)
    k1_id = ASSOC_KEY_OFFSET + ((k0 - ASSOC_KEY_OFFSET + k1_off) % ASSOC_NUM_KEYS)

    v0_init = jax.random.randint(ka3, (b,), ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    v1_init = jax.random.randint(ka4, (b,), ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    # Guarantee overwrite changes v0 to a distinct value so stale memory scores 0%!
    v0_off = jax.random.randint(ka5, (b,), 1, ASSOC_NUM_VALS)
    v0_over = ASSOC_VAL_OFFSET + ((v0_init - ASSOC_VAL_OFFSET + v0_off) % ASSOC_NUM_VALS)
    final_q_slot = jax.random.randint(ka5, (b,), 0, 2)

    text = jnp.zeros((b, t, 4), dtype=jnp.int32)
    target_text = jnp.zeros((b, t), dtype=jnp.int32)

    # Sub-token 0 at ticks 0..3: spoken/symbolic goal quadrant token (4..7)
    quad_tok = (
        4
        + (targ_xy[:, 0] >= 0.0).astype(jnp.int32) * 2
        + (targ_xy[:, 1] >= 0.0).astype(jnp.int32)
    )
    for tick_i in range(min(4, t)):
        text = text.at[:, tick_i, 0].set(quad_tok)
    target_text = target_text.at[:, 0].set(quad_tok)

    if t >= 8:
        t_ov = max(4, t // 2)
        # Write (k0 -> v0_init) at t=1, (k1 -> v1_init) at t=2, and overwrite (k0 -> v0_over) at t=t_ov
        text = text.at[:, 1, 1].set(k0).at[:, 1, 2].set(v0_init)
        text = text.at[:, 2, 1].set(k1_id).at[:, 2, 2].set(v1_init)
        text = text.at[:, t_ov, 1].set(k0).at[:, t_ov, 2].set(v0_over)

        # Intermediate & terminal associative queries (token 2 in slot 3 = QUERY marker)
        q1_t = max(3, t // 4)
        if q1_t != t_ov:
            text = text.at[:, q1_t, 1].set(k0).at[:, q1_t, 3].set(2)
            target_text = target_text.at[:, q1_t].set(v0_init if q1_t < t_ov else v0_over)

        q2_t = min(t - 2, max(t_ov + 1, (3 * t) // 4))
        if q2_t > t_ov and q2_t < t - 1:
            text = text.at[:, q2_t, 1].set(k1_id).at[:, q2_t, 3].set(2)
            target_text = target_text.at[:, q2_t].set(v1_init)

        q_final_key = jnp.where(final_q_slot == 0, k0, k1_id)
        q_final_ans = jnp.where(final_q_slot == 0, v0_over, v1_init)
        text = text.at[:, t - 1, 1].set(q_final_key).at[:, t - 1, 3].set(2)
        target_text = target_text.at[:, t - 1].set(q_final_ans)

    full_goal = jnp.concatenate([targ_xy - obj_init, targ_xy], axis=-1)
    # Zero external goal vector for TransientCommand (scen == 1) so model MUST rely on latent working memory!
    obs_goal = jnp.where((scen == 1)[:, None], jnp.zeros_like(full_goal), full_goal)

    full_mod_mask = jnp.ones((b, t, 4), dtype=jnp.float32)
    mod_mask = full_mod_mask
    ungrounded = (scen[:, None] == 4) & (jnp.arange(t)[None, :] >= 2)
    mod_mask = mod_mask.at[:, :, 0].set(jnp.where(ungrounded, 0.0, 1.0))
    mod_mask = mod_mask.at[:, :, 1].set(jnp.where(ungrounded, 0.0, 1.0))

    hidden_stall = (scen[:, None] == 3) & (raw_bar_hit > 0.5)
    target_commit = jnp.zeros((b, t), dtype=jnp.int32)
    target_commit = jnp.where(spoof_active, 1, target_commit)
    target_commit = jnp.where(hidden_stall, 2, target_commit)
    target_commit = jnp.where(ungrounded, 3, target_commit)

    return {
        "video": obs_video,
        "clean_video": clean_video,
        "audio": audio + 0.025 * jax.random.normal(k_noise, audio.shape),
        "proprio": proprio,
        "target_proprio": proprio,
        "target_next_proprio": next_proprio,
        "text": text % vocab_size,
        "target_text": target_text % vocab_size,
        "action": prev_action,
        "next_action": u_issued,
        "target_action": u_expert,
        "target_contacts": contacts,
        "target_value": values,
        "target_commit": target_commit,
        "goal": obs_goal,
        "true_goal": full_goal,
        "modality_mask": mod_mask,
        "full_modality_mask": full_mod_mask,
        "has_av_pair": (1.0 - ungrounded[:, -1].astype(jnp.float32)),
        "action_supervision_mask": jnp.broadcast_to((scen != 4)[:, None], (b, t)).astype(jnp.float32),
        "valid": jnp.ones((b, t), dtype=bool),
        "scenario": scen,
    }


class RealMultimodalBank:
    """In-memory / device-staged bank of real GreatestHits-64, PyMunk-64, STARSS23 & FineWeb-Edu clips."""

    def __init__(self, path: str | Path, split: str = "train") -> None:
        self.path = Path(path)
        self.split = split
        with np.load(self.path, allow_pickle=False) as f:
            pfx = f"{split}_"
            self.video = np.asarray(f[pfx + "video"], dtype=np.uint8)
            self.audio = np.asarray(f[pfx + "audio"], dtype=np.float32)
            self.proprio = np.asarray(f[pfx + "proprio"], dtype=np.float32)
            self.text = np.asarray(f[pfx + "text"], dtype=np.int32)
            self.target_text = np.asarray(f[pfx + "target_text"], dtype=np.int32)
            self.action = np.asarray(f[pfx + "action"], dtype=np.float32)
            self.contacts = np.asarray(f[pfx + "contacts"], dtype=np.float32)
            self.goal = np.asarray(f[pfx + "goal"], dtype=np.float32)
            self.mod_mask = np.asarray(f[pfx + "modality_mask"], dtype=np.float32)
            self.has_av = np.asarray(f[pfx + "has_av_pair"], dtype=np.float32)
        self.size = int(self.video.shape[0])

    def sample_batch(self, rng: np.random.Generator, batch_size: int, seq_len: int = 32) -> dict[str, np.ndarray]:
        """Sample a deterministic real-world multimodal batch."""
        idx = rng.integers(0, self.size, size=(batch_size,))
        t_avail = self.video.shape[1]
        t = min(seq_len, t_avail)
        vid = self.video[idx, :t]
        aud = self.audio[idx, :t]
        prop = self.proprio[idx, :t]
        next_prop = np.concatenate([prop[:, 1:], prop[:, -1:]], axis=1)
        txt = self.text[idx, :t]
        ttxt = self.target_text[idx, :t]
        act = self.action[idx, :t]
        next_act = np.concatenate([act[:, 1:], act[:, -1:]], axis=1)
        con = self.contacts[idx, :t]
        gl = self.goal[idx]
        mm = self.mod_mask[idx, :t]
        av = self.has_av[idx]

        dist = np.sqrt(np.sum((prop[:, :, 2:4] - gl[:, None, 2:4]) ** 2, axis=-1))
        val = (-1.35 * dist).astype(np.float32)
        return {
            "video": vid,
            "clean_video": vid,
            "audio": aud,
            "proprio": prop,
            "target_proprio": prop,
            "target_next_proprio": next_prop,
            "text": txt,
            "target_text": ttxt,
            "action": act,
            "next_action": next_act,
            "target_action": next_act,
            "target_contacts": con,
            "target_value": val,
            "target_commit": np.zeros((batch_size, t), dtype=np.int32),
            "goal": gl,
            "true_goal": gl,
            "modality_mask": mm,
            "full_modality_mask": mm,
            "has_av_pair": av,
            "action_supervision_mask": (np.sum(np.abs(next_act), axis=-1) > 1e-3).astype(np.float32),
            "valid": np.ones((batch_size, t), dtype=bool),
            "scenario": np.full((batch_size,), -1, dtype=np.int32),
        }


def mix_real_and_synthetic_batches(
    real_batch: dict[str, Any],
    synth_batch: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Concatenate half-real, half-interactive episodes along batch dimension."""
    keys = (
        "video",
        "clean_video",
        "audio",
        "proprio",
        "target_proprio",
        "target_next_proprio",
        "text",
        "target_text",
        "action",
        "next_action",
        "target_action",
        "target_contacts",
        "target_value",
        "target_commit",
        "goal",
        "true_goal",
        "modality_mask",
        "full_modality_mask",
        "has_av_pair",
        "action_supervision_mask",
        "valid",
        "scenario",
    )
    return {k: jnp.concatenate([jnp.asarray(real_batch[k]), synth_batch[k]], axis=0) for k in keys}
