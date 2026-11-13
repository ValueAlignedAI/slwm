"""Geometry-Free Learned-Cost Latent-MPPI & Explicit-Goal-Latch Reactive Baseline (Wave 6 / SLWM-v2).

Eliminates every domain-specific central-wall (`x = 0`) planner heuristic identified during peer review:
  1. ZERO hard-coded wall coordinates (`|x| < 0.14`, `|y| < 0.39` removed).
  2. ZERO opposite-side checks (`x_obj * x_goal < 0.02` removed).
  3. ZERO hand-seeded `y = +/- 0.55` detour waypoints.
  4. Supports strict End-Effector-Only (`ee_only=True`) proprioception where object & obstacle states
     must be inferred purely from raw camera patches + latent world-model transitions.
  5. Includes `plan_explicit_latch_reactive_controller`: a non-world-model reactive controller with an
     explicit goal latch, isolating multi-step latent world rollouts from simple goal-latching heuristics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from models.x1b_jepa import predict_latent_step
from models.x1b_world import SLOT_PROPRIO, SLOTS_AUDIO, SLOTS_REGISTER, SLOTS_VISUAL, X1BSpec


def resolve_deprivileged_goal_and_pose(
    params: dict[str, jax.Array],
    z_curr: jax.Array,
    ext_goal_xy: jax.Array,
    ee_xy: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Decode object pose purely from visual slots and active target from external cue or register memory."""
    zf = z_curr.astype(jnp.float32)
    vis_sum = jnp.mean(zf[:, SLOTS_VISUAL, :], axis=1)
    reg_sum = jnp.mean(zf[:, SLOTS_REGISTER, :], axis=1)

    v_pose = vis_sum @ params["head_vis_pose_w"].astype(jnp.float32) + params["head_vis_pose_b"].astype(jnp.float32)
    vis_obj_xy = v_pose[:, 2:4]

    latched_xy = jnp.tanh(
        reg_sum @ params["head_goal_w"].astype(jnp.float32) + params["head_goal_b"].astype(jnp.float32)
    )
    has_ext = jnp.sum(jnp.abs(ext_goal_xy), axis=-1, keepdims=True) > 1e-4
    target_xy = jnp.where(has_ext, ext_goal_xy, latched_xy)
    return vis_obj_xy, target_xy


def _score_geometry_free_latent_rollout(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_step: jax.Array,
    z_anchor: jax.Array,
    ee0_xy: jax.Array,
    vis_obj0_xy: jax.Array,
    log_var_step: jax.Array,
    target_xy: jax.Array,
    *,
    residual: bool = True,
    trust_proprio_only: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Purely learned rollout objective: Euclidean target distance + learned hazard head + value critic."""
    zf = z_step.astype(jnp.float32)
    z0 = z_anchor.astype(jnp.float32)

    vis_sum = jnp.mean(zf[:, :, SLOTS_VISUAL, :], axis=2)
    vis0_sum = jnp.mean(z0[:, :, SLOTS_VISUAL, :], axis=2)
    vis_max = jnp.max(zf[:, :, SLOTS_VISUAL, :], axis=2)
    aud_sum = jnp.mean(zf[:, :, SLOTS_AUDIO, :], axis=2)
    reg_sum = jnp.mean(zf[:, :, SLOTS_REGISTER, :], axis=2)
    reg0_sum = jnp.mean(z0[:, :, SLOTS_REGISTER, :], axis=2)
    prop_lat = zf[:, :, SLOT_PROPRIO, :]
    prop0_lat = z0[:, :, SLOT_PROPRIO, :]

    head_pw = params["head_proprio_w"].astype(jnp.float32)
    if residual:
        delta_full = (0.75 * (prop_lat - prop0_lat) + 0.25 * (vis_sum - vis0_sum)) @ head_pw
        delta_trust = (0.80 * (prop_lat - prop0_lat) + 0.20 * (reg_sum - reg0_sum)) @ head_pw
        delta = jnp.where(trust_proprio_only[:, None, None], delta_trust, delta_full)
        pred_ee_xy = ee0_xy[:, None, :] + delta[:, :, 0:2]
        pred_obj_xy = vis_obj0_xy[:, None, :] + delta[:, :, 2:4]
    else:
        head_pb = params["head_proprio_b"].astype(jnp.float32)
        raw_p = (0.75 * prop_lat + 0.25 * vis_sum) @ head_pw + head_pb
        pred_ee_xy = raw_p[:, :, 0:2]
        pred_obj_xy = raw_p[:, :, 2:4]

    # Pure Euclidean goal & staging alignment without ANY geometric obstacle coordinates!
    to_goal = target_xy[:, None, :] - pred_obj_xy
    dist_goal = jnp.sqrt(jnp.sum(to_goal * to_goal, axis=-1) + 1e-6)
    push_unit = to_goal / (dist_goal[..., None] + 1e-6)
    push_anchor = pred_obj_xy - 0.16 * push_unit
    dist_reach = jnp.sqrt(jnp.sum((pred_ee_xy - push_anchor) ** 2, axis=-1) + 1e-6)

    # Learned general collision/obstacle hazard strictly from latent representations
    contact_ctx = 0.45 * vis_max + 0.25 * vis_sum + 0.15 * aud_sum + 0.15 * reg_sum
    c_logits = contact_ctx @ params["head_contact_w"].astype(jnp.float32) + params["head_contact_b"].astype(jnp.float32)
    learned_hazard = jax.nn.sigmoid(c_logits[:, :, 1]) * (~trust_proprio_only[:, None]).astype(jnp.float32)

    unc_pen = 0.12 * jnp.mean(jnp.exp(0.5 * jnp.squeeze(log_var_step, axis=-1)), axis=-1)

    rel_vec = jnp.concatenate([to_goal, pred_obj_xy - pred_ee_xy], axis=-1)
    pol_ctx = jnp.where(
        trust_proprio_only[:, None, None],
        jnp.concatenate([jnp.zeros_like(vis_sum), prop_lat, rel_vec], axis=-1),
        jnp.concatenate([0.65 * vis_sum + 0.35 * reg_sum, jnp.zeros_like(prop_lat), rel_vec], axis=-1),
    )
    val_boot = jnp.squeeze(
        jax.nn.silu(pol_ctx @ params["head_value_w1"].astype(jnp.float32))
        @ params["head_value_w2"].astype(jnp.float32),
        axis=-1,
    )

    step_reward = -1.90 * dist_goal - 0.35 * dist_reach - 0.70 * learned_hazard - unc_pen
    return step_reward, val_boot


def plan_geometry_free_latent_mppi(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_curr: jax.Array,
    ext_goal_xy: jax.Array,
    ee_xy: jax.Array,
    rng_key: jax.Array,
    *,
    horizon: int = 8,
    num_candidates: int = 64,
    num_elites: int = 10,
    cem_iters: int = 3,
    residual: bool = True,
    trust_proprio_only: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Zero-prior Geometry-Free Latent-MPPI planner under End-Effector-Only proprioception."""
    b = z_curr.shape[0]
    if trust_proprio_only is None:
        trust_proprio_only = jnp.zeros((b,), dtype=bool)

    vis_obj0_xy, target_xy = resolve_deprivileged_goal_and_pose(params, z_curr, ext_goal_xy, ee_xy)
    zf = z_curr.astype(jnp.float32)
    vis_sum = jnp.mean(zf[:, SLOTS_VISUAL, :], axis=1)
    reg_sum = jnp.mean(zf[:, SLOTS_REGISTER, :], axis=1)
    prop_lat = zf[:, SLOT_PROPRIO, :]

    rel0 = jnp.concatenate([target_xy - vis_obj0_xy, vis_obj0_xy - ee_xy], axis=-1)
    p_ctx0 = jnp.where(
        trust_proprio_only[:, None],
        jnp.concatenate([jnp.zeros_like(vis_sum), prop_lat, rel0], axis=-1),
        jnp.concatenate([0.65 * vis_sum + 0.35 * reg_sum, jnp.zeros_like(prop_lat), rel0], axis=-1),
    )
    prior_act = jnp.tanh(
        jax.nn.silu(p_ctx0 @ params["head_action_w1"].astype(jnp.float32))
        @ params["head_action_w2"].astype(jnp.float32)
    )

    # Isotropic zero-mean + learned policy-prior initialization (ZERO geometric wall/detour priors!)
    mean = jnp.broadcast_to(prior_act[:, None, :], (b, horizon, spec.action_dim))
    std = jnp.full((b, horizon, spec.action_dim), 0.65, dtype=jnp.float32)

    z_init = jnp.broadcast_to(z_curr[:, None, :, :], (b, num_candidates, spec.slots_per_tick, spec.d_model))
    best_ret = jnp.full((b,), -1e9, dtype=jnp.float32)
    best_u0 = prior_act

    for it in range(cem_iters):
        rng_key, subk = jax.random.split(rng_key)
        eps = jax.random.normal(subk, (b, num_candidates - 1, horizon, spec.action_dim), dtype=jnp.float32)
        sampled = jnp.clip(mean[:, None, :, :] + std[:, None, :, :] * eps, -1.0, 1.0)
        act_seq = jnp.concatenate([mean[:, None, :, :], sampled], axis=1)

        def rollout_step(carry: tuple[jax.Array, jax.Array], step_inp: tuple[jax.Array, jax.Array]):
            z_k, ret_acc = carry
            u_k, t_idx = step_inp
            h_idx = jnp.minimum(t_idx, 2)
            z_next, lv_next = predict_latent_step(params, spec, z_k, u_k, horizon_idx=h_idx, residual=residual)
            r_k, v_k = _score_geometry_free_latent_rollout(
                params,
                spec,
                z_next,
                z_init,
                ee_xy,
                vis_obj0_xy,
                lv_next,
                target_xy,
                residual=residual,
                trust_proprio_only=trust_proprio_only,
            )
            disc = 0.94**t_idx.astype(jnp.float32)
            smooth_pen = 0.02 * jnp.sum(u_k * u_k, axis=-1)
            ret_next = ret_acc + disc * (r_k - smooth_pen)
            return (z_next, ret_next), v_k

        (_, cum_ret), val_seq = jax.lax.scan(
            rollout_step,
            (z_init, jnp.zeros((b, num_candidates), dtype=jnp.float32)),
            (jnp.moveaxis(act_seq, 2, 0), jnp.arange(horizon, dtype=jnp.int32)),
        )
        total_ret = cum_ret + (0.94**horizon) * 0.25 * val_seq[-1]

        elite_vals, elite_idx = jax.lax.top_k(total_ret, num_elites)
        b_idx = jnp.arange(b)[:, None]
        elite_acts = act_seq[b_idx, elite_idx]

        # Sharp Boltzmann MPPI weights prevent multimodal trajectory cancellation without geometric assumptions
        w_logits = (elite_vals - jnp.max(elite_vals, axis=-1, keepdims=True)) / 0.06
        weights = jax.nn.softmax(w_logits, axis=-1)[:, :, None, None]
        new_mean = jnp.sum(weights * elite_acts, axis=1)
        diff = elite_acts - new_mean[:, None, :, :]
        new_std = jnp.sqrt(jnp.sum(weights * (diff * diff), axis=1) + 1e-4)

        alpha_m = 0.80 if it < cem_iters - 1 else 1.0
        mean = alpha_m * new_mean + (1.0 - alpha_m) * mean
        std = jnp.clip(new_std, 0.08, 0.70)

        top1_ret = elite_vals[:, 0]
        top1_u0 = elite_acts[:, 0, 0, :]
        better = top1_ret > best_ret
        best_ret = jnp.where(better, top1_ret, best_ret)
        best_u0 = jnp.where(better[:, None], top1_u0, best_u0)

    return best_u0, {"best_return": best_ret, "decoded_obj_xy": vis_obj0_xy, "resolved_target_xy": target_xy}


def plan_explicit_latch_reactive_controller(
    ee_xy: jax.Array,
    decoded_obj_xy: jax.Array,
    ext_goal_xy: jax.Array,
    latched_goal_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reviewer-requested non-world-model baseline: explicit goal latch + direct reactive push controller.

    Stores `ext_goal_xy` verbatim in `latched_goal_state` whenever `||ext_goal_xy|| > 0` (perfect external
    memory latch, zero forgetting) and issues a reactive proportional push command without multi-step rollouts.
    """
    has_ext = jnp.sum(jnp.abs(ext_goal_xy), axis=-1, keepdims=True) > 1e-4
    new_latch = jnp.where(has_ext, ext_goal_xy, latched_goal_state)

    to_goal = new_latch - decoded_obj_xy
    dist_g = jnp.sqrt(jnp.sum(to_goal * to_goal, axis=-1, keepdims=True) + 1e-6)
    u_dir = to_goal / (dist_g + 1e-6)
    rel_ee = ee_xy - decoded_obj_xy
    proj = jnp.sum(rel_ee * u_dir, axis=-1, keepdims=True)
    perp = jnp.stack([-u_dir[..., 1], u_dir[..., 0]], axis=-1)
    lat = jnp.abs(jnp.sum(rel_ee * perp, axis=-1, keepdims=True))
    aligned = (proj <= -0.09) & (lat < 0.10)
    waypoint = jnp.where(aligned, decoded_obj_xy + 0.08 * u_dir, decoded_obj_xy - 0.18 * u_dir)

    u_xy = jnp.clip(3.2 * (waypoint - ee_xy), -1.0, 1.0)
    u_full = jnp.concatenate([u_xy, jnp.zeros_like(u_xy)], axis=-1)
    return u_full, new_latch
