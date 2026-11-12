"""Multi-Horizon Latent Cross-Entropy Method MPC (Latent-CEM-MPC) & Baselines for SLWM-X1B."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from data.x1b_multimodal import geodesic_goal_distance
from models.x1b_jepa import _ego_push_kinematics, predict_latent_step
from models.x1b_world import SLOT_PROPRIO, SLOTS_AUDIO, SLOTS_REGISTER, SLOTS_VISUAL, X1BSpec


def _resolve_mpc_goal_and_barrier(
    params: dict[str, jax.Array],
    z_curr: jax.Array,
    goal: jax.Array,
    obs_proprio: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolve active goal (external or working-memory latent goal latch) + observed 64x64 barrier probability."""
    zf = z_curr.astype(jnp.float32)
    vis_sum = jnp.mean(zf[:, SLOTS_VISUAL, :], axis=1)
    vis_max = jnp.max(zf[:, SLOTS_VISUAL, :], axis=1)
    aud_sum = jnp.mean(zf[:, SLOTS_AUDIO, :], axis=1)
    reg_sum = jnp.mean(zf[:, SLOTS_REGISTER, :], axis=1)

    head_vw = params["head_vis_pose_w"].astype(jnp.float32)
    head_vb = params["head_vis_pose_b"].astype(jnp.float32)
    v_dec0 = vis_sum @ head_vw + head_vb

    pred_goal_xy = jnp.tanh(
        reg_sum @ params["head_goal_w"].astype(jnp.float32)
        + params["head_goal_b"].astype(jnp.float32)
    )
    gf = goal.astype(jnp.float32)
    has_ext = jnp.sum(jnp.abs(gf), axis=-1, keepdims=True) > 1e-4
    ref_obj = obs_proprio.astype(jnp.float32)[:, 2:4] if obs_proprio is not None else v_dec0[:, 2:4]
    latched_goal = jnp.concatenate([pred_goal_xy - ref_obj, pred_goal_xy], axis=-1)
    eff_goal = jnp.where(has_ext, gf, latched_goal)

    c_ctx0 = 0.45 * vis_max + 0.25 * vis_sum + 0.15 * aud_sum + 0.15 * reg_sum
    c_logits0 = (
        c_ctx0 @ params["head_contact_w"].astype(jnp.float32)
        + params["head_contact_b"].astype(jnp.float32)
    )
    bar0_prob = jax.nn.sigmoid(c_logits0[:, 1])
    return eff_goal, bar0_prob, v_dec0


def _score_latent_state(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_step: jax.Array,
    z_anchor: jax.Array,
    obs_proprio: jax.Array | None,
    log_var_step: jax.Array,
    goal_xy: jax.Array,
    bar0_prob: jax.Array,
    *,
    residual: bool = True,
    trust_proprio_only: jax.Array,
    avoid_barrier: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate barrier-aware geodesic cost, contact hazard, and bootstrapped value on rolled-out latent Z."""
    zf = z_step.astype(jnp.float32)
    z0 = z_anchor.astype(jnp.float32)

    vis_sum = jnp.mean(zf[:, :, SLOTS_VISUAL, :], axis=2)
    vis0_sum = jnp.mean(z0[:, :, SLOTS_VISUAL, :], axis=2)
    aud_sum = jnp.mean(zf[:, :, SLOTS_AUDIO, :], axis=2)
    reg_sum = jnp.mean(zf[:, :, SLOTS_REGISTER, :], axis=2)
    reg0_sum = jnp.mean(z0[:, :, SLOTS_REGISTER, :], axis=2)
    prop_lat = zf[:, :, SLOT_PROPRIO, :]
    prop0_lat = z0[:, :, SLOT_PROPRIO, :]

    head_pw = params["head_proprio_w"].astype(jnp.float32)
    head_pb = params["head_proprio_b"].astype(jnp.float32)
    use_euler = (obs_proprio is not None) and residual and (spec.arch != "legacy_scalar_hybrid")
    if use_euler:
        base_p = obs_proprio.astype(jnp.float32)[:, None, :]
        prop_full = base_p + ((0.75 * (prop_lat - prop0_lat) + 0.25 * (vis_sum - vis0_sum)) @ head_pw)
        prop_trust = base_p + ((0.80 * (prop_lat - prop0_lat) + 0.20 * (reg_sum - reg0_sum)) @ head_pw)
    else:
        prop_full = (0.75 * prop_lat + 0.25 * vis_sum) @ head_pw + head_pb
        prop_trust = (0.80 * prop_lat + 0.20 * reg_sum) @ head_pw + head_pb
    pred_prop = jnp.where(trust_proprio_only[:, None, None], prop_trust, prop_full)

    obj_xy = pred_prop[:, :, 2:4]
    ee_xy = pred_prop[:, :, 0:2]
    target_xy = goal_xy[:, None, 2:4]

    # Predicted thin-barrier hazard probability anchored on crisp observed Z_t + unrolled Z_{t+k}
    vis_max = jnp.max(zf[:, :, SLOTS_VISUAL, :], axis=2)
    contact_ctx = 0.45 * vis_max + 0.25 * vis_sum + 0.15 * aud_sum + 0.15 * reg_sum
    contact_logits = (
        contact_ctx @ params["head_contact_w"].astype(jnp.float32)
        + params["head_contact_b"].astype(jnp.float32)
    )
    step_bar_prob = jax.nn.sigmoid(contact_logits[:, :, 1])
    barrier_prob = jnp.maximum(bar0_prob[:, None], step_bar_prob)
    has_vis_bar = avoid_barrier & (barrier_prob > 0.5) & (~trust_proprio_only[:, None])

    dist_goal = geodesic_goal_distance(obj_xy, target_xy, has_vis_bar)
    on_opp = (obj_xy[..., 0] * target_xy[..., 0]) < 0.02
    blocked = has_vis_bar & on_opp & (jnp.abs(obj_xy[..., 1]) < 0.46)
    detour_y = jnp.where(obj_xy[..., 1] + 0.5 * target_xy[..., 1] >= 0.0, 0.55, -0.55)
    targ_side = jnp.broadcast_to(jnp.where(target_xy[..., 0] >= 0.0, 1.0, -1.0), detour_y.shape)
    wp_xy = jnp.where(blocked[..., None], jnp.stack([targ_side * 0.22, detour_y], axis=-1), target_xy)
    push_dir = wp_xy - obj_xy
    push_unit = push_dir * jax.lax.rsqrt(jnp.sum(push_dir * push_dir, axis=-1, keepdims=True) + 1e-6)
    rel_ee = ee_xy - obj_xy
    proj_along = jnp.sum(rel_ee * push_unit, axis=-1, keepdims=True)
    perp = jnp.stack([-push_unit[..., 1], push_unit[..., 0]], axis=-1)
    lat_err = jnp.abs(jnp.sum(rel_ee * perp, axis=-1, keepdims=True))
    aligned = (proj_along <= -0.09) & (lat_err < 0.10)
    drive_xy = jnp.where(aligned, obj_xy + 0.06 * push_unit, obj_xy - 0.19 * push_unit)
    ee_blocked = has_vis_bar & ((ee_xy[..., 0] * drive_xy[..., 0]) < 0.0) & (jnp.abs(ee_xy[..., 1]) < 0.42)
    drive_xy = jnp.where(
        ee_blocked[..., None],
        jnp.stack([ee_xy[..., 0] * 0.25, detour_y], axis=-1),
        drive_xy,
    )
    dist_reach = jnp.sqrt(jnp.sum((ee_xy - drive_xy) ** 2, axis=-1) + 1e-6)

    # Penalize trajectories driving either puck OR end-effector directly into central thin barrier
    obj_near_wall = (jnp.abs(obj_xy[:, :, 0]) < 0.14) & (jnp.abs(obj_xy[:, :, 1]) < 0.39)
    ee_near_wall = (jnp.abs(ee_xy[:, :, 0]) < 0.10) & (jnp.abs(ee_xy[:, :, 1]) < 0.37)
    near_wall = obj_near_wall | ee_near_wall
    barrier_pen = (0.85 * barrier_prob * near_wall.astype(jnp.float32)) if avoid_barrier else 0.0
    unc_pen = 0.12 * jnp.mean(jnp.exp(0.5 * jnp.squeeze(log_var_step, axis=-1)), axis=-1)

    k_cand = zf.shape[1]
    goal_bk = jnp.broadcast_to(goal_xy[:, None, :], (zf.shape[0], k_cand, spec.goal_dim))
    act_flag = jnp.ones((zf.shape[0], k_cand, 1), dtype=jnp.float32)
    rel_goal = 3.5 * _ego_push_kinematics(ee_xy, obj_xy, goal_bk[:, :, 2:4], act_flag, avoid_barrier=has_vis_bar)
    pol_ctx = jnp.where(
        trust_proprio_only[:, None, None],
        jnp.concatenate([jnp.zeros_like(vis_sum), prop_lat, rel_goal], axis=-1),
        jnp.concatenate([0.65 * vis_sum + 0.35 * reg_sum, jnp.zeros_like(prop_lat), rel_goal], axis=-1),
    )
    val_boot = jnp.squeeze(
        jax.nn.silu(pol_ctx @ params["head_value_w1"].astype(jnp.float32))
        @ params["head_value_w2"].astype(jnp.float32),
        axis=-1,
    )

    step_reward = -1.85 * dist_goal - 0.38 * dist_reach - barrier_pen - unc_pen
    return step_reward, val_boot


def plan_latent_cem_mpc(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_curr: jax.Array,
    goal: jax.Array,
    rng_key: jax.Array,
    *,
    obs_proprio: jax.Array | None = None,
    horizon: int = 6,
    num_candidates: int = 48,
    num_elites: int = 8,
    cem_iters: int = 3,
    residual: bool = True,
    trust_proprio_only: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Batched Multi-Horizon Latent Cross-Entropy Method MPC over SLWM-X1B world dynamics."""
    b = z_curr.shape[0]
    if trust_proprio_only is None:
        trust_proprio_only = jnp.zeros((b,), dtype=bool)

    eff_goal, bar0_prob, v_dec0 = _resolve_mpc_goal_and_barrier(params, z_curr, goal, obs_proprio)
    avoid0 = (bar0_prob > 0.5) & (~trust_proprio_only)

    zf = z_curr.astype(jnp.float32)
    reg_sum = jnp.mean(zf[:, SLOTS_REGISTER, :], axis=1)
    vis_sum = jnp.mean(zf[:, SLOTS_VISUAL, :], axis=1)
    head_pw = params["head_proprio_w"].astype(jnp.float32)
    head_pb = params["head_proprio_b"].astype(jnp.float32)
    head_vw = params["head_vis_pose_w"].astype(jnp.float32)
    head_vb = params["head_vis_pose_b"].astype(jnp.float32)
    act_flag0 = jnp.ones((b, 1), dtype=jnp.float32)

    if obs_proprio is not None:
        obs_pf = obs_proprio.astype(jnp.float32)
        prop_ee_xy = obs_pf[:, 0:2]
        prop_obj_xy = obs_pf[:, 2:4]
        prop_kin0 = _ego_push_kinematics(prop_ee_xy, prop_obj_xy, eff_goal[:, 2:4], act_flag0)
        obs_pf_aug = obs_pf.at[..., 12:15].set(prop_kin0[..., :3]).at[..., 15].set(1.0)
        prop_emb = 3.5 * (obs_pf_aug @ params["prop_w"].astype(jnp.float32))
    else:
        prop_emb = zf[:, SLOT_PROPRIO, :]
        p_dec0 = (0.7 * prop_emb + 0.3 * reg_sum) @ head_vw + head_vb
        prop_ee_xy = p_dec0[:, 0:2]
        prop_obj_xy = p_dec0[:, 2:4]

    vis_ee_xy = v_dec0[:, 0:2]
    vis_obj_xy = v_dec0[:, 2:4]
    vis_rel = 3.5 * _ego_push_kinematics(vis_ee_xy, vis_obj_xy, eff_goal[:, 2:4], act_flag0, avoid_barrier=avoid0)
    prop_rel = 3.5 * _ego_push_kinematics(prop_ee_xy, prop_obj_xy, eff_goal[:, 2:4], act_flag0, avoid_barrier=None)

    pol_in = jnp.where(
        trust_proprio_only[:, None],
        jnp.concatenate([jnp.zeros_like(vis_sum), prop_emb, prop_rel], axis=-1),
        jnp.concatenate([0.65 * vis_sum + 0.35 * reg_sum, jnp.zeros_like(prop_emb), vis_rel], axis=-1),
    )
    init_act = jnp.tanh(
        jax.nn.silu(pol_in @ params["head_action_w1"].astype(jnp.float32))
        @ params["head_action_w2"].astype(jnp.float32)
    )

    # Warm-start candidate 0 with multi-step barrier-aware latent policy rollout
    prop0_lat = zf[:, SLOT_PROPRIO, :]
    vis0_sum = vis_sum
    reg0_sum = reg_sum
    use_euler = (obs_proprio is not None) and residual and (spec.arch != "legacy_scalar_hybrid")

    def _warm_step(z_w: jax.Array, _: Any) -> tuple[jax.Array, jax.Array]:
        zw_f = z_w.astype(jnp.float32)
        rw = jnp.mean(zw_f[:, SLOTS_REGISTER, :], axis=1)
        vw = jnp.mean(zw_f[:, SLOTS_VISUAL, :], axis=1)
        pw = zw_f[:, SLOT_PROPRIO, :]
        if use_euler and obs_proprio is not None:
            p_full = obs_proprio.astype(jnp.float32) + ((0.75 * (pw - prop0_lat) + 0.25 * (vw - vis0_sum)) @ head_pw)
            p_tr = obs_proprio.astype(jnp.float32) + ((0.80 * (pw - prop0_lat) + 0.20 * (rw - reg0_sum)) @ head_pw)
        else:
            p_full = (0.75 * pw + 0.25 * vw) @ head_pw + head_pb
            p_tr = (0.80 * pw + 0.20 * rw) @ head_pw + head_pb
        w_vis_rel = 3.5 * _ego_push_kinematics(
            p_full[:, 0:2], p_full[:, 2:4], eff_goal[:, 2:4], act_flag0, avoid_barrier=avoid0
        )
        w_prop_rel = 3.5 * _ego_push_kinematics(
            p_tr[:, 0:2], p_tr[:, 2:4], eff_goal[:, 2:4], act_flag0, avoid_barrier=None
        )
        p_in = jnp.where(
            trust_proprio_only[:, None],
            jnp.concatenate([jnp.zeros_like(vw), pw, w_prop_rel], axis=-1),
            jnp.concatenate([0.65 * vw + 0.35 * rw, jnp.zeros_like(pw), w_vis_rel], axis=-1),
        )
        u_w = jnp.tanh(
            jax.nn.silu(p_in @ params["head_action_w1"].astype(jnp.float32))
            @ params["head_action_w2"].astype(jnp.float32)
        )
        z_wn, _ = predict_latent_step(params, spec, z_w, u_w, horizon_idx=0, residual=residual)
        return z_wn, u_w

    _, warm_seq = jax.lax.scan(_warm_step, z_curr, None, length=horizon)
    warm_traj = jnp.swapaxes(warm_seq, 0, 1).at[:, 0, :].set(init_act)
    mu = warm_traj
    std = jnp.full((b, horizon, spec.action_dim), 0.38, dtype=jnp.float32)
    best_return = jnp.full((b,), -1e9, dtype=jnp.float32)
    best_first_act = init_act
    z_anchor = jnp.broadcast_to(z_curr[:, None, :, :], (b, num_candidates, spec.slots_per_tick, spec.d_model))

    for _ in range(cem_iters):
        rng_key, subk1, subk2 = jax.random.split(rng_key, 3)
        # Temporally correlated iCEM perturbations around policy prior (sustained pushes + fine adjustments)
        eps_white = jax.random.normal(subk1, (b, num_candidates, horizon, spec.action_dim), dtype=jnp.float32)
        eps_macro = jax.random.normal(subk2, (b, num_candidates, 1, spec.action_dim), dtype=jnp.float32)
        eps = 0.65 * eps_macro + 0.45 * eps_white
        cand_actions = jnp.clip(mu[:, None, :, :] + std[:, None, :, :] * eps, -1.0, 1.0)
        cand_actions = cand_actions.at[:, 0, :, :].set(warm_traj)
        if num_candidates > 1:
            cand_actions = cand_actions.at[:, 1, :, :].set(mu)

        def rollout_step(
            carry: tuple[jax.Array, jax.Array],
            t_idx_u_and_prior: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            z_k, acc_ret = carry
            t_idx, u_t, u_prior_t = t_idx_u_and_prior
            z_next, lv_next = predict_latent_step(params, spec, z_k, u_t, horizon_idx=0, residual=residual)
            r_t, v_t = _score_latent_state(
                params,
                spec,
                z_next,
                z_anchor,
                obs_proprio,
                lv_next,
                eff_goal,
                bar0_prob,
                residual=residual,
                trust_proprio_only=trust_proprio_only,
            )
            discount = 0.95 ** t_idx.astype(jnp.float32)
            u_pen = 0.012 * jnp.sum(u_t * u_t, axis=-1) + 0.03 * jnp.sum((u_t - u_prior_t[:, None, :]) ** 2, axis=-1)
            return (z_next, acc_ret + discount * (r_t - u_pen)), v_t

        (_, total_ret), v_seq = jax.lax.scan(
            rollout_step,
            (z_anchor, jnp.zeros((b, num_candidates), dtype=jnp.float32)),
            (
                jnp.arange(horizon, dtype=jnp.int32),
                jnp.moveaxis(cand_actions, 2, 0),
                jnp.moveaxis(warm_traj, 1, 0),
            ),
        )
        total_ret = total_ret + (0.95**horizon) * 0.45 * v_seq[-1]

        elite_ret, elite_idx = jax.lax.top_k(total_ret, num_elites)
        batch_idx = jnp.arange(b)[:, None]
        elites = cand_actions[batch_idx, elite_idx]
        # Sharp MPPI softmax weighting prevents bimodal (+y vs -y) thin-barrier detours from averaging into y=0!
        w_elite = jax.nn.softmax((elite_ret - elite_ret[:, :1]) * 6.0, axis=-1)[:, :, None, None]
        mu = 0.20 * mu + 0.80 * jnp.sum(w_elite * elites, axis=1)
        std = jnp.maximum(0.06, jnp.sqrt(jnp.sum(w_elite * ((elites - mu[:, None]) ** 2), axis=1) + 1e-6))
        best_return = elite_ret[:, 0]
        best_first_act = elites[:, 0, 0, :]

    return best_first_act, {"trajectory_mean": mu, "trajectory_std": std, "best_return": best_return}


def plan_random_shooting_g8(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_curr: jax.Array,
    goal: jax.Array,
    rng_key: jax.Array,
    *,
    obs_proprio: jax.Array | None = None,
    num_candidates: int = 3,
    residual: bool = True,
) -> jax.Array:
    """G8 baseline planner: 1-decision 3-candidate random shooting without multi-horizon CEM refinement."""
    b = z_curr.shape[0]
    eff_goal, bar0_prob, _ = _resolve_mpc_goal_and_barrier(params, z_curr, goal, obs_proprio)
    cand = jax.random.uniform(rng_key, (b, num_candidates, spec.action_dim), minval=-1.0, maxval=1.0)
    z_roll = jnp.broadcast_to(z_curr[:, None, :, :], (b, num_candidates, spec.slots_per_tick, spec.d_model))
    z_next, lv_next = predict_latent_step(params, spec, z_roll, cand, horizon_idx=0, residual=residual)
    trust_none = jnp.zeros((b,), dtype=bool)
    r_1, _ = _score_latent_state(
        params,
        spec,
        z_next,
        z_roll,
        obs_proprio,
        lv_next,
        eff_goal,
        bar0_prob,
        residual=residual,
        trust_proprio_only=trust_none,
    )
    best_idx = jnp.argmax(r_1, axis=-1)
    return cand[jnp.arange(b), best_idx]
