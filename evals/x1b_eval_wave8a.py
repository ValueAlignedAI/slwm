"""Wave 8A Joint Recall + Spatial Non-Regression & Multi-Item Overwrite Evaluation Suite (`SLWM-X2`).

Executes all 6 Wave 8A evaluation panels using `forward_world_model_wave8a` and `decode_world_heads_wave8a`
under strict End-Effector-Only (`ee_only`) proprioception and zero geometric planner heuristics:
  1. `unseen_layout_deprivileged_mpc`: Closed-loop manipulation across 4 unseen blocked obstacle/dynamics
     families (`off_center_wall`, `dual_chicane`, `multi_block_clutter`, `dynamics_shift`).
  2. `beyond_window_T160_distractor_switch`: `T = 160` (`> 2.5x W = 64`) delayed & distracted Goal-Switch
     with post-window physical reset perturbation at `t = 88`.
  3. `symbolic_key_drift_and_recall`: Mechanistic symbolic-key visual cosine drift + `T = 64, 256, 512, 1024`
     dynamic-visual associative overwrite recall.
  4. `unseen_ood_sensor_corruptions`: Zero-shot safety detection AUROC across 4 unseen camera corruptions.
  5. `grounded_spoken_robot_dev` & `untouched_vault_spoken_robot`: Physical `xArm6` 4-step Cartesian forecast
     error (mm), counterfactual causal plan gain (mm), and speaker-disjoint spoken + 16-token grounding.
  6. `multi_item_overwrite_capacity`: Multi-item associative binding + mid-horizon selective overwrite capacity
     curve across `K in {1, 2, 4, 8, 16}` concurrent bindings and horizons `T in {64, 256, 1024}` under live
     dynamic RGB motion, separately reporting overall, overwritten-key, retained-key, and stale-capture rates.
"""

from __future__ import annotations

import functools
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_multimodal import (
    ASSOC_KEY_OFFSET,
    ASSOC_NUM_KEYS,
    ASSOC_NUM_VALS,
    ASSOC_VAL_OFFSET,
    generate_synthetic_world_batch,
)
from data.x1b_unseen_worlds import (
    LAYOUT_DUAL_CHICANE,
    LAYOUT_DYNAMICS_SHIFT,
    LAYOUT_MULTI_CLUTTER,
    LAYOUT_OFF_CENTER,
    apply_unseen_ood_sensor_corruption,
    init_unseen_layout_worlds,
    mask_privileged_proprio,
    render_unseen_world_rgb,
    step_unseen_layout_world,
)
from evals.x1b_eval_wave6 import PHYSICAL_TABLE_SCALE_MM, _binary_auroc
from evals.x1b_eval_wave7 import (
    _build_tick_batch_v7,
    _proportional_push_action,
    compute_learned_value_grid_8x8,
    init_unseen_blocked_worlds,
    plan_explicit_latch_reactive_v7,
    update_self_filtered_occ_8x8,
)
from models.x1b_commit import route_proposals_with_commit_gate
from models.x1b_jepa import l2_normalize, predict_latent_step
from models.x1b_wave8a_core import (
    Wave8AInterventions,
    de_privilege_proprio_wave8a,
    decode_goal_latch_wave8a,
    decode_spatial_hazard_grid_8x8,
    decode_spatial_vis_pose,
    decode_target_block_xy_wave8a,
    decode_world_heads_wave8a,
    encode_goal_query_tokens,
    encode_goal_tokens_2d,
    encode_pure_visual_grid_8x8,
    encode_pure_visual_patches,
    extract_decoder_streaming_carry,
    fork_isolated_latent_streams,
    forward_world_model_wave8a,
    init_decoder_streaming_carry,
    sample_hazard_grid_bilinear,
    spatial_grid_local_context,
)
from models.x1b_world import (
    SLOT_PROPRIO,
    SLOT_TEXT,
    SLOTS_AUDIO,
    SLOTS_REGISTER,
    SLOTS_VISUAL,
    X1BSpec,
    compute_streaming_state_bytes,
    init_streaming_memories,
)


def _propose_subgoal_candidates_shared(
    obj_xy: jax.Array,
    target_xy: jax.Array,
    occ_cal: jax.Array,
    v_grid: jax.Array,
    num_candidates: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shared subgoal candidate generator used identically by learned and oracle MPPI planners."""
    b = obj_xy.shape[0]
    c8 = jnp.linspace(-0.875, 0.875, 8, dtype=jnp.float32)
    yy8, xx8 = jnp.meshgrid(c8, c8, indexing="ij")
    raw_grid_pts = jnp.stack([xx8.reshape(-1), yy8.reshape(-1)], axis=-1)
    safe_grid_pts = jnp.clip(raw_grid_pts, -0.66, 0.66)
    d_obj_all = jnp.sqrt(jnp.sum((safe_grid_pts[None, :, :] - obj_xy[:, None, :]) ** 2, axis=-1) + 1e-6)
    in_reach_all = (
        (d_obj_all >= 0.14)
        & (d_obj_all <= 0.78)
        & (jnp.max(jnp.abs(raw_grid_pts[None, :, :]), axis=-1) < 0.78)
    )
    vg_flat = v_grid.reshape(b, 64)
    occ_flat = occ_cal.reshape(b, 64)
    cell_prior = -vg_flat - 12.0 * occ_flat - jnp.where(in_reach_all, 0.0, 40.0)
    k_grid = min(64, max(8, num_candidates - 1))
    _, top_idx = jax.lax.top_k(cell_prior, k_grid)
    ring_wp = safe_grid_pts[top_idx]
    v_ring = jnp.take_along_axis(vg_flat, top_idx, axis=-1)
    occ_ring = jnp.take_along_axis(occ_flat, top_idx, axis=-1)
    reach_ring = jnp.take_along_axis(in_reach_all, top_idx, axis=-1)

    all_wp = jnp.concatenate([target_xy[:, None, :], ring_wp], axis=1)
    v_wp = jnp.concatenate([jnp.zeros((b, 1), dtype=jnp.float32), v_ring], axis=1)
    occ_wp = jnp.concatenate([jnp.zeros((b, 1), dtype=jnp.float32), occ_ring], axis=1)
    return all_wp, v_wp, occ_wp, reach_ring


def _score_subgoal_candidates_shared(
    obj_xy: jax.Array,
    target_xy: jax.Array,
    all_wp: jax.Array,
    v_wp: jax.Array,
    occ_wp: jax.Array,
    reach_ring: jax.Array,
    occ_cal: jax.Array,
    rollout_ret: jax.Array,
    fin_v: jax.Array,
    prev_wp: jax.Array | None,
    has_prev_wp: jax.Array | bool,
    *,
    use_rollout: bool = True,
) -> jax.Array:
    """Single shared subgoal scoring functional used identically by learned and oracle MPPI controllers."""
    b = obj_xy.shape[0]
    q1_wp = 0.67 * obj_xy[:, None, :] + 0.33 * all_wp
    q2_wp = 0.33 * obj_xy[:, None, :] + 0.67 * all_wp
    edge_haz = jnp.maximum(
        occ_wp,
        jnp.maximum(sample_hazard_grid_bilinear(occ_cal, q1_wp), sample_hazard_grid_bilinear(occ_cal, q2_wp)),
    )
    has_pwp = jnp.broadcast_to(jnp.asarray(has_prev_wp, dtype=bool), (b,))
    pwp_xy = jnp.where(has_pwp[:, None], prev_wp if prev_wp is not None else target_xy, target_xy)
    pwp_bonus = jnp.where(
        has_pwp[:, None],
        -0.16 * jnp.sqrt(jnp.sum((all_wp - pwp_xy[:, None, :]) ** 2, axis=-1) + 1e-6),
        0.0,
    )
    dist_goal_now = jnp.sqrt(jnp.sum((target_xy - obj_xy) ** 2, axis=-1) + 1e-6)
    direct_clear = (dist_goal_now < 0.45) & (edge_haz[:, 0] < 0.25)
    reach_all = jnp.concatenate([direct_clear[:, None], reach_ring], axis=1)

    roll_term = (0.14 * rollout_ret - 0.12 * fin_v) if use_rollout else 0.0
    total_ret = (
        -0.85 * v_wp
        - 20.0 * jnp.clip(edge_haz - 0.40, 0.0, 1.0)
        + roll_term
        + pwp_bonus
        - jnp.where(reach_all, 0.0, 30.0)
    )
    return total_ret.at[:, 0].add(jnp.where(direct_clear, 2.5, 0.0))


def plan_spatial_geometry_free_mppi_v8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_curr: jax.Array,
    interventions: Wave8AInterventions,
    ext_goal_xy: jax.Array,
    ee_xy: jax.Array,
    rng_key: jax.Array,
    *,
    pure_v_patches: jax.Array | None = None,
    pure_v_ti: jax.Array | None = None,
    pure_v_8x8: jax.Array | None = None,
    occ_8x8: jax.Array | None = None,
    vis_obj_vel: jax.Array | None = None,
    prev_wp: jax.Array | None = None,
    has_prev_wp: jax.Array | bool = False,
    horizon: int = 8,
    num_candidates: int = 25,
    enable_jepa_rollout: bool = True,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Obstacle-coordinate-free Spatial-Grid Latent-JEPA + Bellman planner over isotropic 2D subgoals."""
    del rng_key, vis_obj_vel
    b = z_curr.shape[0]
    w6_iv = interventions.as_wave6()
    _, vis_obj0_xy, _ = decode_spatial_vis_pose(
        params, z_curr, w6_iv, pure_v_patches=pure_v_patches, pure_v_8x8=pure_v_8x8
    )
    latched_xy = decode_goal_latch_wave8a(params, z_curr, interventions)
    has_ext = jnp.sum(jnp.abs(ext_goal_xy), axis=-1, keepdims=True) > 1e-4
    target_xy = jnp.where(has_ext, ext_goal_xy, latched_xy)

    if occ_8x8 is None:
        raw_hl8 = decode_spatial_hazard_grid_8x8(
            params, z_curr, w6_iv, pure_v_ti=pure_v_ti, pure_v_8x8=pure_v_8x8
        )
        occ_map = update_self_filtered_occ_8x8(
            jnp.zeros((b, 8, 8), dtype=jnp.float32),
            raw_hl8,
            ee_xy,
            vis_obj0_xy,
            True,
            ext_goal_xy=ext_goal_xy,
        )
    else:
        occ_map = occ_8x8

    occ_sorted = jnp.sort(occ_map.reshape(b, 64), axis=-1)
    bg_med = occ_sorted[:, 32][:, None, None]
    pk_95 = occ_sorted[:, 61][:, None, None]
    occ_span = jnp.maximum(pk_95 - bg_med, 0.065)
    occ_cal = jnp.clip((occ_map - (bg_med + 0.28 * occ_span)) / (0.52 * occ_span), 0.0, 1.0)
    v_grid = compute_learned_value_grid_8x8(occ_cal, target_xy)

    all_wp, v_wp, occ_wp, reach_ring = _propose_subgoal_candidates_shared(
        vis_obj0_xy, target_xy, occ_cal, v_grid, num_candidates
    )
    k_tot = all_wp.shape[1]

    z_init = jnp.broadcast_to(z_curr[:, None, :, :], (b, k_tot, spec.slots_per_tick, spec.d_model))
    ee_init = jnp.broadcast_to(ee_xy[:, None, :], (b, k_tot, 2))
    obj_init = jnp.broadcast_to(vis_obj0_xy[:, None, :], (b, k_tot, 2))
    fg_k = jnp.broadcast_to(target_xy[:, None, :], (b, k_tot, 2))
    p_vti_k = (
        jnp.broadcast_to(pure_v_ti[:, None, :, :], (b, k_tot, 16, spec.d_model))
        if pure_v_ti is not None
        else None
    )
    head_pw = params["head_proprio_w"].astype(jnp.float32)
    head_pb = params["head_proprio_b"].astype(jnp.float32)
    roll_h = min(max(4, horizon), 12)

    def rollout_step(carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], t_idx: jax.Array):
        z_k, pred_ee_prev, pred_obj_prev, jepa_acc = carry
        u_k = _proportional_push_action(pred_ee_prev, pred_obj_prev, all_wp, final_goal_xy=fg_k)
        z_next, lv_next = predict_latent_step(
            params, spec, z_k, u_k, horizon_idx=0, residual=interventions.zerolag_bypass
        )
        zn_f = z_next.astype(jnp.float32)
        zk_f = z_k.astype(jnp.float32)

        vis_s = jnp.mean(zn_f[:, :, SLOTS_VISUAL, :], axis=2)
        vis_prev_s = jnp.mean(zk_f[:, :, SLOTS_VISUAL, :], axis=2)
        aud_s = jnp.mean(zn_f[:, :, SLOTS_AUDIO, :], axis=2)
        reg_s = jnp.mean(zn_f[:, :, SLOTS_REGISTER, :], axis=2)
        prop_s = zn_f[:, :, SLOT_PROPRIO, :]
        prop_prev_s = zk_f[:, :, SLOT_PROPRIO, :]

        if interventions.zerolag_bypass:
            step_delta = (0.75 * (prop_s - prop_prev_s) + 0.25 * (vis_s - vis_prev_s)) @ head_pw
            pred_ee = jnp.clip(pred_ee_prev + step_delta[:, :, 0:2], -0.90, 0.90)
            raw_obj = pred_obj_prev + step_delta[:, :, 2:4]
            _, vis_obj_step, _ = decode_spatial_vis_pose(params, z_next, w6_iv)
            pred_obj = jnp.clip(0.65 * raw_obj + 0.35 * vis_obj_step, -0.90, 0.90)
        else:
            raw_p = (0.75 * prop_s + 0.25 * vis_s) @ head_pw + head_pb
            pred_ee = jnp.clip(raw_p[:, :, 0:2], -0.90, 0.90)
            pred_obj = jnp.clip(raw_p[:, :, 2:4], -0.90, 0.90)

        if interventions.decoupled_heads:
            loc_vis = spatial_grid_local_context(
                zn_f[:, :, SLOTS_VISUAL, :], pred_obj, pred_ee, pure_v_ti=p_vti_k
            )
            contact_ctx = 0.72 * loc_vis + 0.14 * aud_s + 0.14 * reg_s
        else:
            contact_ctx = jnp.mean(zn_f, axis=2)

        c_logits = contact_ctx @ params["head_contact_w"].astype(jnp.float32) + params["head_contact_b"].astype(jnp.float32)
        contact_haz = jax.nn.sigmoid(c_logits[:, :, 1])
        unc_pen = 0.02 * jnp.mean(jnp.exp(0.5 * jnp.squeeze(lv_next, axis=-1)), axis=-1)

        to_goal = fg_k - pred_obj
        rel_vec = jnp.concatenate([to_goal, pred_obj - pred_ee], axis=-1)
        pol_ctx = (
            jnp.concatenate([0.65 * vis_s + 0.35 * reg_s, jnp.zeros_like(prop_s), rel_vec], axis=-1)
            if interventions.decoupled_heads
            else jnp.concatenate([vis_s, prop_s, rel_vec], axis=-1)
        )
        val_boot = jnp.squeeze(
            jax.nn.silu(pol_ctx @ params["head_value_w1"].astype(jnp.float32))
            @ params["head_value_w2"].astype(jnp.float32),
            axis=-1,
        )
        disc = 0.96 ** t_idx.astype(jnp.float32)
        step_score = -0.25 * contact_haz - unc_pen + 0.04 * val_boot
        return (z_next, pred_ee, pred_obj, jepa_acc + disc * step_score), pred_obj

    if enable_jepa_rollout:
        (_, _, pred_obj_fin, jepa_ret), pred_obj_traj = jax.lax.scan(
            rollout_step,
            (z_init, ee_init, obj_init, jnp.zeros((b, k_tot), dtype=jnp.float32)),
            jnp.arange(roll_h, dtype=jnp.int32),
        )
        fin_v = sample_hazard_grid_bilinear(jnp.minimum(v_grid, 3.5), pred_obj_fin)
    else:
        jepa_ret = jnp.zeros((b, k_tot), dtype=jnp.float32)
        fin_v = jnp.zeros((b, k_tot), dtype=jnp.float32)
        pred_obj_traj = jnp.broadcast_to(vis_obj0_xy[None, :, None, :], (roll_h, b, k_tot, 2))

    total_ret = _score_subgoal_candidates_shared(
        vis_obj0_xy,
        target_xy,
        all_wp,
        v_wp,
        occ_wp,
        reach_ring,
        occ_cal,
        jepa_ret,
        fin_v,
        prev_wp,
        has_prev_wp,
        use_rollout=enable_jepa_rollout,
    )

    best_idx = jnp.argmax(total_ret, axis=-1)
    best_ret = total_ret[jnp.arange(b), best_idx]
    best_wp = all_wp[jnp.arange(b), best_idx]
    best_traj = pred_obj_traj[:, jnp.arange(b), best_idx, :]  # [roll_h, B, 2]

    best_u0 = _proportional_push_action(ee_xy, vis_obj0_xy, best_wp, final_goal_xy=target_xy)
    return best_u0, {
        "best_return": best_ret,
        "decoded_obj_xy": vis_obj0_xy,
        "resolved_target_xy": target_xy,
        "best_wp": best_wp,
        "best_pred_obj_traj": jnp.swapaxes(best_traj, 0, 1),
        "occ_8x8": occ_map,
        "v_grid_8x8": v_grid,
    }


def _true_world_occ_8x8(st: dict[str, jax.Array]) -> jax.Array:
    """Compute exact oracle 8x8 obstacle occupancy grid from ground-truth world box coordinates."""
    c8 = jnp.linspace(-0.875, 0.875, 8, dtype=jnp.float32)
    yy8, xx8 = jnp.meshgrid(c8, c8, indexing="ij")
    dx = jnp.abs(xx8[None, :, :, None] - st["box_cx"][:, None, None, :])
    dy = jnp.abs(yy8[None, :, :, None] - st["box_cy"][:, None, None, :])
    hit = (st["box_hx"][:, None, None, :] > 1e-4) & (dx <= (st["box_hx"][:, None, None, :] + 0.115)) & (
        dy <= (st["box_hy"][:, None, None, :] + 0.095)
    )
    return jnp.any(hit, axis=-1).astype(jnp.float32)


def plan_oracle_perception_dynamics_mppi_v8a(
    st: dict[str, jax.Array],
    goal_xy: jax.Array,
    prev_wp: jax.Array,
    has_prev_wp: jax.Array | bool,
    *,
    horizon: int = 6,
    num_candidates: int = 32,
    enable_rollout: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Oracle-Perception x Oracle-Dynamics subgoal MPPI ceiling sharing the exact candidate & scoring functional."""
    b = st["ee_xy"].shape[0]
    true_obj = st["obj_xy"]
    true_ee = st["ee_xy"]
    occ_cal = _true_world_occ_8x8(st)
    v_grid = compute_learned_value_grid_8x8(occ_cal, goal_xy)

    all_wp, v_wp, occ_wp, reach_ring = _propose_subgoal_candidates_shared(
        true_obj, goal_xy, occ_cal, v_grid, num_candidates
    )
    k_tot = all_wp.shape[1]
    roll_h = min(max(4, horizon), 12)

    def _tile_state(a: jax.Array) -> jax.Array:
        return jnp.repeat(a, k_tot, axis=0)

    st_bk = {k: _tile_state(v) for k, v in st.items() if k != "layout_id"}
    wp_bk = all_wp.reshape(b * k_tot, 2)
    fg_bk = jnp.repeat(goal_xy, k_tot, axis=0)

    def _oracle_step(carry: tuple[dict[str, jax.Array], jax.Array], t_idx: jax.Array):
        carry_st, ret_acc = carry
        u_bk = _proportional_push_action(carry_st["ee_xy"], carry_st["obj_xy"], wp_bk, final_goal_xy=fg_bk)
        nxt_st, _ = step_unseen_layout_world(carry_st, u_bk)
        dx_o = jnp.abs(nxt_st["obj_xy"][:, None, 0] - nxt_st["box_cx"])
        dy_o = jnp.abs(nxt_st["obj_xy"][:, None, 1] - nxt_st["box_cy"])
        dx_e = jnp.abs(nxt_st["ee_xy"][:, None, 0] - nxt_st["box_cx"])
        dy_e = jnp.abs(nxt_st["ee_xy"][:, None, 1] - nxt_st["box_cy"])
        active_box = nxt_st["box_hx"] > 1e-4
        near_box = jnp.any(
            active_box
            & (
                ((dx_o < nxt_st["box_hx"] + 0.11) & (dy_o < nxt_st["box_hy"] + 0.06))
                | ((dx_e < nxt_st["box_hx"] + 0.09) & (dy_e < nxt_st["box_hy"] + 0.05))
            ),
            axis=-1,
        ).astype(jnp.float32)
        true_val_t = -1.35 * jnp.sqrt(jnp.sum((nxt_st["obj_xy"] - fg_bk) ** 2, axis=-1) + 1e-6) - 0.65 * near_box
        disc = 0.96 ** t_idx.astype(jnp.float32)
        step_score = -0.25 * near_box + 0.04 * true_val_t
        return (nxt_st, ret_acc + disc * step_score), nxt_st["obj_xy"]

    if enable_rollout:
        (fin_st, oracle_ret_bk), _ = jax.lax.scan(
            _oracle_step,
            (st_bk, jnp.zeros((b * k_tot,), dtype=jnp.float32)),
            jnp.arange(roll_h, dtype=jnp.int32),
        )
        fin_obj = fin_st["obj_xy"].reshape(b, k_tot, 2)
        oracle_ret = oracle_ret_bk.reshape(b, k_tot)
        fin_v = sample_hazard_grid_bilinear(jnp.minimum(v_grid, 3.5), fin_obj)
    else:
        oracle_ret = jnp.zeros((b, k_tot), dtype=jnp.float32)
        fin_v = jnp.zeros((b, k_tot), dtype=jnp.float32)

    total_ret = _score_subgoal_candidates_shared(
        true_obj,
        goal_xy,
        all_wp,
        v_wp,
        occ_wp,
        reach_ring,
        occ_cal,
        oracle_ret,
        fin_v,
        prev_wp,
        has_prev_wp,
        use_rollout=enable_rollout,
    )
    best_idx = jnp.argmax(total_ret, axis=-1)
    best_wp = all_wp[jnp.arange(b), best_idx]
    best_u0 = _proportional_push_action(true_ee, true_obj, best_wp, final_goal_xy=goal_xy)
    return best_u0, best_wp


def eval_unseen_layouts_deprivileged_mpc_v8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    seed: int = 202801,
    n_worlds: int = 64,
    horizon_ticks: int = 116,
) -> dict[str, Any]:
    """Panel W8A-1: Closed-loop control across 4 unseen blocked layouts under `ee_only` proprioception."""
    init_state = init_unseen_blocked_worlds(jax.random.key(seed), batch_size=n_worlds)
    goal_xy = init_state["targ_a"]
    goal_write_tok = encode_goal_tokens_2d(goal_xy, is_distractor=False)
    goal_query_tok = encode_goal_query_tokens((n_worlds,))
    w6_iv = interventions.as_wave6()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _rollout_oracle(use_roll: bool):
        def body_oracle(carry, t_idx):
            st, prev_wp = carry
            u_t, next_wp = plan_oracle_perception_dynamics_mppi_v8a(
                st, goal_xy, prev_wp, t_idx > 0, horizon=6, num_candidates=32, enable_rollout=use_roll
            )
            st_next, _ = step_unseen_layout_world(st, u_t)
            dist_goal = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_xy) ** 2, axis=-1) + 1e-8)
            return (st_next, next_wp), dist_goal

        (st_fin, _), dist_hist = jax.lax.scan(
            body_oracle, (init_state, goal_xy), jnp.arange(horizon_ticks, dtype=jnp.int32)
        )
        tail_w = min(16, horizon_ticks)
        term_reach = (jnp.min(dist_hist[-tail_w:], axis=0) < 0.24).astype(jnp.float32)
        term_hold = (jnp.max(dist_hist[-tail_w:], axis=0) < 0.24).astype(jnp.float32)
        any_reach = (jnp.min(dist_hist, axis=0) < 0.24).astype(jnp.float32)
        return term_reach, term_hold, any_reach, st_fin["layout_id"]

    oracle_succ_j, oracle_hold_j, oracle_any_j, _ = _rollout_oracle(True)
    oracle_noroll_j, _, _, _ = _rollout_oracle(False)
    oracle_succ = np.asarray(jax.device_get(oracle_succ_j))
    oracle_hold = np.asarray(jax.device_get(oracle_hold_j))
    oracle_any = np.asarray(jax.device_get(oracle_any_j))
    oracle_noroll = np.asarray(jax.device_get(oracle_noroll_j))

    def _run_mode(mode: str) -> dict[str, np.ndarray]:
        mems_init = init_streaming_memories(spec, n_worlds, cache_len=max(64, horizon_ticks + 4))
        dec_carry0 = init_decoder_streaming_carry(spec, n_worlds)
        prev_u0 = jnp.zeros((n_worlds, spec.action_dim), dtype=jnp.float32)
        latch0 = goal_xy
        occ0 = jnp.zeros((n_worlds, 8, 8), dtype=jnp.float32)
        prev_vobj0 = jnp.zeros((n_worlds, 2), dtype=jnp.float32)
        c0 = jnp.zeros((n_worlds,), dtype=jnp.float32)

        @jax.jit
        def _rollout(p: dict[str, jax.Array]):
            def body_fn(carry, t_idx):
                st, mems, dec_carry, prev_u, latch_st, occ_st, prev_vobj, contact_c, rk = carry
                rk, pk = jax.random.split(rk)
                show_g = jnp.ones((n_worlds,), dtype=bool)
                ext_g = goal_xy
                txt_t = jnp.where(t_idx < 6, goal_write_tok, goal_query_tok)
                vid = render_unseen_world_rgb(st, goal_xy, show_g)
                ee_prop = mask_privileged_proprio(st["ee_xy"], st["ee_vel"], contact_c)
                tb = _build_tick_batch_v7(vid, ee_prop, prev_u, ext_g, txt_t)
                z_seq, next_mems, _ = forward_world_model_wave8a(
                    p, spec, tb, interventions, memories=mems, offset=t_idx
                )
                z_t = z_seq[:, 0]
                pure_vp_ti = encode_pure_visual_patches(p, spec, vid, include_pos=False)
                pure_v8 = encode_pure_visual_grid_8x8(p, spec, vid)
                vis_ee_xy, vis_obj_xy, _ = decode_spatial_vis_pose(
                    p, z_t, w6_iv, pure_v_patches=pure_vp_ti, pure_v_8x8=pure_v8
                )
                vis_obj_vel = jnp.where(t_idx > 0, jnp.clip(vis_obj_xy - prev_vobj, -0.14, 0.14), 0.0)
                hl8 = decode_spatial_hazard_grid_8x8(
                    p, z_t, w6_iv, pure_v_ti=pure_vp_ti, pure_v_8x8=pure_v8
                )
                occ_next = update_self_filtered_occ_8x8(
                    occ_st, hl8, st["ee_xy"], vis_obj_xy, t_idx == 0, ext_goal_xy=ext_g
                )

                if mode in ("geometry_free_mppi", "no_jepa_subgoal"):
                    u_t, mppi_diag = plan_spatial_geometry_free_mppi_v8a(
                        p,
                        spec,
                        z_t,
                        interventions,
                        ext_g,
                        st["ee_xy"],
                        pk,
                        pure_v_patches=pure_vp_ti,
                        pure_v_ti=pure_vp_ti,
                        pure_v_8x8=pure_v8,
                        occ_8x8=occ_next,
                        vis_obj_vel=vis_obj_vel,
                        prev_wp=latch_st,
                        has_prev_wp=(t_idx > 0),
                        horizon=6,
                        num_candidates=32,
                        enable_jepa_rollout=(mode == "geometry_free_mppi"),
                    )
                    next_latch = mppi_diag["best_wp"]
                    next_dec_carry = dec_carry
                elif mode == "explicit_latch_reactive":
                    u_t, next_latch = plan_explicit_latch_reactive_v7(
                        st["ee_xy"], vis_obj_xy, ext_g, latch_st
                    )
                    next_dec_carry = dec_carry
                else:
                    z_mem, z_task = fork_isolated_latent_streams(z_seq, interventions.zmem_isolation)
                    z_n1, lv1 = predict_latent_step(p, spec, z_task, prev_u[:, None, :], residual=True)
                    hds = decode_world_heads_wave8a(
                        p,
                        spec,
                        z_mem,
                        z_task,
                        z_n1,
                        lv1,
                        tb["goal"],
                        interventions,
                        obs_ee_proprio=tb["proprio"],
                        obs_video=tb["video"],
                        obs_modality_mask=tb["modality_mask"],
                        decoder_carry=dec_carry,
                    )
                    u_t = hds["action_proposal"][:, 0]
                    next_latch = latch_st
                    next_dec_carry = extract_decoder_streaming_carry(hds)

                st_next, c_next = step_unseen_layout_world(st, u_t)
                obj_err = jnp.sqrt(jnp.sum((vis_obj_xy - st["obj_xy"]) ** 2, axis=-1) + 1e-8)
                ee_err = jnp.sqrt(jnp.sum((vis_ee_xy - st["ee_xy"]) ** 2, axis=-1) + 1e-8)
                dist_goal = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_xy) ** 2, axis=-1) + 1e-8)
                return (
                    st_next,
                    next_mems,
                    next_dec_carry,
                    u_t,
                    next_latch,
                    occ_next,
                    vis_obj_xy,
                    c_next,
                    rk,
                ), (
                    dist_goal,
                    obj_err,
                    ee_err,
                )

            (st_fin, _, _, _, _, _, _, _, _), (dist_hist, obj_err_hist, ee_err_hist) = jax.lax.scan(
                body_fn,
                (init_state, mems_init, dec_carry0, prev_u0, latch0, occ0, prev_vobj0, c0, jax.random.key(seed + 17)),
                jnp.arange(horizon_ticks, dtype=jnp.int32),
            )
            tail_w = min(16, horizon_ticks)
            succ = (jnp.min(dist_hist[-tail_w:], axis=0) < 0.24).astype(jnp.float32)
            hold = (jnp.max(dist_hist[-tail_w:], axis=0) < 0.24).astype(jnp.float32)
            any_r = (jnp.min(dist_hist, axis=0) < 0.24).astype(jnp.float32)
            return succ, hold, any_r, jnp.mean(obj_err_hist, axis=0), jnp.mean(ee_err_hist, axis=0), st_fin["layout_id"]

        s_arr, h_arr, a_arr, oe_arr, ee_arr, lid_arr = jax.device_get(_rollout(params))
        return {
            "succ": np.asarray(s_arr),
            "hold": np.asarray(h_arr),
            "any_tick": np.asarray(a_arr),
            "vis_obj_err": np.asarray(oe_arr),
            "vis_ee_err": np.asarray(ee_arr),
            "layout": np.asarray(lid_arr),
        }

    mppi_res = _run_mode("geometry_free_mppi")
    nojepa_res = _run_mode("no_jepa_subgoal")
    react_res = _run_mode("explicit_latch_reactive")
    open_res = _run_mode("open_loop_policy")

    lid = mppi_res["layout"]
    fam_names = {
        LAYOUT_OFF_CENTER: "off_center_wall",
        LAYOUT_DUAL_CHICANE: "dual_chicane",
        LAYOUT_MULTI_CLUTTER: "multi_block_clutter",
        LAYOUT_DYNAMICS_SHIFT: "dynamics_shift",
    }
    by_family = {}
    for code, fname in fam_names.items():
        m = lid == code
        by_family[fname] = {
            "oracle_mppi_ceiling": float(np.mean(oracle_succ[m])),
            "oracle_terminal16_all_ticks_hold_success": float(np.mean(oracle_hold[m])),
            "oracle_any_tick_success": float(np.mean(oracle_any[m])),
            "mppi_success": float(np.mean(mppi_res["succ"][m])),
            "mppi_terminal16_all_ticks_hold_success": float(np.mean(mppi_res["hold"][m])),
            "no_jepa_subgoal_success": float(np.mean(nojepa_res["succ"][m])),
            "explicit_latch_reactive_success": float(np.mean(react_res["succ"][m])),
            "open_loop_success": float(np.mean(open_res["succ"][m])),
        }

    return {
        "endpoint_definition": "terminal_16_tick_window_reach_min_dist_lt_0.24",
        "oracle_perception_dynamics_mppi_ceiling": float(np.mean(oracle_succ)),
        "oracle_no_jepa_subgoal_ceiling": float(np.mean(oracle_noroll)),
        "oracle_terminal16_all_ticks_hold_success": float(np.mean(oracle_hold)),
        "oracle_any_tick_success": float(np.mean(oracle_any)),
        "geometry_free_mppi_success": float(np.mean(mppi_res["succ"])),
        "geometry_free_mppi_terminal16_all_ticks_hold_success": float(np.mean(mppi_res["hold"])),
        "geometry_free_mppi_any_tick_success": float(np.mean(mppi_res["any_tick"])),
        "geometry_free_mppi_succ_vec": [float(x) for x in mppi_res["succ"]],
        "no_jepa_subgoal_success": float(np.mean(nojepa_res["succ"])),
        "no_jepa_subgoal_succ_vec": [float(x) for x in nojepa_res["succ"]],
        "mppi_over_no_jepa_gain_pp": float((np.mean(mppi_res["succ"]) - np.mean(nojepa_res["succ"])) * 100.0),
        "explicit_latch_reactive_success": float(np.mean(react_res["succ"])),
        "explicit_latch_reactive_succ_vec": [float(x) for x in react_res["succ"]],
        "open_loop_policy_success": float(np.mean(open_res["succ"])),
        "vis_decoded_obj_rmse": float(np.mean(mppi_res["vis_obj_err"])),
        "vis_decoded_ee_rmse": float(np.mean(mppi_res["vis_ee_err"])),
        "by_unseen_family": by_family,
    }


def init_shortcut_free_t160_switch_worlds(rng_key: jax.Array, batch_size: int = 64) -> dict[str, jax.Array]:
    """Initialize shortcut-free T=160 Goal-Switch worlds with zero mutual information between poses and goals.

    Samples initial `(ee_xy, obj_xy)`, post-window `t = 88` reset `(reset_ee_xy, reset_obj_xy)`, `goal_a`,
    `goal_b` (distinct quadrant from `goal_a`), counterfactual `goal_b_prime = -goal_b`, and `false_distractor_xy`
    using independent RNG streams so observations at `t >= 20` contain strictly zero information about `goal_b`.
    """
    k_obj0, k_ee0, k_obj_r, k_ee_r, k_qa, k_ga, k_qb, k_gb, k_dist = jax.random.split(rng_key, 9)
    obj0 = jax.random.uniform(k_obj0, (batch_size, 2), minval=-0.12, maxval=0.12)
    th0 = jax.random.uniform(k_ee0, (batch_size,), minval=0.0, maxval=2.0 * np.pi)
    ee0 = obj0 + 0.20 * jnp.stack([jnp.cos(th0), jnp.sin(th0)], axis=-1)

    reset_obj = jax.random.uniform(k_obj_r, (batch_size, 2), minval=-0.12, maxval=0.12)
    th_r = jax.random.uniform(k_ee_r, (batch_size,), minval=0.0, maxval=2.0 * np.pi)
    reset_ee = reset_obj + 0.20 * jnp.stack([jnp.cos(th_r), jnp.sin(th_r)], axis=-1)

    def _quad_to_signs(q: jax.Array) -> tuple[jax.Array, jax.Array]:
        sx = jnp.where((q // 2) == 1, 1.0, -1.0)
        sy = jnp.where((q % 2) == 1, 1.0, -1.0)
        return sx, sy

    quad_a = jax.random.randint(k_qa, (batch_size,), 0, 4)
    quad_b = (quad_a + jax.random.randint(k_qb, (batch_size,), 1, 4)) % 4
    sxa, sya = _quad_to_signs(quad_a)
    sxb, syb = _quad_to_signs(quad_b)

    mag_a = jax.random.uniform(k_ga, (batch_size, 2), minval=jnp.array([0.46, 0.26]), maxval=jnp.array([0.60, 0.44]))
    mag_b = jax.random.uniform(k_gb, (batch_size, 2), minval=jnp.array([0.46, 0.26]), maxval=jnp.array([0.60, 0.44]))
    goal_a = mag_a * jnp.stack([sxa, sya], axis=-1)
    goal_b = mag_b * jnp.stack([sxb, syb], axis=-1)
    goal_b_prime = -goal_b

    qd = jax.random.randint(k_dist, (batch_size,), 0, 4)
    sxd, syd = _quad_to_signs(qd)
    mag_d = jax.random.uniform(
        jax.random.fold_in(k_dist, 1), (batch_size, 2), minval=jnp.array([0.42, 0.24]), maxval=jnp.array([0.58, 0.42])
    )
    false_distractor_xy = mag_d * jnp.stack([sxd, syd], axis=-1)
    zeros_box = jnp.zeros((batch_size, 3), dtype=jnp.float32)

    return {
        "layout_id": jnp.arange(batch_size, dtype=jnp.int32) % 4,
        "ee_xy": ee0,
        "ee_vel": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "obj_xy": obj0,
        "obj_vel": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "reset_ee_xy": reset_ee,
        "reset_obj_xy": reset_obj,
        "targ_a": goal_a,
        "targ_b": goal_b,
        "targ_b_prime": goal_b_prime,
        "false_distractor_xy": false_distractor_xy,
        "box_cx": zeros_box,
        "box_cy": zeros_box,
        "box_hx": zeros_box,
        "box_hy": zeros_box,
        "mass_scale": jnp.ones((batch_size,), dtype=jnp.float32),
        "damping": jnp.full((batch_size,), 0.74, dtype=jnp.float32),
    }


def forced_retention_sweep_action(
    ee_xy: jax.Array,
    t_idx: jax.Array | int,
    action_dim: int = 4,
) -> jax.Array:
    """Goal-independent closed-loop retention sweep for t < 88 so obs_{20..88} has zero Goal-B mutual information."""
    tf = jnp.asarray(t_idx, dtype=jnp.float32)
    sweep_xy = jnp.stack(
        [
            -0.46 + 0.16 * jnp.cos(0.23 * tf),
            0.30 * jnp.sin(0.19 * tf),
        ],
        axis=-1,
    )
    u_xy = jnp.clip((sweep_xy - ee_xy) * 2.8, -0.75, 0.75)
    pad = jnp.zeros((ee_xy.shape[0], max(0, action_dim - 2)), dtype=jnp.float32)
    return jnp.concatenate([u_xy, pad], axis=-1)


def eval_beyond_window_T160_distractor_switch_v8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    seed: int = 202802,
    n_worlds: int = 64,
    return_obs_trace: bool = False,
) -> dict[str, Any]:
    """Panel W8A-2: Shortcut-free T=160 (>2.5x W=64) Delayed & Distracted Goal-Switch + Counterfactual Swap."""
    total_t = 160
    sw_worlds = init_shortcut_free_t160_switch_worlds(jax.random.key(seed), batch_size=n_worlds)
    goal_a = sw_worlds["targ_a"]
    goal_b = sw_worlds["targ_b"]
    goal_b_prime = sw_worlds["targ_b_prime"]
    false_distractor_xy = sw_worlds["false_distractor_xy"]
    reset_ee_xy = sw_worlds["reset_ee_xy"]
    reset_obj_xy = sw_worlds["reset_obj_xy"]
    init_state = {
        k: v
        for k, v in sw_worlds.items()
        if k not in ("reset_ee_xy", "reset_obj_xy", "targ_b_prime", "false_distractor_xy")
    }

    tok_ga = encode_goal_tokens_2d(goal_a, is_distractor=False)
    tok_dist = encode_goal_tokens_2d(false_distractor_xy, is_distractor=True)
    tok_query = encode_goal_query_tokens((n_worlds,))
    zero_ext_g = jnp.zeros((n_worlds, 2), dtype=jnp.float32)
    w6_iv = interventions.as_wave6()
    mems_init = init_streaming_memories(spec, n_worlds, cache_len=total_t + 4)

    @jax.jit
    def _run_t160(p: dict[str, jax.Array], instructed_gb: jax.Array, alt_gb: jax.Array):
        tok_gb_in = encode_goal_tokens_2d(instructed_gb, is_distractor=False)

        def body_fn(carry, t_idx):
            st, mems, prev_u, prev_wp, occ_st, prev_vobj, contact_c, rk = carry
            rk, pk = jax.random.split(rk)

            is_ga = t_idx < 4
            is_gb = (t_idx >= 16) & (t_idx < 20)
            is_dist = ((t_idx >= 48) & (t_idx < 52)) | ((t_idx >= 76) & (t_idx < 80))

            show_g = jnp.broadcast_to(is_ga | is_gb, (n_worlds,))
            shown_xy = jnp.where(is_ga, goal_a, jnp.where(is_gb, instructed_gb, zero_ext_g))
            dist_xy = jnp.where(is_dist, false_distractor_xy, jnp.full_like(false_distractor_xy, 9.0))
            txt_t = jnp.where(
                is_ga,
                tok_ga,
                jnp.where(is_gb, tok_gb_in, jnp.where(is_dist, tok_dist, tok_query)),
            )

            is_perturb = t_idx == 88
            st_eff = {
                **st,
                "ee_xy": jnp.where(is_perturb, reset_ee_xy, st["ee_xy"]),
                "ee_vel": jnp.where(is_perturb, jnp.zeros_like(st["ee_vel"]), st["ee_vel"]),
                "obj_xy": jnp.where(is_perturb, reset_obj_xy, st["obj_xy"]),
                "obj_vel": jnp.where(is_perturb, jnp.zeros_like(st["obj_vel"]), st["obj_vel"]),
            }
            contact_eff = jnp.where(is_perturb, jnp.zeros_like(contact_c), contact_c)
            prev_u_eff = jnp.where(is_perturb, jnp.zeros_like(prev_u), prev_u)
            prev_wp_eff = jnp.where(is_perturb, jnp.zeros_like(prev_wp), prev_wp)
            occ_eff = jnp.where(is_perturb, jnp.zeros_like(occ_st), occ_st)
            prev_vobj_eff = jnp.where(is_perturb, jnp.zeros_like(prev_vobj), prev_vobj)

            vid = render_unseen_world_rgb(st_eff, shown_xy, show_g, distractor_xy=dist_xy)
            ee_prop = mask_privileged_proprio(st_eff["ee_xy"], st_eff["ee_vel"], contact_eff)
            tb = _build_tick_batch_v7(vid, ee_prop, prev_u_eff, zero_ext_g, txt_t)
            z_seq, next_mems, _ = forward_world_model_wave8a(
                p, spec, tb, interventions, memories=mems, offset=t_idx
            )
            z_t = z_seq[:, 0]
            pure_vp_ti = encode_pure_visual_patches(p, spec, vid, include_pos=False)
            pure_v8 = encode_pure_visual_grid_8x8(p, spec, vid)
            _, vis_obj_xy, _ = decode_spatial_vis_pose(
                p, z_t, w6_iv, pure_v_patches=pure_vp_ti, pure_v_8x8=pure_v8
            )
            vis_obj_vel = jnp.where(
                (t_idx > 0) & (~is_perturb), jnp.clip(vis_obj_xy - prev_vobj_eff, -0.14, 0.14), 0.0
            )
            hl8 = decode_spatial_hazard_grid_8x8(
                p, z_t, w6_iv, pure_v_ti=pure_vp_ti, pure_v_8x8=pure_v8
            )
            cue_free = ~(is_ga | is_gb | is_dist)
            occ_cand = update_self_filtered_occ_8x8(
                occ_eff, hl8, st_eff["ee_xy"], vis_obj_xy, (t_idx == 4) | is_perturb
            )
            occ_next = jnp.where(cue_free, occ_cand, occ_eff)
            resolved_targ = decode_goal_latch_wave8a(p, z_t, interventions)

            def _forced_branch(_):
                return (
                    forced_retention_sweep_action(st_eff["ee_xy"], t_idx, spec.action_dim),
                    jnp.zeros_like(prev_wp_eff),
                )

            def _planned_branch(_):
                u_p, mppi_d = plan_spatial_geometry_free_mppi_v8a(
                    p,
                    spec,
                    z_t,
                    interventions,
                    zero_ext_g,
                    st_eff["ee_xy"],
                    pk,
                    pure_v_patches=pure_vp_ti,
                    pure_v_ti=pure_vp_ti,
                    pure_v_8x8=pure_v8,
                    occ_8x8=occ_next,
                    vis_obj_vel=vis_obj_vel,
                    prev_wp=prev_wp_eff,
                    has_prev_wp=(t_idx > 88),
                    horizon=6,
                    num_candidates=17,
                )
                return u_p, mppi_d["best_wp"]

            u_t, next_wp = jax.lax.cond(t_idx < 88, _forced_branch, _planned_branch, operand=None)
            st_next, c_next = step_unseen_layout_world(st_eff, u_t)
            err_latch_b = jnp.mean((resolved_targ - instructed_gb) ** 2, axis=-1)
            dist_b = jnp.sqrt(jnp.sum((st_next["obj_xy"] - instructed_gb) ** 2, axis=-1) + 1e-8)
            dist_alt = jnp.sqrt(jnp.sum((st_next["obj_xy"] - alt_gb) ** 2, axis=-1) + 1e-8)
            dist_a = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_a) ** 2, axis=-1) + 1e-8)
            obs_step = (
                (vid, ee_prop, prev_u_eff, txt_t)
                if return_obs_trace
                else (jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.int32(0))
            )
            return (st_next, next_mems, u_t, next_wp, occ_next, vis_obj_xy, c_next, rk), (
                dist_b,
                dist_alt,
                dist_a,
                err_latch_b,
                obs_step,
            )

        _, (db_hist, dalt_hist, da_hist, latch_mse_hist, obs_hist) = jax.lax.scan(
            body_fn,
            (
                init_state,
                mems_init,
                jnp.zeros((n_worlds, spec.action_dim), dtype=jnp.float32),
                jnp.zeros((n_worlds, 2), dtype=jnp.float32),
                jnp.zeros((n_worlds, 8, 8), dtype=jnp.float32),
                jnp.zeros((n_worlds, 2), dtype=jnp.float32),
                jnp.zeros((n_worlds,), dtype=jnp.float32),
                jax.random.key(seed + 99),
            ),
            jnp.arange(total_t, dtype=jnp.int32),
        )
        succ_b = (jnp.min(db_hist[-15:], axis=0) < 0.25).astype(jnp.float32)
        leak_alt = (jnp.min(dalt_hist[-15:], axis=0) < 0.25).astype(jnp.float32)
        stale_a = (da_hist[-1] < 0.25).astype(jnp.float32)
        pre_w64_mse = jnp.mean(latch_mse_hist[24:64], axis=0)
        post_w64_mse = jnp.mean(latch_mse_hist[92:160], axis=0)
        return succ_b, leak_alt, stale_a, pre_w64_mse, post_w64_mse, obs_hist

    succ_b, _, stale_a, pre_mse, post_mse, obs_b = jax.device_get(_run_t160(params, goal_b, goal_b_prime))
    succ_bp, leak_to_b, _, _, _, obs_bp = jax.device_get(_run_t160(params, goal_b_prime, goal_b))
    paired_causal = np.asarray(succ_b) * np.asarray(succ_bp)
    out: dict[str, Any] = {
        "T160_post_perturbation_goal_b_success": float(np.mean(succ_b)),
        "T160_succ_vec": [float(x) for x in succ_b],
        "T160_counterfactual_goal_b_prime_success": float(np.mean(succ_bp)),
        "T160_paired_causal_switch_rate": float(np.mean(paired_causal)),
        "T160_paired_causal_switch_vec": [float(x) for x in paired_causal],
        "T160_uninstructed_goal_b_leak_rate": float(np.mean(leak_to_b)),
        "T160_stale_goal_a_capture": float(np.mean(stale_a)),
        "latch_mse_inside_W64_t24_64": float(np.mean(pre_mse)),
        "latch_mse_beyond_W64_t92_160": float(np.mean(post_mse)),
    }
    if return_obs_trace:
        out["obs_trace_b"] = {
            "video": np.swapaxes(np.asarray(obs_b[0]), 0, 1),
            "proprio": np.swapaxes(np.asarray(obs_b[1]), 0, 1),
            "action": np.swapaxes(np.asarray(obs_b[2]), 0, 1),
            "text": np.swapaxes(np.asarray(obs_b[3]), 0, 1),
        }
        out["obs_trace_bp"] = {
            "video": np.swapaxes(np.asarray(obs_bp[0]), 0, 1),
            "proprio": np.swapaxes(np.asarray(obs_bp[1]), 0, 1),
            "action": np.swapaxes(np.asarray(obs_bp[2]), 0, 1),
            "text": np.swapaxes(np.asarray(obs_bp[3]), 0, 1),
        }
    return out


def _render_dynamic_visual_stream(seed: int, n_episodes: int) -> jax.Array:
    st = init_unseen_layout_worlds(jax.random.key(seed), batch_size=n_episodes)
    vids = []
    cur = st
    for t in range(32):
        ang = 0.35 * float(t)
        u = jnp.broadcast_to(jnp.array([np.cos(ang), np.sin(ang), 0.0, 0.0], dtype=jnp.float32), (n_episodes, 4))
        cur, _ = step_unseen_layout_world(cur, u)
        vids.append(render_unseen_world_rgb(cur, st["targ_a"], jnp.ones((n_episodes,), dtype=bool)))
    return jnp.stack(vids, axis=1)  # [B, 32, 64, 64, 3]


def eval_symbolic_key_drift_and_recall_v8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    seed: int = 202803,
    n_episodes: int = 32,
) -> dict[str, Any]:
    """Panel W8A-3: Mechanistic Symbolic-Key Cosine Drift + Dynamic-Visual Overwrite Recall up to T=1024."""
    vid_dyn = _render_dynamic_visual_stream(seed, n_episodes)
    vid_static = jnp.broadcast_to(vid_dyn[:, :1], vid_dyn.shape)

    fixed_txt = jnp.broadcast_to(jnp.array([42, 108, 257, 519], dtype=jnp.int32), (n_episodes, 32, 4))
    b_dyn = {
        "video": vid_dyn,
        "audio": jnp.zeros((n_episodes, 32, 64), dtype=jnp.float32),
        "proprio": jnp.zeros((n_episodes, 32, 16), dtype=jnp.float32),
        "text": fixed_txt,
        "action": jnp.zeros((n_episodes, 32, 4), dtype=jnp.float32),
        "goal": jnp.zeros((n_episodes, 32, 4), dtype=jnp.float32),
        "modality_mask": jnp.ones((n_episodes, 32, 4), dtype=jnp.float32),
        "valid": jnp.ones((n_episodes, 32), dtype=bool),
    }
    b_stat = {**b_dyn, "video": vid_static}

    z_dyn, _, sk_dyn = forward_world_model_wave8a(params, spec, b_dyn, interventions, return_symbolic_keys=True)
    z_stat, _, sk_stat = forward_world_model_wave8a(params, spec, b_stat, interventions, return_symbolic_keys=True)

    if sk_dyn is not None and sk_stat is not None:
        cos_vs_static = jnp.sum(sk_dyn[:, 1:] * sk_stat[:, 1:], axis=-1)
        key_cos = float(jnp.mean(cos_vs_static))
        key_drift = float(1.0 - key_cos)
    else:
        t_dyn = z_dyn[:, 1:, SLOT_TEXT, :].astype(jnp.float32)
        t_stat = z_stat[:, 1:, SLOT_TEXT, :].astype(jnp.float32)
        t_dyn = t_dyn * jax.lax.rsqrt(jnp.sum(t_dyn * t_dyn, axis=-1, keepdims=True) + 1e-6)
        t_stat = t_stat * jax.lax.rsqrt(jnp.sum(t_stat * t_stat, axis=-1, keepdims=True) + 1e-6)
        key_cos = float(jnp.mean(jnp.sum(t_dyn * t_stat, axis=-1)))
        key_drift = float(1.0 - key_cos)

    recall_res: dict[str, Any] = {
        "symbolic_key_cosine_stability": key_cos,
        "symbolic_key_visual_drift": key_drift,
    }
    val_slice = slice(ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    for horizon in (64, 256, 512, 1024):
        total_ep = min(n_episodes, 16 if horizon >= 512 else 32)
        c_bs = 2 if (horizon >= 512 and spec.d_model >= 1152) else min(total_ep, 8)
        eval_spec = (
            spec
            if (spec.arch != "dense_transformer" or horizon <= spec.rope_Window)
            else make_extended_rope_spec(spec, horizon)
        )

        @jax.jit
        def _eval_dyn_assoc(p: dict[str, jax.Array], subk: jax.Array, seq_t: int = horizon, bs_c: int = c_bs, sp: X1BSpec = eval_spec):
            probe = generate_synthetic_world_batch(subk, bs_c, seq_len=seq_t, vocab_size=sp.vocab_size)
            reps = (seq_t + 31) // 32
            dyn_v_full = jnp.tile(vid_dyn[:bs_c], (1, reps, 1, 1, 1))[:, :seq_t]
            probe = de_privilege_proprio_wave8a(
                {
                    **probe,
                    "video": dyn_v_full,
                    "modality_mask": jnp.ones((bs_c, seq_t, 4), dtype=jnp.float32),
                }
            )
            zf, _, _ = forward_world_model_wave8a(p, sp, probe, interventions)
            txt_ctx = (
                zf[:, -1, SLOT_TEXT, :].astype(jnp.float32)
                if interventions.decoupled_heads
                else jnp.mean(zf[:, -1].astype(jnp.float32), axis=1)
            )
            logits = (txt_ctx @ p["head_text_w"].astype(jnp.float32))[:, val_slice]
            return logits, probe["target_text"][:, -1]

        acc_parts = []
        base_k = jax.random.key(seed + 7000 + horizon)
        for ci in range(max(1, total_ep // c_bs)):
            v_log, t_val = jax.device_get(_eval_dyn_assoc(params, jax.random.fold_in(base_k, ci)))
            pred_v = np.argmax(v_log, axis=-1) + ASSOC_VAL_OFFSET
            acc_parts.append((pred_v == t_val).astype(np.float32))
        ep_acc = [float(x) for x in np.concatenate(acc_parts, axis=0)]
        recall_res[f"dyn_visual_assoc_T{horizon}"] = float(np.mean(ep_acc))
        recall_res[f"dyn_visual_assoc_T{horizon}_vec"] = ep_acc

    return recall_res


def make_extended_rope_spec(spec: X1BSpec, horizon: int) -> X1BSpec:
    from dataclasses import replace
    return replace(spec, rope_Window=max(spec.rope_Window, horizon))


def build_multi_item_overwrite_probe_batch(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int,
    num_items_k: int,
    vid_dyn_template: jax.Array,
) -> dict[str, jax.Array]:
    """Construct a `seq_len`-tick dynamic-visual multi-item associative binding + selective overwrite batch.

    Schedule for `K = num_items_k in {1, 2, 4, 8, 16}` and `T = seq_len in {64, 256, 1024}`:
      - Ticks `1 .. K`: write `K` distinct key-value bindings `[0, k_j, v_init_j, 0]` (`k_j in [16..31]`, `v in [64..79]`).
      - Ticks `T//2 .. T//2 + M - 1` (`M = 1` if `K <= 2` else `K // 2`): selectively overwrite the first `M` keys
        to guaranteed-distinct values `v_over_j != v_init_j`, leaving the remaining `K - M` keys untouched.
      - Ticks `T - K .. T - 1`: query all `K` keys `[0, k_j, 0, 2]` while vigorous RGB motion continues.
        At `T = 64`, `dist <= 62 < W = 64` (100% inside `SlidingWindow-W64-GQA` cache).
        At `T = 256, 1024`, both initial writes and mid-horizon overwrites lie strictly beyond `W = 64`.
    """
    if not (1 <= num_items_k <= ASSOC_NUM_KEYS):
        raise ValueError(f"Expected 1 <= num_items_k <= {ASSOC_NUM_KEYS}, got {num_items_k}")
    m_over = 1 if num_items_k <= 2 else (num_items_k // 2)
    t_ov = seq_len // 2
    q_start = seq_len - num_items_k

    k_perm, k_vi, k_vo, k_aud = jax.random.split(rng_key, 4)
    all_keys = jnp.arange(ASSOC_KEY_OFFSET, ASSOC_KEY_OFFSET + ASSOC_NUM_KEYS, dtype=jnp.int32)
    perm_keys = jax.vmap(lambda rk: jax.random.permutation(rk, all_keys))(jax.random.split(k_perm, batch_size))
    v_init = jax.random.randint(k_vi, (batch_size, ASSOC_NUM_KEYS), ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    v_off = jax.random.randint(k_vo, (batch_size, ASSOC_NUM_KEYS), 1, ASSOC_NUM_VALS)
    v_over = ASSOC_VAL_OFFSET + ((v_init - ASSOC_VAL_OFFSET + v_off) % ASSOC_NUM_VALS)

    text = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32)
    target_text = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    stale_text = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    for j in range(num_items_k):
        kj = perm_keys[:, j]
        vij = v_init[:, j]
        text = text.at[:, 1 + j, 1].set(kj).at[:, 1 + j, 2].set(vij)

    for j in range(m_over):
        kj = perm_keys[:, j]
        voj = v_over[:, j]
        text = text.at[:, t_ov + j, 1].set(kj).at[:, t_ov + j, 2].set(voj)

    for j in range(num_items_k):
        kj = perm_keys[:, j]
        expected_v = v_over[:, j] if j < m_over else v_init[:, j]
        qt = q_start + j
        text = text.at[:, qt, 1].set(kj).at[:, qt, 3].set(2)
        target_text = target_text.at[:, qt].set(expected_v)
        stale_text = stale_text.at[:, qt].set(v_init[:, j])

    reps = (seq_len + 31) // 32
    vid = jnp.tile(vid_dyn_template[:batch_size], (1, reps, 1, 1, 1))[:, :seq_len]
    aud = 0.03 * jax.random.normal(k_aud, (batch_size, seq_len, 64), dtype=jnp.float32)
    return {
        "video": vid,
        "audio": aud,
        "proprio": jnp.zeros((batch_size, seq_len, 16), dtype=jnp.float32),
        "text": text,
        "target_text": target_text,
        "stale_text": stale_text,
        "action": jnp.zeros((batch_size, seq_len, 4), dtype=jnp.float32),
        "goal": jnp.zeros((batch_size, seq_len, 4), dtype=jnp.float32),
        "modality_mask": jnp.ones((batch_size, seq_len, 4), dtype=jnp.float32),
        "valid": jnp.ones((batch_size, seq_len), dtype=bool),
        "m_over": jnp.asarray(m_over, dtype=jnp.int32),
    }


def eval_multi_item_overwrite_capacity_v8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    seed: int = 202806,
    n_episodes: int = 24,
    k_items_list: tuple[int, ...] = (1, 2, 4, 8, 16),
    horizons: tuple[int, ...] = (64, 256, 1024),
) -> dict[str, Any]:
    """Panel W8A-6: Multi-item associative binding + selective overwrite capacity curve across K and T."""
    vid_dyn = _render_dynamic_visual_stream(seed, max(n_episodes, 8))
    val_slice = slice(ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    results: dict[str, Any] = {"chance_level": 1.0 / float(ASSOC_NUM_VALS), "curves": {}}

    for horizon in horizons:
        total_ep = min(n_episodes, 16 if horizon >= 512 else n_episodes)
        c_bs = 2 if (horizon >= 512 and spec.d_model >= 1152) else min(total_ep, 8)
        eval_spec = (
            spec
            if (spec.arch != "dense_transformer" or horizon <= spec.rope_Window)
            else make_extended_rope_spec(spec, horizon)
        )
        h_dict: dict[str, Any] = {}

        for k_items in k_items_list:
            m_over = 1 if k_items <= 2 else (k_items // 2)

            @jax.jit
            def _eval_chunk(
                p: dict[str, jax.Array],
                subk: jax.Array,
                seq_t: int = horizon,
                kk: int = k_items,
                bs_c: int = c_bs,
                sp: X1BSpec = eval_spec,
            ):
                probe = build_multi_item_overwrite_probe_batch(subk, bs_c, seq_t, kk, vid_dyn)
                zf, _, _ = forward_world_model_wave8a(p, sp, probe, interventions)
                q_z = zf[:, -kk:, :, :].astype(jnp.float32)
                txt_ctx = q_z[:, :, SLOT_TEXT, :] if interventions.decoupled_heads else jnp.mean(q_z, axis=2)
                logits = (txt_ctx @ p["head_text_w"].astype(jnp.float32))[:, :, val_slice]
                preds = jnp.argmax(logits, axis=-1) + ASSOC_VAL_OFFSET
                targs = probe["target_text"][:, -kk:]
                stales = probe["stale_text"][:, -kk:]
                return preds, targs, stales

            all_corr, ov_corr, ret_corr, ov_stale = [], [], [], []
            base_k = jax.random.key(seed + 1000 * horizon + 37 * k_items)
            for ci in range(max(1, total_ep // c_bs)):
                pr_b, tg_b, st_b = jax.device_get(_eval_chunk(params, jax.random.fold_in(base_k, ci)))
                corr = (pr_b == tg_b).astype(np.float32)  # [bs_c, kk]
                all_corr.append(np.mean(corr, axis=-1))
                ov_corr.append(np.mean(corr[:, :m_over], axis=-1))
                ov_stale.append(np.mean((pr_b[:, :m_over] == st_b[:, :m_over]).astype(np.float32), axis=-1))
                if k_items > m_over:
                    ret_corr.append(np.mean(corr[:, m_over:], axis=-1))

            ep_all = [float(x) for x in np.concatenate(all_corr, axis=0)]
            ep_ov = [float(x) for x in np.concatenate(ov_corr, axis=0)]
            ep_st = [float(x) for x in np.concatenate(ov_stale, axis=0)]
            ep_ret = [float(x) for x in np.concatenate(ret_corr, axis=0)] if ret_corr else None

            entry = {
                "overall_acc": float(np.mean(ep_all)),
                "overwritten_acc": float(np.mean(ep_ov)),
                "retained_acc": float(np.mean(ep_ret)) if ep_ret is not None else None,
                "stale_value_capture_rate": float(np.mean(ep_st)),
                "ep_vec": ep_all,
                "overwritten_vec": ep_ov,
            }
            h_dict[f"K{k_items}"] = entry
            results[f"K{k_items}_T{horizon}_acc"] = entry["overall_acc"]
            results[f"K{k_items}_T{horizon}_overwritten_acc"] = entry["overwritten_acc"]
            results[f"K{k_items}_T{horizon}_vec"] = ep_all
            results[f"K{k_items}_T{horizon}_overwritten_vec"] = ep_ov

        results["curves"][f"T{horizon}"] = h_dict

    return results


def _balanced_binary_ece_10bin(clean_p_pos: np.ndarray, corr_p_pos: np.ndarray, n_bins: int = 10) -> float:
    """Class-balanced 10-bin Expected Calibration Error for `P(mode != 0)` across clean (y=0) & corrupted (y=1) ticks."""
    cp = np.clip(np.asarray(clean_p_pos, dtype=np.float64).reshape(-1), 0.0, 1.0)
    op = np.clip(np.asarray(corr_p_pos, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if cp.size == 0 or op.size == 0:
        return 0.0
    probs = np.concatenate([cp, op], axis=0)
    labels = np.concatenate([np.zeros_like(cp), np.ones_like(op)], axis=0)
    weights = np.concatenate(
        [np.full_like(cp, 0.5 / float(cp.size)), np.full_like(op, 0.5 / float(op.size))],
        axis=0,
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bi in range(n_bins):
        if bi == n_bins - 1:
            m = (probs >= edges[bi]) & (probs <= edges[bi + 1])
        else:
            m = (probs >= edges[bi]) & (probs < edges[bi + 1])
        w_sum = float(np.sum(weights[m]))
        if w_sum > 0.0:
            acc_b = float(np.sum(weights[m] * labels[m]) / w_sum)
            conf_b = float(np.sum(weights[m] * probs[m]) / w_sum)
            ece += w_sum * abs(acc_b - conf_b)
    return float(ece)


def eval_unseen_ood_corruptions_v8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    seed: int = 202804,
    n_worlds: int = 48,
) -> dict[str, Any]:
    """Panel W8A-4: Zero-shot detection AUROC + `PolicyCommitGate` selective routing & ECE across 4 camera corruptions."""
    st = init_unseen_blocked_worlds(jax.random.key(seed), batch_size=n_worlds)
    st = {
        **st,
        "box_hx": jnp.zeros_like(st["box_hx"]),
        "box_hy": jnp.zeros_like(st["box_hy"]),
    }
    vids, ee_props, acts = [], [], []
    cur = st
    c_scal = jnp.zeros((n_worlds,), dtype=jnp.float32)
    phase = jnp.linspace(0.0, 2.0 * np.pi, n_worlds, endpoint=False, dtype=jnp.float32)
    for t in range(24):
        sweep_target = jnp.stack(
            [0.64 * jnp.cos(phase + 0.35 * t), -0.22 - 0.45 * jnp.abs(jnp.sin(phase + 0.28 * t))],
            axis=-1,
        )
        u = jnp.concatenate([jnp.clip(3.2 * (sweep_target - cur["ee_xy"]), -1.0, 1.0), jnp.zeros((n_worlds, 2))], axis=-1)
        vids.append(render_unseen_world_rgb(cur, st["targ_a"], jnp.broadcast_to(t < 4, (n_worlds,))))
        ee_props.append(mask_privileged_proprio(cur["ee_xy"], cur["ee_vel"], c_scal))
        acts.append(u)
        cur, c_scal = step_unseen_layout_world(cur, u)

    clean_vid = jnp.stack(vids, axis=1)
    prop_seq = jnp.stack(ee_props, axis=1)
    act_seq = jnp.stack(acts, axis=1)
    prev_act = jnp.concatenate([jnp.zeros((n_worlds, 1, 4)), act_seq[:, :-1]], axis=1)
    goal_seq = jnp.concatenate(
        [jnp.zeros((n_worlds, 24, 2)), jnp.broadcast_to(st["targ_a"][:, None, :], (n_worlds, 24, 2))], axis=-1
    )

    @jax.jit
    def _anomaly_and_gate(p: dict[str, jax.Array], v_in: jax.Array):
        b_in = {
            "video": v_in,
            "audio": jnp.zeros((n_worlds, 24, 64), dtype=jnp.float32),
            "proprio": prop_seq,
            "text": jnp.zeros((n_worlds, 24, 4), dtype=jnp.int32),
            "action": prev_act,
            "goal": goal_seq,
            "modality_mask": jnp.ones((n_worlds, 24, 4), dtype=jnp.float32),
            "valid": jnp.ones((n_worlds, 24), dtype=bool),
        }
        z_raw, _, _ = forward_world_model_wave8a(p, spec, b_in, interventions)
        z_mem, z_task = fork_isolated_latent_streams(z_raw, interventions.zmem_isolation)
        z_n1, lv1 = predict_latent_step(p, spec, z_task, act_seq, residual=True)
        hds = decode_world_heads_wave8a(
            p, spec, z_mem, z_task, z_n1, lv1, goal_seq, interventions, obs_ee_proprio=prop_seq, obs_video=v_in
        )
        routed = route_proposals_with_commit_gate(
            hds["commit_logits"],
            hds["action_proposal"],
            hds["action_proposal_trusted"],
            gate_policy="learned_innovation",
        )

        vis_ee_disc = jnp.sqrt(jnp.sum((hds["vis_ee_curr"] - prop_seq[:, :, 0:2]) ** 2, axis=-1) + 1e-6)
        z_pred_shift = jnp.concatenate([z_task[:, :1], z_n1[:, :-1]], axis=1)
        jepa_vis_err = jnp.mean(
            (l2_normalize(z_task[:, :, SLOTS_VISUAL, :]) - l2_normalize(z_pred_shift[:, :, SLOTS_VISUAL, :])) ** 2,
            axis=(-2, -1),
        )
        probs = routed["mode_probs"]
        p_anomaly = 1.0 - probs[:, :, 0]
        raw_t = 3.0 * vis_ee_disc + 1.2 * jepa_vis_err + 0.4 * p_anomaly
        pad = jnp.pad(raw_t, ((0, 0), (3, 0)), mode="edge")
        causal_max = jnp.maximum(
            jnp.maximum(pad[:, 3:], pad[:, 2:-1]),
            jnp.maximum(pad[:, 1:-2], pad[:, :-3]),
        )
        anom = 0.65 * raw_t + 0.35 * causal_max
        act_delta = jnp.sqrt(jnp.sum((routed["committed_action"] - hds["action_proposal"]) ** 2, axis=-1) + 1e-8)
        return anom, routed["mode_id"], p_anomaly, act_delta

    clean_sc_all, clean_md_all, clean_pnc_all, clean_ad_all = jax.device_get(_anomaly_and_gate(params, clean_vid))
    clean_sc = np.asarray(clean_sc_all)[:, 10:]
    clean_md = np.asarray(clean_md_all)[:, 10:]
    clean_pnc = np.asarray(clean_pnc_all)[:, 10:]
    clean_commit_vec = np.mean(clean_md == 0, axis=1)  # [n_worlds]

    results: dict[str, Any] = {
        "clean_world_scores": [[float(v) for v in row] for row in clean_sc],
        "corrupted_world_scores": {},
        "clean_commit_full_rate": float(np.mean(clean_commit_vec)),
        "clean_abstain_rate": float(np.mean(clean_md == 3)),
        "clean_fallback_mitigated_rate": float(np.mean((clean_md == 1) | (clean_md == 2))),
        "clean_routed_action_l2_delta": float(np.mean(np.asarray(clean_ad_all)[:, 10:])),
        "routing_by_corruption": {},
    }
    aurocs = []
    ep_aurocs_per_kind = []
    ood_non_commit_vecs = []
    ood_abstain_rates = []
    ood_fallback_rates = []
    all_corr_pnc = []
    for kind in ("frozen_camera_frame", "cross_episode_replay_swap", "unseen_90deg_rotation", "half_occluded_camera"):
        corr_vid = apply_unseen_ood_sensor_corruption(clean_vid, kind)
        corr_sc_all, corr_md_all, corr_pnc_all, corr_ad_all = jax.device_get(_anomaly_and_gate(params, corr_vid))
        corr_sc = np.asarray(corr_sc_all)[:, 10:]
        corr_md = np.asarray(corr_md_all)[:, 10:]
        corr_pnc = np.asarray(corr_pnc_all)[:, 10:]
        all_corr_pnc.append(corr_pnc)

        results["corrupted_world_scores"][kind] = [[float(v) for v in row] for row in corr_sc]
        auc = _binary_auroc(corr_sc, clean_sc)
        results[f"{kind}_auroc"] = auc
        aurocs.append(auc)
        ep_aurocs_per_kind.append([_binary_auroc(corr_sc[i], clean_sc) for i in range(n_worlds)])

        nc_vec = np.mean(corr_md != 0, axis=1)
        abs_rate = float(np.mean(corr_md == 3))
        fb_rate = float(np.mean((corr_md == 1) | (corr_md == 2)))
        ood_non_commit_vecs.append(nc_vec)
        ood_abstain_rates.append(abs_rate)
        ood_fallback_rates.append(fb_rate)
        results["routing_by_corruption"][kind] = {
            "non_commit_rate": float(np.mean(nc_vec)),
            "abstain_mode3_rate": abs_rate,
            "fallback_mode1_2_rate": fb_rate,
            "mean_routed_action_l2_delta": float(np.mean(np.asarray(corr_ad_all)[:, 10:])),
        }

    mean_ood_nc_vec = np.mean(np.stack(ood_non_commit_vecs, axis=0), axis=0)  # [n_worlds]
    sel_routing_vec = 0.5 * clean_commit_vec + 0.5 * mean_ood_nc_vec
    results["mean_unseen_ood_corruption_auroc"] = float(np.mean(aurocs))
    results["ood_ep_auroc_vec"] = [float(x) for x in np.mean(np.asarray(ep_aurocs_per_kind, dtype=np.float64), axis=0)]
    results["ood_non_commit_rate"] = float(np.mean(mean_ood_nc_vec))
    results["ood_abstain_rate"] = float(np.mean(ood_abstain_rates))
    results["ood_fallback_mitigated_rate"] = float(np.mean(ood_fallback_rates))
    results["commit_selective_routing_acc"] = float(np.mean(sel_routing_vec))
    results["commit_selective_routing_vec"] = [float(x) for x in sel_routing_vec]
    results["commit_binary_ece"] = _balanced_binary_ece_10bin(clean_pnc, np.stack(all_corr_pnc, axis=0))
    return results


def _load_train_color_prior_means(
    train_reference_npz_path: str | Path | None,
    eval_npz_path: Path,
    *,
    require_nonempty: bool = False,
) -> np.ndarray:
    """Compute training-only per-color mean 2D block coordinates `mu_train[color_id]` (`0` eval image/label access)."""
    mu = np.zeros((3, 2), dtype=np.float32)
    for cand in ([Path(train_reference_npz_path)] if train_reference_npz_path else []) + [eval_npz_path]:
        if cand.is_file():
            with np.load(cand, allow_pickle=False) as d_ref:
                if "train_color_id" in d_ref and "train_target_block_xy" in d_ref:
                    tr_col = d_ref["train_color_id"].astype(np.int32)
                    tr_xy = d_ref["train_target_block_xy"].astype(np.float32)
                    if tr_col.shape[0] > 0:
                        for cid in range(3):
                            m = tr_col == cid
                            if np.any(m):
                                mu[cid] = np.mean(tr_xy[m], axis=0)
                        return mu
    prov_json = Path(__file__).resolve().parents[1] / "configs/x1b/wave8a_dataset_provenance.json"
    if prov_json.is_file():
        import json as _json

        prov_doc = _json.loads(prov_json.read_text())
        rbs = prov_doc.get("rows_by_split", {})
        if isinstance(rbs, dict) and isinstance(rbs.get("train"), list) and rbs["train"]:
            tr_rows = [r for r in rbs["train"] if "target_block_xy" in r and "color_id" in r]
        else:
            rows = prov_doc.get("episodes", [])
            tr_rows = [r for r in rows if r.get("split") == "train" and "target_block_xy" in r and "color_id" in r]
        if tr_rows:
            for cid in range(3):
                pts = [r["target_block_xy"] for r in tr_rows if int(r["color_id"]) == cid]
                if pts:
                    mu[cid] = np.mean(np.asarray(pts, dtype=np.float32), axis=0)
            return mu
    if require_nonempty:
        raise FileNotFoundError(
            f"Unable to load non-empty training-split color coordinate prior from {train_reference_npz_path!r}, "
            f"{eval_npz_path}, or {prov_json}"
        )
    return mu


def _symmetric_two_block_grounding_score(
    pred_tr: np.ndarray,
    pred_wc: np.ndarray,
    tb_xy: np.ndarray,
    db_xy: np.ndarray,
) -> tuple[float, float]:
    """Symmetric 2-block target-vs-distractor localization accuracy + simultaneous dual-pair hit.

    Evaluates BOTH whether the instructed-color readout `pred_tr` is closer to `tb_xy` than `db_xy` AND whether the
    same-scene distractor-color readout `pred_wc` is closer to `db_xy` than `tb_xy` (`0.5` on exact ties).
    Any constant or language-blind predictor scores exact `0.50` (`0.0` dual-pair) without any blank-video clamp.
    """
    d_tr_t = float(np.mean((pred_tr - tb_xy) ** 2))
    d_tr_d = float(np.mean((pred_tr - db_xy) ** 2))
    hit_tr = 1.0 if d_tr_t < d_tr_d - 1e-7 else (0.5 if abs(d_tr_t - d_tr_d) <= 1e-7 else 0.0)

    d_wc_d = float(np.mean((pred_wc - db_xy) ** 2))
    d_wc_t = float(np.mean((pred_wc - tb_xy) ** 2))
    hit_wc = 1.0 if d_wc_d < d_wc_t - 1e-7 else (0.5 if abs(d_wc_d - d_wc_t) <= 1e-7 else 0.0)

    dual_hit = 1.0 if (d_tr_t < d_tr_d - 1e-7 and d_wc_d < d_wc_t - 1e-7) else 0.0
    return 0.5 * (hit_tr + hit_wc), dual_hit


def eval_grounded_spoken_robot_v8a(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    npz_path: str | Path,
    *,
    split: str = "vault",
    max_episodes: int = 256,
    train_reference_npz_path: str | Path | None = None,
) -> dict[str, Any]:
    """Panel W8A-5: Physical xArm6 Dynamics + Scene-Dependent Spoken & 16-Token Grounding (`dev` & `vault`)."""
    p = Path(npz_path)
    if not p.is_file():
        return {"available": False}

    with np.load(p, allow_pickle=False) as d:
        pfx = f"{split}_"
        required_keys = (
            "video",
            "spoken_audio",
            "wrong_spoken_audio",
            "wrong_color_spoken_audio",
            "ee_xy",
            "action_xy",
            "length",
            "true_text_ids",
            "wrong_text_ids",
            "wrong_color_text_ids",
            "cf_index",
            "scene_cluster_id",
            "dir_id",
            "color_id",
            "distractor_color_id",
            "target_block_xy",
            "distractor_block_xy",
        )
        missing = [pfx + k for k in required_keys if (pfx + k) not in d]
        if missing:
            raise KeyError(f"Wave 8A robot eval bank {p} is missing required v2 provenance/counterfactual fields: {missing}")
        all_dir = d[pfx + "dir_id"].astype(np.int32)
        all_cf = d[pfx + "cf_index"].astype(np.int64)
        all_true_txt = d[pfx + "true_text_ids"].astype(np.int32)
        all_wrong_txt = d[pfx + "wrong_text_ids"].astype(np.int32)
        all_wcol_txt = d[pfx + "wrong_color_text_ids"].astype(np.int32)
        all_col = d[pfx + "color_id"].astype(np.int32)
        all_dist_col = d[pfx + "distractor_color_id"].astype(np.int32)
        if np.any((all_cf < 0) | (all_cf >= all_dir.shape[0])):
            raise ValueError(f"Invalid cf_index out of bounds in {p}")
        opp_map = np.array([1, 0, 3, 2], dtype=np.int32)
        if not np.array_equal(all_dir[all_cf], opp_map[all_dir]):
            raise ValueError(f"Counterfactual dir_id in {p} violates strict opposite-direction pairing")
        if not np.array_equal(all_wrong_txt, all_true_txt[all_cf]):
            raise ValueError(f"wrong_text_ids in {p} does not match true_text_ids[cf_index]")
        if np.any(all_dist_col == all_col):
            raise ValueError(f"Same-scene wrong-color distractor_color_id matches instructed color_id in {p}")

        n_tot = int(d[pfx + "video"].shape[0])
        n = min(n_tot, max_episodes)
        videos = d[pfx + "video"][:n]
        audios = d[pfx + "spoken_audio"][:n].astype(np.float32)
        aud_wrong = d[pfx + "wrong_spoken_audio"][:n].astype(np.float32)
        aud_wrong_col = d[pfx + "wrong_color_spoken_audio"][:n].astype(np.float32)
        ee_xy = d[pfx + "ee_xy"][:n].astype(np.float32)
        act_xy = d[pfx + "action_xy"][:n].astype(np.float32)
        lengths = d[pfx + "length"][:n].astype(np.int32)
        true_txt16 = all_true_txt[:n]
        wrong_txt16 = all_wrong_txt[:n]
        wrong_col_txt16 = all_wcol_txt[:n]
        col_ids = all_col[:n]
        dist_col_ids = all_dist_col[:n]
        targ_blk_xy = d[pfx + "target_block_xy"][:n].astype(np.float32)
        dist_blk_xy = d[pfx + "distractor_block_xy"][:n].astype(np.float32)
        raw_cluster_ids = d[pfx + "scene_cluster_id"][:n].astype(np.int64)

    if np.any(lengths < 8):
        raise ValueError(f"Wave 8A robot eval bank {p} contains invalid episode length < 8 ticks")
    if not np.allclose(aud_wrong_col[:, 16:32], audios[:, 16:32], atol=1e-6):
        raise ValueError(f"wrong_color_spoken_audio[:, 16:32] does not preserve direction audio in {p}")
    if np.any(np.linalg.norm(targ_blk_xy - dist_blk_xy, axis=-1) < 0.175):
        raise ValueError(f"target_block_xy and distractor_block_xy violate minimum separation in {p}")

    mu_train_color = _load_train_color_prior_means(
        train_reference_npz_path,
        p,
        require_nonempty=(train_reference_npz_path is not None),
    )
    uniq_clusters = sorted(int(c) for c in set(int(x) for x in raw_cluster_ids.tolist()))
    if len(uniq_clusters) < 2:
        raise ValueError(
            f"Disjoint-scene shuffle in {p} requires at least 2 distinct scene_cluster_id groups; got {len(uniq_clusters)}"
        )
    shuf_perm = np.arange(n, dtype=np.int64)
    shuf_dyad_arr = raw_cluster_ids.copy()
    # Partition scene clusters into disjoint dyads (plus one 3-cluster cycle when len(uniq_clusters) is odd)
    # so recipient and donor episodes never cross super-cluster boundaries (exact zero cross-dyad covariance).
    cluster_dyads: list[list[int]] = []
    uc_i = 0
    while uc_i < len(uniq_clusters):
        rem = len(uniq_clusters) - uc_i
        if rem == 3:
            cluster_dyads.append(uniq_clusters[uc_i : uc_i + 3])
            uc_i += 3
        else:
            cluster_dyads.append(uniq_clusters[uc_i : uc_i + 2])
            uc_i += 2
    for dyad_id, cl_grp in enumerate(cluster_dyads):
        k_g = len(cl_grp)
        members = [np.flatnonzero(raw_cluster_ids == cid) for cid in cl_grp]
        for pos_c, rec_idxs in enumerate(members):
            don_idxs = members[(pos_c + 1) % k_g]
            for rank_j, ep_i in enumerate(rec_idxs):
                shuf_perm[int(ep_i)] = int(don_idxs[rank_j % len(don_idxs)])
                shuf_dyad_arr[int(ep_i)] = int(dyad_id)
    videos_shuf = videos[shuf_perm]

    def _tile_16tok(tok16: np.ndarray) -> np.ndarray:
        out = np.zeros((n, 32, 4), dtype=np.int32)
        out[:, 0:4, :] = tok16.reshape(n, 4, 4)
        return out

    txt_true_seq = _tile_16tok(true_txt16)
    txt_wrong_seq = _tile_16tok(wrong_txt16)
    txt_wrong_col_seq = _tile_16tok(wrong_col_txt16)

    ee_vel = np.zeros_like(ee_xy)
    ee_vel[:, 1:] = ee_xy[:, 1:] - ee_xy[:, :-1]
    moving = (np.linalg.norm(ee_vel, axis=-1, keepdims=True) > 0.015).astype(np.float32)
    prop = np.concatenate(
        [
            ee_xy,
            ee_xy,
            ee_vel,
            np.zeros((n, 32, 1), dtype=np.float32),
            moving,
            np.zeros((n, 32, 8), dtype=np.float32),
        ],
        axis=-1,
    )
    act4 = np.concatenate([act_xy, np.zeros_like(act_xy)], axis=-1)
    prev_act = np.concatenate([np.zeros((n, 1, 4), dtype=np.float32), act4[:, :-1]], axis=1)
    valid = np.arange(32)[None, :] < lengths[:, None]

    step1_mm, persist1_mm, step4_mm, persist4_mm, plan_gain_mm = [], [], [], [], []
    disc_hits, disc_aud_only_hits, disc_txt_only_hits = [], [], []
    scene_col_hits, scene_col_aud_hits, scene_col_txt_hits, blind_col_hits = [], [], [], []
    shuf_col_hits, shuf_col_aud_hits = [], []
    flip_col_hits, paired_flip_dual_hits, paired_flip_dual_aud_hits = [], [], []
    prior_col_hits, prior_legacy_asym_hits = [], []
    targ_blk_norm_rmse_list, targ_blk_px64_list = [], []
    scene_cluster_ids: list[int] = []
    bs = 32
    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        sl = slice(i0, i1)
        nb = i1 - i0
        b_true = {
            "video": jnp.asarray(videos[sl]),
            "audio": jnp.asarray(audios[sl]),
            "proprio": jnp.asarray(prop[sl]),
            "text": jnp.asarray(txt_true_seq[sl]),
            "action": jnp.asarray(prev_act[sl]),
            "goal": jnp.zeros((nb, 32, 4), dtype=jnp.float32),
            "modality_mask": jnp.ones((nb, 32, 4), dtype=jnp.float32),
            "valid": jnp.asarray(valid[sl]),
        }

        z_raw, _, _ = forward_world_model_wave8a(params, spec, b_true, interventions)
        z_mem, z_tr = fork_isolated_latent_streams(z_raw, interventions.zmem_isolation)
        u_seq = jnp.asarray(act4[sl])
        zn1, lv1 = predict_latent_step(params, spec, z_tr, u_seq, residual=True)
        h_tr = decode_world_heads_wave8a(
            params,
            spec,
            z_mem,
            z_tr,
            zn1,
            lv1,
            b_true["goal"],
            interventions,
            obs_ee_proprio=b_true["proprio"],
            obs_video=b_true["video"],
        )

        t_init = 32
        vid_init = jnp.broadcast_to(b_true["video"][:, :1], (nb, t_init, 64, 64, 3))
        vid_shuf = jnp.broadcast_to(jnp.asarray(videos_shuf[sl, :1]), (nb, t_init, 64, 64, 3))
        vid_flip = jnp.flip(vid_init, axis=(2, 3))
        vid_blind = jnp.zeros_like(vid_init)
        prop_init = jnp.broadcast_to(b_true["proprio"][:, :1], (nb, t_init, 16))
        zero_aud_init = jnp.zeros((nb, t_init, 64), dtype=jnp.float32)
        zero_txt_init = jnp.zeros((nb, t_init, 4), dtype=jnp.int32)
        mm_full = jnp.ones((nb, t_init, 4), dtype=jnp.float32)
        mm_aud_only = mm_full.at[:, :, 3].set(0.0)
        mm_txt_only = mm_full.at[:, :, 0].set(0.0)
        mm_blind = mm_full.at[:, :, 1].set(0.0)
        pv8_init_last = encode_pure_visual_grid_8x8(params, spec, vid_init[:, -1:])
        pv8_shuf_last = encode_pure_visual_grid_8x8(params, spec, vid_shuf[:, -1:])
        pv8_flip_last = encode_pure_visual_grid_8x8(params, spec, vid_flip[:, -1:])

        def _decode_init_goal_and_block(
            aud_in: jax.Array,
            txt_in: jax.Array,
            mm_in: jax.Array,
            *,
            vid_override: jax.Array | None = None,
            pv8_override: jax.Array | None = None,
            blind_video: bool = False,
        ) -> tuple[np.ndarray, np.ndarray]:
            v_use = vid_blind if blind_video else (vid_init if vid_override is None else vid_override)
            pv8_use = None if blind_video else (pv8_init_last if pv8_override is None else pv8_override)
            b_in = {
                "video": v_use,
                "audio": aud_in,
                "proprio": prop_init,
                "text": txt_in,
                "action": jnp.zeros((nb, t_init, 4), dtype=jnp.float32),
                "goal": jnp.zeros((nb, t_init, 4), dtype=jnp.float32),
                "modality_mask": mm_in,
                "valid": jnp.ones((nb, t_init), dtype=bool),
            }
            zr_in, _, _ = forward_world_model_wave8a(params, spec, b_in, interventions)
            _, zt_in = fork_isolated_latent_streams(zr_in, interventions.zmem_isolation)
            g_last = decode_goal_latch_wave8a(params, zt_in, interventions)[:, -1]
            blk_last = decode_target_block_xy_wave8a(
                params, zt_in[:, -1:], interventions, pure_v_8x8=pv8_use
            )[:, 0]
            g_arr, b_arr = jax.device_get((g_last, blk_last))
            return np.asarray(g_arr), np.asarray(b_arr)

        aud_tr_init = jnp.asarray(audios[sl, :t_init])
        aud_wr_init = jnp.asarray(aud_wrong[sl, :t_init])
        aud_wc_init = jnp.asarray(aud_wrong_col[sl, :t_init])
        txt_tr_init = jnp.asarray(txt_true_seq[sl, :t_init])
        txt_wr_init = jnp.asarray(txt_wrong_seq[sl, :t_init])
        txt_wc_init = jnp.asarray(txt_wrong_col_seq[sl, :t_init])

        goal_tr_init, blk_tr_init = _decode_init_goal_and_block(aud_tr_init, txt_tr_init, mm_full)
        goal_wr_init, _ = _decode_init_goal_and_block(aud_wr_init, txt_wr_init, mm_full)
        _, blk_wc_init = _decode_init_goal_and_block(aud_wc_init, txt_wc_init, mm_full)

        goal_tr_aud, blk_tr_aud = _decode_init_goal_and_block(aud_tr_init, zero_txt_init, mm_aud_only)
        goal_wr_aud, _ = _decode_init_goal_and_block(aud_wr_init, zero_txt_init, mm_aud_only)
        _, blk_wc_aud = _decode_init_goal_and_block(aud_wc_init, zero_txt_init, mm_aud_only)

        goal_tr_txt, blk_tr_txt = _decode_init_goal_and_block(zero_aud_init, txt_tr_init, mm_txt_only)
        goal_wr_txt, _ = _decode_init_goal_and_block(zero_aud_init, txt_wr_init, mm_txt_only)
        _, blk_wc_txt = _decode_init_goal_and_block(zero_aud_init, txt_wc_init, mm_txt_only)

        _, blk_tr_shuf = _decode_init_goal_and_block(
            aud_tr_init, txt_tr_init, mm_full, vid_override=vid_shuf, pv8_override=pv8_shuf_last
        )
        _, blk_wc_shuf = _decode_init_goal_and_block(
            aud_wc_init, txt_wc_init, mm_full, vid_override=vid_shuf, pv8_override=pv8_shuf_last
        )
        _, blk_tr_aud_shuf = _decode_init_goal_and_block(
            aud_tr_init, zero_txt_init, mm_aud_only, vid_override=vid_shuf, pv8_override=pv8_shuf_last
        )
        _, blk_wc_aud_shuf = _decode_init_goal_and_block(
            aud_wc_init, zero_txt_init, mm_aud_only, vid_override=vid_shuf, pv8_override=pv8_shuf_last
        )

        _, blk_tr_flip = _decode_init_goal_and_block(
            aud_tr_init, txt_tr_init, mm_full, vid_override=vid_flip, pv8_override=pv8_flip_last
        )
        _, blk_wc_flip = _decode_init_goal_and_block(
            aud_wc_init, txt_wc_init, mm_full, vid_override=vid_flip, pv8_override=pv8_flip_last
        )
        _, blk_tr_aud_flip = _decode_init_goal_and_block(
            aud_tr_init, zero_txt_init, mm_aud_only, vid_override=vid_flip, pv8_override=pv8_flip_last
        )
        _, blk_wc_aud_flip = _decode_init_goal_and_block(
            aud_wc_init, zero_txt_init, mm_aud_only, vid_override=vid_flip, pv8_override=pv8_flip_last
        )

        _, blk_tr_blind = _decode_init_goal_and_block(aud_tr_init, txt_tr_init, mm_blind, blind_video=True)
        _, blk_wc_blind = _decode_init_goal_and_block(aud_wc_init, txt_wc_init, mm_blind, blind_video=True)

        def _roll4(u_in: jax.Array) -> jax.Array:
            z1, _ = predict_latent_step(params, spec, z_tr[:, :-4], u_in[:, :-4], residual=True)
            z2, _ = predict_latent_step(params, spec, z1, u_in[:, 1:-3], residual=True)
            z3, _ = predict_latent_step(params, spec, z2, u_in[:, 2:-2], residual=True)
            z4, _ = predict_latent_step(params, spec, z3, u_in[:, 3:-1], residual=True)
            hw = params["head_proprio_w"].astype(jnp.float32)
            hb = params["head_proprio_b"].astype(jnp.float32)
            zf4, zf0 = z4.astype(jnp.float32), z_tr[:, :-4].astype(jnp.float32)
            if interventions.zerolag_bypass:
                d4 = (
                    0.75 * (zf4[:, :, SLOT_PROPRIO, :] - zf0[:, :, SLOT_PROPRIO, :])
                    + 0.25 * (jnp.mean(zf4[:, :, SLOTS_VISUAL, :], axis=2) - jnp.mean(zf0[:, :, SLOTS_VISUAL, :], axis=2))
                ) @ hw
                return b_true["proprio"][:, :-4, :2] + d4[:, :, :2]
            return ((0.75 * zf4[:, :, SLOT_PROPRIO, :] + 0.25 * jnp.mean(zf4[:, :, SLOTS_VISUAL, :], axis=2)) @ hw + hb)[:, :, :2]

        pred4_pos = np.asarray(jax.device_get(_roll4(u_seq)))
        pred4_neg = np.asarray(jax.device_get(_roll4(-u_seq)))
        pred1_pos = np.asarray(jax.device_get(h_tr["proprio_next"][:, :-1, :2]))

        ee_b = ee_xy[sl]
        len_b = lengths[sl]
        clust_b = raw_cluster_ids[sl]
        tblk_b = targ_blk_xy[sl]
        dblk_b = dist_blk_xy[sl]
        col_b = col_ids[sl]
        dcol_b = dist_col_ids[sl]
        for bi in range(nb):
            L = int(len_b[bi])
            if L < 8:
                continue
            scene_cluster_ids.append(int(clust_b[bi]))
            e1 = np.sqrt(np.sum((pred1_pos[bi, : L - 1] - ee_b[bi, 1:L]) ** 2, axis=-1)) * PHYSICAL_TABLE_SCALE_MM
            p1 = np.sqrt(np.sum((ee_b[bi, : L - 1] - ee_b[bi, 1:L]) ** 2, axis=-1)) * PHYSICAL_TABLE_SCALE_MM
            step1_mm.append(float(np.mean(e1)))
            persist1_mm.append(float(np.mean(p1)))

            e4 = np.sqrt(np.sum((pred4_pos[bi, : L - 4] - ee_b[bi, 4:L]) ** 2, axis=-1)) * PHYSICAL_TABLE_SCALE_MM
            n4 = np.sqrt(np.sum((pred4_neg[bi, : L - 4] - ee_b[bi, 4:L]) ** 2, axis=-1)) * PHYSICAL_TABLE_SCALE_MM
            p4 = np.sqrt(np.sum((ee_b[bi, : L - 4] - ee_b[bi, 4:L]) ** 2, axis=-1)) * PHYSICAL_TABLE_SCALE_MM
            step4_mm.append(float(np.mean(e4)))
            persist4_mm.append(float(np.mean(p4)))
            plan_gain_mm.append(float(np.mean(n4 - e4)))

            true_disp = np.clip((ee_b[bi, L - 1] - ee_b[bi, 0]) * 1.25, -0.95, 0.95)
            err_tr = float(np.mean((goal_tr_init[bi] - true_disp) ** 2))
            err_wr = float(np.mean((goal_wr_init[bi] - true_disp) ** 2))
            disc_hits.append(1.0 if err_tr < err_wr else (0.5 if abs(err_tr - err_wr) < 1e-7 else 0.0))

            err_tr_a = float(np.mean((goal_tr_aud[bi] - true_disp) ** 2))
            err_wr_a = float(np.mean((goal_wr_aud[bi] - true_disp) ** 2))
            disc_aud_only_hits.append(1.0 if err_tr_a < err_wr_a else (0.5 if abs(err_tr_a - err_wr_a) < 1e-7 else 0.0))

            err_tr_t = float(np.mean((goal_tr_txt[bi] - true_disp) ** 2))
            err_wr_t = float(np.mean((goal_wr_txt[bi] - true_disp) ** 2))
            disc_txt_only_hits.append(1.0 if err_tr_t < err_wr_t else (0.5 if abs(err_tr_t - err_wr_t) < 1e-7 else 0.0))

            tb_xy = tblk_b[bi]
            db_xy = dblk_b[bi]
            b_l2 = float(np.sqrt(np.sum((blk_tr_init[bi] - tb_xy) ** 2)))
            targ_blk_norm_rmse_list.append(b_l2)
            targ_blk_px64_list.append(b_l2 * 32.0)

            sc_hit, dual_orig = _symmetric_two_block_grounding_score(blk_tr_init[bi], blk_wc_init[bi], tb_xy, db_xy)
            sc_aud_hit, dual_aud_orig = _symmetric_two_block_grounding_score(
                blk_tr_aud[bi], blk_wc_aud[bi], tb_xy, db_xy
            )
            sc_txt_hit, _ = _symmetric_two_block_grounding_score(blk_tr_txt[bi], blk_wc_txt[bi], tb_xy, db_xy)
            sc_blind_hit, _ = _symmetric_two_block_grounding_score(blk_tr_blind[bi], blk_wc_blind[bi], tb_xy, db_xy)
            sc_shuf_hit, _ = _symmetric_two_block_grounding_score(blk_tr_shuf[bi], blk_wc_shuf[bi], tb_xy, db_xy)
            sc_shuf_aud_hit, _ = _symmetric_two_block_grounding_score(
                blk_tr_aud_shuf[bi], blk_wc_aud_shuf[bi], tb_xy, db_xy
            )
            sc_flip_hit, dual_flip = _symmetric_two_block_grounding_score(
                blk_tr_flip[bi], blk_wc_flip[bi], -tb_xy, -db_xy
            )
            _, dual_aud_flip = _symmetric_two_block_grounding_score(
                blk_tr_aud_flip[bi], blk_wc_aud_flip[bi], -tb_xy, -db_xy
            )

            mu_tr = mu_train_color[int(np.clip(col_b[bi], 0, 2))]
            mu_wc = mu_train_color[int(np.clip(dcol_b[bi], 0, 2))]
            pr_sym_hit, _ = _symmetric_two_block_grounding_score(mu_tr, mu_wc, tb_xy, db_xy)
            pr_asym_d_tr = float(np.mean((mu_tr - tb_xy) ** 2))
            pr_asym_d_wc = float(np.mean((mu_wc - tb_xy) ** 2))
            pr_asym_hit = (
                1.0
                if pr_asym_d_tr < pr_asym_d_wc - 1e-7
                else (0.5 if abs(pr_asym_d_tr - pr_asym_d_wc) <= 1e-7 else 0.0)
            )

            scene_col_hits.append(sc_hit)
            scene_col_aud_hits.append(sc_aud_hit)
            scene_col_txt_hits.append(sc_txt_hit)
            blind_col_hits.append(sc_blind_hit)
            shuf_col_hits.append(sc_shuf_hit)
            shuf_col_aud_hits.append(sc_shuf_aud_hit)
            flip_col_hits.append(sc_flip_hit)
            paired_flip_dual_hits.append(dual_orig * dual_flip)
            paired_flip_dual_aud_hits.append(dual_aud_orig * dual_aud_flip)
            prior_col_hits.append(pr_sym_hit)
            prior_legacy_asym_hits.append(pr_asym_hit)

    s4_m = float(np.mean(step4_mm))
    p4_m = float(np.mean(persist4_mm))
    norm_rmse_m = float(np.mean(targ_blk_norm_rmse_list))
    return {
        "available": True,
        "split": split,
        "num_episodes": len(step4_mm),
        "physical_scale_mm": PHYSICAL_TABLE_SCALE_MM,
        "step1_mm": float(np.mean(step1_mm)),
        "step1_persistence_mm": float(np.mean(persist1_mm)),
        "step4_mm": s4_m,
        "step4_mm_norm250": float(s4_m * (250.0 / PHYSICAL_TABLE_SCALE_MM)),
        "step4_mm_vec": step4_mm,
        "step4_persistence_mm": p4_m,
        "step4_skill_vs_persistence": float(1.0 - s4_m / max(p4_m, 1e-6)),
        "causal_plan_gain_mm": float(np.mean(plan_gain_mm)),
        "spoken_16tok_discrimination_acc": float(np.mean(disc_hits)),
        "spoken_audio_only_discrimination_acc": float(np.mean(disc_aud_only_hits)),
        "text_16tok_only_discrimination_acc": float(np.mean(disc_txt_only_hits)),
        "spoken_scene_color_disc_acc": float(np.mean(scene_col_hits)),
        "spoken_audio_only_scene_color_disc_acc": float(np.mean(scene_col_aud_hits)),
        "text_16tok_only_scene_color_disc_acc": float(np.mean(scene_col_txt_hits)),
        "shuffled_scene_color_disc_acc": float(np.mean(shuf_col_hits)),
        "shuffled_scene_color_audio_only_disc_acc": float(np.mean(shuf_col_aud_hits)),
        "mirrored_180_scene_color_disc_acc": float(np.mean(flip_col_hits)),
        "paired_flip_dual_scene_color_disc_acc": float(np.mean(paired_flip_dual_hits)),
        "paired_flip_dual_audio_only_disc_acc": float(np.mean(paired_flip_dual_aud_hits)),
        "color_prior_scene_color_disc_acc": float(np.mean(prior_col_hits)),
        "color_prior_legacy_asym_disc_acc": float(np.mean(prior_legacy_asym_hits)),
        "color_prior_paired_flip_dual_acc": 0.0,
        "blind_video_scene_color_disc_acc": float(np.mean(blind_col_hits)),
        "target_block_norm_rmse": norm_rmse_m,
        "target_block_rmse": norm_rmse_m,
        "target_block_px64_err": float(np.mean(targ_blk_px64_list)),
        "disc_vec": disc_hits,
        "disc_audio_only_vec": disc_aud_only_hits,
        "disc_text_only_vec": disc_txt_only_hits,
        "scene_color_disc_vec": scene_col_hits,
        "scene_color_audio_only_vec": scene_col_aud_hits,
        "scene_color_text_only_vec": scene_col_txt_hits,
        "shuffled_scene_color_disc_vec": shuf_col_hits,
        "shuffled_scene_color_audio_only_vec": shuf_col_aud_hits,
        "paired_flip_dual_scene_color_vec": paired_flip_dual_hits,
        "paired_flip_dual_audio_only_vec": paired_flip_dual_aud_hits,
        "color_prior_scene_color_vec": prior_col_hits,
        "scene_cluster_ids": scene_cluster_ids,
        "num_unique_scene_clusters": len(set(scene_cluster_ids)),
        "shuffled_dyad_cluster_ids": [int(x) for x in shuf_dyad_arr.tolist()],
        "shuffled_donor_indices": [int(x) for x in shuf_perm.tolist()],
        "num_shuffled_dyad_clusters": len(set(int(x) for x in shuf_dyad_arr.tolist())),
    }


def benchmark_streaming_tick_latency_ms(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    smoke: bool = False,
) -> float:
    """Measure compiled B=1 single-tick causal streaming latency (milliseconds/tick) after XLA warmup."""
    mems = init_streaming_memories(spec, 1, cache_len=64)
    tick_b = {
        "video": jnp.zeros((1, 1, 64, 64, 3), dtype=jnp.uint8),
        "audio": jnp.zeros((1, 1, 64), dtype=jnp.float32),
        "proprio": jnp.zeros((1, 1, 16), dtype=jnp.float32),
        "text": jnp.zeros((1, 1, 4), dtype=jnp.int32),
        "action": jnp.zeros((1, 1, spec.action_dim), dtype=jnp.float32),
        "goal": jnp.zeros((1, spec.goal_dim), dtype=jnp.float32),
        "modality_mask": jnp.ones((1, 1, 4), dtype=jnp.float32),
        "valid": jnp.ones((1, 1), dtype=bool),
    }

    @jax.jit
    def _one_tick(p, m_in, off):
        z, m_out, _ = forward_world_model_wave8a(p, spec, tick_b, interventions, memories=m_in, offset=off)
        return z, m_out

    for wi in range(2 if smoke else 3):
        z_w, mems = _one_tick(params, mems, jnp.asarray(wi, dtype=jnp.int32))
        jax.block_until_ready(z_w)

    n_iters = 4 if smoke else 16
    t0 = time.monotonic()
    for ti in range(n_iters):
        z_w, mems = _one_tick(params, mems, jnp.asarray(10 + ti, dtype=jnp.int32))
    jax.block_until_ready(z_w)
    return round(((time.monotonic() - t0) * 1000.0) / float(n_iters), 4)


def run_wave8a_evaluation(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    seed: int = 2001,
    grounded_npz_path: str | Path | None = None,
    vault_npz_path: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute all 6 Wave 8A Evaluation Panels on canonical fixed eval worlds + latency & state accounting."""
    del seed  # Evaluation worlds are locked to canonical seeds across all training seeds for crossed (seed x world) CIs
    nw = 8 if smoke else 64
    no = 8 if smoke else 48
    nr = 16 if smoke else 256
    nk = 8 if smoke else 24

    ms_per_tick = benchmark_streaming_tick_latency_ms(params, spec, interventions, smoke=smoke)
    p1 = eval_unseen_layouts_deprivileged_mpc_v8a(
        params, spec, interventions, seed=202801, n_worlds=nw, horizon_ticks=(20 if smoke else 116)
    )
    p2 = eval_beyond_window_T160_distractor_switch_v8a(
        params, spec, interventions, seed=202802, n_worlds=(8 if smoke else 64)
    )
    p3 = eval_symbolic_key_drift_and_recall_v8a(
        params, spec, interventions, seed=202803, n_episodes=(8 if smoke else 32)
    )
    p4 = eval_unseen_ood_corruptions_v8a(params, spec, interventions, seed=202804, n_worlds=no)
    p5_dev = (
        eval_grounded_spoken_robot_v8a(
            params,
            spec,
            interventions,
            grounded_npz_path,
            split="dev",
            max_episodes=nr,
            train_reference_npz_path=grounded_npz_path,
        )
        if grounded_npz_path
        else {"available": False}
    )
    p5_vault = (
        eval_grounded_spoken_robot_v8a(
            params,
            spec,
            interventions,
            vault_npz_path,
            split="vault",
            max_episodes=(16 if smoke else 512),
            train_reference_npz_path=grounded_npz_path,
        )
        if vault_npz_path
        else {"available": False}
    )
    p6 = eval_multi_item_overwrite_capacity_v8a(
        params,
        spec,
        interventions,
        seed=202806,
        n_episodes=nk,
        k_items_list=(1, 2, 4, 8, 16),
        horizons=((64, 256) if smoke else (64, 256, 1024)),
    )
    return {
        "streaming_state_memory": compute_streaming_state_bytes(spec, batch_size=1, cache_len=64),
        "compiled_inference_ms_per_tick": ms_per_tick,
        "canonical_eval_seeds": {
            "unseen_layout_mpc": 202801,
            "beyond_window_T160": 202802,
            "symbolic_key_drift_and_recall": 202803,
            "unseen_ood_corruptions": 202804,
            "multi_item_overwrite_capacity": 202806,
        },
        "unseen_layout_deprivileged_mpc": p1,
        "beyond_window_T160_distractor_switch": p2,
        "symbolic_key_drift_and_recall": p3,
        "multi_item_overwrite_capacity": p6,
        "unseen_ood_sensor_corruptions": p4,
        "grounded_spoken_robot_dev": p5_dev,
        "untouched_vault_spoken_robot": p5_vault,
    }
