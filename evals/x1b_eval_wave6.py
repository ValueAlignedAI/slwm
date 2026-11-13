"""Wave 6 Zero-Compromise Grounded Evaluation Suite (`SLWM-v2`).

Implements all five reviewer-clincher evaluation panels with zero geometric planner priors and strict
End-Effector-Only (`ee_only`) proprioception:
  1. `unseen_layout_deprivileged_mpc`: Closed-loop manipulation across 4 unseen obstacle/dynamics families
     (`off_center_wall`, `dual_chicane`, `multi_block_clutter`, `dynamics_shift`) comparing Geometry-Free
     Learned-Cost `Latent-MPPI` vs `Explicit-Goal-Latch + Reactive Controller` vs `Open-Loop Policy`.
  2. `beyond_window_T160_distractor_switch`: T=160 (16.0s > 2.5x W=64) delayed & distracted Goal-Switch
     with a post-window physical perturbation at t=88 (69 ticks after Goal B vanishes > W=64), making early
     coasting impossible and isolating persistent working memory from W=64 sliding-window attention.
  3. `symbolic_key_drift_and_recall`: Direct cosine-drift measurement of GatedDelta symbolic write keys under
     vigorous visual motion + T=64..1024 multi-overwrite associative recall with dynamic visual distractors.
  4. `unseen_ood_sensor_corruptions`: Zero-shot safety detection AUROC & closed-loop regret across 4 unseen
     corruptions (`frozen_camera_frame`, `cross_episode_replay_swap`, `unseen_90deg_rotation`, `half_occluded_camera`).
  5. `grounded_spoken_robot_transfer`: Multi-shard physical xArm6 Cartesian trajectory prediction (mm),
     counterfactual plan gain (mm), and speaker-disjoint real human spoken + 16-token instruction grounding
     on both `dev` and the untouched holdout `vault` (`00147`, `00419`, fresh speakers `33-34`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_multimodal import (
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
from evals.x1b_learned_mpc import (
    plan_explicit_latch_reactive_controller,
    plan_geometry_free_latent_mppi,
    resolve_deprivileged_goal_and_pose,
)
from models.x1b_jepa import predict_latent_step
from models.x1b_wave6_core import (
    Wave6Interventions,
    decode_world_heads_wave6,
    forward_world_model_wave6,
)
from models.x1b_world import (
    SLOT_PROPRIO,
    SLOT_TEXT,
    SLOTS_REGISTER,
    SLOTS_VISUAL,
    X1BSpec,
    init_streaming_memories,
)

PHYSICAL_TABLE_SCALE_MM = 350.0


def _binary_auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    pos = np.asarray(scores_pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(scores_neg, dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)))


def _build_tick_batch(
    vid_t: jax.Array,
    ee_prop_t: jax.Array,
    prev_act: jax.Array,
    ext_goal_xy: jax.Array,
    *,
    aud_t: jax.Array | None = None,
    txt_t: jax.Array | None = None,
) -> dict[str, jax.Array]:
    b = vid_t.shape[0]
    aud = jnp.zeros((b, 1, 64), dtype=jnp.float32) if aud_t is None else aud_t[:, None, :]
    txt = jnp.zeros((b, 1, 4), dtype=jnp.int32) if txt_t is None else txt_t[:, None, :]
    g4 = jnp.concatenate([jnp.zeros_like(ext_goal_xy), ext_goal_xy], axis=-1)[:, None, :]
    return {
        "video": vid_t[:, None, :, :, :],
        "audio": aud,
        "proprio": ee_prop_t[:, None, :],
        "text": txt,
        "action": prev_act[:, None, :],
        "goal": g4,
        "modality_mask": jnp.ones((b, 1, 4), dtype=jnp.float32),
        "valid": jnp.ones((b, 1), dtype=bool),
    }


def eval_unseen_layouts_deprivileged_mpc(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave6Interventions,
    *,
    seed: int = 202601,
    n_worlds: int = 64,
    horizon_ticks: int = 56,
) -> dict[str, Any]:
    """Panel W6-1: Closed-loop control across 4 unseen obstacle/dynamics layouts under `ee_only` proprioception."""
    init_state = init_unseen_layout_worlds(jax.random.key(seed), batch_size=n_worlds)
    goal_xy = init_state["targ_a"]

    def _run_mode(mode: str) -> dict[str, np.ndarray]:
        mems_init = init_streaming_memories(spec, n_worlds, cache_len=max(64, horizon_ticks + 4))
        prev_u0 = jnp.zeros((n_worlds, spec.action_dim), dtype=jnp.float32)
        latch0 = jnp.zeros((n_worlds, 2), dtype=jnp.float32)
        c0 = jnp.zeros((n_worlds,), dtype=jnp.float32)

        @jax.jit
        def _rollout(p: dict[str, jax.Array]):
            def body_fn(carry, t_idx):
                st, mems, prev_u, latch_st, contact_c, rk = carry
                rk, pk = jax.random.split(rk)
                show_g = jnp.broadcast_to(t_idx < 6, (n_worlds,))
                ext_g = jnp.where(show_g[:, None], goal_xy, 0.0)
                vid = render_unseen_world_rgb(st, goal_xy, show_g)
                ee_prop = mask_privileged_proprio(st["ee_xy"], st["ee_vel"], contact_c)
                tb = _build_tick_batch(vid, ee_prop, prev_u, ext_g)
                z_seq, next_mems, _ = forward_world_model_wave6(
                    p, spec, tb, interventions, memories=mems, offset=t_idx
                )
                z_t = z_seq[:, 0]
                vis_obj_xy, _ = resolve_deprivileged_goal_and_pose(p, z_t, ext_g, st["ee_xy"])

                if mode == "geometry_free_mppi":
                    u_t, _ = plan_geometry_free_latent_mppi(
                        p,
                        spec,
                        z_t,
                        ext_g,
                        st["ee_xy"],
                        pk,
                        horizon=8,
                        num_candidates=48,
                        num_elites=8,
                        cem_iters=3,
                        residual=interventions.zerolag_bypass,
                    )
                    next_latch = latch_st
                elif mode == "explicit_latch_reactive":
                    u_t, next_latch = plan_explicit_latch_reactive_controller(
                        st["ee_xy"], vis_obj_xy, ext_g, latch_st
                    )
                else:
                    z_n1, lv1 = predict_latent_step(p, spec, z_seq, prev_u[:, None, :], residual=True)
                    hds = decode_world_heads_wave6(
                        p, spec, z_seq, z_n1, lv1, tb["goal"], interventions, obs_ee_proprio=tb["proprio"]
                    )
                    u_t = hds["action_proposal"][:, 0]
                    next_latch = latch_st

                st_next, c_next = step_unseen_layout_world(st, u_t)
                obj_err = jnp.sqrt(jnp.sum((vis_obj_xy - st["obj_xy"]) ** 2, axis=-1) + 1e-8)
                dist_goal = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_xy) ** 2, axis=-1) + 1e-8)
                return (st_next, next_mems, u_t, next_latch, c_next, rk), (dist_goal, obj_err)

            (st_fin, _, _, _, _, _), (dist_hist, obj_err_hist) = jax.lax.scan(
                body_fn,
                (init_state, mems_init, prev_u0, latch0, c0, jax.random.key(seed + 17)),
                jnp.arange(horizon_ticks, dtype=jnp.int32),
            )
            min_dist = jnp.min(dist_hist[-10:], axis=0)
            succ = (min_dist < 0.24).astype(jnp.float32)
            return succ, jnp.mean(obj_err_hist, axis=0), st_fin["layout_id"]

        s_arr, e_arr, lid_arr = jax.device_get(_rollout(params))
        return {"succ": np.asarray(s_arr), "vis_obj_err": np.asarray(e_arr), "layout": np.asarray(lid_arr)}

    mppi_res = _run_mode("geometry_free_mppi")
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
            "mppi_success": float(np.mean(mppi_res["succ"][m])),
            "explicit_latch_reactive_success": float(np.mean(react_res["succ"][m])),
            "open_loop_success": float(np.mean(open_res["succ"][m])),
        }

    return {
        "geometry_free_mppi_success": float(np.mean(mppi_res["succ"])),
        "geometry_free_mppi_succ_vec": [float(x) for x in mppi_res["succ"]],
        "explicit_latch_reactive_success": float(np.mean(react_res["succ"])),
        "explicit_latch_reactive_succ_vec": [float(x) for x in react_res["succ"]],
        "open_loop_policy_success": float(np.mean(open_res["succ"])),
        "vis_decoded_obj_rmse": float(np.mean(mppi_res["vis_obj_err"])),
        "by_unseen_family": by_family,
    }


def eval_beyond_window_T160_distractor_switch(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave6Interventions,
    *,
    seed: int = 202602,
    n_worlds: int = 64,
) -> dict[str, Any]:
    """Panel W6-2: T=160 (16.0s >> W=64) Delayed & Distracted Goal-Switch with post-window perturbation at t=88."""
    total_t = 160
    init_state = init_unseen_layout_worlds(jax.random.key(seed), batch_size=n_worlds)
    # Use clean/shifted single-obstacle dynamics so memory retention + overwrite is cleanly isolated from S-chicanes
    init_state = {
        **init_state,
        "box_hx": jnp.zeros_like(init_state["box_hx"]),
        "box_hy": jnp.zeros_like(init_state["box_hy"]),
        "mass_scale": jnp.ones((n_worlds,), dtype=jnp.float32),
        "damping": jnp.full((n_worlds,), 0.74, dtype=jnp.float32),
    }
    goal_a = init_state["targ_a"]
    goal_b = init_state["targ_b"]
    false_distractor_xy = jnp.stack([-goal_b[:, 0], goal_b[:, 1]], axis=-1)
    mems_init = init_streaming_memories(spec, n_worlds, cache_len=total_t + 4)

    @jax.jit
    def _run_t160(p: dict[str, jax.Array]):
        def body_fn(carry, t_idx):
            st, mems, prev_u, contact_c, rk = carry
            rk, pk = jax.random.split(rk)

            # Goal A flashed at t=0..3; Goal B overwrite flashed at t=16..19; ZERO goal cue for t=20..159 (140 ticks > 2x W=64!)
            is_ga = t_idx < 4
            is_gb = (t_idx >= 16) & (t_idx < 20)
            show_g = jnp.broadcast_to(is_ga | is_gb, (n_worlds,))
            shown_xy = jnp.where(is_ga, goal_a, goal_b)
            ext_g = jnp.where((is_ga | is_gb), shown_xy, 0.0)

            # Magenta visual distractor flashes at t=48..51 and t=76..79 (must NOT overwrite Goal B!)
            is_dist = ((t_idx >= 48) & (t_idx < 52)) | ((t_idx >= 76) & (t_idx < 80))
            dist_xy = jnp.where(is_dist, false_distractor_xy, jnp.full_like(false_distractor_xy, 9.0))

            # At t=88 (69 ticks > W=64 after Goal B vanished!), displace object away from Goal B back toward center-start
            # so NO finite-window W=64 agent can pass merely by having parked the block on Goal B prior to t=84!
            is_perturb = t_idx == 88
            st_eff = {
                **st,
                "obj_xy": jnp.where(is_perturb, init_state["obj_xy"], st["obj_xy"]),
                "obj_vel": jnp.where(is_perturb, jnp.zeros_like(st["obj_vel"]), st["obj_vel"]),
            }

            vid = render_unseen_world_rgb(st_eff, shown_xy, show_g, distractor_xy=dist_xy)
            ee_prop = mask_privileged_proprio(st_eff["ee_xy"], st_eff["ee_vel"], contact_c)
            tb = _build_tick_batch(vid, ee_prop, prev_u, ext_g)
            z_seq, next_mems, _ = forward_world_model_wave6(
                p, spec, tb, interventions, memories=mems, offset=t_idx
            )
            z_t = z_seq[:, 0]
            _, resolved_targ = resolve_deprivileged_goal_and_pose(p, z_t, ext_g, st_eff["ee_xy"])

            u_t, _ = plan_geometry_free_latent_mppi(
                p,
                spec,
                z_t,
                ext_g,
                st_eff["ee_xy"],
                pk,
                horizon=6,
                num_candidates=32,
                num_elites=6,
                cem_iters=2,
                residual=interventions.zerolag_bypass,
            )
            st_next, c_next = step_unseen_layout_world(st_eff, u_t)
            err_latch_b = jnp.mean((resolved_targ - goal_b) ** 2, axis=-1)
            dist_b = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_b) ** 2, axis=-1) + 1e-8)
            dist_a = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_a) ** 2, axis=-1) + 1e-8)
            return (st_next, next_mems, u_t, c_next, rk), (dist_b, dist_a, err_latch_b)

        _, (db_hist, da_hist, latch_mse_hist) = jax.lax.scan(
            body_fn,
            (
                init_state,
                mems_init,
                jnp.zeros((n_worlds, spec.action_dim), dtype=jnp.float32),
                jnp.zeros((n_worlds,), dtype=jnp.float32),
                jax.random.key(seed + 99),
            ),
            jnp.arange(total_t, dtype=jnp.int32),
        )
        # Must re-deliver block to Goal B in the post-perturbation window (t=145..159, >125 ticks after Goal B cue!)
        post_min_b = jnp.min(db_hist[-15:], axis=0)
        succ_b = (post_min_b < 0.25).astype(jnp.float32)
        stale_a = (da_hist[-1] < 0.25).astype(jnp.float32)
        pre_w64_mse = jnp.mean(latch_mse_hist[24:64], axis=0)
        post_w64_mse = jnp.mean(latch_mse_hist[92:160], axis=0)
        return succ_b, stale_a, pre_w64_mse, post_w64_mse

    succ_b, stale_a, pre_mse, post_mse = jax.device_get(_run_t160(params))
    return {
        "T160_post_perturbation_goal_b_success": float(np.mean(succ_b)),
        "T160_succ_vec": [float(x) for x in succ_b],
        "T160_stale_goal_a_capture": float(np.mean(stale_a)),
        "latch_mse_inside_W64_t24_64": float(np.mean(pre_mse)),
        "latch_mse_beyond_W64_t92_160": float(np.mean(post_mse)),
    }


def eval_symbolic_key_drift_and_recall(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave6Interventions,
    *,
    seed: int = 202603,
    n_episodes: int = 32,
) -> dict[str, Any]:
    """Panel W6-3: Mechanistic Symbolic-Key Cosine Drift + Dynamic-Visual Overwrite Recall up to T=1024."""
    st = init_unseen_layout_worlds(jax.random.key(seed), batch_size=n_episodes)
    # Render 32 frames of vigorous visual motion while holding symbolic text token constant!
    vids = []
    cur = st
    for t in range(32):
        ang = 0.35 * float(t)
        u = jnp.broadcast_to(jnp.array([np.cos(ang), np.sin(ang), 0.0, 0.0], dtype=jnp.float32), (n_episodes, 4))
        cur, _ = step_unseen_layout_world(cur, u)
        vids.append(render_unseen_world_rgb(cur, st["targ_a"], jnp.ones((n_episodes,), dtype=bool)))
    vid_dyn = jnp.stack(vids, axis=1)  # [B, 32, 64, 64, 3]
    vid_static = jnp.broadcast_to(vid_dyn[:, :1], vid_dyn.shape)

    fixed_txt = jnp.broadcast_to(jnp.array([42, 108, 257, 519], dtype=jnp.int32), (n_episodes, 32, 4))
    zero_p = jnp.zeros((n_episodes, 32, 16), dtype=jnp.float32)
    zero_a = jnp.zeros((n_episodes, 32, 64), dtype=jnp.float32)
    zero_u = jnp.zeros((n_episodes, 32, 4), dtype=jnp.float32)
    zero_g = jnp.zeros((n_episodes, 32, 4), dtype=jnp.float32)

    b_dyn = {
        "video": vid_dyn,
        "audio": zero_a,
        "proprio": zero_p,
        "text": fixed_txt,
        "action": zero_u,
        "goal": zero_g,
        "modality_mask": jnp.ones((n_episodes, 32, 4), dtype=jnp.float32),
        "valid": jnp.ones((n_episodes, 32), dtype=bool),
    }
    b_stat = {**b_dyn, "video": vid_static}

    z_dyn, _, sk_dyn = forward_world_model_wave6(params, spec, b_dyn, interventions, return_symbolic_keys=True)
    z_stat, _, sk_stat = forward_world_model_wave6(params, spec, b_stat, interventions, return_symbolic_keys=True)

    if sk_dyn is not None and sk_stat is not None:
        # Cosine similarity between symbolic write keys under dynamic vs static visual scenes for identical text!
        cos_vs_static = jnp.sum(sk_dyn[:, 1:] * sk_stat[:, 1:], axis=-1)
        key_drift = float(1.0 - jnp.mean(cos_vs_static))
        key_cos = float(jnp.mean(cos_vs_static))
    else:
        # For non-GatedDelta architectures, measure SLOT_TEXT representation drift under dynamic vs static video
        t_dyn = z_dyn[:, 1:, SLOT_TEXT, :].astype(jnp.float32)
        t_stat = z_stat[:, 1:, SLOT_TEXT, :].astype(jnp.float32)
        t_dyn = t_dyn * jax.lax.rsqrt(jnp.sum(t_dyn * t_dyn, axis=-1, keepdims=True) + 1e-6)
        t_stat = t_stat * jax.lax.rsqrt(jnp.sum(t_stat * t_stat, axis=-1, keepdims=True) + 1e-6)
        key_cos = float(jnp.mean(jnp.sum(t_dyn * t_stat, axis=-1)))
        key_drift = float(1.0 - key_cos)

    # Dynamic-visual associative overwrite recall at T = 64, 256, 512, 1024 (with moving RGB frames throughout T!)
    recall_res: dict[str, Any] = {
        "symbolic_key_cosine_stability": key_cos,
        "symbolic_key_visual_drift": key_drift,
    }
    val_slice = slice(ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    for horizon in (64, 256, 512, 1024):
        total_ep = min(n_episodes, 16 if horizon >= 512 else 32)
        c_bs = 2 if (horizon >= 512 and spec.d_model >= 1152) else min(total_ep, 8)

        @jax.jit
        def _eval_dyn_assoc(p: dict[str, jax.Array], subk: jax.Array, seq_t: int = horizon, bs_c: int = c_bs):
            probe = generate_synthetic_world_batch(subk, bs_c, seq_len=seq_t, vocab_size=spec.vocab_size)
            reps = (seq_t + 31) // 32
            dyn_v_full = jnp.tile(vid_dyn[:bs_c], (1, reps, 1, 1, 1))[:, :seq_t]
            probe = {
                **probe,
                "video": dyn_v_full,
                "modality_mask": jnp.ones((bs_c, seq_t, 4), dtype=jnp.float32),
            }
            zf, _, _ = forward_world_model_wave6(p, spec, probe, interventions)
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


def eval_unseen_ood_corruptions(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave6Interventions,
    *,
    seed: int = 202604,
    n_worlds: int = 48,
) -> dict[str, Any]:
    """Panel W6-4: Zero-shot detection AUROC across 4 strictly unseen camera corruptions."""
    st = init_unseen_layout_worlds(jax.random.key(seed), batch_size=n_worlds)
    vids, ee_props, acts = [], [], []
    cur = st
    c_scal = jnp.zeros((n_worlds,), dtype=jnp.float32)
    for t in range(24):
        to_o = cur["obj_xy"] - cur["ee_xy"]
        u = jnp.concatenate([jnp.clip(3.0 * to_o, -1.0, 1.0), jnp.zeros((n_worlds, 2))], axis=-1)
        vids.append(render_unseen_world_rgb(cur, st["targ_a"], jnp.broadcast_to(t < 4, (n_worlds,))))
        ee_props.append(mask_privileged_proprio(cur["ee_xy"], cur["ee_vel"], c_scal))
        acts.append(u)
        cur, c_scal = step_unseen_layout_world(cur, u)

    clean_vid = jnp.stack(vids, axis=1)
    prop_seq = jnp.stack(ee_props, axis=1)
    act_seq = jnp.stack(acts, axis=1)
    prev_act = jnp.concatenate([jnp.zeros((n_worlds, 1, 4)), act_seq[:, :-1]], axis=1)
    goal_seq = jnp.concatenate([jnp.zeros((n_worlds, 24, 2)), jnp.broadcast_to(st["targ_a"][:, None, :], (n_worlds, 24, 2))], axis=-1)

    def _anomaly_scores(v_in: jax.Array) -> np.ndarray:
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
        z, _, _ = forward_world_model_wave6(params, spec, b_in, interventions)
        z_n1, lv1 = predict_latent_step(params, spec, z, act_seq, residual=True)
        hds = decode_world_heads_wave6(params, spec, z, z_n1, lv1, goal_seq, interventions, obs_ee_proprio=prop_seq)
        # Combine calibrated visual innovation + visual-vs-joint discrepancy + commit non-clean log-odds
        probs = jax.nn.softmax(hds["commit_logits"], axis=-1)
        p_anomaly = 1.0 - probs[:, :, 0]
        innov_score = hds["innov_feats"][:, :, 0] + 1.5 * hds["innov_feats"][:, :, 4] + 2.0 * p_anomaly
        return np.asarray(jax.device_get(innov_score))

    clean_sc = _anomaly_scores(clean_vid)[:, 10:]
    results: dict[str, Any] = {}
    aurocs = []
    for kind in ("frozen_camera_frame", "cross_episode_replay_swap", "unseen_90deg_rotation", "half_occluded_camera"):
        corr_vid = apply_unseen_ood_sensor_corruption(clean_vid, kind)
        corr_sc = _anomaly_scores(corr_vid)[:, 10:]
        auc = _binary_auroc(corr_sc, clean_sc)
        results[f"{kind}_auroc"] = auc
        aurocs.append(auc)
    results["mean_unseen_ood_corruption_auroc"] = float(np.mean(aurocs))
    return results


def eval_grounded_spoken_robot_npz(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave6Interventions,
    npz_path: str | Path,
    *,
    split: str = "vault",
    max_episodes: int = 256,
) -> dict[str, Any]:
    """Panel W6-5: Physical xArm6 + Speaker-Disjoint Real Human Spoken Audio + Untruncated 16-Token Grounding."""
    p = Path(npz_path)
    if not p.is_file():
        return {"available": False}

    with np.load(p, allow_pickle=False) as d:
        pfx = f"{split}_"
        n_tot = int(d[pfx + "video"].shape[0])
        n = min(n_tot, max_episodes)
        videos = d[pfx + "video"][:n]
        audios = d[pfx + "spoken_audio"][:n].astype(np.float32)
        ee_xy = d[pfx + "ee_xy"][:n].astype(np.float32)
        act_xy = d[pfx + "action_xy"][:n].astype(np.float32)
        lengths = d[pfx + "length"][:n].astype(np.int32)
        true_txt16 = d[pfx + "true_text_ids"][:n].astype(np.int32)
        wrong_txt16 = d[pfx + "wrong_text_ids"][:n].astype(np.int32)

    # Stream untruncated 16-token instructions across ticks t=0..3 (4 ticks x 4 subtokens/tick = 16 tokens, ZERO truncation!)
    def _tile_16tok(tok16: np.ndarray) -> np.ndarray:
        out = np.zeros((n, 32, 4), dtype=np.int32)
        out[:, 0:4, :] = tok16.reshape(n, 4, 4)
        return out

    txt_true_seq = _tile_16tok(true_txt16)
    txt_wrong_seq = _tile_16tok(wrong_txt16)
    aud_wrong = np.roll(audios, shift=7, axis=0)

    ee_vel = np.zeros_like(ee_xy)
    ee_vel[:, 1:] = ee_xy[:, 1:] - ee_xy[:, :-1]
    prop = np.concatenate(
        [ee_xy, ee_xy, ee_vel, np.zeros((n, 32, 1), dtype=np.float32), np.zeros((n, 32, 9), dtype=np.float32)],
        axis=-1,
    )
    act4 = np.concatenate([act_xy, np.zeros_like(act_xy)], axis=-1)
    prev_act = np.concatenate([np.zeros((n, 1, 4), dtype=np.float32), act4[:, :-1]], axis=1)
    valid = (np.arange(32)[None, :] < lengths[:, None])

    step1_mm, persist1_mm, step4_mm, persist4_mm, plan_gain_mm, disc_hits = [], [], [], [], [], []
    bs = 32
    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        sl = slice(i0, i1)
        b_true = {
            "video": jnp.asarray(videos[sl]),
            "audio": jnp.asarray(audios[sl]),
            "proprio": jnp.asarray(prop[sl]),
            "text": jnp.asarray(txt_true_seq[sl]),
            "action": jnp.asarray(prev_act[sl]),
            "goal": jnp.zeros((i1 - i0, 32, 4), dtype=jnp.float32),
            "modality_mask": jnp.ones((i1 - i0, 32, 4), dtype=jnp.float32),
            "valid": jnp.asarray(valid[sl]),
        }
        b_wrong = {
            **b_true,
            "audio": jnp.asarray(aud_wrong[sl]),
            "text": jnp.asarray(txt_wrong_seq[sl]),
        }

        z_tr, _, _ = forward_world_model_wave6(params, spec, b_true, interventions)
        z_wr, _, _ = forward_world_model_wave6(params, spec, b_wrong, interventions)

        u_seq = jnp.asarray(act4[sl])
        zn1, lv1 = predict_latent_step(params, spec, z_tr, u_seq, residual=True)
        h_tr = decode_world_heads_wave6(params, spec, z_tr, zn1, lv1, b_true["goal"], interventions, obs_ee_proprio=b_true["proprio"])
        zn1_w, lvw = predict_latent_step(params, spec, z_wr, u_seq, residual=True)
        h_wr = decode_world_heads_wave6(params, spec, z_wr, zn1_w, lvw, b_wrong["goal"], interventions, obs_ee_proprio=b_wrong["proprio"])

        # 4-step rollout under true motor plan (+u) vs counterfactual negated motor plan (-u)
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
        act_tr_p = np.asarray(jax.device_get(h_tr["action_proposal"][:, :, :2]))
        act_wr_p = np.asarray(jax.device_get(h_wr["action_proposal"][:, :, :2]))
        goal_tr_p = np.asarray(jax.device_get(h_tr["pred_goal_xy"]))
        goal_wr_p = np.asarray(jax.device_get(h_wr["pred_goal_xy"]))

        ee_b = ee_xy[sl]
        act_b = act_xy[sl]
        len_b = lengths[sl]
        for bi in range(i1 - i0):
            L = int(len_b[bi])
            if L < 8:
                continue
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

            # Pairwise grounded command discrimination combining policy action match + final endpoint goal latch match
            end_disp = ee_b[bi, L - 1]
            err_tr = float(np.mean((act_tr_p[bi, 3:L] - act_b[bi, 3:L]) ** 2)) + 0.5 * float(
                np.mean((goal_tr_p[bi, 3:L] - end_disp[None, :]) ** 2)
            )
            err_wr = float(np.mean((act_wr_p[bi, 3:L] - act_b[bi, 3:L]) ** 2)) + 0.5 * float(
                np.mean((goal_wr_p[bi, 3:L] - end_disp[None, :]) ** 2)
            )
            disc_hits.append(1.0 if err_tr < err_wr else (0.5 if abs(err_tr - err_wr) < 1e-7 else 0.0))

    s4_m = float(np.mean(step4_mm))
    p4_m = float(np.mean(persist4_mm))
    return {
        "available": True,
        "split": split,
        "num_episodes": len(step4_mm),
        "step1_mm": float(np.mean(step1_mm)),
        "step1_persistence_mm": float(np.mean(persist1_mm)),
        "step4_mm": s4_m,
        "step4_mm_vec": step4_mm,
        "step4_persistence_mm": p4_m,
        "step4_skill_vs_persistence": float(1.0 - s4_m / max(p4_m, 1e-6)),
        "causal_plan_gain_mm": float(np.mean(plan_gain_mm)),
        "spoken_16tok_discrimination_acc": float(np.mean(disc_hits)),
        "disc_vec": disc_hits,
    }


def run_wave6_evaluation(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave6Interventions,
    *,
    seed: int = 2001,
    grounded_npz_path: str | Path | None = None,
    vault_npz_path: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute all 5 Wave 6 Zero-Compromise Grounded Evaluation Panels."""
    nw = 8 if smoke else 64
    no = 8 if smoke else 48
    nr = 16 if smoke else 256

    p1 = eval_unseen_layouts_deprivileged_mpc(
        params, spec, interventions, seed=seed + 100, n_worlds=nw, horizon_ticks=(16 if smoke else 56)
    )
    p2 = eval_beyond_window_T160_distractor_switch(
        params, spec, interventions, seed=seed + 200, n_worlds=(8 if smoke else 64)
    )
    p3 = eval_symbolic_key_drift_and_recall(
        params, spec, interventions, seed=seed + 300, n_episodes=(8 if smoke else 32)
    )
    p4 = eval_unseen_ood_corruptions(params, spec, interventions, seed=seed + 400, n_worlds=no)
    p5_dev = (
        eval_grounded_spoken_robot_npz(params, spec, interventions, grounded_npz_path, split="dev", max_episodes=nr)
        if grounded_npz_path
        else {"available": False}
    )
    p5_vault = (
        eval_grounded_spoken_robot_npz(
            params, spec, interventions, vault_npz_path, split="vault", max_episodes=(16 if smoke else 512)
        )
        if vault_npz_path
        else {"available": False}
    )
    return {
        "unseen_layout_deprivileged_mpc": p1,
        "beyond_window_T160_distractor_switch": p2,
        "symbolic_key_drift_and_recall": p3,
        "unseen_ood_sensor_corruptions": p4,
        "grounded_spoken_robot_dev": p5_dev,
        "untouched_vault_spoken_robot": p5_vault,
    }
