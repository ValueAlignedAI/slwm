"""Deterministic Qualitative Inference Gallery Exporter & Renderer for Study X2 Wave 8A (`AGENTS.md` §13.2).

Produces uncurated, pre-registered `dev` inference bundles (`showcase_bundle.npz`) and side-by-side
multi-cohort comparison galleries (`inference_showcase/wave8a/dev/`) with:
  1. Pre-registered common cross-model episode indices (`[0, 1, 2, 3]`, spanning all 4 unseen layout families)
     plus deterministic post-evaluation percentile selection (`P05` failure, `P25`, `P50`, `P75`, `P95`)
     ranked by final object-to-goal distance on the median seed of `w8a_slwm_x2_isolated_450m`.
  2. Explicit epistemic channel labels (`[observed]`, `[reconstructed]`, `[predicted]`, `[abstained]`)
     distinguishing diagnostic readout from `PolicyCommitGate`-routed output.
  3. Synchronized multi-cohort comparisons across unseen-layout closed-loop control, `T = 160` (`> 2.5x W = 64`)
     post-perturbation goal-switching, `K = 16` dynamic-visual overwrite at `T = 1024`, unseen OOD camera
     corruption abstention, and speaker-disjoint spoken `xArm6` 4-step forecasting.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_plt():
    if "MPLCONFIGDIR" not in os.environ:
        try:
            os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="slwm_mpl_")
        except OSError:
            pass
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

from data.x1b_multimodal import ASSOC_NUM_VALS, ASSOC_VAL_OFFSET
from data.x1b_unseen_worlds import (
    LAYOUT_DUAL_CHICANE,
    LAYOUT_DYNAMICS_SHIFT,
    LAYOUT_MULTI_CLUTTER,
    LAYOUT_OFF_CENTER,
    apply_unseen_ood_sensor_corruption,
    mask_privileged_proprio,
    render_unseen_world_rgb,
    step_unseen_layout_world,
)
from evals.x1b_eval_wave6 import PHYSICAL_TABLE_SCALE_MM
from evals.x1b_eval_wave7 import (
    _build_tick_batch_v7,
    init_unseen_blocked_worlds,
    update_self_filtered_occ_8x8,
)
from evals.x1b_eval_wave8a import (
    _render_dynamic_visual_stream,
    build_multi_item_overwrite_probe_batch,
    forced_retention_sweep_action,
    init_shortcut_free_t160_switch_worlds,
    make_extended_rope_spec,
    plan_spatial_geometry_free_mppi_v8a,
)
from models.x1b_commit import route_proposals_with_commit_gate
from models.x1b_jepa import l2_normalize, predict_latent_step
from models.x1b_wave8a_core import (
    Wave8AInterventions,
    decode_goal_latch_wave8a,
    decode_spatial_hazard_grid_8x8,
    decode_spatial_vis_pose,
    decode_world_heads_wave8a,
    encode_goal_query_tokens,
    encode_goal_tokens_2d,
    encode_pure_visual_grid_8x8,
    encode_pure_visual_patches,
    fork_isolated_latent_streams,
    forward_world_model_wave8a,
)
from models.x1b_world import (
    SLOT_PROPRIO,
    SLOT_TEXT,
    SLOTS_VISUAL,
    X1BSpec,
    init_streaming_memories,
)

PRE_REGISTERED_COMMON_EPISODES = (0, 1, 2, 3)
LAYOUT_NAMES = {
    int(LAYOUT_OFF_CENTER): "off_center_wall",
    int(LAYOUT_DUAL_CHICANE): "dual_chicane",
    int(LAYOUT_MULTI_CLUTTER): "multi_block_clutter",
    int(LAYOUT_DYNAMICS_SHIFT): "dynamics_shift",
}


def collect_wave8a_showcase_bundle(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    interventions: Wave8AInterventions,
    *,
    cohort: str,
    seed: int = 2001,
    grounded_npz_path: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, np.ndarray]:
    """Collect deterministic `dev` inference traces for `inference_showcase/wave8a/dev/`."""
    n_worlds = 8 if smoke else 64
    horizon_ticks = 20 if smoke else 116
    snap_ticks = np.linspace(0, horizon_ticks - 1, 4, dtype=np.int32)
    w6_iv = interventions.as_wave6()

    # 1. Unseen-layout closed-loop rollout across all `n_worlds` worlds (fixed dev seed 202891)
    init_state = init_unseen_blocked_worlds(jax.random.key(202891), batch_size=n_worlds)
    goal_xy = init_state["targ_a"]
    goal_write_tok = encode_goal_tokens_2d(goal_xy, is_distractor=False)
    goal_query_tok = encode_goal_query_tokens((n_worlds,))
    mems_init = init_streaming_memories(spec, n_worlds, cache_len=max(64, horizon_ticks + 4))

    @jax.jit
    def _roll_unseen(p: dict[str, jax.Array]):
        def body_fn(carry, t_idx):
            st, mems, prev_u, latch_st, occ_st, prev_vobj, contact_c, rk = carry
            rk, pk = jax.random.split(rk)
            show_g = jnp.ones((n_worlds,), dtype=bool)
            txt_t = jnp.where(t_idx < 6, goal_write_tok, goal_query_tok)
            vid = render_unseen_world_rgb(st, goal_xy, show_g)
            ee_prop = mask_privileged_proprio(st["ee_xy"], st["ee_vel"], contact_c)
            tb = _build_tick_batch_v7(vid, ee_prop, prev_u, goal_xy, txt_t)
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
            hl8 = decode_spatial_hazard_grid_8x8(p, z_t, w6_iv, pure_v_ti=pure_vp_ti, pure_v_8x8=pure_v8)
            occ_next = update_self_filtered_occ_8x8(
                occ_st, hl8, st["ee_xy"], vis_obj_xy, t_idx == 0, ext_goal_xy=goal_xy
            )
            u_t, mppi_diag = plan_spatial_geometry_free_mppi_v8a(
                p,
                spec,
                z_t,
                interventions,
                goal_xy,
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
            )
            # Diagnostic frame reconstruction (`[reconstructed]`, stop-gradient head)
            vis_patches_dec = z_t.astype(jnp.float32)[:, SLOTS_VISUAL, :] @ p["diag_vis_patch_w"].astype(jnp.float32)
            diag_rgb = jnp.clip(
                (jnp.swapaxes(vis_patches_dec.reshape(n_worlds, 4, 4, 16, 16, 3), 2, 3).reshape(n_worlds, 64, 64, 3) + 1.0)
                * 127.5,
                0.0,
                255.0,
            ).astype(jnp.uint8)
            st_next, c_next = step_unseen_layout_world(st, u_t)
            dist_g = jnp.sqrt(jnp.sum((st_next["obj_xy"] - goal_xy) ** 2, axis=-1) + 1e-8)
            return (st_next, next_mems, u_t, mppi_diag["best_wp"], occ_next, vis_obj_xy, c_next, rk), (
                st["ee_xy"],
                st["obj_xy"],
                vis_ee_xy,
                vis_obj_xy,
                dist_g,
                vid,
                diag_rgb,
                occ_next,
            )

        _, outs = jax.lax.scan(
            body_fn,
            (
                init_state,
                mems_init,
                jnp.zeros((n_worlds, spec.action_dim), dtype=jnp.float32),
                goal_xy,
                jnp.zeros((n_worlds, 8, 8), dtype=jnp.float32),
                jnp.zeros((n_worlds, 2), dtype=jnp.float32),
                jnp.zeros((n_worlds,), dtype=jnp.float32),
                jax.random.key(202892),
            ),
            jnp.arange(horizon_ticks, dtype=jnp.int32),
        )
        ee_tr, obj_tr, vee_tr, vobj_tr, dist_tr, vid_tr, diag_tr, occ_tr = outs
        snaps_obs = jnp.swapaxes(vid_tr[snap_ticks], 0, 1)  # [N, 4, 64, 64, 3]
        snaps_rec = jnp.swapaxes(diag_tr[snap_ticks], 0, 1)  # [N, 4, 64, 64, 3]
        return (
            jnp.swapaxes(ee_tr, 0, 1),
            jnp.swapaxes(obj_tr, 0, 1),
            jnp.swapaxes(vee_tr, 0, 1),
            jnp.swapaxes(vobj_tr, 0, 1),
            jnp.min(dist_tr[-min(16, horizon_ticks) :], axis=0),
            snaps_obs,
            snaps_rec,
            occ_tr[-1],
        )

    ee_tr, obj_tr, vee_tr, vobj_tr, min_dist, snaps_obs, snaps_rec, occ_fin = jax.device_get(_roll_unseen(params))

    # 2. Shortcut-free beyond-window T=160 Goal-Switch trace on fixed common dev worlds (n=8)
    n_sw = min(8, n_worlds)
    total_t = 36 if smoke else 160
    sw_worlds = init_shortcut_free_t160_switch_worlds(jax.random.key(202893), batch_size=n_sw)
    ga = sw_worlds["targ_a"]
    gb = sw_worlds["targ_b"]
    fdist = sw_worlds["false_distractor_xy"]
    reset_ee_xy = sw_worlds["reset_ee_xy"]
    reset_obj_xy = sw_worlds["reset_obj_xy"]
    sw_init = {
        k: v
        for k, v in sw_worlds.items()
        if k not in ("reset_ee_xy", "reset_obj_xy", "targ_b_prime", "false_distractor_xy")
    }
    tok_ga = encode_goal_tokens_2d(ga, is_distractor=False)
    tok_gb = encode_goal_tokens_2d(gb, is_distractor=False)
    tok_dist = encode_goal_tokens_2d(fdist, is_distractor=True)
    tok_q = encode_goal_query_tokens((n_sw,))
    z_ext = jnp.zeros((n_sw, 2), dtype=jnp.float32)
    mems_sw = init_streaming_memories(spec, n_sw, cache_len=total_t + 4)

    pert_t = 20 if smoke else 88

    @jax.jit
    def _roll_t160(p: dict[str, jax.Array]):
        def body_sw(carry, t_idx):
            st, mems, prev_u, prev_wp, occ_st, prev_vobj, contact_c, rk = carry
            rk, pk = jax.random.split(rk)
            is_ga = t_idx < 4
            is_gb = (t_idx >= 16) & (t_idx < 20)
            is_dist = ((t_idx >= 48) & (t_idx < 52)) | ((t_idx >= 76) & (t_idx < 80))
            show_g = jnp.broadcast_to(is_ga | is_gb, (n_sw,))
            shown_xy = jnp.where(is_ga, ga, jnp.where(is_gb, gb, z_ext))
            dist_xy = jnp.where(is_dist, fdist, jnp.full_like(fdist, 9.0))
            txt_t = jnp.where(is_ga, tok_ga, jnp.where(is_gb, tok_gb, jnp.where(is_dist, tok_dist, tok_q)))
            is_pert = t_idx == pert_t
            st_eff = {
                **st,
                "ee_xy": jnp.where(is_pert, reset_ee_xy, st["ee_xy"]),
                "ee_vel": jnp.where(is_pert, jnp.zeros_like(st["ee_vel"]), st["ee_vel"]),
                "obj_xy": jnp.where(is_pert, reset_obj_xy, st["obj_xy"]),
                "obj_vel": jnp.where(is_pert, jnp.zeros_like(st["obj_vel"]), st["obj_vel"]),
            }
            contact_eff = jnp.where(is_pert, jnp.zeros_like(contact_c), contact_c)
            prev_u_eff = jnp.where(is_pert, jnp.zeros_like(prev_u), prev_u)
            prev_wp_eff = jnp.where(is_pert, jnp.zeros_like(prev_wp), prev_wp)
            occ_eff = jnp.where(is_pert, jnp.zeros_like(occ_st), occ_st)
            prev_vobj_eff = jnp.where(is_pert, jnp.zeros_like(prev_vobj), prev_vobj)

            vid = render_unseen_world_rgb(st_eff, shown_xy, show_g, distractor_xy=dist_xy)
            ee_prop = mask_privileged_proprio(st_eff["ee_xy"], st_eff["ee_vel"], contact_eff)
            tb = _build_tick_batch_v7(vid, ee_prop, prev_u_eff, z_ext, txt_t)
            z_seq, next_mems, _ = forward_world_model_wave8a(
                p, spec, tb, interventions, memories=mems, offset=t_idx
            )
            z_t = z_seq[:, 0]
            pure_vp_ti = encode_pure_visual_patches(p, spec, vid, include_pos=False)
            pure_v8 = encode_pure_visual_grid_8x8(p, spec, vid)
            _, vis_obj_xy, _ = decode_spatial_vis_pose(
                p, z_t, w6_iv, pure_v_patches=pure_vp_ti, pure_v_8x8=pure_v8
            )
            vis_obj_vel = jnp.where((t_idx > 0) & (~is_pert), jnp.clip(vis_obj_xy - prev_vobj_eff, -0.14, 0.14), 0.0)
            hl8 = decode_spatial_hazard_grid_8x8(p, z_t, w6_iv, pure_v_ti=pure_vp_ti, pure_v_8x8=pure_v8)
            occ_cand = update_self_filtered_occ_8x8(
                occ_eff, hl8, st_eff["ee_xy"], vis_obj_xy, (t_idx == 4) | is_pert
            )
            occ_next = jnp.where(~(is_ga | is_gb | is_dist), occ_cand, occ_eff)
            latched = decode_goal_latch_wave8a(p, z_t, interventions)

            def _forced_sw(_):
                return (
                    forced_retention_sweep_action(st_eff["ee_xy"], t_idx, spec.action_dim),
                    jnp.zeros_like(prev_wp_eff),
                )

            def _planned_sw(_):
                u_p, mppi_d = plan_spatial_geometry_free_mppi_v8a(
                    p,
                    spec,
                    z_t,
                    interventions,
                    z_ext,
                    st_eff["ee_xy"],
                    pk,
                    pure_v_patches=pure_vp_ti,
                    pure_v_ti=pure_vp_ti,
                    pure_v_8x8=pure_v8,
                    occ_8x8=occ_next,
                    vis_obj_vel=vis_obj_vel,
                    prev_wp=prev_wp_eff,
                    has_prev_wp=(t_idx > pert_t),
                    horizon=6,
                    num_candidates=17,
                )
                return u_p, mppi_d["best_wp"]

            u_t, next_wp = jax.lax.cond(t_idx < pert_t, _forced_sw, _planned_sw, operand=None)
            st_next, c_next = step_unseen_layout_world(st_eff, u_t)
            return (st_next, next_mems, u_t, next_wp, occ_next, vis_obj_xy, c_next, rk), (
                st_eff["obj_xy"],
                latched,
            )

        _, (sw_obj, sw_lat) = jax.lax.scan(
            body_sw,
            (
                sw_init,
                mems_sw,
                jnp.zeros((n_sw, spec.action_dim), dtype=jnp.float32),
                jnp.zeros((n_sw, 2), dtype=jnp.float32),
                jnp.zeros((n_sw, 8, 8), dtype=jnp.float32),
                jnp.zeros((n_sw, 2), dtype=jnp.float32),
                jnp.zeros((n_sw,), dtype=jnp.float32),
                jax.random.key(202894),
            ),
            jnp.arange(total_t, dtype=jnp.int32),
        )
        return jnp.swapaxes(sw_obj, 0, 1), jnp.swapaxes(sw_lat, 0, 1)

    sw_obj_tr, sw_lat_tr = jax.device_get(_roll_t160(params))

    # 3. K=16 dynamic-visual selective overwrite at T=1024 (T=128 in smoke)
    t_k16 = 128 if smoke else 1024
    vid_dyn = _render_dynamic_visual_stream(202895, 2)
    sp_k16 = (
        spec
        if (spec.arch != "dense_transformer" or t_k16 <= spec.rope_Window)
        else make_extended_rope_spec(spec, t_k16)
    )
    probe_k16 = build_multi_item_overwrite_probe_batch(jax.random.key(202896), 2, t_k16, 16, vid_dyn)
    zf_k16, _, _ = forward_world_model_wave8a(params, sp_k16, probe_k16, interventions)
    qz_k16 = zf_k16[:, -16:, :, :].astype(jnp.float32)
    tctx_k16 = qz_k16[:, :, SLOT_TEXT, :] if interventions.decoupled_heads else jnp.mean(qz_k16, axis=2)
    val_sl = slice(ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    k16_preds = np.asarray(
        jax.device_get(jnp.argmax((tctx_k16 @ params["head_text_w"].astype(jnp.float32))[:, :, val_sl], axis=-1) + ASSOC_VAL_OFFSET)
    )
    k16_targs = np.asarray(jax.device_get(probe_k16["target_text"][:, -16:]))
    k16_stales = np.asarray(jax.device_get(probe_k16["stale_text"][:, -16:]))

    # 4. Unseen OOD corruption (`half_occluded_camera`) epistemic routing trace
    ood_st = init_unseen_blocked_worlds(jax.random.key(202897), batch_size=4)
    ood_st = {**ood_st, "box_hx": jnp.zeros_like(ood_st["box_hx"]), "box_hy": jnp.zeros_like(ood_st["box_hy"])}
    v_clean_l, p_l, u_l = [], [], []
    cur_o = ood_st
    c_o = jnp.zeros((4,), dtype=jnp.float32)
    ph_o = jnp.linspace(0.0, 1.5 * np.pi, 4, endpoint=False, dtype=jnp.float32)
    for t in range(24):
        sw_t = jnp.stack([0.62 * jnp.cos(ph_o + 0.35 * t), -0.22 - 0.42 * jnp.abs(jnp.sin(ph_o + 0.28 * t))], axis=-1)
        u_o = jnp.concatenate([jnp.clip(3.2 * (sw_t - cur_o["ee_xy"]), -1.0, 1.0), jnp.zeros((4, 2))], axis=-1)
        v_clean_l.append(render_unseen_world_rgb(cur_o, ood_st["targ_a"], jnp.broadcast_to(t < 4, (4,))))
        p_l.append(mask_privileged_proprio(cur_o["ee_xy"], cur_o["ee_vel"], c_o))
        u_l.append(u_o)
        cur_o, c_o = step_unseen_layout_world(cur_o, u_o)
    v_clean = jnp.stack(v_clean_l, axis=1)
    v_occ = apply_unseen_ood_sensor_corruption(v_clean, "half_occluded_camera")
    p_seq = jnp.stack(p_l, axis=1)
    u_seq = jnp.stack(u_l, axis=1)
    pu_seq = jnp.concatenate([jnp.zeros((4, 1, 4)), u_seq[:, :-1]], axis=1)
    g_seq = jnp.concatenate([jnp.zeros((4, 24, 2)), jnp.broadcast_to(ood_st["targ_a"][:, None, :], (4, 24, 2))], axis=-1)

    def _eval_ood_stream(v_stream: jax.Array):
        b_in = {
            "video": v_stream,
            "audio": jnp.zeros((4, 24, 64), dtype=jnp.float32),
            "proprio": p_seq,
            "text": jnp.zeros((4, 24, 4), dtype=jnp.int32),
            "action": pu_seq,
            "goal": g_seq,
            "modality_mask": jnp.ones((4, 24, 4), dtype=jnp.float32),
            "valid": jnp.ones((4, 24), dtype=bool),
        }
        zr, _, _ = forward_world_model_wave8a(params, spec, b_in, interventions)
        zm, zt = fork_isolated_latent_streams(zr, interventions.zmem_isolation)
        zn1, lv1 = predict_latent_step(params, spec, zt, u_seq, residual=True)
        hds = decode_world_heads_wave8a(
            params, spec, zm, zt, zn1, lv1, g_seq, interventions, obs_ee_proprio=p_seq, obs_video=v_stream
        )
        routed = route_proposals_with_commit_gate(
            hds["commit_logits"],
            hds["action_proposal"],
            hds["action_proposal_trusted"],
            gate_policy="learned_innovation",
        )
        vis_disc = jnp.sqrt(jnp.sum((hds["vis_ee_curr"] - p_seq[:, :, 0:2]) ** 2, axis=-1) + 1e-6)
        z_ps = jnp.concatenate([zt[:, :1], zn1[:, :-1]], axis=1)
        j_err = jnp.mean(
            (l2_normalize(zt[:, :, SLOTS_VISUAL, :]) - l2_normalize(z_ps[:, :, SLOTS_VISUAL, :])) ** 2,
            axis=(-2, -1),
        )
        probs = routed["mode_probs"]
        anom = 3.0 * vis_disc + 1.2 * j_err + 0.4 * (1.0 - probs[:, :, 0])
        act_delta = jnp.sqrt(jnp.sum((routed["committed_action"] - hds["action_proposal"]) ** 2, axis=-1) + 1e-8)
        rec_f = jnp.clip((hds["diag_video"][0, 16] + 1.0) * 127.5, 0.0, 255.0).astype(jnp.uint8)
        return (
            np.asarray(jax.device_get(anom[0])),
            np.asarray(jax.device_get(probs[0])),
            np.asarray(jax.device_get(routed["mode_id"][0])),
            np.asarray(jax.device_get(act_delta[0])),
            np.asarray(jax.device_get(rec_f)),
        )

    ood_clean_score, ood_clean_probs, ood_clean_modes, ood_clean_act_delta, _ = _eval_ood_stream(v_clean)
    ood_occ_score, ood_occ_probs, ood_occ_modes, ood_occ_act_delta, ood_occ_rec = _eval_ood_stream(v_occ)

    bundle: dict[str, np.ndarray] = {
        "cohort": np.asarray(cohort),
        "seed": np.asarray(seed, dtype=np.int32),
        "snap_ticks": snap_ticks,
        "unseen_layout_id": np.asarray(jax.device_get(init_state["layout_id"]), dtype=np.int32),
        "unseen_goal_xy": np.asarray(jax.device_get(goal_xy), dtype=np.float32),
        "unseen_box_cx": np.asarray(jax.device_get(init_state["box_cx"]), dtype=np.float32),
        "unseen_box_cy": np.asarray(jax.device_get(init_state["box_cy"]), dtype=np.float32),
        "unseen_box_hx": np.asarray(jax.device_get(init_state["box_hx"]), dtype=np.float32),
        "unseen_box_hy": np.asarray(jax.device_get(init_state["box_hy"]), dtype=np.float32),
        "unseen_ee_traj": np.asarray(ee_tr, dtype=np.float32),
        "unseen_obj_traj": np.asarray(obj_tr, dtype=np.float32),
        "unseen_vis_ee_traj": np.asarray(vee_tr, dtype=np.float32),
        "unseen_vis_obj_traj": np.asarray(vobj_tr, dtype=np.float32),
        "unseen_min_dist": np.asarray(min_dist, dtype=np.float32),
        "unseen_snaps_obs": np.asarray(snaps_obs, dtype=np.uint8),
        "unseen_snaps_rec": np.asarray(snaps_rec, dtype=np.uint8),
        "unseen_occ_fin": np.asarray(occ_fin, dtype=np.float32),
        "t160_goal_a": np.asarray(jax.device_get(ga), dtype=np.float32),
        "t160_goal_b": np.asarray(jax.device_get(gb), dtype=np.float32),
        "t160_obj_traj": np.asarray(sw_obj_tr, dtype=np.float32),
        "t160_latch_traj": np.asarray(sw_lat_tr, dtype=np.float32),
        "k16_horizon": np.asarray(t_k16, dtype=np.int32),
        "k16_preds": k16_preds.astype(np.int32),
        "k16_targs": k16_targs.astype(np.int32),
        "k16_stales": k16_stales.astype(np.int32),
        "ood_clean_frame": np.asarray(jax.device_get(v_clean[0, 16]), dtype=np.uint8),
        "ood_occ_frame": np.asarray(jax.device_get(v_occ[0, 16]), dtype=np.uint8),
        "ood_occ_rec": ood_occ_rec,
        "ood_clean_score": ood_clean_score.astype(np.float32),
        "ood_occ_score": ood_occ_score.astype(np.float32),
        "ood_clean_probs": ood_clean_probs.astype(np.float32),
        "ood_occ_probs": ood_occ_probs.astype(np.float32),
        "ood_clean_modes": ood_clean_modes.astype(np.int32),
        "ood_occ_modes": ood_occ_modes.astype(np.int32),
        "ood_clean_act_delta": ood_clean_act_delta.astype(np.float32),
        "ood_occ_act_delta": ood_occ_act_delta.astype(np.float32),
    }

    # 5. Optional physical spoken xArm6 `dev` episode `#0`
    if grounded_npz_path and Path(grounded_npz_path).is_file():
        with np.load(grounded_npz_path, allow_pickle=False) as d:
            vid_r = d["dev_video"][:2]
            aud_r = d["dev_spoken_audio"][:2].astype(np.float32)
            ee_r = d["dev_ee_xy"][:2].astype(np.float32)
            act_r = d["dev_action_xy"][:2].astype(np.float32)
            len_r = d["dev_length"][:2].astype(np.int32)
            txt_r = d["dev_true_text_ids"][:2].astype(np.int32)
        txt_seq = np.zeros((2, 32, 4), dtype=np.int32)
        txt_seq[:, :4, :] = txt_r.reshape(2, 4, 4)
        ee_vel = np.zeros_like(ee_r)
        ee_vel[:, 1:] = ee_r[:, 1:] - ee_r[:, :-1]
        moving_r = (np.linalg.norm(ee_vel, axis=-1, keepdims=True) > 0.015).astype(np.float32)
        prop_r = np.concatenate(
            [
                ee_r,
                ee_r,
                ee_vel,
                np.zeros((2, 32, 1), dtype=np.float32),
                moving_r,
                np.zeros((2, 32, 8), dtype=np.float32),
            ],
            axis=-1,
        )
        u4_r = np.concatenate([act_r, np.zeros_like(act_r)], axis=-1)
        pu4_r = np.concatenate([np.zeros((2, 1, 4), dtype=np.float32), u4_r[:, :-1]], axis=1)
        val_r = np.arange(32)[None, :] < len_r[:, None]
        b_rob = {
            "video": jnp.asarray(vid_r),
            "audio": jnp.asarray(aud_r),
            "proprio": jnp.asarray(prop_r),
            "text": jnp.asarray(txt_seq),
            "action": jnp.asarray(pu4_r),
            "goal": jnp.zeros((2, 32, 4), dtype=jnp.float32),
            "modality_mask": jnp.ones((2, 32, 4), dtype=jnp.float32),
            "valid": jnp.asarray(val_r),
        }
        zr_rob, _, _ = forward_world_model_wave8a(params, spec, b_rob, interventions)
        _, zt_rob = fork_isolated_latent_streams(zr_rob, interventions.zmem_isolation)
        u_j = jnp.asarray(u4_r)
        z1, _ = predict_latent_step(params, spec, zt_rob[:, :-4], u_j[:, :-4], residual=True)
        z2, _ = predict_latent_step(params, spec, z1, u_j[:, 1:-3], residual=True)
        z3, _ = predict_latent_step(params, spec, z2, u_j[:, 2:-2], residual=True)
        z4, _ = predict_latent_step(params, spec, z3, u_j[:, 3:-1], residual=True)
        hw = params["head_proprio_w"].astype(jnp.float32)
        zf4, zf0 = z4.astype(jnp.float32), zt_rob[:, :-4].astype(jnp.float32)
        d4 = (
            0.75 * (zf4[:, :, SLOT_PROPRIO, :] - zf0[:, :, SLOT_PROPRIO, :])
            + 0.25 * (jnp.mean(zf4[:, :, SLOTS_VISUAL, :], axis=2) - jnp.mean(zf0[:, :, SLOTS_VISUAL, :], axis=2))
        ) @ hw
        pred4_ee = np.asarray(jax.device_get(b_rob["proprio"][0, :-4, :2] + d4[0, :, :2]), dtype=np.float32)
        bundle["xarm_frame0"] = vid_r[0, 0]
        bundle["xarm_spoken_spec"] = aud_r[0]
        bundle["xarm_ee_true"] = ee_r[0]
        bundle["xarm_ee_pred4"] = pred4_ee
        bundle["xarm_len"] = len_r[0:1]

    return bundle


def save_showcase_bundle_npz(path: Path, bundle: dict[str, np.ndarray]) -> str:
    """Atomically write compressed `showcase_bundle.npz` and return its SHA-256 hex digest."""
    import hashlib
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **bundle)
    raw = buf.getvalue()
    tmp = path.with_suffix(".npz.tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def select_percentile_episodes(min_dists: np.ndarray) -> dict[str, int]:
    """Deterministically rank episodes from worst (`P05` failure) to best (`P95`) by `-min_dist_to_goal`."""
    n = int(min_dists.shape[0])
    scores = -np.asarray(min_dists, dtype=np.float64)
    # Stable deterministic tie-break by episode index
    order = np.argsort(scores, kind="mergesort")
    pcts = {"P05": 0.05, "P25": 0.25, "P50": 0.50, "P75": 0.75, "P95": 0.95}
    return {name: int(order[min(n - 1, max(0, int(round(q * (n - 1)))))]) for name, q in pcts.items()}


def render_wave8a_showcase_gallery(
    bundles_by_cohort: dict[str, Path],
    out_dir: Path,
    *,
    primary_cohort: str = "w8a_slwm_x2_isolated_450m",
) -> dict[str, Any]:
    """Render deterministic `P05-P95` and cross-cohort synchronized galleries + `showcase_manifest.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for c_key, p in bundles_by_cohort.items():
        if p.is_file():
            with np.load(p, allow_pickle=False) as d:
                loaded[c_key] = {k: np.asarray(d[k]) for k in d.files}
    if not loaded:
        return {"rendered": False, "reason": "no_bundles"}

    prereg_path = REPO_ROOT / "configs/x1b/wave8a_showcase_preregistration.json"
    if not prereg_path.is_file():
        raise FileNotFoundError(f"Missing frozen showcase preregistration: {prereg_path}")
    prereg = json.loads(prereg_path.read_text())
    if list(PRE_REGISTERED_COMMON_EPISODES) != list(prereg["common_cross_model_episode_ids"]):
        raise RuntimeError("PRE_REGISTERED_COMMON_EPISODES drifted from frozen wave8a_showcase_preregistration.json")

    plt = _get_plt()
    ref_key = primary_cohort if primary_cohort in loaded else next(iter(loaded))
    ref = loaded[ref_key]
    pct_eps = select_percentile_episodes(ref["unseen_min_dist"])
    common_eps = [i for i in PRE_REGISTERED_COMMON_EPISODES if i < int(ref["unseen_min_dist"].shape[0])]

    plt.rcParams.update(
        {
            "font.size": 8.6,
            "axes.titlesize": 9.3,
            "axes.titleweight": "bold",
            "figure.dpi": 165,
        }
    )

    # Gallery Figure 1: Pre-registered Percentile Strip (`P05` failure -> `P95`) on SLWM-X2
    pct_order = list(prereg["percentile_quantiles"])
    fig1, axes1 = plt.subplots(len(pct_order), 4, figsize=(13.8, 12.2))
    snap_ticks = ref["snap_ticks"]
    for row_i, p_lbl in enumerate(pct_order):
        ep = pct_eps[p_lbl]
        lid = int(ref["unseen_layout_id"][ep])
        lname = LAYOUT_NAMES.get(lid, f"layout_{lid}")
        md = float(ref["unseen_min_dist"][ep])
        status = "SUCCESS" if md < 0.24 else "FAILURE"

        # Col 0: [observed] RGB start/end strip
        obs_strip = np.concatenate([ref["unseen_snaps_obs"][ep, 0], ref["unseen_snaps_obs"][ep, -1]], axis=1)
        axes1[row_i, 0].imshow(obs_strip)
        axes1[row_i, 0].set_xticks([])
        axes1[row_i, 0].set_yticks([])
        axes1[row_i, 0].set_title(f"{p_lbl} (ep #{ep}, {lname})\n[observed] RGB t={snap_ticks[0]} | t={snap_ticks[-1]}")

        # Col 1: [reconstructed] diagnostic visual decoder
        rec_strip = np.concatenate([ref["unseen_snaps_rec"][ep, 0], ref["unseen_snaps_rec"][ep, -1]], axis=1)
        axes1[row_i, 1].imshow(rec_strip)
        axes1[row_i, 1].set_xticks([])
        axes1[row_i, 1].set_yticks([])
        axes1[row_i, 1].set_title(f"[reconstructed] Diagnostic Readout\n(stop-grad patch decoder)")

        # Col 2: [predicted] 8x8 spatial hazard belief map
        im = axes1[row_i, 2].imshow(
            ref["unseen_occ_fin"][ep],
            origin="lower",
            extent=[-1.0, 1.0, -1.0, 1.0],
            cmap="magma",
            vmin=0.0,
            vmax=max(0.25, float(np.max(ref["unseen_occ_fin"][ep]))),
        )
        axes1[row_i, 2].scatter(
            [ref["unseen_goal_xy"][ep, 0]],
            [ref["unseen_goal_xy"][ep, 1]],
            c="#2ECC71",
            s=55,
            marker="*",
            label="Goal",
        )
        axes1[row_i, 2].set_xlim(-1.0, 1.0)
        axes1[row_i, 2].set_ylim(-1.0, 1.0)
        axes1[row_i, 2].set_title("[predicted] 8x8 Spatial Hazard Map")
        if row_i == 0:
            axes1[row_i, 2].legend(loc="upper right", fontsize=7.2)

        # Col 3: [ground-truth eval] simulator object trajectory vs [predicted] visually decoded trajectory
        ax_t = axes1[row_i, 3]
        for bi in range(3):
            hx = float(ref["unseen_box_hx"][ep, bi])
            hy = float(ref["unseen_box_hy"][ep, bi])
            if hx > 1e-4 and hy > 1e-4:
                cx = float(ref["unseen_box_cx"][ep, bi])
                cy = float(ref["unseen_box_cy"][ep, bi])
                rect = plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="#566573", alpha=0.45)
                ax_t.add_patch(rect)
        ax_t.plot(
            ref["unseen_obj_traj"][ep, :, 0],
            ref["unseen_obj_traj"][ep, :, 1],
            color="#1F618D",
            linewidth=1.8,
            label="[ground-truth eval] True Obj",
        )
        ax_t.plot(
            ref["unseen_vis_obj_traj"][ep, :, 0],
            ref["unseen_vis_obj_traj"][ep, :, 1],
            color="#E67E22",
            linestyle="--",
            linewidth=1.5,
            label="[predicted] Decoded Obj",
        )
        ax_t.scatter([ref["unseen_goal_xy"][ep, 0]], [ref["unseen_goal_xy"][ep, 1]], c="#2ECC71", s=65, marker="*")
        ax_t.set_xlim(-1.0, 1.0)
        ax_t.set_ylim(-1.0, 1.0)
        ax_t.grid(True, alpha=0.25)
        ax_t.set_title(f"Closed-Loop Trajectory ({status}, d={md:.2f})")
        if row_i == 0:
            ax_t.legend(loc="lower right", fontsize=7.0)

    fig1.suptitle(
        "Wave 8A Uncurated Percentile Showcase (P05 Failure to P95) — De-Privileged Unseen Layouts (Dev Gallery)",
        fontsize=11.0,
        fontweight="bold",
        y=0.995,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.985])
    p_fig1 = out_dir / "wave8a_p05_to_p95_unseen_mpc_gallery.png"
    fig1.savefig(p_fig1, bbox_inches="tight")
    plt.close(fig1)

    # Gallery Figure 2: Synchronized Cross-Cohort Comparison (T=160 Switch, K=16 T=1024 Overwrite, OOD Abstention, xArm6)
    cohort_order = [
        ("SLWM-X2 (Iso)", "w8a_slwm_x2_isolated_450m", "#117A65"),
        ("No-Zmem-Iso", "w8a_ablation_no_zmem_iso_450m", "#C0392B"),
        ("Spatial-Only", "w8a_ablation_spatial_control_only_450m", "#2874A6"),
        ("SWin-W64-GQA", "w8a_sliding_window_w64_gqa_450m", "#D35400"),
        ("DiagonalSSM", "w8a_diagonal_ssm_450m", "#8E44AD"),
    ]
    present = [(lbl, ck, col) for lbl, ck, col in cohort_order if ck in loaded]

    fig2, axes2 = plt.subplots(2, 2, figsize=(14.4, 9.6))

    # Panel A: T=160 Goal-B Latch Error vs Time across cohorts + Trivial Zero-Latch Baseline
    ax_a = axes2[0, 0]
    t_len = int(ref["t160_latch_traj"].shape[1])
    ts = np.arange(t_len)
    triv_scalar = float(np.mean(np.sqrt(np.sum(ref["t160_goal_b"] ** 2, axis=-1))))
    triv_err = np.full((t_len,), triv_scalar, dtype=np.float32)
    ax_a.plot(ts, triv_err, color="#95A5A6", linestyle=":", linewidth=1.4, label="Trivial Zero-Memory Baseline")
    for lbl, ck, col in present:
        b = loaded[ck]
        err_t = np.mean(np.sqrt(np.sum((b["t160_latch_traj"] - b["t160_goal_b"][:, None, :]) ** 2, axis=-1)), axis=0)
        ax_a.plot(ts, err_t, color=col, linewidth=1.8, label=f"[predicted] {lbl}")
    if t_len >= 90:
        ax_a.axvline(20 + 64, color="#D35400", linestyle="--", alpha=0.65, label="W=64 Eviction Horizon (t=84)")
        ax_a.axvline(88, color="#2C3E50", linestyle="-.", alpha=0.65, label="Physical Reset Perturbation (t=88)")
    ax_a.set_xlabel("Streaming Tick t (10 Hz)")
    ax_a.set_ylabel("Goal-B Latch Euclidean Error (lower is better)")
    ax_a.set_title("A. T=160 Delayed & Distracted Goal-Switch Memory Trace")
    ax_a.grid(True, alpha=0.28)
    ax_a.legend(fontsize=7.2, loc="upper left")

    # Panel B: K=16 Multi-Item Selective Overwrite Matrix at T=1024 (Episode #0, 16 queried keys)
    ax_b = axes2[0, 1]
    mat_rows = []
    row_labels = []
    # Trivial stale-copy baseline (always predicts initial un-overwritten value -> 0/8 on overwritten, 8/8 on retained)
    stale_status = np.array([1.0] * 8 + [2.0] * 8, dtype=np.float32)  # 2=correct, 1=stale, 0=other_err
    mat_rows.append(stale_status)
    row_labels.append("Trivial Initial-Write Copy")
    for lbl, ck, _ in present:
        b = loaded[ck]
        pr = b["k16_preds"][0]
        tg = b["k16_targs"][0]
        st = b["k16_stales"][0]
        code = np.where(pr == tg, 2.0, np.where(pr == st, 1.0, 0.0))
        mat_rows.append(code)
        row_labels.append(f"[predicted] {lbl}")
    mat = np.stack(mat_rows, axis=0)
    from matplotlib.colors import ListedColormap
    cmap_k16 = ListedColormap(["#E74C3C", "#F39C12", "#27AE60"])
    ax_b.imshow(mat, aspect="auto", cmap=cmap_k16, vmin=0.0, vmax=2.0)
    ax_b.axvline(7.5, color="white", linewidth=2.0)
    ax_b.set_yticks(np.arange(len(row_labels)))
    ax_b.set_yticklabels(row_labels, fontsize=7.8)
    ax_b.set_xticks(np.arange(16))
    ax_b.set_xticklabels([f"Ov{j+1}" for j in range(8)] + [f"Rt{j+1}" for j in range(8)], fontsize=7.0, rotation=35)
    ax_b.set_title(
        f"B. K=16 Dynamic-Visual Overwrite at T={int(ref['k16_horizon'])} (Green=Exact, Amber=Stale, Red=Err)"
    )

    # Panel C: Unseen OOD Camera Corruption & PolicyCommitGate Diagnostic Stream Routing
    ax_c = axes2[1, 0]
    ts_ood = np.arange(24)
    ax_c.plot(
        ts_ood,
        ref["ood_clean_score"],
        color="#27AE60",
        linewidth=1.8,
        label="[gate-routed diagnostic] Clean Stream Anomaly Score",
    )
    ood_routing_counts: dict[str, dict[str, Any]] = {}
    for lbl, ck, col in present:
        b = loaded[ck]
        post_probs = b["ood_occ_probs"][10:]
        modes = (
            b["ood_occ_modes"][10:].astype(np.int32)
            if "ood_occ_modes" in b
            else np.argmax(post_probs, axis=-1).astype(np.int32)
        )
        clean_modes = (
            b["ood_clean_modes"][10:].astype(np.int32)
            if "ood_clean_modes" in b
            else np.argmax(b["ood_clean_probs"][10:], axis=-1).astype(np.int32)
        )
        n_post = int(modes.shape[0])
        n_commit = int(np.sum(modes == 0))
        n_fallback = int(np.sum((modes == 1) | (modes == 2)))
        n_abstain = int(np.sum(modes == 3))
        n_clean_commit = int(np.sum(clean_modes == 0))
        p_non_commit = 1.0 - float(np.mean(post_probs[:, 0]))
        mean_act_delta = float(np.mean(b["ood_occ_act_delta"][10:])) if "ood_occ_act_delta" in b else 0.0
        ood_routing_counts[ck] = {
            "post_onset_ticks": n_post,
            "clean_committed_mode0_ticks": n_clean_commit,
            "committed_mode0_ticks": n_commit,
            "fallback_mode1_2_ticks": n_fallback,
            "abstained_mode3_ticks": n_abstain,
            "mean_non_commit_prob": round(p_non_commit, 4),
            "mean_routed_action_l2_delta": round(mean_act_delta, 4),
            "gate_invoked": "route_proposals_with_commit_gate",
        }
        ax_c.plot(
            ts_ood,
            b["ood_occ_score"],
            color=col,
            linestyle="--" if ck != ref_key else "-",
            linewidth=1.8 if ck == ref_key else 1.2,
            label=f"[gate-predicted non-commit: {n_abstain + n_fallback}/{n_post} (abstain={n_abstain}, fallback={n_fallback})] {lbl}",
        )
    ax_c.axvline(8, color="#7F8C8D", linestyle=":", label="Corruption Onset (t=8)")
    ax_c.set_xlabel("Streaming Tick t")
    ax_c.set_ylabel("Multimodal Innovation Discrepancy Score")
    ax_c.set_title("C. Zero-Shot OOD Camera Fault & PolicyCommitGate Routing (Diagnostic Stream)")
    ax_c.grid(True, alpha=0.28)
    ax_c.legend(fontsize=6.4, loc="upper left")

    # Panel D: Spoken Physical xArm6 4-Step Cartesian Rollout vs Trivial Persistence (or Common Episode #1 Dual-Chicane)
    ax_d = axes2[1, 1]
    if "xarm_ee_true" in ref:
        L = int(ref["xarm_len"][0])
        ee_gt = ref["xarm_ee_true"][:L] * PHYSICAL_TABLE_SCALE_MM
        ax_d.plot(ee_gt[:, 0], ee_gt[:, 1], color="#1B2631", linewidth=2.2, label="[observed] True xArm6 EE (mm)")
        pers = ref["xarm_ee_true"][: max(1, L - 4)] * PHYSICAL_TABLE_SCALE_MM
        ax_d.scatter(pers[:, 0], pers[:, 1], c="#95A5A6", s=16, label="Trivial Persistence Copy (t -> t+4)")
        for lbl, ck, col in present:
            b = loaded[ck]
            if "xarm_ee_pred4" in b:
                pr4 = b["xarm_ee_pred4"][: max(1, L - 4)] * PHYSICAL_TABLE_SCALE_MM
                ax_d.plot(pr4[:, 0], pr4[:, 1], color=col, linestyle="--", linewidth=1.5, label=f"[predicted] {lbl} (t+4)")
        ax_d.set_xlabel("Workspace X (mm)")
        ax_d.set_ylabel("Workspace Y (mm)")
        ax_d.set_title("D. Grounded Spoken xArm6 4-Step Forecast (Dev Episode #0)")
        ax_d.grid(True, alpha=0.28)
        ax_d.legend(fontsize=7.0, loc="best")
    else:
        ep_c = common_eps[min(1, len(common_eps) - 1)]
        for lbl, ck, col in present:
            b = loaded[ck]
            ax_d.plot(
                b["unseen_obj_traj"][ep_c, :, 0],
                b["unseen_obj_traj"][ep_c, :, 1],
                color=col,
                linewidth=1.6,
                label=f"[ground-truth eval] {lbl}",
            )
        ax_d.scatter([ref["unseen_goal_xy"][ep_c, 0]], [ref["unseen_goal_xy"][ep_c, 1]], c="#2ECC71", s=70, marker="*")
        ax_d.set_xlim(-1.0, 1.0)
        ax_d.set_ylim(-1.0, 1.0)
        ax_d.grid(True, alpha=0.28)
        ax_d.set_title(f"D. Common Cross-Cohort Episode #{ep_c} Rollout")
        ax_d.legend(fontsize=7.2, loc="best")

    fig2.suptitle(
        "Wave 8A Synchronized Cross-Cohort Inference Showcase — Memory, Control, Safety & Grounded Robotics",
        fontsize=11.0,
        fontweight="bold",
        y=0.995,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.982])
    p_fig2 = out_dir / "wave8a_cross_cohort_t160_and_k16_showcase.png"
    fig2.savefig(p_fig2, bbox_inches="tight")
    plt.close(fig2)

    # Gallery Figure 3: Pre-Registered Common Cross-Cohort Episodes [0, 1, 2, 3] across all 4 Unseen Layout Families
    fig3, axes3 = plt.subplots(2, 2, figsize=(13.6, 9.4))
    for panel_idx in range(4):
        ax_ce = axes3[panel_idx // 2, panel_idx % 2]
        ep_c = common_eps[panel_idx] if panel_idx < len(common_eps) else 0
        lid_c = int(ref["unseen_layout_id"][ep_c])
        lname_c = LAYOUT_NAMES.get(lid_c, f"layout_{lid_c}")
        for bi in range(3):
            hx = float(ref["unseen_box_hx"][ep_c, bi])
            hy = float(ref["unseen_box_hy"][ep_c, bi])
            if hx > 1e-4 and hy > 1e-4:
                cx = float(ref["unseen_box_cx"][ep_c, bi])
                cy = float(ref["unseen_box_cy"][ep_c, bi])
                ax_ce.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color="#566573", alpha=0.45))
        ax_ce.scatter(
            [ref["unseen_obj_traj"][ep_c, 0, 0]],
            [ref["unseen_obj_traj"][ep_c, 0, 1]],
            c="#7F8C8D",
            s=42,
            marker="o",
            label="Trivial Persistence (Initial Obj)",
        )
        for lbl, ck, col in present:
            b = loaded[ck]
            md_c = float(b["unseen_min_dist"][ep_c])
            ax_ce.plot(
                b["unseen_obj_traj"][ep_c, :, 0],
                b["unseen_obj_traj"][ep_c, :, 1],
                color=col,
                linewidth=1.9 if ck == ref_key else 1.45,
                linestyle="-" if ck == ref_key else "--",
                label=f"[ground-truth eval] {lbl} (d={md_c:.2f})",
            )
        ax_ce.scatter(
            [ref["unseen_goal_xy"][ep_c, 0]],
            [ref["unseen_goal_xy"][ep_c, 1]],
            c="#2ECC71",
            s=75,
            marker="*",
            label="Goal",
        )
        ax_ce.set_xlim(-1.0, 1.0)
        ax_ce.set_ylim(-1.0, 1.0)
        ax_ce.grid(True, alpha=0.28)
        ax_ce.set_title(f"Pre-Registered Common Episode #{ep_c} ({lname_c})")
        ax_ce.legend(fontsize=6.8, loc="best")

    fig3.suptitle(
        "Wave 8A Pre-Registered Common Episodes [0, 1, 2, 3] — Side-by-Side Cross-Cohort Trajectories (Dev Gallery)",
        fontsize=11.0,
        fontweight="bold",
        y=0.995,
    )
    fig3.tight_layout(rect=[0, 0, 1, 0.982])
    p_fig3 = out_dir / "wave8a_common_episodes_cross_cohort_gallery.png"
    fig3.savefig(p_fig3, bbox_inches="tight")
    plt.close(fig3)

    t160_n = int(ref["t160_latch_traj"].shape[0])
    rendered_artifacts = [p_fig1.name, p_fig2.name, p_fig3.name]
    if rendered_artifacts != list(prereg["artifacts"]):
        raise RuntimeError(
            f"Rendered showcase artifact names {rendered_artifacts} do not match preregistered {prereg['artifacts']}"
        )
    manifest = {
        "rendered": True,
        "stage": "rendered",
        "preregistration_verified": True,
        "wave": "wave8a",
        "split": "dev",
        "weights_frozen": True,
        "online_weight_updates": False,
        "selection_protocol": {
            "common_cross_model_episode_ids": common_eps,
            "cohort_seed_selection_rule": prereg["cohort_seed_selection_rule"],
            "cohort_seeds_used": {ck: int(loaded[ck]["seed"]) for _, ck, _ in present},
            "percentile_quantiles": pct_order,
            "percentile_ranking_metric": prereg["percentile_ranking_metric"],
            "percentile_reference_cohort": ref_key,
            "percentile_reference_seed": int(ref["seed"]),
            "selected_percentile_episodes": {
                k: {
                    "episode_id": int(v),
                    "layout_family": LAYOUT_NAMES.get(int(ref["unseen_layout_id"][v]), "unknown"),
                    "min_goal_distance": round(float(ref["unseen_min_dist"][v]), 4),
                    "succeeded": bool(float(ref["unseen_min_dist"][v]) < 0.24),
                }
                for k, v in pct_eps.items()
            },
            "figure2_fixed_episode_ids": {
                "panel_a_t160_mean_episode_ids": list(range(t160_n)),
                "panel_b_k16_overwrite_episode_id": int(prereg["figure2_fixed_episode_ids"]["panel_b_k16_overwrite_episode_id"]),
                "panel_c_ood_camera_episode_id": int(prereg["figure2_fixed_episode_ids"]["panel_c_ood_camera_episode_id"]),
                "panel_d_xarm6_dev_episode_id": (
                    int(prereg["figure2_fixed_episode_ids"]["panel_d_xarm6_dev_episode_id"])
                    if "xarm_ee_true" in ref
                    else int(common_eps[min(1, len(common_eps) - 1)])
                ),
            },
        },
        "ood_post_onset_routing_counts": ood_routing_counts,
        "epistemic_channel_definitions": {
            "observed": "Model-visible sensory frame, spectrogram, or end-effector proprioception recorded from environment.",
            "ground_truth_eval": "Simulator ground-truth object coordinate trajectory plotted strictly for evaluation comparison (never observed by model proprioception).",
            "reconstructed": "Stop-gradient diagnostic patch/audio linear readout from current latent state Z_task.",
            "predicted": "Action-conditioned future latent rollout, decoded spatial pose/hazard map, or recalled symbolic binding.",
            "abstained": "Tick routed to NO_OP abstention (mode 3: noop_abstain) by route_proposals_with_commit_gate on the diagnostic stream.",
            "fallback_mitigated": "Tick routed to trusted-proprioception fallback (mode 1: commit_trusted_proprio) or cautious probe (mode 2: probe_cautious) by route_proposals_with_commit_gate.",
        },
        "cohorts_included": [ck for _, ck, _ in present],
        "artifacts": rendered_artifacts,
    }
    manifest_path = out_dir / "showcase_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
