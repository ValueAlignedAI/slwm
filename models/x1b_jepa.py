"""Action-conditioned Residual EMA-JEPA, SIGReg anti-collapse, and multimodal heads for SLWM-X1B."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from models.x1b_world import (
    SLOT_PROPRIO,
    SLOT_TEXT,
    SLOTS_AUDIO,
    SLOTS_REGISTER,
    SLOTS_VISUAL,
    X1BSpec,
    forward_world_model,
    rms_norm,
)

VALID_OBJECTIVES = (
    "residual_ema_jepa_sigreg",
    "raw_pixel_linear_mse",
    "direct_jepa_no_residual",
    "residual_jepa_no_sigreg",
)


def l2_normalize(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalize along the last axis in float32."""
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.sum(xf * xf, axis=-1, keepdims=True) + eps)


def predict_latent_step(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z_t: jax.Array,
    action_t: jax.Array,
    horizon_idx: int = 0,
    *,
    residual: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Predict future world field Z_{t+k} and per-slot innovation log-variance."""
    dtype = getattr(jnp, spec.dtype)
    z_in = z_t.astype(dtype)
    act_emb = (action_t.astype(dtype) @ params["jepa_act_w"].astype(dtype))[..., None, :]
    h_emb = params["jepa_horizon_emb"].astype(dtype)[horizon_idx]
    h = rms_norm(z_in + act_emb + h_emb, params["jepa_norm"])

    global_ctx = h.mean(axis=-2, keepdims=True)
    h_mix = h + 0.25 * global_ctx
    gate = jax.nn.silu(h_mix @ params["jepa_gate"].astype(dtype))
    up = h_mix @ params["jepa_w1"].astype(dtype)
    raw_delta = (gate * up) @ params["jepa_w2"].astype(dtype)
    # Smoothly bound multi-step rollout drift while preserving full unclipped 1-step contact/impact jumps:
    delta = (1.50 * jnp.tanh(raw_delta.astype(jnp.float32) / 1.50)).astype(dtype)
    z_hat = (z_in + delta) if residual else rms_norm(raw_delta, params["final_norm"])

    u_hid = jax.nn.silu(h_mix.astype(jnp.float32) @ params["unc_w1"].astype(jnp.float32))
    log_var = jnp.clip(
        u_hid @ params["unc_w2"].astype(jnp.float32) + params["unc_b2"].astype(jnp.float32),
        -5.0,
        4.0,
    )
    return z_hat, log_var


def sigreg_loss_and_pr(z_samples: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute isotropic Variance-Covariance anti-collapse loss (SIGReg/VICReg) + Participation Ratio."""
    zf = z_samples.astype(jnp.float32)
    n, d = zf.shape
    zf_centered = zf - jnp.mean(zf, axis=0, keepdims=True)
    std = jnp.sqrt(jnp.mean(zf_centered * zf_centered, axis=0) + 1e-4)
    var_loss = jnp.mean(jnp.maximum(0.0, 1.0 - std))

    cov = (zf_centered.T @ zf_centered) / jnp.maximum(float(n - 1), 1.0)
    off_diag = cov - jnp.diag(jnp.diag(cov))
    cov_loss = jnp.sum(off_diag * off_diag) / float(max(1, d * (d - 1)))

    tr_c = jnp.trace(cov)
    tr_c2 = jnp.sum(cov * cov)
    pr = (tr_c * tr_c) / jnp.maximum(float(d) * tr_c2, 1e-8)
    return var_loss + 4.0 * cov_loss, pr


def compute_innovation_features(
    params: dict[str, jax.Array],
    z_obs: jax.Array,
    z_pred_prev: jax.Array,
    log_var_prev: jax.Array,
    vis_pose_discrepancy: jax.Array | None = None,
) -> jax.Array:
    """Compute 6-D Kalman/Mahalanobis cross-modal & geometric innovation statistics at each tick t."""
    diff_sq = jnp.mean((z_obs.astype(jnp.float32) - z_pred_prev.astype(jnp.float32)) ** 2, axis=-1)
    inv_var = jnp.exp(-jnp.squeeze(log_var_prev, axis=-1))
    mahal = diff_sq * inv_var

    m_vis = jnp.mean(mahal[:, :, SLOTS_VISUAL], axis=-1)
    m_aud = jnp.mean(mahal[:, :, SLOTS_AUDIO], axis=-1)
    m_prop = mahal[:, :, SLOT_PROPRIO]
    spoof_gap = jnp.log1p(m_vis) - jnp.log1p(m_prop)

    if vis_pose_discrepancy is not None:
        disagree = vis_pose_discrepancy
    else:
        v_proj = l2_normalize(
            jnp.mean(z_obs[:, :, SLOTS_VISUAL, :], axis=2).astype(jnp.float32) @ params["align_vis_w"].astype(jnp.float32)
        )
        p_proj = l2_normalize(z_obs[:, :, SLOT_PROPRIO, :].astype(jnp.float32) @ params["align_prop_w"].astype(jnp.float32))
        disagree = 1.0 - jnp.sum(v_proj * p_proj, axis=-1)
    mean_lv = jnp.mean(jnp.squeeze(log_var_prev, axis=-1), axis=-1)

    return jnp.stack(
        [jnp.log1p(m_vis), jnp.log1p(m_aud), jnp.log1p(m_prop), spoof_gap, disagree, mean_lv],
        axis=-1,
    )


def _ego_push_kinematics(ee_xy: jax.Array, obj_xy: jax.Array, goal_xy: jax.Array, has_ext_goal: jax.Array) -> jax.Array:
    """Causal tick-t relative pushing geometry [drive_vec (2), obj_minus_ee (2)] from decoded or proprio pose."""
    to_goal = goal_xy - obj_xy
    push_unit = to_goal * jax.lax.rsqrt(jnp.sum(to_goal * to_goal, axis=-1, keepdims=True) + 1e-6)
    rel_ee = ee_xy - obj_xy
    proj_along = jnp.sum(rel_ee * push_unit, axis=-1, keepdims=True)
    perp = jnp.stack([-push_unit[..., 1], push_unit[..., 0]], axis=-1)
    lat_err = jnp.sum(rel_ee * perp, axis=-1, keepdims=True)
    perp_side = jnp.where(lat_err >= 0.0, 1.0, -1.0)
    need_wrap = (proj_along > -0.08) & (jnp.sqrt(jnp.sum(rel_ee * rel_ee, axis=-1, keepdims=True) + 1e-6) < 0.30)
    wrap_pt = obj_xy + perp * perp_side * 0.25 - 0.14 * push_unit
    aligned = (proj_along <= -0.10) & (jnp.abs(lat_err) < 0.09)
    staging = obj_xy - 0.20 * push_unit
    drive_pt = jnp.where(need_wrap, wrap_pt, jnp.where(aligned, obj_xy + 0.08 * push_unit, staging))
    drive_vec = has_ext_goal * jnp.clip((drive_pt - ee_xy) * 3.5, -2.0, 2.0)
    return jnp.concatenate([drive_vec, obj_xy - ee_xy], axis=-1)


def decode_world_heads(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z: jax.Array,
    z_next_pred: jax.Array,
    log_var_next: jax.Array,
    goal: jax.Array,
    *,
    obs_proprio: jax.Array | None = None,
    use_residual: bool = True,
    stop_grad_diagnostics: bool = True,
) -> dict[str, jax.Array]:
    """Decode physical state, contacts, action/value proposals, commitment logits, text, and diagnostic media."""
    b, t, _, _ = z.shape
    zf = z.astype(jnp.float32)
    z_next_f = z_next_pred.astype(jnp.float32)

    vis_sum = jnp.mean(zf[:, :, SLOTS_VISUAL, :], axis=2)
    vis_next_sum = jnp.mean(z_next_f[:, :, SLOTS_VISUAL, :], axis=2)
    aud_sum = jnp.mean(zf[:, :, SLOTS_AUDIO, :], axis=2)
    reg_sum = jnp.mean(zf[:, :, SLOTS_REGISTER, :], axis=2)
    reg_next_sum = jnp.mean(z_next_f[:, :, SLOTS_REGISTER, :], axis=2)
    prop_lat = zf[:, :, SLOT_PROPRIO, :]
    prop_next_lat = z_next_f[:, :, SLOT_PROPRIO, :]

    # Uncorrupted Euclidean proprioceptive embedding directly from trusted joint/state sensors (immune to visual spoofing!)
    g_f = goal.astype(jnp.float32)
    if g_f.ndim == 2:
        g_f = g_f[:, None, :]
    goal_bt = jnp.broadcast_to(g_f, (b, t, spec.goal_dim))
    has_ext_goal = (jnp.sum(jnp.abs(goal_bt), axis=-1, keepdims=True) > 1e-4).astype(jnp.float32)

    head_pw = params["head_proprio_w"].astype(jnp.float32)
    # Pure visual pose readout enables continuous cross-modal visual-vs-proprioception spoofing verification,
    # while proprio_curr matches the exact (0.5 prop + 0.5 vis) basis of the Euler latent difference!
    vis_proprio_curr = vis_sum @ head_pw

    if obs_proprio is not None:
        obs_pf = obs_proprio.astype(jnp.float32)
        prop_kin = _ego_push_kinematics(obs_pf[:, :, 0:2], obs_pf[:, :, 2:4], goal_bt[:, :, 2:4], has_ext_goal)
        obs_pf_aug = obs_pf.at[..., 12:15].set(prop_kin[..., :3]).at[..., 15].set(1.0)
        clean_prop_emb = 3.5 * (obs_pf_aug @ params["prop_w"].astype(jnp.float32))
    else:
        clean_prop_emb = prop_lat

    proprio_curr = (0.5 * prop_lat + 0.5 * vis_sum) @ head_pw
    proprio_trusted = (0.7 * clean_prop_emb + 0.3 * reg_sum) @ head_pw

    # Unit-scale Euler Physical Residual Integration: s_hat_{t+1} = s_t + (s_hat(z_hat_{t+1}) - s_hat(z_t))
    use_euler = (obs_proprio is not None) and use_residual and (spec.arch != "legacy_scalar_hybrid")
    if use_euler:
        obs_pf = obs_proprio.astype(jnp.float32)
        delta_full = (0.5 * (prop_next_lat - prop_lat) + 0.5 * (vis_next_sum - vis_sum)) @ head_pw
        delta_trust = (0.7 * (prop_next_lat - prop_lat) + 0.3 * (reg_next_sum - reg_sum)) @ head_pw
        proprio_next = obs_pf + delta_full
        proprio_next_trusted = obs_pf + delta_trust
    else:
        proprio_next = (0.5 * prop_next_lat + 0.5 * vis_next_sum) @ head_pw
        proprio_next_trusted = (0.7 * prop_next_lat + 0.3 * reg_next_sum) @ head_pw

    # Impact & 64x64 thin-barrier contact logits (draws from spatial visual max + mean + audio + registers)
    vis_max = jnp.max(zf[:, :, SLOTS_VISUAL, :], axis=2)
    contact_ctx = 0.45 * vis_max + 0.25 * vis_sum + 0.15 * aud_sum + 0.15 * reg_sum
    contact_logits = contact_ctx @ params["head_contact_w"].astype(jnp.float32)

    vis_ee_xy = jax.lax.stop_gradient(vis_proprio_curr[:, :, 0:2])
    vis_obj_xy = jax.lax.stop_gradient(vis_proprio_curr[:, :, 2:4])
    if obs_proprio is not None:
        prop_ee_xy = obs_proprio.astype(jnp.float32)[:, :, 0:2]
        prop_obj_xy = obs_proprio.astype(jnp.float32)[:, :, 2:4]
    else:
        prop_ee_xy = jax.lax.stop_gradient(proprio_trusted[:, :, 0:2])
        prop_obj_xy = jax.lax.stop_gradient(proprio_trusted[:, :, 2:4])
    vis_rel_goal = 3.5 * _ego_push_kinematics(vis_ee_xy, vis_obj_xy, goal_bt[:, :, 2:4], has_ext_goal)
    prop_rel_goal = 3.5 * _ego_push_kinematics(prop_ee_xy, prop_obj_xy, goal_bt[:, :, 2:4], has_ext_goal)

    # Disjoint subspace routing in head_action_w1 ([0:D] = visual+register, [D:2D] = trusted proprio):
    # Mode 0 (commit_full): relies on high-res visual slots + working-memory registers + visual-relative target
    pol_in = jnp.concatenate([0.65 * vis_sum + 0.35 * reg_sum, jnp.zeros_like(clean_prop_emb), vis_rel_goal], axis=-1)
    act_prop = jnp.tanh(
        jax.nn.silu(pol_in @ params["head_action_w1"].astype(jnp.float32))
        @ params["head_action_w2"].astype(jnp.float32)
    )
    value_pred = jnp.squeeze(
        jax.nn.silu(pol_in @ params["head_value_w1"].astype(jnp.float32))
        @ params["head_value_w2"].astype(jnp.float32),
        axis=-1,
    )

    # Mode 1 (commit_trusted_proprio): zeroes visual block and uses visually-isolated proprioception + proprio-relative target!
    pol_trusted_in = jnp.concatenate([jnp.zeros_like(vis_sum), clean_prop_emb, prop_rel_goal], axis=-1)
    act_prop_trusted = jnp.tanh(
        jax.nn.silu(pol_trusted_in @ params["head_action_w1"].astype(jnp.float32))
        @ params["head_action_w2"].astype(jnp.float32)
    )

    z_pred_shifted = jnp.concatenate([z[:, :1], z_next_pred[:, :-1]], axis=1)
    lv_shifted = jnp.concatenate([log_var_next[:, :1], log_var_next[:, :-1]], axis=1)
    if obs_proprio is not None:
        vis_pose_disc = jnp.sqrt(
            jnp.sum((vis_proprio_curr[:, :, :4] - obs_proprio.astype(jnp.float32)[:, :, :4]) ** 2, axis=-1) + 1e-6
        )
    else:
        vis_pose_disc = None
    innov_feats = compute_innovation_features(
        params,
        z,
        jax.lax.stop_gradient(z_pred_shifted),
        jax.lax.stop_gradient(lv_shifted),
        vis_pose_discrepancy=jax.lax.stop_gradient(vis_pose_disc) if vis_pose_disc is not None else None,
    )

    commit_in = jnp.concatenate([reg_sum, goal_bt, innov_feats], axis=-1)
    commit_logits = (
        jax.nn.silu(commit_in @ params["commit_w1"].astype(jnp.float32))
        @ params["commit_w2"].astype(jnp.float32)
        + params["commit_b2"].astype(jnp.float32)
    )

    # Subword text / associative recall logits:
    # Symbol-pure GatedDelta world models decode directly from SLOT_TEXT without physical register variance,
    # while non-GatedDelta baselines preserve exact Wave-2 readout parity.
    if spec.arch in ("slwm_x1b", "slwm_no_registers", "separate_modality_cores"):
        txt_ctx = zf[:, :, SLOT_TEXT, :]
    else:
        txt_ctx = 0.50 * zf[:, :, SLOT_TEXT, :] + 0.50 * reg_sum
    text_logits = txt_ctx @ params["head_text_w"].astype(jnp.float32)

    z_diag = jax.lax.stop_gradient(zf) if stop_grad_diagnostics else zf
    vis_patches_dec = z_diag[:, :, SLOTS_VISUAL, :] @ params["diag_vis_patch_w"].astype(jnp.float32)
    vis_grid = vis_patches_dec.reshape(b, t, 4, 4, 16, 16, 3)
    diag_video = jnp.swapaxes(vis_grid, 3, 4).reshape(b, t, 64, 64, 3)
    diag_audio = jnp.mean(z_diag[:, :, SLOTS_AUDIO, :], axis=2) @ params["diag_aud_w"].astype(jnp.float32)

    return {
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


def update_ema_params(
    ema_params: dict[str, jax.Array],
    online_params: dict[str, jax.Array],
    decay: float = 0.995,
) -> dict[str, jax.Array]:
    """Exponential moving average update for the stop-gradient JEPA target encoder."""
    return jax.tree.map(
        lambda e, p: (decay * e.astype(jnp.float32) + (1.0 - decay) * p.astype(jnp.float32)).astype(e.dtype),
        ema_params,
        online_params,
    )


def world_model_loss(
    params: dict[str, jax.Array],
    ema_params: dict[str, jax.Array],
    spec: X1BSpec,
    batch: dict[str, jax.Array],
    objective_mode: str = "residual_ema_jepa_sigreg",
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Unified pre-registered training objective supporting all 4 Study X1B objective variants."""
    if objective_mode not in VALID_OBJECTIVES:
        raise ValueError(f"Unknown objective_mode {objective_mode!r}")

    is_raw_mse = objective_mode == "raw_pixel_linear_mse"
    use_residual = objective_mode != "direct_jepa_no_residual"
    use_sigreg = objective_mode == "residual_ema_jepa_sigreg"

    z, _ = forward_world_model(params, spec, batch)
    b, t, _, _ = z.shape
    valid = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(jnp.float32)
    valid_pair = valid[:, :-1] * valid[:, 1:]

    # Causal action u_t executed at tick t to transition Z_t -> Z_{t+1}
    u_exec = batch.get("next_action", batch["action"])
    z_next_1, log_var_1 = predict_latent_step(params, spec, z, u_exec, horizon_idx=0, residual=use_residual)

    if not is_raw_mse:
        clean_batch = dict(batch)
        if "clean_video" in batch:
            clean_batch["video"] = batch["clean_video"]
        if "full_modality_mask" in batch:
            clean_batch["modality_mask"] = batch["full_modality_mask"]
        z_target, _ = forward_world_model(jax.lax.stop_gradient(ema_params), spec, clean_batch)
        z_target = jax.lax.stop_gradient(z_target)

        err_1 = jnp.mean((l2_normalize(z_next_1[:, :-1]) - l2_normalize(z_target[:, 1:])) ** 2, axis=(-2, -1))
        jepa_1 = jnp.sum(err_1 * valid_pair) / jnp.maximum(jnp.sum(valid_pair), 1.0)

        if t > 2:
            z_next_2, _ = predict_latent_step(
                params, spec, z_next_1[:, :-1], u_exec[:, 1:], horizon_idx=0, residual=use_residual
            )
            v2 = valid[:, :-2] * valid[:, 2:]
            err_2 = jnp.mean((l2_normalize(z_next_2[:, :-1]) - l2_normalize(z_target[:, 2:])) ** 2, axis=(-2, -1))
            jepa_2 = jnp.sum(err_2 * v2) / jnp.maximum(jnp.sum(v2), 1.0)
        else:
            z_next_2 = None
            jepa_2 = jnp.zeros((), dtype=jnp.float32)

        raw_sq = jax.lax.stop_gradient(
            jnp.mean(
                (z_next_1[:, :-1].astype(jnp.float32) - z_target[:, 1:].astype(jnp.float32)) ** 2,
                axis=-1,
                keepdims=True,
            )
        )
        lv = log_var_1[:, :-1]
        nll = 0.5 * (raw_sq * jnp.exp(-lv) + lv)
        unc_loss = jnp.sum(jnp.mean(nll, axis=(-2, -1)) * valid_pair) / jnp.maximum(jnp.sum(valid_pair), 1.0)
        jepa_loss = jepa_1 + 0.5 * jepa_2
    else:
        z_next_2 = None
        jepa_loss = jnp.zeros((), dtype=jnp.float32)
        unc_loss = jnp.zeros((), dtype=jnp.float32)

    # 2. SIGReg Anti-Collapse & Cross-Modal Alignment (Visual <-> Audio & Clean Visual <-> Proprio)
    vis_pool = jnp.mean(z[:, :, SLOTS_VISUAL, :], axis=2).astype(jnp.float32) @ params["align_vis_w"].astype(jnp.float32)
    aud_pool = jnp.mean(z[:, :, SLOTS_AUDIO, :], axis=2).astype(jnp.float32) @ params["align_aud_w"].astype(jnp.float32)
    sig_loss_v, pr_val = sigreg_loss_and_pr(vis_pool.reshape(b * t, -1))
    sig_loss_a, _ = sigreg_loss_and_pr(aud_pool.reshape(b * t, -1))
    sig_loss = 0.5 * (sig_loss_v + sig_loss_a) if use_sigreg else jnp.zeros((), dtype=jnp.float32)

    v_clip = l2_normalize(jnp.mean(vis_pool, axis=1))
    a_clip = l2_normalize(jnp.mean(aud_pool, axis=1))
    temp = jnp.clip(jnp.exp(params["align_logit_scale"].astype(jnp.float32))[0], 1.0, 50.0)
    sim = temp * (v_clip @ a_clip.T)
    av_nce = 0.5 * (
        -jnp.mean(jnp.diag(jax.nn.log_softmax(sim, axis=-1)))
        - jnp.mean(jnp.diag(jax.nn.log_softmax(sim, axis=0)))
    )
    has_av = batch.get("has_av_pair", jnp.ones((b,), dtype=jnp.float32)).astype(jnp.float32)

    # Calibrate visual-proprio cosine consistency on clean non-spoofed ticks so spoofing spikes disagree!
    commit_target = batch["target_commit"].astype(jnp.int32)
    clean_tick_mask = (commit_target == 0).astype(jnp.float32) * valid
    clean_prop_emb = batch["proprio"].astype(jnp.float32) @ params["prop_w"].astype(jnp.float32)
    vp_cos = jnp.sum(
        l2_normalize(vis_pool) * l2_normalize(clean_prop_emb @ params["align_prop_w"].astype(jnp.float32)),
        axis=-1,
    )
    vp_align = jnp.sum((1.0 - vp_cos) * clean_tick_mask) / jnp.maximum(jnp.sum(clean_tick_mask), 1.0)
    align_loss = av_nce * jnp.mean(has_av) + 0.35 * vp_align

    # 3. Task, Control, Commit Gate, Language & Diagnostic Decoders
    heads = decode_world_heads(
        params,
        spec,
        z,
        z_next_1,
        log_var_1,
        batch["goal"],
        obs_proprio=batch["proprio"],
        use_residual=use_residual,
        stop_grad_diagnostics=(not is_raw_mse),
    )

    target_prop = batch["target_proprio"].astype(jnp.float32)
    # Supervise visual pose readout on clean visual-grounded ticks so camera spoofing triggers sharp discrepancy!
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
        last_valid = jnp.where(scen_vec < 0, 0.0, 1.0)[:, None]
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
    # Scale 100ms step transition MSE by 1/dt^2 (~35x) so velocity errors O(0.045^2 = 0.002) match position scale O(0.35)!
    state_loss = 1.5 * prop_curr_loss + 35.0 * prop_next_loss + 10.0 * prop_trusted_loss

    # Contact & 64x64 Thin-Barrier Hazard BCE
    target_contacts = batch["target_contacts"].astype(jnp.float32)
    cl = heads["contact_logits"]
    contact_bce = jnp.mean(
        jax.nn.softplus(cl) * (1.0 - target_contacts) + 2.5 * jax.nn.softplus(-cl) * target_contacts
    )

    # Action proposals: Mode 0 supervised on clean sensors; Mode 1 supervised on clean + spoofed proprio!
    act_target = batch["target_action"].astype(jnp.float32)
    base_act_mask = batch.get("action_supervision_mask", jnp.ones((b, t), dtype=jnp.float32)).astype(jnp.float32)
    mode0_mask = base_act_mask * (commit_target == 0).astype(jnp.float32)
    mode1_mask = base_act_mask * (commit_target <= 1).astype(jnp.float32)
    loss_act0 = jnp.sum(
        jnp.mean((heads["action_proposal"] - act_target) ** 2, axis=-1) * mode0_mask
    ) / jnp.maximum(jnp.sum(mode0_mask), 1.0)
    loss_act1 = jnp.sum(
        jnp.mean((heads["action_proposal_trusted"] - act_target) ** 2, axis=-1) * mode1_mask
    ) / jnp.maximum(jnp.sum(mode1_mask), 1.0)
    act_loss = loss_act0 + 1.0 * loss_act1

    val_target = batch["target_value"].astype(jnp.float32)
    val_loss = jnp.sum(((heads["value_pred"] - val_target) ** 2) * mode0_mask) / jnp.maximum(jnp.sum(mode0_mask), 1.0)

    commit_lp = jax.nn.log_softmax(heads["commit_logits"], axis=-1)
    commit_cls_w = jnp.array([1.0, 2.4, 4.5, 1.8], dtype=jnp.float32)[commit_target]
    commit_nll = -jnp.take_along_axis(commit_lp, commit_target[..., None], axis=-1)[..., 0]
    commit_loss = jnp.sum(commit_nll * commit_cls_w) / jnp.maximum(jnp.sum(commit_cls_w), 1.0)

    # Balanced Subword Language Modeling (FineWeb) + Multi-Query Associative Overwrite Recall
    txt_target = batch["target_text"].astype(jnp.int32)
    txt_mask = (txt_target > 0).astype(jnp.float32) * valid
    txt_lp = jax.nn.log_softmax(heads["text_logits"], axis=-1)
    txt_nll = -jnp.take_along_axis(txt_lp, jnp.clip(txt_target, 0, spec.vocab_size - 1)[..., None], axis=-1)[..., 0]

    scen_arr = batch.get("scenario", jnp.zeros((b,), dtype=jnp.int32))
    is_synth = (scen_arr >= 0).astype(jnp.float32)[:, None]
    synth_tmask = txt_mask * is_synth
    real_tmask = txt_mask * (1.0 - is_synth)
    assoc_ce = jnp.sum(txt_nll * synth_tmask) / jnp.maximum(jnp.sum(synth_tmask), 1.0)
    lm_ce = jnp.sum(txt_nll * real_tmask) / jnp.maximum(jnp.sum(real_tmask), 1.0)
    text_loss = assoc_ce + 0.5 * lm_ce

    clean_vid = batch.get("clean_video", batch["video"]).astype(jnp.float32) / 127.5 - 1.0
    vid_mse = jnp.mean((heads["diag_video"] - clean_vid) ** 2)
    aud_mse = jnp.mean((heads["diag_audio"] - batch["audio"].astype(jnp.float32)) ** 2)
    recon_loss = vid_mse + 0.5 * aud_mse

    if is_raw_mse:
        total_loss = (
            2.5 * recon_loss
            + 1.0 * state_loss
            + 0.5 * contact_bce
            + 1.4 * act_loss
            + 0.25 * val_loss
            + 0.5 * commit_loss
            + 0.85 * text_loss
        )
    else:
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
            + 0.2 * recon_loss
        )

    metrics = {
        "loss": total_loss,
        "jepa": jepa_loss,
        "sigreg": sig_loss,
        "pr": pr_val,
        "state_mse": prop_next_loss,
        "contact_bce": contact_bce,
        "commit_ce": commit_loss,
        "text_ce": assoc_ce,
        "recon_mse": recon_loss,
    }
    return total_loss, metrics
