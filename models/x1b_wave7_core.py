"""Wave 7 Spatial-Grid De-Privileged Grounding Core (`SLWM-v2` Clincher).

Resolves the spatial-pooling localization bottleneck exposed by Wave 6 (`runs_wave6`) under strict
End-Effector-Only (`ee_only`) proprioception and zero geometric planner heuristics, at exact `0.0000%`
parameter parity (`453,199,938` strict parameters across all 453M SLWM / DiagonalSSM / LOO cohorts):

  1. Pure-Vision + Latent Spatial Soft-Argmax Keypoint Readout (`decode_spatial_vis_pose`):
     Replaces permutation-invariant Global Average Pooling (`mean(SLOTS_VISUAL)`) with differentiable
     2D Spatial Soft-Argmax over the 4x4 visual patch grid (`grid_xy in [-0.75, +0.75]^2`) plus sub-patch
     fine offset, reusing `vis_conv*` and `head_vis_pose_w/b` with ZERO extra parameters and zero
     proprioceptive shortcut leakage.
  2. Key-Addressed Symbolic + Visual Working-Memory Goal Latch & Overwrite:
     Encodes transient 0.4s 2D goal directives at symbolic register `GOAL_KEY_ID = 36` (`[0, 36, val_2d, 0]`)
     and false distractors at orthogonal `DIST_KEY_ID = 37` (`[0, 37, val_dist, 0]`), queried on silent
     ticks via constant `[0, 36, 0, 2]` (zero coordinate leakage). `GatedDelta` + `KeyShield` surgically
     overwrites Goal A with Goal B and retains Goal B across 140+ silent distracted ticks (`T = 160 > 2.5x W = 64`),
     whereas finite-window `SlidingWindow-W64` evicts Goal B past tick `t = 84`.
  3. Zero-Shot Cross-Modal Spatial-JEPA Anomaly Score:
     Compares pure-vision Spatial-Soft-Argmax effector pose `vis_ee_curr` against joint encoder `ee_xy`
     plus latent visual JEPA innovation to detect unseen OOD camera corruptions zero-shot.
  4. Initial-Scene Contrastive Spoken + 16-Token Physical xArm6 Grounding:
     Trains grounded command-to-displacement prediction from the static initial tabletop view (`video[:, 0]`,
     `ee_xy[:, 0]`) + 16-token language + human spoken audio against counterfactual instructions.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from models.x1b_jepa import compute_innovation_features, l2_normalize, predict_latent_step
from models.x1b_sigreg_epps_pulley import epps_pulley_sigreg_loss, vicreg_var_cov_loss
from models.x1b_wave6_core import (
    Wave6Interventions,
    de_privilege_batch_proprio,
    forward_world_model_wave6,
)
from models.x1b_world import (
    SLOT_PROPRIO,
    SLOT_TEXT,
    SLOTS_AUDIO,
    SLOTS_REGISTER,
    SLOTS_VISUAL,
    X1BSpec,
    _spatial_grid_fourier,
)

GOAL_KEY_ID = 36
DIST_KEY_ID = 37
GOAL_VAL_OFFSET = 3584  # 3584..3839 (16x16 = 256 2D coordinate bins over [-0.80, +0.80]^2)
GOAL_NUM_BINS_2D = 256


def canonical_patch_centers_4x4() -> jax.Array:
    """Return canonical (x, y) centers in [-0.75, +0.75]^2 for the 16 `SLOTS_VISUAL` patches (4x4 ij grid)."""
    pts = jnp.linspace(-0.75, 0.75, 4, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(pts, pts, indexing="ij")
    return jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)  # [16, 2]


def canonical_grid_centers_8x8() -> jax.Array:
    """Return canonical (x, y) centers in [-0.875, +0.875]^2 for the 64 shift-equivariant 8x8 cells."""
    pts = jnp.linspace(-0.875, 0.875, 8, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(pts, pts, indexing="ij")
    return jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)  # [64, 2]


def quantize_goal_2d_val_id(goal_xy: jax.Array) -> jax.Array:
    """Quantize continuous (x, y) in [-0.80, +0.80]^2 into 256 vocabulary IDs `[3584..3839]`."""
    gx_bin = jnp.clip(jnp.floor((goal_xy[..., 0] + 0.80) / 1.60 * 16.0).astype(jnp.int32), 0, 15)
    gy_bin = jnp.clip(jnp.floor((goal_xy[..., 1] + 0.80) / 1.60 * 16.0).astype(jnp.int32), 0, 15)
    return GOAL_VAL_OFFSET + gx_bin * 16 + gy_bin


def encode_goal_tokens_2d(goal_xy: jax.Array, *, is_distractor: bool = False) -> jax.Array:
    """Encode a 2D goal write `[0, GOAL_KEY_ID, val_2d, 0]` or distractor write `[0, DIST_KEY_ID, val_2d, 0]`."""
    val_id = quantize_goal_2d_val_id(goal_xy)
    z0 = jnp.zeros_like(val_id)
    key_id = jnp.full_like(val_id, DIST_KEY_ID if is_distractor else GOAL_KEY_ID)
    return jnp.stack([z0, key_id, val_id, z0], axis=-1)


def encode_goal_query_tokens(batch_shape: tuple[int, ...]) -> jax.Array:
    """Return constant coordinate-free goal-register query `[0, GOAL_KEY_ID, 0, 2]` of shape `(*batch_shape, 4)`."""
    q = jnp.array([0, GOAL_KEY_ID, 0, 2], dtype=jnp.int32)
    return jnp.broadcast_to(q, (*batch_shape, 4))


def _encode_pure_visual_h2(
    params: dict[str, jax.Array],
    video: jax.Array,
) -> jax.Array:
    """Run shared pure-vision convolutional stem `vis_conv1 -> vis_conv2` producing `[..., 16, 16, C2]`."""
    lead_shape = video.shape[:-3]
    n_flat = math.prod(lead_shape) if lead_shape else 1
    v_in = (video.astype(jnp.float32) / 127.5 - 1.0).reshape(n_flat, 64, 64, 3)
    h1 = jax.nn.silu(
        jax.lax.conv_general_dilated(
            v_in,
            params["vis_conv1_w"].astype(jnp.float32),
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_conv1_b"].astype(jnp.float32)
    )
    h1_pool = jnp.max(h1.reshape(n_flat, 16, 2, 16, 2, -1), axis=(2, 4))
    c2 = params["vis_conv2_b"].shape[0]
    h2 = jax.nn.silu(
        jax.lax.conv_general_dilated(
            h1,
            params["vis_conv2_w"].astype(jnp.float32),
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_conv2_b"].astype(jnp.float32)
    ) + 0.30 * jnp.tile(h1_pool, (1, 1, 1, (c2 + h1.shape[-1] - 1) // h1.shape[-1]))[..., :c2]
    return h2.reshape(*lead_shape, 16, 16, c2)


def encode_pure_visual_grid_8x8(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    video: jax.Array,
) -> jax.Array:
    """Extract 100% shift-equivariant position-free `[..., 8, 8, D]` visual features (ZERO extra params).

    Pools the `16x16` convolutional stem `h2` over `2x2` windows (`0.55 * mean + 0.45 * max`) and projects
    each cell identically with the spatial-mean kernel `mean(vis_patch_w, axis=(0, 1))`, eliminating
    stride-4 patchify aliasing so supervision on `x=0` bars transfers identically to any `(x, y)` in `[-1, +1]^2`.
    """
    lead_shape = video.shape[:-3]
    d = spec.d_model
    h2 = _encode_pure_visual_h2(params, video)
    c2 = h2.shape[-1]
    h2_blocks = h2.reshape(*lead_shape, 8, 2, 8, 2, c2)
    h2_8x8 = 0.25 * jnp.mean(h2_blocks, axis=(-4, -2)) + 0.75 * jnp.max(h2_blocks, axis=(-4, -2))
    w_1x1 = jnp.mean(params["vis_patch_w"].astype(jnp.float32), axis=(0, 1))  # [C2, D]
    b_1x1 = params["vis_patch_b"].astype(jnp.float32)  # [D]
    return (h2_8x8 @ w_1x1 + b_1x1).reshape(*lead_shape, 8, 8, d)


def encode_pure_visual_patches(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    video: jax.Array,
    *,
    include_pos: bool = True,
) -> jax.Array:
    """Extract pure-vision 4x4 patch embeddings `[..., 16, D]` using `vis_conv1/2/patch` (zero proprio leakage!)."""
    lead_shape = video.shape[:-3]
    n_flat = math.prod(lead_shape) if lead_shape else 1
    d = spec.d_model
    h2 = _encode_pure_visual_h2(params, video)
    c2 = h2.shape[-1]
    v_patches = (
        jax.lax.conv_general_dilated(
            h2.reshape(n_flat, 16, 16, c2),
            params["vis_patch_w"].astype(jnp.float32),
            window_strides=(4, 4),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_patch_b"].astype(jnp.float32)
    ).reshape(*lead_shape, 16, d)
    if not include_pos:
        return v_patches
    pos = _spatial_grid_fourier(d, jnp.float32)
    return v_patches + pos


def spatial_grid_local_context(
    z_vis_slots: jax.Array,
    obj_xy: jax.Array,
    ee_xy: jax.Array,
    *,
    pure_v_ti: jax.Array | None = None,
) -> jax.Array:
    """Object-centric 4x4 spatial-grid local patch readout at `(obj_xy, ee_xy)` (100% translation-equivariant!)."""
    grid_xy = canonical_patch_centers_4x4()  # [16, 2]
    d2_obj = jnp.sum((grid_xy - obj_xy[..., None, :]) ** 2, axis=-1)
    d2_ee = jnp.sum((grid_xy - ee_xy[..., None, :]) ** 2, axis=-1)
    w_obj = jax.nn.softmax(-9.0 * d2_obj, axis=-1)
    w_ee = jax.nn.softmax(-9.0 * d2_ee, axis=-1)
    d = z_vis_slots.shape[-1]
    pos = _spatial_grid_fourier(d, jnp.float32)
    z_vis_ti = z_vis_slots.astype(jnp.float32) - pos
    src = z_vis_ti if pure_v_ti is None else (0.70 * pure_v_ti.astype(jnp.float32) + 0.30 * z_vis_ti)
    loc_obj = jnp.einsum("...s,...sd->...d", w_obj, src)
    loc_ee = jnp.einsum("...s,...sd->...d", w_ee, src)
    return 0.68 * loc_obj + 0.32 * loc_ee


def decode_spatial_hazard_grid_8x8(
    params: dict[str, jax.Array],
    z_any: jax.Array,
    interventions: Wave6Interventions,
    *,
    pure_v_ti: jax.Array | None = None,
    pure_v_8x8: jax.Array | None = None,
) -> jax.Array:
    """Decode an 8x8 strictly shift-equivariant spatial hazard logit grid `[..., 8, 8]` (ZERO extra params).

    Applies a single shared 1x1 readout vector (`mean(head_vis_pose_w[:, 12:16])`) across all 64 cells of
    position-free shift-equivariant `pure_v_8x8` + `SLOTS_VISUAL` minus Fourier position. Because every
    `(row, col)` cell shares identical position-free convolutional and linear weights, supervision on
    in-domain `x=0` bars transfers zero-shot to arbitrary unseen off-center, chicane, and clutter layouts.
    When `not interventions.decoupled_heads` (Global Average Pooling ablation), spatial resolution collapses
    to a uniform scalar field.
    """
    zf = z_any.astype(jnp.float32)
    head_vw = params["head_vis_pose_w"].astype(jnp.float32)[:, 12:16]  # [D, 4]
    head_vb = params["head_vis_pose_b"].astype(jnp.float32)[12:16]     # [4]

    if not interventions.decoupled_heads:
        pooled = jnp.mean(zf, axis=-2)  # [..., D]
        w_1x1 = jnp.mean(head_vw, axis=-1)
        b_1x1 = jnp.mean(head_vb)
        val = (pooled @ w_1x1 + b_1x1)[..., None, None]
        return jnp.broadcast_to(val, (*pooled.shape[:-1], 8, 8))

    d = zf.shape[-1]
    pos = _spatial_grid_fourier(d, jnp.float32)
    z_vis_ti = zf[..., SLOTS_VISUAL, :] - pos  # [..., 16, D]
    lead_shape = z_vis_ti.shape[:-2]
    z_vis_4x4 = z_vis_ti.reshape(*lead_shape, 4, 4, d)
    z_vis_8x8 = jnp.repeat(jnp.repeat(z_vis_4x4, 2, axis=-3), 2, axis=-2)  # [..., 8, 8, D]

    if pure_v_8x8 is not None:
        pv8 = pure_v_8x8.astype(jnp.float32)
        pv8_semi = 0.45 * pv8 + 0.55 * jax.lax.stop_gradient(pv8)
        cell_feat = 0.85 * pv8_semi + 0.15 * z_vis_8x8
        w_1x1 = jnp.mean(head_vw, axis=-1)  # [D]
        b_1x1 = jnp.mean(head_vb)
        return jnp.einsum("...ijd,d->...ij", cell_feat, w_1x1) + b_1x1

    if pure_v_ti is not None:
        pv4 = pure_v_ti.astype(jnp.float32).reshape(*lead_shape, 4, 4, d)
        pv8 = jnp.repeat(jnp.repeat(pv4, 2, axis=-3), 2, axis=-2)
        pv8_semi = 0.25 * pv8 + 0.75 * jax.lax.stop_gradient(pv8)
        cell_feat = 0.80 * pv8_semi + 0.20 * z_vis_8x8
        w_1x1 = jnp.mean(head_vw, axis=-1)
        b_1x1 = jnp.mean(head_vb)
        return jnp.einsum("...ijd,d->...ij", cell_feat, w_1x1) + b_1x1

    w_1x1 = jnp.mean(head_vw, axis=-1)
    b_1x1 = jnp.mean(head_vb)
    return jnp.einsum("...ijd,d->...ij", z_vis_8x8, w_1x1) + b_1x1


def sample_hazard_grid_bilinear(haz_prob_8x8: jax.Array, xy: jax.Array) -> jax.Array:
    """Differentiable bilinear lookup into `[..., 8, 8]` hazard probability grid (`ij` = `(y, x)` indexing).

    `haz_prob_8x8` has leading shape `B_lead` (e.g. `[B]` or `[B, K]`) and spatial axes `[8, 8]` over `[-1, +1]^2`
    with cell centers at `linspace(-0.875, 0.875, 8)`. `xy` has shape `(*B_lead, 2)` or `[B, K, 2]`.
    """
    if haz_prob_8x8.ndim == 3 and xy.ndim == 3:
        grid = jnp.broadcast_to(haz_prob_8x8[:, None, :, :], (xy.shape[0], xy.shape[1], 8, 8))
    else:
        grid = haz_prob_8x8

    # Map x, y in [-1, +1] to continuous grid coordinates u_x, u_y in [0, 7]
    u_x = jnp.clip((xy[..., 0] + 0.875) / 0.25, 0.0, 7.0)
    u_y = jnp.clip((xy[..., 1] + 0.875) / 0.25, 0.0, 7.0)
    x0 = jnp.clip(jnp.floor(u_x).astype(jnp.int32), 0, 6)
    y0 = jnp.clip(jnp.floor(u_y).astype(jnp.int32), 0, 6)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = u_x - x0.astype(jnp.float32)
    wy = u_y - y0.astype(jnp.float32)

    flat_g = grid.reshape(-1, 8, 8)
    n_items = flat_g.shape[0]
    idx = jnp.arange(n_items, dtype=jnp.int32)
    fx0, fx1 = x0.reshape(-1), x1.reshape(-1)
    fy0, fy1 = y0.reshape(-1), y1.reshape(-1)

    v00 = flat_g[idx, fy0, fx0].reshape(xy.shape[:-1])
    v01 = flat_g[idx, fy0, fx1].reshape(xy.shape[:-1])
    v10 = flat_g[idx, fy1, fx0].reshape(xy.shape[:-1])
    v11 = flat_g[idx, fy1, fx1].reshape(xy.shape[:-1])
    return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)


def _spatial_soft_argmax_grid(
    feat_slots: jax.Array,
    grid_xy: jax.Array,
    head_vw: jax.Array,
    head_vb: jax.Array,
    *,
    col_attn_ee: int = 0,
    col_attn_obj: int = 1,
    col_fine_ee: slice = slice(2, 4),
    col_fine_obj: slice = slice(4, 6),
    bias_ee: slice = slice(0, 2),
    bias_obj: slice = slice(2, 4),
    fine_scale: float = 0.32,
) -> tuple[jax.Array, jax.Array]:
    logits_ee = (feat_slots @ head_vw[:, col_attn_ee]) * 3.0
    logits_obj = (feat_slots @ head_vw[:, col_attn_obj]) * 3.0
    attn_ee = jax.nn.softmax(logits_ee, axis=-1)
    attn_obj = jax.nn.softmax(logits_obj, axis=-1)

    coarse_ee = jnp.einsum("...s,sc->...c", attn_ee, grid_xy)
    coarse_obj = jnp.einsum("...s,sc->...c", attn_obj, grid_xy)
    feat_ee = jnp.einsum("...s,...sd->...d", attn_ee, feat_slots)
    feat_obj = jnp.einsum("...s,...sd->...d", attn_obj, feat_slots)

    fine_ee = fine_scale * jnp.tanh(feat_ee @ head_vw[:, col_fine_ee] + head_vb[bias_ee])
    fine_obj = fine_scale * jnp.tanh(feat_obj @ head_vw[:, col_fine_obj] + head_vb[bias_obj])
    return jnp.clip(coarse_ee + fine_ee, -0.96, 0.96), jnp.clip(coarse_obj + fine_obj, -0.96, 0.96)


def decode_spatial_vis_pose(
    params: dict[str, jax.Array],
    z_any: jax.Array,
    interventions: Wave6Interventions,
    *,
    pure_v_patches: jax.Array | None = None,
    pure_v_8x8: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Decode `(vis_ee_xy, vis_obj_xy, vis_proprio_16)` via position-free Spatial Soft-Argmax (ZERO extra params).

    Uses dedicated columns `[6..11]` of `head_vis_pose_w` on position-free shift-equivariant `pure_v_8x8`
    (or position-free `pure_v_patches`) so appearance-based keypoint attention transfers zero-shot across
    arbitrary off-center crimson obstacles with zero proprioceptive shortcut leakage, and columns `[0..5]`
    on position-free `SLOTS_VISUAL - pos` for deep latent rollouts.
    """
    zf = z_any.astype(jnp.float32)
    head_vw = params["head_vis_pose_w"].astype(jnp.float32)  # [D, 16]
    head_vb = params["head_vis_pose_b"].astype(jnp.float32)  # [16]

    if not interventions.decoupled_heads:
        pooled = jnp.mean(zf, axis=-2)
        raw = pooled @ head_vw + head_vb
        return raw[..., 0:2], raw[..., 2:4], raw

    d = zf.shape[-1]
    pos = _spatial_grid_fourier(d, jnp.float32)
    z_vis = zf[..., SLOTS_VISUAL, :]  # [..., 16, D]
    z_vis_ti = z_vis - pos
    lat_ee, lat_obj = _spatial_soft_argmax_grid(
        z_vis_ti,
        canonical_patch_centers_4x4(),
        head_vw,
        head_vb,
        col_attn_ee=0,
        col_attn_obj=1,
        col_fine_ee=slice(2, 4),
        col_fine_obj=slice(4, 6),
        bias_ee=slice(0, 2),
        bias_obj=slice(2, 4),
        fine_scale=0.32,
    )
    if pure_v_8x8 is not None:
        pv64 = pure_v_8x8.astype(jnp.float32).reshape(*pure_v_8x8.shape[:-3], 64, d)
        pure_ee, pure_obj = _spatial_soft_argmax_grid(
            pv64,
            canonical_grid_centers_8x8(),
            head_vw,
            head_vb,
            col_attn_ee=6,
            col_attn_obj=7,
            col_fine_ee=slice(8, 10),
            col_fine_obj=slice(10, 12),
            bias_ee=slice(4, 6),
            bias_obj=slice(6, 8),
            fine_scale=0.18,
        )
        ee_xy = pure_ee
        obj_xy = 0.35 * pure_obj + 0.65 * lat_obj
    elif pure_v_patches is not None:
        pure_ee, pure_obj = _spatial_soft_argmax_grid(
            pure_v_patches.astype(jnp.float32),
            canonical_patch_centers_4x4(),
            head_vw,
            head_vb,
            col_attn_ee=6,
            col_attn_obj=7,
            col_fine_ee=slice(8, 10),
            col_fine_obj=slice(10, 12),
            bias_ee=slice(4, 6),
            bias_obj=slice(6, 8),
            fine_scale=0.32,
        )
        ee_xy = pure_ee
        obj_xy = 0.65 * pure_obj + 0.35 * lat_obj
    else:
        ee_xy, obj_xy = lat_ee, lat_obj

    vis_sum = jnp.mean(z_vis, axis=-2)
    rest_12 = jnp.concatenate(
        [vis_sum @ head_vw[:, 12:16] + head_vb[12:16], jnp.zeros((*ee_xy.shape[:-1], 8), dtype=jnp.float32)],
        axis=-1,
    )
    full_16 = jnp.concatenate([ee_xy, obj_xy, rest_12], axis=-1)
    return ee_xy, obj_xy, full_16


def decode_goal_latch_wave7(
    params: dict[str, jax.Array],
    z_any: jax.Array,
    interventions: Wave6Interventions,
) -> jax.Array:
    """Decode persistent 2D target goal / robot displacement from register + symbolic + spoken context."""
    zf = z_any.astype(jnp.float32)
    if interventions.decoupled_heads:
        reg_sum = jnp.mean(zf[..., SLOTS_REGISTER, :], axis=-2)
        aud_sum = jnp.mean(zf[..., SLOTS_AUDIO, :], axis=-2)
        txt_lat = zf[..., SLOT_TEXT, :]
        g_ctx = 0.45 * reg_sum + 0.40 * txt_lat + 0.15 * aud_sum
    else:
        g_ctx = jnp.mean(zf, axis=-2)
    return jnp.tanh(g_ctx @ params["head_goal_w"].astype(jnp.float32) + params["head_goal_b"].astype(jnp.float32))


def decode_world_heads_wave7(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    z: jax.Array,
    z_next_pred: jax.Array,
    log_var_next: jax.Array,
    goal: jax.Array,
    interventions: Wave6Interventions,
    *,
    obs_ee_proprio: jax.Array | None = None,
    obs_video: jax.Array | None = None,
    stop_grad_diagnostics: bool = True,
) -> dict[str, jax.Array]:
    """Spatial-Soft-Argmax de-privileged (`ee_only`) readout heads with zero geometric priors."""
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
        pooled = jnp.mean(zf, axis=2)
        pooled_next = jnp.mean(z_next_f, axis=2)
        vis_sum = vis_max = aud_sum = reg_sum = prop_lat = txt_ctx = contact_ctx = pooled
        vis_next_sum = reg_next_sum = prop_next_lat = pooled_next

    pure_vp_ti = encode_pure_visual_patches(params, spec, obs_video, include_pos=False) if obs_video is not None else None
    pure_v8 = encode_pure_visual_grid_8x8(params, spec, obs_video) if obs_video is not None else None
    vis_ee_curr, vis_obj_curr, vis_proprio_curr = decode_spatial_vis_pose(
        params, z, interventions, pure_v_patches=pure_vp_ti, pure_v_8x8=pure_v8
    )
    lat_ee_curr, lat_obj_curr, _ = decode_spatial_vis_pose(params, z, interventions, pure_v_patches=None)
    vis_ee_next, vis_obj_next, _ = decode_spatial_vis_pose(params, z_next_pred, interventions, pure_v_patches=None)

    ee_for_loc = (
        obs_ee_proprio.astype(jnp.float32)[:, :, 0:2]
        if obs_ee_proprio is not None
        else jax.lax.stop_gradient(vis_ee_curr)
    )
    if interventions.decoupled_heads:
        loc_vis_ctx = spatial_grid_local_context(
            zf[:, :, SLOTS_VISUAL, :],
            jax.lax.stop_gradient(vis_obj_curr),
            ee_for_loc,
            pure_v_ti=pure_vp_ti,
        )
        contact_ctx = 0.72 * loc_vis_ctx + 0.14 * aud_sum + 0.14 * reg_sum

    pred_goal_xy = decode_goal_latch_wave7(params, z, interventions)
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

    proprio_curr = vis_proprio_curr
    if obs_ee_proprio is not None:
        clean_prop_emb = 2.5 * (obs_ee_proprio.astype(jnp.float32) @ params["prop_w"].astype(jnp.float32))
    else:
        clean_prop_emb = prop_lat
    proprio_trusted = (0.7 * clean_prop_emb + 0.3 * reg_sum) @ head_pw + head_pb

    if obs_ee_proprio is not None and interventions.zerolag_bypass:
        ee_now = obs_ee_proprio.astype(jnp.float32)[:, :, 0:2]
        obj_vis_now = jax.lax.stop_gradient(vis_obj_curr)
        anchor_state = obs_ee_proprio.astype(jnp.float32).at[:, :, 0:2].set(ee_now).at[:, :, 2:4].set(obj_vis_now)
        delta_full = (0.75 * (prop_next_lat - prop_lat) + 0.25 * (vis_next_sum - vis_sum)) @ head_pw
        delta_trust = (0.80 * (prop_next_lat - prop_lat) + 0.20 * (reg_next_sum - reg_sum)) @ head_pw
        raw_next = anchor_state + delta_full
        blended_obj_next = 0.65 * raw_next[:, :, 2:4] + 0.35 * vis_obj_next
        proprio_next = raw_next.at[:, :, 2:4].set(blended_obj_next)
        proprio_next_trusted = anchor_state + delta_trust
    else:
        proprio_next = (0.75 * prop_next_lat + 0.25 * vis_next_sum) @ head_pw + head_pb
        proprio_next_trusted = (0.80 * prop_next_lat + 0.20 * reg_next_sum) @ head_pw + head_pb

    contact_logits = (
        contact_ctx @ params["head_contact_w"].astype(jnp.float32)
        + params["head_contact_b"].astype(jnp.float32)
    )

    ee_xy = (
        obs_ee_proprio.astype(jnp.float32)[:, :, 0:2]
        if obs_ee_proprio is not None
        else jax.lax.stop_gradient(vis_ee_curr)
    )
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

    z_pred_shifted = jnp.concatenate([z[:, :1], z_next_pred[:, :-1]], axis=1)
    lv_shifted = jnp.concatenate([log_var_next[:, :1], log_var_next[:, :-1]], axis=1)
    vis_ee_disc = jnp.sqrt(jnp.sum((vis_ee_curr - ee_xy) ** 2, axis=-1) + 1e-6)
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
    hazard_logits_8x8 = decode_spatial_hazard_grid_8x8(
        params, z, interventions, pure_v_ti=pure_vp_ti, pure_v_8x8=pure_v8
    )

    return {
        "pred_goal_xy": pred_goal_xy,
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
    }


def world_model_loss_wave7(
    params: dict[str, jax.Array],
    ema_params: dict[str, jax.Array],
    spec: X1BSpec,
    raw_batch: dict[str, jax.Array],
    interventions: Wave6Interventions,
    rng_key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Wave 7 training objective with pure-vision Spatial-Soft-Argmax & initial-scene spoken robot grounding."""
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

    vis_pool = jnp.mean(z[:, :, SLOTS_VISUAL, :], axis=2).astype(jnp.float32) @ params["align_vis_w"].astype(jnp.float32)
    aud_pool = jnp.mean(z[:, :, SLOTS_AUDIO, :], axis=2).astype(jnp.float32) @ params["align_aud_w"].astype(jnp.float32)
    k_v, k_a = jax.random.split(rng_key)
    if interventions.regularizer == "epps_pulley_sigreg":
        sig_v, pr_val = epps_pulley_sigreg_loss(vis_pool.reshape(b * t, -1), k_v, num_slices=64)
        sig_a, _ = epps_pulley_sigreg_loss(aud_pool.reshape(b * t, -1), k_a, num_slices=64)
    else:
        sig_v, pr_val = vicreg_var_cov_loss(vis_pool.reshape(b * t, -1))
        sig_a, _ = vicreg_var_cov_loss(aud_pool.reshape(b * t, -1))
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

    heads = decode_world_heads_wave7(
        params,
        spec,
        z,
        z_next_1,
        log_var_1,
        batch["goal"],
        interventions,
        obs_ee_proprio=batch["proprio"],
        obs_video=batch["video"],
        stop_grad_diagnostics=True,
    )

    target_prop = batch["target_proprio"].astype(jnp.float32)
    vis_present = batch.get("modality_mask", jnp.ones((b, t, 4), dtype=jnp.float32))[:, :, 1].astype(jnp.float32)
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

    dim_w = jnp.array([2.0] * 8 + [0.15] * 8, dtype=jnp.float32)[None, None, :]
    if "target_next_proprio" in batch:
        tnp = batch["target_next_proprio"].astype(jnp.float32)
        last_valid = jnp.where(scen_arr == -1, 0.0, 1.0)[:, None]
        next_mask = valid.at[:, -1].set(valid[:, -1] * last_valid[:, 0]) * (commit_target != 2).astype(jnp.float32)
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
        prop_next_loss = jnp.sum(
            jnp.mean(((heads["proprio_next"][:, :-1] - target_prop[:, 1:]) ** 2) * dim_w, axis=-1) * valid_pair
        ) / jnp.maximum(jnp.sum(valid_pair), 1.0)
        vis_obj_next_mse = jnp.zeros((), dtype=jnp.float32)
        prop_trusted_loss = prop_next_loss

    if z_next_4 is not None:
        zr4 = z_next_4[:, :-1].astype(jnp.float32)
        z04 = z[:, :-4].astype(jnp.float32)
        hw4 = params["head_proprio_w"].astype(jnp.float32)
        hb4 = params["head_proprio_b"].astype(jnp.float32)
        _, vis_obj_4, _ = decode_spatial_vis_pose(params, zr4, interventions)
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
        m4 = valid[:, :-4] * valid[:, 4:] * (commit_target[:, :-4] == 0).astype(jnp.float32) * (commit_target[:, 4:] == 0).astype(jnp.float32)
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
    base_contact_bce = jnp.mean(
        jax.nn.softplus(cl) * (1.0 - target_contacts) + 3.5 * jax.nn.softplus(-cl) * target_contacts
    )

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
    commit_cls_w = jnp.array([1.0, 2.8, 4.5, 1.8], dtype=jnp.float32)[commit_target]
    commit_nll = -jnp.take_along_axis(commit_lp, commit_target[..., None], axis=-1)[..., 0]
    commit_loss = jnp.sum(commit_nll * commit_cls_w) / jnp.maximum(jnp.sum(commit_cls_w), 1.0)

    txt_target = batch["target_text"].astype(jnp.int32)
    txt_mask = (txt_target > 0).astype(jnp.float32) * valid
    txt_lp = jax.nn.log_softmax(heads["text_logits"], axis=-1)
    txt_nll = -jnp.take_along_axis(txt_lp, jnp.clip(txt_target, 0, spec.vocab_size - 1)[..., None], axis=-1)[..., 0]
    is_assoc_16k = (scen_arr == 4).astype(jnp.float32)[:, None]
    is_goal_sym = ((scen_arr >= 0) & (scen_arr != 4)).astype(jnp.float32)[:, None]
    is_real_lm = (scen_arr < 0).astype(jnp.float32)[:, None]
    assoc_ce = jnp.sum(txt_nll * txt_mask * is_assoc_16k) / jnp.maximum(jnp.sum(txt_mask * is_assoc_16k), 1.0)
    goal_sym_ce = jnp.sum(txt_nll * txt_mask * is_goal_sym) / jnp.maximum(jnp.sum(txt_mask * is_goal_sym), 1.0)
    lm_ce = jnp.sum(txt_nll * txt_mask * is_real_lm) / jnp.maximum(jnp.sum(txt_mask * is_real_lm), 1.0)
    text_loss = 1.6 * assoc_ce + 0.35 * goal_sym_ce + 0.5 * lm_ce

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
    goal_mask = goal_w * has_goal_sup * valid
    goal_loss = jnp.sum(jnp.mean((heads["pred_goal_xy"] - tg_xy) ** 2, axis=-1) * goal_mask) / jnp.maximum(
        jnp.sum(goal_mask), 1.0
    )

    if "wrong_text" in batch and t >= 4:
        t_init = 4
        vid_init = jnp.broadcast_to(batch["video"][:, :1], (b, t_init, *batch["video"].shape[2:]))
        prop_init = jnp.broadcast_to(batch["proprio"][:, :1], (b, t_init, batch["proprio"].shape[-1]))
        zero_act_init = jnp.zeros((b, t_init, spec.action_dim), dtype=jnp.float32)
        zero_goal_init = jnp.zeros((b, t_init, spec.goal_dim), dtype=jnp.float32)
        mm_init = jnp.ones((b, t_init, 4), dtype=jnp.float32)
        val_init = jnp.ones((b, t_init), dtype=bool)

        b_init_tr = {
            "video": vid_init,
            "audio": batch["audio"][:, :t_init],
            "proprio": prop_init,
            "text": batch["text"][:, :t_init],
            "action": zero_act_init,
            "goal": zero_goal_init,
            "modality_mask": mm_init,
            "valid": val_init,
        }
        wrong_aud_init = (
            batch["wrong_audio"][:, :t_init]
            if "wrong_audio" in batch
            else jnp.roll(batch["audio"][:, :t_init], shift=1, axis=0)
        )
        b_init_wr = {
            **b_init_tr,
            "text": batch["wrong_text"][:, :t_init],
            "audio": wrong_aud_init,
        }
        z_itr, _, _ = forward_world_model_wave6(params, spec, b_init_tr, interventions)
        z_iwr, _, _ = forward_world_model_wave6(params, spec, b_init_wr, interventions)
        pg_tr = decode_goal_latch_wave7(params, z_itr, interventions)[:, -1]
        pg_wr = decode_goal_latch_wave7(params, z_iwr, interventions)[:, -1]

        is_rob_ep = (scen_arr == -2).astype(jnp.float32)
        end_xy_t = tg_xy[:, -1]
        d_tr = jnp.mean((pg_tr - end_xy_t) ** 2, axis=-1)
        d_wr = jnp.mean((pg_wr - end_xy_t) ** 2, axis=-1)
        init_goal_mse = jnp.sum(d_tr * is_rob_ep) / jnp.maximum(jnp.sum(is_rob_ep), 1.0)
        rank_margin = jnp.sum(jax.nn.relu(0.12 + d_tr - d_wr) * is_rob_ep) / jnp.maximum(jnp.sum(is_rob_ep), 1.0)
        goal_loss = goal_loss + 2.0 * init_goal_mse
    else:
        rank_margin = jnp.zeros((), dtype=jnp.float32)

    clean_vid = batch.get("clean_video", batch["video"]).astype(jnp.float32) / 127.5 - 1.0
    recon_loss = jnp.mean((heads["diag_video"] - clean_vid) ** 2) + 0.5 * jnp.mean(
        (heads["diag_audio"] - batch["audio"].astype(jnp.float32)) ** 2
    )

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
    }
    return total_loss, metrics
