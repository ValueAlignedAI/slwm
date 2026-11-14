"""Unit tests for Wave 7 Spatial-Grid De-Privileged Grounding modules (`SLWM-v2`)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from evals.x1b_eval_wave7 import (
    init_unseen_blocked_worlds,
    plan_spatial_geometry_free_mppi,
    run_wave7_evaluation,
)
from models.x1b_wave6_core import Wave6Interventions
from models.x1b_wave7_core import (
    GOAL_KEY_ID,
    DIST_KEY_ID,
    GOAL_VAL_OFFSET,
    canonical_patch_centers_4x4,
    decode_spatial_vis_pose,
    encode_goal_query_tokens,
    encode_goal_tokens_2d,
    world_model_loss_wave7,
)
from models.x1b_world import count_parameters, init_x1b_params, make_x1b_spec
from scripts.train.x1b_run_wave7_lane import build_wave7_lanes
from scripts.train.x1b_train_wave7 import generate_in_domain_deprivileged_sim_batch_v7


def test_spatial_soft_argmax_zero_extra_params_and_shapes() -> None:
    spec = make_x1b_spec(profile="x_tiny", arch="slwm_x1b")
    params = init_x1b_params(spec, seed=2001)
    assert count_parameters(params)["strict_total"] > 0

    grid = canonical_patch_centers_4x4()
    assert grid.shape == (16, 2)
    assert float(jnp.min(grid)) == -0.75 and float(jnp.max(grid)) == 0.75

    iv = Wave6Interventions()
    z_seq = jax.random.normal(jax.random.key(0), (3, 5, 24, spec.d_model))
    ee_s, obj_s, full_s = decode_spatial_vis_pose(params, z_seq, iv)
    assert ee_s.shape == (3, 5, 2)
    assert obj_s.shape == (3, 5, 2)
    assert full_s.shape == (3, 5, 16)

    z_mppi = jax.random.normal(jax.random.key(1), (3, 16, 24, spec.d_model))
    ee_k, obj_k, _ = decode_spatial_vis_pose(params, z_mppi, iv)
    assert ee_k.shape == (3, 16, 2)
    assert obj_k.shape == (3, 16, 2)


def test_goal_token_encoding_disjoint_keys_and_query() -> None:
    pts = jnp.array([[-0.75, -0.75], [0.75, 0.75]], dtype=jnp.float32)
    t_goal = encode_goal_tokens_2d(pts, is_distractor=False)
    t_dist = encode_goal_tokens_2d(pts, is_distractor=True)
    t_q = encode_goal_query_tokens((2,))
    assert t_goal.shape == (2, 4)
    assert jnp.all(t_goal[:, 1] == GOAL_KEY_ID)
    assert jnp.all(t_dist[:, 1] == DIST_KEY_ID)
    assert jnp.all((t_goal[:, 2] >= GOAL_VAL_OFFSET) & (t_goal[:, 2] < GOAL_VAL_OFFSET + 256))
    assert int(t_goal[0, 2]) != int(t_goal[1, 2])
    assert jnp.all(t_q[:, 1] == GOAL_KEY_ID) and jnp.all(t_q[:, 3] == 2) and jnp.all(t_q[:, 2] == 0)


def test_wave7_loss_and_smoke_evaluation_finite() -> None:
    spec = make_x1b_spec(profile="x_tiny", arch="slwm_x1b")
    params = init_x1b_params(spec, seed=2001)
    iv = Wave6Interventions()
    batch = generate_in_domain_deprivileged_sim_batch_v7(jax.random.key(42), 4, seq_len=24, vocab_size=spec.vocab_size)
    loss, metrics = world_model_loss_wave7(params, params, spec, batch, iv, jax.random.key(99))
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["vis_obj_mse"])

    st = init_unseen_blocked_worlds(jax.random.key(7), batch_size=8)
    z0 = jax.random.normal(jax.random.key(8), (8, 24, spec.d_model))
    u0, info = plan_spatial_geometry_free_mppi(
        params, spec, z0, iv, st["targ_a"], st["ee_xy"], jax.random.key(9), horizon=6, num_candidates=16, num_elites=4, cem_iters=2
    )
    assert u0.shape == (8, 4)
    assert info["decoded_obj_xy"].shape == (8, 2)

    ev = run_wave7_evaluation(params, spec, iv, seed=2001, smoke=True)
    assert "unseen_layout_deprivileged_mpc" in ev
    assert "beyond_window_T160_distractor_switch" in ev
    assert "unseen_ood_sensor_corruptions" in ev


def test_wave7_453m_parameter_parity_and_lane_partition() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs/x1b/study_x1b_wave7.json"
    cfg = json.loads(cfg_path.read_text())
    lanes = build_wave7_lanes(cfg)
    assert sorted(lanes) == ["w7_lane1", "w7_lane2", "w7_lane3"]
    all_tags = [j["tag"] for _, jobs in lanes.values() for j in jobs]
    assert len(all_tags) == 12 and len(set(all_tags)) == 12

    s_slwm = make_x1b_spec(profile="x450m", arch="slwm_x1b")
    s_swa = make_x1b_spec(profile="x450m", arch="sliding_window_transformer")
    s_ssm = make_x1b_spec(profile="x450m", arch="diagonal_ssm_hybrid")
    assert s_slwm.d_model == s_swa.d_model == s_ssm.d_model == 1152
