"""Unit tests for Wave 6 zero-compromise modules: Epps-Pulley SIGReg, Seed-Aware Bootstrap,
Geometry-Free De-Privileged Latent-MPPI, Explicit-Latch Controller, and Unseen Layouts/Corruptions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_unseen_worlds import (
    apply_unseen_ood_sensor_corruption,
    init_unseen_layout_worlds,
    mask_privileged_proprio,
    render_unseen_world_rgb,
    step_unseen_layout_world,
)
from evals.x1b_learned_mpc import (
    plan_explicit_latch_reactive_controller,
    plan_geometry_free_latent_mppi,
)
from evals.x1b_stats import hierarchical_seed_bootstrap_ci
from models.x1b_sigreg_epps_pulley import epps_pulley_sigreg_loss, vicreg_var_cov_loss
from models.x1b_world import forward_world_model, init_x1b_params, make_x1b_spec


def test_epps_pulley_sigreg_distinguishes_isotropic_gaussian_from_collapse() -> None:
    k_gauss, k_slice = jax.random.split(jax.random.key(42))
    z_gauss = jax.random.normal(k_gauss, (128, 64), dtype=jnp.float32)
    # Rank-1 collapsed distribution and strongly bimodal non-Gaussian distribution (unit variance!)
    u1 = jax.random.normal(k_gauss, (128, 1), dtype=jnp.float32)
    z_rank1 = jnp.broadcast_to(u1, (128, 64))
    z_bimodal = jnp.where(z_gauss >= 0.0, 1.0, -1.0)

    loss_g, pr_g = epps_pulley_sigreg_loss(z_gauss, k_slice, num_slices=64)
    loss_r1, pr_r1 = epps_pulley_sigreg_loss(z_rank1, k_slice, num_slices=64)
    loss_bi, _ = epps_pulley_sigreg_loss(z_bimodal, k_slice, num_slices=64)
    v_loss_bi, _ = vicreg_var_cov_loss(z_bimodal)

    assert float(pr_g) > 0.50 and float(pr_r1) < 0.05
    assert float(loss_g) < float(loss_r1)
    # Crucial theoretical distinction: Rademacher +/-1 has unit variance & zero covariance (VICReg ~ 0),
    # whereas Epps-Pulley SIGReg detects its non-Gaussian characteristic function!
    assert float(loss_bi) > float(loss_g) * 2.0
    assert float(v_loss_bi) < 0.15


def test_hierarchical_seed_bootstrap_captures_between_seed_variance() -> None:
    # 3 seeds x 96 episodes where seeds differ substantially (0.95, 0.70, 0.45)
    rng = np.random.default_rng(7)
    s0 = rng.binomial(1, 0.95, size=96)
    s1 = rng.binomial(1, 0.70, size=96)
    s2 = rng.binomial(1, 0.45, size=96)
    mat = np.stack([s0, s1, s2], axis=0).astype(np.float64)

    res = hierarchical_seed_bootstrap_ci(mat, n_boot=5000)
    hier_width = res["hierarchical_ci95"][1] - res["hierarchical_ci95"][0]
    pool_width = res["episode_pooled_ci95"][1] - res["episode_pooled_ci95"][0]
    assert res["n_seeds"] == 3 and res["n_episodes_per_seed"] == 96
    assert hier_width > pool_width * 2.0


def test_geometry_free_mppi_and_unseen_beyond_window_worlds() -> None:
    spec = make_x1b_spec("x_tiny", "slwm_x1b")
    params = init_x1b_params(spec, seed=11)
    st = init_unseen_layout_worlds(jax.random.key(99), batch_size=8)

    # Verify de-privileged proprioception zeroes all object coordinates
    ee_only_p = mask_privileged_proprio(st["ee_xy"], st["ee_vel"], jnp.zeros((8,), dtype=jnp.float32))
    assert ee_only_p.shape == (8, 16)
    assert np.allclose(np.asarray(ee_only_p[:, 0:2]), np.asarray(ee_only_p[:, 2:4]))
    assert np.allclose(np.asarray(ee_only_p[:, 8:]), 0.0)

    rgb = render_unseen_world_rgb(st, st["targ_a"], jnp.ones((8,), dtype=bool), res=64)
    assert rgb.shape == (8, 64, 64, 3) and rgb.dtype == jnp.uint8

    batch = {
        "video": rgb[:, None, ...],
        "audio": jnp.zeros((8, 1, 64), dtype=jnp.float32),
        "proprio": ee_only_p[:, None, :],
        "text": jnp.zeros((8, 1, 4), dtype=jnp.int32),
        "action": jnp.zeros((8, 1, 4), dtype=jnp.float32),
    }
    z, _ = forward_world_model(params, spec, batch)
    u_mppi, info = plan_geometry_free_latent_mppi(
        params,
        spec,
        z[:, 0],
        st["targ_a"],
        st["ee_xy"],
        jax.random.key(123),
        horizon=4,
        num_candidates=12,
        num_elites=4,
        cem_iters=2,
    )
    assert u_mppi.shape == (8, 4) and jnp.all(jnp.isfinite(u_mppi))

    u_latch, latched = plan_explicit_latch_reactive_controller(
        st["ee_xy"], info["decoded_obj_xy"], st["targ_a"], jnp.zeros_like(st["targ_a"])
    )
    assert u_latch.shape == (8, 4) and np.allclose(np.asarray(latched), np.asarray(st["targ_a"]))

    # Step T=160 > W=64 ticks and test all 4 unseen OOD sensor corruptions
    st_next, c_hit = step_unseen_layout_world(st, u_mppi)
    assert st_next["obj_xy"].shape == (8, 2) and c_hit.shape == (8,)

    vid_seq = jnp.broadcast_to(rgb[:, None, ...], (8, 12, 64, 64, 3))
    for kind in ("frozen_camera_frame", "cross_episode_replay_swap", "unseen_90deg_rotation", "half_occluded_camera"):
        corr = apply_unseen_ood_sensor_corruption(vid_seq, kind)
        assert corr.shape == vid_seq.shape and corr.dtype == jnp.uint8
        assert np.array_equal(np.asarray(corr[:, :8]), np.asarray(vid_seq[:, :8]))


def test_wave6_end_to_end_loo_key_shield_theorem_and_vault(tmp_path) -> None:
    from evals.x1b_eval_wave6 import eval_symbolic_key_drift_and_recall
    from models.x1b_wave6_core import Wave6Interventions
    from scripts.train.x1b_train_wave6 import train_and_evaluate_wave6

    spec = make_x1b_spec("x_tiny", "slwm_x1b")
    params = init_x1b_params(spec, seed=303)

    # Verify Symbolic-Key Shielding invariance theorem directly:
    res_shielded = eval_symbolic_key_drift_and_recall(
        params, spec, Wave6Interventions(key_shield=True), n_episodes=4
    )
    res_unshielded = eval_symbolic_key_drift_and_recall(
        params, spec, Wave6Interventions(key_shield=False), n_episodes=4
    )
    assert res_shielded["symbolic_key_visual_drift"] < 1e-5
    assert res_unshielded["symbolic_key_visual_drift"] > 1e-4

    # Create tiny mock grounded + untouched vault NPZs and run full Wave 6 train + 5-panel eval smoke
    rng = np.random.default_rng(19)
    for fname, sp_keys in [("grounded.npz", ("train", "dev")), ("vault.npz", ("vault",))]:
        arrs = {}
        for sp in sp_keys:
            n = 12
            arrs[f"{sp}_video"] = rng.integers(0, 255, size=(n, 32, 64, 64, 3), dtype=np.uint8)
            arrs[f"{sp}_spoken_audio"] = rng.standard_normal((n, 32, 64)).astype(np.float16)
            arrs[f"{sp}_ee_xy"] = np.clip(np.cumsum(0.03 * rng.standard_normal((n, 32, 2)), axis=1), -0.9, 0.9).astype(np.float32)
            arrs[f"{sp}_action_xy"] = np.clip(0.4 * rng.standard_normal((n, 32, 2)), -1.0, 1.0).astype(np.float32)
            arrs[f"{sp}_length"] = rng.integers(12, 32, size=(n,), dtype=np.int32)
            arrs[f"{sp}_true_text_ids"] = rng.integers(10, 500, size=(n, 16), dtype=np.int32)
            arrs[f"{sp}_wrong_text_ids"] = rng.integers(10, 500, size=(n, 16), dtype=np.int32)
        np.savez_compressed(tmp_path / fname, **arrs)

    rec = train_and_evaluate_wave6(
        tag="w6-smoke-slwm",
        profile="x_tiny",
        arch="slwm_x1b",
        regularizer="epps_pulley_sigreg",
        key_shield=True,
        zerolag_bypass=True,
        decoupled_heads=True,
        seed=2001,
        steps=3,
        batch_size=4,
        seq_len=16,
        out_dir=tmp_path / "out",
        grounded_robot_path=str(tmp_path / "grounded.npz"),
        vault_robot_path=str(tmp_path / "vault.npz"),
        smoke=True,
    )
    assert rec["eval_summary"]["vault_step4_mm"] is not None
    assert 0.0 <= rec["eval_summary"]["T160_post_perturb_goal_b_success"] <= 1.0
