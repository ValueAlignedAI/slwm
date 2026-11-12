"""Unit & integration tests for SLWM-X1B (~1.008B scale parity, causality, streaming, JEPA, MPC, CommitGate)."""

from __future__ import annotations

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from data.x1b_multimodal import generate_synthetic_world_batch, render_scene_64x64, step_physics_world
from evals.x1b_eval_suite import binary_auroc, paired_bootstrap_ci, run_full_evaluation
from evals.x1b_mpc import plan_latent_cem_mpc, plan_random_shooting_g8
from models.x1b_commit import route_proposals_with_commit_gate
from models.x1b_jepa import VALID_OBJECTIVES, predict_latent_step, sigreg_loss_and_pr, world_model_loss
from models.x1b_world import (
    TOTAL_SLOTS,
    VALID_ARCHS,
    count_parameters,
    count_spec_parameters,
    encode_signals,
    forward_world_model,
    init_streaming_memories,
    init_x1b_params,
    make_x1b_spec,
)
from scripts.train.x1b_train import load_bf16_weights, save_bf16_weights, save_portable_int8, train_and_evaluate


def test_1b_parameter_scale_and_strict_baseline_parity() -> None:
    """Verify flagship X1B is ~1.008B strict parameters and matches baselines within <0.01%."""
    s_1b = make_x1b_spec("x1b", "slwm_x1b")
    c_1b = count_spec_parameters(s_1b)["strict_total"]
    assert 0.98e9 <= c_1b <= 1.03e9, f"Expected ~1.00B parameters, got {c_1b:,}"

    for prof in ("x150m", "x450m", "x1b"):
        ref = count_spec_parameters(make_x1b_spec(prof, "slwm_x1b"))["strict_total"]
        for base_arch in ("dense_transformer", "scalar_retention_hybrid", "legacy_scalar_hybrid", "slwm_no_registers", "separate_modality_cores"):
            other = count_spec_parameters(make_x1b_spec(prof, base_arch))["strict_total"]
            rel_diff = abs(ref - other) / ref
            assert rel_diff < 1e-4, f"Parameter mismatch on {prof}/{base_arch}: {ref:,} vs {other:,} ({rel_diff:.6%})"


@pytest.mark.parametrize("arch", VALID_ARCHS)
def test_strict_temporal_causality_zero_future_leakage(arch: str) -> None:
    """Perturbing multimodal inputs at ticks t > t_cut must produce ZERO change in Z[:, :t_cut+1]."""
    spec = make_x1b_spec("x_tiny", arch)
    params = init_x1b_params(spec, seed=101)
    batch_a = generate_synthetic_world_batch(jax.random.key(7), batch_size=2, seq_len=10, vocab_size=spec.vocab_size)
    batch_b = dict(batch_a)
    t_cut = 4
    # Massively corrupt all modalities from tick 5 onwards
    batch_b["video"] = batch_a["video"].at[:, t_cut + 1 :].set(255 - batch_a["video"][:, t_cut + 1 :])
    batch_b["audio"] = batch_a["audio"].at[:, t_cut + 1 :].set(batch_a["audio"][:, t_cut + 1 :] + 50.0)
    batch_b["proprio"] = batch_a["proprio"].at[:, t_cut + 1 :].set(-99.0)
    batch_b["text"] = batch_a["text"].at[:, t_cut + 1 :].set(123)
    batch_b["action"] = batch_a["action"].at[:, t_cut + 1 :].set(-1.0)

    z_a, _ = forward_world_model(params, spec, batch_a)
    z_b, _ = forward_world_model(params, spec, batch_b)
    assert z_a.shape == (2, 10, TOTAL_SLOTS, spec.d_model)

    max_prefix_diff = float(jnp.max(jnp.abs(z_a[:, : t_cut + 1] - z_b[:, : t_cut + 1])))
    future_diff = float(jnp.max(jnp.abs(z_a[:, t_cut + 1 :] - z_b[:, t_cut + 1 :])))
    assert max_prefix_diff < 1e-5, f"Future leakage detected in {arch}: {max_prefix_diff}"
    assert future_diff > 1e-3, f"Model ignored future inputs in {arch}"


@pytest.mark.parametrize("arch", ("slwm_x1b", "dense_transformer", "scalar_retention_hybrid"))
def test_online_streaming_recurrence_matches_parallel_sequence(arch: str) -> None:
    """Step-by-step online carry execution must match full-sequence parallel forward pass."""
    spec = make_x1b_spec("x_tiny", arch, grad_checkpoint=False)
    params = init_x1b_params(spec, seed=202)
    batch = generate_synthetic_world_batch(jax.random.key(9), batch_size=2, seq_len=6, vocab_size=spec.vocab_size)

    z_full, _ = forward_world_model(params, spec, batch)
    memories = init_streaming_memories(spec, batch_size=2, cache_len=16)
    steps_z = []
    for t in range(6):
        step_b = {
            "video": batch["video"][:, t : t + 1],
            "audio": batch["audio"][:, t : t + 1],
            "proprio": batch["proprio"][:, t : t + 1],
            "text": batch["text"][:, t : t + 1],
            "action": batch["action"][:, t : t + 1],
            "modality_mask": batch["modality_mask"][:, t : t + 1],
            "valid": batch["valid"][:, t : t + 1],
        }
        z_t, memories = forward_world_model(params, spec, step_b, memories=memories, offset=t)
        steps_z.append(z_t)
    z_stream = jnp.concatenate(steps_z, axis=1)
    np.testing.assert_allclose(np.asarray(z_full), np.asarray(z_stream), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("objective_mode", VALID_OBJECTIVES)
def test_all_jepa_and_mse_objectives_produce_finite_gradients(objective_mode: str) -> None:
    """Every pre-registered objective variant must yield finite loss, finite gradients, and bounded PR."""
    spec = make_x1b_spec("x_tiny", "slwm_x1b", grad_checkpoint=True)
    params = init_x1b_params(spec, seed=303)
    ema_params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
    batch = generate_synthetic_world_batch(jax.random.key(11), batch_size=4, seq_len=8, vocab_size=spec.vocab_size)

    (loss, metrics), grads = jax.value_and_grad(
        lambda w: world_model_loss(w, ema_params, spec, batch, objective_mode=objective_mode),
        has_aux=True,
    )(params)
    assert np.isfinite(float(loss))
    assert 0.0 < float(metrics["pr"]) <= 1.0001
    grad_norms = [float(jnp.linalg.norm(g)) for g in jax.tree.leaves(grads)]
    assert all(np.isfinite(gn) for gn in grad_norms)
    assert sum(grad_norms) > 0.0


def test_thin_barrier_64x64_visibility_and_mpc_commit_gate() -> None:
    """Verify 64x64 rendering resolves the thin central barrier, CEM-MPC plans, and CommitGate routes."""
    ee = jnp.array([[-0.3, 0.0], [-0.3, 0.0]])
    obj = jnp.array([[-0.2, 0.0], [-0.2, 0.0]])
    theta = jnp.zeros((2,))
    targ = jnp.array([[0.4, 0.0], [0.4, 0.0]])
    img = render_scene_64x64(ee, obj, theta, targ, jnp.array([True, False]), jnp.array([True, True]))
    assert img.shape == (2, 64, 64, 3)
    # Central column (x=31..32) must differ clearly when thin barrier is present vs absent!
    center_diff = float(jnp.mean(jnp.abs(img[0, 22:42, 31:33].astype(jnp.float32) - img[1, 22:42, 31:33].astype(jnp.float32))))
    assert center_diff > 40.0, f"Thin barrier insufficiently visible at 64x64: diff={center_diff}"

    spec = make_x1b_spec("x_tiny", "slwm_x1b", grad_checkpoint=False)
    params = init_x1b_params(spec, seed=404)
    z_dummy = jax.random.normal(jax.random.key(1), (3, TOTAL_SLOTS, spec.d_model))
    goal = jnp.zeros((3, 4))
    act_mpc, info = plan_latent_cem_mpc(params, spec, z_dummy, goal, jax.random.key(2), horizon=4, num_candidates=12, num_elites=4, cem_iters=2)
    act_g8 = plan_random_shooting_g8(params, spec, z_dummy, goal, jax.random.key(3))
    assert act_mpc.shape == (3, 4)
    assert act_g8.shape == (3, 4)
    assert jnp.all(jnp.abs(act_mpc) <= 1.0)

    # Test PolicyCommitGate routing across all 4 modes + exploration safety lock
    logits = jnp.array([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    af = jnp.ones((4, 4)) * 0.8
    at = jnp.ones((4, 4)) * -0.5
    routed = route_proposals_with_commit_gate(logits, af, at)
    np.testing.assert_array_equal(np.asarray(routed["mode_id"]), np.array([0, 1, 2, 3]))
    np.testing.assert_allclose(np.asarray(routed["committed_action"][0]), 0.8, atol=1e-5)
    np.testing.assert_allclose(np.asarray(routed["committed_action"][1]), -0.5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(routed["committed_action"][3]), 0.0, atol=1e-5)

    exp_routed = route_proposals_with_commit_gate(logits, af, at, exploration_mode=True)
    assert exp_routed["status"] == "diagnostic-only"
    np.testing.assert_allclose(np.asarray(exp_routed["committed_action"]), 0.0, atol=1e-6)


def test_smoke_train_eval_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """End-to-end smoke test of x1b_train + 7-panel eval + bf16/int8 checkpoint serialization."""
    receipt = train_and_evaluate(
        tag="x1b-unit-smoke",
        profile="x_tiny",
        arch="slwm_x1b",
        objective="residual_ema_jepa_sigreg",
        seed=2001,
        steps=3,
        batch_size=4,
        seq_len=8,
        out_dir=tmp_path,
    )
    assert receipt["parameter_counts"]["strict_total"] == count_spec_parameters(make_x1b_spec("x_tiny", "slwm_x1b"))["strict_total"]
    bf16_file = tmp_path / "x1b-unit-smoke/weights_bf16.npz"
    int8_file = tmp_path / "x1b-unit-smoke/portable_int8.npz"
    assert bf16_file.is_file() and int8_file.is_file()
    restored = load_bf16_weights(bf16_file)
    assert set(restored.keys()) == set(init_x1b_params(make_x1b_spec("x_tiny", "slwm_x1b")).keys())
