import numpy as np

from training.synthetic_metrics import frequency_recovery_error, masked_mse, phase_error, prediction_metric_bundle, spectral_magnitude_loss
from training.synthetic_signals import SUPPORTED_SYNTHETIC_TASKS, make_synthetic_signal_batch


def test_t0_synthetic_batches_are_deterministic_and_shaped() -> None:
    for task in SUPPORTED_SYNTHETIC_TASKS:
        batch_a = make_synthetic_signal_batch(task, batch_size=2, context_length=16, latent_dim=3, seed=11, split="train")
        batch_b = make_synthetic_signal_batch(task, batch_size=2, context_length=16, latent_dim=3, seed=11, split="train")

        assert batch_a.input_latents.shape == (2, 16, 3)
        assert batch_a.target_latents.shape == (2, 16, 3)
        assert batch_a.input_mask.shape == (2, 16)
        assert batch_a.loss_mask.shape == (2, 16)
        assert np.array_equal(batch_a.input_latents, batch_b.input_latents)
        assert np.array_equal(batch_a.target_latents, batch_b.target_latents)
        assert batch_a.metadata["dataset"] == "synthetic_signal_t0"
        assert batch_a.metadata["source_tags"]["input"] == "observed"
        assert batch_a.metadata["modality_mix"] == {"synthetic_signal": 1.0}


def test_t0_missing_span_uses_observed_input_mask_and_reconstruction_loss_mask() -> None:
    batch = make_synthetic_signal_batch(
        "missing_span_reconstruction",
        batch_size=2,
        context_length=20,
        latent_dim=2,
        seed=3,
        split="train",
        missing_fraction=0.25,
    )

    assert np.any(~batch.input_mask)
    assert np.array_equal(batch.loss_mask, ~batch.input_mask)
    assert np.all(batch.input_latents[~batch.input_mask] == 0.0)
    assert batch.metadata["source_tags"]["target"] == "reconstructed"


def test_t0_metric_bundle_rewards_exact_prediction() -> None:
    target = make_synthetic_signal_batch("sine_mixture", batch_size=1, context_length=32, latent_dim=2, seed=5).target_latents
    noisy = target + 0.25

    exact_metrics = prediction_metric_bundle(target, target)
    noisy_metrics = prediction_metric_bundle(noisy, target)

    assert masked_mse(target, target) == 0.0
    assert spectral_magnitude_loss(target, target) == 0.0
    assert phase_error(target, target) < 1e-9
    assert frequency_recovery_error(target, target) == 0.0
    assert exact_metrics["mse"] < noisy_metrics["mse"]
    assert exact_metrics["spectral_loss"] <= noisy_metrics["spectral_loss"]
