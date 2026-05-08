import math

from models.baselines.continuous_transformer import ContinuousSignalTransformerConfig, NumpyContinuousSignalTransformerBaseline
from models.slwm_config import SLWMCoreConfig
from models.slwm_signal_predictor import NumpySLWMSignalPredictor
from training.synthetic_signals import make_synthetic_signal_batch


def test_continuous_transformer_forward_backward_on_t0_batch() -> None:
    cfg = ContinuousSignalTransformerConfig(context_length=12, latent_dim=4, n_layer=1, n_head=2, intermediate_size=8, seed=1)
    model = NumpyContinuousSignalTransformerBaseline(cfg)
    batch = make_synthetic_signal_batch("sine_mixture", batch_size=2, context_length=12, latent_dim=4, seed=1)
    optimizer = model.make_optimizer(learning_rate=0.001)

    optimizer.zero_grad()
    loss, output = model.loss_and_backward(
        batch.input_latents,
        batch.target_latents,
        input_mask=batch.input_mask,
        loss_mask=batch.loss_mask,
    )
    grad_norm = optimizer.step()

    assert output["prediction"].shape == (2, 12, 4)
    assert math.isfinite(loss)
    assert loss > 0.0
    assert math.isfinite(grad_norm)
    assert model.parameter_count() == model.module_parameter_counts()["total"]


def test_slwm_t0_signal_predictor_ablation_counts_and_backward() -> None:
    full_cfg = SLWMCoreConfig(
        context_length=12,
        latent_dim=4,
        n_layer=1,
        intermediate_size=8,
        spectral_modes=6,
        use_uncertainty_head=False,
        use_output_heads=False,
        use_policy_gate=False,
        seed=2,
    )
    no_spectral_cfg = SLWMCoreConfig(
        context_length=12,
        latent_dim=4,
        n_layer=1,
        intermediate_size=8,
        spectral_modes=6,
        use_spectral_mixer=False,
        use_uncertainty_head=False,
        use_output_heads=False,
        use_policy_gate=False,
        seed=2,
    )
    full = NumpySLWMSignalPredictor(full_cfg)
    no_spectral = NumpySLWMSignalPredictor(no_spectral_cfg)
    batch = make_synthetic_signal_batch("noisy_periodic_denoising", batch_size=2, context_length=12, latent_dim=4, seed=2)
    optimizer = full.make_optimizer(learning_rate=0.001)

    optimizer.zero_grad()
    loss, output = full.loss_and_backward(
        batch.input_latents,
        batch.target_latents,
        input_mask=batch.input_mask,
        loss_mask=batch.loss_mask,
    )
    grad_norm = optimizer.step()

    assert output["prediction"].shape == (2, 12, 4)
    assert math.isfinite(loss)
    assert math.isfinite(grad_norm)
    assert full.parameter_count_breakdown().registry_module_counts()["adapters"] == 0
    assert full.parameter_count() > no_spectral.parameter_count()
