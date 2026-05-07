import math

import numpy as np

from data import SOURCE_TAGS
from models import NumpySLWMCore, SLWMCoreConfig, make_i2_dummy_batch, slwm_parameter_breakdown_from_config
from models.adapters import AudioSignalAdapter, TextSignalAdapter, VisualSignalAdapter
from models.heads import UncertaintyHead
from models.latent_field import LatentSignalField
from models.processor import SignalWorldProcessor
from utils import load_config


def _tiny_config() -> SLWMCoreConfig:
    return SLWMCoreConfig.from_mapping(load_config("configs/slwm/slwm_i2_tiny_smoke.json"))


def test_i2_config_flags_and_parameter_ablation_counts() -> None:
    full = _tiny_config()
    no_spectral = SLWMCoreConfig.from_mapping(load_config("configs/slwm/slwm_i2_no_spectral_smoke.json"))
    no_longconv = SLWMCoreConfig.from_mapping(load_config("configs/slwm/slwm_i2_no_longconv_smoke.json"))

    assert full.use_spectral_mixer is True
    assert no_spectral.use_spectral_mixer is False
    assert full.use_long_conv is True
    assert no_longconv.use_long_conv is False
    assert full.use_gated_mlp is True

    full_model = NumpySLWMCore(full)
    ablated_model = NumpySLWMCore(no_spectral)
    no_longconv_model = NumpySLWMCore(no_longconv)
    full_counts = full_model.parameter_count_breakdown().as_dict()
    ablated_counts = ablated_model.parameter_count_breakdown().as_dict()
    no_longconv_counts = no_longconv_model.parameter_count_breakdown().as_dict()

    assert full_counts["adapters"]["text_code"] > 0
    assert full_counts["adapters"]["audio"] > 0
    assert full_counts["adapters"]["visual_video"] > 0
    assert full_counts["processor"] > ablated_counts["processor"]
    assert full_counts["processor"] > no_longconv_counts["processor"]
    assert full_counts["heads"]["latent_prediction"] > 0
    assert full_counts["heads"]["uncertainty"] > 0
    assert full_counts["policy"] == 0
    assert full_counts["total"] == full_model.parameter_count_breakdown().registry_module_counts()["total"]
    assert slwm_parameter_breakdown_from_config(full).as_dict() == full_counts


def test_i2_124m_style_formula_parameter_count_does_not_instantiate_large_model() -> None:
    cfg = SLWMCoreConfig.from_mapping(load_config("configs/slwm/slwm_i2_124m_style.json"))
    breakdown = slwm_parameter_breakdown_from_config(cfg).as_dict()

    assert cfg.context_length == 1024
    assert cfg.latent_dim == 768
    assert breakdown["adapters"]["text_code"] > 0
    assert breakdown["processor"] > 0
    assert breakdown["heads"]["latent_prediction"] > 0
    assert breakdown["heads"]["uncertainty"] > 0
    assert breakdown["policy"] == 0
    assert breakdown["total"] == (
        breakdown["adapters"]["total"] + breakdown["processor"] + breakdown["heads"]["total"] + breakdown["policy"]
    )


def test_i2_adapters_and_latent_field_pack_real_multimodal_arrays() -> None:
    cfg = _tiny_config()
    batch = make_i2_dummy_batch(cfg, batch_size=2, seed=7)
    adapters = [
        TextSignalAdapter(latent_length=cfg.context_length, latent_dim=cfg.latent_dim, vocab_size=cfg.text_vocab_size, seed=1),
        AudioSignalAdapter(
            latent_length=cfg.context_length,
            latent_dim=cfg.latent_dim,
            audio_feature_dim=cfg.audio_feature_dim,
            seed=2,
        ),
        VisualSignalAdapter(
            latent_length=cfg.context_length,
            latent_dim=cfg.latent_dim,
            visual_feature_dim=cfg.visual_feature_dim,
            seed=3,
        ),
    ]
    packets = [
        adapters[0]({"input_ids": batch["text_tokens"]}),
        adapters[1]({"features": batch["audio_features"]}),
        adapters[2]({"features": batch["visual_features"]}),
    ]

    for packet, modality in zip(packets, ["text_code", "audio", "visual_video"], strict=True):
        assert packet["z"].shape == (2, cfg.context_length, cfg.latent_dim)
        assert packet["mask"].shape == (2, cfg.context_length)
        assert packet["metadata"]["modality"] == modality
        assert packet["metadata"]["observed"] is True
        assert packet["metadata"]["implementation"].startswith("i2_trainable_numpy")

    field = LatentSignalField(latent_length=cfg.context_length, latent_dim=cfg.latent_dim).from_adapter_outputs(packets)
    assert field["z"].shape == (2, cfg.context_length, cfg.latent_dim)
    assert field["mask"].shape == (2, cfg.context_length)
    assert field["metadata"]["modalities"] == ["text_code", "audio", "visual_video"]
    assert field["metadata"]["implementation"] == "i2_numpy_concat_pad_pack"
    # All three dummy modalities fit into the tiny context because packing uses
    # each adapter's observed copied length, not its padded latent_length.
    assert int(np.sum(field["mask"][0])) == sum(int(packet["metadata"]["copied_length"]) for packet in packets)


def test_i2_processor_preserves_shape_with_and_without_spectral_block() -> None:
    cfg = _tiny_config()
    z = np.random.default_rng(0).normal(size=(2, cfg.context_length, cfg.latent_dim)).astype(np.float64)
    mask = np.ones((2, cfg.context_length), dtype=bool)

    processor = SignalWorldProcessor(config=cfg)
    output = processor(z, mask=mask)
    assert output["z_world"].shape == z.shape
    assert output["aux"]["ablation_flags"]["use_spectral_mixer"] is True

    no_spectral = SLWMCoreConfig.from_mapping(load_config("configs/slwm/slwm_i2_no_spectral_smoke.json"))
    ablated_output = SignalWorldProcessor(config=no_spectral)(z, mask=mask)
    assert ablated_output["z_world"].shape == z.shape
    assert ablated_output["aux"]["ablation_flags"]["use_spectral_mixer"] is False

    no_longconv = SLWMCoreConfig.from_mapping(load_config("configs/slwm/slwm_i2_no_longconv_smoke.json"))
    no_longconv_output = SignalWorldProcessor(config=no_longconv)(z, mask=mask)
    assert no_longconv_output["z_world"].shape == z.shape
    assert no_longconv_output["aux"]["ablation_flags"]["use_long_conv"] is False


def test_i2_core_forward_backward_on_text_audio_visual_dummy_batch() -> None:
    cfg = _tiny_config()
    model = NumpySLWMCore(cfg)
    batch = make_i2_dummy_batch(cfg, batch_size=2, seed=3)
    target = np.random.default_rng(4).normal(size=(2, cfg.context_length, cfg.latent_dim)).astype(np.float64)
    optimizer = model.make_optimizer(learning_rate=0.001)

    optimizer.zero_grad()
    loss, output = model.loss_and_backward(batch, target)
    grad_norm = sum(float(np.sum(np.abs(param.grad))) for param in model.parameters())

    assert math.isfinite(loss)
    assert loss > 0.0
    assert output["z_world"].shape == (2, cfg.context_length, cfg.latent_dim)
    assert output["latent_prediction"]["latent_prediction"].shape == (2, cfg.context_length, cfg.latent_dim)
    assert output["uncertainty"]["uncertainty"].shape == (2, cfg.context_length, 1)
    assert output["uncertainty"]["source_logits"].shape == (2, cfg.context_length, len(SOURCE_TAGS))
    assert grad_norm > 0.0
    assert math.isfinite(optimizer.step())


def test_i2_uncertainty_head_backward_interface_ready() -> None:
    cfg = _tiny_config()
    head = UncertaintyHead(cfg.latent_dim, seed=5)
    z_world = np.random.default_rng(8).normal(size=(2, cfg.context_length, cfg.latent_dim)).astype(np.float64)
    output = head(z_world)
    grad_z = head.backward(
        grad_uncertainty=np.ones_like(output["uncertainty"]),
        grad_source_logits=np.ones_like(output["source_logits"]),
    )

    assert output["metadata"]["source_tags"] == list(SOURCE_TAGS)
    assert grad_z.shape == z_world.shape
    assert np.all(np.isfinite(grad_z))
    assert sum(float(np.sum(np.abs(param.grad))) for param in head.parameters()) > 0.0
