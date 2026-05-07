import numpy as np

from models.baselines.gpt2_decoder import GPT2DecoderConfig, NumpyGPT2DecoderBaseline
from models.baselines.null_random import RandomLogitBaseline, UniformLogitBaseline, shuffled_targets
from models.baselines.vanilla_multimodal_transformer import NumpyVanillaMultimodalTransformerBaseline, VanillaMultimodalConfig
from training.baseline_smoke import make_gpt2_tiny_batch, make_multimodal_tiny_batch


def test_gpt2_decoder_forward_backward_shapes_and_gradients() -> None:
    cfg = GPT2DecoderConfig(vocab_size=32, context_length=8, n_layer=1, n_embd=16, n_head=2, intermediate_size=64, seed=0)
    model = NumpyGPT2DecoderBaseline(cfg)
    input_ids, target_ids = make_gpt2_tiny_batch(cfg, batch_size=2, length=8)

    logits = model.forward(input_ids)
    assert logits.shape == (2, 8, 32)

    optimizer = model.make_optimizer(learning_rate=0.01)
    optimizer.zero_grad()
    loss, _ = model.loss_and_backward(input_ids, target_ids)
    grad_norm = sum(float(np.sum(np.abs(param.grad))) for param in model.parameters())

    assert loss > 0.0
    assert grad_norm > 0.0
    optimizer.step()


def test_vanilla_multimodal_forward_backward_shapes_and_controls() -> None:
    cfg = VanillaMultimodalConfig(
        text_vocab_size=32,
        target_vocab_size=24,
        context_length=9,
        n_layer=1,
        n_embd=16,
        n_head=2,
        intermediate_size=64,
        audio_feature_dim=5,
        visual_feature_dim=6,
        seed=1,
    )
    model = NumpyVanillaMultimodalTransformerBaseline(cfg)
    batch = make_multimodal_tiny_batch(cfg, batch_size=2, seed=1)

    logits = model.forward(
        text_tokens=batch["text_tokens"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
    )
    assert logits.shape == (2, 9, 24)

    optimizer = model.make_optimizer(learning_rate=0.01)
    optimizer.zero_grad()
    loss, _ = model.loss_and_backward(
        text_tokens=batch["text_tokens"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        target_ids=batch["target_ids"],
    )
    grad_norm = sum(float(np.sum(np.abs(param.grad))) for param in model.parameters())
    assert loss > 0.0
    assert grad_norm > 0.0
    optimizer.step()

    assert UniformLogitBaseline(cfg.target_vocab_size).evaluate(batch["target_ids"]).parameter_count == 0
    assert RandomLogitBaseline(cfg.target_vocab_size, seed=1).evaluate(batch["target_ids"]).loss > 0.0
    assert shuffled_targets(batch["target_ids"], seed=2).shape == batch["target_ids"].shape
