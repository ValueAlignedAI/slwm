from models.baselines.gpt2_decoder import GPT2DecoderConfig, NumpyGPT2DecoderBaseline
from models.baselines.parameter_count import gpt2_parameter_breakdown, vanilla_multimodal_parameter_breakdown
from models.baselines.vanilla_multimodal_transformer import NumpyVanillaMultimodalTransformerBaseline, VanillaMultimodalConfig


def test_gpt2_small_style_parameter_count_matches_reference() -> None:
    breakdown = gpt2_parameter_breakdown(
        vocab_size=50257,
        context_length=1024,
        n_layer=12,
        n_embd=768,
        intermediate_size=3072,
        tie_embeddings=True,
    )

    assert breakdown.total == 124_439_808
    assert breakdown.transformer_blocks == 85_054_464


def test_vanilla_multimodal_reference_count_is_within_documented_tolerance() -> None:
    breakdown = vanilla_multimodal_parameter_breakdown(
        text_vocab_size=50257,
        target_vocab_size=1024,
        context_length=1024,
        n_layer=12,
        n_embd=768,
        intermediate_size=3072,
        audio_feature_dim=80,
        visual_feature_dim=256,
    )

    assert breakdown.total == 125_489_152
    assert abs(breakdown.total - 124_000_000) / 124_000_000 < 0.02


def test_tiny_instantiated_counts_match_formula_counts() -> None:
    gpt_cfg = GPT2DecoderConfig(vocab_size=32, context_length=8, n_layer=1, n_embd=16, n_head=2, intermediate_size=64)
    gpt_model = NumpyGPT2DecoderBaseline(gpt_cfg)
    assert gpt_model.parameter_count() == gpt_cfg.parameter_breakdown().total == 3_952

    mm_cfg = VanillaMultimodalConfig(
        text_vocab_size=32,
        target_vocab_size=24,
        context_length=9,
        n_layer=1,
        n_embd=16,
        n_head=2,
        intermediate_size=64,
        audio_feature_dim=5,
        visual_feature_dim=6,
    )
    mm_model = NumpyVanillaMultimodalTransformerBaseline(mm_cfg)
    assert mm_model.parameter_count() == mm_cfg.parameter_breakdown().total == 4_632
