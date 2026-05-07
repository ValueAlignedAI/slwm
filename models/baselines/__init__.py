"""Sprint I1 baseline models and controls.

This namespace is limited to required baselines and shared utilities needed by
those baselines. It intentionally excludes SLWM novelty modules such as spectral
mixers, long-convolution/SSM blocks, policy learning, and uncertainty heads.
"""

from models.baselines.gpt2_decoder import GPT2DecoderConfig, NumpyGPT2DecoderBaseline
from models.baselines.null_random import RandomLogitBaseline, UniformLogitBaseline, shuffled_targets
from models.baselines.parameter_count import (
    GPT2ParameterBreakdown,
    MultimodalParameterBreakdown,
    gpt2_module_counts_for_registry,
    gpt2_parameter_breakdown,
    multimodal_module_counts_for_registry,
    vanilla_multimodal_parameter_breakdown,
)
from models.baselines.vanilla_multimodal_transformer import NumpyVanillaMultimodalTransformerBaseline, VanillaMultimodalConfig

__all__ = [
    "GPT2DecoderConfig",
    "GPT2ParameterBreakdown",
    "MultimodalParameterBreakdown",
    "NumpyGPT2DecoderBaseline",
    "NumpyVanillaMultimodalTransformerBaseline",
    "RandomLogitBaseline",
    "UniformLogitBaseline",
    "VanillaMultimodalConfig",
    "gpt2_module_counts_for_registry",
    "gpt2_parameter_breakdown",
    "multimodal_module_counts_for_registry",
    "shuffled_targets",
    "vanilla_multimodal_parameter_breakdown",
]
