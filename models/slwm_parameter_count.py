"""Parameter accounting for Sprint I2 SLWM core modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from data.contract import SOURCE_TAGS
from models.baselines.numpy_nn import Parameter
from models.slwm_config import SLWMCoreConfig


def count_parameters(parameters: Sequence[Parameter]) -> int:
    """Count trainable NumPy ``Parameter`` objects exactly."""

    return int(sum(parameter.size for parameter in parameters))


@dataclass(frozen=True)
class SLWMParameterBreakdown:
    """Exact instantiated trainable parameter count split by I2 buckets.

    Buckets follow the project reporting rule:
        adapters, processor, heads, policy, and total. Sprint I2 keeps policy at
        zero because advanced policy behavior is out of scope.
    """

    adapters: Mapping[str, int]
    processor: int
    heads: Mapping[str, int]
    policy: int = 0

    @property
    def adapter_total(self) -> int:
        return int(sum(self.adapters.values()))

    @property
    def head_total(self) -> int:
        return int(sum(self.heads.values()))

    @property
    def total(self) -> int:
        return self.adapter_total + int(self.processor) + self.head_total + int(self.policy)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable nested breakdown."""

        return {
            "adapters": {**dict(self.adapters), "total": self.adapter_total},
            "processor": int(self.processor),
            "heads": {**dict(self.heads), "total": self.head_total},
            "policy": int(self.policy),
            "total": self.total,
        }

    def registry_module_counts(self) -> dict[str, int]:
        """Return flat counts compatible with the experiment registry buckets."""

        return {
            "adapters": self.adapter_total,
            "processor": int(self.processor),
            "heads": self.head_total,
            "policy": int(self.policy),
            "decoders": 0,
            "embeddings": int(self.adapters.get("text_code", 0)),
            "total": self.total,
        }


def slwm_parameter_breakdown_from_config(config: SLWMCoreConfig) -> SLWMParameterBreakdown:
    """Compute formula parameter counts from config without instantiating arrays.

    This is used for 124M-style accounting reports where allocating the full
    NumPy smoke model would be unnecessarily expensive. The formula mirrors the
    I2 module definitions exactly.
    """

    d_model = int(config.latent_dim)
    context = int(config.context_length)
    d_ff = int(config.d_ff)

    adapters = {
        "text_code": int(config.text_vocab_size) * d_model + context * d_model,
        "audio": int(config.audio_feature_dim) * d_model + d_model + context * d_model,
        "visual_video": int(config.visual_feature_dim) * d_model + d_model + context * d_model,
    }

    per_block = 2 * d_model  # LayerNorm gamma/beta.
    if config.use_local_temporal_mixer:
        per_block += int(config.local_kernel_size) * d_model + d_model
    if config.use_spectral_mixer:
        per_block += int(config.n_spectral_modes) * d_model
    if config.use_long_conv:
        per_block += int(config.long_conv_kernel_size) * d_model + d_model
    if config.use_gated_mlp:
        per_block += (d_model * d_ff + d_ff)  # value projection
        per_block += (d_model * d_ff + d_ff)  # gate projection
        per_block += (d_ff * d_model + d_model)  # output projection
    processor = int(config.n_layer) * per_block

    heads: dict[str, int] = {}
    if config.use_latent_prediction_head:
        heads["latent_prediction"] = d_model * d_model + d_model
    if config.use_uncertainty_head:
        heads["uncertainty"] = (d_model + 1) + (d_model * len(SOURCE_TAGS) + len(SOURCE_TAGS))

    return SLWMParameterBreakdown(adapters=adapters, processor=processor, heads=heads, policy=0)


__all__ = ["SLWMParameterBreakdown", "count_parameters", "slwm_parameter_breakdown_from_config"]
