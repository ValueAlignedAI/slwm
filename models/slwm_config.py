"""Configuration objects for Sprint I2 SLWM core modules.

The Sprint I2 implementation remains NumPy-only and tiny-run friendly while
preserving the project-wide canonical latent contract ``Z: FloatTensor[B,T,D]``.
Every novel processor component has an explicit boolean flag so ablations can
disable it from configuration without changing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SLWMCoreConfig:
    """Config for the Sprint I2 SLWM adapters, processor, and diagnostic heads.

    Shape contract:
        text tokens: ``IntTensor[B,T_text]``.
        audio latents/features: ``FloatTensor[B,T_audio,A]``.
        visual latents/features: ``FloatTensor[B,T_visual,V]``.
        shared latent field and processor IO: ``FloatTensor[B,T,D]`` where
        ``T=context_length`` and ``D=latent_dim``.

    Ablation flags:
        ``use_local_temporal_mixer``, ``use_spectral_mixer``,
        ``use_long_conv``, and ``use_gated_mlp`` gate the I2 processor blocks.
        ``use_latent_prediction_head`` and ``use_uncertainty_head`` gate the two
        I2 trainable heads.
    """

    context_length: int = 1024
    latent_dim: int = 768
    n_layer: int = 12
    text_vocab_size: int = 50_257
    audio_feature_dim: int = 80
    visual_feature_dim: int = 256
    intermediate_size: int | None = None
    local_kernel_size: int = 3
    long_conv_kernel_size: int = 15
    spectral_modes: int | None = None
    seed: int = 0
    parameter_accounting_mode: str = "strict"
    use_local_temporal_mixer: bool = True
    use_spectral_mixer: bool = True
    use_long_conv: bool = True
    use_gated_mlp: bool = True
    use_latent_prediction_head: bool = True
    use_uncertainty_head: bool = True

    @property
    def d_ff(self) -> int:
        """Hidden width for the gated MLP block."""

        return int(self.intermediate_size if self.intermediate_size is not None else 4 * self.latent_dim)

    @property
    def n_spectral_modes(self) -> int:
        """Number of DCT-like frequency modes used by the spectral mixer."""

        if self.spectral_modes is None:
            return int(self.context_length)
        return max(1, min(int(self.spectral_modes), int(self.context_length)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SLWMCoreConfig":
        """Construct from a config mapping.

        The loader accepts either a top-level model section or a direct model
        mapping. Boolean ablation flags may be supplied under
        ``architecture_flags`` or as direct ``use_*`` fields.
        """

        model = mapping.get("model", mapping) if isinstance(mapping.get("model", mapping), Mapping) else mapping
        flags = model.get("architecture_flags", {}) if isinstance(model.get("architecture_flags", {}), Mapping) else {}
        codecs = model.get("codecs", {}) if isinstance(model.get("codecs", {}), Mapping) else {}
        runtime = mapping.get("runtime", {}) if isinstance(mapping.get("runtime", {}), Mapping) else {}

        def flag(name: str, fallback: bool) -> bool:
            return bool(model.get(name, flags.get(name, fallback)))

        context_length = int(model.get("context_length", model.get("latent_length", cls.context_length)))
        return cls(
            context_length=context_length,
            latent_dim=int(model.get("latent_dim", model.get("n_embd", cls.latent_dim))),
            n_layer=int(model.get("n_layer", model.get("processor_layers", cls.n_layer))),
            text_vocab_size=int(model.get("text_vocab_size", model.get("vocab_size", cls.text_vocab_size))),
            audio_feature_dim=int(model.get("audio_feature_dim", codecs.get("audio_feature_dim", cls.audio_feature_dim))),
            visual_feature_dim=int(model.get("visual_feature_dim", codecs.get("visual_feature_dim", cls.visual_feature_dim))),
            intermediate_size=(None if model.get("intermediate_size") is None else int(model["intermediate_size"])),
            local_kernel_size=int(model.get("local_kernel_size", cls.local_kernel_size)),
            long_conv_kernel_size=int(model.get("long_conv_kernel_size", cls.long_conv_kernel_size)),
            spectral_modes=(None if model.get("spectral_modes") is None else int(model["spectral_modes"])),
            seed=int(model.get("seed", runtime.get("seed", cls.seed))),
            parameter_accounting_mode=str(model.get("parameter_accounting_mode", cls.parameter_accounting_mode)),
            use_local_temporal_mixer=flag("use_local_temporal_mixer", bool(flags.get("local_temporal_mixer", True))),
            use_spectral_mixer=flag("use_spectral_mixer", bool(flags.get("spectral_mixer", True))),
            use_long_conv=flag("use_long_conv", bool(flags.get("longconv_or_ssm", True))),
            use_gated_mlp=flag("use_gated_mlp", True),
            use_latent_prediction_head=flag("use_latent_prediction_head", bool(flags.get("latent_prediction", True))),
            use_uncertainty_head=flag("use_uncertainty_head", bool(flags.get("uncertainty_head", True))),
        )


__all__ = ["SLWMCoreConfig"]
