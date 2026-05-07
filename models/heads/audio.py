"""Audio decoder head stub."""

from __future__ import annotations

from typing import Any, Mapping

from models.heads.base import BaseOutputHead, output_spec_from_latent


class AudioDecoderHead(BaseOutputHead):
    """Return audio-latent output shape metadata.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shape: ``audio_latents: FloatTensor[B,T,A]``.
    """

    head_name = "audio_decoder"
    channel = "audio"
    modality = "audio"
    diagnostic_only = False
    default_score = 0.5
    default_source_tag = "predicted"
    default_intention = "emit_audio_latents"

    def __init__(self, audio_dim: int = 80) -> None:
        self.audio_dim = int(audio_dim)

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        output = super().forward(z_world, query=query, metadata=metadata)
        output["audio_latents"] = output_spec_from_latent(z_world, self.audio_dim, "audio_latents")
        output["metadata"]["audio_dim"] = self.audio_dim
        output["proposal"]["output_key"] = "audio_latents"
        return output
