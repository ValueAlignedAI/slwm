"""Visual/video decoder head stub."""

from __future__ import annotations

from typing import Any, Mapping

from models.heads.base import BaseOutputHead, output_spec_from_latent


class VisualDecoderHead(BaseOutputHead):
    """Return visual/video latent output shape metadata.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shape: ``visual_latents: FloatTensor[B,T,Vd]``.
    """

    head_name = "visual_decoder"
    channel = "visual"
    diagnostic_only = False

    def __init__(self, visual_dim: int = 768) -> None:
        self.visual_dim = int(visual_dim)

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        output = super().forward(z_world, query=query, metadata=metadata)
        output["visual_latents"] = output_spec_from_latent(z_world, self.visual_dim, "visual_latents")
        output["metadata"]["visual_dim"] = self.visual_dim
        return output
