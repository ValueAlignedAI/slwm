"""Reconstruction head stub."""

from __future__ import annotations

from typing import Any, Mapping

from models.heads.base import BaseOutputHead, output_spec_from_latent


class ReconstructionHead(BaseOutputHead):
    """Return reconstruction shape metadata.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shape: ``reconstruction: FloatTensor[B,T,D_out]`` where ``D_out``
    defaults to ``D``.
    """

    head_name = "reconstruction"
    channel = "internal"
    default_source_tag = "reconstructed"
    default_intention = "diagnostic_reconstruction"

    def __init__(self, output_dim: int | None = None) -> None:
        self.output_dim = output_dim

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        latent_shape = self._proposal(z_world, metadata)["latent_shape"]
        output_dim = self.output_dim or int(latent_shape[2])
        output = BaseOutputHead.forward(self, z_world, query=query, metadata=metadata)
        output["reconstruction"] = output_spec_from_latent(z_world, output_dim, "reconstruction")
        return output
