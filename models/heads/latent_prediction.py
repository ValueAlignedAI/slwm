"""Latent prediction head stub."""

from __future__ import annotations

from typing import Any, Mapping

from models.heads.base import BaseOutputHead
from models.types import TensorSpec, ensure_latent


class LatentPredictionHead(BaseOutputHead):
    """Return future-latent prediction shape metadata.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shape: ``latent_prediction: FloatTensor[B,T,D]``.
    """

    head_name = "latent_prediction"
    channel = "internal"

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        b, t, d = ensure_latent(z_world, "z_world")
        output = super().forward(z_world, query=query, metadata=metadata)
        output["latent_prediction"] = TensorSpec((b, t, d), "float32", "latent_prediction")
        return output
