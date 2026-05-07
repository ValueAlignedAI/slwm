"""Uncertainty/source head stub."""

from __future__ import annotations

from typing import Any, Mapping

from data.contract import SOURCE_TAGS
from models.heads.base import BaseOutputHead
from models.types import TensorSpec, ensure_latent


class UncertaintyHead(BaseOutputHead):
    """Return uncertainty/source shape metadata.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shapes:
        ``uncertainty: FloatTensor[B,T,1]``
        ``source_logits: FloatTensor[B,T,S]`` where ``S=len(SOURCE_TAGS)``.
    """

    head_name = "uncertainty"
    channel = "internal"

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        b, t, _ = ensure_latent(z_world, "z_world")
        output = super().forward(z_world, query=query, metadata=metadata)
        output["uncertainty"] = TensorSpec((b, t, 1), "float32", "uncertainty")
        output["source_logits"] = TensorSpec((b, t, len(SOURCE_TAGS)), "float32", "source_logits")
        output["metadata"]["source_tags"] = list(SOURCE_TAGS)
        return output
