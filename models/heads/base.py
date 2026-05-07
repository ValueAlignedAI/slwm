"""Base output-head proposal contract."""

from __future__ import annotations

from typing import Any, Mapping

from models.module import ShapeModule
from models.types import TensorSpec, ensure_latent


class BaseOutputHead(ShapeModule):
    """Shape-only proposal head.

    Forward input shape:
        ``z_world: FloatTensor[B,T,D]``.

    Forward output contract:
        ``{"proposal": dict, "metadata": dict}``, optionally with an output
        ``TensorSpec`` under a modality-specific key.
    """

    head_name = "base"
    channel = "internal"
    diagnostic_only = True

    def _proposal(self, z_world: Any, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        b, t, d = ensure_latent(z_world, "z_world")
        return {
            "head": self.head_name,
            "channel": self.channel,
            "latent_shape": [b, t, d],
            "diagnostic_only": self.diagnostic_only,
            "committed": False,
            "status": "proposal",
            "metadata": dict(metadata or {}),
        }

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Create a proposal from ``z_world`` without decoding real outputs."""

        return {
            "proposal": self._proposal(z_world, metadata),
            "metadata": {
                "head": self.head_name,
                "query_provided": query is not None,
                "implementation": "i0_shape_contract_stub",
            },
        }


def output_spec_from_latent(z_world: Any, output_dim: int, name: str) -> TensorSpec:
    """Create ``FloatTensor[B,T,output_dim]`` metadata from latent shape."""

    b, t, _ = ensure_latent(z_world, "z_world")
    return TensorSpec((b, t, int(output_dim)), "float32", name)
