"""No-op output head stub."""

from __future__ import annotations

from typing import Any, Mapping

from models.heads.base import BaseOutputHead


class NoOpHead(BaseOutputHead):
    """Return a no-op proposal without external decoding.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output contract: proposal channel ``none`` with no tensor output.
    """

    head_name = "noop"
    channel = "none"
    diagnostic_only = False

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        output = super().forward(z_world, query=query, metadata=metadata)
        output["proposal"]["intention"] = "observe_wait"
        return output
