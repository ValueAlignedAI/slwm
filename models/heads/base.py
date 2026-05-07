"""Base output-head proposal contract for Sprint I3.

Output heads are proposal producers, not automatic external decoders.  The
policy/commit gate is responsible for changing a proposal into a committed
behavior.  Diagnostic/probe outputs are marked internal-only so exploration
cannot be confused with committed external behavior.
"""

from __future__ import annotations

from typing import Any, Mapping

from data.contract import MODALITY_IDS, SOURCE_TAGS
from models.module import ShapeModule
from models.types import TensorSpec, ensure_latent


CHANNEL_TO_MODALITY: dict[str, str | None] = {
    "none": "noop",
    "text": "text_code",
    "audio": "audio",
    "visual": "visual_video",
    "internal": None,
}


def _metadata_flag(metadata: Mapping[str, Any], name: str) -> bool:
    value = metadata.get(name, False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


class BaseOutputHead(ShapeModule):
    """Shape-only proposal head.

    Forward input shape:
        ``z_world: FloatTensor[B,T,D]``.

    Forward output contract:
        ``{"proposal": dict, "metadata": dict}``, optionally with an output
        ``TensorSpec`` under a modality-specific key.

    Proposal contract:
        Each proposal includes the head name, output channel, optional data
        contract modality ID, source tag, score, latent shape, and a status.  A
        proposal starts as ``"proposal"`` unless it is explicitly diagnostic, in
        which case it is ``"diagnostic-only"`` and must not be externally
        committed by the default policy.
    """

    head_name = "base"
    channel = "internal"
    diagnostic_only = True
    modality: str | None = None
    default_score = 0.0
    default_source_tag = "unknown"
    default_intention = "diagnostic_probe"

    def _proposal(self, z_world: Any, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        b, t, d = ensure_latent(z_world, "z_world")
        proposal_metadata = dict(metadata or {})
        mode = str(proposal_metadata.get("mode", "commit"))
        diagnostic_only = bool(
            self.diagnostic_only
            or mode == "explore"
            or _metadata_flag(proposal_metadata, "diagnostic_only")
            or _metadata_flag(proposal_metadata, "internal_only")
        )
        source_tag = str(proposal_metadata.get("source_tag", self.default_source_tag))
        if source_tag not in SOURCE_TAGS:
            known = ", ".join(SOURCE_TAGS)
            raise ValueError(f"Unknown source_tag {source_tag!r}; expected one of: {known}")

        modality = proposal_metadata.get("modality", self.modality)
        if modality is None:
            modality = CHANNEL_TO_MODALITY.get(self.channel)
        if modality is not None and modality not in MODALITY_IDS:
            known = ", ".join(sorted(MODALITY_IDS))
            raise ValueError(f"Unknown proposal modality {modality!r}; expected one of: {known}")
        modality_id = None if modality is None else MODALITY_IDS[str(modality)]

        score = float(proposal_metadata.get("score", self.default_score))
        status = "diagnostic-only" if diagnostic_only else "proposal"
        return {
            "proposal_id": f"{self.head_name}:{self.channel}",
            "head": self.head_name,
            "channel": self.channel,
            "modality": modality,
            "modality_id": modality_id,
            "latent_shape": [b, t, d],
            "score": score,
            "source_tag": source_tag,
            "diagnostic_only": diagnostic_only,
            "head_diagnostic_only": self.diagnostic_only,
            "internal_only": diagnostic_only,
            "committed": False,
            "status": status,
            "intention": str(proposal_metadata.get("intention", self.default_intention)),
            "metadata": proposal_metadata,
        }

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Create a proposal from ``z_world`` without decoding real outputs."""

        return {
            "proposal": self._proposal(z_world, metadata),
            "metadata": {
                "head": self.head_name,
                "query_provided": query is not None,
                "implementation": "i3_proposal_contract_stub",
            },
        }


def output_spec_from_latent(z_world: Any, output_dim: int, name: str) -> TensorSpec:
    """Create ``FloatTensor[B,T,output_dim]`` metadata from latent shape."""

    b, t, _ = ensure_latent(z_world, "z_world")
    return TensorSpec((b, t, int(output_dim)), "float32", name)
