"""Text/code decoder head stub."""

from __future__ import annotations

from typing import Any, Mapping

from models.heads.base import BaseOutputHead, output_spec_from_latent


class TextDecoderHead(BaseOutputHead):
    """Return text/code logits shape metadata.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shape: ``text_logits: FloatTensor[B,T,V]``.
    """

    head_name = "text_decoder"
    channel = "text"
    diagnostic_only = False

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = int(vocab_size)

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        output = super().forward(z_world, query=query, metadata=metadata)
        output["text_logits"] = output_spec_from_latent(z_world, self.vocab_size, "text_logits")
        output["metadata"]["vocab_size"] = self.vocab_size
        return output
