"""Text/code decoder heads.

The Sprint I3 proposal API remains shape-only by default.  Sprint T1 can opt in
to a tiny trainable NumPy language-model head by passing ``latent_dim`` at
construction time; this adds logits and a backward path while preserving the
proposal contract used by earlier tests.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from models.baselines.numpy_nn import Linear, Parameter, cross_entropy_loss
from models.heads.base import BaseOutputHead, output_spec_from_latent


class TextDecoderHead(BaseOutputHead):
    """Return text/code logits shape metadata.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shape: ``text_logits: FloatTensor[B,T,V]``.
    """

    head_name = "text_decoder"
    channel = "text"
    modality = "text_code"
    diagnostic_only = False
    default_score = 0.5
    default_source_tag = "inferred"
    default_intention = "write_text_or_code"

    def __init__(self, vocab_size: int = 50257, *, latent_dim: int | None = None, seed: int = 0) -> None:
        self.vocab_size = int(vocab_size)
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self._projection = None
        if self.latent_dim is not None:
            rng = np.random.default_rng(int(seed))
            self._projection = Linear(rng, self.latent_dim, self.vocab_size, name="text_decoder.lm_head")

    @property
    def trainable(self) -> bool:
        """Whether this head has a real NumPy logits projection."""

        return self._projection is not None

    def parameters(self) -> list[Parameter]:
        """Return trainable text decoder parameters, if enabled."""

        return [] if self._projection is None else self._projection.parameters()

    def parameter_count(self) -> int:
        """Exact trainable parameter count for the optional LM head."""

        return int(sum(param.size for param in self.parameters()))

    def logits(self, z_world: Any) -> np.ndarray:
        """Project latent states to text/code logits.

        Shape contract:
            ``z_world: FloatTensor[B,T,D]`` ->
            ``text_logits: FloatTensor[B,T,V]``.

        Raises:
            RuntimeError: if the head was not constructed with ``latent_dim``.
        """

        if self._projection is None:
            raise RuntimeError("TextDecoderHead.logits requires latent_dim to enable the trainable T1 LM head")
        return self._projection.forward(np.asarray(z_world, dtype=np.float64))

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        """Backpropagate from ``FloatTensor[B,T,V]`` logits to latent states."""

        if self._projection is None:
            raise RuntimeError("TextDecoderHead.backward requires a trainable T1 LM head")
        return self._projection.backward(np.asarray(grad_logits, dtype=np.float64))

    def loss(self, logits: np.ndarray, target_ids: np.ndarray) -> tuple[float, np.ndarray]:
        """Return next-token cross-entropy and gradient wrt logits."""

        return cross_entropy_loss(logits, target_ids)

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        output = super().forward(z_world, query=query, metadata=metadata)
        output["text_logits"] = output_spec_from_latent(z_world, self.vocab_size, "text_logits")
        output["metadata"]["vocab_size"] = self.vocab_size
        output["metadata"]["trainable"] = self.trainable
        output["proposal"]["output_key"] = "text_logits"
        return output
