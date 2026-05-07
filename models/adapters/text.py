"""Text/code signal adapter for Sprint I2.

The adapter treats tokens/bytes as an edge codec only. It maps integer text/code
IDs into the canonical shared latent packet ``z: FloatTensor[B,T,D]`` while the
shared processor operates on continuous latent fields.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from models.adapters.base import BaseModalityAdapter
from models.baselines.numpy_nn import Embedding, Parameter
from models.types import TensorSpec


class TextSignalAdapter(BaseModalityAdapter):
    """Map text/code edge samples to ``z: FloatTensor[B,T,D]``.

    Shape contract:
        ``input_ids``/``tokens``/``data`` as ``IntTensor[B,T_in]`` ->
        ``z: FloatTensor[B,T,D]`` and ``mask: BoolTensor[B,T]``.

    The tokenizer/BPE choice remains external to the core; this adapter only
    embeds supplied IDs and adds a learned position signal. If no IDs are
    supplied, it falls back to the I0 ``TensorSpec`` shape-only behavior.
    """

    modality = "text_code"

    def __init__(
        self,
        latent_length: int = 1024,
        latent_dim: int = 768,
        *,
        vocab_size: int = 50_257,
        seed: int = 0,
        codec_name: str = "integer_token_edge_codec",
    ) -> None:
        super().__init__(latent_length=latent_length, latent_dim=latent_dim)
        self.vocab_size = int(vocab_size)
        self.codec_name = str(codec_name)
        rng = np.random.default_rng(int(seed))
        self.token_embedding = Embedding(rng, self.vocab_size, self.latent_dim, name="text_code.token_embedding")
        self.position_embedding = Parameter(
            rng.normal(0.0, 0.02, size=(self.latent_length, self.latent_dim)).astype(np.float64),
            "text_code.position_embedding",
        )
        self._last_length: int | None = None
        self._last_mask: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        """Return trainable token and position parameters."""

        return self.token_embedding.parameters() + [self.position_embedding]

    def parameter_count(self) -> int:
        """Exact instantiated trainable parameter count for this adapter."""

        return int(sum(param.size for param in self.parameters()))

    def _ids_from_sample(self, sample: Mapping[str, Any]) -> Any | None:
        for key in ("input_ids", "tokens", "data"):
            if key in sample:
                return sample[key]
        return None

    def _coerce_ids(self, value: Any) -> np.ndarray:
        if isinstance(value, str):
            ids = np.asarray([[byte for byte in value.encode("utf-8")]], dtype=np.int64)
        elif isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
            encoded = [np.asarray([byte for byte in item.encode("utf-8")], dtype=np.int64) for item in value]
            max_len = max(1, max(int(row.size) for row in encoded))
            ids = np.zeros((len(encoded), max_len), dtype=np.int64)
            for index, row in enumerate(encoded):
                ids[index, : row.size] = row
        else:
            ids = np.asarray(value, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError(f"text_code IDs must have shape [B,T], got {ids.shape}")
        if np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError(f"text_code IDs must be in [0,{self.vocab_size})")
        return ids

    def forward(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Embed text/code IDs into a canonical latent packet."""

        ids_value = self._ids_from_sample(sample)
        if ids_value is None or isinstance(ids_value, TensorSpec) or "z" in sample:
            return super().forward(sample)

        ids = self._coerce_ids(ids_value)
        batch_size, source_length = ids.shape
        copied_length = min(source_length, self.latent_length)
        ids_copied = ids[:, :copied_length]

        if "mask" in sample:
            source_mask = np.asarray(sample["mask"], dtype=bool)
            if source_mask.shape != (batch_size, source_length):
                raise ValueError(f"text_code mask must have shape {(batch_size, source_length)}, got {source_mask.shape}")
            copied_mask = source_mask[:, :copied_length]
        else:
            copied_mask = np.ones((batch_size, copied_length), dtype=bool)

        z = np.zeros((batch_size, self.latent_length, self.latent_dim), dtype=np.float64)
        mask = np.zeros((batch_size, self.latent_length), dtype=bool)
        if copied_length:
            z[:, :copied_length, :] = self.token_embedding.forward(ids_copied) + self.position_embedding.value[:copied_length][None, :, :]
            mask[:, :copied_length] = copied_mask
            z[:, :copied_length, :] *= copied_mask[:, :, None]

        self._last_length = copied_length
        self._last_mask = copied_mask

        metadata = dict(sample.get("metadata", {}))
        metadata.update(
            {
                "modality": self.modality,
                "modality_id": self.modality_id,
                "observed": True,
                "adapter": self.__class__.__name__,
                "codec": self.codec_name,
                "source_length": int(source_length),
                "copied_length": int(copied_length),
                "implementation": "i2_trainable_numpy_text_adapter",
            }
        )
        return {"z": z, "mask": mask, "metadata": metadata}

    def backward(self, grad_z: np.ndarray) -> None:
        """Backpropagate from ``grad_z: FloatTensor[B,T,D]`` into embeddings."""

        if self._last_length is None or self._last_mask is None:
            raise RuntimeError("TextSignalAdapter.backward called before a real forward pass")
        grad = np.asarray(grad_z, dtype=np.float64)
        if grad.ndim != 3 or grad.shape[1:] != (self.latent_length, self.latent_dim):
            raise ValueError(f"grad_z must have shape [B,{self.latent_length},{self.latent_dim}], got {grad.shape}")
        copied_length = self._last_length
        if copied_length == 0:
            return
        grad_copied = grad[:, :copied_length, :] * self._last_mask[:, :, None]
        self.position_embedding.grad[:copied_length] += np.sum(grad_copied, axis=0)
        self.token_embedding.backward(grad_copied)


__all__ = ["TextSignalAdapter"]
