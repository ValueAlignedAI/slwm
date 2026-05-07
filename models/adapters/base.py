"""Base modality adapter contracts and shared Sprint I2 feature adapter.

All adapters preserve the canonical packet contract:
``{"z": FloatTensor[B,T,D], "mask": BoolTensor[B,T], "metadata": dict}``.
The I0 shape-only path remains available for ``TensorSpec`` tests, while Sprint
I2 subclasses may return real NumPy arrays and expose trainable parameters.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from data.contract import MODALITY_IDS
from models.baselines.numpy_nn import Linear, Parameter
from models.module import ShapeModule
from models.types import TensorSpec
from models.types import ensure_latent, ensure_mask, make_latent_spec, make_mask_spec


class BaseModalityAdapter(ShapeModule):
    """Shape-only base modality adapter.

    Forward input:
        ``sample`` dictionary with optional ``batch_size`` and optional existing
        ``z``/``mask`` shape carriers.

    Forward output shape contract:
        ``{"z": FloatTensor[B,T,D], "mask": BoolTensor[B,T], "metadata": dict}``.

    The base class intentionally remains parameter-free. Concrete I2 adapters
    add NumPy trainable embeddings/projections while preserving this fallback for
    dependency-free shape-contract tests.
    """

    modality: str = "unknown"

    def __init__(self, latent_length: int = 1024, latent_dim: int = 768, modality: str | None = None) -> None:
        self.latent_length = int(latent_length)
        self.latent_dim = int(latent_dim)
        if modality is not None:
            self.modality = modality
        if self.modality not in MODALITY_IDS:
            raise ValueError(f"Unknown modality {self.modality!r}")

    @property
    def modality_id(self) -> int:
        """Stable numeric modality ID for metadata."""

        return MODALITY_IDS[self.modality]

    def forward(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Return a canonical latent packet for this modality."""

        batch_size = int(sample.get("batch_size", 1))
        if "z" in sample:
            z = sample["z"]
            b, t, d = ensure_latent(z)
            if d != self.latent_dim:
                raise ValueError(f"Adapter expected D={self.latent_dim}; got D={d}")
            batch_size = b
        else:
            t = self.latent_length
            z = make_latent_spec(batch_size, t, self.latent_dim, name=f"z_{self.modality}")

        if "mask" in sample:
            mask = sample["mask"]
            ensure_mask(mask, (batch_size, t))
        else:
            mask = make_mask_spec(batch_size, t, name=f"mask_{self.modality}")

        metadata = dict(sample.get("metadata", {}))
        metadata.update(
            {
                "modality": self.modality,
                "modality_id": self.modality_id,
                "observed": True,
                "adapter": self.__class__.__name__,
                "implementation": "i0_shape_contract_stub",
            }
        )
        return {"z": z, "mask": mask, "metadata": metadata}


class ProjectedFeatureAdapter(BaseModalityAdapter):
    """Trainable projection adapter for precomputed modality features.

    Forward input shape:
        ``features``/``latents``/``data``: ``FloatTensor[B,T_in,F]`` where
        ``F=input_dim``. Optional ``mask`` has shape ``BoolTensor[B,T_in]``.

    Forward output shape:
        ``z: FloatTensor[B,T,D]`` and ``mask: BoolTensor[B,T]`` with
        ``T=latent_length`` and ``D=latent_dim``.

    Backward input shape:
        gradient wrt output ``z``: ``FloatTensor[B,T,D]``. Gradients are
        accumulated into the projection and positional parameters.
    """

    feature_keys: tuple[str, ...] = ("features", "latents", "data")

    def __init__(
        self,
        latent_length: int = 1024,
        latent_dim: int = 768,
        *,
        input_dim: int = 80,
        modality: str | None = None,
        seed: int = 0,
        codec_name: str = "provided_latents",
    ) -> None:
        super().__init__(latent_length=latent_length, latent_dim=latent_dim, modality=modality)
        self.input_dim = int(input_dim)
        self.codec_name = str(codec_name)
        rng = np.random.default_rng(int(seed))
        self.projection = Linear(rng, self.input_dim, self.latent_dim, name=f"{self.modality}.projection")
        self.position_embedding = Parameter(
            rng.normal(0.0, 0.02, size=(self.latent_length, self.latent_dim)).astype(np.float64),
            f"{self.modality}.position_embedding",
        )
        self._last_length: int | None = None
        self._last_mask: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        """Return trainable projection and positional parameters."""

        return self.projection.parameters() + [self.position_embedding]

    def parameter_count(self) -> int:
        """Exact instantiated trainable parameter count for this adapter."""

        return int(sum(param.size for param in self.parameters()))

    def _feature_array(self, sample: Mapping[str, Any]) -> Any | None:
        for key in self.feature_keys:
            if key in sample:
                return sample[key]
        return None

    def _normalized_mask(self, sample: Mapping[str, Any], batch_size: int, source_length: int, copied_length: int) -> np.ndarray:
        if "mask" in sample:
            source_mask = np.asarray(sample["mask"], dtype=bool)
            if source_mask.shape != (batch_size, source_length):
                raise ValueError(f"{self.modality} mask must have shape {(batch_size, source_length)}, got {source_mask.shape}")
            copied_mask = source_mask[:, :copied_length]
        else:
            copied_mask = np.ones((batch_size, copied_length), dtype=bool)

        mask = np.zeros((batch_size, self.latent_length), dtype=bool)
        if copied_length:
            mask[:, :copied_length] = copied_mask
        return mask

    def forward(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Project modality features into the shared latent field.

        If no real feature tensor is supplied, falls back to the I0 shape-only
        behavior in ``BaseModalityAdapter``.
        """

        features_value = self._feature_array(sample)
        if features_value is None or isinstance(features_value, TensorSpec) or "z" in sample:
            return super().forward(sample)

        features = np.asarray(features_value, dtype=np.float64)
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"{self.modality} features must have shape [B,T,{self.input_dim}], got {features.shape}")

        batch_size, source_length, _ = features.shape
        copied_length = min(source_length, self.latent_length)
        projected = self.projection.forward(features[:, :copied_length, :]) if copied_length else np.zeros((batch_size, 0, self.latent_dim))
        mask = self._normalized_mask(sample, batch_size, source_length, copied_length)

        z = np.zeros((batch_size, self.latent_length, self.latent_dim), dtype=np.float64)
        if copied_length:
            z[:, :copied_length, :] = projected + self.position_embedding.value[:copied_length][None, :, :]
            z[:, :copied_length, :] *= mask[:, :copied_length, None]

        self._last_length = copied_length
        self._last_mask = mask[:, :copied_length]

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
                "implementation": "i2_trainable_numpy_feature_adapter",
            }
        )
        return {"z": z, "mask": mask, "metadata": metadata}

    def backward(self, grad_z: np.ndarray) -> None:
        """Backpropagate from ``grad_z: FloatTensor[B,T,D]`` into parameters."""

        if self._last_length is None or self._last_mask is None:
            raise RuntimeError(f"{self.__class__.__name__}.backward called before a real forward pass")
        grad = np.asarray(grad_z, dtype=np.float64)
        if grad.ndim != 3 or grad.shape[1:] != (self.latent_length, self.latent_dim):
            raise ValueError(f"grad_z must have shape [B,{self.latent_length},{self.latent_dim}], got {grad.shape}")
        copied_length = self._last_length
        if copied_length == 0:
            return
        grad_copied = grad[:, :copied_length, :] * self._last_mask[:, :, None]
        self.position_embedding.grad[:copied_length] += np.sum(grad_copied, axis=0)
        self.projection.backward(grad_copied)


__all__ = ["BaseModalityAdapter", "ProjectedFeatureAdapter"]
