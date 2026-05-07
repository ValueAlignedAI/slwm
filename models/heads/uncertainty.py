"""Uncertainty/source head for Sprint I0/I2."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from data.contract import SOURCE_TAGS
from models.baselines.numpy_nn import Linear, Parameter
from models.heads.base import BaseOutputHead
from models.types import TensorSpec, ensure_latent


class UncertaintyHead(BaseOutputHead):
    """Predict uncertainty and source-tag logits from world latents.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shapes:
        ``uncertainty: FloatTensor[B,T,1]``
        ``source_logits: FloatTensor[B,T,S]`` where ``S=len(SOURCE_TAGS)``.

    I2 provides an interface-ready trainable head; calibration quality is not
    claimed until later evaluation sprints.
    """

    head_name = "uncertainty"
    channel = "internal"
    default_source_tag = "inferred"
    default_intention = "estimate_uncertainty_and_source"

    def __init__(self, latent_dim: int | None = None, *, seed: int = 0) -> None:
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self.seed = int(seed)
        self.uncertainty_proj: Linear | None = None
        self.source_proj: Linear | None = None
        self._last_uncertainty: np.ndarray | None = None
        if self.latent_dim is not None:
            self._ensure_projections(self.latent_dim)

    def _ensure_projections(self, latent_dim: int) -> None:
        if self.uncertainty_proj is None or self.source_proj is None:
            self.latent_dim = int(latent_dim)
            rng = np.random.default_rng(self.seed)
            self.uncertainty_proj = Linear(rng, self.latent_dim, 1, name="heads.uncertainty.scalar")
            self.source_proj = Linear(rng, self.latent_dim, len(SOURCE_TAGS), name="heads.uncertainty.source")
        elif int(latent_dim) != self.latent_dim:
            raise ValueError(f"UncertaintyHead expected D={self.latent_dim}; got D={latent_dim}")

    def parameters(self) -> list[Parameter]:
        """Return trainable uncertainty/source parameters."""

        if self.uncertainty_proj is None or self.source_proj is None:
            return []
        return self.uncertainty_proj.parameters() + self.source_proj.parameters()

    def parameter_count(self) -> int:
        """Exact instantiated trainable parameter count for this head."""

        return int(sum(param.size for param in self.parameters()))

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        b, t, _ = ensure_latent(z_world, "z_world")
        output = super().forward(z_world, query=query, metadata=metadata)
        if isinstance(z_world, TensorSpec):
            output["uncertainty"] = TensorSpec((b, t, 1), "float32", "uncertainty")
            output["source_logits"] = TensorSpec((b, t, len(SOURCE_TAGS)), "float32", "source_logits")
        else:
            self._ensure_projections(z_world.shape[-1])
            assert self.uncertainty_proj is not None and self.source_proj is not None
            logits = self.uncertainty_proj.forward(np.asarray(z_world, dtype=np.float64))
            uncertainty = 1.0 / (1.0 + np.exp(-logits))
            output["uncertainty"] = uncertainty
            output["source_logits"] = self.source_proj.forward(np.asarray(z_world, dtype=np.float64))
            output["metadata"]["implementation"] = "i2_trainable_numpy_uncertainty_head"
            self._last_uncertainty = uncertainty
        output["metadata"]["source_tags"] = list(SOURCE_TAGS)
        return output

    def backward(self, grad_uncertainty: np.ndarray | None = None, grad_source_logits: np.ndarray | None = None) -> np.ndarray:
        """Backpropagate artificial/tiny-run gradients through the head.

        Args:
            grad_uncertainty: Optional gradient wrt sigmoid uncertainty output
                with shape ``FloatTensor[B,T,1]``.
            grad_source_logits: Optional gradient wrt source logits with shape
                ``FloatTensor[B,T,S]``.

        Returns:
            Accumulated gradient wrt ``z_world: FloatTensor[B,T,D]``.
        """

        if self.uncertainty_proj is None or self.source_proj is None:
            raise RuntimeError("UncertaintyHead.backward called before a real forward pass")
        grad_z: np.ndarray | None = None
        if grad_source_logits is not None:
            grad_z = self.source_proj.backward(np.asarray(grad_source_logits, dtype=np.float64))
        if grad_uncertainty is not None:
            if self._last_uncertainty is None:
                raise RuntimeError("uncertainty sigmoid cache missing")
            grad_logits = np.asarray(grad_uncertainty, dtype=np.float64) * self._last_uncertainty * (1.0 - self._last_uncertainty)
            grad_scalar = self.uncertainty_proj.backward(grad_logits)
            grad_z = grad_scalar if grad_z is None else grad_z + grad_scalar
        if grad_z is None:
            raise ValueError("At least one uncertainty/source gradient is required")
        return grad_z


__all__ = ["UncertaintyHead"]
