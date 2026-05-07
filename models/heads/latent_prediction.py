"""Latent prediction head for Sprint I0/I2."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from models.baselines.numpy_nn import Linear, Parameter
from models.heads.base import BaseOutputHead
from models.types import TensorSpec, ensure_latent


class LatentPredictionHead(BaseOutputHead):
    """Predict future/target latent fields from processed world latents.

    Input shape: ``z_world: FloatTensor[B,T,D]``.
    Output shape: ``latent_prediction: FloatTensor[B,T,D]``.

    TensorSpec inputs use the I0 shape-only path. Real NumPy inputs use a
    trainable linear projection and support MSE backward for tiny I2 smoke tests.
    """

    head_name = "latent_prediction"
    channel = "internal"
    default_score = 0.0
    default_source_tag = "predicted"
    default_intention = "predict_future_latent"

    def __init__(self, latent_dim: int | None = None, *, seed: int = 0) -> None:
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self.seed = int(seed)
        self.projection: Linear | None = None
        if self.latent_dim is not None:
            self._ensure_projection(self.latent_dim)

    def _ensure_projection(self, latent_dim: int) -> None:
        if self.projection is None:
            self.latent_dim = int(latent_dim)
            rng = np.random.default_rng(self.seed)
            self.projection = Linear(rng, self.latent_dim, self.latent_dim, name="heads.latent_prediction.projection")
        elif int(latent_dim) != self.latent_dim:
            raise ValueError(f"LatentPredictionHead expected D={self.latent_dim}; got D={latent_dim}")

    def parameters(self) -> list[Parameter]:
        """Return trainable head parameters."""

        return [] if self.projection is None else self.projection.parameters()

    def parameter_count(self) -> int:
        """Exact instantiated trainable parameter count for this head."""

        return int(sum(param.size for param in self.parameters()))

    def forward(self, z_world: Any, query: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        b, t, d = ensure_latent(z_world, "z_world")
        output = super().forward(z_world, query=query, metadata=metadata)
        if isinstance(z_world, TensorSpec):
            output["latent_prediction"] = TensorSpec((b, t, d), "float32", "latent_prediction")
        else:
            self._ensure_projection(d)
            assert self.projection is not None
            output["latent_prediction"] = self.projection.forward(np.asarray(z_world, dtype=np.float64))
            output["metadata"]["implementation"] = "i2_trainable_numpy_latent_prediction_head"
        return output

    def backward(self, grad_prediction: np.ndarray) -> np.ndarray:
        """Backpropagate from ``grad_prediction: FloatTensor[B,T,D]``."""

        if self.projection is None:
            raise RuntimeError("LatentPredictionHead.backward called before a real forward pass")
        return self.projection.backward(np.asarray(grad_prediction, dtype=np.float64))

    @staticmethod
    def mse_loss(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """Return MSE loss and gradient wrt prediction.

        Shape contract:
            prediction/target: ``FloatTensor[B,T,D]``.
            optional mask: ``BoolTensor[B,T]`` where true positions contribute.
        """

        pred = np.asarray(prediction, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        if pred.shape != tgt.shape or pred.ndim != 3:
            raise ValueError(f"prediction and target must share [B,T,D] shape, got {pred.shape} and {tgt.shape}")
        diff = pred - tgt
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != pred.shape[:2]:
                raise ValueError(f"mask must have shape {pred.shape[:2]}, got {mask_array.shape}")
            diff = diff * mask_array[:, :, None]
            denom = int(np.sum(mask_array)) * pred.shape[-1]
        else:
            denom = int(pred.size)
        if denom <= 0:
            raise ValueError("MSE loss requires at least one valid element")
        loss = float(np.sum(np.power(diff, 2)) / float(denom))
        grad = 2.0 * diff / float(denom)
        return loss, grad

    def mse_loss_and_backward(self, z_world: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """Forward, compute MSE, backpropagate, and return loss/prediction."""

        output = self.forward(z_world)
        prediction = output["latent_prediction"]
        loss, grad_prediction = self.mse_loss(prediction, target, mask=mask)
        self.backward(grad_prediction)
        return loss, prediction


__all__ = ["LatentPredictionHead"]
