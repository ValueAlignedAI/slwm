"""Sprint T0 direct latent-signal SLWM predictor.

This wrapper is intentionally narrower than ``NumpySLWMCore``: it trains the
SLWM processor and latent prediction head directly on synthetic latent signals
``FloatTensor[B,T,D]``.  It does not route synthetic data through text, audio, or
visual adapters, preserving Sprint T0's synthetic-only scope.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.baselines.numpy_nn import AdamW, Parameter
from models.heads import LatentPredictionHead
from models.processor import SignalWorldProcessor
from models.slwm_config import SLWMCoreConfig
from models.slwm_parameter_count import SLWMParameterBreakdown
from training.synthetic_metrics import masked_mse_loss


class NumpySLWMSignalPredictor:
    """Trainable SLWM signal predictor for Sprint T0 synthetic latents.

    Forward input shape:
        ``input_latents: FloatTensor[B,T,D]`` plus optional
        ``input_mask: BoolTensor[B,T]`` where true positions are observed.

    Forward output shape:
        ``prediction: FloatTensor[B,T,D]`` in a dict with processor/head metadata.
    """

    def __init__(self, config: SLWMCoreConfig) -> None:
        self.config = config
        if not config.use_latent_prediction_head:
            raise ValueError("T0 SLWM signal predictor requires use_latent_prediction_head=True")
        self.processor = SignalWorldProcessor(config=config)
        self.latent_prediction_head = LatentPredictionHead(config.latent_dim, seed=config.seed + 4)

    def parameters(self) -> list[Parameter]:
        """Return trainable processor and latent-head parameters."""

        return self.processor.parameters() + self.latent_prediction_head.parameters()

    def make_optimizer(self, *, learning_rate: float = 3e-4, weight_decay: float = 0.0, grad_clip_norm: float | None = 1.0) -> AdamW:
        """Create a deterministic AdamW optimizer for T0 synthetic training."""

        return AdamW(self.parameters(), learning_rate=learning_rate, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm)

    def parameter_count_breakdown(self) -> SLWMParameterBreakdown:
        """Exact trainable parameter counts for T0 direct latent mode.

        T0 uses no modality adapters and no policy/decoder parameters.
        """

        return SLWMParameterBreakdown(
            adapters={},
            processor=self.processor.parameter_count(),
            heads={"latent_prediction": self.latent_prediction_head.parameter_count()},
            policy=0,
        )

    def parameter_count(self) -> int:
        """Return total trainable parameter count."""

        return int(sum(param.size for param in self.parameters()))

    def forward(self, input_latents: np.ndarray, input_mask: np.ndarray | None = None) -> dict[str, Any]:
        """Run processor and latent prediction head on canonical latent inputs."""

        z = np.asarray(input_latents, dtype=np.float64)
        if z.ndim != 3 or z.shape[-1] != self.config.latent_dim:
            raise ValueError(f"input_latents must have shape [B,T,{self.config.latent_dim}], got {z.shape}")
        if z.shape[1] != self.config.context_length:
            raise ValueError(f"input_latents T must match context_length={self.config.context_length}, got {z.shape[1]}")
        mask = None if input_mask is None else np.asarray(input_mask, dtype=bool)
        processed = self.processor(z, mask=mask)
        head_output = self.latent_prediction_head(processed["z_world"])
        return {
            "z_world": processed["z_world"],
            "processed": processed,
            "latent_prediction": head_output,
            "prediction": head_output["latent_prediction"],
            "metadata": {
                "model": "NumpySLWMSignalPredictor",
                "sprint": "T0",
                "synthetic_only": True,
                "ablation_flags": processed["aux"].get("ablation_flags"),
            },
        }

    def loss_and_backward(
        self,
        input_latents: np.ndarray,
        target_latents: np.ndarray,
        *,
        input_mask: np.ndarray | None = None,
        loss_mask: np.ndarray | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Run T0 MSE latent prediction objective and backpropagate.

        Args:
            input_latents: synthetic observed latents ``FloatTensor[B,T,D]``.
            target_latents: synthetic target latents ``FloatTensor[B,T,D]``.
            input_mask: optional observed-position mask passed into the processor.
            loss_mask: optional target-position mask for the MSE objective.
        """

        output = self.forward(input_latents, input_mask=input_mask)
        loss, grad_prediction = masked_mse_loss(output["prediction"], target_latents, mask=loss_mask)
        grad_z_world = self.latent_prediction_head.backward(grad_prediction)
        self.processor.backward(grad_z_world)
        return loss, output


__all__ = ["NumpySLWMSignalPredictor"]
