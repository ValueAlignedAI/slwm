"""Sprint I2 NumPy SLWM core wrapper.

This wrapper is intentionally limited to I2 scope: adapters, shared latent field,
signal processor, latent prediction head, and uncertainty head. It does not
implement policy routing, output commitment, exploration dashboards, datasets, or
large training loops.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from models.adapters import AudioSignalAdapter, TextSignalAdapter, VisualSignalAdapter
from models.baselines.numpy_nn import AdamW, Parameter
from models.heads import LatentPredictionHead, UncertaintyHead
from models.latent_field import LatentSignalField
from models.processor import SignalWorldProcessor
from models.slwm_config import SLWMCoreConfig
from models.slwm_parameter_count import SLWMParameterBreakdown


class NumpySLWMCore:
    """Minimal Sprint I2 SLWM core for deterministic smoke tests.

    Forward input batch keys:
        ``text_tokens``: ``IntTensor[B,T_text]``.
        ``audio_features``: ``FloatTensor[B,T_audio,A]``.
        ``visual_features``: ``FloatTensor[B,T_visual,V]``.

    Forward outputs:
        ``field['z']`` and ``processed['z_world']`` preserve the canonical
        ``FloatTensor[B,T,D]`` contract. Optional I2 heads emit
        ``latent_prediction: FloatTensor[B,T,D]`` and uncertainty/source logits.

    Training smoke contract:
        ``loss_and_backward`` applies an MSE latent prediction objective against a
        dummy ``target_latents: FloatTensor[B,T,D]`` and accumulates gradients in
        adapters, processor, and latent prediction head.
    """

    def __init__(self, config: SLWMCoreConfig) -> None:
        self.config = config
        self.text_adapter = TextSignalAdapter(
            latent_length=config.context_length,
            latent_dim=config.latent_dim,
            vocab_size=config.text_vocab_size,
            seed=config.seed + 1,
        )
        self.audio_adapter = AudioSignalAdapter(
            latent_length=config.context_length,
            latent_dim=config.latent_dim,
            audio_feature_dim=config.audio_feature_dim,
            seed=config.seed + 2,
        )
        self.visual_adapter = VisualSignalAdapter(
            latent_length=config.context_length,
            latent_dim=config.latent_dim,
            visual_feature_dim=config.visual_feature_dim,
            seed=config.seed + 3,
        )
        self.latent_field = LatentSignalField(latent_length=config.context_length, latent_dim=config.latent_dim)
        self.processor = SignalWorldProcessor(config=config)
        self.latent_prediction_head = (
            LatentPredictionHead(config.latent_dim, seed=config.seed + 4) if config.use_latent_prediction_head else None
        )
        self.uncertainty_head = UncertaintyHead(config.latent_dim, seed=config.seed + 5) if config.use_uncertainty_head else None
        self._last_adapter_order: list[Any] = []
        self._last_output: dict[str, Any] | None = None

    def parameters(self) -> list[Parameter]:
        """Return all trainable I2 parameters in deterministic order."""

        params: list[Parameter] = []
        for adapter in (self.text_adapter, self.audio_adapter, self.visual_adapter):
            params.extend(adapter.parameters())
        params.extend(self.processor.parameters())
        if self.latent_prediction_head is not None:
            params.extend(self.latent_prediction_head.parameters())
        if self.uncertainty_head is not None:
            params.extend(self.uncertainty_head.parameters())
        return params

    def make_optimizer(self, *, learning_rate: float = 3e-4, weight_decay: float = 0.0, grad_clip_norm: float | None = 1.0) -> AdamW:
        """Create a deterministic AdamW optimizer over core I2 parameters."""

        return AdamW(self.parameters(), learning_rate=learning_rate, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm)

    def parameter_count_breakdown(self) -> SLWMParameterBreakdown:
        """Exact trainable parameter counts by adapter, processor, and head."""

        head_counts: dict[str, int] = {}
        if self.latent_prediction_head is not None:
            head_counts["latent_prediction"] = self.latent_prediction_head.parameter_count()
        if self.uncertainty_head is not None:
            head_counts["uncertainty"] = self.uncertainty_head.parameter_count()
        return SLWMParameterBreakdown(
            adapters={
                "text_code": self.text_adapter.parameter_count(),
                "audio": self.audio_adapter.parameter_count(),
                "visual_video": self.visual_adapter.parameter_count(),
            },
            processor=self.processor.parameter_count(),
            heads=head_counts,
            policy=0,
        )

    def _pack_batch(self, batch: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
        packets: list[dict[str, Any]] = []
        adapters: list[Any] = []
        if "text_tokens" in batch:
            sample = {"input_ids": batch["text_tokens"]}
            if "text_mask" in batch:
                sample["mask"] = batch["text_mask"]
            packets.append(self.text_adapter(sample))
            adapters.append(self.text_adapter)
        if "audio_features" in batch:
            sample = {"features": batch["audio_features"]}
            if "audio_mask" in batch:
                sample["mask"] = batch["audio_mask"]
            packets.append(self.audio_adapter(sample))
            adapters.append(self.audio_adapter)
        if "visual_features" in batch:
            sample = {"features": batch["visual_features"]}
            if "visual_mask" in batch:
                sample["mask"] = batch["visual_mask"]
            packets.append(self.visual_adapter(sample))
            adapters.append(self.visual_adapter)
        if not packets:
            raise ValueError("NumpySLWMCore.forward requires at least one modality in the batch")
        return packets, adapters

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Run adapters -> latent field -> processor -> I2 diagnostic heads."""

        packets, adapter_order = self._pack_batch(batch)
        field = self.latent_field.from_adapter_outputs(packets)
        processed = self.processor(field["z"], mask=field["mask"])
        z_world = processed["z_world"]
        output: dict[str, Any] = {"packets": packets, "field": field, "processed": processed, "z_world": z_world}
        if self.latent_prediction_head is not None:
            output["latent_prediction"] = self.latent_prediction_head(z_world)
        if self.uncertainty_head is not None:
            output["uncertainty"] = self.uncertainty_head(z_world)
        self._last_adapter_order = adapter_order
        self._last_output = output
        return output

    def loss_and_backward(self, batch: Mapping[str, Any], target_latents: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Run a tiny latent-prediction MSE objective and backpropagate.

        Args:
            batch: Multimodal dummy batch accepted by ``forward``.
            target_latents: ``FloatTensor[B,T,D]`` target for the latent
                prediction head.

        Returns:
            ``(loss, output)`` where loss is finite MSE over valid packed mask
            positions.
        """

        if self.latent_prediction_head is None:
            raise RuntimeError("loss_and_backward requires use_latent_prediction_head=True")
        output = self.forward(batch)
        prediction = output["latent_prediction"]["latent_prediction"]
        loss, grad_prediction = self.latent_prediction_head.mse_loss(prediction, target_latents, mask=output["field"]["mask"])
        grad_z_world = self.latent_prediction_head.backward(grad_prediction)
        grad_context = self.processor.backward(grad_z_world)
        packet_grads = self.latent_field.backward(grad_context)
        for adapter, packet_grad in zip(self._last_adapter_order, packet_grads, strict=True):
            adapter.backward(packet_grad)
        return loss, output


def make_i2_dummy_batch(config: SLWMCoreConfig, *, batch_size: int = 2, seed: int | None = None) -> dict[str, np.ndarray]:
    """Create deterministic text/audio/visual dummy inputs for I2 tests.

    Output shapes:
        ``text_tokens: IntTensor[B,T_text]``;
        ``audio_features: FloatTensor[B,T_audio,A]``;
        ``visual_features: FloatTensor[B,T_visual,V]``.
    """

    rng = np.random.default_rng(config.seed if seed is None else int(seed))
    # Keep per-modality lengths small so all required modalities fit inside the
    # fixed context during tiny tests.
    text_length = max(1, min(4, config.context_length // 3))
    audio_length = max(1, min(3, config.context_length // 3))
    visual_length = max(1, min(3, config.context_length - text_length - audio_length))
    return {
        "text_tokens": rng.integers(0, config.text_vocab_size, size=(int(batch_size), text_length), dtype=np.int64),
        "audio_features": rng.normal(0.0, 1.0, size=(int(batch_size), audio_length, config.audio_feature_dim)).astype(np.float64),
        "visual_features": rng.normal(0.0, 1.0, size=(int(batch_size), visual_length, config.visual_feature_dim)).astype(np.float64),
    }


__all__ = ["NumpySLWMCore", "make_i2_dummy_batch"]
