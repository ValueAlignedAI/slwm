"""Vanilla continuous-signal Transformer baseline for Sprint T0.

This baseline intentionally excludes SLWM novelty components: no spectral mixer,
no long-convolution/SSM block, no policy gate, and no uncertainty head.  It maps
synthetic latent signals directly from ``FloatTensor[B,T,D]`` to
``FloatTensor[B,T,D]`` with standard Transformer blocks and an MSE objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from models.baselines.numpy_nn import AdamW, LayerNorm, Linear, Parameter, TransformerBlock
from training.synthetic_metrics import masked_mse_loss


@dataclass(frozen=True)
class ContinuousSignalTransformerConfig:
    """Config for the T0 vanilla continuous Transformer baseline.

    Shape contract:
        input and output latents use ``FloatTensor[B,T,D]`` where
        ``T=context_length`` and ``D=latent_dim``. Optional masks use
        ``BoolTensor[B,T]``.
    """

    context_length: int = 64
    latent_dim: int = 16
    n_layer: int = 2
    n_head: int = 2
    intermediate_size: int | None = None
    seed: int = 0
    causal: bool = False
    parameter_accounting_mode: str = "strict"

    @property
    def d_ff(self) -> int:
        return int(self.intermediate_size if self.intermediate_size is not None else 4 * self.latent_dim)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ContinuousSignalTransformerConfig":
        """Construct from either a top-level config or a model section."""

        model = mapping.get("model", mapping) if isinstance(mapping.get("model", mapping), Mapping) else mapping
        runtime = mapping.get("runtime", {}) if isinstance(mapping.get("runtime", {}), Mapping) else {}
        return cls(
            context_length=int(model.get("context_length", model.get("latent_length", cls.context_length))),
            latent_dim=int(model.get("latent_dim", model.get("n_embd", cls.latent_dim))),
            n_layer=int(model.get("n_layer", model.get("processor_layers", cls.n_layer))),
            n_head=int(model.get("n_head", model.get("attention_heads", cls.n_head))),
            intermediate_size=(None if model.get("intermediate_size") is None else int(model.get("intermediate_size"))),
            seed=int(model.get("seed", runtime.get("seed", cls.seed))),
            causal=bool(model.get("causal", cls.causal)),
            parameter_accounting_mode=str(model.get("parameter_accounting_mode", cls.parameter_accounting_mode)),
        )


class NumpyContinuousSignalTransformerBaseline:
    """Vanilla Transformer regression baseline for synthetic signal latents.

    Forward input shape:
        ``input_latents: FloatTensor[B,T,D]`` and optional
        ``input_mask: BoolTensor[B,T]``.

    Forward output shape:
        ``prediction: FloatTensor[B,T,D]``.
    """

    def __init__(self, config: ContinuousSignalTransformerConfig) -> None:
        self.config = config
        if config.latent_dim % config.n_head != 0:
            raise ValueError("latent_dim must be divisible by n_head")
        rng = np.random.default_rng(int(config.seed) + 3000)
        self.input_projection = Linear(rng, config.latent_dim, config.latent_dim, name="t0_vanilla.input_projection")
        self.position_embedding = Parameter(
            rng.normal(0.0, 0.02, size=(config.context_length, config.latent_dim)).astype(np.float64),
            "t0_vanilla.position_embedding",
        )
        self.blocks = [
            TransformerBlock(
                rng,
                config.latent_dim,
                config.n_head,
                config.d_ff,
                name=f"t0_vanilla.blocks.{layer}",
                causal=config.causal,
            )
            for layer in range(config.n_layer)
        ]
        self.final_layer_norm = LayerNorm(config.latent_dim, name="t0_vanilla.final_layer_norm")
        self.output_projection = Linear(rng, config.latent_dim, config.latent_dim, name="t0_vanilla.output_projection")
        self._last_length: int | None = None
        self._last_input_mask: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        """Return trainable parameters in deterministic order."""

        params: list[Parameter] = []
        params.extend(self.input_projection.parameters())
        params.append(self.position_embedding)
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_layer_norm.parameters())
        params.extend(self.output_projection.parameters())
        return params

    def parameter_count(self) -> int:
        """Return exact trainable parameter count."""

        return int(sum(param.size for param in self.parameters()))

    def module_parameter_counts(self) -> dict[str, int]:
        """Map parameter counts onto shared registry buckets."""

        embeddings = int(self.position_embedding.size)
        input_adapter = int(sum(param.size for param in self.input_projection.parameters()))
        processor = int(sum(param.size for block in self.blocks for param in block.parameters()))
        processor += int(sum(param.size for param in self.final_layer_norm.parameters()))
        heads = int(sum(param.size for param in self.output_projection.parameters()))
        total = embeddings + input_adapter + processor + heads
        return {
            "adapters": input_adapter,
            "processor": processor,
            "heads": heads,
            "policy": 0,
            "decoders": 0,
            "embeddings": embeddings,
            "total": total,
        }

    def make_optimizer(self, *, learning_rate: float = 3e-4, weight_decay: float = 0.0, grad_clip_norm: float | None = 1.0) -> AdamW:
        """Create deterministic AdamW optimizer for the baseline."""

        return AdamW(self.parameters(), learning_rate=learning_rate, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm)

    def forward(self, input_latents: np.ndarray, input_mask: np.ndarray | None = None) -> dict[str, Any]:
        """Predict target latent signals from input latent signals."""

        x = np.asarray(input_latents, dtype=np.float64)
        if x.ndim != 3 or x.shape[-1] != self.config.latent_dim:
            raise ValueError(f"input_latents must have shape [B,T,{self.config.latent_dim}], got {x.shape}")
        if x.shape[1] != self.config.context_length:
            raise ValueError(f"input T must match context_length={self.config.context_length}, got {x.shape[1]}")
        mask = None if input_mask is None else np.asarray(input_mask, dtype=bool)
        if mask is not None and mask.shape != x.shape[:2]:
            raise ValueError(f"input_mask must have shape {x.shape[:2]}, got {mask.shape}")
        self._last_length = x.shape[1]
        self._last_input_mask = mask

        hidden = self.input_projection.forward(x) + self.position_embedding.value[: x.shape[1]][None, :, :]
        if mask is not None:
            hidden = hidden * mask[:, :, None]
        for block in self.blocks:
            hidden = block.forward(hidden, attention_mask=mask)
        hidden = self.final_layer_norm.forward(hidden)
        prediction = self.output_projection.forward(hidden)
        return {
            "prediction": prediction,
            "metadata": {
                "model": "NumpyContinuousSignalTransformerBaseline",
                "sprint": "T0",
                "synthetic_only": True,
                "architecture_flags": {
                    "vanilla_transformer": True,
                    "spectral_mixer": False,
                    "longconv_or_ssm": False,
                    "policy_commit_gate": False,
                },
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
        """Run MSE signal prediction objective and backpropagate."""

        output = self.forward(input_latents, input_mask=input_mask)
        loss, grad_prediction = masked_mse_loss(output["prediction"], target_latents, mask=loss_mask)
        grad_hidden = self.output_projection.backward(grad_prediction)
        grad_hidden = self.final_layer_norm.backward(grad_hidden)
        for block in reversed(self.blocks):
            grad_hidden = block.backward(grad_hidden)
        if self._last_length is None:
            raise RuntimeError("loss_and_backward called before forward cached length")
        if self._last_input_mask is not None:
            grad_hidden = grad_hidden * self._last_input_mask[:, :, None]
        self.position_embedding.grad[: self._last_length] += np.sum(grad_hidden, axis=0)
        self.input_projection.backward(grad_hidden)
        return loss, output


__all__ = ["ContinuousSignalTransformerConfig", "NumpyContinuousSignalTransformerBaseline"]
