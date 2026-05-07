"""Vanilla multimodal Transformer baseline for Sprint I1.

This baseline serializes text, audio-feature, and visual-feature positions into a
single sequence and applies ordinary non-causal Transformer blocks. It is a
control model for later SLWM comparisons and intentionally avoids SLWM-specific
shared-latent novelty such as spectral/SSM/long-conv processors or policy gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np

from models.baselines.numpy_nn import AdamW, Embedding, LayerNorm, Linear, Parameter, TransformerBlock, cross_entropy_loss
from models.baselines.parameter_count import MultimodalParameterBreakdown, vanilla_multimodal_parameter_breakdown


@dataclass(frozen=True)
class VanillaMultimodalConfig:
    """Configuration for a vanilla multimodal Transformer baseline.

    Shape contract:
        text tokens: ``IntTensor[B,T_text]``.
        audio features: ``FloatTensor[B,T_audio,A]``.
        visual features: ``FloatTensor[B,T_visual,V]``.
        concatenated logits: ``FloatTensor[B,T_total,target_vocab_size]``.
    """

    text_vocab_size: int = 32_000
    target_vocab_size: int = 512
    context_length: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    audio_feature_dim: int = 80
    visual_feature_dim: int = 256
    intermediate_size: int | None = None
    seed: int = 0
    text_codec: str = "gpt2_bpe"
    audio_codec_or_features: str = "provided_audio_latents"
    visual_codec_or_features: str = "provided_visual_latents"

    @property
    def d_ff(self) -> int:
        return int(self.intermediate_size if self.intermediate_size is not None else 4 * self.n_embd)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "VanillaMultimodalConfig":
        """Construct from a JSON config ``model`` section."""

        codecs = mapping.get("codecs", {}) if isinstance(mapping.get("codecs", {}), Mapping) else {}
        return cls(
            text_vocab_size=int(mapping.get("text_vocab_size", cls.text_vocab_size)),
            target_vocab_size=int(mapping.get("target_vocab_size", cls.target_vocab_size)),
            context_length=int(mapping.get("context_length", cls.context_length)),
            n_layer=int(mapping.get("n_layer", cls.n_layer)),
            n_embd=int(mapping.get("n_embd", cls.n_embd)),
            n_head=int(mapping.get("n_head", cls.n_head)),
            audio_feature_dim=int(mapping.get("audio_feature_dim", cls.audio_feature_dim)),
            visual_feature_dim=int(mapping.get("visual_feature_dim", cls.visual_feature_dim)),
            intermediate_size=(None if mapping.get("intermediate_size") is None else int(mapping["intermediate_size"])),
            seed=int(mapping.get("seed", mapping.get("runtime_seed", 0))),
            text_codec=str(codecs.get("text", "gpt2_bpe")),
            audio_codec_or_features=str(codecs.get("audio", "provided_audio_latents")),
            visual_codec_or_features=str(codecs.get("visual", "provided_visual_latents")),
        )

    def parameter_breakdown(self) -> MultimodalParameterBreakdown:
        """Exact parameter count for this vanilla multimodal config."""

        return vanilla_multimodal_parameter_breakdown(
            text_vocab_size=self.text_vocab_size,
            target_vocab_size=self.target_vocab_size,
            context_length=self.context_length,
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            audio_feature_dim=self.audio_feature_dim,
            visual_feature_dim=self.visual_feature_dim,
            intermediate_size=self.d_ff,
        )


class NumpyVanillaMultimodalTransformerBaseline:
    """Ordinary Transformer over serialized multimodal features.

    Forward shape contract:
        ``text_tokens``: ``IntTensor[B,T_text]``.
        ``audio_features``: ``FloatTensor[B,T_audio,A]``.
        ``visual_features``: ``FloatTensor[B,T_visual,V]``.
        returned logits: ``FloatTensor[B,T_total,target_vocab_size]``.
    """

    TEXT_MODALITY = 0
    AUDIO_MODALITY = 1
    VISUAL_MODALITY = 2

    def __init__(self, config: VanillaMultimodalConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.text_embedding = Embedding(rng, config.text_vocab_size, config.n_embd, name="text_embedding")
        self.audio_projection = Linear(rng, config.audio_feature_dim, config.n_embd, name="audio_projection")
        self.visual_projection = Linear(rng, config.visual_feature_dim, config.n_embd, name="visual_projection")
        self.position_embedding = Parameter(
            rng.normal(0.0, 0.02, size=(config.context_length, config.n_embd)).astype(np.float64),
            "position_embedding",
        )
        self.modality_embedding = Parameter(
            rng.normal(0.0, 0.02, size=(3, config.n_embd)).astype(np.float64),
            "modality_embedding",
        )
        self.blocks = [
            TransformerBlock(
                rng,
                config.n_embd,
                config.n_head,
                config.d_ff,
                name=f"blocks.{layer}",
                causal=False,
            )
            for layer in range(config.n_layer)
        ]
        self.final_layer_norm = LayerNorm(config.n_embd, name="ln_f")
        self.output_head = Linear(rng, config.n_embd, config.target_vocab_size, name="output_head")
        self._last_lengths: tuple[int, int, int] | None = None
        self._last_modality_ids: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        """Return trainable parameters in deterministic order."""

        params: list[Parameter] = []
        params.extend(self.text_embedding.parameters())
        params.extend(self.audio_projection.parameters())
        params.extend(self.visual_projection.parameters())
        params.append(self.position_embedding)
        params.append(self.modality_embedding)
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_layer_norm.parameters())
        params.extend(self.output_head.parameters())
        return params

    def parameter_count(self) -> int:
        """Exact instantiated trainable parameter count."""

        return int(sum(param.size for param in self.parameters()))

    def formula_parameter_count(self) -> int:
        """Exact formula count for the configured baseline."""

        return self.config.parameter_breakdown().total

    def forward(self, *, text_tokens: np.ndarray, audio_features: np.ndarray, visual_features: np.ndarray) -> np.ndarray:
        """Compute per-position logits over serialized multimodal inputs."""

        text = np.asarray(text_tokens, dtype=np.int64)
        audio = np.asarray(audio_features, dtype=np.float64)
        visual = np.asarray(visual_features, dtype=np.float64)
        if text.ndim != 2:
            raise ValueError(f"text_tokens must have shape [B,T_text], got {text.shape}")
        if audio.ndim != 3 or audio.shape[-1] != self.config.audio_feature_dim:
            raise ValueError(f"audio_features must have shape [B,T_audio,{self.config.audio_feature_dim}], got {audio.shape}")
        if visual.ndim != 3 or visual.shape[-1] != self.config.visual_feature_dim:
            raise ValueError(f"visual_features must have shape [B,T_visual,{self.config.visual_feature_dim}], got {visual.shape}")
        bsz = text.shape[0]
        if audio.shape[0] != bsz or visual.shape[0] != bsz:
            raise ValueError("text, audio, and visual inputs must share batch size")

        text_hidden = self.text_embedding.forward(text)
        audio_hidden = self.audio_projection.forward(audio)
        visual_hidden = self.visual_projection.forward(visual)
        hidden = np.concatenate([text_hidden, audio_hidden, visual_hidden], axis=1)
        total_length = hidden.shape[1]
        if total_length > self.config.context_length:
            raise ValueError(f"serialized length {total_length} exceeds context_length={self.config.context_length}")
        modality_ids = np.concatenate(
            [
                np.full((text.shape[1],), self.TEXT_MODALITY, dtype=np.int64),
                np.full((audio.shape[1],), self.AUDIO_MODALITY, dtype=np.int64),
                np.full((visual.shape[1],), self.VISUAL_MODALITY, dtype=np.int64),
            ]
        )
        hidden = hidden + self.position_embedding.value[np.arange(total_length)][None, :, :]
        hidden = hidden + self.modality_embedding.value[modality_ids][None, :, :]
        for block in self.blocks:
            hidden = block.forward(hidden)
        hidden = self.final_layer_norm.forward(hidden)
        self._last_lengths = (text.shape[1], audio.shape[1], visual.shape[1])
        self._last_modality_ids = modality_ids
        return self.output_head.forward(hidden)

    def backward(self, grad_logits: np.ndarray) -> None:
        """Backpropagate from serialized multimodal logits."""

        if self._last_lengths is None or self._last_modality_ids is None:
            raise RuntimeError("backward called before forward")
        grad_hidden = self.output_head.backward(grad_logits)
        grad_hidden = self.final_layer_norm.backward(grad_hidden)
        for block in reversed(self.blocks):
            grad_hidden = block.backward(grad_hidden)

        total_length = grad_hidden.shape[1]
        self.position_embedding.grad[np.arange(total_length)] += np.sum(grad_hidden, axis=0)
        for position, modality_id in enumerate(self._last_modality_ids):
            self.modality_embedding.grad[modality_id] += np.sum(grad_hidden[:, position, :], axis=0)

        text_len, audio_len, visual_len = self._last_lengths
        grad_text = grad_hidden[:, :text_len, :]
        grad_audio = grad_hidden[:, text_len : text_len + audio_len, :]
        grad_visual = grad_hidden[:, text_len + audio_len : text_len + audio_len + visual_len, :]
        self.text_embedding.backward(grad_text)
        self.audio_projection.backward(grad_audio)
        self.visual_projection.backward(grad_visual)

    def loss_and_backward(
        self,
        *,
        text_tokens: np.ndarray,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        target_ids: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute cross-entropy loss and accumulate gradients."""

        logits = self.forward(text_tokens=text_tokens, audio_features=audio_features, visual_features=visual_features)
        loss, grad_logits = cross_entropy_loss(logits, target_ids)
        self.backward(grad_logits)
        return loss, logits

    def make_optimizer(self, *, learning_rate: float = 3e-4, weight_decay: float = 0.0, grad_clip_norm: float | None = 1.0) -> AdamW:
        """Create an AdamW optimizer over all baseline parameters."""

        return AdamW(self.parameters(), learning_rate=learning_rate, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm)


__all__ = ["NumpyVanillaMultimodalTransformerBaseline", "VanillaMultimodalConfig"]
