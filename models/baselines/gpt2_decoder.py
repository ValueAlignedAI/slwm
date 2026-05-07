"""GPT-2-small-style decoder-only baseline for Sprint I1.

The production-scale configuration is represented by exact parameter-count
formulas. The NumPy implementation is intentionally tiny-config friendly so the
repo can run deterministic forward/backward and overfit smoke tests without
requiring a heavyweight training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from models.baselines.numpy_nn import AdamW, Embedding, LayerNorm, Linear, Parameter, TransformerBlock, cross_entropy_loss
from models.baselines.parameter_count import GPT2ParameterBreakdown, gpt2_parameter_breakdown


@dataclass(frozen=True)
class GPT2DecoderConfig:
    """Configuration for a GPT-2-style decoder-only Transformer.

    Shape contract:
        ``input_ids`` and ``target_ids`` use shape ``IntTensor[B,T]`` with
        ``T <= context_length``. The model returns logits with shape
        ``FloatTensor[B,T,vocab_size]``.
    """

    vocab_size: int = 50_257
    context_length: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    intermediate_size: int | None = None
    tie_embeddings: bool = True
    seed: int = 0
    tokenizer: str = "gpt2_bpe"

    @property
    def d_ff(self) -> int:
        """Transformer MLP width."""

        return int(self.intermediate_size if self.intermediate_size is not None else 4 * self.n_embd)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "GPT2DecoderConfig":
        """Construct from a JSON config ``model`` section."""

        tokenizer_config = mapping.get("tokenizer", "gpt2_bpe")
        tokenizer = tokenizer_config.get("type", "gpt2_bpe") if isinstance(tokenizer_config, Mapping) else tokenizer_config
        return cls(
            vocab_size=int(mapping.get("vocab_size", cls.vocab_size)),
            context_length=int(mapping.get("context_length", mapping.get("block_size", cls.context_length))),
            n_layer=int(mapping.get("n_layer", cls.n_layer)),
            n_embd=int(mapping.get("n_embd", cls.n_embd)),
            n_head=int(mapping.get("n_head", cls.n_head)),
            intermediate_size=(None if mapping.get("intermediate_size") is None else int(mapping["intermediate_size"])),
            tie_embeddings=bool(mapping.get("tie_embeddings", True)),
            seed=int(mapping.get("seed", mapping.get("runtime_seed", 0))),
            tokenizer=str(tokenizer),
        )

    def parameter_breakdown(self) -> GPT2ParameterBreakdown:
        """Exact parameter count for this GPT-2-style configuration."""

        return gpt2_parameter_breakdown(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            intermediate_size=self.d_ff,
            tie_embeddings=self.tie_embeddings,
        )


class NumpyGPT2DecoderBaseline:
    """Decoder-only causal Transformer language-model baseline.

    Forward shape contract:
        input token IDs: ``IntTensor[B,T]``.
        output logits: ``FloatTensor[B,T,V]``.

    Training contract:
        ``loss_and_backward(input_ids, target_ids)`` computes next-token-style
        cross-entropy and accumulates gradients in all trainable parameters.
    """

    def __init__(self, config: GPT2DecoderConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.token_embedding = Embedding(rng, config.vocab_size, config.n_embd, name="token_embedding")
        self.position_embedding = Parameter(
            rng.normal(0.0, 0.02, size=(config.context_length, config.n_embd)).astype(np.float64),
            "position_embedding",
        )
        self.blocks = [
            TransformerBlock(
                rng,
                config.n_embd,
                config.n_head,
                config.d_ff,
                name=f"blocks.{layer}",
                causal=True,
            )
            for layer in range(config.n_layer)
        ]
        self.final_layer_norm = LayerNorm(config.n_embd, name="ln_f")
        self.lm_head = None if config.tie_embeddings else Linear(rng, config.n_embd, config.vocab_size, name="lm_head")
        self._last_input_ids: np.ndarray | None = None
        self._last_length: int | None = None
        self._last_hidden: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        """Return trainable parameters in deterministic order."""

        params: list[Parameter] = []
        params.extend(self.token_embedding.parameters())
        params.append(self.position_embedding)
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.final_layer_norm.parameters())
        if self.lm_head is None:
            pass
        else:
            params.extend(self.lm_head.parameters())
        return params

    def parameter_count(self) -> int:
        """Exact instantiated trainable parameter count."""

        return int(sum(param.size for param in self.parameters()))

    def formula_parameter_count(self) -> int:
        """Exact formula count for the configured GPT-2-style model."""

        return self.config.parameter_breakdown().total

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Compute causal language-model logits.

        Args:
            input_ids: ``IntTensor[B,T]`` with ``T <= context_length``.
        Returns:
            ``FloatTensor[B,T,vocab_size]`` logits.
        """

        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,T], got {ids.shape}")
        _, length = ids.shape
        if length > self.config.context_length:
            raise ValueError(f"input length {length} exceeds context_length={self.config.context_length}")
        hidden = self.token_embedding.forward(ids) + self.position_embedding.value[np.arange(length)][None, :, :]
        for block in self.blocks:
            hidden = block.forward(hidden)
        hidden = self.final_layer_norm.forward(hidden)
        self._last_input_ids = ids
        self._last_length = length
        self._last_hidden = hidden
        if self.lm_head is None:
            logits = np.matmul(hidden, self.token_embedding.weight.value.T)
        else:
            logits = self.lm_head.forward(hidden)
        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        """Backpropagate from logits through the decoder baseline."""

        if self._last_input_ids is None or self._last_length is None or self._last_hidden is None:
            raise RuntimeError("backward called before forward")
        hidden = self._last_hidden
        if self.lm_head is None:
            grad_flat = grad_logits.reshape(-1, grad_logits.shape[-1])
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            self.token_embedding.weight.grad += np.matmul(grad_flat.T, hidden_flat)
            grad_hidden = np.matmul(grad_logits, self.token_embedding.weight.value)
        else:
            grad_hidden = self.lm_head.backward(grad_logits)

        grad_hidden = self.final_layer_norm.backward(grad_hidden)
        for block in reversed(self.blocks):
            grad_hidden = block.backward(grad_hidden)
        self.position_embedding.grad[np.arange(self._last_length)] += np.sum(grad_hidden, axis=0)
        self.token_embedding.backward(grad_hidden)

    def loss_and_backward(self, input_ids: np.ndarray, target_ids: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute cross-entropy loss and accumulate gradients."""

        logits = self.forward(input_ids)
        loss, grad_logits = cross_entropy_loss(logits, target_ids)
        self.backward(grad_logits)
        return loss, logits

    def make_optimizer(self, *, learning_rate: float = 3e-4, weight_decay: float = 0.0, grad_clip_norm: float | None = 1.0) -> AdamW:
        """Create an AdamW optimizer over all model parameters."""

        return AdamW(self.parameters(), learning_rate=learning_rate, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm)


__all__ = ["GPT2DecoderConfig", "NumpyGPT2DecoderBaseline"]
