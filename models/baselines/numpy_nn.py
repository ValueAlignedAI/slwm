"""Small NumPy neural-network primitives for Sprint I1 baseline smoke tests.

This file intentionally implements only generic Transformer baseline machinery:
linear layers, embeddings, layer norm, multi-head self-attention, MLP blocks,
cross-entropy, and AdamW. It does **not** include SLWM novelty components such as
spectral mixers, long-convolution/SSM blocks, learned policy gates, or
uncertainty heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


@dataclass
class Parameter:
    """Trainable NumPy parameter with an accumulated gradient.

    Shape contract:
        ``value`` and ``grad`` always have identical shapes. Gradients are
        accumulated until ``zero_grad`` is called.
    """

    value: Array
    name: str

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=np.float64)
        self.grad = np.zeros_like(self.value)

    @property
    def size(self) -> int:
        """Number of scalar trainable parameters."""

        return int(self.value.size)

    def zero_grad(self) -> None:
        """Reset the accumulated gradient to zero."""

        self.grad.fill(0.0)


def _normal(rng: np.random.Generator, shape: tuple[int, ...], *, scale: float = 0.02) -> Array:
    return rng.normal(0.0, scale, size=shape).astype(np.float64)


def gelu(x: Array) -> Array:
    """GPT-style approximate GELU activation."""

    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * np.power(x, 3))))


def gelu_grad(x: Array) -> Array:
    """Derivative of the approximate GELU activation."""

    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * np.power(x, 3))
    tanh_u = np.tanh(u)
    du = c * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
    return 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - np.power(tanh_u, 2)) * du


def softmax(logits: Array, axis: int = -1) -> Array:
    """Numerically stable softmax."""

    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def softmax_backward(grad_output: Array, probabilities: Array) -> Array:
    """Backward pass for softmax along the last axis."""

    dot = np.sum(grad_output * probabilities, axis=-1, keepdims=True)
    return probabilities * (grad_output - dot)


def causal_attention_mask(length: int) -> Array:
    """Return a boolean causal mask with shape ``[1,1,T,T]``."""

    return np.tril(np.ones((1, 1, int(length), int(length)), dtype=bool))


def cross_entropy_loss(logits: Array, targets: Array, ignore_index: int | None = None) -> tuple[float, Array]:
    """Return mean cross-entropy and gradient w.r.t. logits.

    Shape contract:
        ``logits``: ``FloatTensor[B,T,V]``.
        ``targets``: ``IntTensor[B,T]`` with values in ``[0,V)`` unless equal
        to ``ignore_index``.
        returned gradient: ``FloatTensor[B,T,V]``.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,T,V], got {logits.shape}")
    if targets.shape != logits.shape[:2]:
        raise ValueError(f"targets shape {targets.shape} must match logits [B,T] {logits.shape[:2]}")

    bsz, length, vocab_size = logits.shape
    flat_logits = logits.reshape(bsz * length, vocab_size)
    flat_targets = np.asarray(targets, dtype=np.int64).reshape(bsz * length)
    if ignore_index is None:
        valid = np.ones_like(flat_targets, dtype=bool)
    else:
        valid = flat_targets != int(ignore_index)
    valid_count = int(np.sum(valid))
    if valid_count <= 0:
        raise ValueError("cross_entropy_loss requires at least one valid target")
    if np.any((flat_targets[valid] < 0) | (flat_targets[valid] >= vocab_size)):
        raise ValueError("targets must be in [0, vocab_size) for valid positions")

    probabilities = softmax(flat_logits, axis=-1)
    picked = probabilities[np.arange(flat_targets.size)[valid], flat_targets[valid]]
    loss = float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))

    grad = probabilities
    valid_rows = np.arange(flat_targets.size)[valid]
    grad[valid_rows, flat_targets[valid]] -= 1.0
    grad[~valid, :] = 0.0
    grad /= float(valid_count)
    return loss, grad.reshape(logits.shape)


class Embedding:
    """Trainable embedding lookup.

    Forward shape contract:
        input IDs ``IntTensor[B,T]`` -> embeddings ``FloatTensor[B,T,D]``.
    """

    def __init__(self, rng: np.random.Generator, num_embeddings: int, embedding_dim: int, *, name: str) -> None:
        self.weight = Parameter(_normal(rng, (int(num_embeddings), int(embedding_dim))), f"{name}.weight")
        self._last_ids: Array | None = None

    def parameters(self) -> list[Parameter]:
        return [self.weight]

    def forward(self, input_ids: Array) -> Array:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError(f"embedding input must have shape [B,T], got {ids.shape}")
        if np.any(ids < 0) or np.any(ids >= self.weight.value.shape[0]):
            raise ValueError("embedding input IDs out of range")
        self._last_ids = ids
        return self.weight.value[ids]

    def backward(self, grad_output: Array) -> None:
        if self._last_ids is None:
            raise RuntimeError("Embedding.backward called before forward")
        if grad_output.shape != self._last_ids.shape + (self.weight.value.shape[1],):
            raise ValueError("embedding grad shape does not match cached IDs")
        np.add.at(self.weight.grad, self._last_ids, grad_output)


class Linear:
    """Trainable affine projection over the last dimension.

    Forward shape contract:
        ``FloatTensor[...,in_features]`` -> ``FloatTensor[...,out_features]``.
    """

    def __init__(self, rng: np.random.Generator, in_features: int, out_features: int, *, name: str, bias: bool = True) -> None:
        scale = 1.0 / np.sqrt(max(1, int(in_features)))
        self.weight = Parameter(_normal(rng, (int(in_features), int(out_features)), scale=scale), f"{name}.weight")
        self.bias = Parameter(np.zeros((int(out_features),), dtype=np.float64), f"{name}.bias") if bias else None
        self._last_x: Array | None = None

    def parameters(self) -> list[Parameter]:
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def forward(self, x: Array) -> Array:
        if x.shape[-1] != self.weight.value.shape[0]:
            raise ValueError(f"linear expected last dim {self.weight.value.shape[0]}, got {x.shape[-1]}")
        self._last_x = x
        y = np.matmul(x, self.weight.value)
        if self.bias is not None:
            y = y + self.bias.value
        return y

    def backward(self, grad_output: Array) -> Array:
        if self._last_x is None:
            raise RuntimeError("Linear.backward called before forward")
        x = self._last_x
        x2 = x.reshape(-1, x.shape[-1])
        g2 = grad_output.reshape(-1, grad_output.shape[-1])
        self.weight.grad += np.matmul(x2.T, g2)
        if self.bias is not None:
            self.bias.grad += np.sum(g2, axis=0)
        return np.matmul(grad_output, self.weight.value.T)


class LayerNorm:
    """Layer normalization over the final dimension.

    Forward shape contract:
        ``FloatTensor[...,D]`` -> ``FloatTensor[...,D]``.
    """

    def __init__(self, normalized_shape: int, *, name: str, eps: float = 1e-5) -> None:
        self.gamma = Parameter(np.ones((int(normalized_shape),), dtype=np.float64), f"{name}.gamma")
        self.beta = Parameter(np.zeros((int(normalized_shape),), dtype=np.float64), f"{name}.beta")
        self.eps = float(eps)
        self._cache: tuple[Array, Array, Array, Array] | None = None

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]

    def forward(self, x: Array) -> Array:
        if x.shape[-1] != self.gamma.value.shape[0]:
            raise ValueError(f"layer norm expected last dim {self.gamma.value.shape[0]}, got {x.shape[-1]}")
        mean = np.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        var = np.mean(np.power(centered, 2), axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        xhat = centered * inv_std
        self._cache = (centered, inv_std, xhat, x)
        return xhat * self.gamma.value + self.beta.value

    def backward(self, grad_output: Array) -> Array:
        if self._cache is None:
            raise RuntimeError("LayerNorm.backward called before forward")
        centered, inv_std, xhat, x = self._cache
        reduction_axes = tuple(range(grad_output.ndim - 1))
        self.gamma.grad += np.sum(grad_output * xhat, axis=reduction_axes)
        self.beta.grad += np.sum(grad_output, axis=reduction_axes)

        n = x.shape[-1]
        grad_xhat = grad_output * self.gamma.value
        grad_var = np.sum(grad_xhat * centered * -0.5 * np.power(inv_std, 3), axis=-1, keepdims=True)
        grad_mean = np.sum(grad_xhat * -inv_std, axis=-1, keepdims=True) + grad_var * np.mean(-2.0 * centered, axis=-1, keepdims=True)
        return grad_xhat * inv_std + grad_var * 2.0 * centered / n + grad_mean / n


class MultiHeadSelfAttention:
    """Vanilla multi-head self-attention for decoder/encoder baselines.

    Forward shape contract:
        hidden states ``FloatTensor[B,T,D]`` -> ``FloatTensor[B,T,D]``.
        Optional attention mask uses shape ``BoolTensor[B,T]`` where true means
        valid key/value position. Causal masking is controlled by config.
    """

    def __init__(self, rng: np.random.Generator, n_embd: int, n_head: int, *, name: str, causal: bool) -> None:
        if int(n_embd) % int(n_head) != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_embd = int(n_embd)
        self.n_head = int(n_head)
        self.head_dim = self.n_embd // self.n_head
        self.causal = bool(causal)
        self.q_proj = Linear(rng, self.n_embd, self.n_embd, name=f"{name}.q_proj")
        self.k_proj = Linear(rng, self.n_embd, self.n_embd, name=f"{name}.k_proj")
        self.v_proj = Linear(rng, self.n_embd, self.n_embd, name=f"{name}.v_proj")
        self.out_proj = Linear(rng, self.n_embd, self.n_embd, name=f"{name}.out_proj")
        self._cache: tuple[Array, Array, Array, Array, Array | None] | None = None

    def parameters(self) -> list[Parameter]:
        return self.q_proj.parameters() + self.k_proj.parameters() + self.v_proj.parameters() + self.out_proj.parameters()

    def _split_heads(self, x: Array) -> Array:
        bsz, length, _ = x.shape
        return x.reshape(bsz, length, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Array) -> Array:
        bsz, _, length, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(bsz, length, self.n_embd)

    def _allowed_mask(self, batch_size: int, length: int, attention_mask: Array | None) -> Array | None:
        allowed: Array | None = None
        if self.causal:
            allowed = causal_attention_mask(length)
        if attention_mask is not None:
            mask = np.asarray(attention_mask, dtype=bool)
            if mask.shape != (batch_size, length):
                raise ValueError(f"attention mask must have shape {(batch_size, length)}, got {mask.shape}")
            key_allowed = mask[:, None, None, :]
            allowed = key_allowed if allowed is None else (allowed & key_allowed)
        return allowed

    def forward(self, x: Array, attention_mask: Array | None = None) -> Array:
        if x.ndim != 3 or x.shape[-1] != self.n_embd:
            raise ValueError(f"attention input must have shape [B,T,{self.n_embd}], got {x.shape}")
        bsz, length, _ = x.shape
        q = self._split_heads(self.q_proj.forward(x))
        k = self._split_heads(self.k_proj.forward(x))
        v = self._split_heads(self.v_proj.forward(x))
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(float(self.head_dim))
        allowed = self._allowed_mask(bsz, length, attention_mask)
        if allowed is not None:
            scores = np.where(allowed, scores, -1e9)
        probabilities = softmax(scores, axis=-1)
        attended = np.matmul(probabilities, v)
        merged = self._merge_heads(attended)
        self._cache = (q, k, v, probabilities, allowed)
        return self.out_proj.forward(merged)

    def backward(self, grad_output: Array) -> Array:
        if self._cache is None:
            raise RuntimeError("MultiHeadSelfAttention.backward called before forward")
        q, k, v, probabilities, allowed = self._cache
        grad_merged = self.out_proj.backward(grad_output)
        grad_attended = self._split_heads(grad_merged)
        grad_probabilities = np.matmul(grad_attended, v.transpose(0, 1, 3, 2))
        grad_v = np.matmul(probabilities.transpose(0, 1, 3, 2), grad_attended)
        grad_scores = softmax_backward(grad_probabilities, probabilities)
        if allowed is not None:
            grad_scores = np.where(allowed, grad_scores, 0.0)
        scale = 1.0 / np.sqrt(float(self.head_dim))
        grad_q = np.matmul(grad_scores, k) * scale
        grad_k = np.matmul(grad_scores.transpose(0, 1, 3, 2), q) * scale

        grad_q_merged = self._merge_heads(grad_q)
        grad_k_merged = self._merge_heads(grad_k)
        grad_v_merged = self._merge_heads(grad_v)
        grad_x = self.q_proj.backward(grad_q_merged)
        grad_x += self.k_proj.backward(grad_k_merged)
        grad_x += self.v_proj.backward(grad_v_merged)
        return grad_x


class FeedForward:
    """Transformer MLP with GELU activation.

    Forward shape contract:
        ``FloatTensor[B,T,D]`` -> ``FloatTensor[B,T,D]``.
    """

    def __init__(self, rng: np.random.Generator, n_embd: int, intermediate_size: int, *, name: str) -> None:
        self.fc_in = Linear(rng, int(n_embd), int(intermediate_size), name=f"{name}.fc_in")
        self.fc_out = Linear(rng, int(intermediate_size), int(n_embd), name=f"{name}.fc_out")
        self._pre_activation: Array | None = None

    def parameters(self) -> list[Parameter]:
        return self.fc_in.parameters() + self.fc_out.parameters()

    def forward(self, x: Array) -> Array:
        pre = self.fc_in.forward(x)
        self._pre_activation = pre
        return self.fc_out.forward(gelu(pre))

    def backward(self, grad_output: Array) -> Array:
        if self._pre_activation is None:
            raise RuntimeError("FeedForward.backward called before forward")
        grad_hidden = self.fc_out.backward(grad_output)
        grad_pre = grad_hidden * gelu_grad(self._pre_activation)
        return self.fc_in.backward(grad_pre)


class TransformerBlock:
    """Pre-layer-norm Transformer block for baseline models.

    Forward shape contract:
        ``FloatTensor[B,T,D]`` plus optional ``BoolTensor[B,T]`` attention mask
        -> ``FloatTensor[B,T,D]``.
    """

    def __init__(self, rng: np.random.Generator, n_embd: int, n_head: int, intermediate_size: int, *, name: str, causal: bool) -> None:
        self.ln_1 = LayerNorm(int(n_embd), name=f"{name}.ln_1")
        self.attn = MultiHeadSelfAttention(rng, int(n_embd), int(n_head), name=f"{name}.attn", causal=causal)
        self.ln_2 = LayerNorm(int(n_embd), name=f"{name}.ln_2")
        self.mlp = FeedForward(rng, int(n_embd), int(intermediate_size), name=f"{name}.mlp")

    def parameters(self) -> list[Parameter]:
        return self.ln_1.parameters() + self.attn.parameters() + self.ln_2.parameters() + self.mlp.parameters()

    def forward(self, x: Array, attention_mask: Array | None = None) -> Array:
        attn_out = self.attn.forward(self.ln_1.forward(x), attention_mask=attention_mask)
        x_with_attn = x + attn_out
        mlp_out = self.mlp.forward(self.ln_2.forward(x_with_attn))
        return x_with_attn + mlp_out

    def backward(self, grad_output: Array) -> Array:
        grad_x_with_attn = grad_output + self.ln_2.backward(self.mlp.backward(grad_output))
        grad_x = grad_x_with_attn + self.ln_1.backward(self.attn.backward(grad_x_with_attn))
        return grad_x


def iter_parameters(modules: Iterable[object]) -> list[Parameter]:
    """Collect trainable parameters from modules exposing ``parameters()``."""

    params: list[Parameter] = []
    for module in modules:
        getter = getattr(module, "parameters", None)
        if getter is None:
            continue
        params.extend(getter())
    return params


class AdamW:
    """Small deterministic AdamW optimizer for baseline smoke runs."""

    def __init__(
        self,
        parameters: Sequence[Parameter],
        *,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2 = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.step_count = 0
        self._m = [np.zeros_like(param.value) for param in self.parameters]
        self._v = [np.zeros_like(param.value) for param in self.parameters]

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.zero_grad()

    def _global_grad_norm(self) -> float:
        total = 0.0
        for param in self.parameters:
            total += float(np.sum(np.power(param.grad, 2)))
        return float(np.sqrt(total))

    def step(self) -> float:
        """Apply one AdamW update and return the pre-clipping gradient norm."""

        self.step_count += 1
        grad_norm = self._global_grad_norm()
        scale = 1.0
        if self.grad_clip_norm is not None and grad_norm > self.grad_clip_norm > 0.0:
            scale = self.grad_clip_norm / (grad_norm + 1e-12)
        for index, param in enumerate(self.parameters):
            grad = param.grad * scale
            if self.weight_decay:
                grad = grad + self.weight_decay * param.value
            self._m[index] = self.beta1 * self._m[index] + (1.0 - self.beta1) * grad
            self._v[index] = self.beta2 * self._v[index] + (1.0 - self.beta2) * np.power(grad, 2)
            m_hat = self._m[index] / (1.0 - self.beta1**self.step_count)
            v_hat = self._v[index] / (1.0 - self.beta2**self.step_count)
            param.value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return grad_norm


__all__ = [
    "AdamW",
    "Array",
    "Embedding",
    "FeedForward",
    "LayerNorm",
    "Linear",
    "MultiHeadSelfAttention",
    "Parameter",
    "TransformerBlock",
    "causal_attention_mask",
    "cross_entropy_loss",
    "gelu",
    "gelu_grad",
    "iter_parameters",
    "softmax",
]
