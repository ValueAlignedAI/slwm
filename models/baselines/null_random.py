"""Null and random controls for Sprint I1 baseline/probe sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from models.baselines.numpy_nn import cross_entropy_loss


@dataclass(frozen=True)
class NullBaselineResult:
    """Metric container for a parameter-free null baseline."""

    loss: float
    parameter_count: int = 0


class UniformLogitBaseline:
    """Parameter-free baseline that emits uniform zero logits.

    Forward shape contract:
        requested ``batch_size`` and ``length`` -> logits
        ``FloatTensor[B,T,V]`` where every token has equal probability.
    """

    parameter_count = 0

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)

    def forward(self, *, batch_size: int, length: int) -> np.ndarray:
        """Return zero logits with shape ``[B,T,V]``."""

        return np.zeros((int(batch_size), int(length), self.vocab_size), dtype=np.float64)

    def evaluate(self, targets: np.ndarray) -> NullBaselineResult:
        """Compute cross-entropy against target IDs without training."""

        logits = self.forward(batch_size=targets.shape[0], length=targets.shape[1])
        loss, _ = cross_entropy_loss(logits, targets)
        return NullBaselineResult(loss=loss)


class RandomLogitBaseline:
    """Seeded parameter-free random-logit control."""

    parameter_count = 0

    def __init__(self, vocab_size: int, *, seed: int = 0, scale: float = 1.0) -> None:
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.scale = float(scale)

    def forward(self, *, batch_size: int, length: int) -> np.ndarray:
        """Return deterministic random logits with shape ``[B,T,V]``."""

        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, self.scale, size=(int(batch_size), int(length), self.vocab_size)).astype(np.float64)

    def evaluate(self, targets: np.ndarray) -> NullBaselineResult:
        """Compute cross-entropy against target IDs without training."""

        logits = self.forward(batch_size=targets.shape[0], length=targets.shape[1])
        loss, _ = cross_entropy_loss(logits, targets)
        return NullBaselineResult(loss=loss)


def shuffled_targets(targets: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Return a deterministic shuffled-position target control.

    Shape contract:
        ``IntTensor[B,T]`` -> ``IntTensor[B,T]`` with the same values but a
        seeded permutation over flattened positions.
    """

    flat = np.asarray(targets, dtype=np.int64).reshape(-1).copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(flat)
    return flat.reshape(targets.shape)


__all__ = ["NullBaselineResult", "RandomLogitBaseline", "UniformLogitBaseline", "shuffled_targets"]
