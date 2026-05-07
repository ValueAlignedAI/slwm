"""Shape and dtype helpers for SLWM-124M Sprint I0 contracts.

The canonical latent tensor form is ``Z: FloatTensor[B,T,D]`` with a matching
``mask: BoolTensor[B,T]``. Because PyTorch is intentionally not required in I0,
``TensorSpec`` carries shape/dtype metadata for tests and docs. The validators
also accept tensor-like objects with a ``shape`` attribute so later PyTorch
modules can reuse the same contract checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


Shape = tuple[int, ...]


@dataclass(frozen=True)
class TensorSpec:
    """Dependency-free shape carrier for tensor contracts.

    Args:
        shape: Tensor shape, e.g. ``(B, T, D)`` for latent fields.
        dtype: Logical dtype name such as ``"float32"`` or ``"bool"``.
        name: Human-readable tensor name for diagnostics.

    Shape contract examples:
        latent field: ``TensorSpec((B,T,D), "float32", "z")``
        mask: ``TensorSpec((B,T), "bool", "mask")``
    """

    shape: Shape
    dtype: str = "float32"
    name: str = "tensor"

    def __post_init__(self) -> None:
        normalized = tuple(int(dim) for dim in self.shape)
        if not normalized:
            raise ValueError("TensorSpec shape must have at least one dimension")
        if any(dim <= 0 for dim in normalized):
            raise ValueError(f"TensorSpec dimensions must be positive, got {normalized}")
        object.__setattr__(self, "shape", normalized)

    @property
    def ndim(self) -> int:
        """Number of dimensions in ``shape``."""

        return len(self.shape)

    def as_dict(self) -> dict[str, Any]:
        """Serialize this spec for registry/config metadata."""

        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


def _normalize_shape(shape: Iterable[Any]) -> Shape:
    return tuple(int(dim) for dim in tuple(shape))


def shape_of(tensor_like: Any) -> Shape:
    """Return a tuple shape from ``TensorSpec`` or a tensor-like object."""

    if isinstance(tensor_like, TensorSpec):
        return tensor_like.shape
    if hasattr(tensor_like, "shape"):
        return _normalize_shape(getattr(tensor_like, "shape"))
    raise TypeError(f"Object {type(tensor_like)!r} does not expose a tensor shape")


def dtype_name(tensor_like: Any) -> str:
    """Return a best-effort dtype name from ``TensorSpec`` or tensor-like object."""

    if isinstance(tensor_like, TensorSpec):
        return tensor_like.dtype
    dtype = getattr(tensor_like, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def ensure_rank(tensor_like: Any, rank: int, name: str) -> Shape:
    """Validate tensor rank and return its shape."""

    shape = shape_of(tensor_like)
    if len(shape) != rank:
        raise ValueError(f"{name} must have rank {rank}; got shape {shape}")
    return shape


def ensure_latent(z: Any, name: str = "z") -> tuple[int, int, int]:
    """Validate canonical latent shape ``FloatTensor[B,T,D]``.

    Returns:
        ``(B, T, D)`` from the checked latent shape.
    """

    shape = ensure_rank(z, 3, name)
    dtype = dtype_name(z)
    if "float" not in dtype and dtype != "unknown":
        raise ValueError(f"{name} must be a floating tensor/spec; got dtype {dtype!r}")
    return shape  # type: ignore[return-value]


def ensure_mask(mask: Any, expected_shape: tuple[int, int] | None = None, name: str = "mask") -> tuple[int, int]:
    """Validate mask shape ``BoolTensor[B,T]``.

    Args:
        mask: Mask tensor/spec.
        expected_shape: Optional ``(B,T)`` shape to match.
        name: Name used in error messages.
    """

    shape = ensure_rank(mask, 2, name)
    dtype = dtype_name(mask)
    if "bool" not in dtype and dtype != "unknown":
        raise ValueError(f"{name} must be a bool tensor/spec; got dtype {dtype!r}")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"{name} shape {shape} does not match expected {expected_shape}")
    return shape  # type: ignore[return-value]


def make_latent_spec(batch_size: int, latent_length: int, latent_dim: int, name: str = "z") -> TensorSpec:
    """Create ``FloatTensor[B,T,D]`` shape metadata."""

    return TensorSpec((batch_size, latent_length, latent_dim), "float32", name)


def make_mask_spec(batch_size: int, latent_length: int, name: str = "mask") -> TensorSpec:
    """Create ``BoolTensor[B,T]`` shape metadata."""

    return TensorSpec((batch_size, latent_length), "bool", name)
