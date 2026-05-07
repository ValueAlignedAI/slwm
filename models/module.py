"""Tiny module base used before PyTorch is introduced as a dependency.

I0 cannot require real neural-network behavior. This class mirrors the small
portion of ``nn.Module`` needed for importable, callable stubs while keeping the
current repository dependency-free.
"""

from __future__ import annotations

from typing import Any


class ShapeModule:
    """Callable shape-only module.

    Forward shape contract: subclasses accept and return canonical dictionaries
    containing shape carriers such as ``TensorSpec``. No parameters or trainable
    state are defined in I0.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract guard
        raise NotImplementedError

    def parameters(self) -> tuple[()]:
        """Return no trainable parameters for I0 stubs."""

        return ()
