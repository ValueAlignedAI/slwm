"""Signal world processor shape contract."""

from __future__ import annotations

from typing import Any

from models.module import ShapeModule
from models.types import TensorSpec, ensure_latent, ensure_mask


class SignalWorldProcessor(ShapeModule):
    """Shape-preserving processor stub.

    Forward input shape:
        ``z: FloatTensor[B,T,D]`` and optional ``mask: BoolTensor[B,T]``.

    Forward output shape:
        ``{"z_world": FloatTensor[B,T,D], "aux": dict}``.

    I0 implements no spectral mixer, attention, long convolution, SSM, or MLP.
    """

    def forward(self, z: Any, mask: Any | None = None, state: Any | None = None) -> dict[str, Any]:
        b, t, d = ensure_latent(z)
        if mask is not None:
            ensure_mask(mask, (b, t))

        z_world = TensorSpec((b, t, d), "float32", "z_world") if isinstance(z, TensorSpec) else z
        return {
            "z_world": z_world,
            "aux": {
                "processor": self.__class__.__name__,
                "implementation": "i0_shape_contract_stub",
                "state_provided": state is not None,
                "mask_shape": [b, t] if mask is not None else None,
            },
        }
