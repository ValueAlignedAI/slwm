"""Signal world processor for SLWM Sprint I0/I2.

The default constructor preserves the I0 shape-only behavior for ``TensorSpec``
tests. Passing an ``SLWMCoreConfig`` or processor dimensions enables the Sprint
I2 NumPy implementation: stacked signal blocks with configurable local temporal,
spectral, long-conv, and gated-MLP components.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.baselines.numpy_nn import Parameter
from models.module import ShapeModule
from models.processor.blocks import SignalProcessorBlock
from models.slwm_config import SLWMCoreConfig
from models.types import TensorSpec, ensure_latent, ensure_mask


class SignalWorldProcessor(ShapeModule):
    """Shape-preserving signal world processor.

    Forward input shape:
        ``z: FloatTensor[B,T,D]`` and optional ``mask: BoolTensor[B,T]``.

    Forward output shape:
        ``{"z_world": FloatTensor[B,T,D], "aux": dict}``.

    Configured I2 behavior:
        stacked blocks preserve the canonical ``FloatTensor[B,T,D]`` shape.
        Each novel block can be disabled by config for ablation.
    """

    def __init__(self, config: SLWMCoreConfig | None = None, **kwargs: Any) -> None:
        if config is None and kwargs:
            config = SLWMCoreConfig(**kwargs)
        self.config = config
        self.blocks: list[SignalProcessorBlock] = []
        if self.config is not None:
            rng = np.random.default_rng(int(self.config.seed) + 1000)
            self.blocks = [SignalProcessorBlock(rng, self.config, layer_index=layer) for layer in range(int(self.config.n_layer))]
        self._last_mask: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        """Return trainable processor parameters in deterministic order."""

        params: list[Parameter] = []
        for block in self.blocks:
            params.extend(block.parameters())
        return params

    def parameter_count(self) -> int:
        """Exact instantiated trainable parameter count for the processor."""

        return int(sum(param.size for param in self.parameters()))

    def forward(self, z: Any, mask: Any | None = None, state: Any | None = None) -> dict[str, Any]:
        b, t, d = ensure_latent(z)
        if mask is not None:
            ensure_mask(mask, (b, t))

        if isinstance(z, TensorSpec) or self.config is None:
            z_world = TensorSpec((b, t, d), "float32", "z_world") if isinstance(z, TensorSpec) else z
            implementation = "i0_shape_contract_stub" if isinstance(z, TensorSpec) else "i2_passthrough_unconfigured"
        else:
            if d != self.config.latent_dim:
                raise ValueError(f"Processor expected D={self.config.latent_dim}; got D={d}")
            z_world = np.asarray(z, dtype=np.float64)
            mask_array = None if mask is None else np.asarray(mask, dtype=bool)
            self._last_mask = mask_array
            for block in self.blocks:
                z_world = block.forward(z_world, mask=mask_array)
            implementation = "i2_numpy_signal_world_processor"

        return {
            "z_world": z_world,
            "aux": {
                "processor": self.__class__.__name__,
                "implementation": implementation,
                "state_provided": state is not None,
                "mask_shape": [b, t] if mask is not None else None,
                "layer_count": len(self.blocks),
                "ablation_flags": (
                    None
                    if self.config is None
                    else {
                        "use_local_temporal_mixer": self.config.use_local_temporal_mixer,
                        "use_spectral_mixer": self.config.use_spectral_mixer,
                        "use_long_conv": self.config.use_long_conv,
                        "use_gated_mlp": self.config.use_gated_mlp,
                    }
                ),
            },
        }

    def backward(self, grad_z_world: np.ndarray) -> np.ndarray:
        """Backpropagate through configured I2 processor blocks.

        Args:
            grad_z_world: ``FloatTensor[B,T,D]`` gradient wrt processor output.

        Returns:
            ``FloatTensor[B,T,D]`` gradient wrt processor input.
        """

        if self.config is None:
            return np.asarray(grad_z_world, dtype=np.float64)
        grad = np.asarray(grad_z_world, dtype=np.float64)
        if grad.ndim != 3 or grad.shape[-1] != self.config.latent_dim:
            raise ValueError(f"processor grad must have shape [B,T,{self.config.latent_dim}], got {grad.shape}")
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        if self._last_mask is not None:
            grad *= self._last_mask[:, :, None]
        return grad


__all__ = ["SignalWorldProcessor"]
