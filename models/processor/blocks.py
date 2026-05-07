"""Sprint I2 SLWM processor blocks implemented with NumPy.

These blocks are intentionally small and deterministic so shape and
forward/backward smoke tests can run without a heavyweight autograd stack. Every
novel component is instantiated only when its corresponding config flag is
enabled, supporting ablations such as no-spectral and no-long-conv.
"""

from __future__ import annotations

import numpy as np

from models.baselines.numpy_nn import LayerNorm, Linear, Parameter
from models.slwm_config import SLWMCoreConfig


Array = np.ndarray


def silu(x: Array) -> Array:
    """SiLU activation used by the gated MLP."""

    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return x * sigmoid


def silu_grad(x: Array) -> Array:
    """Derivative of SiLU."""

    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return sigmoid + x * sigmoid * (1.0 - sigmoid)


class DepthwiseTemporalConv:
    """Depthwise temporal convolution over the latent sequence.

    Forward shape contract:
        ``FloatTensor[B,T,D]`` -> ``FloatTensor[B,T,D]`` using same-length
        zero-padded convolution with one kernel per channel.
    """

    def __init__(self, rng: np.random.Generator, latent_dim: int, kernel_size: int, *, name: str) -> None:
        kernel = int(kernel_size)
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("Temporal convolution kernel_size must be a positive odd integer")
        self.latent_dim = int(latent_dim)
        self.kernel_size = kernel
        self.name = str(name)
        scale = 0.02 / np.sqrt(float(kernel))
        self.kernel = Parameter(rng.normal(0.0, scale, size=(kernel, self.latent_dim)).astype(np.float64), f"{name}.kernel")
        self.bias = Parameter(np.zeros((self.latent_dim,), dtype=np.float64), f"{name}.bias")
        self._last_x: Array | None = None

    def parameters(self) -> list[Parameter]:
        return [self.kernel, self.bias]

    def forward(self, x: Array) -> Array:
        if x.ndim != 3 or x.shape[-1] != self.latent_dim:
            raise ValueError(f"{self.name} expected [B,T,{self.latent_dim}], got {x.shape}")
        self._last_x = x
        pad = self.kernel_size // 2
        padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="constant")
        y = np.zeros_like(x)
        for offset in range(self.kernel_size):
            y += padded[:, offset : offset + x.shape[1], :] * self.kernel.value[offset][None, None, :]
        return y + self.bias.value[None, None, :]

    def backward(self, grad_output: Array) -> Array:
        if self._last_x is None:
            raise RuntimeError(f"{self.name}.backward called before forward")
        x = self._last_x
        if grad_output.shape != x.shape:
            raise ValueError(f"{self.name} grad shape {grad_output.shape} does not match input {x.shape}")
        pad = self.kernel_size // 2
        padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="constant")
        grad_padded = np.zeros_like(padded)
        for offset in range(self.kernel_size):
            window = padded[:, offset : offset + x.shape[1], :]
            self.kernel.grad[offset] += np.sum(grad_output * window, axis=(0, 1))
            grad_padded[:, offset : offset + x.shape[1], :] += grad_output * self.kernel.value[offset][None, None, :]
        self.bias.grad += np.sum(grad_output, axis=(0, 1))
        return grad_padded[:, pad : pad + x.shape[1], :]


class SpectralMixer:
    """DCT-like spectral mixer with learned frequency/channel gains.

    Forward shape contract:
        ``FloatTensor[B,T,D]`` -> ``FloatTensor[B,T,D]``. The mixer projects the
        time axis onto an orthonormal cosine basis, applies learned gains per
        frequency/channel, and reconstructs the sequence.
    """

    def __init__(self, rng: np.random.Generator, latent_dim: int, context_length: int, modes: int, *, name: str) -> None:
        self.latent_dim = int(latent_dim)
        self.context_length = int(context_length)
        self.modes = max(1, min(int(modes), self.context_length))
        self.name = str(name)
        self.filter = Parameter(
            rng.normal(0.0, 0.02, size=(self.modes, self.latent_dim)).astype(np.float64),
            f"{name}.filter",
        )
        self._basis_cache: dict[tuple[int, int], Array] = {}
        self._last_basis: Array | None = None
        self._last_spectrum: Array | None = None
        self._last_modes: int | None = None

    def parameters(self) -> list[Parameter]:
        return [self.filter]

    def _basis(self, length: int, modes: int) -> Array:
        key = (int(length), int(modes))
        if key not in self._basis_cache:
            t = np.arange(length, dtype=np.float64)[:, None]
            k = np.arange(modes, dtype=np.float64)[None, :]
            basis = np.cos(np.pi * (t + 0.5) * k / float(length))
            basis[:, 0] *= 1.0 / np.sqrt(float(length))
            if modes > 1:
                basis[:, 1:] *= np.sqrt(2.0 / float(length))
            self._basis_cache[key] = basis
        return self._basis_cache[key]

    def forward(self, x: Array) -> Array:
        if x.ndim != 3 or x.shape[-1] != self.latent_dim:
            raise ValueError(f"{self.name} expected [B,T,{self.latent_dim}], got {x.shape}")
        length = x.shape[1]
        modes = min(self.modes, length)
        basis = self._basis(length, modes)
        spectrum = np.einsum("tm,btd->bmd", basis, x)
        filtered = spectrum * self.filter.value[:modes][None, :, :]
        y = np.einsum("tm,bmd->btd", basis, filtered)
        self._last_basis = basis
        self._last_spectrum = spectrum
        self._last_modes = modes
        return y

    def backward(self, grad_output: Array) -> Array:
        if self._last_basis is None or self._last_spectrum is None or self._last_modes is None:
            raise RuntimeError(f"{self.name}.backward called before forward")
        basis = self._last_basis
        spectrum = self._last_spectrum
        modes = self._last_modes
        grad_spectral_output = np.einsum("tm,btd->bmd", basis, grad_output)
        self.filter.grad[:modes] += np.sum(grad_spectral_output * spectrum, axis=0)
        grad_spectrum = grad_spectral_output * self.filter.value[:modes][None, :, :]
        return np.einsum("tm,bmd->btd", basis, grad_spectrum)


class GatedMLP:
    """SwiGLU-style channel mixer for latent fields.

    Forward shape contract:
        ``FloatTensor[B,T,D]`` -> ``FloatTensor[B,T,D]``.
    """

    def __init__(self, rng: np.random.Generator, latent_dim: int, hidden_dim: int, *, name: str) -> None:
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.value_proj = Linear(rng, self.latent_dim, self.hidden_dim, name=f"{name}.value_proj")
        self.gate_proj = Linear(rng, self.latent_dim, self.hidden_dim, name=f"{name}.gate_proj")
        self.out_proj = Linear(rng, self.hidden_dim, self.latent_dim, name=f"{name}.out_proj")
        self._last_value: Array | None = None
        self._last_gate: Array | None = None
        self._last_gate_activation: Array | None = None

    def parameters(self) -> list[Parameter]:
        return self.value_proj.parameters() + self.gate_proj.parameters() + self.out_proj.parameters()

    def forward(self, x: Array) -> Array:
        value = self.value_proj.forward(x)
        gate = self.gate_proj.forward(x)
        gate_activation = silu(gate)
        self._last_value = value
        self._last_gate = gate
        self._last_gate_activation = gate_activation
        return self.out_proj.forward(value * gate_activation)

    def backward(self, grad_output: Array) -> Array:
        if self._last_value is None or self._last_gate is None or self._last_gate_activation is None:
            raise RuntimeError("GatedMLP.backward called before forward")
        grad_hidden = self.out_proj.backward(grad_output)
        grad_value = grad_hidden * self._last_gate_activation
        grad_gate = grad_hidden * self._last_value * silu_grad(self._last_gate)
        return self.value_proj.backward(grad_value) + self.gate_proj.backward(grad_gate)


class SignalProcessorBlock:
    """Minimal Sprint I2 signal processor block.

    Forward shape contract:
        ``z: FloatTensor[B,T,D]`` plus optional ``mask: BoolTensor[B,T]`` ->
        ``FloatTensor[B,T,D]``. The sequence of enabled modules is:
        norm -> local temporal mixer -> spectral mixer -> long-conv mixer ->
        gated MLP -> residual output.
    """

    def __init__(self, rng: np.random.Generator, config: SLWMCoreConfig, *, layer_index: int) -> None:
        self.config = config
        self.layer_index = int(layer_index)
        d_model = int(config.latent_dim)
        name = f"processor.blocks.{self.layer_index}"
        self.norm = LayerNorm(d_model, name=f"{name}.norm")
        self.local_temporal_mixer = (
            DepthwiseTemporalConv(rng, d_model, config.local_kernel_size, name=f"{name}.local_temporal_mixer")
            if config.use_local_temporal_mixer
            else None
        )
        self.spectral_mixer = (
            SpectralMixer(rng, d_model, config.context_length, config.n_spectral_modes, name=f"{name}.spectral_mixer")
            if config.use_spectral_mixer
            else None
        )
        self.long_conv = (
            DepthwiseTemporalConv(rng, d_model, config.long_conv_kernel_size, name=f"{name}.long_conv")
            if config.use_long_conv
            else None
        )
        self.gated_mlp = GatedMLP(rng, d_model, config.d_ff, name=f"{name}.gated_mlp") if config.use_gated_mlp else None
        self._last_mask: Array | None = None

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        params.extend(self.norm.parameters())
        for module in (self.local_temporal_mixer, self.spectral_mixer, self.long_conv, self.gated_mlp):
            if module is not None:
                params.extend(module.parameters())
        return params

    def forward(self, x: Array, mask: Array | None = None) -> Array:
        if x.ndim != 3 or x.shape[-1] != self.config.latent_dim:
            raise ValueError(f"SignalProcessorBlock expected [B,T,{self.config.latent_dim}], got {x.shape}")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != x.shape[:2]:
                raise ValueError(f"processor mask must have shape {x.shape[:2]}, got {mask.shape}")
            x_in = x * mask[:, :, None]
        else:
            x_in = x
        self._last_mask = mask

        h = self.norm.forward(x_in)
        if self.local_temporal_mixer is not None:
            h = h + self.local_temporal_mixer.forward(h)
        if self.spectral_mixer is not None:
            h = h + self.spectral_mixer.forward(h)
        if self.long_conv is not None:
            h = h + self.long_conv.forward(h)
        if self.gated_mlp is not None:
            h = h + self.gated_mlp.forward(h)
        y = x_in + h
        if mask is not None:
            y = y * mask[:, :, None]
        return y

    def backward(self, grad_output: Array) -> Array:
        if self._last_mask is not None:
            grad = np.asarray(grad_output, dtype=np.float64) * self._last_mask[:, :, None]
        else:
            grad = np.asarray(grad_output, dtype=np.float64)

        grad_residual = grad.copy()
        grad_h = grad.copy()
        if self.gated_mlp is not None:
            upstream = grad_h
            grad_h = upstream + self.gated_mlp.backward(upstream)
        if self.long_conv is not None:
            upstream = grad_h
            grad_h = upstream + self.long_conv.backward(upstream)
        if self.spectral_mixer is not None:
            upstream = grad_h
            grad_h = upstream + self.spectral_mixer.backward(upstream)
        if self.local_temporal_mixer is not None:
            upstream = grad_h
            grad_h = upstream + self.local_temporal_mixer.backward(upstream)
        grad_x = grad_residual + self.norm.backward(grad_h)
        if self._last_mask is not None:
            grad_x *= self._last_mask[:, :, None]
        return grad_x


__all__ = [
    "Array",
    "DepthwiseTemporalConv",
    "GatedMLP",
    "SignalProcessorBlock",
    "SpectralMixer",
    "silu",
    "silu_grad",
]
