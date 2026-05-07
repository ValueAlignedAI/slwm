"""Shared latent field packer for SLWM-124M.

The I0 ``TensorSpec`` contract remains supported. Sprint I2 adds real NumPy
packing by concatenating adapter packets into a fixed-length context field and
padding/truncating to preserve ``Z: FloatTensor[B,T,D]``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from models.types import TensorSpec, ensure_latent, ensure_mask, make_latent_spec, make_mask_spec


class LatentSignalField:
    """Packer for the shared latent signal field.

    Input shape contract:
        adapter output packets contain ``z: FloatTensor[B,T,D]`` and
        ``mask: BoolTensor[B,T]``.

    Output shape contract:
        ``{"z": FloatTensor[B,T_context,D], "mask": BoolTensor[B,T_context],
        "metadata": dict}``.

    TensorSpec packets use the I0 shape-only path. NumPy packets are concatenated
    in packet order, truncated/padded to ``T_context``, and cached so
    ``backward`` can split gradients back to adapter packet shapes.
    """

    def __init__(self, latent_length: int = 1024, latent_dim: int = 768) -> None:
        self.latent_length = int(latent_length)
        self.latent_dim = int(latent_dim)
        self._last_segments: list[dict[str, Any]] | None = None

    def from_adapter_outputs(self, packets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Validate adapter packets and create a shared-field shape contract."""

        if not packets:
            raise ValueError("At least one adapter packet is required")

        batch_size: int | None = None
        modalities: list[str] = []
        source_shapes: list[list[int]] = []

        for packet in packets:
            if "z" not in packet or "mask" not in packet or "metadata" not in packet:
                raise ValueError("Adapter packet must contain z, mask, and metadata")
            b, t, d = ensure_latent(packet["z"])
            ensure_mask(packet["mask"], (b, t))
            if d != self.latent_dim:
                raise ValueError(f"Adapter latent dim {d} does not match field dim {self.latent_dim}")
            if batch_size is None:
                batch_size = b
            elif b != batch_size:
                raise ValueError(f"All adapter packets must share B={batch_size}; got {b}")
            metadata = packet.get("metadata", {})
            modalities.append(str(metadata.get("modality", "unknown")))
            source_shapes.append([b, t, d])

        assert batch_size is not None

        if all(isinstance(packet["z"], TensorSpec) for packet in packets):
            self._last_segments = None
            return {
                "z": make_latent_spec(batch_size, self.latent_length, self.latent_dim, name="z_context"),
                "mask": make_mask_spec(batch_size, self.latent_length, name="context_mask"),
                "metadata": {
                    "field": "shared_latent_signal_field",
                    "packet_count": len(packets),
                    "modalities": modalities,
                    "source_shapes": source_shapes,
                    "implementation": "i0_shape_contract_stub",
                },
            }

        if any(isinstance(packet["z"], TensorSpec) for packet in packets):
            raise ValueError("Cannot mix TensorSpec and real tensor adapter packets")

        z_context = np.zeros((batch_size, self.latent_length, self.latent_dim), dtype=np.float64)
        context_mask = np.zeros((batch_size, self.latent_length), dtype=bool)
        segments: list[dict[str, Any]] = []
        cursor = 0
        for index, packet in enumerate(packets):
            z = np.asarray(packet["z"], dtype=np.float64)
            mask = np.asarray(packet["mask"], dtype=bool)
            b, source_length, d = z.shape
            metadata = packet.get("metadata", {})
            effective_length = max(0, min(source_length, int(metadata.get("copied_length", source_length))))
            remaining = max(0, self.latent_length - cursor)
            copied = min(effective_length, remaining)
            start = cursor
            end = cursor + copied
            if copied:
                z_context[:, start:end, :] = z[:, :copied, :] * mask[:, :copied, None]
                context_mask[:, start:end] = mask[:, :copied]
            segments.append(
                {
                    "packet_index": index,
                    "modality": packet.get("metadata", {}).get("modality", "unknown"),
                    "source_shape": (b, source_length, d),
                    "effective_length": effective_length,
                    "start": start,
                    "end": end,
                    "copied_length": copied,
                }
            )
            cursor = end

        self._last_segments = segments
        return {
            "z": z_context,
            "mask": context_mask,
            "metadata": {
                "field": "shared_latent_signal_field",
                "packet_count": len(packets),
                "modalities": modalities,
                "source_shapes": source_shapes,
                "segments": segments,
                "filled_length": int(cursor),
                "implementation": "i2_numpy_concat_pad_pack",
            },
        }

    def backward(self, grad_context: np.ndarray) -> list[np.ndarray]:
        """Split ``grad_context: FloatTensor[B,T_context,D]`` by packed packet.

        Returns:
            A list of gradients matching each adapter packet's original
            ``z: FloatTensor[B,T_source,D]`` shape. Positions truncated out of
            the context receive zero gradients.
        """

        if self._last_segments is None:
            raise RuntimeError("LatentSignalField.backward requires a prior real-tensor forward pass")
        grad = np.asarray(grad_context, dtype=np.float64)
        if grad.shape != (self._last_segments[0]["source_shape"][0], self.latent_length, self.latent_dim):
            raise ValueError(f"grad_context must have shape [B,{self.latent_length},{self.latent_dim}], got {grad.shape}")
        packet_grads: list[np.ndarray] = []
        for segment in self._last_segments:
            source_shape = segment["source_shape"]
            packet_grad = np.zeros(source_shape, dtype=np.float64)
            copied = int(segment["copied_length"])
            if copied:
                packet_grad[:, :copied, :] = grad[:, int(segment["start"]): int(segment["end"]), :]
            packet_grads.append(packet_grad)
        return packet_grads


__all__ = ["LatentSignalField", "TensorSpec"]
