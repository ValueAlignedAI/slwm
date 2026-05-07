"""Shared latent field contract for SLWM-124M Sprint I0."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from models.types import TensorSpec, ensure_latent, ensure_mask, make_latent_spec, make_mask_spec


class LatentSignalField:
    """Shape-only packer for the shared latent signal field.

    Input shape contract:
        adapter output packets contain ``z: FloatTensor[B,T,D]`` and
        ``mask: BoolTensor[B,T]``.

    Output shape contract:
        ``{"z": FloatTensor[B,T_context,D], "mask": BoolTensor[B,T_context],
        "metadata": dict}``.

    I0 does not concatenate or resample real tensors. It validates packet shapes
    and returns a canonical context-shaped ``TensorSpec``.
    """

    def __init__(self, latent_length: int = 1024, latent_dim: int = 768) -> None:
        self.latent_length = int(latent_length)
        self.latent_dim = int(latent_dim)

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


__all__ = ["LatentSignalField", "TensorSpec"]
