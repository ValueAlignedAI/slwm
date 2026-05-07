"""Base modality adapter shape contract."""

from __future__ import annotations

from typing import Any, Mapping

from data.contract import MODALITY_IDS
from models.module import ShapeModule
from models.types import ensure_latent, ensure_mask, make_latent_spec, make_mask_spec


class BaseModalityAdapter(ShapeModule):
    """Shape-only modality adapter.

    Forward input:
        ``sample`` dictionary with optional ``batch_size`` and optional existing
        ``z``/``mask`` shape carriers.

    Forward output shape contract:
        ``{"z": FloatTensor[B,T,D], "mask": BoolTensor[B,T], "metadata": dict}``.

    No tokenization, codec logic, convolution, or learned projection is
    implemented in I0.
    """

    modality: str = "unknown"

    def __init__(self, latent_length: int = 1024, latent_dim: int = 768, modality: str | None = None) -> None:
        self.latent_length = int(latent_length)
        self.latent_dim = int(latent_dim)
        if modality is not None:
            self.modality = modality
        if self.modality not in MODALITY_IDS:
            raise ValueError(f"Unknown modality {self.modality!r}")

    @property
    def modality_id(self) -> int:
        """Stable numeric modality ID for metadata."""

        return MODALITY_IDS[self.modality]

    def forward(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Return a canonical latent packet for this modality."""

        batch_size = int(sample.get("batch_size", 1))
        if "z" in sample:
            z = sample["z"]
            b, t, d = ensure_latent(z)
            if d != self.latent_dim:
                raise ValueError(f"Adapter expected D={self.latent_dim}; got D={d}")
            batch_size = b
        else:
            t = self.latent_length
            z = make_latent_spec(batch_size, t, self.latent_dim, name=f"z_{self.modality}")

        if "mask" in sample:
            mask = sample["mask"]
            ensure_mask(mask, (batch_size, t))
        else:
            mask = make_mask_spec(batch_size, t, name=f"mask_{self.modality}")

        metadata = dict(sample.get("metadata", {}))
        metadata.update(
            {
                "modality": self.modality,
                "modality_id": self.modality_id,
                "observed": True,
                "adapter": self.__class__.__name__,
                "implementation": "i0_shape_contract_stub",
            }
        )
        return {"z": z, "mask": mask, "metadata": metadata}
