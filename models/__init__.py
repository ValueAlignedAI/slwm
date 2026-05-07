"""SLWM-124M Sprint I0 model skeleton.

The package contains shape-contract stubs only. Real model logic belongs to
later implementation sprints.
"""

from models.latent_field import LatentSignalField
from models.types import TensorSpec, ensure_latent, ensure_mask, make_latent_spec, make_mask_spec

__all__ = [
    "LatentSignalField",
    "TensorSpec",
    "ensure_latent",
    "ensure_mask",
    "make_latent_spec",
    "make_mask_spec",
]
