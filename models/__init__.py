"""SLWM-124M model package.

I0 shape-contract stubs remain importable. Sprint I2 adds a tiny NumPy SLWM core
for adapters, latent packing, processor blocks, latent prediction, uncertainty,
and parameter accounting smoke tests.
"""

from models.latent_field import LatentSignalField
from models.slwm_config import SLWMCoreConfig
from models.slwm_core import NumpySLWMCore, make_i2_dummy_batch
from models.slwm_parameter_count import SLWMParameterBreakdown, slwm_parameter_breakdown_from_config
from models.types import TensorSpec, ensure_latent, ensure_mask, make_latent_spec, make_mask_spec

__all__ = [
    "LatentSignalField",
    "NumpySLWMCore",
    "SLWMCoreConfig",
    "SLWMParameterBreakdown",
    "TensorSpec",
    "ensure_latent",
    "ensure_mask",
    "make_i2_dummy_batch",
    "make_latent_spec",
    "make_mask_spec",
    "slwm_parameter_breakdown_from_config",
]
