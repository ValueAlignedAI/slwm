"""SLWM-v3: General-Purpose Content-Routed Multi-Timescale Selective Memory Architecture.

Pure-JAX, XLA/TPU v6e-1 native implementation strictly adhering to:
  - Zero hardcoded slot indices, reserved token-ID ranges, or task opcodes in the shared world core.
  - Exact online step-by-step == offline chunked scan equivalence via a shared single-step causal kernel.
  - Strictly bounded O(1) persistent streaming state (float32 memory banks + causal ring buffer).
  - Base-shape muP parameterization (input/hidden/readout scaling + CompleteP depth residual scaling).
  - Matched-parameter & matched-state-byte comparators and orthogonal single-factor causal ablations.
"""

from models.v3.spec import (
    BACKBONE_TYPES,
    REPRESENTATION_MODES,
    SLWMv3Spec,
    build_wave_a_comparator_specs,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)

__all__ = [
    "BACKBONE_TYPES",
    "REPRESENTATION_MODES",
    "SLWMv3Spec",
    "build_wave_a_comparator_specs",
    "compute_core_streaming_state_bytes",
    "count_tree_params",
    "make_slwm_v3_spec",
]
