"""SLWM-v4: Live Streaming Humanoid World Model (`160 ms` Causal Interleaved Core + Shared Generative Heads)."""

from models.v4.blocks import classify_v4_param_mup_role
from models.v4.heads import COMMIT_ACTIONS
from models.v4.model import (
    apply_v4_core_seq,
    compute_v4_training_loss,
    embed_v4_packet_step,
    embed_v4_packets_seq,
    init_v4_core_params,
    init_v4_core_state,
    init_v4_model_params,
    make_v4_mup_optimizer,
    run_v4_live_heartbeat_rollout,
    step_v4_core,
    step_v4_live_heartbeat,
    summarize_v4_accounting,
)
from models.v4.spec import (
    SLWMv4Spec,
    V4_S3_DISCOVERY_RUN_IDS,
    build_v4_s3_discovery_specs,
    compute_v4_core_streaming_state_bytes,
    count_state_tree_bytes,
    count_tree_params,
    estimate_v4_active_core_param_count,
    estimate_v4_core_param_count,
    make_slwm_v4_spec,
)

__all__ = [
    "COMMIT_ACTIONS",
    "SLWMv4Spec",
    "V4_S3_DISCOVERY_RUN_IDS",
    "apply_v4_core_seq",
    "build_v4_s3_discovery_specs",
    "classify_v4_param_mup_role",
    "compute_v4_core_streaming_state_bytes",
    "compute_v4_training_loss",
    "count_state_tree_bytes",
    "count_tree_params",
    "embed_v4_packet_step",
    "embed_v4_packets_seq",
    "estimate_v4_active_core_param_count",
    "estimate_v4_core_param_count",
    "init_v4_core_params",
    "init_v4_core_state",
    "init_v4_model_params",
    "make_slwm_v4_spec",
    "make_v4_mup_optimizer",
    "run_v4_live_heartbeat_rollout",
    "step_v4_core",
    "step_v4_live_heartbeat",
    "summarize_v4_accounting",
]
