"""SLWM-v4 Architecture Specification, Exact Streaming State Byte Counter & Single-Factor V4-S3 Discovery Factory.

Implements every pre-registered architecture, ablation, and baseline in `docs/research/slwm_v4_gate1_spec.md §5`:
  - Exact Dual-Track Accounting (`AGENTS.md §4.1`): core backbone params & streaming state bytes are matched
    across all compared backbones (`A0..A2`, `B0..B1`, `C1..C7`), while shared generative heads (`Flow-Video`,
    `Depth-Audio 16-RVQ`, `Unified 52/68 Motor`, `Text Byte`, `PolicyCommitGate`) are byte-for-byte identical.
  - Zero-`M_anchor` multi-timescale routing (`C5_routed_2bank`, `C6_routed_4bank`): `H_tot = 8` heads split
    across `B = 1 / 2 / 4` banks (`1x8`, `2x4`, `4x2`) keeps the matrix memory state strictly identical (`1.0x`)
    while using a lightweight `float32[B, d]` causal input-EMA novelty tracker (`~1.7%` extra bytes vs `2.0x` `M_anchor`)
    so ALL `A0..C7` configs execute full Chunkwise Parallel WY DeltaNet on TPU `v6e-1`.
  - Complete base-shape muP (`m_d = d_model / d_base`) and CompleteP `alpha = 1.0` depth residual scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from data.v4_heartbeat_contract import (
    SLWMv4HeartbeatSpec,
    UNIFIED_ACTION_DIM,
    UNIFIED_PROPRIO_DIM,
)

V4_BACKBONE_FAMILIES = (
    "hybrid_gdn_gqa",     # SLWM-v4 macro hybrid (3:1 / 5:1 / 1:1 Gated DeltaNet : SW-GQA+Sink)
    "sw_gqa_only",        # Pure equal-state-byte Sliding-Window GQA + Attention Sink (B1)
    "slwm_v3_dual_ref",   # Prior-generation SLWM-v3 per-layer dual-branch GQA + M_anchor routed memory (B0)
)

V4_VIDEO_HEAD_TYPES = ("flow", "huber")
V4_AUDIO_HEAD_TYPES = ("depth_transformer", "flat_rvq")

V4_S3_DISCOVERY_RUN_IDS = (
    "B0_v3_ref",
    "B1_sw_gqa",
    "A0_3to1_single_bank",
    "A1_5to1_single_bank",
    "A2_1to1_single_bank",
    "C1_no_conv1d",
    "C2_beta_0_1",
    "C3_no_ogate",
    "C4_rank128",
    "C5_routed_2bank",
    "C6_routed_4bank",
    "C7_ms_moe_ffn",
    "O1_jepa_sync",
    "H1_huber_video_abl",
    "H2_flat_rvq_abl",
    "H3_self_forcing",
    "H4_no_policy_gate",
)


@dataclass(frozen=True)
class SLWMv4Spec:
    """Immutable specification for SLWM-v4 cores, shared generative heads, and single-factor ablations."""

    name: str
    backbone_family: str = "hybrid_gdn_gqa"

    # Core sequence backbone dimensions
    d_model: int = 512
    d_base: int = 256
    n_layers: int = 16
    n_base_layers: int = 16
    residual_base_gain: float = 0.25
    ffn_mult: float = 2.6667
    ffn_dim_override: int | None = None

    # Macro hybrid schedule: 1 SW-GQA+Sink layer every `(gdn_per_gqa + 1)` layers (e.g. 3 -> 3:1, 5 -> 5:1, 1 -> 1:1)
    gdn_per_gqa: int = 3

    # Sliding-Window GQA + Learnable Attention Sink parameters
    window_size: int = 64
    n_sinks: int = 4
    n_attn_heads: int = 8
    n_kv_heads: int = 2
    attn_head_dim: int = 64

    # Gated DeltaNet / Zero-M_anchor Routed Multi-Timescale Matrix Memory parameters
    n_banks: int = 1
    n_mem_heads: int = 8  # Heads per bank; total heads H_tot = n_banks * n_mem_heads (8 -> 1x8, 2x4, 4x2 exact parity!)
    mem_key_dim: int = 64
    mem_val_dim: int = 64
    init_timescales_s: tuple[float, ...] = (64.0,)
    use_short_conv: bool = True
    conv_kernel_size: int = 4
    beta_max: float = 2.0  # Householder beta in (0, 2) by default; 1.0 for C2_beta_0_1
    use_output_gate: bool = True  # SiLU output gate o_gate; False for C3_no_ogate
    use_surprise_router: bool = False  # True for C5_routed_2bank, C6_routed_4bank, B0_v3_ref
    use_nowrite_option: bool = True
    use_m_anchor_legacy: bool = False  # True ONLY for B0_v3_ref (SLWM-v3 prior-generation reference)
    surprise_ema_alpha: float = 0.25  # Causal novelty EMA decay for zero-M_anchor surprise routing

    # FFN routing (C7_ms_moe_ffn)
    use_ms_moe_ffn: bool = False  # True -> 4 modality-specific SwiGLU experts (video/audio/text/motor) at matched active FLOPs

    # Shared Generative & Control Heads (identical across all compared backbones unless ablated in H1..H4 / O1)
    video_head_type: str = "flow"  # "flow" (Rectified Flow Matching) or "huber" (H1_huber_video_abl)
    video_flow_dim: int = 512
    video_flow_layers: int = 3
    video_flow_nfe: int = 6

    audio_head_type: str = "depth_transformer"  # "depth_transformer" (Moshi 16-RVQ) or "flat_rvq" (H2_flat_rvq_abl)
    audio_depth_dim: int = 384
    audio_depth_layers: int = 2
    audio_depth_heads: int = 6

    use_jepa_sync: bool = False  # O1_jepa_sync: causal windowed cross-modal AV/motor InfoNCE synchrony
    jepa_proj_dim: int = 256

    use_diffusion_forcing: bool = False  # H3_self_forcing: causal per-heartbeat video history noise
    use_self_forcing: bool = False  # H3_self_forcing: short closed-loop latent rollout consistency
    diffusion_forcing_max_sigma: float = 0.35

    use_policy_commit_gate: bool = True  # False ONLY for H4_no_policy_gate (AGENTS.md §9 mandatory ablation)
    policy_hidden_dim: int = 256

    # Physical 160 ms heartbeat edge interface (Gate 1 & Gate 2 certified contract)
    n_modalities: int = 4  # 0:video, 1:audio, 2:text_byte, 3:motor_proprio
    video_patch_dim: int = SLWMv4HeartbeatSpec().video_patch_dim  # 1024
    video_spatial_packets: int = SLWMv4HeartbeatSpec().video_patches_per_heartbeat  # 15
    audio_codebooks: int = SLWMv4HeartbeatSpec().audio_codebooks_stored  # 16
    audio_vocab_size: int = 2048
    text_vocab_size: int = 264
    action_dim: int = UNIFIED_ACTION_DIM  # 52
    proprio_dim: int = UNIFIED_PROPRIO_DIM  # 68

    # Precision & numerical hygiene
    state_dtype: str = "float32"
    ring_kv_bytes_per_elem: int = 2  # bf16 sliding-window KV cache
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.backbone_family not in V4_BACKBONE_FAMILIES:
            raise ValueError(f"Invalid backbone_family={self.backbone_family!r}")
        if self.video_head_type not in V4_VIDEO_HEAD_TYPES:
            raise ValueError(f"Invalid video_head_type={self.video_head_type!r}")
        if self.audio_head_type not in V4_AUDIO_HEAD_TYPES:
            raise ValueError(f"Invalid audio_head_type={self.audio_head_type!r}")
        if self.d_model % self.n_attn_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_attn_heads ({self.n_attn_heads})")
        if self.n_attn_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_attn_heads ({self.n_attn_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.n_banks < 1 or len(self.init_timescales_s) != self.n_banks:
            raise ValueError(f"len(init_timescales_s) must equal n_banks ({self.n_banks})")
        if self.audio_depth_dim % self.audio_depth_heads != 0:
            raise ValueError("audio_depth_dim must be divisible by audio_depth_heads")
        if self.conv_kernel_size < 2:
            raise ValueError("conv_kernel_size must be >= 2")

    @property
    def width_mult(self) -> float:
        return float(self.d_model) / float(self.d_base)

    @property
    def depth_residual_scale(self) -> float:
        return float(self.residual_base_gain) * (float(self.n_base_layers) / float(self.n_layers))

    @property
    def ffn_dim(self) -> int:
        if self.ffn_dim_override is not None:
            return int(self.ffn_dim_override)
        raw = int(round(self.d_model * self.ffn_mult))
        return max(64, ((raw + 63) // 64) * 64)

    @property
    def total_mem_heads(self) -> int:
        return self.n_banks * self.n_mem_heads

    def layer_types(self) -> tuple[str, ...]:
        """Return exact per-layer sequence mixer schedule (`gdn`, `gqa`, or `v3_dual`)."""
        if self.backbone_family == "sw_gqa_only":
            return tuple("gqa" for _ in range(self.n_layers))
        if self.backbone_family == "slwm_v3_dual_ref":
            return tuple("v3_dual" for _ in range(self.n_layers))
        period = max(1, int(self.gdn_per_gqa) + 1)
        return tuple("gqa" if ((i + 1) % period == 0) else "gdn" for i in range(self.n_layers))


def count_tree_params(params: Any) -> int:
    """Count exact scalar parameters in any JAX/NumPy pytree."""
    if hasattr(params, "size") and hasattr(params, "shape"):
        return int(params.size)
    if isinstance(params, dict):
        return sum(count_tree_params(v) for v in params.values())
    if isinstance(params, (list, tuple)):
        return sum(count_tree_params(v) for v in params)
    return 0


def count_state_tree_bytes(state: Any) -> int:
    """Count exact byte footprint of any allocated JAX/NumPy recurrent/KV state pytree."""
    if hasattr(state, "nbytes"):
        return int(state.nbytes)
    if hasattr(state, "size") and hasattr(state, "dtype"):
        return int(state.size) * int(state.dtype.itemsize)
    if isinstance(state, dict):
        return sum(count_state_tree_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(count_state_tree_bytes(v) for v in state)
    return 0


def compute_v4_core_streaming_state_bytes(spec: SLWMv4Spec, batch_size: int = 1) -> dict[str, int]:
    """Compute exact persistent streaming state bytes matching allocated JAX state arrays byte-for-byte."""
    mem_b = 4 if spec.state_dtype == "float32" else 2
    kv_b = int(spec.ring_kv_bytes_per_elem)
    ltypes = spec.layer_types()

    mem_bytes = 0
    conv_bytes = 0
    ring_kv_bytes = 0
    book_bytes = 0

    h_tot = spec.total_mem_heads
    qkv_dim = h_tot * (2 * spec.mem_key_dim + spec.mem_val_dim)

    for lt in ltypes:
        if lt in ("gdn", "v3_dual"):
            # Matrix memory M: float32[B, n_banks, n_mem_heads, d_v, d_k]
            m_layer = spec.n_banks * spec.n_mem_heads * spec.mem_val_dim * spec.mem_key_dim * mem_b
            mem_bytes += m_layer
            if lt == "v3_dual" and spec.use_m_anchor_legacy:
                # SLWM-v3 legacy M_anchor doubles matrix memory state + cum_decay + int32 phase
                mem_bytes += m_layer + spec.n_banks * spec.n_mem_heads * 4
                book_bytes += 4
            if spec.use_surprise_router and not spec.use_m_anchor_legacy:
                # Zero-M_anchor causal novelty EMA state: float32[B, d_model] + bool has_ema[B]
                mem_bytes += spec.d_model * 4
                book_bytes += 1
            if spec.use_short_conv and lt == "gdn":
                # Depthwise causal conv1d ring buffer: float32[B, k-1, qkv_dim]
                conv_bytes += (spec.conv_kernel_size - 1) * qkv_dim * 4
        if lt in ("gqa", "v3_dual"):
            # Sliding-window GQA ring buffer: K+V [B, W, H_kv, d_h] (bf16) + bool valid [B, W] + int32 step [B]
            ring_kv_bytes += 2 * spec.window_size * spec.n_kv_heads * spec.attn_head_dim * kv_b
            book_bytes += spec.window_size * 1 + 4

    mem_bytes *= batch_size
    conv_bytes *= batch_size
    ring_kv_bytes *= batch_size
    book_bytes *= batch_size
    total = mem_bytes + conv_bytes + ring_kv_bytes + book_bytes
    return {
        "matrix_memory_bytes": int(mem_bytes),
        "short_conv_bytes": int(conv_bytes),
        "local_ring_kv_bytes": int(ring_kv_bytes),
        "bookkeeping_bytes": int(book_bytes),
        "total_core_state_bytes": int(total),
    }


def estimate_v4_core_param_count(spec: SLWMv4Spec) -> int:
    """Exact analytic trainable parameter count for the SLWM-v4 core backbone (excluding shared edge & heads)."""
    d = spec.d_model
    ffn = spec.ffn_dim
    h_tot = spec.total_mem_heads
    dk, dv = spec.mem_key_dim, spec.mem_val_dim
    hq = spec.n_attn_heads * spec.attn_head_dim
    hkv = spec.n_kv_heads * spec.attn_head_dim
    qkv_dim = h_tot * (2 * dk + dv)

    total = d  # final_norm_w
    for lt in spec.layer_types():
        # FFN sublayer (dense SwiGLU or 4-expert MS-MoE)
        n_exp = spec.n_modalities if spec.use_ms_moe_ffn else 1
        total += d + n_exp * (3 * d * ffn)

        if lt in ("gdn", "v3_dual"):
            # Pre-norm + q, k, v, o + per-head o_norm_w + decay gate w_g, g_bias + log_tau
            total += d + d * h_tot * dk + d * h_tot * dk + d * h_tot * dv + (h_tot * dv) * d
            total += h_tot * dv + d * h_tot + h_tot + spec.n_banks
            if spec.use_short_conv and lt == "gdn":
                total += qkv_dim * spec.conv_kernel_size + qkv_dim
            if spec.use_output_gate and lt == "gdn":
                total += d * (h_tot * dv)
            if not spec.use_surprise_router:
                total += d * h_tot + h_tot
            else:
                mod_d = 16
                meta_in = mod_d + 3
                meta_hid = 32
                router_hid = max(32, d // 4)
                n_out = h_tot + (spec.n_banks if spec.use_nowrite_option else 0)
                total += spec.n_modalities * mod_d
                total += d * n_out + d * router_hid + router_hid + router_hid * n_out + n_out
                total += meta_in * n_out + meta_in * meta_hid + meta_hid + meta_hid * n_out
                if spec.use_m_anchor_legacy:
                    total += (h_tot * dv) * d  # w_prior_proj in SLWM-v3 ref
        if lt in ("gqa", "v3_dual"):
            # Pre-norm + w_q, w_k, w_v, w_o + rel_pos_bias + learnable sink KV (gqa only)
            total += d + d * hq + d * hkv + d * hkv + hq * d + spec.n_attn_heads * spec.window_size
            if lt == "gqa" and spec.n_sinks > 0:
                total += 2 * spec.n_sinks * spec.n_kv_heads * spec.attn_head_dim
    return int(total)


def estimate_v4_active_core_param_count(spec: SLWMv4Spec) -> int:
    """Exact analytic ACTIVE trainable core parameters per packet (equals total except for MS-MoE `C7`)."""
    if not spec.use_ms_moe_ffn:
        return estimate_v4_core_param_count(spec)
    return estimate_v4_core_param_count(replace(spec, use_ms_moe_ffn=False))


def _solve_ffn_dim_for_target_core_params(spec: SLWMv4Spec, target_core_params: int) -> SLWMv4Spec:
    """Solve exact multiple-of-16 `ffn_dim_override` so core trainable params match `target_core_params` within <= 0.6%."""
    base_zero = replace(spec, ffn_dim_override=0)
    fixed_p = estimate_v4_core_param_count(base_zero)
    # Note: for MS-MoE (`C7`), match ACTIVE params per token (identical `ffn_dim` as `A0`), NOT total inactive expert params!
    if spec.use_ms_moe_ffn:
        return spec
    denom = 3 * spec.d_model * spec.n_layers
    raw_ffn = max(64.0, (target_core_params - fixed_p) / float(denom))
    best_ffn = max(64, int(round(raw_ffn / 16.0)) * 16)
    return replace(spec, ffn_dim_override=best_ffn)


def make_slwm_v4_spec(
    profile: str = "discovery_45m",
    *,
    run_id: str = "A0_3to1_single_bank",
) -> SLWMv4Spec:
    """Build any named SLWM-v4 scale profile (`tiny_unit`, `discovery_45m`, `confirmation_150m`, `scale_700m`, `scale_1p3b`)."""
    if profile == "tiny_unit":
        anchor = SLWMv4Spec(
            name=f"{profile}_{run_id}",
            d_model=128,
            d_base=128,
            n_layers=4,
            n_base_layers=4,
            gdn_per_gqa=3,
            window_size=8,
            n_sinks=2,
            n_attn_heads=4,
            n_kv_heads=2,
            attn_head_dim=32,
            n_banks=1,
            n_mem_heads=4,
            mem_key_dim=32,
            mem_val_dim=32,
            init_timescales_s=(64.0,),
            video_flow_dim=128,
            video_flow_layers=2,
            video_flow_nfe=4,
            audio_depth_dim=96,
            audio_depth_layers=2,
            audio_depth_heads=4,
            jepa_proj_dim=64,
            policy_hidden_dim=64,
        )
    elif profile == "discovery_45m":
        # ~45.3M trainable core (`d=512, L=16`, 3:1 hybrid: 12 GDN + 4 SW-GQA+Sink; 1.84 MiB core state @ B=1)
        anchor = SLWMv4Spec(
            name=f"{profile}_{run_id}",
            d_model=512,
            d_base=256,
            n_layers=16,
            n_base_layers=16,
            ffn_dim_override=1088,
            gdn_per_gqa=3,
            window_size=64,
            n_sinks=4,
            n_attn_heads=8,
            n_kv_heads=2,
            attn_head_dim=64,
            n_banks=1,
            n_mem_heads=8,
            mem_key_dim=64,
            mem_val_dim=64,
            init_timescales_s=(64.0,),
            video_flow_dim=512,
            video_flow_layers=3,
            video_flow_nfe=6,
            audio_depth_dim=384,
            audio_depth_layers=2,
            audio_depth_heads=6,
            jepa_proj_dim=256,
            policy_hidden_dim=256,
        )
    elif profile == "confirmation_150m":
        # ~150.8M trainable core (`d=896, L=20`, 3:1 hybrid: 15 GDN + 5 SW-GQA+Sink)
        anchor = SLWMv4Spec(
            name=f"{profile}_{run_id}",
            d_model=896,
            d_base=256,
            n_layers=20,
            n_base_layers=16,
            ffn_dim_override=1664,
            gdn_per_gqa=3,
            window_size=128,
            n_sinks=4,
            n_attn_heads=14,
            n_kv_heads=2,
            attn_head_dim=64,
            n_banks=1,
            n_mem_heads=12,
            mem_key_dim=64,
            mem_val_dim=64,
            init_timescales_s=(64.0,),
            video_flow_dim=768,
            video_flow_layers=4,
            video_flow_nfe=6,
            audio_depth_dim=512,
            audio_depth_layers=3,
            audio_depth_heads=8,
            jepa_proj_dim=384,
            policy_hidden_dim=256,
        )
    elif profile == "scale_700m":
        anchor = SLWMv4Spec(
            name=f"{profile}_{run_id}",
            d_model=1536,
            d_base=256,
            n_layers=24,
            n_base_layers=16,
            ffn_dim_override=4096,
            gdn_per_gqa=3,
            window_size=128,
            n_sinks=4,
            n_attn_heads=24,
            n_kv_heads=4,
            attn_head_dim=64,
            n_banks=1,
            n_mem_heads=16,
            mem_key_dim=96,
            mem_val_dim=96,
            init_timescales_s=(64.0,),
            video_flow_dim=1024,
            video_flow_layers=4,
            video_flow_nfe=6,
            audio_depth_dim=640,
            audio_depth_layers=4,
            audio_depth_heads=8,
            jepa_proj_dim=512,
            policy_hidden_dim=384,
        )
    elif profile == "scale_1p3b":
        anchor = SLWMv4Spec(
            name=f"{profile}_{run_id}",
            d_model=2048,
            d_base=256,
            n_layers=28,
            n_base_layers=16,
            ffn_dim_override=4608,
            gdn_per_gqa=3,
            window_size=128,
            n_sinks=4,
            n_attn_heads=32,
            n_kv_heads=4,
            attn_head_dim=64,
            n_banks=1,
            n_mem_heads=16,
            mem_key_dim=128,
            mem_val_dim=128,
            init_timescales_s=(64.0,),
            video_flow_dim=1280,
            video_flow_layers=4,
            video_flow_nfe=6,
            audio_depth_dim=768,
            audio_depth_layers=4,
            audio_depth_heads=8,
            jepa_proj_dim=512,
            policy_hidden_dim=384,
        )
    else:
        raise ValueError(f"Unknown SLWM-v4 profile={profile!r}")

    target_core_p = estimate_v4_core_param_count(anchor)
    target_state_b = compute_v4_core_streaming_state_bytes(anchor, batch_size=1)["total_core_state_bytes"]

    # Apply EXACT single-factor modification for `run_id` relative to `A0_3to1_single_bank`!
    if run_id == "A0_3to1_single_bank":
        return anchor

    if run_id == "A1_5to1_single_bank":
        s = replace(anchor, gdn_per_gqa=5)
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "A2_1to1_single_bank":
        s = replace(anchor, gdn_per_gqa=1)
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "B0_v3_ref":
        # Prior-generation SLWM-v3: dual-branch per-layer (local GQA + M_anchor routed 4-bank memory), no conv1d, beta in (0,1), no o_gate
        nb = 4
        nh = max(1, anchor.n_mem_heads // (2 * nb))
        dk_v3 = anchor.mem_key_dim
        taus_v3 = (1.0, 8.0, 64.0, 512.0)
        s = replace(
            anchor,
            backbone_family="slwm_v3_dual_ref",
            window_size=max(8, anchor.window_size // 2),
            n_sinks=0,
            n_banks=nb,
            n_mem_heads=nh,
            mem_key_dim=dk_v3,
            mem_val_dim=dk_v3,
            init_timescales_s=taus_v3,
            use_short_conv=False,
            beta_max=1.0,
            use_output_gate=False,
            use_surprise_router=True,
            use_m_anchor_legacy=True,
        )
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "B1_sw_gqa":
        # Pure equal-state-byte SW-GQA + Attention Sink: solve exact window_size matching `A0`'s streaming state bytes!
        per_slot_bytes = anchor.n_layers * (2 * anchor.n_kv_heads * anchor.attn_head_dim * anchor.ring_kv_bytes_per_elem + 1)
        fixed_book = anchor.n_layers * 4
        w_eq = max(8, int(round((target_state_b - fixed_book) / float(per_slot_bytes))))
        s = replace(anchor, backbone_family="sw_gqa_only", window_size=w_eq)
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "C1_no_conv1d":
        return replace(anchor, use_short_conv=False)

    if run_id == "C2_beta_0_1":
        return replace(anchor, beta_max=1.0)

    if run_id == "C3_no_ogate":
        s = replace(anchor, use_output_gate=False)
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "C4_rank128":
        # Double per-head rank (64->128) while halving head count (8->4) so inner dim & core params match A0!
        nh_half = max(1, anchor.n_mem_heads // 2)
        s = replace(anchor, n_mem_heads=nh_half, mem_key_dim=anchor.mem_key_dim * 2, mem_val_dim=anchor.mem_val_dim * 2)
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "C5_routed_2bank":
        # Zero-M_anchor pooled-surprise B=2 routed SLWM (exact 1.0x matrix state bytes!)
        nh2 = max(1, anchor.n_mem_heads // 2)
        s = replace(
            anchor,
            n_banks=2,
            n_mem_heads=nh2,
            init_timescales_s=(4.0, 128.0),
            use_surprise_router=True,
            use_m_anchor_legacy=False,
        )
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "C6_routed_4bank":
        # Zero-M_anchor pooled-surprise B=4 routed SLWM (exact 1.0x matrix state bytes!)
        nh4 = max(1, anchor.n_mem_heads // 4)
        s = replace(
            anchor,
            n_banks=4,
            n_mem_heads=nh4,
            init_timescales_s=(1.0, 8.0, 64.0, 512.0),
            use_surprise_router=True,
            use_m_anchor_legacy=False,
        )
        return _solve_ffn_dim_for_target_core_params(s, target_core_p)

    if run_id == "C7_ms_moe_ffn":
        # Modality-Specific MS-MoE FFN (matched active FLOPs/params per token, identical ffn_dim!)
        return replace(anchor, use_ms_moe_ffn=True)

    if run_id == "O1_jepa_sync":
        return replace(anchor, use_jepa_sync=True)

    if run_id == "H1_huber_video_abl":
        return replace(anchor, video_head_type="huber")

    if run_id == "H2_flat_rvq_abl":
        return replace(anchor, audio_head_type="flat_rvq")

    if run_id == "H3_self_forcing":
        return replace(anchor, use_diffusion_forcing=True, use_self_forcing=True)

    if run_id == "H4_no_policy_gate":
        return replace(anchor, use_policy_commit_gate=False)

    raise ValueError(f"Unknown SLWM-v4 run_id={run_id!r}; expected one of {V4_S3_DISCOVERY_RUN_IDS}")


def build_v4_s3_discovery_specs(profile: str = "discovery_45m") -> dict[str, SLWMv4Spec]:
    """Return all 17 pre-registered single-factor `V4-S3` specs indexed by canonical `run_id`."""
    return {rid: make_slwm_v4_spec(profile=profile, run_id=rid) for rid in V4_S3_DISCOVERY_RUN_IDS}
