"""SLWM-v3 Model Specifications, Dual-Track State Accounting & Exact Matched Comparator Factory (Rev 2).

Provides:
  - `SLWMv3Spec`: typed immutable configuration covering Discovery (`~40.5M`, `L=12, W=32`),
    Confirmation (`~130M`, `L=16, W=64`), Pilot (`d=256, L=12, W=32`), and CPU Unit profiles.
  - Base-shape muP (`m_d = d_model / d_base`, exact `1.0` LR mult at `d == d_base`) and
    CompleteP `alpha = 1.0` depth residual scaling (`L_base / L`, `arXiv:2505.01618`).
  - Exact persistent streaming state byte calculation (`compute_core_streaming_state_bytes`)
    verified byte-for-byte against actual allocated JAX state arrays across all backbones.
  - `build_wave_a_comparator_specs`: constructs matched-parameter (within `±1.0%` via exact
    `ffn_dim` solving!) AND matched-state-byte specs for every Wave A comparator and ablation:
      1. `slwm_v3_routed`            (full model: content routing + multi-timescale + no-write + zero-lag bypass)
      2. `slwm_v3_no_routing`        (ablation: holds `tau_b`, `g_b`, and `read_w` fixed; replaces content-routed bank write allocation with content-independent round-robin one-hot allocation at exact equal total write budget `1 - r_nowrite`)
      3. `slwm_v3_no_nowrite`        (ablation: explicit `r_nowrite` channel removed)
      4. `slwm_v3_no_zerolag`        (ablation: extra zero-lag projection bypass removed)
      5. `single_bank_gated_deltanet` (faithful `B=1` Gated DeltaNet: NO surprise router, 2x `(d_k, d_v)` for equal bytes)
      6. `sliding_window_gqa_eqparam` (GQA-only matched in trainable params within `±1.0%`)
      7. `sliding_window_gqa_eqbyte`  (GQA-only matched in total streaming state bytes & trainable params)
      8. `diag_ssm`                  (diagonal scalar-decay SSM comparator matched in bytes & params)
      9. `explicit_bounded_kv`       (differentiable multi-head bounded LRU/overwrite KV controller matched in bytes & params)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

MATRIX_MEMORY_BACKBONES = (
    "slwm_v3_routed",
    "slwm_v3_tied_tau",
    "slwm_v3_no_routing",
    "slwm_v3_no_nowrite",
    "slwm_v3_forced_write",
    "slwm_v3_no_zerolag",
    "single_bank_spread_tau",
    "single_bank_gated_deltanet",
)

BACKBONE_TYPES = MATRIX_MEMORY_BACKBONES + (
    "sliding_window_gqa",
    "diag_ssm",
    "explicit_bounded_kv",
)

REPRESENTATION_MODES = (
    "frozen_codec_edge",
    "scratch_edge",
)


@dataclass(frozen=True)
class SLWMv3Spec:
    """Immutable architecture specification for SLWM-v3 and matched Wave A comparators.

    Zero task-specific IDs, vocabulary ranges, or hardcoded slot opcodes exist in this specification.
    """

    name: str
    backbone_type: str = "slwm_v3_routed"
    representation_mode: str = "frozen_codec_edge"

    # Core backbone dimensions
    d_model: int = 512
    d_base: int = 256  # Base width for base-shape muP multiplier m_d = d_model / d_base
    n_layers: int = 12
    n_base_layers: int = 12  # Base depth for CompleteP alpha=1.0 residual scaling (residual_base_gain * L_base / L)
    residual_base_gain: float = 0.25  # Normalizes 3-sublayer block gain to O(1) at L_base (3.0 / L for L_base=12)
    ffn_mult: float = 2.6667
    ffn_dim_override: int | None = None  # Exact multiple-of-64 override for ±1.0% comparator param parity

    # Short-window O(1) causal local GQA ring buffer
    window_size: int = 32
    n_attn_heads: int = 8
    n_kv_heads: int = 2
    attn_head_dim: int = 64

    # Head-routed multi-timescale matrix memory (or faithful single-bank Gated DeltaNet at identical rank & bytes)
    n_banks: int = 4
    n_mem_heads: int = 1  # Heads per timescale bank (total heads H_tot = n_banks * n_mem_heads = 4)
    mem_key_dim: int = 32
    mem_val_dim: int = 32
    mem_base_dim: int = 128  # Base memory readout fan_in (4 * 32) for exact muP fan_in scaling at confirmation/1B
    init_timescales_s: tuple[float, ...] = (1.0, 8.0, 64.0, 512.0)

    # Multi-head differentiable explicit bounded LRU/overwrite KV controller
    explicit_kv_heads: int = 4
    explicit_kv_slots: int = 16
    explicit_kv_dim: int = 32

    # Diagonal SSM state dimension per channel
    ssm_state_dim: int = 8

    # Orthogonal single-factor ablation toggles
    use_learned_routing: bool = True
    use_nowrite_option: bool = True
    use_zero_lag_residual: bool = True
    tie_bank_timescales: bool = False
    use_surprise_router: bool = True  # False for faithful single-bank Gated DeltaNet!
    force_write_floor: float = 0.0  # >0.0 for preregistered forced-write control (eta >= floor on valid ticks)

    # Edge packet interface dimensions (identical across ALL compared backbones!)
    n_modalities: int = 5  # 0:video, 1:audio, 2:text_byte, 3:proprio_action, 4:signal_probe
    video_patch_dim: int = 1024  # Wan 2.1 VAE 8x8x16 isotropic patch (or matched scratch patch)
    video_spatial_packets: int = 15  # 192x320 -> 24x40x16 -> 3x5=15 isotropic 8x8 packets (0% black bars)
    audio_codebooks: int = 32  # Kyutai Mimi 32-RVQ codebooks (12.5 Hz)
    audio_vocab_size: int = 2048  # Mimi codebook cardinality
    audio_cont_dim: int = 64  # Continuous audio / scratch log-mel feature dim
    text_vocab_size: int = 264  # Lossless ByT5 256 UTF-8 bytes + 8 control symbols
    proprio_dim: int = 16
    action_dim: int = 14
    probe_dim: int = 32
    target_feat_dim: int = 64
    pred_horizons: tuple[int, ...] = (1, 4, 8)

    # Precision & numerical stability
    state_dtype: str = "float32"  # Recurrent/KV memory state ALWAYS fp32
    ring_kv_bytes_per_elem: int = 2  # bf16 local ring-buffer KV cache
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.backbone_type not in BACKBONE_TYPES:
            raise ValueError(f"Unsupported backbone_type={self.backbone_type!r}; expected one of {BACKBONE_TYPES}")
        if self.representation_mode not in REPRESENTATION_MODES:
            raise ValueError(
                f"Unsupported representation_mode={self.representation_mode!r}; expected one of {REPRESENTATION_MODES}"
            )
        if self.d_model % self.n_attn_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_attn_heads ({self.n_attn_heads})")
        if self.n_attn_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_attn_heads ({self.n_attn_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.n_banks < 1:
            raise ValueError("n_banks must be >= 1")
        if len(self.init_timescales_s) != self.n_banks:
            raise ValueError(
                f"len(init_timescales_s) ({len(self.init_timescales_s)}) must equal n_banks ({self.n_banks})"
            )

    @property
    def width_mult(self) -> float:
        """Base-shape muP width multiplier m_d = d_model / d_base (equals 1.0 at d_model == d_base)."""
        return float(self.d_model) / float(self.d_base)

    @property
    def depth_residual_scale(self) -> float:
        """CompleteP alpha=1.0 normalized depth residual branch multiplier `residual_base_gain * L_base / L`."""
        return float(self.residual_base_gain) * (float(self.n_base_layers) / float(self.n_layers))

    @property
    def ffn_dim(self) -> int:
        if self.ffn_dim_override is not None:
            return int(self.ffn_dim_override)
        raw = int(round(self.d_model * self.ffn_mult))
        return max(64, ((raw + 63) // 64) * 64)

    @property
    def stacked_window_receptive_field_ticks(self) -> int:
        """Exact maximum causal receptive field L*(W-1) of an L-layer window-W local attention stack."""
        return self.n_layers * (self.window_size - 1)


def compute_core_streaming_state_bytes(spec: SLWMv3Spec, batch_size: int = 1) -> dict[str, int]:
    """Compute exact persistent streaming state bytes matching allocated JAX state arrays byte-for-byte."""
    mem_elem_bytes = 4 if spec.state_dtype == "float32" else 2
    kv_elem_bytes = int(spec.ring_kv_bytes_per_elem)

    # Local sliding-window GQA ring buffer per layer: K + V [B, W, H_kv, d_head] + bool valid [B, W] + int32 step [B]
    ring_kv_per_layer = 2 * spec.window_size * spec.n_kv_heads * spec.attn_head_dim * kv_elem_bytes
    book_per_layer = spec.window_size * 1 + 4

    if spec.backbone_type in MATRIX_MEMORY_BACKBONES:
        mem_per_layer = spec.n_banks * spec.n_mem_heads * spec.mem_val_dim * spec.mem_key_dim * mem_elem_bytes
    elif spec.backbone_type == "sliding_window_gqa":
        mem_per_layer = 0
    elif spec.backbone_type == "diag_ssm":
        mem_per_layer = spec.d_model * spec.ssm_state_dim * mem_elem_bytes
    elif spec.backbone_type == "explicit_bounded_kv":
        # Multi-head KV table: keys [B, H_kv, K_slots, D_kv] + vals [B, H_kv, K_slots, D_kv] + age [B, H_kv, K_slots] (fp32)
        # + valid [B, H_kv, K_slots] (bool)
        h_e = spec.explicit_kv_heads
        k_s = spec.explicit_kv_slots
        d_e = spec.explicit_kv_dim
        mem_per_layer = h_e * (2 * k_s * d_e + k_s) * mem_elem_bytes
        book_per_layer += h_e * k_s * 1
    else:
        raise ValueError(f"Unknown backbone_type={spec.backbone_type}")

    mem_total = batch_size * spec.n_layers * mem_per_layer
    ring_total = batch_size * spec.n_layers * ring_kv_per_layer
    book_total = batch_size * spec.n_layers * book_per_layer
    return {
        "memory_state_bytes": int(mem_total),
        "local_ring_kv_bytes": int(ring_total),
        "bookkeeping_bytes": int(book_total),
        "total_core_state_bytes": int(mem_total + ring_total + book_total),
    }


def count_tree_params(params: Any) -> int:
    """Count exact scalar parameters in any JAX pytree."""
    if hasattr(params, "size") and hasattr(params, "shape"):
        return int(params.size)
    if isinstance(params, dict):
        return sum(count_tree_params(v) for v in params.values())
    if isinstance(params, (list, tuple)):
        return sum(count_tree_params(v) for v in params)
    return 0


def make_slwm_v3_spec(
    profile: str = "discovery_45m",
    *,
    backbone_type: str = "slwm_v3_routed",
    representation_mode: str = "frozen_codec_edge",
    d_model_override: int | None = None,
    n_layers_override: int | None = None,
) -> SLWMv3Spec:
    """Build a named SLWM-v3 scale profile (`tiny_unit`, `pilot_d256`, `discovery_45m`, `confirmation_135m`)."""
    if profile == "tiny_unit":
        base = SLWMv3Spec(
            name=f"tiny_unit_{backbone_type}",
            backbone_type=backbone_type,
            representation_mode=representation_mode,
            d_model=128,
            d_base=128,
            n_layers=4,
            n_base_layers=4,
            window_size=8,
            n_attn_heads=4,
            n_kv_heads=2,
            attn_head_dim=32,
            n_banks=4,
            n_mem_heads=1,
            mem_key_dim=32,
            mem_val_dim=32,
            init_timescales_s=(1.0, 8.0, 64.0, 512.0),
            explicit_kv_heads=4,
            explicit_kv_slots=16,
            explicit_kv_dim=32,
            ssm_state_dim=32,
        )
    elif profile == "pilot_d256":
        # Compact TPU pilot profile (exact base width d=256, L=12, W=32 -> RF=372 ticks)
        base = SLWMv3Spec(
            name=f"pilot_d256_{backbone_type}",
            backbone_type=backbone_type,
            representation_mode=representation_mode,
            d_model=256,
            d_base=256,
            n_layers=12,
            n_base_layers=12,
            window_size=32,
            n_attn_heads=4,
            n_kv_heads=2,
            attn_head_dim=64,
            n_banks=4,
            n_mem_heads=1,
            mem_key_dim=32,
            mem_val_dim=32,
            init_timescales_s=(1.0, 8.0, 64.0, 512.0),
            explicit_kv_heads=4,
            explicit_kv_slots=16,
            explicit_kv_dim=32,
            ssm_state_dim=16,
        )
    elif profile == "discovery_45m":
        # ~40.5M trainable core (L=12, W=32 -> RF=372; 16 KiB mem + 16 KiB ring KV/layer -> W_eq=64)
        base = SLWMv3Spec(
            name=f"discovery_45m_{backbone_type}",
            backbone_type=backbone_type,
            representation_mode=representation_mode,
            d_model=512,
            d_base=256,
            n_layers=12,
            n_base_layers=12,
            window_size=32,
            n_attn_heads=8,
            n_kv_heads=2,
            attn_head_dim=64,
            n_banks=4,
            n_mem_heads=1,
            mem_key_dim=32,
            mem_val_dim=32,
            init_timescales_s=(1.0, 8.0, 64.0, 512.0),
            explicit_kv_heads=4,
            explicit_kv_slots=16,
            explicit_kv_dim=32,
            ssm_state_dim=8,
        )
    elif profile == "confirmation_135m":
        # ~130M trainable core (SLWM-124M-150M anchor; L=16, W=64 -> RF=1008; 64 KiB mem + 64 KiB ring KV -> W_eq=128)
        base = SLWMv3Spec(
            name=f"confirmation_135m_{backbone_type}",
            backbone_type=backbone_type,
            representation_mode=representation_mode,
            d_model=768,
            d_base=256,
            n_layers=16,
            n_base_layers=12,
            ffn_mult=3.1667,
            window_size=64,
            n_attn_heads=12,
            n_kv_heads=4,
            attn_head_dim=64,
            n_banks=4,
            n_mem_heads=1,
            mem_key_dim=64,
            mem_val_dim=64,
            init_timescales_s=(1.0, 8.0, 64.0, 512.0),
            explicit_kv_heads=4,
            explicit_kv_slots=32,
            explicit_kv_dim=64,
            ssm_state_dim=21,
        )
    else:
        raise ValueError(f"Unknown profile={profile!r}")

    if d_model_override is not None:
        n_heads = max(2, d_model_override // base.attn_head_dim)
        n_kv = max(1, min(base.n_kv_heads, n_heads))
        while n_heads % n_kv != 0:
            n_kv -= 1
        base = replace(base, d_model=d_model_override, n_attn_heads=n_heads, n_kv_heads=n_kv)

    if n_layers_override is not None:
        base = replace(base, n_layers=int(n_layers_override))

    # Apply orthogonal single-factor toggle defaults
    if backbone_type == "slwm_v3_tied_tau":
        # 2x2 factorial cell: IDENTICAL surprise router + no-write + FFN width as `slwm_v3_routed`,
        # changing ONLY timescale prior from spread `(1, 8, 64, 512)s` to tied `64s`!
        base = replace(base, tie_bank_timescales=True, init_timescales_s=(64.0,) * base.n_banks)
    elif backbone_type == "slwm_v3_no_routing":
        # Strict single-factor isolation of content-to-timescale bank routing: holds physical timescales `tau_b`,
        # bank decay gates `g_b`, per-head (dk, dv) rank, and aggregate write/no-write mass IDENTICAL to `slwm_v3_routed`,
        # replacing ONLY per-bank content routing with uniform bank-averaged write allocation!
        base = replace(base, use_learned_routing=False, tie_bank_timescales=False)
    elif backbone_type == "slwm_v3_no_nowrite":
        base = replace(base, use_nowrite_option=False)
    elif backbone_type == "slwm_v3_forced_write":
        # Preregistered forced-write control: identical surprise router & timescales to `slwm_v3_routed`,
        # floor-clamping write rate `eta >= 0.5` on every valid tick (prohibits write abstention)!
        base = replace(base, use_nowrite_option=False, force_write_floor=0.5)
    elif backbone_type == "slwm_v3_no_zerolag":
        base = replace(base, use_zero_lag_residual=False)
    elif backbone_type == "single_bank_spread_tau":
        # 2x2 factorial cell: IDENTICAL linear DeltaNet `w_beta` controller as `single_bank_gated_deltanet`,
        # changing ONLY timescale prior from tied `64s` to spread `(1, 8, 64, 512)s`!
        base = replace(
            base,
            use_learned_routing=False,
            use_nowrite_option=False,
            use_surprise_router=False,
            tie_bank_timescales=False,
        )
    elif backbone_type == "single_bank_gated_deltanet":
        # Faithful B=1 Gated DeltaNet with IDENTICAL total heads (H_tot = 4), IDENTICAL per-head rank (dk, dv),
        # and IDENTICAL matrix state bytes as `slwm_v3_routed` — single shared timescale (64s), NO surprise/no-write router!
        base = replace(
            base,
            n_banks=1,
            n_mem_heads=base.n_banks * base.n_mem_heads,
            init_timescales_s=(64.0,),
            use_learned_routing=False,
            use_nowrite_option=False,
            use_surprise_router=False,
            tie_bank_timescales=True,
        )
    return base


def _estimate_param_count_via_eval_shape(spec: SLWMv3Spec) -> int:
    """Compute exact parameter count for any `SLWMv3Spec` with zero memory allocation via `jax.eval_shape`."""
    import jax
    from models.v3.core import init_slwm_v3_params

    tree = jax.eval_shape(lambda k: init_slwm_v3_params(k, spec), jax.random.PRNGKey(0))
    return count_tree_params(tree)


def _match_params_via_ffn_dim(spec: SLWMv3Spec, target_params: int) -> SLWMv3Spec:
    """Solve exact multiple-of-16 `ffn_dim_override` so comparator trainable params match `target_params` within <= 1%."""
    cur_params = _estimate_param_count_via_eval_shape(spec)
    # Each +16 in ffn_dim adds exactly 3 * d_model * 16 * n_layers scalar parameters (SwiGLU gate, up, down)
    step = 16
    delta_per_step = 3 * spec.d_model * step * spec.n_layers
    diff = target_params - cur_params
    steps_n = int(round(float(diff) / float(delta_per_step)))
    cand_ffn = max(128, ((spec.ffn_dim + step * steps_n) // step) * step)
    best_spec = replace(spec, ffn_dim_override=cand_ffn)
    best_err = abs(_estimate_param_count_via_eval_shape(best_spec) - target_params)
    for adj in (-step, step):
        c2 = max(128, cand_ffn + adj)
        s2 = replace(spec, ffn_dim_override=c2)
        e2 = abs(_estimate_param_count_via_eval_shape(s2) - target_params)
        if e2 < best_err:
            best_spec, best_err = s2, e2
    return best_spec


def build_wave_a_comparator_specs(
    profile: str = "discovery_45m",
    *,
    representation_mode: str = "frozen_codec_edge",
    d_model_override: int | None = None,
    match_params_to_anchor: bool = True,
) -> dict[str, SLWMv3Spec]:
    """Construct complete Wave A comparator & ablation suite with exact byte parity AND ±1% param parity."""
    anchor = make_slwm_v3_spec(
        profile,
        backbone_type="slwm_v3_routed",
        representation_mode=representation_mode,
        d_model_override=d_model_override,
    )
    anchor_params = _estimate_param_count_via_eval_shape(anchor) if match_params_to_anchor else 0
    anchor_bytes = compute_core_streaming_state_bytes(anchor, batch_size=1)
    anchor_mem_per_layer = anchor_bytes["memory_state_bytes"] // anchor.n_layers
    anchor_tot_per_layer = anchor_bytes["total_core_state_bytes"] // anchor.n_layers

    # Exact byte-matched SlidingWindow-GQA window size W_eq
    kv_bytes_per_tick = 2 * anchor.n_kv_heads * anchor.attn_head_dim * anchor.ring_kv_bytes_per_elem + 1
    w_eq = max(anchor.window_size, int(round((anchor_tot_per_layer - 4) / kv_bytes_per_tick)))

    # Exact byte-matched multi-head Explicit Bounded LRU/Overwrite KV slots
    h_e = anchor.explicit_kv_heads
    d_e = anchor.explicit_kv_dim
    slot_bytes_all_heads = h_e * ((2 * d_e + 1) * 4 + 1)
    k_kv_eq = max(4, int(round(anchor_mem_per_layer / slot_bytes_all_heads)))

    # Exact byte-matched Diagonal SSM state_dim
    ssm_d = max(4, int(round(anchor_mem_per_layer / (anchor.d_model * 4))))

    raw_specs: dict[str, SLWMv3Spec] = {
        "slwm_v3_routed": anchor,
        "slwm_v3_tied_tau": make_slwm_v3_spec(
            profile, backbone_type="slwm_v3_tied_tau", representation_mode=representation_mode, d_model_override=d_model_override
        ),
        "slwm_v3_no_routing": make_slwm_v3_spec(
            profile, backbone_type="slwm_v3_no_routing", representation_mode=representation_mode, d_model_override=d_model_override
        ),
        "slwm_v3_no_nowrite": make_slwm_v3_spec(
            profile, backbone_type="slwm_v3_no_nowrite", representation_mode=representation_mode, d_model_override=d_model_override
        ),
        "slwm_v3_forced_write": make_slwm_v3_spec(
            profile, backbone_type="slwm_v3_forced_write", representation_mode=representation_mode, d_model_override=d_model_override
        ),
        "slwm_v3_no_zerolag": make_slwm_v3_spec(
            profile, backbone_type="slwm_v3_no_zerolag", representation_mode=representation_mode, d_model_override=d_model_override
        ),
        "single_bank_spread_tau": make_slwm_v3_spec(
            profile, backbone_type="single_bank_spread_tau", representation_mode=representation_mode, d_model_override=d_model_override
        ),
        "single_bank_gated_deltanet": make_slwm_v3_spec(
            profile, backbone_type="single_bank_gated_deltanet", representation_mode=representation_mode, d_model_override=d_model_override
        ),
        "sliding_window_gqa_eqparam": replace(
            make_slwm_v3_spec(
                profile, backbone_type="sliding_window_gqa", representation_mode=representation_mode, d_model_override=d_model_override
            ),
            name=f"{profile}_sliding_window_gqa_eqparam",
        ),
        "sliding_window_gqa_eqbyte": replace(
            make_slwm_v3_spec(
                profile, backbone_type="sliding_window_gqa", representation_mode=representation_mode, d_model_override=d_model_override
            ),
            name=f"{profile}_sliding_window_gqa_eqbyte",
            window_size=w_eq,
        ),
        "diag_ssm": replace(
            make_slwm_v3_spec(
                profile, backbone_type="diag_ssm", representation_mode=representation_mode, d_model_override=d_model_override
            ),
            ssm_state_dim=ssm_d,
        ),
        "explicit_bounded_kv": replace(
            make_slwm_v3_spec(
                profile, backbone_type="explicit_bounded_kv", representation_mode=representation_mode, d_model_override=d_model_override
            ),
            explicit_kv_heads=h_e,
            explicit_kv_slots=k_kv_eq,
            explicit_kv_dim=d_e,
        ),
    }

    if not match_params_to_anchor:
        return raw_specs

    # Match trainable parameter count within <= 1% for every structural comparator
    # (Keep single-factor ablations `slwm_v3_*` at exact identical FFN width so ONLY the ablated mechanism changes!)
    matched: dict[str, SLWMv3Spec] = {}
    for k, sp in raw_specs.items():
        if k in (
            "slwm_v3_routed",
            "slwm_v3_tied_tau",
            "slwm_v3_no_routing",
            "slwm_v3_no_nowrite",
            "slwm_v3_forced_write",
            "slwm_v3_no_zerolag",
        ):
            matched[k] = sp
        else:
            matched[k] = _match_params_via_ffn_dim(sp, anchor_params)
    return matched
