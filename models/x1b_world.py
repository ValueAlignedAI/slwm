"""SLWM-X1B: Structured Spatiotemporal Register-Delta World Model (~150M to ~1.00B parameters).

Architecture overview:
    Edge Signals:
        video   : uint8/float32[B, T, 64, 64, 3] -> VisualSpatialAdapter (Conv2D stem) -> [B, T, 16, D]
        audio   : float32[B, T, 64]              -> AudioSpectralAdapter               -> [B, T, 2, D]
        proprio : float32[B, T, 16]              -> ProprioSensorAdapter               -> [B, T, 1, D]
        text    : int32[B, T, 4]                 -> TextSubwordAdapter (vocab 4096)    -> [B, T, 1, D]
        action  : float32[B, T, 4]               -> ActionMotorAdapter                 -> [B, T, 1, D]
        regs    : learned persistent seeds       -> WorkingMemoryRegisters             -> [B, T, 3, D]

    Structured Latent World Field:
        Z_0 : bfloat16/float32[B, T, S=24, D]
        Slot layout:
            0..15 : 4x4 spatial visual patches (with 2D spatial Fourier encodings)
            16..17: low/high spectral + onset audio tokens
            18    : continuous proprioception / physical state token
            19    : subword language / code token
            20    : issued continuous/discrete motor action token
            21..23: persistent working-memory register slots (R_0, R_1, R_2)

    Core Spatiotemporal Processor (L layers):
        1. Within-tick SpatialCrossModalAttention across S=24 slots (strictly causal across T).
        2. Across-tick Causal Temporal Mixer:
           - slwm_x1b: 3:1 interleaved Multi-Head Gated Delta Matrix Recurrence + Causal RoPE Attention
           - dense_transformer: Causal RoPE Attention at all L layers
           - scalar_retention_hybrid / legacy_scalar_hybrid: 8-group scalar EMA (G5/G10 control)
           - slwm_no_registers: slwm_x1b with register recurrence detached across ticks
           - separate_modality_cores: slwm_x1b with block-diagonal within-tick modality mask
        3. SwiGLU channel mixer with pre-RMSNorm and optional jax.checkpoint rematerialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SLOTS_VISUAL = slice(0, 16)
SLOTS_AUDIO = slice(16, 18)
SLOT_PROPRIO = 18
SLOT_TEXT = 19
SLOT_ACTION = 20
SLOTS_REGISTER = slice(21, 24)
TOTAL_SLOTS = 24

VALID_ARCHS = (
    "slwm_x1b",
    "dense_transformer",
    "scalar_retention_hybrid",
    "legacy_scalar_hybrid",
    "slwm_no_registers",
    "separate_modality_cores",
)


@dataclass(frozen=True)
class X1BSpec:
    """Immutable model specification for SLWM-X1B and matched baselines."""

    profile: str
    arch: str
    layers: int
    d_model: int
    n_heads: int
    head_dim: int
    d_ff: int
    slots_per_tick: int = TOTAL_SLOTS
    vocab_size: int = 4096
    audio_dim: int = 64
    proprio_dim: int = 16
    action_dim: int = 4
    goal_dim: int = 4
    dtype: str = "bfloat16"
    grad_checkpoint: bool = True
    rope_Window: int = 512


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "x_tiny": {
        "layers": 4,
        "d_model": 128,
        "n_heads": 4,
        "head_dim": 32,
        "d_ff": 320,
        "dtype": "float32",
        "grad_checkpoint": False,
    },
    "x150m": {
        "layers": 13,
        "d_model": 768,
        "n_heads": 12,
        "head_dim": 64,
        "d_ff": 2048,
        "dtype": "bfloat16",
        "grad_checkpoint": True,
    },
    "x450m": {
        "layers": 18,
        "d_model": 1152,
        "n_heads": 18,
        "head_dim": 64,
        "d_ff": 3072,
        "dtype": "bfloat16",
        "grad_checkpoint": True,
    },
    "x1b": {
        "layers": 23,
        "d_model": 1536,
        "n_heads": 24,
        "head_dim": 64,
        "d_ff": 4096,
        "dtype": "bfloat16",
        "grad_checkpoint": True,
    },
}


def make_x1b_spec(profile: str = "x1b", arch: str = "slwm_x1b", **overrides: Any) -> X1BSpec:
    """Construct an X1BSpec with automatic FFN width alignment for strict parameter parity (<0.01%)."""
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {tuple(PROFILE_DEFAULTS)}")
    if arch not in VALID_ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; expected one of {VALID_ARCHS}")
    cfg = dict(PROFILE_DEFAULTS[profile])
    cfg.update(overrides)
    d = int(cfg["d_model"])
    h = int(cfg["n_heads"])
    layers = int(cfg["layers"])
    base_ff = int(cfg["d_ff"])

    # GatedDelta layer uses tq(D^2) + tkv_write(2D^2) + tslot_u(D^2) + tgates(D*(2H+D)) + tgates_b(2H+D) + to(D^2)
    # = 6*D^2 + 2*D*H + (2*H + D), whereas Dense/Scalar layer uses tqkv(3D^2) + to(D^2) = 4*D^2.
    # Extra per GatedDelta layer = 2*D^2 + 2*D*H + 2*H + D.
    delta_layers_ref = sum(1 for i in range(layers) if (i + 1) % 4 != 0)
    extra_ref = delta_layers_ref * (2 * d * d + 2 * d * h + 2 * h + d)
    if arch in ("dense_transformer", "scalar_retention_hybrid", "legacy_scalar_hybrid") and "d_ff" not in overrides:
        add_ff = int(round(extra_ref / max(1, 3 * d * layers)))
        cfg["d_ff"] = base_ff + add_ff

    return X1BSpec(profile=profile, arch=arch, **cfg)


def rms_norm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm in float32 precision returning x.dtype.

    Args:
        x: Tensor[..., D]
        scale: Tensor[D]
    Returns:
        Normalized Tensor[..., D] in x.dtype.
    """
    xf = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * inv * scale.astype(jnp.float32)).astype(x.dtype)


def _spatial_grid_fourier(d_model: int, dtype: Any) -> jax.Array:
    """Deterministic 2D spatial Fourier positional basis for the 4x4 visual patch grid.

    Returns:
        Tensor[16, d_model]
    """
    ys, xs = jnp.meshgrid(jnp.linspace(-1.0, 1.0, 4), jnp.linspace(-1.0, 1.0, 4), indexing="ij")
    coords = jnp.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)  # [16, 2]
    n_freq = d_model // 4
    freqs = jnp.geomspace(1.0, 8.0, n_freq) * jnp.pi  # [n_freq]
    ang_y = coords[:, 0:1] * freqs[None, :]
    ang_x = coords[:, 1:2] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang_y), jnp.cos(ang_y), jnp.sin(ang_x), jnp.cos(ang_x)], axis=-1)
    if emb.shape[-1] < d_model:
        emb = jnp.pad(emb, ((0, 0), (0, d_model - emb.shape[-1])))
    return (emb[:, :d_model] * 0.25).astype(dtype)


def _apply_rope_1d(x: jax.Array, positions: jax.Array) -> jax.Array:
    """Apply 1D Rotary Position Embedding over time dimension.

    Args:
        x: Tensor[B_flat, T, H, Dh]
        positions: int32/float32 Tensor[T]
    Returns:
        Rotated Tensor[B_flat, T, H, Dh] in x.dtype.
    """
    dh = x.shape[-1]
    half = dh // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=jnp.float32) / max(1, half)))
    angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]  # [T, half]
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    x1 = x[..., :half].astype(jnp.float32)
    x2 = x[..., half:].astype(jnp.float32)
    out = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return out.astype(x.dtype)


def _separate_modality_mask() -> jax.Array:
    """Block-diagonal 24x24 mask for the 'separate_modality_cores' ablation (AGENTS.md §9).

    Visual (0..15), Audio (16..17), Proprio+Action (18, 20), Text (19), Registers (21..23 isolated).
    """
    groups = np.zeros(TOTAL_SLOTS, dtype=np.int32)
    groups[0:16] = 0
    groups[16:18] = 1
    groups[18] = 2
    groups[20] = 2
    groups[19] = 3
    groups[21:24] = np.arange(4, 7)
    return jnp.asarray(groups[:, None] == groups[None, :])


def _shield_text_slot_mask() -> jax.Array:
    """Directed cross-modal grounding mask (24x24) for unified SLWM-X1B world models.

    Embodied & persistent register slots (0..18, 20..23) attend freely across ALL 24 slots
    (including SLOT_TEXT=19) to ground symbolic/spoken instructions into physical control,
    while SLOT_TEXT=19 attends strictly to itself so continuous visual/proprioceptive pose
    drift never shifts discrete associative keys across long horizons (T=512).
    """
    mask = np.ones((TOTAL_SLOTS, TOTAL_SLOTS), dtype=bool)
    mask[SLOT_TEXT, :] = False
    mask[SLOT_TEXT, SLOT_TEXT] = True
    return jnp.asarray(mask)


def init_x1b_params(spec: X1BSpec, seed: int = 2001) -> dict[str, jax.Array]:
    """Deterministically initialize all strict-accounted trainable parameters in float32.

    Includes:
        - Edge signal codecs (Conv2D visual stem, spectral audio, proprio, text, action, registers)
        - Core L-layer spatiotemporal processor (spatial attn + temporal mixer + SwiGLU)
        - Action-conditioned Residual-JEPA predictor + heteroscedastic UncertaintyHead
        - Learned Innovation PolicyCommitGate + ValueCritic + ActionProposal + Diagnostic Heads
    """
    rng = np.random.default_rng(seed)
    d = spec.d_model
    h = spec.n_heads
    dh = spec.head_dim
    assert h * dh == d, f"Expected n_heads * head_dim == d_model, got {h}*{dh} != {d}"
    dff = spec.d_ff

    def glorot(shape: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
        fan_in = int(np.prod(shape[:-1]))
        std = scale / np.sqrt(max(1, fan_in))
        return (rng.standard_normal(shape, dtype=np.float32) * std).astype(np.float32)

    c1 = max(32, d // 8)
    c2 = max(64, d // 4)
    p: dict[str, np.ndarray] = {}

    # 1. Edge Signal Adapters
    # Visual 64x64x3 -> 32x32xc1 (3x3 s2) -> 16x16xc2 (3x3 s2) -> 4x4xd (4x4 s4)
    p["vis_conv1_w"] = glorot((3, 3, 3, c1), 1.4)
    p["vis_conv1_b"] = np.zeros((c1,), dtype=np.float32)
    p["vis_conv2_w"] = glorot((3, 3, c1, c2), 1.4)
    p["vis_conv2_b"] = np.zeros((c2,), dtype=np.float32)
    p["vis_patch_w"] = glorot((4, 4, c2, d), 1.0)
    p["vis_patch_b"] = np.zeros((d,), dtype=np.float32)

    p["aud_w1"] = glorot((spec.audio_dim, d), 1.0)
    p["aud_w2"] = glorot((spec.audio_dim, d), 1.0)
    p["prop_w"] = glorot((spec.proprio_dim, d), 1.0)
    p["text_emb"] = (rng.standard_normal((spec.vocab_size, d), dtype=np.float32) * 0.6).astype(np.float32)
    p["text_proj"] = glorot((d, d), 1.0)
    p["act_w"] = glorot((spec.action_dim, d), 1.0)
    p["reg_seeds"] = (rng.standard_normal((3, d), dtype=np.float32) * 0.02).astype(np.float32)
    p["slot_emb"] = (rng.standard_normal((spec.slots_per_tick, d), dtype=np.float32) * 0.02).astype(np.float32)
    p["mod_present_emb"] = (rng.standard_normal((4, d), dtype=np.float32) * 0.02).astype(np.float32)

    # 2. Core Processor Blocks
    res_scale = float((2.0 * spec.layers) ** -0.5)
    for i in range(spec.layers):
        pfx = f"L{i}_"
        # Within-tick spatial/cross-modal attention
        p[pfx + "snorm"] = np.ones((d,), dtype=np.float32)
        p[pfx + "sqkv"] = glorot((d, 3 * d), 1.0)
        p[pfx + "so"] = glorot((d, d), res_scale)

        # Across-tick causal temporal mixer
        p[pfx + "tnorm"] = np.ones((d,), dtype=np.float32)
        is_attn = spec.arch == "dense_transformer" or ((i + 1) % 4 == 0)
        if is_attn:
            p[pfx + "tqkv"] = glorot((d, 3 * d), 1.0)
            p[pfx + "to"] = glorot((d, d), res_scale)
        elif spec.arch in ("scalar_retention_hybrid", "legacy_scalar_hybrid"):
            p[pfx + "tqkv"] = glorot((d, 3 * d), 1.0)
            p[pfx + "to"] = glorot((d, d), res_scale)
        else:
            # Multi-Head Gated Delta Matrix Recurrence + Per-Slot Kinematics State
            p[pfx + "tq"] = glorot((d, d), 1.0)
            p[pfx + "tkv_write"] = glorot((d, 2 * d), 1.0)
            p[pfx + "tslot_u"] = glorot((d, d), 1.0)
            # Per-head (alpha, beta) gates + multi-timescale per-slot kinematics gate
            h_long = h // 2
            c_prior = np.exp(-1.0 / np.geomspace(1.2, 128.0, d)).astype(np.float32)
            c_bias = np.log(c_prior / (1.0 - c_prior)).astype(np.float32)
            p[pfx + "tgates"] = glorot((d, 2 * h + d), 2.0)
            p[pfx + "tgates_b"] = np.concatenate(
                [
                    np.full((h_long,), 8.5, dtype=np.float32),       # long-horizon associative retention (~0.9998/tick -> 90% @ T=512)
                    np.full((h - h_long,), 6.0, dtype=np.float32),   # medium-horizon working memory (~0.9975/tick)
                    np.full((h_long,), -4.2, dtype=np.float32),      # content-selective Householder write/overwrite gate
                    np.full((h - h_long,), -3.5, dtype=np.float32),  # event-triggered write gate (idle ~0.029)
                    c_bias,                                          # log-spaced kinematics & register retention [120ms..12.8s]
                ]
            )
            p[pfx + "to"] = glorot((d, d), res_scale)

        # SwiGLU Channel Mixer
        p[pfx + "fnorm"] = np.ones((d,), dtype=np.float32)
        p[pfx + "fgate"] = glorot((d, dff), 1.0)
        p[pfx + "fup"] = glorot((d, dff), 1.0)
        p[pfx + "fdown"] = glorot((dff, d), res_scale)

    p["final_norm"] = np.ones((d,), dtype=np.float32)

    # 3. Action-Conditioned Residual-JEPA Predictor & Uncertainty Head
    p["jepa_act_w"] = glorot((spec.action_dim, d), 1.0)
    p["jepa_horizon_emb"] = (rng.standard_normal((4, d), dtype=np.float32) * 0.02).astype(np.float32)
    p["jepa_norm"] = np.ones((d,), dtype=np.float32)
    p["jepa_w1"] = glorot((d, 2 * d), 1.0)
    p["jepa_gate"] = glorot((d, 2 * d), 1.0)
    p["jepa_w2"] = glorot((2 * d, d), 0.03)
    p["unc_w1"] = glorot((d, d // 2), 1.0)
    p["unc_w2"] = glorot((d // 2, 1), 0.1)
    p["unc_b2"] = np.array([-1.0], dtype=np.float32)

    # SIGReg / Cross-Modal Alignment Projections
    proj_d = min(256, d)
    p["align_vis_w"] = glorot((d, proj_d), 1.0)
    p["align_aud_w"] = glorot((d, proj_d), 1.0)
    p["align_prop_w"] = glorot((d, proj_d), 1.0)
    p["align_logit_scale"] = np.array([np.log(10.0)], dtype=np.float32)

    # 4. Grounded Task, Policy Commit Gate, Value Critic & Diagnostic Decoder Heads
    p["head_proprio_w"] = glorot((d, spec.proprio_dim), 0.5)
    p["head_contact_w"] = glorot((d, 2), 0.5)  # [impact_event, barrier_contact]
    act_w1 = glorot((2 * d + spec.goal_dim, d), 1.2)
    act_w1[2 * d :, :] = (rng.standard_normal((spec.goal_dim, d), dtype=np.float32) * 0.45).astype(np.float32)
    p["head_action_w1"] = act_w1
    p["head_action_w2"] = glorot((d, spec.action_dim), 0.2)
    val_w1 = glorot((2 * d + spec.goal_dim, d), 1.2)
    val_w1[2 * d :, :] = (rng.standard_normal((spec.goal_dim, d), dtype=np.float32) * 0.45).astype(np.float32)
    p["head_value_w1"] = val_w1
    p["head_value_w2"] = glorot((d, 1), 0.2)

    # Learned PolicyCommitGate: [reg_summary (D) + goal (4) + innov_stats (6)] -> 4 commitment modes
    p["commit_w1"] = glorot((d + spec.goal_dim + 6, d // 2), 1.0)
    p["commit_w2"] = glorot((d // 2, 4), 0.2)
    p["commit_b2"] = np.array([0.5, 0.0, -0.2, -0.3], dtype=np.float32)

    # Subword Text Decoder Head & Diagnostic Patch/Audio Readouts
    p["head_text_w"] = glorot((d, spec.vocab_size), 0.5)
    # Each of the 16 spatial patches decodes its 16x16x3 = 768 local pixel neighborhood!
    p["diag_vis_patch_w"] = glorot((d, 16 * 16 * 3), 0.5)
    p["diag_aud_w"] = glorot((d, spec.audio_dim), 0.5)

    return {k: jnp.asarray(v) for k, v in p.items()}


def count_spec_parameters(spec: X1BSpec) -> dict[str, int]:
    """Zero-allocation analytical strict parameter counter matching init_x1b_params bit-for-bit."""
    d = spec.d_model
    h = spec.n_heads
    dff = spec.d_ff
    c1 = max(32, d // 8)
    c2 = max(64, d // 4)
    proj_d = min(256, d)

    adapters = (
        (3 * 3 * 3 * c1 + c1)
        + (3 * 3 * c1 * c2 + c2)
        + (4 * 4 * c2 * d + d)
        + 2 * (spec.audio_dim * d)
        + (spec.proprio_dim * d)
        + (spec.vocab_size * d + d * d)
        + (spec.action_dim * d)
        + (3 * d)
        + (spec.slots_per_tick * d)
        + (4 * d)
    )
    core = d  # final_norm
    for i in range(spec.layers):
        core += d + 3 * d * d + d * d  # spatial
        core += d  # tnorm
        is_attn = spec.arch == "dense_transformer" or ((i + 1) % 4 == 0)
        if is_attn or spec.arch in ("scalar_retention_hybrid", "legacy_scalar_hybrid"):
            core += 3 * d * d + d * d
        else:
            core += d * d + 2 * d * d + d * d + d * (2 * h + d) + (2 * h + d) + d * d
        core += d + 3 * d * dff  # SwiGLU

    jepa_unc = (
        spec.action_dim * d
        + 4 * d
        + d
        + 2 * (d * 2 * d)
        + (2 * d * d)
        + (d * (d // 2))
        + (d // 2)
        + 1
        + 3 * (d * proj_d)
        + 1
    )
    heads = (
        d * spec.proprio_dim
        + d * 2
        + (2 * d + spec.goal_dim) * d
        + d * spec.action_dim
        + (2 * d + spec.goal_dim) * d
        + d * 1
        + (d + spec.goal_dim + 6) * (d // 2)
        + (d // 2) * 4
        + 4
        + d * spec.vocab_size
        + d * (16 * 16 * 3)
        + d * spec.audio_dim
    )
    return {
        "adapters": adapters,
        "processor_core": core,
        "jepa_and_uncertainty": jepa_unc,
        "policy_and_heads": heads,
        "strict_total": adapters + core + jepa_unc + heads,
    }


def count_parameters(params: dict[str, Any]) -> dict[str, int]:
    """Compute strict total and component-level parameter counts (AGENTS.md §4 & §10)."""
    buckets = {
        "adapters": 0,
        "processor_core": 0,
        "jepa_and_uncertainty": 0,
        "policy_and_heads": 0,
    }
    for k, v in params.items():
        n = int(np.prod(v.shape))
        if k.startswith(("vis_", "aud_", "prop_", "text_", "act_", "reg_", "slot_", "mod_")):
            buckets["adapters"] += n
        elif k.startswith("L") or k == "final_norm":
            buckets["processor_core"] += n
        elif k.startswith(("jepa_", "unc_", "align_")):
            buckets["jepa_and_uncertainty"] += n
        else:
            buckets["policy_and_heads"] += n
    buckets["strict_total"] = sum(buckets.values())
    return buckets


def _subtoken_role_basis(d_model: int, dtype: Any) -> jax.Array:
    """Disjoint quarter-subspace orthogonal basis across sub-token slots [cmd, key, val, query_flag].

    Guarantees <role_r, role_s> == 0 for r != s AND exact scale invariance of key/value embeddings
    regardless of which other sub-token slots are active on tick t.
    """
    q = max(1, d_model // 4)
    role_idx = jnp.minimum(jnp.arange(d_model, dtype=jnp.int32) // q, 3)
    basis = jnp.where(jnp.arange(4, dtype=jnp.int32)[:, None] == role_idx[None, :], 2.0, 0.0)
    return basis.astype(dtype)


def encode_signals(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    batch: dict[str, jax.Array],
) -> jax.Array:
    """Encode raw multimodal edge signals into the structured latent field Z_0[B, T, S=24, D].

    Args:
        batch:
            video         : uint8 or float32[B, T, 64, 64, 3]
            audio         : float32[B, T, 64]
            proprio       : float32[B, T, 16]
            text          : int32[B, T, 4]
            action        : float32[B, T, 4]
            modality_mask : bool/float32[B, T, 4] for (audio, video, proprio, text)

    Returns:
        Z_0 : Tensor[B, T, 24, D] in spec.dtype.
    """
    dtype = getattr(jnp, spec.dtype)
    video = batch["video"]
    b, t = video.shape[:2]
    d = spec.d_model

    # Modality mask: [B, T, 4] -> (audio, video, proprio, text)
    mod_mask = batch.get("modality_mask")
    if mod_mask is None:
        mod_mask = jnp.ones((b, t, 4), dtype=dtype)
    else:
        mod_mask = mod_mask.astype(dtype)

    # 1. Visual 2-stage overlapping Conv2D + 4x4 patch stem (preserves 1-px thin barriers!)
    v_in = (video.astype(jnp.float32) / 127.5 - 1.0).astype(dtype).reshape(b * t, 64, 64, 3)
    h1 = jax.nn.silu(
        jax.lax.conv_general_dilated(
            v_in,
            params["vis_conv1_w"].astype(dtype),
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_conv1_b"].astype(dtype)
    )
    h1_pool = jnp.max(h1.reshape(b * t, 16, 2, 16, 2, -1), axis=(2, 4))
    c2 = params["vis_conv2_b"].shape[0]
    h2 = jax.nn.silu(
        jax.lax.conv_general_dilated(
            h1,
            params["vis_conv2_w"].astype(dtype),
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_conv2_b"].astype(dtype)
    ) + 0.30 * jnp.tile(h1_pool, (1, 1, 1, (c2 + h1.shape[-1] - 1) // h1.shape[-1]))[..., :c2]
    v_patches = (
        jax.lax.conv_general_dilated(
            h2,
            params["vis_patch_w"].astype(dtype),
            window_strides=(4, 4),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_patch_b"].astype(dtype)
    ).reshape(b, t, 16, d)
    v_patches = (v_patches + _spatial_grid_fourier(d, dtype)[None, None, :, :]) * mod_mask[:, :, 1:2, None]

    # 2. Spectral + onset audio tokens: [B, T, 2, D]
    aud = batch["audio"].astype(dtype)
    a_tok1 = aud @ params["aud_w1"].astype(dtype)
    a_tok2 = jax.nn.silu(aud @ params["aud_w2"].astype(dtype))
    a_tokens = jnp.stack([a_tok1, a_tok2], axis=2) * mod_mask[:, :, 0:1, None]

    # 3. Proprioceptive physical state token: [B, T, 1, D]
    p_tok = (batch["proprio"].astype(dtype) @ params["prop_w"].astype(dtype))[:, :, None, :]
    p_tok = p_tok * mod_mask[:, :, 2:3, None]

    # 4. Role-structured subword language/code token (4096 vocab, zero hash collisions): [B, T, 1, D]
    txt_ids = jnp.clip(batch["text"].astype(jnp.int32), 0, spec.vocab_size - 1)
    txt_valid = (txt_ids > 0).astype(dtype)[..., None]
    role_mod = _subtoken_role_basis(d, dtype)[None, None, :, :]
    txt_emb = params["text_emb"].astype(dtype)[txt_ids] * role_mod * txt_valid
    w_tok = (txt_emb.sum(axis=2) @ params["text_proj"].astype(dtype))[:, :, None, :]
    w_tok = w_tok * mod_mask[:, :, 3:4, None]

    # 5. Motor action conditioning token: [B, T, 1, D]
    u_tok = (batch["action"].astype(dtype) @ params["act_w"].astype(dtype))[:, :, None, :]

    # 6. Persistent working-memory register seed slots: [B, T, 3, D]
    mod_context = (mod_mask @ params["mod_present_emb"].astype(dtype))[:, :, None, :]
    reg_tok = jnp.broadcast_to(params["reg_seeds"].astype(dtype)[None, None, :, :], (b, t, 3, d)) + mod_context

    z0 = jnp.concatenate([v_patches, a_tokens, p_tok, w_tok, u_tok, reg_tok], axis=2)
    z0 = z0 + params["slot_emb"].astype(dtype)[None, None, :, :]
    return z0


def _spatial_cross_modal_attn(
    x: jax.Array,
    qkv_w: jax.Array,
    o_w: jax.Array,
    n_heads: int,
    head_dim: int,
    separate_cores: bool,
    shield_text_slot: bool = False,
) -> jax.Array:
    """Within-tick attention across S=24 slots. Strictly local to tick t (zero future leakage).

    Args:
        x: Tensor[B, T, S, D]
    Returns:
        Tensor[B, T, S, D]
    """
    b, t, s, d = x.shape
    qkv = (x @ qkv_w).reshape(b * t, s, 3, n_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [BT, S, H, Dh]
    scores = jnp.einsum("bshd,bthd->bhst", q, k).astype(jnp.float32) * (head_dim**-0.5)
    if separate_cores:
        mask = _separate_modality_mask()[None, None, :, :]
        scores = jnp.where(mask, scores, -1e9)
    elif shield_text_slot:
        mask = _shield_text_slot_mask()[None, None, :, :]
        scores = jnp.where(mask, scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("bhst,bthd->bshd", probs, v).reshape(b, t, s, d)
    return out @ o_w


def _causal_rope_temporal_attn(
    x: jax.Array,
    qkv_w: jax.Array,
    o_w: jax.Array,
    valid: jax.Array,
    n_heads: int,
    head_dim: int,
    window: int,
    offset: int = 0,
    prev_cache: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Across-tick Causal RoPE Self-Attention per slot across time 1..T.

    Args:
        x: Tensor[B, T, S, D]
        valid: bool Tensor[B, T]
    Returns:
        (out: Tensor[B, T, S, D], cache: (K, V, valid))
    """
    b, t, s, d = x.shape
    # Transpose to [B*S, T, D] so every slot attends causally along its temporal trajectory
    xs = jnp.swapaxes(x, 1, 2).reshape(b * s, t, d)
    qkv = (xs @ qkv_w).reshape(b * s, t, 3, n_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    pos_q = jnp.arange(t, dtype=jnp.int32) + offset
    q = _apply_rope_1d(q, pos_q)
    k = _apply_rope_1d(k, pos_q)

    valid_bs = jnp.repeat(valid, s, axis=0)  # [B*S, T]
    if prev_cache is not None:
        c_len = prev_cache[0].shape[1]
        if t < c_len:
            k = jnp.concatenate([prev_cache[0][:, t:], k], axis=1)
            v = jnp.concatenate([prev_cache[1][:, t:], v], axis=1)
            valid_bs = jnp.concatenate([prev_cache[2][:, t:], valid_bs], axis=1)

    k_len = k.shape[1]
    dist = (jnp.arange(t)[:, None] + (k_len - t)) - jnp.arange(k_len)[None, :]
    causal_mask = (dist >= 0) & (dist < window)
    scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) * (head_dim**-0.5)
    full_mask = causal_mask[None, None, :, :] & valid_bs[:, None, None, :]
    probs = jax.nn.softmax(jnp.where(full_mask, scores, -1e9), axis=-1).astype(x.dtype)
    out_bs = jnp.einsum("bhts,bshd->bthd", probs, v).reshape(b, s, t, d)
    out = jnp.swapaxes(out_bs, 1, 2) @ o_w
    keep = k_len if prev_cache is not None else min(window, k_len)
    return out, (k[:, -keep:], v[:, -keep:], valid_bs[:, -keep:])


def init_streaming_memories(spec: X1BSpec, batch_size: int, cache_len: int = 64) -> tuple[Any, ...]:
    """Allocate static-shape recurrent carries / KV ring buffers for zero-recompilation XLA streaming."""
    dtype = getattr(jnp, spec.dtype)
    b = batch_size
    s = spec.slots_per_tick
    d = spec.d_model
    h = spec.n_heads
    dh = spec.head_dim
    mems = []
    for i in range(spec.layers):
        is_attn = spec.arch == "dense_transformer" or ((i + 1) % 4 == 0)
        if is_attn:
            mems.append(
                (
                    jnp.zeros((b * s, cache_len, h, dh), dtype=dtype),
                    jnp.zeros((b * s, cache_len, h, dh), dtype=dtype),
                    jnp.zeros((b * s, cache_len), dtype=bool),
                )
            )
        elif spec.arch in ("scalar_retention_hybrid", "legacy_scalar_hybrid"):
            mems.append(jnp.zeros((b, s, 8, d // 8), dtype=jnp.float32))
        else:
            mems.append(
                (
                    jnp.zeros((b, h, dh, dh), dtype=jnp.float32),
                    jnp.zeros((b, s, d), dtype=jnp.float32),
                )
            )
    return tuple(mems)


def _scalar_retention_temporal(
    x: jax.Array,
    qkv_w: jax.Array,
    o_w: jax.Array,
    valid: jax.Array,
    prev_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Matched G5/G10-style 8-group scalar exponential moving average recurrence (no matrix KV state).

    Args:
        x: Tensor[B, T, S, D]
    Returns:
        (out: Tensor[B, T, S, D], last_state: float32[B, S, 8, D//8])
    """
    b, t, s, d = x.shape
    groups = 8
    gd = d // groups
    qkv = (x @ qkv_w).reshape(b, t, s, 3, d)
    q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]

    prior = jnp.exp(-1.0 / jnp.geomspace(1.0, 128.0, groups))
    base = jnp.log(prior / (1.0 - prior))  # [8]

    k_grp = k.reshape(b, t, s, groups, gd)
    gate = k_grp.astype(jnp.float32).mean(axis=-1) + base[None, None, None, :]
    decay = jax.nn.sigmoid(gate)  # [B, T, S, 8]
    obs = (q * jax.nn.sigmoid(k)).reshape(b, t, s, groups, gd).astype(jnp.float32)

    init = (
        jnp.zeros((b, s, groups, gd), dtype=jnp.float32)
        if prev_state is None
        else prev_state.astype(jnp.float32)
    )

    def step_fn(state: jax.Array, inp: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        dec_t, obs_t, ok_t = inp  # [B, S, 8], [B, S, 8, gd], [B]
        cand = dec_t[..., None] * state + (1.0 - dec_t[..., None]) * obs_t
        nxt = jnp.where(ok_t[:, None, None, None], cand, state)
        return nxt, nxt.astype(x.dtype)

    last, smooth = jax.lax.scan(
        step_fn,
        init,
        (jnp.swapaxes(decay, 0, 1), jnp.swapaxes(obs, 0, 1), valid.T),
    )
    smooth_bt = jnp.swapaxes(smooth, 0, 1).reshape(b, t, s, d)
    mix = jax.nn.sigmoid(v)
    out = (mix * smooth_bt + (1.0 - mix) * obs.reshape(b, t, s, d).astype(x.dtype)) @ o_w
    return out, last


def _gated_delta_temporal(
    x: jax.Array,
    tq_w: jax.Array,
    tkv_write_w: jax.Array,
    tslot_u_w: jax.Array,
    tgates_w: jax.Array,
    tgates_b: jax.Array,
    to_w: jax.Array,
    valid: jax.Array,
    n_heads: int,
    head_dim: int,
    detach_registers: bool = False,
    prev_state: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Multi-Head Gated Delta Matrix Recurrence + Per-Slot Local Kinematics State.

    Mathematical update at each causal tick t:
        M_t^(h) = alpha_t^(h) * M_{t-1}^(h) + beta_t^(h) * (v_t^(h) - M_{t-1}^(h) k_t^(h)) (k_t^(h))^T
        o_{t,s}^(h) = M_t^(h) q_{t,s}^(h)
        c_{t,s} = gamma_{t,s} * c_{t-1,s} + (1 - gamma_{t,s}) * u_{t,s}

    Args:
        x: Tensor[B, T, S, D]
        valid: bool Tensor[B, T]
    Returns:
        (out: Tensor[B, T, S, D], (M_T: float32[B, H, Dh, Dh], c_T: float32[B, S, D]))
    """
    b, t, s, d = x.shape

    # Per-slot associative queries and local kinematics proposals
    q_all = (x @ tq_w).reshape(b, t, s, n_heads, head_dim)
    q_all = q_all * jax.lax.rsqrt(jnp.sum(q_all.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(x.dtype)
    u_slot = x @ tslot_u_w  # [B, T, S, D]

    # Write summary splits heads cleanly into long-horizon symbolic KV heads vs embodied working-memory heads
    h_long = n_heads // 2
    if detach_registers:
        # Ablation: zero register slots in recurrence write summary & dilute across all 21 sensory slots
        write_src_long = x[:, :, :21, :].mean(axis=2)
        write_src_emb = write_src_long
        gates_raw = (write_src_long @ tgates_w).astype(jnp.float32) + tgates_b[None, None, :]
        alpha = jax.nn.sigmoid(gates_raw[:, :, :n_heads])
        beta = jax.nn.sigmoid(gates_raw[:, :, n_heads : 2 * n_heads])
        gamma_raw = gates_raw[:, :, 2 * n_heads :]
    else:
        # Symbol-pure long-horizon heads [0:h_long] bind/overwrite KV pairs with exact pose invariance across T=512;
        # embodied heads [h_long:H] latch transient visual + proprioceptive goals into persistent working memory!
        reg_mean = x[:, :, SLOTS_REGISTER, :].mean(axis=2)
        write_src_long = x[:, :, SLOT_TEXT, :]
        write_src_emb = (
            0.60 * reg_mean
            + 0.22 * x[:, :, SLOT_PROPRIO, :]
            + 0.18 * x[:, :, SLOTS_VISUAL, :].mean(axis=2)
        )
        w_long_gates = jnp.concatenate(
            [tgates_w[:, :h_long], tgates_w[:, n_heads : n_heads + h_long]], axis=-1
        )
        b_long_gates = jnp.concatenate(
            [tgates_b[:h_long], tgates_b[n_heads : n_heads + h_long]], axis=-1
        )
        gates_long = (write_src_long @ w_long_gates).astype(jnp.float32) + b_long_gates[None, None, :]

        w_emb_gates = jnp.concatenate(
            [tgates_w[:, h_long:n_heads], tgates_w[:, n_heads + h_long : 2 * n_heads], tgates_w[:, 2 * n_heads :]],
            axis=-1,
        )
        b_emb_gates = jnp.concatenate(
            [tgates_b[h_long:n_heads], tgates_b[n_heads + h_long : 2 * n_heads], tgates_b[2 * n_heads :]],
            axis=-1,
        )
        gates_emb = (write_src_emb @ w_emb_gates).astype(jnp.float32) + b_emb_gates[None, None, :]

        h_emb = n_heads - h_long
        alpha = jax.nn.sigmoid(
            jnp.concatenate([gates_long[:, :, :h_long], gates_emb[:, :, :h_emb]], axis=-1)
        )
        beta = jax.nn.sigmoid(
            jnp.concatenate([gates_long[:, :, h_long:], gates_emb[:, :, h_emb : 2 * h_emb]], axis=-1)
        )
        gamma_raw = gates_emb[:, :, 2 * h_emb :]

    kv_w_reshaped = tkv_write_w.reshape(d, 2, n_heads, head_dim)
    kv_long = jnp.einsum("btd,dchk->btchk", write_src_long, kv_w_reshaped[:, :, :h_long, :])
    kv_emb = jnp.einsum("btd,dchk->btchk", write_src_emb, kv_w_reshaped[:, :, h_long:, :])
    kv_write = jnp.concatenate([kv_long, kv_emb], axis=3)
    k_w = kv_write[:, :, 0].astype(jnp.float32)
    v_w = kv_write[:, :, 1].astype(jnp.float32)
    # L2-normalize key so (I - beta k k^T) is a strict non-expansive Householder contraction!
    k_w = k_w * jax.lax.rsqrt(jnp.sum(k_w * k_w, axis=-1, keepdims=True) + 1e-6)

    # Multi-rate kinematics: fast sub-second velocity filter on sensory slots (0..20) vs multi-timescale
    # [120ms..12.8s] retention on persistent working-memory registers (21..23).
    gamma_sens = jax.nn.sigmoid(gamma_raw - 2.5)  # [B, T, D]
    gamma_reg = jax.nn.sigmoid(gamma_raw)         # [B, T, D]

    if prev_state is None:
        m_init = jnp.zeros((b, n_heads, head_dim, head_dim), dtype=jnp.float32)
        c_init = jnp.zeros((b, s, d), dtype=jnp.float32)
    else:
        m_init, c_init = prev_state[0].astype(jnp.float32), prev_state[1].astype(jnp.float32)

    slot_u_weight = jnp.concatenate(
        [
            jnp.full((19,), 0.82, dtype=jnp.float32),
            jnp.array([0.82 if detach_registers else 1.0, 0.82], dtype=jnp.float32),
            jnp.full((3,), 1.0 if detach_registers else 0.20, dtype=jnp.float32),
        ]
    )[None, :, None]
    slot_c_weight = 1.0 - slot_u_weight
    slot_assoc_gain = jnp.concatenate(
        [
            jnp.full((16,), 0.10, dtype=jnp.float32),
            jnp.full((3,), 0.30, dtype=jnp.float32),
            jnp.array([1.0, 0.30], dtype=jnp.float32),
            jnp.full((3,), 0.0 if detach_registers else 1.0, dtype=jnp.float32),
        ]
    )[None, :, None]

    def scan_step(
        carry: tuple[jax.Array, jax.Array],
        step_inp: tuple[
            jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        m_prev, c_prev = carry
        q_t, k_t, v_t, a_t, b_t, gs_t, gr_t, u_t, ok_t = step_inp
        # Causal Read-Before-Write: query m_prev BEFORE tick-t Householder update so queries never self-erase!
        q_scaled = q_t.astype(jnp.float32) * (head_dim**0.5)
        assoc_heads = jnp.einsum("bhij,bshj->bshi", m_prev, q_scaled)
        if not detach_registers:
            assoc_heads = assoc_heads.at[:, SLOT_TEXT, h_long:, :].set(0.0)
        assoc_out = assoc_heads.reshape(b, s, d)

        # Predicted value currently stored at key k_t: M_{t-1} k_t -> [B, H, Dh]
        v_pred = jnp.einsum("bhij,bhj->bhi", m_prev, k_t)
        delta_v = v_t - a_t[:, :, None] * v_pred
        m_cand = a_t[:, :, None, None] * m_prev + b_t[:, :, None, None] * jnp.einsum("bhi,bhj->bhij", delta_v, k_t)

        g_slot = jnp.concatenate(
            [
                jnp.broadcast_to(gs_t[:, None, :], (b, 21, d)),
                jnp.broadcast_to(gr_t[:, None, :], (b, 3, d)),
            ],
            axis=1,
        )
        u_f32 = u_t.astype(jnp.float32)
        c_cand = g_slot * c_prev + (1.0 - g_slot) * u_f32
        if detach_registers:
            c_cand = c_cand.at[:, SLOTS_REGISTER, :].set(u_f32[:, SLOTS_REGISTER, :])
        else:
            c_cand = c_cand.at[:, SLOT_TEXT, :].set(u_f32[:, SLOT_TEXT, :])

        m_next = jnp.where(ok_t[:, None, None, None], m_cand, m_prev)
        c_next = jnp.where(ok_t[:, None, None], c_cand, c_prev)

        # Zero-Lag Lead-Lag Sensory Bypass (0.82*u_t + 0.18*c_next) eliminates multi-layer phase lag at L=23
        # while SLOTS_REGISTER retain persistent working memory (0.20*u_t + 0.80*c_next + assoc_out).
        local_kin = slot_u_weight * u_f32 + slot_c_weight * c_next
        combined = (slot_assoc_gain * assoc_out + local_kin).astype(x.dtype)
        return (m_next, c_next), combined

    (m_last, c_last), seq_out = jax.lax.scan(
        scan_step,
        (m_init, c_init),
        (
            jnp.swapaxes(q_all, 0, 1),
            jnp.swapaxes(k_w, 0, 1),
            jnp.swapaxes(v_w, 0, 1),
            jnp.swapaxes(alpha, 0, 1),
            jnp.swapaxes(beta, 0, 1),
            jnp.swapaxes(gamma_sens, 0, 1),
            jnp.swapaxes(gamma_reg, 0, 1),
            jnp.swapaxes(u_slot, 0, 1),
            valid.T,
        ),
    )
    out = jnp.swapaxes(seq_out, 0, 1) @ to_w
    return out, (m_last, c_last)


def _run_single_block(
    x: jax.Array,
    layer_idx: int,
    params: dict[str, jax.Array],
    spec: X1BSpec,
    valid: jax.Array,
    offset: int,
    prev_mem: Any,
) -> tuple[jax.Array, Any]:
    """Execute one spatiotemporal world-processor block."""
    dtype = getattr(jnp, spec.dtype)
    pfx = f"L{layer_idx}_"

    # 1. Within-tick Spatial & Cross-Modal Attention
    z_s = rms_norm(x, params[pfx + "snorm"])
    x = x + _spatial_cross_modal_attn(
        z_s,
        params[pfx + "sqkv"].astype(dtype),
        params[pfx + "so"].astype(dtype),
        spec.n_heads,
        spec.head_dim,
        separate_cores=(spec.arch == "separate_modality_cores"),
        shield_text_slot=(spec.arch in ("slwm_x1b", "slwm_no_registers")),
    )

    # 2. Across-tick Causal Temporal Mixer
    z_t = rms_norm(x, params[pfx + "tnorm"])
    is_attn = spec.arch == "dense_transformer" or ((layer_idx + 1) % 4 == 0)
    if is_attn:
        # Hybrid world models use bounded O(1) sub-second sliding-window attention (W=8 ticks = 0.8s) paired with
        # global position-free GatedDelta so multi-second associative recall & overwrite MUST flow through M_t;
        # DenseTransformer relies on full-window RoPE attention across the whole sequence.
        attn_win = spec.rope_Window if spec.arch == "dense_transformer" else min(8, spec.rope_Window)
        t_out, mem_out = _causal_rope_temporal_attn(
            z_t,
            params[pfx + "tqkv"].astype(dtype),
            params[pfx + "to"].astype(dtype),
            valid,
            spec.n_heads,
            spec.head_dim,
            window=attn_win,
            offset=offset,
            prev_cache=prev_mem,
        )
    elif spec.arch in ("scalar_retention_hybrid", "legacy_scalar_hybrid"):
        t_out, mem_out = _scalar_retention_temporal(
            z_t,
            params[pfx + "tqkv"].astype(dtype),
            params[pfx + "to"].astype(dtype),
            valid,
            prev_state=prev_mem,
        )
    else:
        t_out, mem_out = _gated_delta_temporal(
            z_t,
            params[pfx + "tq"].astype(dtype),
            params[pfx + "tkv_write"].astype(dtype),
            params[pfx + "tslot_u"].astype(dtype),
            params[pfx + "tgates"].astype(dtype),
            params[pfx + "tgates_b"],
            params[pfx + "to"].astype(dtype),
            valid,
            spec.n_heads,
            spec.head_dim,
            detach_registers=(spec.arch == "slwm_no_registers"),
            prev_state=prev_mem,
        )
    x = x + t_out

    # 3. SwiGLU Channel Mixer
    z_f = rms_norm(x, params[pfx + "fnorm"])
    gate = jax.nn.silu(z_f @ params[pfx + "fgate"].astype(dtype))
    up = z_f @ params[pfx + "fup"].astype(dtype)
    x = (x + (gate * up) @ params[pfx + "fdown"].astype(dtype)) * valid[:, :, None, None].astype(dtype)
    return x, mem_out


def forward_world_model(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    batch: dict[str, jax.Array],
    *,
    memories: tuple[Any, ...] | None = None,
    offset: int = 0,
) -> tuple[jax.Array, tuple[Any, ...]]:
    """Map multimodal edge signals to the processed causal world field Z[B, T, S=24, D].

    For `legacy_scalar_hybrid`, pools the emitted field across slots to replicate G5-G10's
    1-vector-per-tick bottleneck at matched 1.00B parameter scale.
    """
    dtype = getattr(jnp, spec.dtype)
    x = encode_signals(params, spec, batch)
    b, t = x.shape[:2]
    valid = batch.get("valid")
    if valid is None:
        valid = jnp.ones((b, t), dtype=bool)
    else:
        valid = valid.astype(bool)

    x = x * valid[:, :, None, None].astype(dtype)
    next_memories = []

    for i in range(spec.layers):
        prev = None if memories is None else memories[i]
        if spec.grad_checkpoint and memories is None:
            block_fn = jax.checkpoint(
                lambda x_in, p_in, v_in, idx=i: _run_single_block(
                    x_in, idx, p_in, spec, v_in, offset, None
                )[0]
            )
            x = block_fn(x, params, valid)
            next_memories.append(None)
        else:
            x, m_out = _run_single_block(x, i, params, spec, valid, offset, prev)
            next_memories.append(m_out)

    z = rms_norm(x, params["final_norm"])
    if spec.arch == "legacy_scalar_hybrid":
        # Replicate G5-G10's collapse of all modalities/patches into 1 shared vector per tick
        pooled = z.mean(axis=2, keepdims=True)
        z = jnp.broadcast_to(pooled, z.shape)
    return z, tuple(next_memories)
