"""Study X3 (`SLWM-1B-480p`): Native 480p (640x480) O(1) Causal Spatiotemporal World Model & Any-to-Any Composition.

Resolves all three Wave 8A 24-slot architectural falsifications while scaling to 1.020B parameters and 640x480 (480p):
  1. Physically split symbolic-memory vs. goal/subtask language slots (`SLOT_SYM_MEM = 54` with exact null-tick
     `beta_sym = 0` feeding `H_sym`, strictly isolated from `SLOT_GOAL_LANG = 55` feeding `H_goal` across all layers).
  2. Causal 3-tap depthwise-temporal acoustic Conv1D (`0` future lookahead, `+512 B` explicit streaming carry) paired
     with a learned sigmoid acoustic activity gate `beta_aud in [0, 1]` preventing trailing log-mel silence overwrite.
  3. Online kinematic system-identification register (`SLOT_SYSID = 59`) fed causal `(prev_action, d_vis_pose, d_mov)`
     innovations to adapt closed-loop control under unseen mass/damping shifts without privileged physics parameters.
  4. Native `640x480` (`480p`) 3-stage causal Conv2D stem -> `6x8 = 48` macro visual slots (`4:3` aspect) + `15x20`
     fine spatial localization grid, with a single-factor `240p` (`320x240`) bottleneck ablation preserving exact
     parameter and tensor parity.
  5. Six causal cross-modal composition heads (`Video/Audio -> Text`, `Text + Context -> 480p Video`,
     `Video + Action -> 16 kHz Audio`, `Text/Video -> 32D Movement`, `Audio + Video -> 16D Actions`, and
     `Audio + Proprio -> Occluded Scene State & PolicyCommitGate`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from models.x1b_sigreg_epps_pulley import epps_pulley_sigreg_loss
from models.x1b_world import _apply_rope_1d, _causal_rope_temporal_attn, rms_norm

X3_TOTAL_SLOTS = 60
X3_SLOTS_VISUAL = slice(0, 48)       # 6x8 (4:3) 480p macro patch grid
X3_SLOTS_AUDIO = slice(48, 51)       # 3 causal temporal-conv spectral/onset slots
X3_SLOTS_MOVEMENT = slice(51, 53)    # 2 embodiment-tagged 32-D movement/proprio slots
X3_SLOT_ACTION = 53                  # 1 16-D motor/movement action conditioning slot
X3_SLOT_SYM_MEM = 54                 # 1 dedicated discrete associative memory slot (Z_mem ONLY)
X3_SLOT_GOAL_LANG = 55               # 1 dedicated continuous language/subtask/goal slot (Z_task)
X3_SLOTS_REGISTER = slice(56, 60)    # 3 spatial/goal registers (56..58) + 1 sysid register (59)
X3_SLOT_SYSID = 59

X3_ASSOC_KEY_OFFSET = 32
X3_ASSOC_NUM_KEYS = 32
X3_ASSOC_VAL_OFFSET = 64
X3_ASSOC_NUM_VALS = 32


@dataclass(frozen=True)
class X3Spec:
    profile: str
    arch: str
    resolution_mode: str = "480p"
    n_layers: int = 24
    d_model: int = 1536
    n_heads: int = 24
    head_dim: int = 64
    d_ff: int = 4352
    n_kv_heads_gqa: int = 3
    h_sym: int = 8
    h_goal: int = 6
    h_phys: int = 10
    vocab_size: int = 4096
    movement_dim: int = 32
    action_dim: int = 16
    audio_dim: int = 64
    num_embodiments: int = 8
    local_attn_window: int = 8
    gqa_window: int = 64
    slots_per_tick: int = X3_TOTAL_SLOTS
    dtype: str = "bfloat16"
    grad_checkpoint: bool = True


def make_x3_spec(
    profile: str = "x3_1b",
    arch: str = "slwm_x3",
    *,
    resolution_mode: str = "480p",
    dtype: str = "bfloat16",
    grad_checkpoint: bool | None = None,
    vocab_size: int = 4096,
) -> X3Spec:
    if arch not in ("slwm_x3", "sliding_window_gqa_x3", "diagonal_ssm_x3"):
        raise ValueError(f"Unsupported Study X3 architecture: {arch!r}")
    if resolution_mode not in ("480p", "240p"):
        raise ValueError(f"Unsupported resolution_mode: {resolution_mode!r}")

    table = {
        "x3_tiny": dict(
            n_layers=4, d_model=192, n_heads=6, head_dim=32,
            d_ff_slwm=512, d_ff_gqa=650, n_kv_heads_gqa=2,
            h_sym=2, h_goal=2, h_phys=2,
        ),
        "x3_150m": dict(
            n_layers=12, d_model=768, n_heads=12, head_dim=64,
            d_ff_slwm=2240, d_ff_gqa=2868, n_kv_heads_gqa=2,
            h_sym=4, h_goal=3, h_phys=5,
        ),
        "x3_450m": dict(
            n_layers=20, d_model=1152, n_heads=18, head_dim=64,
            d_ff_slwm=3072, d_ff_gqa=4056, n_kv_heads_gqa=2,
            h_sym=6, h_goal=4, h_phys=8,
        ),
        "x3_1b": dict(
            n_layers=24, d_model=1536, n_heads=24, head_dim=64,
            d_ff_slwm=4352, d_ff_gqa=5650, n_kv_heads_gqa=3,
            h_sym=8, h_goal=6, h_phys=10,
        ),
    }
    if profile not in table:
        raise ValueError(f"Unknown Study X3 profile {profile!r}; expected one of {sorted(table)}")
    cfg = table[profile]
    d_ff = cfg["d_ff_gqa"] if arch == "sliding_window_gqa_x3" else cfg["d_ff_slwm"]
    use_ckpt = (profile != "x3_tiny") if grad_checkpoint is None else bool(grad_checkpoint)
    return X3Spec(
        profile=profile,
        arch=arch,
        resolution_mode=resolution_mode,
        n_layers=cfg["n_layers"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        head_dim=cfg["head_dim"],
        d_ff=d_ff,
        n_kv_heads_gqa=cfg["n_kv_heads_gqa"],
        h_sym=cfg["h_sym"],
        h_goal=cfg["h_goal"],
        h_phys=cfg["h_phys"],
        vocab_size=vocab_size,
        dtype=dtype,
        grad_checkpoint=use_ckpt,
    )


def _spatial_grid_fourier_6x8(d_model: int, dtype: Any) -> jax.Array:
    """Deterministic 2D Fourier positional basis for the 6x8 (4:3, 480p) visual patch grid."""
    ys, xs = jnp.meshgrid(jnp.linspace(-1.0, 1.0, 6), jnp.linspace(-1.0, 1.0, 8), indexing="ij")
    coords = jnp.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)  # [48, 2]
    n_freq = max(1, d_model // 4)
    freqs = jnp.geomspace(1.0, 12.0, n_freq) * jnp.pi
    ang_y = coords[:, 0:1] * freqs[None, :]
    ang_x = coords[:, 1:2] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(ang_y), jnp.cos(ang_y), jnp.sin(ang_x), jnp.cos(ang_x)], axis=-1)
    if emb.shape[-1] < d_model:
        emb = jnp.pad(emb, ((0, 0), (0, d_model - emb.shape[-1])))
    return (emb[:, :d_model] * 0.22).astype(dtype)


def canonical_grid_centers_15x20() -> jax.Array:
    """Canonical [-1, 1]^2 cell centers (x, y) for the 15x20 fine 480p spatial localization grid."""
    ys = -1.0 + (2.0 * (jnp.arange(15, dtype=jnp.float32) + 0.5) / 15.0)
    xs = -1.0 + (2.0 * (jnp.arange(20, dtype=jnp.float32) + 0.5) / 20.0)
    gy, gx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)  # [300, 2]


def _subtoken_role_basis_4(d_model: int, dtype: Any) -> jax.Array:
    """Orthogonal 4-slot subtoken role modulation [4, D]."""
    idx = jnp.arange(d_model, dtype=jnp.float32)[None, :]
    roles = jnp.arange(1, 5, dtype=jnp.float32)[:, None]
    return (1.0 + 0.35 * jnp.cos(roles * idx * (jnp.pi / max(8.0, float(d_model))))).astype(dtype)


def _shield_x3_slots_mask() -> jax.Array:
    """Directed 60x60 within-tick attention mask for SLWM-X3.

    - `SLOT_SYM_MEM` (54) attends strictly to itself (zero visual/goal/kinematic drift on discrete associative keys).
    - `SLOT_GOAL_LANG` (55) attends only to `{SLOT_GOAL_LANG} U SLOTS_AUDIO` (zero visual pose drift on goal keys,
      while fusing speech + language cleanly).
    - Embodied, visual, audio, action, and register slots attend across sensory/goal/register slots `{0..53, 55..59}`,
      EXCLUDING `SLOT_SYM_MEM` (54) so task gradients never leak into `SLOT_SYM_MEM` at any layer!
    """
    mask = np.ones((X3_TOTAL_SLOTS, X3_TOTAL_SLOTS), dtype=bool)
    # Block all non-sym slots from attending to SLOT_SYM_MEM:
    mask[:, X3_SLOT_SYM_MEM] = False
    # SLOT_SYM_MEM attends strictly to itself:
    mask[X3_SLOT_SYM_MEM, :] = False
    mask[X3_SLOT_SYM_MEM, X3_SLOT_SYM_MEM] = True
    # SLOT_GOAL_LANG attends only to itself + audio slots (48..50):
    mask[X3_SLOT_GOAL_LANG, :] = False
    mask[X3_SLOT_GOAL_LANG, X3_SLOT_GOAL_LANG] = True
    mask[X3_SLOT_GOAL_LANG, 48:51] = True
    # Register 56 (object/color latch) attends strictly to {56, SLOT_GOAL_LANG, SLOTS_AUDIO}:
    mask[56, :] = False
    mask[56, 56] = True
    mask[56, X3_SLOT_GOAL_LANG] = True
    mask[56, 48:51] = True
    # Registers 57..58 (spatial goal/direction latch) attend strictly to {57, 58, SLOT_GOAL_LANG, SLOTS_AUDIO}:
    mask[57:59, :] = False
    mask[57:59, 57:59] = True
    mask[57:59, X3_SLOT_GOAL_LANG] = True
    mask[57:59, 48:51] = True
    return jnp.asarray(mask)


def init_x3_params(spec: X3Spec, seed: int = 3001) -> dict[str, jax.Array]:
    """Deterministically initialize all strict-accounted Study X3 parameters in float32."""
    rng = np.random.default_rng(seed)
    d = spec.d_model
    h = spec.n_heads
    dh = spec.head_dim
    c1, c2 = 64, 192

    def glorot(shape: tuple[int, ...], scale: float = 1.0) -> jax.Array:
        fan_in = shape[0] if len(shape) == 2 else int(np.prod(shape[:-1]))
        fan_out = shape[-1]
        std = scale * np.sqrt(2.0 / max(1, fan_in + fan_out))
        return jnp.asarray(rng.normal(0.0, std, size=shape).astype(np.float32))

    p: dict[str, jax.Array] = {}
    # 1. 480p 3-stage visual stem + 15x20 fine localization projection
    p["vis_conv1_w"] = glorot((5, 5, 3, c1))
    p["vis_conv1_b"] = jnp.zeros((c1,), dtype=jnp.float32)
    p["vis_conv2_w"] = glorot((3, 3, c1, c2))
    p["vis_conv2_b"] = jnp.zeros((c2,), dtype=jnp.float32)
    p["vis_patch_w"] = glorot((5, 5, c2, d))
    p["vis_patch_b"] = jnp.zeros((d,), dtype=jnp.float32)
    p["vis_fine_w"] = glorot((c2, d), 0.7)
    p["vis_fine_b"] = jnp.zeros((d,), dtype=jnp.float32)

    # 2. Causal 3-tap depthwise-temporal audio Conv1D + learned sigmoid acoustic activity gate + 3 slot projections
    dw = np.zeros((3, spec.audio_dim), dtype=np.float32)
    dw[0, :] = 0.20
    dw[1, :] = 0.30
    dw[2, :] = 0.50
    p["aud_dwconv_w"] = jnp.asarray(dw)
    p["aud_act_gate_w"] = glorot((2 * spec.audio_dim, 1), 0.5)
    p["aud_act_gate_b"] = jnp.array([-0.5], dtype=jnp.float32)
    p["aud_slot_w"] = glorot((spec.audio_dim, 3 * d))
    p["aud_slot_b"] = jnp.zeros((3, d), dtype=jnp.float32)

    # 3. Movement/proprioception (2 slots + 8 embodiment IDs) & 16-D action conditioning
    p["mov_w"] = glorot((spec.movement_dim, 2 * d))
    p["embodiment_emb"] = glorot((spec.num_embodiments, d), 0.4)
    p["act_w"] = glorot((spec.action_dim, d))

    # 4. Shared 4096-vocab embedding + separate SLOT_SYM_MEM and SLOT_GOAL_LANG projections + registers
    p["text_emb"] = glorot((spec.vocab_size, d), 0.6)
    p["sym_proj_w"] = glorot((d, d), 0.8)
    p["lang_proj_w"] = glorot((d, d), 0.8)
    p["reg_seeds"] = glorot((4, d), 0.25)
    p["sysid_in_w"] = glorot((spec.action_dim + 4 + spec.movement_dim, d), 0.5)
    p["slot_emb"] = glorot((X3_TOTAL_SLOTS, d), 0.2)
    p["mod_present_emb"] = glorot((4, d), 0.2)

    # 5. Core L-layer spatiotemporal processor (< 0.007% strict parity across SLWM-X3, GQA, DiagonalSSM)
    for li in range(spec.n_layers):
        pfx = f"L{li}_"
        p[pfx + "snorm"] = jnp.ones((d,), dtype=jnp.float32)
        p[pfx + "sqkv"] = glorot((d, 3 * d), 0.7)
        p[pfx + "so"] = glorot((d, d), 0.5 / np.sqrt(spec.n_layers))

        p[pfx + "tnorm"] = jnp.ones((d,), dtype=jnp.float32)
        is_attn = (spec.arch == "sliding_window_gqa_x3") or ((li + 1) % 4 == 0)
        if is_attn:
            h_kv = spec.n_kv_heads_gqa if spec.arch == "sliding_window_gqa_x3" else h
            p[pfx + "tqkv"] = glorot((d, d + 2 * h_kv * dh), 0.7)
            p[pfx + "to"] = glorot((d, d), 0.5 / np.sqrt(spec.n_layers))
        else:
            p[pfx + "tq"] = glorot((d, d), 0.7)
            p[pfx + "tslot_u"] = glorot((d, d), 0.7)
            p[pfx + "tkv_write"] = glorot((d, 2 * d), 0.7)
            p[pfx + "tgates_w"] = glorot((d, 3 * h), 0.3)
            # Bias: high retention alpha (+2.6) on H_sym, ultra-persistent (+3.1) on H_obj, overwritable (+2.2) on H_dir, moderate (+1.2) on H_phys
            h_obj = max(1, spec.h_goal // 2)
            h_dir = spec.h_goal - h_obj
            b_alpha = np.concatenate(
                [
                    np.full((spec.h_sym,), 2.6, dtype=np.float32),
                    np.full((h_obj,), 3.1, dtype=np.float32),
                    np.full((h_dir,), 2.2, dtype=np.float32),
                    np.full((spec.h_phys,), 1.2, dtype=np.float32),
                ]
            )
            b_beta = np.full((h,), -0.4, dtype=np.float32)
            b_gamma = np.full((h,), 0.8, dtype=np.float32)
            p[pfx + "tgates_b"] = jnp.asarray(np.concatenate([b_alpha, b_beta, b_gamma]))
            p[pfx + "to"] = glorot((d, d), 0.5 / np.sqrt(spec.n_layers))

        p[pfx + "mnorm"] = jnp.ones((d,), dtype=jnp.float32)
        p[pfx + "mw1"] = glorot((d, spec.d_ff), 0.7)
        p[pfx + "mw2"] = glorot((d, spec.d_ff), 0.7)
        p[pfx + "mw3"] = glorot((spec.d_ff, d), 0.5 / np.sqrt(spec.n_layers))

    p["final_norm"] = jnp.ones((d,), dtype=jnp.float32)

    # 6. Action-conditioned Residual-EMA-JEPA predictor + heteroscedastic uncertainty
    p["jepa_w1"] = glorot((d + spec.action_dim, 2 * d), 0.7)
    p["jepa_w2"] = glorot((2 * d, d), 0.25)
    p["unc_w"] = glorot((d, 1), 0.2)
    p["unc_b"] = jnp.array([-1.5], dtype=jnp.float32)

    # 7. Six Any-to-Any Cross-Modal Composition & Control Heads
    # (a) Sym-mem readout & Video/Audio -> Text subword decoder (projects to D, ties with text_emb)
    p["head_sym_w"] = glorot((d, d), 0.7)
    p["head_lang_dec_w"] = glorot((d, d), 0.7)
    p["head_lang_dec_b"] = jnp.zeros((d,), dtype=jnp.float32)

    # (b) Coarse-to-fine 480p (640x480) Video Synthesis Decoder (6x8 -> 30x40 -> 120x160 -> 480x640)
    p["vid_dec_up1_w"] = glorot((d, 5 * 5 * 96), 0.6)
    p["vid_dec_up1_b"] = jnp.zeros((96,), dtype=jnp.float32)
    p["vid_dec_up2_w"] = glorot((3, 3, 96, 32 * 16), 0.5)   # pixel-shuffle x4 -> 120x160x32
    p["vid_dec_up2_b"] = jnp.zeros((32 * 16,), dtype=jnp.float32)
    p["vid_dec_rgb_w"] = glorot((3, 3, 32, 3 * 16), 0.3)    # pixel-shuffle x4 -> 480x640x3
    p["vid_dec_rgb_b"] = jnp.zeros((3 * 16,), dtype=jnp.float32)
    p["vid_dec_ctx_gate_w"] = glorot((d, 1), 0.3)

    # (c) Video (+ Action) -> 64-band Causal Spectrogram Audio Synthesis Head
    p["aud_dec_w1"] = glorot((d, d), 0.7)
    p["aud_dec_b1"] = jnp.zeros((d,), dtype=jnp.float32)
    p["aud_dec_w2"] = glorot((d, spec.audio_dim), 0.4)
    p["aud_dec_b2"] = jnp.zeros((spec.audio_dim,), dtype=jnp.float32)

    # (d) Text/Video -> 32-D Movement Flow + Audio/Video -> 16-D Continuous Actions + Goal/Spatial/Commit
    p["ctrl_trunk_w"] = glorot((d, d), 0.7)
    p["ctrl_trunk_b"] = jnp.zeros((d,), dtype=jnp.float32)
    p["head_mov_next_w"] = glorot((d, spec.movement_dim), 0.4)
    p["head_mov_next_b"] = jnp.zeros((spec.movement_dim,), dtype=jnp.float32)
    p["head_mov_step4_w"] = glorot((d, spec.movement_dim), 0.4)
    p["head_mov_step4_b"] = jnp.zeros((spec.movement_dim,), dtype=jnp.float32)
    p["head_act_w"] = glorot((d, spec.action_dim), 0.4)
    p["head_act_b"] = jnp.zeros((spec.action_dim,), dtype=jnp.float32)
    p["head_val_w"] = glorot((d, 1), 0.3)
    p["head_val_b"] = jnp.zeros((1,), dtype=jnp.float32)
    p["head_goal_xy_w"] = glorot((d, 2), 0.4)
    p["head_goal_xy_b"] = jnp.zeros((2,), dtype=jnp.float32)
    p["head_contact_w"] = glorot((d, 2), 0.4)
    p["head_contact_b"] = jnp.zeros((2,), dtype=jnp.float32)

    # Fine 480p spatial pose (4x4 / 6x8 / 15x20) + 8x8 learned occupancy grid + calibrated PolicyCommitGate
    p["spatial_pose_w"] = glorot((d, 12), 0.5)   # [0:4] ee/obj heatmaps, [4:8] lang-vis block query/key, [8:12] fine offsets
    p["spatial_pose_b"] = jnp.zeros((12,), dtype=jnp.float32)
    p["occ_grid_w"] = glorot((d, 64), 0.4)
    p["occ_grid_b"] = jnp.full((64,), -1.2, dtype=jnp.float32)
    p["commit_w1"] = glorot((d + 8, 64), 0.6)
    p["commit_b1"] = jnp.zeros((64,), dtype=jnp.float32)
    p["commit_w2"] = glorot((64, 4), 0.5)
    p["commit_b2"] = jnp.array([0.8, -0.4, -0.5, -0.6], dtype=jnp.float32)
    return p


def count_x3_parameters(params: dict[str, jax.Array]) -> dict[str, int]:
    """Strict parameter accounting broken down by functional subsystem."""
    adapters = 0
    core = 0
    jepa_unc = 0
    heads = 0
    for k, v in params.items():
        n = int(v.size)
        if k.startswith("L") or k == "final_norm":
            core += n
        elif k.startswith(("jepa_", "unc_")):
            jepa_unc += n
        elif k.startswith(("vis_", "aud_dwconv", "aud_act_gate", "aud_slot", "mov_w", "embodiment_", "act_w", "text_emb", "sym_proj", "lang_proj", "reg_seeds", "sysid_in", "slot_emb", "mod_present")):
            adapters += n
        else:
            heads += n
    return {
        "adapters": adapters,
        "processor_core": core,
        "jepa_and_uncertainty": jepa_unc,
        "any_to_any_and_control_heads": heads,
        "strict_total": adapters + core + jepa_unc + heads,
    }


def compute_x3_streaming_state_bytes(spec: X3Spec, batch_size: int = 1) -> dict[str, float | int]:
    """Exact Dual-Track (`Core` + `Strict System`) streaming state accounting for Study X3 (`S = 60` slots)."""
    b = int(batch_size)
    s = spec.slots_per_tick
    d = spec.d_model
    h = spec.n_heads
    dh = spec.head_dim
    bf16_b = 2
    f32_b = 4

    n_attn = spec.n_layers if spec.arch == "sliding_window_gqa_x3" else (spec.n_layers // 4)
    n_rec = spec.n_layers - n_attn
    win = spec.gqa_window if spec.arch == "sliding_window_gqa_x3" else spec.local_attn_window
    h_kv = spec.n_kv_heads_gqa if spec.arch == "sliding_window_gqa_x3" else h

    kv_tensors_bytes = n_attn * b * s * win * 2 * h_kv * dh * bf16_b
    kv_valid_bytes = n_attn * b * s * win * 1
    kv_total_bytes = kv_tensors_bytes + kv_valid_bytes

    if spec.arch == "slwm_x3":
        # f32 matrix M [B, H, Dh, Dh] + f32 slot filter C [B, S, D]
        rec_bytes = n_rec * b * (h * dh * dh * f32_b + s * d * f32_b)
    elif spec.arch == "diagonal_ssm_x3":
        # f32 vector s [B, H, Dh] + f32 slot filter C [B, S, D]
        rec_bytes = n_rec * b * (h * dh * f32_b + s * d * f32_b)
    else:
        rec_bytes = 0

    core_bytes = kv_total_bytes + rec_bytes
    # Explicit streaming frontend/carry bytes (worst-case across 16/24/48 kHz corpora):
    # 1. 48->16 kHz 3:1 causal boxcar (8 B) + 100 ms rolling Hann STFT buffer (1,600 * 4 B = 6,400 B at 16 kHz;
    #    or 2,400 * 4 B = 9,600 B native 24 kHz; or 10,876 B GRID MP2) + 3-tap causal Conv1D carry [B, 2, 64] f32 (512 B)
    #    -> strict upper bound across all active Study X3 & Wave 8A corpora = 10,876 + 512 = 11,388 B.
    audio_frontend_bytes = b * (10876 + 2 * spec.audio_dim * f32_b)
    # 2. Online sysid + innovation decoder carry: prev_vis_pose [B, 4] f32 + prev_mov [B, 32] f32 + prev_z_next [B, S, D] bf16 + prev_lv [B, S, 1] f32 + flags [B, 5]
    decoder_carry_bytes = b * ((4 + spec.movement_dim) * f32_b + s * d * bf16_b + s * f32_b + 5)
    controller_bytes = b * 292
    strict_sys_bytes = core_bytes + audio_frontend_bytes + decoder_carry_bytes + controller_bytes
    mib = float(1024 * 1024)
    return {
        "core_state_bytes": int(core_bytes),
        "core_state_mib": round(core_bytes / mib, 4),
        "kv_cache_bytes": int(kv_total_bytes),
        "recurrent_state_bytes": int(rec_bytes),
        "audio_frontend_state_bytes": int(audio_frontend_bytes),
        "decoder_carry_bytes": int(decoder_carry_bytes),
        "controller_state_bytes": int(controller_bytes),
        "strict_system_state_bytes": int(strict_sys_bytes),
        "strict_system_state_mib": round(strict_sys_bytes / mib, 4),
    }


def apply_240p_bottleneck_if_needed(video_480p: jax.Array, resolution_mode: str) -> jax.Array:
    """Single-factor 240p resolution ablation: downsample 480x640 -> 240x320 boxcar, then bilinear back to 480x640."""
    if resolution_mode == "480p":
        return video_480p
    b, t, h, w, c = video_480p.shape
    vf = video_480p.astype(jnp.float32)
    v_240p = jnp.mean(vf.reshape(b * t, h // 2, 2, w // 2, 2, c), axis=(2, 4))
    v_up = jax.image.resize(v_240p, (b * t, h, w, c), method="linear")
    return jnp.clip(v_up, 0.0, 255.0).astype(video_480p.dtype).reshape(b, t, h, w, c)


def encode_480p_visual_stem(
    params: dict[str, jax.Array],
    spec: X3Spec,
    video_u8: jax.Array,
    vis_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Causal per-frame 480p (480x640x3) visual stem -> (48 macro slots [B,T,48,D], 15x20 fine grid [B,T,300,D], 4x4 pose [B,T,4])."""
    dtype = getattr(jnp, spec.dtype)
    v_in = apply_240p_bottleneck_if_needed(video_u8, spec.resolution_mode)
    b, t, h, w, _ = v_in.shape
    vn = (v_in.astype(dtype) / jnp.asarray(127.5, dtype=dtype) - jnp.asarray(1.0, dtype=dtype)).reshape(b * t, h, w, 3)

    # Stage 1: 480x640 -> 120x160x64 (stride 4)
    h1 = jax.nn.silu(
        jax.lax.conv_general_dilated(
            vn,
            params["vis_conv1_w"].astype(dtype),
            window_strides=(4, 4),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_conv1_b"].astype(dtype)
    )
    # High-freq 4x4 max-pool skip (preserves thin 480p edges/grippers/blocks -> 30x40x64)
    h1_hi = jnp.max(h1.reshape(b * t, 30, 4, 40, 4, 64), axis=(2, 4))

    # Stage 2: 120x160 -> 30x40x192 (stride 4) + high-freq skip
    h2_raw = jax.nn.silu(
        jax.lax.conv_general_dilated(
            h1,
            params["vis_conv2_w"].astype(dtype),
            window_strides=(4, 4),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_conv2_b"].astype(dtype)
    )
    h2 = h2_raw + jnp.asarray(0.30, dtype=dtype) * jnp.tile(h1_hi, (1, 1, 1, 3))

    # Fine 15x20 (300-cell) pure-vision spatial grid for sub-cm 480p localization
    h2_15x20 = jnp.mean(h2.reshape(b * t, 15, 2, 20, 2, 192), axis=(2, 4))
    fine_300 = (
        h2_15x20.reshape(b, t, 300, 192) @ params["vis_fine_w"].astype(dtype)
        + params["vis_fine_b"].astype(dtype)
    ) * vis_mask[:, :, None, None].astype(dtype)

    # Stage 3: 30x40 -> 6x8 = 48 macro 480p visual slots (stride 5x5)
    v_macro = (
        jax.lax.conv_general_dilated(
            h2,
            params["vis_patch_w"].astype(dtype),
            window_strides=(5, 5),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vis_patch_b"].astype(dtype)
    ).reshape(b, t, 48, spec.d_model)
    v_macro = (v_macro + _spatial_grid_fourier_6x8(spec.d_model, dtype)[None, None, :, :]) * vis_mask[:, :, None, None].astype(dtype)

    # Quick equivariant 2D pose estimate (ee_xy, obj_xy) from fine_300 spatial soft-argmax for causal sysid innovation
    centers_300 = canonical_grid_centers_15x20()
    fc = fine_300.astype(jnp.float32)
    fc_zero = fc - jnp.mean(fc, axis=2, keepdims=True)
    heat = fc_zero @ params["spatial_pose_w"].astype(jnp.float32)[:, 0:2]  # [B, T, 300, 2]
    att_ee = jax.nn.softmax(heat[:, :, :, 0], axis=-1)
    att_ob = jax.nn.softmax(heat[:, :, :, 1], axis=-1)
    ee_2d = jnp.einsum("bts,sc->btc", att_ee, centers_300)
    ob_2d = jnp.einsum("bts,sc->btc", att_ob, centers_300)
    vis_pose4 = jnp.concatenate([ee_2d, ob_2d], axis=-1) * vis_mask[:, :, None]
    return v_macro, fine_300, vis_pose4


def encode_causal_audio_conv(
    params: dict[str, jax.Array],
    spec: X3Spec,
    audio_bt: jax.Array,
    aud_mask: jax.Array,
    prev_aud_2: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Causal 3-tap depthwise-temporal Conv1D (`0` future lookahead) + learned sigmoid acoustic activity gate."""
    dtype = getattr(jnp, spec.dtype)
    b, t, ca = audio_bt.shape
    aud_f32 = audio_bt.astype(jnp.float32) * aud_mask[:, :, None]
    carry_in = jnp.zeros((b, 2, ca), dtype=jnp.float32) if prev_aud_2 is None else prev_aud_2.astype(jnp.float32)
    padded = jnp.concatenate([carry_in, aud_f32], axis=1)  # [B, T+2, 64]
    next_aud_2 = padded[:, -2:, :]

    dw = params["aud_dwconv_w"].astype(jnp.float32)  # [3, 64]
    conv_out = (
        dw[0][None, None, :] * padded[:, 0:t, :]
        + dw[1][None, None, :] * padded[:, 1 : t + 1, :]
        + dw[2][None, None, :] * padded[:, 2 : t + 2, :]
    ) * aud_mask[:, :, None]

    # Learned acoustic activity gate in [0, 1] from instantaneous spectral energy + causal spectral flux (a_t - a_{t-1})
    flux = jnp.abs(aud_f32 - padded[:, 1 : t + 1, :])
    gate_in = jnp.concatenate([aud_f32, flux], axis=-1)
    beta_aud = jax.nn.sigmoid(gate_in @ params["aud_act_gate_w"].astype(jnp.float32) + params["aud_act_gate_b"].astype(jnp.float32))
    beta_aud = beta_aud * aud_mask[:, :, None]  # [B, T, 1], exact 0 when audio modality is masked!

    proj = (conv_out.astype(dtype) @ params["aud_slot_w"].astype(dtype)).reshape(b, t, 3, spec.d_model)
    proj = (proj + params["aud_slot_b"].astype(dtype)[None, None, :, :]) * aud_mask[:, :, None, None].astype(dtype)
    proj = proj.at[:, :, 1, :].set(jax.nn.silu(proj[:, :, 1, :]))
    proj = proj.at[:, :, 2, :].set(proj[:, :, 2, :] * beta_aud.astype(dtype))
    return proj, beta_aud, next_aud_2


def init_x3_streaming_carry(spec: X3Spec, batch_size: int = 1) -> dict[str, Any]:
    """Allocate static-shape O(1) streaming state across trunk + causal audio conv + sysid/commit carry."""
    dtype = getattr(jnp, spec.dtype)
    b = int(batch_size)
    s = spec.slots_per_tick
    d = spec.d_model
    h = spec.n_heads
    dh = spec.head_dim

    layers: list[Any] = []
    for li in range(spec.n_layers):
        is_attn = (spec.arch == "sliding_window_gqa_x3") or ((li + 1) % 4 == 0)
        if is_attn:
            win = spec.gqa_window if spec.arch == "sliding_window_gqa_x3" else spec.local_attn_window
            h_kv = spec.n_kv_heads_gqa if spec.arch == "sliding_window_gqa_x3" else h
            layers.append(
                (
                    jnp.zeros((b * s, win, h_kv, dh), dtype=dtype),
                    jnp.zeros((b * s, win, h_kv, dh), dtype=dtype),
                    jnp.zeros((b * s, win), dtype=bool),
                )
            )
        elif spec.arch == "diagonal_ssm_x3":
            layers.append(
                (
                    jnp.zeros((b, h, dh), dtype=jnp.float32),
                    jnp.zeros((b, s, d), dtype=jnp.float32),
                )
            )
        else:
            layers.append(
                (
                    jnp.zeros((b, h, dh, dh), dtype=jnp.float32),
                    jnp.zeros((b, s, d), dtype=jnp.float32),
                )
            )
    return {
        "offset": jnp.int32(0),
        "layers": tuple(layers),
        "prev_aud_2": jnp.zeros((b, 2, spec.audio_dim), dtype=jnp.float32),
        "prev_vis_pose4": jnp.zeros((b, 4), dtype=jnp.float32),
        "prev_mov": jnp.zeros((b, spec.movement_dim), dtype=jnp.float32),
        "has_prev_tick": jnp.zeros((b,), dtype=bool),
        "prev_z_next": jnp.zeros((b, 1, s, d), dtype=dtype),
        "prev_log_var": jnp.zeros((b, 1, s, 1), dtype=jnp.float32),
    }


def _spatial_attn_x3(
    x: jax.Array,
    qkv_w: jax.Array,
    o_w: jax.Array,
    n_heads: int,
    head_dim: int,
) -> jax.Array:
    """Within-tick attention across S=60 slots with exact structural shielding of SLOT_SYM_MEM & SLOT_GOAL_LANG."""
    b, t, s, d = x.shape
    qkv = (x @ qkv_w).reshape(b * t, s, 3, n_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scores = jnp.einsum("bshd,bthd->bhst", q, k).astype(jnp.float32) * (head_dim**-0.5)
    mask = _shield_x3_slots_mask()[None, None, :, :]
    scores = jnp.where(mask, scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("bhst,bthd->bshd", probs, v).reshape(b, t, s, d)
    return out @ o_w


def _recurrent_temporal_x3(
    x: jax.Array,
    tq_w: jax.Array,
    tslot_u_w: jax.Array,
    tkv_write_w: jax.Array,
    tgates_w: jax.Array,
    tgates_b: jax.Array,
    to_w: jax.Array,
    valid: jax.Array,
    spec: X3Spec,
    *,
    sym_write_active: jax.Array,
    txt_present: jax.Array,
    beta_aud: jax.Array,
    prev_state: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Split-head (`H_sym + H_goal + H_phys`) temporal recurrence (`GatedDelta` matrix or `DiagonalSSM` vector).

    Enforces:
      - Exact `beta_sym = 0, alpha_sym = 1` on `H_sym` whenever `sym_write_active == 0` (zero passive associative decay!).
      - `H_goal` writes from `SLOT_GOAL_LANG` when `txt_present > 0.5`, else from `beta_aud`-gated `SLOTS_AUDIO`
        (preventing trailing log-mel silence from overwriting latched spoken attributes!).
      - Zero cross-contamination between `SLOT_SYM_MEM` (`H_sym`) and `SLOT_GOAL_LANG` / `SLOTS_REGISTER` (`H_goal + H_phys`).
    """
    b, t, s, d = x.shape
    h = spec.n_heads
    dh = spec.head_dim
    h_sym = spec.h_sym
    h_goal = spec.h_goal
    h_obj = max(1, h_goal // 2)
    h_obj_end = h_sym + h_obj
    h_long = h_sym + h_goal
    h_phys = spec.h_phys

    tq_reshaped = tq_w.reshape(d, h, dh)
    q_raw = jnp.einsum("btsd,dhk->btshk", x, tq_reshaped)
    q_all = q_raw * jax.lax.rsqrt(
        jnp.sum(q_raw.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6
    ).astype(x.dtype)

    # Registers (56..59) query H_goal using blended register + goal_lang query when text is present
    goal_slot = x[:, :, X3_SLOT_GOAL_LANG, :]
    q_goal_raw = jnp.einsum("btd,dhk->bthk", goal_slot, tq_reshaped[:, h_sym:h_long, :]).astype(jnp.float32)
    q_goal_n = q_goal_raw * jax.lax.rsqrt(jnp.sum(q_goal_raw * q_goal_raw, axis=-1, keepdims=True) + 1e-6)
    q_reg_g = q_all[:, :, X3_SLOTS_REGISTER, h_sym:h_long, :].astype(jnp.float32)
    q_blend = 0.40 * q_reg_g + 0.60 * q_goal_n[:, :, None, :, :]
    q_blend = q_blend * jax.lax.rsqrt(jnp.sum(q_blend * q_blend, axis=-1, keepdims=True) + 1e-6)
    q_use = jnp.where(txt_present[:, :, None, None, None] > 0.5, q_blend, q_reg_g)
    q_all = q_all.at[:, :, X3_SLOTS_REGISTER, h_sym:h_long, :].set(q_use.astype(x.dtype))

    u_slot = x @ tslot_u_w
    sym_slot = x[:, :, X3_SLOT_SYM_MEM, :]
    aud_mean = x[:, :, X3_SLOTS_AUDIO, :].mean(axis=2)
    # Language/Goal write stream: pure SLOT_GOAL_LANG when text present, else activity-gated audio!
    tp = txt_present[:, :, None].astype(x.dtype)
    write_src_goal = tp * goal_slot + (1.0 - tp) * aud_mean
    write_src_phys = (
        0.55 * x[:, :, X3_SLOTS_REGISTER, :].mean(axis=2)
        + 0.25 * x[:, :, X3_SLOTS_MOVEMENT, :].mean(axis=2)
        + 0.20 * x[:, :, X3_SLOTS_VISUAL, :].mean(axis=2)
    )

    w_sym_g = jnp.concatenate([tgates_w[:, :h_sym], tgates_w[:, h : h + h_sym]], axis=-1)
    b_sym_g = jnp.concatenate([tgates_b[:h_sym], tgates_b[h : h + h_sym]], axis=-1)
    g_sym = (sym_slot @ w_sym_g).astype(jnp.float32) + b_sym_g[None, None, :]

    w_goal_g = jnp.concatenate([tgates_w[:, h_sym:h_long], tgates_w[:, h + h_sym : h + h_long]], axis=-1)
    b_goal_g = jnp.concatenate([tgates_b[h_sym:h_long], tgates_b[h + h_sym : h + h_long]], axis=-1)
    g_goal = (write_src_goal @ w_goal_g).astype(jnp.float32) + b_goal_g[None, None, :]

    w_phys_g = jnp.concatenate(
        [tgates_w[:, h_long:h], tgates_w[:, h + h_long : 2 * h], tgates_w[:, 2 * h :]], axis=-1
    )
    b_phys_g = jnp.concatenate(
        [tgates_b[h_long:h], tgates_b[h + h_long : 2 * h], tgates_b[2 * h :]], axis=-1
    )
    g_phys = (write_src_phys @ w_phys_g).astype(jnp.float32) + b_phys_g[None, None, :]

    # Exact null-tick freeze on H_sym (`sym_write_active == 0` -> alpha=1, beta=0)
    # AND learned acoustic-activity gate modulation on H_goal when text is absent (`txt_present == 0`)!
    sym_act_f = sym_write_active[:, :, None].astype(jnp.float32)
    alpha_sym = jnp.where(sym_act_f > 0.5, jax.nn.sigmoid(g_sym[:, :, :h_sym]), 1.0)
    beta_sym = jnp.where(sym_act_f > 0.5, jax.nn.sigmoid(g_sym[:, :, h_sym:]), 0.0)

    goal_act_gate = jnp.where(txt_present[:, :, None] > 0.5, 1.0, beta_aud)
    beta_goal_raw = jax.nn.sigmoid(g_goal[:, :, h_goal:]) * goal_act_gate
    alpha_goal_raw = 1.0 - (1.0 - jax.nn.sigmoid(g_goal[:, :, :h_goal])) * goal_act_gate

    alpha = jnp.concatenate([alpha_sym, alpha_goal_raw, jax.nn.sigmoid(g_phys[:, :, :h_phys])], axis=-1)
    beta = jnp.concatenate([beta_sym, beta_goal_raw, jax.nn.sigmoid(g_phys[:, :, h_phys : 2 * h_phys])], axis=-1)
    gamma_raw = jnp.mean(g_phys[:, :, 2 * h_phys :], axis=-1, keepdims=True)  # [B, T, 1]

    kv_w = tkv_write_w.reshape(d, 2, h, dh)
    kv_sym = jnp.einsum("btd,dchk->btchk", sym_slot, kv_w[:, :, :h_sym, :])
    kv_goal = jnp.einsum("btd,dchk->btchk", write_src_goal, kv_w[:, :, h_sym:h_long, :])
    kv_phys = jnp.einsum("btd,dchk->btchk", write_src_phys, kv_w[:, :, h_long:, :])
    kv_all = jnp.concatenate([kv_sym, kv_goal, kv_phys], axis=3)
    k_w = kv_all[:, :, 0].astype(jnp.float32)
    v_w = kv_all[:, :, 1].astype(jnp.float32)
    k_w = k_w * jax.lax.rsqrt(jnp.sum(k_w * k_w, axis=-1, keepdims=True) + 1e-6)

    gamma_sens = jax.nn.sigmoid(gamma_raw - 2.5)
    gamma_reg = jax.nn.sigmoid(gamma_raw)

    # Zero-lag lead-lag slot weights: sensory fast (0.82), sym_mem & goal_lang exact (1.0), registers smoothed (0.20)
    slot_u_w = jnp.concatenate(
        [
            jnp.full((54,), 0.82, dtype=jnp.float32),
            jnp.array([1.0, 1.0], dtype=jnp.float32),
            jnp.full((4,), 0.20, dtype=jnp.float32),
        ]
    )[None, :, None]
    slot_c_w = 1.0 - slot_u_w
    slot_gain = jnp.concatenate(
        [
            jnp.full((48,), 0.12, dtype=jnp.float32),
            jnp.full((3,), 0.35, dtype=jnp.float32),
            jnp.full((3,), 0.25, dtype=jnp.float32),
            jnp.array([1.0, 0.85], dtype=jnp.float32),
            jnp.full((4,), 1.0, dtype=jnp.float32),
        ]
    )[None, :, None]

    if spec.arch == "diagonal_ssm_x3":
        if prev_state is None:
            s_init = jnp.zeros((b, h, dh), dtype=jnp.float32)
            c_init = jnp.zeros((b, s, d), dtype=jnp.float32)
        else:
            s_init, c_init = prev_state[0].astype(jnp.float32), prev_state[1].astype(jnp.float32)

        def scan_ssm(carry, inp):
            s_prev, c_prev = carry
            q_t, k_t, v_t, a_t, b_t, gs_t, gr_t, u_t, ok_t = inp
            assoc_h = q_t.astype(jnp.float32) * s_prev[:, None, :, :]
            # Strict subspace readout partitioning:
            #   - SLOT_SYM_MEM (54) reads ONLY H_sym
            #   - Register 56 (object/color latch) reads H_obj + H_phys (zero H_dir second-utterance interference!)
            #   - Registers 57..58 (spatial direction/goal latch) read H_dir + H_phys (zero H_obj interference!)
            assoc_h = assoc_h.at[:, X3_SLOT_SYM_MEM, h_sym:, :].set(0.0)
            assoc_h = assoc_h.at[:, :X3_SLOT_SYM_MEM, :h_sym, :].set(0.0)
            assoc_h = assoc_h.at[:, X3_SLOT_SYM_MEM + 1 :, :h_sym, :].set(0.0)
            assoc_h = assoc_h.at[:, 56, h_obj_end:h_long, :].set(0.0)
            assoc_h = assoc_h.at[:, 57:59, h_sym:h_obj_end, :].set(0.0)
            assoc_out = assoc_h.reshape(b, s, d)

            s_cand = a_t[:, :, None] * s_prev + b_t[:, :, None] * (jax.nn.tanh(k_t * (dh**0.5)) * v_t)
            g_sl = jnp.concatenate(
                [jnp.broadcast_to(gs_t[:, None, :], (b, 56, 1)), jnp.broadcast_to(gr_t[:, None, :], (b, 4, 1))],
                axis=1,
            )
            u_f32 = u_t.astype(jnp.float32)
            c_cand = g_sl * c_prev + (1.0 - g_sl) * u_f32
            c_cand = c_cand.at[:, X3_SLOT_SYM_MEM, :].set(u_f32[:, X3_SLOT_SYM_MEM, :])
            c_cand = c_cand.at[:, X3_SLOT_GOAL_LANG, :].set(u_f32[:, X3_SLOT_GOAL_LANG, :])
            s_next = jnp.where(ok_t[:, None, None], s_cand, s_prev)
            c_next = jnp.where(ok_t[:, None, None], c_cand, c_prev)
            comb = (slot_gain * assoc_out + slot_u_w * u_f32 + slot_c_w * c_next).astype(x.dtype)
            return (s_next, c_next), comb

        (s_fin, c_fin), seq_out = jax.lax.scan(
            scan_ssm,
            (s_init, c_init),
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
        return jnp.swapaxes(seq_out, 0, 1) @ to_w, (s_fin, c_fin)

    if prev_state is None:
        m_init = jnp.zeros((b, h, dh, dh), dtype=jnp.float32)
        c_init = jnp.zeros((b, s, d), dtype=jnp.float32)
    else:
        m_init, c_init = prev_state[0].astype(jnp.float32), prev_state[1].astype(jnp.float32)

    def scan_delta(carry, inp):
        m_prev, c_prev = carry
        q_t, k_t, v_t, a_t, b_t, gs_t, gr_t, u_t, ok_t = inp
        q_sc = q_t.astype(jnp.float32) * (dh**0.5)
        assoc_h = jnp.einsum("bhij,bshj->bshi", m_prev, q_sc)
        assoc_h = assoc_h.at[:, X3_SLOT_SYM_MEM, h_sym:, :].set(0.0)
        assoc_h = assoc_h.at[:, :X3_SLOT_SYM_MEM, :h_sym, :].set(0.0)
        assoc_h = assoc_h.at[:, X3_SLOT_SYM_MEM + 1 :, :h_sym, :].set(0.0)
        assoc_h = assoc_h.at[:, 56, h_obj_end:h_long, :].set(0.0)
        assoc_h = assoc_h.at[:, 57:59, h_sym:h_obj_end, :].set(0.0)
        assoc_out = assoc_h.reshape(b, s, d)

        v_pred = jnp.einsum("bhij,bhj->bhi", m_prev, k_t)
        delta_v = v_t - a_t[:, :, None] * v_pred
        m_cand = a_t[:, :, None, None] * m_prev + b_t[:, :, None, None] * jnp.einsum("bhi,bhj->bhij", delta_v, k_t)

        g_sl = jnp.concatenate(
            [jnp.broadcast_to(gs_t[:, None, :], (b, 56, 1)), jnp.broadcast_to(gr_t[:, None, :], (b, 4, 1))],
            axis=1,
        )
        u_f32 = u_t.astype(jnp.float32)
        c_cand = g_sl * c_prev + (1.0 - g_sl) * u_f32
        c_cand = c_cand.at[:, X3_SLOT_SYM_MEM, :].set(u_f32[:, X3_SLOT_SYM_MEM, :])
        c_cand = c_cand.at[:, X3_SLOT_GOAL_LANG, :].set(u_f32[:, X3_SLOT_GOAL_LANG, :])
        m_next = jnp.where(ok_t[:, None, None, None], m_cand, m_prev)
        c_next = jnp.where(ok_t[:, None, None], c_cand, c_prev)
        comb = (slot_gain * assoc_out + slot_u_w * u_f32 + slot_c_w * c_next).astype(x.dtype)
        return (m_next, c_next), comb

    (m_fin, c_fin), seq_out = jax.lax.scan(
        scan_delta,
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
    return jnp.swapaxes(seq_out, 0, 1) @ to_w, (m_fin, c_fin)


def _run_x3_block(
    x: jax.Array,
    layer_idx: int,
    params: dict[str, jax.Array],
    spec: X3Spec,
    valid: jax.Array,
    offset: jax.Array | int,
    prev_mem: Any,
    sym_write_active: jax.Array,
    txt_present: jax.Array,
    beta_aud: jax.Array,
) -> tuple[jax.Array, Any]:
    dtype = getattr(jnp, spec.dtype)
    pfx = f"L{layer_idx}_"
    z_s = rms_norm(x, params[pfx + "snorm"])
    x = x + _spatial_attn_x3(
        z_s,
        params[pfx + "sqkv"].astype(dtype),
        params[pfx + "so"].astype(dtype),
        spec.n_heads,
        spec.head_dim,
    )

    z_t = rms_norm(x, params[pfx + "tnorm"])
    is_attn = (spec.arch == "sliding_window_gqa_x3") or ((layer_idx + 1) % 4 == 0)
    if is_attn:
        win = spec.gqa_window if spec.arch == "sliding_window_gqa_x3" else spec.local_attn_window
        h_kv = spec.n_kv_heads_gqa if spec.arch == "sliding_window_gqa_x3" else spec.n_heads
        t_out, mem_out = _causal_rope_temporal_attn(
            z_t,
            params[pfx + "tqkv"].astype(dtype),
            params[pfx + "to"].astype(dtype),
            valid,
            spec.n_heads,
            spec.head_dim,
            win,
            offset=offset,
            prev_cache=prev_mem,
            n_kv_heads=h_kv,
        )
    else:
        t_out, mem_out = _recurrent_temporal_x3(
            z_t,
            params[pfx + "tq"].astype(dtype),
            params[pfx + "tslot_u"].astype(dtype),
            params[pfx + "tkv_write"].astype(dtype),
            params[pfx + "tgates_w"].astype(dtype),
            params[pfx + "tgates_b"],
            params[pfx + "to"].astype(dtype),
            valid,
            spec,
            sym_write_active=sym_write_active,
            txt_present=txt_present,
            beta_aud=beta_aud,
            prev_state=prev_mem,
        )
    x = x + t_out

    z_m = rms_norm(x, params[pfx + "mnorm"])
    g1 = jax.nn.silu(z_m @ params[pfx + "mw1"].astype(dtype))
    g2 = z_m @ params[pfx + "mw2"].astype(dtype)
    x = x + (g1 * g2) @ params[pfx + "mw3"].astype(dtype)
    return x, mem_out


def forward_x3_world_model(
    params: dict[str, jax.Array],
    spec: X3Spec,
    batch: dict[str, jax.Array],
    streaming_carry: dict[str, Any] | None = None,
    *,
    skip_visual_conv: bool = False,
    precomputed_visual: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    sysid_mode: str = "normal",
    sysid_swap_perm: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Causal forward pass of SLWM-X3-480p returning `(z_top, fine_300, beta_aud, next_streaming_carry)`."""
    dtype = getattr(jnp, spec.dtype)
    b, t = batch["audio"].shape[:2] if (skip_visual_conv or precomputed_visual is not None) else batch["video"].shape[:2]
    valid = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(bool)
    vf = valid.astype(jnp.float32)

    mm_raw = batch.get("modality_mask", jnp.ones((b, t, 4), dtype=jnp.float32)).astype(jnp.float32)
    mm = mm_raw * vf[:, :, None]
    aud_m = mm[:, :, 0]
    vis_m = jnp.zeros((b, t), dtype=jnp.float32) if skip_visual_conv else mm[:, :, 1]
    mov_m = mm[:, :, 2]
    txt_m = mm[:, :, 3]

    prev_aud_2 = None if streaming_carry is None else streaming_carry["prev_aud_2"]
    if skip_visual_conv:
        v_macro = jnp.zeros((b, t, 48, spec.d_model), dtype=dtype)
        fine_300 = jnp.zeros((b, t, 300, spec.d_model), dtype=dtype)
        vis_pose4 = jnp.zeros((b, t, 4), dtype=jnp.float32)
    elif precomputed_visual is not None:
        vm_in, f3_in, vp_in = precomputed_visual
        v_macro = vm_in.astype(dtype) * vis_m[:, :, None, None].astype(dtype)
        fine_300 = f3_in.astype(dtype) * vis_m[:, :, None, None].astype(dtype)
        vis_pose4 = vp_in.astype(jnp.float32) * vis_m[:, :, None]
    else:
        v_macro, fine_300, vis_pose4 = encode_480p_visual_stem(params, spec, batch["video"], vis_m)
    a_slots, beta_aud, next_aud_2 = encode_causal_audio_conv(params, spec, batch["audio"], aud_m, prev_aud_2=prev_aud_2)

    # Movement / proprioception (2 slots + embodiment ID embedding)
    mov_clean = batch["movement"].astype(jnp.float32) * mov_m[:, :, None]
    emb_id = jnp.clip(batch.get("embodiment_id", jnp.zeros((b,), dtype=jnp.int32)), 0, spec.num_embodiments - 1)
    emb_vec = params["embodiment_emb"].astype(dtype)[emb_id][:, None, None, :]
    m_slots = (mov_clean.astype(dtype) @ params["mov_w"].astype(dtype)).reshape(b, t, 2, spec.d_model)
    m_slots = (m_slots + emb_vec) * mov_m[:, :, None, None].astype(dtype)

    # Action conditioning slot
    act_in = batch["action"].astype(dtype) * vf[:, :, None].astype(dtype)
    u_slot = (act_in @ params["act_w"].astype(dtype))[:, :, None, :]

    # Dedicated SLOT_SYM_MEM (54): discrete associative [32..95] + query=2 ONLY!
    sym_ids = jnp.clip(batch.get("sym_tokens", jnp.zeros((b, t, 4), dtype=jnp.int32)), 0, spec.vocab_size - 1)
    sym_valid = (sym_ids > 0).astype(dtype)[..., None]
    role4 = _subtoken_role_basis_4(spec.d_model, dtype)[None, None, :, :]
    sym_emb = params["text_emb"].astype(dtype)[sym_ids] * role4 * sym_valid
    sym_tok = (sym_emb.sum(axis=2) @ params["sym_proj_w"].astype(dtype))[:, :, None, :] * vf[:, :, None, None].astype(dtype)
    # Write is active ONLY when a value token (slot 2 > 0, i.e., write/overwrite, NOT query where slot 3 == 2) is present!
    sym_write_active = (sym_ids[:, :, 2] > 0) & valid

    # Dedicated SLOT_GOAL_LANG (55): natural language subtask/goal/transcript tokens (stop_grad on [32..95] rows so text_emb[32..95] stays 100% pure!)
    lang_ids = jnp.clip(batch.get("lang_tokens", jnp.zeros((b, t, 4), dtype=jnp.int32)), 0, spec.vocab_size - 1)
    lang_valid = (lang_ids > 0).astype(dtype)[..., None]
    te_full = params["text_emb"].astype(dtype)
    te_sg = jax.lax.stop_gradient(te_full)
    is_assoc_row = (jnp.arange(spec.vocab_size) >= 32) & (jnp.arange(spec.vocab_size) < 96)
    te_for_lang = jnp.where(is_assoc_row[:, None], te_sg, te_full)
    lang_emb = te_for_lang[lang_ids] * role4 * lang_valid
    lang_tok = (lang_emb.sum(axis=2) @ params["lang_proj_w"].astype(dtype))[:, :, None, :] * txt_m[:, :, None, None].astype(dtype)

    # Causal online system-ID register input (`SLOT_SYSID = 59`): [prev_action, d_vis_pose4, d_mov]
    if streaming_carry is None:
        prev_vp0 = jnp.zeros((b, 1, 4), dtype=jnp.float32)
        prev_mv0 = jnp.zeros((b, 1, spec.movement_dim), dtype=jnp.float32)
        has_prev0 = jnp.zeros((b, 1), dtype=jnp.float32)
    else:
        prev_vp0 = streaming_carry["prev_vis_pose4"][:, None, :]
        prev_mv0 = streaming_carry["prev_mov"][:, None, :]
        has_prev0 = streaming_carry["has_prev_tick"].astype(jnp.float32)[:, None]

    vp_prev_seq = jnp.concatenate([prev_vp0, vis_pose4[:, :-1, :]], axis=1)
    mv_prev_seq = jnp.concatenate([prev_mv0, mov_clean[:, :-1, :]], axis=1)
    has_prev_seq = jnp.concatenate([has_prev0, vf[:, :-1]], axis=1) * vf
    d_vp = (vis_pose4 - vp_prev_seq) * has_prev_seq[:, :, None] * vis_m[:, :, None]
    d_mv = (mov_clean - mv_prev_seq) * has_prev_seq[:, :, None] * mov_m[:, :, None]
    sysid_feat = jnp.concatenate([batch["action"].astype(jnp.float32) * vf[:, :, None], d_vp, d_mv], axis=-1)
    if sysid_mode == "zeroed":
        sysid_feat = jnp.zeros_like(sysid_feat)
    elif sysid_mode == "swapped":
        if sysid_swap_perm is not None:
            sysid_feat = sysid_feat[sysid_swap_perm]
        else:
            sysid_feat = jnp.roll(sysid_feat, shift=max(1, b // 4), axis=0)
    elif sysid_mode != "normal":
        raise ValueError(f"Unsupported sysid_mode={sysid_mode!r}; expected 'normal', 'zeroed', or 'swapped'")
    sysid_delta = (sysid_feat.astype(dtype) @ params["sysid_in_w"].astype(dtype))[:, :, None, :]

    mod_ctx = (mm.astype(dtype) @ params["mod_present_emb"].astype(dtype))[:, :, None, :]
    reg_base = jnp.broadcast_to(params["reg_seeds"].astype(dtype)[None, None, :, :], (b, t, 4, spec.d_model)) + mod_ctx
    reg_slots = jnp.concatenate([reg_base[:, :, :3, :], reg_base[:, :, 3:4, :] + sysid_delta], axis=2)

    x = jnp.concatenate([v_macro, a_slots, m_slots, u_slot, sym_tok, lang_tok, reg_slots], axis=2)
    x = x + params["slot_emb"].astype(dtype)[None, None, :, :]

    offset = jnp.int32(0) if streaming_carry is None else streaming_carry["offset"]
    prev_layers = (None,) * spec.n_layers if streaming_carry is None else streaming_carry["layers"]
    new_layers: list[Any] = []

    for li in range(spec.n_layers):
        if spec.grad_checkpoint and streaming_carry is None:
            def _ckpt_fn(x_in, swa, tp, ba, idx=li):
                y, _ = _run_x3_block(x_in, idx, params, spec, valid, 0, None, swa, tp, ba)
                return y
            x = jax.checkpoint(_ckpt_fn)(x, sym_write_active, txt_m, beta_aud)
            new_layers.append(None)
        else:
            x, m_out = _run_x3_block(
                x, li, params, spec, valid, offset, prev_layers[li], sym_write_active, txt_m, beta_aud
            )
            new_layers.append(m_out)

    z_top = rms_norm(x, params["final_norm"])
    # Update carry only on valid ticks (preserve last valid tick for padded tails)
    last_valid_vp = jnp.where(valid[:, -1:], vis_pose4[:, -1, :], prev_vp0[:, 0, :])
    last_valid_mv = jnp.where(valid[:, -1:], mov_clean[:, -1, :], prev_mv0[:, 0, :])
    next_has_prev = (has_prev0[:, 0] > 0.5) | jnp.any(valid, axis=1)
    next_offset = offset + jnp.sum(valid[0].astype(jnp.int32))
    next_carry = {
        "offset": next_offset,
        "layers": tuple(new_layers),
        "prev_aud_2": next_aud_2,
        "prev_vis_pose4": last_valid_vp,
        "prev_mov": last_valid_mv,
        "has_prev_tick": next_has_prev,
        "prev_z_next": jnp.zeros((b, 1, spec.slots_per_tick, spec.d_model), dtype=dtype)
        if streaming_carry is None
        else streaming_carry["prev_z_next"],
        "prev_log_var": jnp.zeros((b, 1, spec.slots_per_tick, 1), dtype=jnp.float32)
        if streaming_carry is None
        else streaming_carry["prev_log_var"],
    }
    return z_top, fine_300, beta_aud, next_carry


def fork_x3_streams(z_top: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Fork isolated `z_sym` (`SLOT_SYM_MEM = 54`) and `z_task` (`all other 59 slots`, stop_grad on slot 54)."""
    sym = z_top[:, :, X3_SLOT_SYM_MEM, :]
    z_task = z_top.at[:, :, X3_SLOT_SYM_MEM, :].set(jax.lax.stop_gradient(sym))
    return sym, z_task


def predict_x3_causal_action_from_ztask(
    params: dict[str, jax.Array],
    z_task: jax.Array,
    batch: dict[str, jax.Array],
) -> jax.Array:
    """Compute causal `16-D` motor action proposal directly from current `z_task` (`0` future action leak)."""
    zf = z_task.astype(jnp.float32)
    b, t = zf.shape[:2]
    vf = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(jnp.float32)
    mm = batch.get("modality_mask", jnp.ones((b, t, 4), dtype=jnp.float32)).astype(jnp.float32) * vf[:, :, None]
    aud_m, vis_m, mov_m = mm[:, :, 0], mm[:, :, 1], mm[:, :, 2]
    vis_sum = jnp.mean(zf[:, :, X3_SLOTS_VISUAL, :] * vis_m[:, :, None, None], axis=2)
    aud_sum = jnp.mean(zf[:, :, X3_SLOTS_AUDIO, :] * aud_m[:, :, None, None], axis=2)
    mov_sum = jnp.mean(zf[:, :, X3_SLOTS_MOVEMENT, :] * mov_m[:, :, None, None], axis=2)
    reg_goal_sum = jnp.mean(zf[:, :, 56:59, :], axis=2)
    sysid_lat = zf[:, :, X3_SLOT_SYSID, :]
    ctrl_in = 0.32 * vis_sum + 0.18 * aud_sum + 0.20 * mov_sum + 0.20 * reg_goal_sum + 0.10 * sysid_lat
    ctrl_h = jax.nn.silu(ctrl_in @ params["ctrl_trunk_w"].astype(jnp.float32) + params["ctrl_trunk_b"].astype(jnp.float32))
    return jnp.tanh(ctrl_h @ params["head_act_w"].astype(jnp.float32) + params["head_act_b"].astype(jnp.float32))


def predict_x3_latent_step(
    params: dict[str, jax.Array],
    spec: X3Spec,
    z_task: jax.Array,
    next_action: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Action-conditioned Residual-JEPA latent step predictor + heteroscedastic log-variance."""
    dtype = getattr(jnp, spec.dtype)
    b, t, s, d = z_task.shape
    u_broad = jnp.broadcast_to(next_action.astype(dtype)[:, :, None, :], (b, t, s, spec.action_dim))
    inp = jnp.concatenate([z_task, u_broad], axis=-1)
    h = jax.nn.silu(inp @ params["jepa_w1"].astype(dtype))
    dz = h @ params["jepa_w2"].astype(dtype)
    # Keep SLOT_SYM_MEM invariant under physical latent rollouts
    dz = dz.at[:, :, X3_SLOT_SYM_MEM, :].set(0.0)
    z_next = z_task + dz
    lv = jnp.clip(
        z_next.astype(jnp.float32) @ params["unc_w"].astype(jnp.float32) + params["unc_b"].astype(jnp.float32),
        -5.0,
        3.0,
    )
    return z_next, lv


def _pixel_shuffle_2d(x: jax.Array, r: int) -> jax.Array:
    """Depth-to-space upsampling `[N, H, W, C * r * r] -> [N, H * r, W * r, C]`."""
    n, h, w, c_rr = x.shape
    c = c_rr // (r * r)
    xr = x.reshape(n, h, w, r, r, c)
    return jnp.transpose(xr, (0, 1, 3, 2, 4, 5)).reshape(n, h * r, w * r, c)


def decode_480p_video_from_latents(
    params: dict[str, jax.Array],
    spec: X3Spec,
    z_vis_slots: jax.Array,
    reg_ctx: jax.Array,
    context_frame_480p: jax.Array | None = None,
) -> jax.Array:
    """Synthesize native `480x640x3` (`480p`) RGB frames in `[-1, 1]` from `6x8` latent slots + register directive."""
    dtype = getattr(jnp, spec.dtype)
    b, t, _, d = z_vis_slots.shape
    cond_slots = z_vis_slots.astype(dtype) + reg_ctx.astype(dtype)[:, :, None, :]  # [B, T, 48, D]
    # 6x8xD -> 30x40x96
    up1 = jax.nn.silu(
        (cond_slots @ params["vid_dec_up1_w"].astype(dtype)).reshape(b * t, 6, 8, 5, 5, 96)
    )
    feat_30x40 = jnp.transpose(up1, (0, 1, 3, 2, 4, 5)).reshape(b * t, 30, 40, 96) + params["vid_dec_up1_b"].astype(dtype)

    # 30x40x96 -> 120x160x32
    c2 = (
        jax.lax.conv_general_dilated(
            feat_30x40,
            params["vid_dec_up2_w"].astype(dtype),
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vid_dec_up2_b"].astype(dtype)
    )
    feat_120x160 = jax.nn.silu(_pixel_shuffle_2d(c2, 4))

    # 120x160x32 -> 480x640x3
    c3 = (
        jax.lax.conv_general_dilated(
            feat_120x160,
            params["vid_dec_rgb_w"].astype(dtype),
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + params["vid_dec_rgb_b"].astype(dtype)
    )
    delta_rgb = jnp.tanh(_pixel_shuffle_2d(c3, 4).astype(jnp.float32)).reshape(b, t, 480, 640, 3)

    if context_frame_480p is None:
        return delta_rgb
    ctx_in = apply_240p_bottleneck_if_needed(context_frame_480p, spec.resolution_mode)
    ctx_norm = ctx_in.astype(jnp.float32) / 127.5 - 1.0
    if ctx_norm.shape[1] == 1 and t > 1:
        ctx_norm = jnp.broadcast_to(ctx_norm, (b, t, 480, 640, 3))
    alpha_ctx = 0.65 * jax.nn.sigmoid(
        reg_ctx.astype(jnp.float32) @ params["vid_dec_ctx_gate_w"].astype(jnp.float32)
    )[:, :, None, None, :]
    return jnp.clip(alpha_ctx * ctx_norm + (1.0 - 0.35 * alpha_ctx) * delta_rgb, -1.0, 1.0)


def decode_x3_target_block_xy(
    params: dict[str, jax.Array],
    z_task: jax.Array,
    fine_300: jax.Array,
) -> jax.Array:
    """Equivariant 2-block scene localization over the `15x20` (`300`-cell) `480p` fine visual grid (`0` blank clamp)."""
    zf = z_task.astype(jnp.float32)
    spw = params["spatial_pose_w"].astype(jnp.float32)
    spb = params["spatial_pose_b"].astype(jnp.float32)

    # Read strictly from dedicated object/color latch register 56 (`H_obj`), with zero trailing-tick `aud_sum` dilution!
    lang_ctx = zf[..., 56, :]
    q_lang = lang_ctx @ spw[:, 4:6] + spb[4:6]  # [..., 2]

    fc = fine_300.astype(jnp.float32)
    fc_centered = fc - jnp.mean(fc, axis=-2, keepdims=True)
    k_vis = fc_centered @ spw[:, 6:8]           # [..., 300, 2]
    logits = jnp.einsum("...sr,...r->...s", k_vis, q_lang)
    attn = jax.nn.softmax(logits, axis=-1)
    centers = canonical_grid_centers_15x20()
    coarse_xy = jnp.einsum("...s,sc->...c", attn, centers)
    vis_att = jnp.einsum("...s,...sd->...d", attn, fc_centered)
    fine_xy = 0.12 * jnp.tanh(vis_att @ spw[:, 8:10] + spb[8:10])
    return jnp.clip(coarse_xy + fine_xy, -0.96, 0.96)


def decode_x3_any_to_any_heads(
    params: dict[str, jax.Array],
    spec: X3Spec,
    z_sym: jax.Array,
    z_task: jax.Array,
    z_next_pred: jax.Array,
    log_var_next: jax.Array,
    fine_300: jax.Array,
    batch: dict[str, jax.Array],
    *,
    decode_480p_frames: bool = True,
    context_frame_480p: jax.Array | None = None,
    streaming_carry: dict[str, Any] | None = None,
) -> dict[str, jax.Array]:
    """Decode all 6 Any-to-Any modality composition & epistemic world-model heads."""
    zf = z_task.astype(jnp.float32)
    znf = z_next_pred.astype(jnp.float32)
    b, t, s, _ = zf.shape
    vf = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(jnp.float32)
    mm = batch.get("modality_mask", jnp.ones((b, t, 4), dtype=jnp.float32)).astype(jnp.float32) * vf[:, :, None]
    aud_m, vis_m, mov_m, txt_m = mm[:, :, 0], mm[:, :, 1], mm[:, :, 2], mm[:, :, 3]

    vis_sum = jnp.mean(zf[:, :, X3_SLOTS_VISUAL, :] * vis_m[:, :, None, None], axis=2)
    vis_next_sum = jnp.mean(znf[:, :, X3_SLOTS_VISUAL, :], axis=2)
    aud_sum = jnp.mean(zf[:, :, X3_SLOTS_AUDIO, :] * aud_m[:, :, None, None], axis=2)
    mov_sum = jnp.mean(zf[:, :, X3_SLOTS_MOVEMENT, :] * mov_m[:, :, None, None], axis=2)
    mov_next_sum = jnp.mean(znf[:, :, X3_SLOTS_MOVEMENT, :], axis=2)
    reg_obj_lat = zf[:, :, 56, :]
    reg_dir_sum = jnp.mean(zf[:, :, 57:59, :], axis=2)
    reg_goal_sum = jnp.mean(zf[:, :, 56:59, :], axis=2)
    sysid_lat = zf[:, :, X3_SLOT_SYSID, :]
    reg_all_sum = jnp.mean(zf[:, :, X3_SLOTS_REGISTER, :], axis=2)

    # 1. Associative Recall Readout (`SLOT_SYM_MEM` -> `[64..95]` value subspace ONLY)
    sym_h = z_sym.astype(jnp.float32) @ params["head_sym_w"].astype(jnp.float32)
    assoc_val_w = params["text_emb"].astype(jnp.float32)[X3_ASSOC_VAL_OFFSET : X3_ASSOC_VAL_OFFSET + X3_ASSOC_NUM_VALS].T
    sym_val_logits = (sym_h @ assoc_val_w) * (spec.d_model**-0.5)  # [B, T, 32]

    # 2. Direction (1): `Video / Audio -> Text` subword generation head (`SLOT_GOAL_LANG` excluded when generating!)
    mm_gen_txt = 0.50 * vis_sum + 0.30 * aud_sum + 0.20 * reg_goal_sum
    lang_dec_h = jax.nn.silu(
        mm_gen_txt @ params["head_lang_dec_w"].astype(jnp.float32) + params["head_lang_dec_b"].astype(jnp.float32)
    )
    te_non_assoc = jax.lax.stop_gradient(params["text_emb"].astype(jnp.float32))
    pred_lang_logits = (lang_dec_h @ te_non_assoc.T) * (spec.d_model**-0.5)  # [B, T, V]

    # 3. Direction (2): `Text (+ Context + Action) -> 480p Video` synthesis
    pred_video_480p = (
        decode_480p_video_from_latents(
            params, spec, z_next_pred[:, :, X3_SLOTS_VISUAL, :], reg_all_sum, context_frame_480p=context_frame_480p
        )
        if decode_480p_frames
        else jnp.zeros((b, 0, 480, 640, 3), dtype=jnp.float32)
    )

    # 4. Direction (3): `Video (+ Action) -> 64-band Causal Audio` synthesis (`SLOTS_AUDIO` excluded so zero input copy!)
    va_ctx = 0.55 * vis_sum + 0.20 * vis_next_sum + 0.15 * zf[:, :, X3_SLOT_ACTION, :] + 0.10 * reg_all_sum
    aud_h = jax.nn.silu(va_ctx @ params["aud_dec_w1"].astype(jnp.float32) + params["aud_dec_b1"].astype(jnp.float32))
    pred_audio_mel = aud_h @ params["aud_dec_w2"].astype(jnp.float32) + params["aud_dec_b2"].astype(jnp.float32)

    # 5. Directions (4) & (5): `Text/Video -> 32D Movement` & `Audio+Video+Text+SysID -> 16D Continuous Robot Actions`
    ctrl_in = 0.32 * vis_sum + 0.18 * aud_sum + 0.20 * mov_sum + 0.20 * reg_goal_sum + 0.10 * sysid_lat
    ctrl_h = jax.nn.silu(ctrl_in @ params["ctrl_trunk_w"].astype(jnp.float32) + params["ctrl_trunk_b"].astype(jnp.float32))
    mov_flow_ctx = 0.35 * vis_next_sum + 0.35 * reg_goal_sum + 0.20 * mov_next_sum + 0.10 * sysid_lat
    pred_mov_next = mov_flow_ctx @ params["head_mov_next_w"].astype(jnp.float32) + params["head_mov_next_b"].astype(jnp.float32)
    pred_mov_step4 = ctrl_h @ params["head_mov_step4_w"].astype(jnp.float32) + params["head_mov_step4_b"].astype(jnp.float32)
    pred_action = jnp.tanh(ctrl_h @ params["head_act_w"].astype(jnp.float32) + params["head_act_b"].astype(jnp.float32))
    value_pred = (ctrl_h @ params["head_val_w"].astype(jnp.float32) + params["head_val_b"].astype(jnp.float32))[..., 0]
    pred_goal_xy = jnp.tanh(reg_dir_sum @ params["head_goal_xy_w"].astype(jnp.float32) + params["head_goal_xy_b"].astype(jnp.float32))
    _ = reg_obj_lat
    contact_logits = (
        0.35 * vis_sum + 0.25 * aud_sum + 0.20 * mov_sum + 0.20 * reg_all_sum
    ) @ params["head_contact_w"].astype(jnp.float32) + params["head_contact_b"].astype(jnp.float32)

    # Fine 480p spatial pose & 2-block localization + 8x8 occupancy grid
    centers_300 = canonical_grid_centers_15x20()
    fc_c = fine_300.astype(jnp.float32) - jnp.mean(fine_300.astype(jnp.float32), axis=2, keepdims=True)
    heat2 = fc_c @ params["spatial_pose_w"].astype(jnp.float32)[:, 0:2]
    vis_ee_xy = jnp.einsum("bts,sc->btc", jax.nn.softmax(heat2[:, :, :, 0], axis=-1), centers_300)
    vis_obj_xy = jnp.einsum("bts,sc->btc", jax.nn.softmax(heat2[:, :, :, 1], axis=-1), centers_300)
    pred_block_xy = decode_x3_target_block_xy(params, z_task, fine_300)
    occ_logits_8x8 = (vis_sum @ params["occ_grid_w"].astype(jnp.float32) + params["occ_grid_b"].astype(jnp.float32)).reshape(b, t, 8, 8)

    # 6. Direction (6): Calibrated Innovation-Gated `PolicyCommitGate` (`commit_logits`)
    if streaming_carry is None:
        prev_zn0 = jnp.zeros((b, 1, s, spec.d_model), dtype=zf.dtype)
        prev_lv0 = jnp.zeros((b, 1, s, 1), dtype=jnp.float32)
        has_p0 = jnp.zeros((b, 1), dtype=jnp.float32)
    else:
        prev_zn0 = streaming_carry["prev_z_next"].astype(jnp.float32)
        prev_lv0 = streaming_carry["prev_log_var"].astype(jnp.float32)
        has_p0 = streaming_carry["has_prev_tick"].astype(jnp.float32)[:, None]

    zn_prev_seq = jnp.concatenate([prev_zn0, znf[:, :-1]], axis=1)
    lv_prev_seq = jnp.concatenate([prev_lv0, log_var_next.astype(jnp.float32)[:, :-1]], axis=1)
    hp_seq = jnp.concatenate([has_p0, vf[:, :-1]], axis=1) * vf
    inv_var = jnp.exp(-lv_prev_seq)
    mahal_vis = jnp.mean(((zf[:, :, X3_SLOTS_VISUAL, :] - zn_prev_seq[:, :, X3_SLOTS_VISUAL, :]) ** 2) * inv_var[:, :, X3_SLOTS_VISUAL, :], axis=(2, 3)) * hp_seq * vis_m
    mahal_mov = jnp.mean(((zf[:, :, X3_SLOTS_MOVEMENT, :] - zn_prev_seq[:, :, X3_SLOTS_MOVEMENT, :]) ** 2) * inv_var[:, :, X3_SLOTS_MOVEMENT, :], axis=(2, 3)) * hp_seq * mov_m
    vis_ee_disc = jnp.sqrt(jnp.sum((vis_ee_xy - batch["movement"].astype(jnp.float32)[:, :, 0:2]) ** 2, axis=-1) + 1e-8) * vis_m * mov_m
    lv_mean = jnp.mean(log_var_next.astype(jnp.float32), axis=(2, 3))
    innov8 = jnp.stack(
        [
            jnp.log1p(mahal_vis) * 1.8,
            jnp.log1p(mahal_mov) * 1.5,
            vis_ee_disc * 4.5,
            lv_mean * 0.5,
            1.0 - vis_m,
            1.0 - aud_m,
            1.0 - mov_m,
            1.0 - txt_m,
        ],
        axis=-1,
    )
    reg_norm = reg_all_sum * jax.lax.rsqrt(jnp.sum(reg_all_sum * reg_all_sum, axis=-1, keepdims=True) + 1e-6)
    commit_in = jnp.concatenate([0.08 * jax.lax.stop_gradient(reg_norm), jax.lax.stop_gradient(innov8)], axis=-1)
    ch = jax.nn.silu(commit_in @ params["commit_w1"].astype(jnp.float32) + params["commit_b1"].astype(jnp.float32))
    commit_logits = ch @ params["commit_w2"].astype(jnp.float32) + params["commit_b2"].astype(jnp.float32)
    anomaly_score = 0.55 * vis_ee_disc + 0.30 * jnp.log1p(mahal_vis) + 0.15 * jax.nn.softplus(commit_logits[..., 1:].max(axis=-1) - commit_logits[..., 0])

    return {
        "sym_val_logits": sym_val_logits,
        "pred_lang_logits": pred_lang_logits,
        "pred_video_480p": pred_video_480p,
        "pred_audio_mel": pred_audio_mel,
        "pred_mov_next": pred_mov_next,
        "pred_mov_step4": pred_mov_step4,
        "pred_action": pred_action,
        "value_pred": value_pred,
        "pred_goal_xy": pred_goal_xy,
        "pred_block_xy": pred_block_xy,
        "vis_ee_xy": vis_ee_xy,
        "vis_obj_xy": vis_obj_xy,
        "occ_logits_8x8": occ_logits_8x8,
        "contact_logits": contact_logits,
        "commit_logits": commit_logits,
        "anomaly_score": anomaly_score,
        "next_prev_z_next": znf[:, -1:].astype(z_task.dtype),
        "next_prev_log_var": log_var_next[:, -1:].astype(jnp.float32),
    }


def mel64_to_waveform_griffin_lim_16khz(
    mel_frames: np.ndarray,
    *,
    sr: int = 16000,
    hop_samples: int = 1600,
    n_fft: int = 2048,
    n_iter: int = 24,
) -> np.ndarray:
    """Synthesize playable `16 kHz` PCM waveform from predicted `64`-band `[60, 7680] Hz` log-power frames."""
    t_frames = int(mel_frames.shape[0])
    if t_frames == 0:
        return np.zeros((0,), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    edges = np.geomspace(60.0, 7680.0, 65)
    # Denormalize `(log10(power) - (-4.523)) / 2.524` -> amplitude spectrum
    log10_p = np.clip(mel_frames.astype(np.float64) * 2.524 - 4.523, -7.5, 1.5)
    band_amp = np.sqrt(np.power(10.0, log10_p))

    mag = np.zeros((t_frames, freqs.shape[0]), dtype=np.float64)
    for bi in range(64):
        idx = np.where((freqs >= edges[bi]) & (freqs < edges[bi + 1]))[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(freqs - 0.5 * (edges[bi] + edges[bi + 1]))))])
        mag[:, idx] = band_amp[:, bi : bi + 1]

    out_len = max(n_fft, (t_frames - 1) * hop_samples + n_fft)
    win = np.hanning(n_fft)
    rng = np.random.default_rng(3001)
    angles = np.exp(2j * np.pi * rng.random(mag.shape))
    spec_complex = mag * angles

    for _ in range(n_iter):
        y = np.zeros((out_len,), dtype=np.float64)
        wsum = np.zeros((out_len,), dtype=np.float64)
        for ti in range(t_frames):
            frame_t = np.fft.irfft(spec_complex[ti], n=n_fft) * win
            i0 = ti * hop_samples
            y[i0 : i0 + n_fft] += frame_t
            wsum[i0 : i0 + n_fft] += win * win
        y = y / np.maximum(wsum, 1e-8)
        for ti in range(t_frames):
            i0 = ti * hop_samples
            stft_t = np.fft.rfft(y[i0 : i0 + n_fft] * win, n=n_fft)
            phase = stft_t / np.maximum(np.abs(stft_t), 1e-8)
            spec_complex[ti] = mag[ti] * phase

    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = 0.88 * (y / peak)
    return y[: t_frames * hop_samples].astype(np.float32)
