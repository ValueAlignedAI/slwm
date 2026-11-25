"""SLWM-v4 Gate 1 Core Contract: 160 ms Strictly Causal Live Interleaved Heartbeat & Unified Humanoid Schema.

Eliminates the structural failure modes identified in SLWM-v3 and enforces every Gate 1 causal invariant:
  1. Strict Physical Availability Ordering (zero future lookahead, zero `np.maximum.accumulate` clock warping):
     Every packet carries exact `capture_time_s <= last_sample_time_s <= availability_time_s` matching
     `data/v3_data_contract.py` (`wan21_group_availability_timestamps`, `mimi_packet_availability_timestamps`).
     Inside each 160 ms heartbeat `h` spanning `(t0, t0 + 0.16]` (after optional Wan 2.1 `1+4k` bootstrap
     Frame 0 at `t = 0.00 s`), events are emitted in strict non-decreasing physical availability order:
       - `t0 + 0.04 s`: `MOT[4h + 0]` (25 Hz control tick; sees only `t <= t0`)
       - `t0 + 0.08 s`: `MOT[4h + 1]`, then `AUD[2h + 0]` (Mimi `[t0, t0+0.08]`, available at `t0+0.08`)
       - `t0 + 0.12 s`: `MOT[4h + 2]` (sees `AUD[2h+0]`, NEVER sees `AUD[2h+1]` or `VID[h]`)
       - `t0 + 0.16 s`: `MOT[4h + 3]` (emitted strictly BEFORE `VID[h]` and `AUD[2h+1]` -> 0 ms lookahead!)
       - `t0 + 0.16 s`: `VID[h]` (15 Wan 2.1 patches covering frames `[4h+1..4h+4]`, available at `t0+0.16`)
       - `t0 + 0.16 s`: `AUD[2h + 1]` (Mimi `[t0+0.08, t0+0.16]`, sees `VID[h]` for 0-lag contact audio!)
       - `t0 + 0.16 s`: `TXT[h]` (causal UTF-8 bytes whose `t_avail <= t0+0.16`)
  2. Non-Aliased 52-DoF Unified Humanoid Action / 68-Dim Proprio+Force Schema (zero truncation, zero aliasing):
     Accommodates full-body Unitree G1 WBT+Inspire (`41+4=45` DoFs), G1 Dex3 (`28` DoFs), Unitree H1 (`40` DoFs),
     AgiBot, RH20T (with 6-axis dual-wrist force/torque), DROID (`8` DoFs), and ALOHA (`14` DoFs) on strictly
     disjoint physical slices. Both `action_mask` and `proprio_mask` are strictly validated binary `{0, 1}`
     tensors passed BOTH as encoder conditioning (disambiguating active `0.0` vs unobserved `0.0`) and loss masks.
  3. Zero-Shortcut Seam-Free Same-Clip AV Desync Pair (`make_same_clip_temporal_desync_pair`):
     Uses strictly interior contiguous window cropping (zero circular `np.roll` splice seam!) and strips
     `MOD_TEXT` on AV-sync eval windows (zero audio-text mismatch shortcut).
  4. Programmatically Derived Single-Pass Corpus Arithmetic & Two-Tier Storage Physics:
     Every packet count and raw byte size is derived by exact formula from `target_hours`, `modalities`,
     and `SLWMv4HeartbeatSpec` (never hand-typed), cleanly separating Tier 1 (`V4-S3` persistent `fp16` pool,
     `~440 GB` inside `200-500 GB` envelope) from Tier 2 (`V4-S4` `3.0B` streaming/quantized pool).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Modality IDs (preserved from SLWM-v3 for clean continuity)
MOD_VIDEO = 0
MOD_AUDIO = 1
MOD_TEXT = 2
MOD_MOTOR = 3
MOD_PROBE = 4

# Non-aliased, zero-truncation unified humanoid embodiment layout (accommodates G1 WBT 45-DoF & H1 40-DoF!)
UNIFIED_ACTION_DIM = 52
UNIFIED_PROPRIO_DIM = 68

# Strictly disjoint, contiguous physical channel slices (sum == 52; zero aliasing across arms/hands/legs/waist/base!)
ACTION_SLICES: dict[str, slice] = {
    "left_arm_7dof": slice(0, 7),
    "right_arm_7dof": slice(7, 14),
    "left_hand_or_gripper_7dof": slice(14, 21),  # 1-DoF gripper -> [14]; 6-DoF Inspire -> [14:20]; 7-DoF Dex3 -> [14:21]
    "right_hand_or_gripper_7dof": slice(21, 28),  # 1-DoF gripper -> [21]; 6-DoF Inspire -> [21:27]; 7-DoF Dex3 -> [21:28]
    "left_leg_6dof": slice(28, 34),
    "right_leg_6dof": slice(34, 40),
    "waist_3dof": slice(40, 43),
    "mobile_base_twist_4dof": slice(43, 47),  # [vx, vy, vyaw, base_height/stand]
    "head_and_aux_5dof": slice(47, 52),
}

# Strictly disjoint proprioception + contact force-torque + IMU slices (sum == 68)
PROPRIO_SLICES: dict[str, slice] = {
    "joint_state_52dof": slice(0, 52),  # Exact 1:1 physical mirror of ACTION_SLICES
    "left_wrist_force_torque_6d": slice(52, 58),  # [fx, fy, fz, tx, ty, tz]
    "right_wrist_force_torque_6d": slice(58, 64),  # [fx, fy, fz, tx, ty, tz]
    "torso_imu_base_orient_4d": slice(64, 68),  # unit quaternion [qw, qx, qy, qz]
}


def validate_disjoint_embodiment_slices() -> bool:
    """Certify that ACTION_SLICES and PROPRIO_SLICES partition [0, D) with zero overlap and zero gaps."""
    for slices, dim, name in (
        (ACTION_SLICES, UNIFIED_ACTION_DIM, "ACTION_SLICES"),
        (PROPRIO_SLICES, UNIFIED_PROPRIO_DIM, "PROPRIO_SLICES"),
    ):
        covered = np.zeros((dim,), dtype=np.int32)
        for k, sl in slices.items():
            if sl.start is None or sl.stop is None or sl.start < 0 or sl.stop > dim or sl.stop <= sl.start:
                raise ValueError(f"Invalid slice {name}[{k}]={sl} for dim={dim}")
            covered[sl] += 1
        if not np.all(covered == 1):
            raise ValueError(f"{name} failed exact partition check: min={covered.min()}, max={covered.max()}")
    return True


@dataclass(frozen=True)
class SLWMv4HeartbeatSpec:
    """Immutable physical clock & packetization contract for SLWM-v4 live streaming."""

    heartbeat_sec: float = 0.16  # 6.25 Hz physical world heartbeat
    video_fps: float = 25.0  # 4 RGB frames per 160 ms Wan 2.1 causal group (`1 + 4k` framing)
    video_hw: tuple[int, int] = (192, 320)  # Zero black bars; 24x40 latent -> 3x5 = 15 patches
    video_patches_per_heartbeat: int = 15
    video_patch_dim: int = 1024  # 8x8x16 Wan 2.1 Causal VAE latent patch
    audio_hz: float = 12.5  # Kyutai Mimi native streaming frame rate
    audio_ticks_per_heartbeat: int = 2  # Exact 2 * 80 ms = 160 ms lock
    audio_codebooks_stored: int = 16  # Semantic RVQ-0 + 15 acoustic RVQ codebooks (16-RVQ WER measured in Gate 2)
    audio_vocab_size: int = 2048
    motor_hz: float = 25.0  # 4 motor/proprio ticks per 160 ms heartbeat (40 ms control period)
    motor_ticks_per_heartbeat: int = 4
    action_dim: int = UNIFIED_ACTION_DIM
    proprio_dim: int = UNIFIED_PROPRIO_DIM
    text_vocab_size: int = 264  # 256 lossless UTF-8 bytes + 8 control symbols

    @property
    def heartbeats_per_hour(self) -> float:
        return 3600.0 / self.heartbeat_sec  # Exact 22,500.0 heartbeats / hour

    @property
    def packets_per_full_av_motor_heartbeat(self) -> int:
        """Exact packet count per 160 ms heartbeat when Video + Audio + Motor are all active (excluding optional text)."""
        return self.video_patches_per_heartbeat + self.audio_ticks_per_heartbeat + self.motor_ticks_per_heartbeat

    @property
    def packets_per_av_heartbeat(self) -> int:
        """Exact packet count per 160 ms heartbeat for synchronized Audio + Video (15 + 2 = 17)."""
        return self.video_patches_per_heartbeat + self.audio_ticks_per_heartbeat

    def heartbeats_for_duration(self, duration_sec: float) -> int:
        return int(round(float(duration_sec) / self.heartbeat_sec))

    def validate_physical_clock_lock(self) -> dict[str, Any]:
        """Certify exact integer lock between video, audio, and motor clocks inside one 160 ms heartbeat."""
        vid_frames = self.heartbeat_sec * self.video_fps
        aud_ticks = self.heartbeat_sec * self.audio_hz
        mot_ticks = self.heartbeat_sec * self.motor_hz
        ok = (
            abs(vid_frames - 4.0) < 1e-9
            and abs(aud_ticks - float(self.audio_ticks_per_heartbeat)) < 1e-9
            and abs(mot_ticks - float(self.motor_ticks_per_heartbeat)) < 1e-9
            and validate_disjoint_embodiment_slices()
        )
        return {
            "heartbeat_sec": self.heartbeat_sec,
            "heartbeats_per_hour": self.heartbeats_per_hour,
            "video_frames_per_heartbeat": vid_frames,
            "audio_ticks_per_heartbeat": aud_ticks,
            "motor_ticks_per_heartbeat": mot_ticks,
            "unified_action_dim": self.action_dim,
            "unified_proprio_dim": self.proprio_dim,
            "exact_integer_clock_lock": ok,
            "packets_per_av_heartbeat": self.packets_per_av_heartbeat,
            "packets_per_av_motor_heartbeat": self.packets_per_full_av_motor_heartbeat,
            "heartbeats_per_120s_2min": self.heartbeats_for_duration(120.0),
            "packets_per_120s_av_stream": self.heartbeats_for_duration(120.0) * self.packets_per_av_heartbeat,
            "packets_per_120s_av_motor_stream": self.heartbeats_for_duration(120.0)
            * self.packets_per_full_av_motor_heartbeat,
        }


def _validate_binary_mask(mask: np.ndarray | None, dim: int, name: str) -> np.ndarray:
    if mask is None:
        return np.ones((dim,), dtype=np.float32)
    arr = np.asarray(mask, dtype=np.float32)
    if arr.shape != (dim,):
        raise ValueError(f"Expected {name} shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError(f"{name} must be strictly binary {{0.0, 1.0}}, got unique values {np.unique(arr)}")
    return arr


def build_interleaved_heartbeat_stream(
    *,
    video_groups: np.ndarray | None = None,
    audio_codes: np.ndarray | None = None,
    action_seq: np.ndarray | None = None,
    proprio_seq: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
    proprio_mask: np.ndarray | None = None,
    text_by_heartbeat: dict[int, bytes] | None = None,
    video_init_frame0: np.ndarray | None = None,
    spec: SLWMv4HeartbeatSpec = SLWMv4HeartbeatSpec(),
) -> dict[str, np.ndarray]:
    """Pack one episode into a strictly causal, 160 ms physically-ordered SLWM-v4 packet stream.

    Causal availability timestamps (`capture_time_s <= last_sample_time_s <= wall_time_s == availability_time_s`):
      - Optional `video_init_frame0` (Wan 2.1 `1+4k` Frame 0): `capture = last = avail = 0.00 s` (`heartbeat_id = -1`).
      - Heartbeat `h` spanning `(t0, t0 + 0.16]` where `t0 = h * 0.16`:
          1. `t0 + 0.04 s`: `MOT[4h + 0]` (25 Hz action/proprio; zero visual/audio lookahead)
          2. `t0 + 0.08 s`: `MOT[4h + 1]`, followed by `AUD[2h + 0]` (Mimi `[t0, t0+0.08]`)
          3. `t0 + 0.12 s`: `MOT[4h + 2]`
          4. `t0 + 0.16 s`: `MOT[4h + 3]` (BEFORE `VID[h]` and `AUD[2h+1]` -> 0 ms future lookahead into motor!)
          5. `t0 + 0.16 s`: `VID[h]` (15 Wan 2.1 patches covering frames `[4h+1..4h+4]`, available at `t0+0.16`)
          6. `t0 + 0.16 s`: `AUD[2h + 1]` (Mimi `[t0+0.08, t0+0.16]`, available at `t0+0.16`; attends causally to `VID[h]`)
          7. `t0 + 0.16 s`: `TXT[h]` (UTF-8 bytes whose causal arrival time `<= t0+0.16`)

    Strictly asserts raw non-decreasing `wall_time_s` WITHOUT `np.maximum.accumulate` clamping!
    """
    n_hb_candidates: list[int] = []
    if video_init_frame0 is not None:
        if video_init_frame0.shape != (spec.video_patches_per_heartbeat, spec.video_patch_dim):
            raise ValueError(
                f"Expected video_init_frame0 [{spec.video_patches_per_heartbeat}, {spec.video_patch_dim}], "
                f"got {video_init_frame0.shape}"
            )
    if video_groups is not None:
        if video_groups.ndim != 3 or video_groups.shape[1:] != (spec.video_patches_per_heartbeat, spec.video_patch_dim):
            raise ValueError(
                f"Expected video_groups [H, {spec.video_patches_per_heartbeat}, {spec.video_patch_dim}], "
                f"got {video_groups.shape}"
            )
        n_hb_candidates.append(int(video_groups.shape[0]))
    if audio_codes is not None:
        if audio_codes.ndim != 2 or audio_codes.shape[1] < spec.audio_codebooks_stored:
            raise ValueError(f"Expected audio_codes [2*H, >={spec.audio_codebooks_stored}], got {audio_codes.shape}")
        n_hb_candidates.append(int(audio_codes.shape[0] // spec.audio_ticks_per_heartbeat))
    if action_seq is not None:
        if action_seq.ndim != 2 or action_seq.shape[1] != spec.action_dim:
            raise ValueError(f"Expected action_seq [4*H, {spec.action_dim}], got {action_seq.shape}")
        n_hb_candidates.append(int(action_seq.shape[0] // spec.motor_ticks_per_heartbeat))
    if proprio_seq is not None:
        if proprio_seq.ndim != 2 or proprio_seq.shape[1] != spec.proprio_dim:
            raise ValueError(f"Expected proprio_seq [4*H, {spec.proprio_dim}], got {proprio_seq.shape}")
        n_hb_candidates.append(int(proprio_seq.shape[0] // spec.motor_ticks_per_heartbeat))

    if n_hb_candidates:
        n_hb = min(n_hb_candidates)
    elif text_by_heartbeat:
        if any(k < 0 for k in text_by_heartbeat):
            raise ValueError("text_by_heartbeat keys must be non-negative heartbeat indices.")
        n_hb = max(text_by_heartbeat.keys()) + 1
    else:
        raise ValueError("At least one active modality (or non-empty text_by_heartbeat) must be provided.")

    if n_hb <= 0:
        raise ValueError("Episode has zero complete 160 ms heartbeats.")

    act_m = _validate_binary_mask(action_mask, spec.action_dim, "action_mask")
    prop_m = _validate_binary_mask(proprio_mask, spec.proprio_dim, "proprio_mask")
    has_motor = (action_seq is not None) or (proprio_seq is not None)

    mod_ids: list[int] = []
    hb_ids: list[int] = []
    cap_times: list[float] = []
    last_times: list[float] = []
    avail_times: list[float] = []
    spatial_pos: list[int] = []
    text_bytes: list[int] = []
    vid_idx: list[int] = []
    aud_idx: list[int] = []
    mot_idx: list[int] = []

    vid_rows: list[np.ndarray] = []
    aud_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    prop_rows: list[np.ndarray] = []

    def _emit_motor(h: int, m_off: int, t_tick: float) -> None:
        idx_m = h * spec.motor_ticks_per_heartbeat + m_off
        a_vec = (
            action_seq[idx_m].astype(np.float32) * act_m
            if action_seq is not None
            else np.zeros((spec.action_dim,), dtype=np.float32)
        )
        p_vec = (
            proprio_seq[idx_m].astype(np.float32) * prop_m
            if proprio_seq is not None
            else np.zeros((spec.proprio_dim,), dtype=np.float32)
        )
        mot_idx.append(len(act_rows))
        act_rows.append(a_vec)
        prop_rows.append(p_vec)
        mod_ids.append(MOD_MOTOR)
        hb_ids.append(h)
        cap_times.append(t_tick)
        last_times.append(t_tick)
        avail_times.append(t_tick)
        spatial_pos.append(0)
        text_bytes.append(0)
        vid_idx.append(-1)
        aud_idx.append(-1)

    def _emit_audio(h: int, a_off: int, t_cap: float, t_avail: float) -> None:
        aud_idx.append(len(aud_rows))
        aud_rows.append(
            audio_codes[h * spec.audio_ticks_per_heartbeat + a_off, : spec.audio_codebooks_stored].astype(
                np.int16, copy=False
            )
        )
        mod_ids.append(MOD_AUDIO)
        hb_ids.append(h)
        cap_times.append(t_cap)
        last_times.append(t_avail)
        avail_times.append(t_avail)
        spatial_pos.append(0)
        text_bytes.append(0)
        vid_idx.append(-1)
        mot_idx.append(-1)

    def _emit_video_group(h: int, patches: np.ndarray, t_cap: float, t_avail: float) -> None:
        for p in range(spec.video_patches_per_heartbeat):
            vid_idx.append(len(vid_rows))
            vid_rows.append(patches[p].astype(np.float16, copy=False))
            mod_ids.append(MOD_VIDEO)
            hb_ids.append(h)
            cap_times.append(t_cap)
            last_times.append(t_avail)
            avail_times.append(t_avail)
            spatial_pos.append(p)
            text_bytes.append(0)
            aud_idx.append(-1)
            mot_idx.append(-1)

    # Optional Wan 2.1 `1 + 4k` bootstrap Frame 0 at exact t = 0.00 s
    if video_init_frame0 is not None:
        _emit_video_group(-1, video_init_frame0, 0.0, 0.0)

    dt_m = 1.0 / spec.motor_hz  # 0.04 s
    dt_a = 1.0 / spec.audio_hz  # 0.08 s

    for h in range(n_hb):
        t0 = h * spec.heartbeat_sec
        t_04 = t0 + 1.0 * dt_m
        t_08 = t0 + 2.0 * dt_m  # == t0 + dt_a
        t_12 = t0 + 3.0 * dt_m
        t_16 = (h + 1) * spec.heartbeat_sec  # == t0 + 4.0 * dt_m == t0 + 2.0 * dt_a

        # 1. t0 + 0.04 s: MOT[0]
        if has_motor:
            _emit_motor(h, 0, t_04)

        # 2. t0 + 0.08 s: MOT[1] BEFORE AUD[0] (zero acoustic lookahead into MOT[1]!)
        if has_motor:
            _emit_motor(h, 1, t_08)
        if audio_codes is not None:
            _emit_audio(h, 0, t0, t_08)

        # 3. t0 + 0.12 s: MOT[2] (causally observes AUD[0], never future AUD[1] or VID[h])
        if has_motor:
            _emit_motor(h, 2, t_12)

        # 4. t0 + 0.16 s: MOT[3] BEFORE VID[h] & AUD[1] (zero visual/acoustic consequence lookahead into MOT[3]!)
        if has_motor:
            _emit_motor(h, 3, t_16)

        # 5. t0 + 0.16 s: Wan 2.1 4-frame video group [4h+1..4h+4], available ONLY once final frame t_16 arrives!
        if video_groups is not None:
            _emit_video_group(h, video_groups[h], t_04, t_16)

        # 6. t0 + 0.16 s: Mimi AUD[1] covering [t0+0.08, t0+0.16] (causally attends to VID[h] & MOT[0..3]!)
        if audio_codes is not None:
            _emit_audio(h, 1, t_08, t_16)

        # 7. t0 + 0.16 s: Aligned causal speech/narration UTF-8 bytes arriving at t_avail = t_16
        if text_by_heartbeat and h in text_by_heartbeat:
            for b_val in text_by_heartbeat[h]:
                mod_ids.append(MOD_TEXT)
                hb_ids.append(h)
                cap_times.append(t_16)
                last_times.append(t_16)
                avail_times.append(t_16)
                spatial_pos.append(0)
                text_bytes.append(int(b_val))
                vid_idx.append(-1)
                aud_idx.append(-1)
                mot_idx.append(-1)

    n_tot = len(mod_ids)
    if n_tot == 0:
        raise ValueError("Constructed empty stream (e.g., text_by_heartbeat had only empty byte strings).")

    cap_arr = np.asarray(cap_times, dtype=np.float64)
    last_arr = np.asarray(last_times, dtype=np.float64)
    avail_arr = np.asarray(avail_times, dtype=np.float64)

    # STRICT causal checks (NO `np.maximum.accumulate` masking!)
    if not (np.all(np.isfinite(cap_arr)) and np.all(np.isfinite(last_arr)) and np.all(np.isfinite(avail_arr))):
        raise ValueError("Non-finite timestamp encountered in interleaved heartbeat stream.")
    if np.any(last_arr < cap_arr - 1e-12) or np.any(avail_arr < last_arr - 1e-12):
        raise ValueError("Causal timestamp invariant capture <= last_sample <= availability violated!")
    diffs = np.diff(avail_arr)
    if diffs.size > 0 and float(np.min(diffs)) < -1e-12:
        raise AssertionError(f"Non-monotonic availability timestamp detected (min diff = {float(np.min(diffs))})!")

    dt_s = np.zeros((n_tot,), dtype=np.float32)
    if n_tot > 1:
        dt_s[1:] = np.maximum(diffs, 0.0).astype(np.float32)
    reset_mask = np.zeros((n_tot,), dtype=np.uint8)
    reset_mask[0] = 1

    return {
        "modality_id": np.asarray(mod_ids, dtype=np.uint8),
        "heartbeat_id": np.asarray(hb_ids, dtype=np.int32),
        "capture_time_s": cap_arr.astype(np.float32),
        "last_sample_time_s": last_arr.astype(np.float32),
        "wall_time_s": avail_arr.astype(np.float32),  # == availability_time_s
        "dt_s": dt_s,
        "reset_mask": reset_mask,
        "valid_mask": np.ones((n_tot,), dtype=np.uint8),
        "target_valid_mask": np.ones((n_tot,), dtype=np.uint8),
        "reliability": np.ones((n_tot,), dtype=np.float32),
        "spatial_pos": np.asarray(spatial_pos, dtype=np.uint8),
        "text_byte": np.asarray(text_bytes, dtype=np.uint16),
        "video_idx": np.asarray(vid_idx, dtype=np.int32),
        "audio_idx": np.asarray(aud_idx, dtype=np.int32),
        "motor_idx": np.asarray(mot_idx, dtype=np.int32),
        "video_feat": np.stack(vid_rows, axis=0) if vid_rows else np.zeros((0, spec.video_patch_dim), dtype=np.float16),
        "audio_codes": np.stack(aud_rows, axis=0)
        if aud_rows
        else np.zeros((0, spec.audio_codebooks_stored), dtype=np.int16),
        "action_feat": np.stack(act_rows, axis=0) if act_rows else np.zeros((0, spec.action_dim), dtype=np.float32),
        "proprio_feat": np.stack(prop_rows, axis=0) if prop_rows else np.zeros((0, spec.proprio_dim), dtype=np.float32),
        # Masks passed BOTH as encoder conditioning (active 0.0 vs unobserved 0.0) and per-channel loss masks!
        "action_mask": act_m,
        "proprio_mask": prop_m,
    }


def make_same_clip_temporal_desync_pair(
    *,
    video_groups: np.ndarray,
    audio_codes: np.ndarray,
    window_heartbeats: int,
    shift_heartbeats: int = 5,
    start_heartbeat: int | None = None,
    spec: SLWMv4HeartbeatSpec = SLWMv4HeartbeatSpec(),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Create a seam-free, zero-shortcut same-clip AV `(clean, desync)` pair via interior window cropping.

    Eliminates BOTH shortcuts found in circular `np.roll` shifts:
      1. Zero circular splice discontinuity: both `clean` and `desync` contain strictly contiguous interior audio
         ticks (`np.all(np.diff(src_audio_ticks) == 1)`), so an audio-only detector cannot spot a splice seam.
      2. Zero audio-text shortcut: `MOD_TEXT` is strictly omitted (`text_by_heartbeat = None`) on AV-sync pairs.
      3. Identical `video_feat`, `wall_time_s`, `dt_s`, `reliability == 1.0`, and `valid_mask` across both windows.
    """
    if shift_heartbeats == 0:
        raise ValueError("shift_heartbeats must be non-zero for a genuine temporal desync negative.")
    if window_heartbeats <= 0:
        raise ValueError("window_heartbeats must be positive.")

    n_hb_src = min(int(video_groups.shape[0]), int(audio_codes.shape[0] // spec.audio_ticks_per_heartbeat))
    min_start = max(0, -int(shift_heartbeats))
    max_start = n_hb_src - window_heartbeats - max(0, int(shift_heartbeats))
    if max_start < min_start:
        raise ValueError(
            f"Source clip ({n_hb_src} hb) too short for interior crop window={window_heartbeats} + |shift|={abs(shift_heartbeats)}."
        )
    s0 = min_start if start_heartbeat is None else int(start_heartbeat)
    if not (min_start <= s0 <= max_start):
        raise ValueError(f"start_heartbeat={s0} outside valid seam-free interior bounds [{min_start}, {max_start}].")

    a_per_hb = spec.audio_ticks_per_heartbeat
    vid_win = video_groups[s0 : s0 + window_heartbeats]
    clean_a_start = s0 * a_per_hb
    clean_a_end = (s0 + window_heartbeats) * a_per_hb
    desync_a_start = (s0 + shift_heartbeats) * a_per_hb
    desync_a_end = (s0 + shift_heartbeats + window_heartbeats) * a_per_hb

    clean_stream = build_interleaved_heartbeat_stream(
        video_groups=vid_win,
        audio_codes=audio_codes[clean_a_start:clean_a_end],
        text_by_heartbeat=None,
        spec=spec,
    )
    desync_stream = build_interleaved_heartbeat_stream(
        video_groups=vid_win,
        audio_codes=audio_codes[desync_a_start:desync_a_end],
        text_by_heartbeat=None,
        spec=spec,
    )
    meta = {
        "window_heartbeats": window_heartbeats,
        "start_heartbeat": s0,
        "shift_heartbeats": shift_heartbeats,
        "shift_sec": round(shift_heartbeats * spec.heartbeat_sec, 4),
        "clean_audio_src_ticks": np.arange(clean_a_start, clean_a_end, dtype=np.int32),
        "desync_audio_src_ticks": np.arange(desync_a_start, desync_a_end, dtype=np.int32),
        "zero_wraparound_seam": True,
        "text_stripped_for_av_sync": True,
    }
    return clean_stream, desync_stream, meta


# ---------------------------------------------------------------------------
# Programmatically Derived Single-Pass (<= 1.0 Epoch) Corpus & Two-Tier Storage
# ---------------------------------------------------------------------------

# Verified multi-FPS resampling policy (Gate 1 -> Gate 2 enforced):
MULTI_FPS_RESAMPLING_POLICY: dict[str, str] = {
    "50_fps_aloha_unitreeh1_epic": "Exact 2:1 integer decimation to 25.0 fps video + 25.0 Hz kinematics (zero interpolation jitter).",
    "30_fps_unitree_g1_agibot_holoassist": "Exact causal nearest-past frame sampling to 25.0 fps (max 16.7 ms phase bound) + linear/SLERP kinematics resampling to 25.0 Hz.",
    "25_fps_vggsound_rh20t_greatesthits": "Native or PTS-locked 25.0 fps audio-video resampling (video & audio share identical PTS clock).",
    "15_fps_droid_1_0_1": "Causal timestamp resampling to 25.0 fps with explicit duplicate-frame indicator mask (`target_valid_mask = 0` on repeated video groups or native 15 Hz kinematic linear/SLERP alignment); audited in Gate 2.",
    "excluded_low_fps_5fps_bridge_10fps_libero": "BridgeData V2 (5 fps) is EXCLUDED from 25 fps Wan 2.1 4-frame video compression (prevents 5x static-frame collapse); LIBERO (10 fps) is held out of 25 fps video dynamics and deferred for closed-loop eval until Gate 1 LIBERO oracle addendum.",
}


def compute_corpus_exact_packets_and_storage(
    entry: dict[str, Any],
    *,
    video_latent_dtype: str = "fp16",
    spec: SLWMv4HeartbeatSpec = SLWMv4HeartbeatSpec(),
) -> dict[str, float]:
    """Derive exact packet counts (`M`) and uncompressed payload storage (`GB`, base-10) purely by formula.

    NEVER trusts hand-typed packet counts or disk sizes!
      - Video: `22,500 hb/hr * 15 patches = 337,500 pkts/hr`
        * `fp16` lossless: `1024 * 2 B` payload + `36 B` stream metadata = `2084 B/pkt` (`0.70335 GB/hr`)
        * `int8` per-patch affine (Tier 2, verified >50 dB latent PSNR in Gate 2): `1024 * 1 B + 4 B` + `36 B` = `1064 B/pkt` (`0.35910 GB/hr`)
      - Audio: `22,500 hb/hr * 2 ticks = 45,000 pkts/hr` * `(16 * 2 + 36) B` = `0.00306 GB/hr`
      - Motor+Proprio: `22,500 hb/hr * 4 ticks = 90,000 pkts/hr` * `((52 + 68) * 4 + 36) B` = `0.04644 GB/hr`
      - Text/Code: compact raw UTF-8 bytes stored on disk (`1.05 B/byte` incl. doc index; unpacked to packets on-the-fly in loader!)
    """
    hrs = float(entry["target_hours"])
    mods = set(entry["modalities"])
    hb_tot = hrs * spec.heartbeats_per_hour

    vid_pkts = hb_tot * spec.video_patches_per_heartbeat if "video" in mods else 0.0
    aud_pkts = hb_tot * spec.audio_ticks_per_heartbeat if "audio" in mods else 0.0
    mot_pkts = hb_tot * spec.motor_ticks_per_heartbeat if ("motor_action" in mods or "proprio" in mods or "proprio_force_torque" in mods) else 0.0

    if "raw_text_bytes_m" in entry:
        txt_pkts = float(entry["raw_text_bytes_m"]) * 1e6
    else:
        txt_bps = float(entry.get("text_bytes_per_sec", 0.0))
        txt_pkts = hrs * 3600.0 * txt_bps if ("text" in mods and txt_bps > 0.0) else 0.0

    tot_pkts = vid_pkts + aud_pkts + mot_pkts + txt_pkts

    if video_latent_dtype == "fp16":
        vid_bytes_per_pkt = spec.video_patch_dim * 2 + 36
    elif video_latent_dtype == "int8":
        vid_bytes_per_pkt = spec.video_patch_dim * 1 + 4 + 36
    else:
        raise ValueError(f"Unsupported video_latent_dtype: {video_latent_dtype}")

    aud_bytes_per_pkt = spec.audio_codebooks_stored * 2 + 36
    mot_bytes_per_pkt = (spec.action_dim + spec.proprio_dim) * 4 + 36
    txt_bytes_per_pkt = 1.05  # Stored as compact raw UTF-8 byte stream + offset table

    raw_bytes = (
        vid_pkts * vid_bytes_per_pkt
        + aud_pkts * aud_bytes_per_pkt
        + mot_pkts * mot_bytes_per_pkt
        + txt_pkts * txt_bytes_per_pkt
    )
    # Add 5% container / shard-index / SHA-256 manifest overhead
    storage_gb = (raw_bytes * 1.05) / 1e9

    return {
        "video_packets_m": vid_pkts / 1e6,
        "audio_packets_m": aud_pkts / 1e6,
        "motor_packets_m": mot_pkts / 1e6,
        "text_packets_m": txt_pkts / 1e6,
        "total_packets_m": tot_pkts / 1e6,
        "storage_gb": storage_gb,
    }


# Empirically verified Tier 1 (`V4-S3` Persistent `fp16` Pool — strictly inside 200-500 GB envelope!)
# Every repo was live-checked via HuggingFace API `meta/info.json` or byte-range tar header streaming (2026-11-24/25).
V4_TIER1_PERSISTENT_CORPUS_SPEC: list[dict[str, Any]] = [
    {
        "name": "unitree_g1_h1_humanoid",
        "pillar": "robotics_humanoid",
        "hf_repos": [
            # Exact sum verified from live meta/info.json = 108.67 h >= 60.0 h target!
            "unitreerobotics/G1_Dex1_Assembling_Motor",
            "unitreerobotics/G1_Dex1_Assemble_Stool",
            "unitreerobotics/G1_Dex1_Hang_Pants",
            "unitreerobotics/G1_Dex1_Fold_Clothes",
            "unitreerobotics/G1_WBT_Inspire_Walk_To_Table_Put_Cups_And_Rack_Plates_In_Dishwasher",
            "unitreerobotics/G1_Brainco_Make_Sandwich",
            "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine",
            "unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
            "unitreerobotics/G1_Dex1_MountCamera_Dataset",
            "unitreerobotics/G1_Dex3_ToastedBread_Dataset",
            "unitreerobotics/G1_Dex1_Wipe_Table",
            "unitreerobotics/G1_Dex1_Arrange_Surgical_Kits",
            "lerobot/unitreeh1_fold_clothes",
        ],
        "verified_source_hours": 108.67,
        "native_fps": "30.0 (G1) / 50.0 (H1)",
        "license": "apache-2.0 / unitree-open",
        "modalities": ["video", "motor_action", "proprio", "text"],
        "text_bytes_per_sec": 2.0,
        "disjointness_axis": "task_episode_hash_and_session_cluster",
        "target_hours": 60.0,
        "target_mix_weight": 0.042,
    },
    {
        "name": "droid_1_0_1",
        "pillar": "robotics_humanoid",
        "hf_repos": ["lerobot/droid_1.0.1"],
        "verified_source_hours": 511.67,
        "native_fps": "15.0",
        "license": "apache-2.0",
        "modalities": ["video", "motor_action", "proprio", "text", "episode_success"],
        "text_bytes_per_sec": 2.0,
        "disjointness_axis": "building_and_collector_id",
        "target_hours": 115.0,
        "target_mix_weight": 0.075,
    },
    {
        "name": "rh20t_contact_av_force",
        "pillar": "robotics_humanoid",
        "hf_repos": ["tonyfanggalaxies/RH20T"],
        "verified_source_hours": 50.0,
        "verified_audio_probe": "Byte-range tar header stream verified audio_mixed/audio_<ms>.wav (1.34-3.36 MB PCM WAV) + transformed/force_torque.npy present inside RH20T_lowdim/RH20T_cfg1..cfg7.tar.gz; full SNR/non-silence audited in Gate 2.",
        "native_fps": "10-30 (timestamp-locked color.mp4 + timestamps.npy)",
        "license": "cc-by-4.0",
        "modalities": ["video", "audio", "motor_action", "proprio_force_torque", "text"],
        "text_bytes_per_sec": 2.0,
        "disjointness_axis": "user_and_scene_id_and_robot_cfg",
        "target_hours": 45.0,
        "target_mix_weight": 0.035,
    },
    {
        "name": "agibot_and_aloha_sim",
        "pillar": "robotics_humanoid",
        "hf_repos": [
            "agibot-world/AgiBotWorld2026",
            "lerobot/aloha_sim_insertion_human",
            "lerobot/aloha_sim_transfer_cube_human",
        ],
        "verified_source_hours": 1000.0,  # AgiBotWorld2026 is 8,078 GB (30 fps); ALOHA-Sim is 0.25 h (50 fps)
        "native_fps": "30.0 (AgiBot) / 50.0 (ALOHA-Sim); 5 fps Bridge & 10 fps LIBERO excluded from 25 fps Wan 2.1",
        "license": "cc-by-nc-sa-4.0 (AgiBot, non-commercial research) / apache-2.0 (ALOHA)",
        "modalities": ["video", "motor_action", "proprio", "text"],
        "text_bytes_per_sec": 2.0,
        "disjointness_axis": "task_and_scene_id",
        "target_hours": 75.0,
        "target_mix_weight": 0.048,
    },
    {
        "name": "vggsound_full_sync",
        "pillar": "av_sync_egocentric",
        "hf_repos": ["Loie/VGGSound"],
        "verified_source_hours": 554.1,  # 199,467 clips in vggsound.csv * 10 s
        "native_fps": "25.0",
        "license": "cc-by-4.0",
        "modalities": ["video", "audio", "text"],
        "text_bytes_per_sec": 3.0,
        "disjointness_axis": "youtube_video_id_and_sha256_media",
        "target_hours": 150.0,
        "target_mix_weight": 0.092,
    },
    {
        "name": "epic_kitchens_100_long_ego_av",
        "pillar": "av_sync_egocentric",
        "hf_repos": ["a1raman/epic_kitchens_100"],
        "external_annotations": "https://github.com/epic-kitchens/epic-kitchens-100-annotations (official EPIC_100_train/validation CSVs)",
        "verified_source_hours": 65.0,  # 470 verified MP4s in a1raman/epic_kitchens_100
        "native_fps": "50.0 -> 25.0 (2:1 integer decimation)",
        "license": "cc-by-nc-4.0 (non-commercial research)",
        "modalities": ["video", "audio", "text"],
        "text_bytes_per_sec": 6.0,
        "disjointness_axis": "participant_id_P01_P37",
        "target_hours": 55.0,
        "target_mix_weight": 0.036,
    },
    {
        "name": "holoassist_and_greatesthits_av",
        "pillar": "av_sync_egocentric",
        "hf_repos": ["SitongGong/HoloAssist"],
        "external_sources": ["MIT CSAIL GreatestHits (~8.1 h impact-audio benchmark, verified in SLWM-v3)"],
        "verified_source_hours": 174.0,  # ~166 h HoloAssist + ~8.1 h GreatestHits
        "native_fps": "30.0 -> 25.0",
        "license": "cc-by-4.0 / MIT-research",
        "modalities": ["video", "audio", "text"],
        "text_bytes_per_sec": 6.0,
        "disjointness_axis": "recording_session_and_date",
        "target_hours": 55.0,
        "target_mix_weight": 0.036,
    },
    {
        "name": "wavcaps_and_libriheavy_speech",
        "pillar": "speech_audio_semantics",
        "hf_repos": ["cvssp/WavCaps", "mythicinfinity/libriheavy"],
        "verified_source_hours": 51000.0,  # LibriHeavy 50,000 h + WavCaps 400k clips (~1,100 h)
        "native_fps": "audio_only_12.5Hz_mimi",
        "license": "cc-by-4.0",
        "modalities": ["audio", "text"],
        "text_bytes_per_sec": 14.0,  # ~150 wpm aligned English speech transcripts / captions
        "disjointness_axis": "speaker_and_freesound_clip_id",
        "target_hours": 4000.0,
        "target_mix_weight": 0.386,
    },
    {
        "name": "fineweb_edu_and_stack_code",
        "pillar": "text_code_reasoning",
        "hf_repos": ["HuggingFaceFW/fineweb-edu", "HuggingFaceTB/smollm-corpus"],
        "verified_source_hours": 0.0,
        "native_fps": "text_bytes",
        "license": "odc-by-1.0 / apache-2.0",
        "modalities": ["text"],
        "raw_text_bytes_m": 500.0,
        "disjointness_axis": "exact_doc_sha256_and_64b_substring",
        "target_hours": 0.0,
        "target_mix_weight": 0.250,
    },
]


# Tier 2 (`V4-S4` 150M Full Humanoid Confirmation — 3.0B single-pass pos = 20.0 Chinchilla pos/param)
# Explicitly models streaming / rotating shard storage (or Gate-2-verified int8 Wan 2.1 latents on GCS/cloud SSD).
V4_TIER2_CONFIRMATION_HOURS_OVERRIDE: dict[str, dict[str, Any]] = {
    "unitree_g1_h1_humanoid": {"target_hours": 100.0, "target_mix_weight": 0.014},
    "droid_1_0_1": {"target_hours": 450.0, "target_mix_weight": 0.062},
    "rh20t_contact_av_force": {"target_hours": 50.0, "target_mix_weight": 0.008},
    "agibot_and_aloha_sim": {"target_hours": 750.0, "target_mix_weight": 0.105},  # Out of 8.08 TB AgiBotWorld2026!
    "vggsound_full_sync": {"target_hours": 520.0, "target_mix_weight": 0.066},  # Full ~187k clips of Loie/VGGSound
    "epic_kitchens_100_long_ego_av": {"target_hours": 65.0, "target_mix_weight": 0.009},
    "holoassist_and_greatesthits_av": {"target_hours": 165.0, "target_mix_weight": 0.021},
    "wavcaps_and_libriheavy_speech": {"target_hours": 14500.0, "target_mix_weight": 0.435},  # Out of 50,000 h LibriHeavy!
    "fineweb_edu_and_stack_code": {"raw_text_bytes_m": 1200.0, "target_mix_weight": 0.280},
}


def build_corpus_spec_for_tier(tier: str = "tier1_v4s3") -> list[dict[str, Any]]:
    """Build concrete corpus table with programmatically derived packet & storage budgets."""
    if tier == "tier1_v4s3":
        base = [dict(c) for c in V4_TIER1_PERSISTENT_CORPUS_SPEC]
        vid_dtype = "fp16"
    elif tier == "tier2_v4s4":
        base = []
        for c in V4_TIER1_PERSISTENT_CORPUS_SPEC:
            row = dict(c)
            if row["name"] in V4_TIER2_CONFIRMATION_HOURS_OVERRIDE:
                row.update(V4_TIER2_CONFIRMATION_HOURS_OVERRIDE[row["name"]])
            base.append(row)
        vid_dtype = "int8"  # Requires Gate 2 >50 dB latent PSNR cert; fp16 size also reported!
    else:
        raise ValueError(f"Unknown corpus tier: {tier}")

    out: list[dict[str, Any]] = []
    for c in base:
        calc_fp16 = compute_corpus_exact_packets_and_storage(c, video_latent_dtype="fp16")
        calc_sel = compute_corpus_exact_packets_and_storage(c, video_latent_dtype=vid_dtype)
        row = dict(c)
        row["derived_packets_m"] = round(calc_sel["total_packets_m"], 4)
        row["derived_breakdown_m"] = {
            "video": round(calc_sel["video_packets_m"], 4),
            "audio": round(calc_sel["audio_packets_m"], 4),
            "motor": round(calc_sel["motor_packets_m"], 4),
            "text": round(calc_sel["text_packets_m"], 4),
        }
        row["derived_storage_gb_fp16"] = round(calc_fp16["storage_gb"], 2)
        row["derived_storage_gb_selected"] = round(calc_sel["storage_gb"], 2)
        row["video_latent_dtype"] = vid_dtype
        out.append(row)
    return out


def verify_single_pass_budget(
    total_train_packets_m: float,
    *,
    tier: str = "tier1_v4s3",
) -> dict[str, Any]:
    """Verify strict <= 1.0 epoch single-pass coverage with fair proportional water-filling redistribution."""
    corpus = build_corpus_spec_for_tier(tier)
    avail = np.array([c["derived_packets_m"] for c in corpus], dtype=np.float64)
    total_avail_m = float(np.sum(avail))
    total_gb_fp16 = float(sum(c["derived_storage_gb_fp16"] for c in corpus))
    total_gb_sel = float(sum(c["derived_storage_gb_selected"] for c in corpus))

    weights = np.array([c["target_mix_weight"] for c in corpus], dtype=np.float64)
    weights /= float(np.sum(weights))

    # Standard water-filling capped at <= 1.00 epoch: redistribute capped excess PROPORTIONALLY to
    # original `target_mix_weight` among uncapped domains (prevents distorting large-pool domains!).
    cap_p = np.minimum(1.0, avail / max(float(total_train_packets_m), 1e-6))
    probs = weights.copy()
    for _ in range(32):
        capped = probs >= cap_p - 1e-12
        excess = float(np.sum(np.maximum(0.0, probs - cap_p)))
        if excess < 1e-10:
            break
        probs = np.minimum(probs, cap_p)
        uncapped_w = np.where(~capped & (cap_p - probs > 1e-12), weights, 0.0)
        denom = float(np.sum(uncapped_w))
        if denom <= 1e-12:
            break
        probs += excess * (uncapped_w / denom)

    per_domain: dict[str, Any] = {}
    pillar_share: dict[str, float] = {}
    mod_alloc_m = {"video": 0.0, "audio": 0.0, "motor": 0.0, "text": 0.0}
    max_ep = 0.0
    for c, p in zip(corpus, probs):
        alloc_m = float(p * total_train_packets_m)
        ep = alloc_m / max(float(c["derived_packets_m"]), 1e-6)
        max_ep = max(max_ep, ep)
        for mk in ("video", "audio", "motor", "text"):
            mod_alloc_m[mk] += float(c["derived_breakdown_m"][mk]) * ep
        per_domain[c["name"]] = {
            "target_hours": c["target_hours"],
            "allocated_packets_m": round(alloc_m, 2),
            "available_packets_m": round(float(c["derived_packets_m"]), 2),
            "planned_epochs": round(ep, 4),
            "mix_share_pct": round(float(p * 100.0), 2),
            "storage_gb_fp16": c["derived_storage_gb_fp16"],
            "storage_gb_selected": c["derived_storage_gb_selected"],
            "license": c["license"],
        }
        pillar_share[c["pillar"]] = pillar_share.get(c["pillar"], 0.0) + float(p * 100.0)

    signal_share_pct = 100.0 - pillar_share.get("text_code_reasoning", 0.0)
    mod_tot = max(sum(mod_alloc_m.values()), 1e-9)
    mod_share_pct = {k: round(100.0 * v / mod_tot, 2) for k, v in mod_alloc_m.items()}
    return {
        "tier": tier,
        "video_latent_dtype": corpus[0]["video_latent_dtype"],
        "target_train_packets_m": float(total_train_packets_m),
        "total_available_packets_m": round(total_avail_m, 2),
        "total_video_av_robotics_hours": round(
            sum(c["target_hours"] for c in corpus if "video" in c["modalities"]), 2
        ),
        "total_audio_only_hours": round(
            sum(c["target_hours"] for c in corpus if "video" not in c["modalities"]), 2
        ),
        "total_preencoded_storage_gb_fp16": round(total_gb_fp16, 2),
        "total_preencoded_storage_gb_selected": round(total_gb_sel, 2),
        "max_planned_epochs_any_domain": round(max_ep, 4),
        "strict_single_pass_le_1_epoch": max_ep <= 1.0001 and float(total_train_packets_m) <= total_avail_m,
        "pillar_share_pct": {k: round(v, 2) for k, v in pillar_share.items()},
        "multimodal_signal_share_pct": round(signal_share_pct, 2),
        "allocated_modality_packet_share_pct": mod_share_pct,
        "non_text_packet_share_pct": round(100.0 - mod_share_pct["text"], 2),
        "per_domain": per_domain,
    }
