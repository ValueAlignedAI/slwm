"""SLWM-v4 Gate 2: Universal Disjoint Splits, Non-Aliased 52/68 Embodiment Mappers, Causal Multi-FPS Resampler & Non-Vacuous Auditors.

Implements all Gate 2 connected-data invariants (AGENTS.md §12 Gate 2 & Gate 1 advisories):
  1. Leakage-proof deterministic splitting (`assign_v4_split`) keyed strictly on physical cluster boundaries
     (`building+collector` for DROID, `user+scene` for RH20T, `participant_id` for EPIC-KITCHENS, `youtube_id`
     for VGGSound, `session_prefix` for HoloAssist, `recording_date` for GreatestHits, `speaker_id` for LibriHeavy).
  2. Lossless non-aliased embodiment mapping (`map_embodiment_to_unified_52_68`) into `UNIFIED_ACTION_DIM = 52`
     and `UNIFIED_PROPRIO_DIM = 68` covering Unitree G1 (`Dex1`/`Brainco`/`Dex3`/`WBT+Inspire` 45-DoF),
     Unitree H1 (40-DoF), DROID (8-DoF), RH20T (arm+gripper+6D wrist FT), AgiBotWorld2026, and ALOHA-Sim (14-DoF).
  3. Strictly causal multi-FPS resampler (`resample_video_and_kinematics_to_25hz`): rejects `< 12 fps` corpora
     (`5 fps` Bridge, `10 fps` LIBERO), uses exact `2:1` decimation for `50 fps`, and strictly causal zero-order hold
     (`src_ts <= tgt_ts`, `0.000 ms` future lookahead) for `30 fps` / `15 fps` video and kinematics with explicit
     `video_group_novel_mask` for `15 fps` DROID.
  4. Episode-level audio non-silence / SNR filter (`audit_pcm_nonsilence`) for RH20T & AV corpora.
  5. Non-vacuous SHA-256 zero-train-eval-overlap auditor (`audit_v4_splits_non_vacuous`, enforcing `n_hashes > 0`
     on every expected active modality!) + audio-only clean-vs-desync shortcut null auditor (`AUROC ~ 0.50`).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np

from data.v4_heartbeat_contract import (
    ACTION_SLICES,
    PROPRIO_SLICES,
    UNIFIED_ACTION_DIM,
    UNIFIED_PROPRIO_DIM,
)

# ---------------------------------------------------------------------------
# 1. Deterministic Leakage-Boundary Disjoint Split Assignment
# ---------------------------------------------------------------------------

_RH20T_DIR_RE = re.compile(r"task_(\d+)_user_(\d+)_scene_(\d+)_cfg_(\d+)")


def extract_disjoint_cluster_key(corpus_name: str, meta: dict[str, Any]) -> str:
    """Return the mandatory leakage-boundary cluster key for `corpus_name` (raises on missing key!)."""
    if corpus_name == "unitree_g1_h1_humanoid":
        repo = str(meta["repo_id"]).strip().lower()
        blk_sz = 5 if "unitreeh1" in repo else 20
        sess = str(meta.get("session_id") or f"epblock_{int(meta['episode_idx']) // blk_sz}")
        return f"g1h1::{repo}::{sess}"

    if corpus_name == "droid_1_0_1":
        building = str(meta["building"]).strip().lower()
        collector = str(meta["collector_id"]).strip().lower()
        if not building or not collector:
            raise ValueError(f"DROID requires non-empty building & collector_id, got {meta}")
        return f"droid::{building}::{collector}"

    if corpus_name == "rh20t_contact_av_force":
        if "episode_dir" in meta:
            m = _RH20T_DIR_RE.search(str(meta["episode_dir"]))
            if not m:
                raise ValueError(f"Cannot parse RH20T user/scene from episode_dir={meta['episode_dir']!r}")
            user_id, scene_id = m.group(2), m.group(3)
        else:
            user_id, scene_id = str(meta["user_id"]), str(meta["scene_id"])
        # Split strictly by user + scene so no user/scene pair crosses train/dev/test!
        return f"rh20t::user_{user_id}::scene_{scene_id}"

    if corpus_name == "agibot_and_aloha_sim":
        src = str(meta["source"]).strip().lower()
        if "agibot" in src:
            return f"agibot::task_{meta['task_id']}::scene_{meta.get('scene_id', 'default')}"
        ep_blk = int(meta["episode_idx"]) // 5
        return f"aloha::{src}::block_{ep_blk}"

    if corpus_name == "vggsound_full_sync":
        yt_id = str(meta["youtube_id"]).strip()
        if len(yt_id) < 6:
            raise ValueError(f"Invalid VGGSound youtube_id: {yt_id!r}")
        # Shared canonical YouTube cluster key so any clip appearing in BOTH VGGSound and AudioCaps
        # is guaranteed to land in the exact same split (zero cross-corpus YouTube leakage!).
        return f"yt::{yt_id}"

    if corpus_name == "epic_kitchens_100_long_ego_av":
        pid = str(meta["participant_id"]).strip().upper()
        if not re.match(r"^P\d{2}$", pid):
            raise ValueError(f"Invalid EPIC-KITCHENS participant_id: {pid!r} (expected P01..P37)")
        return f"epic100::{pid}"

    if corpus_name == "holoassist_and_greatesthits_av":
        src = str(meta["source"]).strip().lower()
        if "greatest" in src:
            rec_date = str(meta["recording_date"]).strip()
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", rec_date):
                raise ValueError(f"Invalid GreatestHits recording_date: {rec_date!r}")
            return f"gh::{rec_date}"
        sess_prefix = str(meta["session_prefix"]).strip().upper()
        if not sess_prefix:
            raise ValueError(f"Missing HoloAssist session_prefix in {meta}")
        return f"holo::{sess_prefix}"

    if corpus_name == "wavcaps_and_libriheavy_speech":
        src = str(meta["source"]).strip().lower()
        if "libri" in src:
            spk = str(meta["speaker_id"]).strip()
            if not spk:
                raise ValueError("LibriHeavy requires speaker_id")
            return f"libriheavy::spk_{spk}"
        if "audiocaps" in src or "youtube_id" in meta:
            yt_id = str(meta["youtube_id"]).strip()
            if len(yt_id) < 6:
                raise ValueError(f"Invalid AudioCaps youtube_id: {yt_id!r}")
            return f"yt::{yt_id}"
        cid = str(meta["source_clip_id"]).strip()
        if not cid:
            raise ValueError("WavCaps requires source_clip_id")
        return f"wavcaps::{cid}"

    if corpus_name == "fineweb_edu_and_stack_code":
        if "repo_name" in meta and str(meta["repo_name"]).strip():
            repo = str(meta["repo_name"]).strip().lower()
            return f"code_repo::{repo}"
        doc_sha = str(meta["doc_sha256"]).strip().lower()
        if len(doc_sha) < 16:
            raise ValueError("Text/code split requires doc_sha256 or repo_name")
        return f"textcode::{doc_sha}"

    raise ValueError(f"Unknown SLWM-v4 corpus_name: {corpus_name!r}")


def assign_v4_split(
    corpus_name: str,
    meta: dict[str, Any],
    *,
    dev_frac: float | None = None,
    test_frac: float | None = None,
) -> tuple[str, str]:
    """Deterministically map `(corpus_name, meta)` -> `(split_name, cluster_key)` in `{train, dev, test}`."""
    # Small-cluster robotics pools require larger eval split fractions so both dev and test carry >= 4 disjoint
    # clusters across all embodiment families (`agibot`: 4 dev / 5 test task-scene clusters + 2 dev / 3 test ALOHA blocks;
    # `unitree_g1_h1`: 80/10/10 -> ~44 dev / ~31 test session blocks covering WBT, Inspire, Brainco, Dex3, Dex1, and H1).
    if dev_frac is None:
        dev_frac = 0.17 if corpus_name == "agibot_and_aloha_sim" else (0.10 if corpus_name == "unitree_g1_h1_humanoid" else 0.05)
    if test_frac is None:
        test_frac = 0.15 if corpus_name == "agibot_and_aloha_sim" else (0.10 if corpus_name == "unitree_g1_h1_humanoid" else 0.05)
    if not (0.0 < dev_frac < 0.4 and 0.0 < test_frac < 0.4 and dev_frac + test_frac < 0.6):
        raise ValueError(f"Invalid split fractions: dev={dev_frac}, test={test_frac}")
    cluster_key = extract_disjoint_cluster_key(corpus_name, meta)
    h_u32 = int(hashlib.sha256(f"slwm-v4-gate2::{cluster_key}".encode("utf-8")).hexdigest()[:8], 16)
    u = h_u32 / float(0xFFFFFFFF)
    if u < test_frac:
        return "test", cluster_key
    if u < test_frac + dev_frac:
        return "dev", cluster_key
    return "train", cluster_key


# ---------------------------------------------------------------------------
# 2. Lossless Non-Aliased 52-DoF Action & 68-Dim Proprio+Force Mappers
# ---------------------------------------------------------------------------


def map_embodiment_to_unified_52_68(
    embodiment: str,
    *,
    raw_action: np.ndarray,
    raw_proprio: np.ndarray | None = None,
    raw_force_torque_right: np.ndarray | None = None,
    raw_force_torque_left: np.ndarray | None = None,
    raw_imu_quat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map heterogeneous real-world robot trajectories into disjoint `52`/`68` tensors with ZERO aliasing or fabrication.

    Returns:
      `action_52: float32[T, 52]`, `action_mask: float32[52]`, `proprio_68: float32[T, 68]`, `proprio_mask: float32[68]`.
    """
    act_in = np.asarray(raw_action, dtype=np.float32)
    if act_in.ndim != 2:
        raise ValueError(f"Expected 2D raw_action [T, D], got {act_in.shape}")
    t_len, d_in = act_in.shape

    act = np.zeros((t_len, UNIFIED_ACTION_DIM), dtype=np.float32)
    act_m = np.zeros((UNIFIED_ACTION_DIM,), dtype=np.float32)
    prop = np.zeros((t_len, UNIFIED_PROPRIO_DIM), dtype=np.float32)
    prop_m = np.zeros((UNIFIED_PROPRIO_DIM,), dtype=np.float32)

    prop_in = None if raw_proprio is None else np.asarray(raw_proprio, dtype=np.float32)
    if prop_in is not None and (prop_in.ndim != 2 or prop_in.shape[0] != t_len):
        raise ValueError(f"raw_proprio shape {prop_in.shape} incompatible with T={t_len}")

    def _copy_slice(dst_sl: slice, src_a: np.ndarray, src_p: np.ndarray | None, width: int | None = None) -> None:
        w = (dst_sl.stop - dst_sl.start) if width is None else width
        s_end = dst_sl.start + w
        act[:, dst_sl.start : s_end] = src_a[:, :w]
        act_m[dst_sl.start : s_end] = 1.0
        if src_p is not None:
            prop[:, dst_sl.start : s_end] = src_p[:, :w]
            prop_m[dst_sl.start : s_end] = 1.0

    if embodiment == "aloha_bimanual_14dof":
        if d_in != 14:
            raise ValueError(f"aloha_bimanual_14dof expects 14 dims, got {d_in}")
        # [left_arm(6), left_grip(1), right_arm(6), right_grip(1)] — leaves 7th arm DoF strictly masked 0!
        _copy_slice(ACTION_SLICES["left_arm_7dof"], act_in[:, 0:6], None if prop_in is None else prop_in[:, 0:6], width=6)
        _copy_slice(ACTION_SLICES["left_hand_or_gripper_7dof"], act_in[:, 6:7], None if prop_in is None else prop_in[:, 6:7], width=1)
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 7:13], None if prop_in is None else prop_in[:, 7:13], width=6)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 13:14], None if prop_in is None else prop_in[:, 13:14], width=1)

    elif embodiment in ("droid_franka_8dof", "rh20t_single_arm_8dof"):
        if d_in != 8:
            raise ValueError(f"{embodiment} expects 8 dims [7 arm + 1 gripper], got {d_in}")
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 0:7], None if prop_in is None else prop_in[:, 0:7], width=7)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 7:8], None if prop_in is None else prop_in[:, 7:8], width=1)

    elif embodiment == "rh20t_single_arm_7dof":
        # 6-DoF UR5 / Kuka arm + 1-DoF gripper (leaves 7th arm DoF strictly masked 0.0 — zero fabrication!)
        if d_in != 7:
            raise ValueError(f"rh20t_single_arm_7dof expects 7 dims [6 arm + 1 gripper], got {d_in}")
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 0:6], None if prop_in is None else prop_in[:, 0:6], width=6)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 6:7], None if prop_in is None else prop_in[:, 6:7], width=1)

    elif embodiment == "unitree_g1_dex1_16dof":
        if d_in != 16:
            raise ValueError(f"unitree_g1_dex1_16dof expects 16 dims [7L+1L+7R+1R], got {d_in}")
        _copy_slice(ACTION_SLICES["left_arm_7dof"], act_in[:, 0:7], None if prop_in is None else prop_in[:, 0:7], width=7)
        _copy_slice(ACTION_SLICES["left_hand_or_gripper_7dof"], act_in[:, 7:8], None if prop_in is None else prop_in[:, 7:8], width=1)
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 8:15], None if prop_in is None else prop_in[:, 8:15], width=7)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 15:16], None if prop_in is None else prop_in[:, 15:16], width=1)

    elif embodiment == "unitree_g1_brainco_26dof":
        if d_in != 26:
            raise ValueError(f"unitree_g1_brainco_26dof expects 26 dims [7L+6L+7R+6R], got {d_in}")
        _copy_slice(ACTION_SLICES["left_arm_7dof"], act_in[:, 0:7], None if prop_in is None else prop_in[:, 0:7], width=7)
        _copy_slice(ACTION_SLICES["left_hand_or_gripper_7dof"], act_in[:, 7:13], None if prop_in is None else prop_in[:, 7:13], width=6)
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 13:20], None if prop_in is None else prop_in[:, 13:20], width=7)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 20:26], None if prop_in is None else prop_in[:, 20:26], width=6)

    elif embodiment == "unitree_g1_dex3_28dof":
        if d_in != 28:
            raise ValueError(f"unitree_g1_dex3_28dof expects 28 dims [7L+7L+7R+7R], got {d_in}")
        _copy_slice(ACTION_SLICES["left_arm_7dof"], act_in[:, 0:7], None if prop_in is None else prop_in[:, 0:7], width=7)
        _copy_slice(ACTION_SLICES["left_hand_or_gripper_7dof"], act_in[:, 7:14], None if prop_in is None else prop_in[:, 7:14], width=7)
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 14:21], None if prop_in is None else prop_in[:, 14:21], width=7)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 21:28], None if prop_in is None else prop_in[:, 21:28], width=7)

    elif embodiment == "unitree_g1_wbt_inspire_45dof":
        if d_in not in (41, 45):
            raise ValueError(f"unitree_g1_wbt_inspire_45dof expects 41 or 45 dims [7L+7R+6LH+6RH+6LL+6RL+3W(+4B)], got {d_in}")
        _copy_slice(ACTION_SLICES["left_arm_7dof"], act_in[:, 0:7], None if prop_in is None else prop_in[:, 0:7], width=7)
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 7:14], None if prop_in is None else prop_in[:, 7:14], width=7)
        _copy_slice(ACTION_SLICES["left_hand_or_gripper_7dof"], act_in[:, 14:20], None if prop_in is None else prop_in[:, 14:20], width=6)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 20:26], None if prop_in is None else prop_in[:, 20:26], width=6)
        _copy_slice(ACTION_SLICES["left_leg_6dof"], act_in[:, 26:32], None if prop_in is None else prop_in[:, 26:32], width=6)
        _copy_slice(ACTION_SLICES["right_leg_6dof"], act_in[:, 32:38], None if prop_in is None else prop_in[:, 32:38], width=6)
        _copy_slice(ACTION_SLICES["waist_3dof"], act_in[:, 38:41], None if prop_in is None else prop_in[:, 38:41], width=3)
        if d_in == 45:
            _copy_slice(ACTION_SLICES["mobile_base_twist_4dof"], act_in[:, 41:45], None if prop_in is None else prop_in[:, 41:45], width=4)

    elif embodiment == "unitree_h1_40dof":
        # Empirically verified on `lerobot/unitreeh1_fold_clothes` (50 fps, action[40], state[19]):
        #   action[0:5]=left_leg(5), [5:10]=right_leg(5), [10]=waist(1), [11:15]=left_arm(4), [15:19]=right_arm(4)
        #   (matching observation.state[0:19] 1:1 with r=0.97-0.98 on active arm joints),
        #   action[19:24]=head/aux(5), [24:30]=left_hand_inspire(6), [30:32]=left_wrist_aux(2 -> left_arm slots 4..5),
        #   [32:38]=right_hand_inspire(6), [38:40]=right_wrist_aux(2 -> right_arm slots 4..5).
        #   All 40/40 action dims are preserved losslessly; when prop_in has 19 dims, ONLY the 19 measured body
        #   joints are marked active in proprio_mask (zero fabrication!).
        if d_in != 40:
            raise ValueError(f"unitree_h1_40dof expects 40 action dims, got {d_in}")
        p_has_full = prop_in is not None and prop_in.shape[1] >= 40
        p_has_body = prop_in is not None and prop_in.shape[1] >= 19
        _copy_slice(ACTION_SLICES["left_leg_6dof"], act_in[:, 0:5], prop_in[:, 0:5] if p_has_body else None, width=5)
        _copy_slice(ACTION_SLICES["right_leg_6dof"], act_in[:, 5:10], prop_in[:, 5:10] if p_has_body else None, width=5)
        _copy_slice(ACTION_SLICES["waist_3dof"], act_in[:, 10:11], prop_in[:, 10:11] if p_has_body else None, width=1)
        _copy_slice(
            ACTION_SLICES["left_arm_7dof"],
            np.concatenate([act_in[:, 11:15], act_in[:, 30:32]], axis=-1),
            np.concatenate([prop_in[:, 11:15], prop_in[:, 30:32]], axis=-1) if p_has_full else None,
            width=6,
        )
        _copy_slice(
            ACTION_SLICES["right_arm_7dof"],
            np.concatenate([act_in[:, 15:19], act_in[:, 38:40]], axis=-1),
            np.concatenate([prop_in[:, 15:19], prop_in[:, 38:40]], axis=-1) if p_has_full else None,
            width=6,
        )
        if p_has_body and not p_has_full:
            la_s = ACTION_SLICES["left_arm_7dof"].start
            ra_s = ACTION_SLICES["right_arm_7dof"].start
            prop[:, la_s : la_s + 4] = prop_in[:, 11:15]
            prop_m[la_s : la_s + 4] = 1.0
            prop[:, ra_s : ra_s + 4] = prop_in[:, 15:19]
            prop_m[ra_s : ra_s + 4] = 1.0
        _copy_slice(ACTION_SLICES["head_and_aux_5dof"], act_in[:, 19:24], prop_in[:, 19:24] if p_has_full else None, width=5)
        _copy_slice(ACTION_SLICES["left_hand_or_gripper_7dof"], act_in[:, 24:30], prop_in[:, 24:30] if p_has_full else None, width=6)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 32:38], prop_in[:, 32:38] if p_has_full else None, width=6)

    elif embodiment == "agibot_g2a_bimanual":
        # Exact verified AgiBotWorld2026 G2A LeRobot v2.1 `meta/info.json` `field_descriptions`:
        #   action[44]: [0]=left_eff, [1]=right_eff, [16:23]=left_arm(7), [23:30]=right_arm(7),
        #               [30:33]=head(3), [33:36]=waist(3), [36:38]=waist_aux(2), [38:42]=base_vel(4)
        #   state[216]: [0]=left_eff, [1]=right_eff, [2:26]=state/end/wrench(24) where empirically
        #               state[:, 2:14] == state[:, 14:26] bit-identically ([2:8]=left 6D wrench, [8:14]=right 6D wrench),
        #               [65:72]=left_arm(7), [72:79]=right_arm(7), [121:124]=head(3), [127:132]=waist(5), [140:144]=base_quat(4)
        if d_in != 44:
            raise ValueError(f"agibot_g2a_bimanual expects verified AgiBotWorld2026 action[44], got {d_in}")
        p_la = None if prop_in is None else prop_in[:, 65:72]
        p_ra = None if prop_in is None else prop_in[:, 72:79]
        p_lg = None if prop_in is None else prop_in[:, 0:1]
        p_rg = None if prop_in is None else prop_in[:, 1:2]
        p_w = None if prop_in is None else prop_in[:, 127:130]
        p_h = None if prop_in is None else np.concatenate([prop_in[:, 121:124], prop_in[:, 130:132]], axis=-1)
        _copy_slice(ACTION_SLICES["left_arm_7dof"], act_in[:, 16:23], p_la, width=7)
        _copy_slice(ACTION_SLICES["right_arm_7dof"], act_in[:, 23:30], p_ra, width=7)
        _copy_slice(ACTION_SLICES["left_hand_or_gripper_7dof"], act_in[:, 0:1], p_lg, width=1)
        _copy_slice(ACTION_SLICES["right_hand_or_gripper_7dof"], act_in[:, 1:2], p_rg, width=1)
        _copy_slice(ACTION_SLICES["waist_3dof"], act_in[:, 33:36], p_w, width=3)
        _copy_slice(ACTION_SLICES["mobile_base_twist_4dof"], act_in[:, 38:42], None, width=4)
        _copy_slice(
            ACTION_SLICES["head_and_aux_5dof"],
            np.concatenate([act_in[:, 30:33], act_in[:, 36:38]], axis=-1),
            p_h,
            width=5,
        )
        if prop_in is not None and prop_in.shape[1] >= 144:
            if raw_force_torque_left is None:
                raw_force_torque_left = prop_in[:, 2:8]
            if raw_force_torque_right is None:
                raw_force_torque_right = prop_in[:, 8:14]
            if raw_imu_quat is None:
                raw_imu_quat = prop_in[:, 140:144]
    else:
        raise ValueError(f"Unknown embodiment: {embodiment!r}")

    # Dedicated force-torque & IMU proprio channels (zero aliasing with joints!)
    if raw_force_torque_left is not None:
        ft_l = np.asarray(raw_force_torque_left, dtype=np.float32)
        if ft_l.shape != (t_len, 6):
            raise ValueError(f"Expected raw_force_torque_left [T, 6], got {ft_l.shape}")
        prop[:, PROPRIO_SLICES["left_wrist_force_torque_6d"]] = ft_l
        prop_m[PROPRIO_SLICES["left_wrist_force_torque_6d"]] = 1.0
    if raw_force_torque_right is not None:
        ft_r = np.asarray(raw_force_torque_right, dtype=np.float32)
        if ft_r.shape != (t_len, 6):
            raise ValueError(f"Expected raw_force_torque_right [T, 6], got {ft_r.shape}")
        prop[:, PROPRIO_SLICES["right_wrist_force_torque_6d"]] = ft_r
        prop_m[PROPRIO_SLICES["right_wrist_force_torque_6d"]] = 1.0
    if raw_imu_quat is not None:
        imu = np.asarray(raw_imu_quat, dtype=np.float32)
        if imu.shape != (t_len, 4):
            raise ValueError(f"Expected raw_imu_quat [T, 4], got {imu.shape}")
        # Unit-normalize quaternions safely
        norms = np.maximum(np.linalg.norm(imu, axis=-1, keepdims=True), 1e-8)
        prop[:, PROPRIO_SLICES["torso_imu_base_orient_4d"]] = imu / norms
        prop_m[PROPRIO_SLICES["torso_imu_base_orient_4d"]] = 1.0

    # Strict post-condition: zero non-zero values on inactive channels!
    assert float(np.max(np.abs(act[:, act_m == 0.0]))) == 0.0
    assert float(np.max(np.abs(prop[:, prop_m == 0.0]))) == 0.0
    return act, act_m, prop, prop_m


def map_unitree_g1_named_columns(
    cols: dict[str, np.ndarray],
    t_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map exact verified Unitree G1 LeRobot v3.0 named parquet columns directly into non-aliased `52`/`68` tensors."""
    act = np.zeros((t_len, UNIFIED_ACTION_DIM), dtype=np.float32)
    act_m = np.zeros((UNIFIED_ACTION_DIM,), dtype=np.float32)
    prop = np.zeros((t_len, UNIFIED_PROPRIO_DIM), dtype=np.float32)
    prop_m = np.zeros((UNIFIED_PROPRIO_DIM,), dtype=np.float32)

    def _put(dst: np.ndarray, mask: np.ndarray, sl: slice, arr: np.ndarray | None, width: int | None = None) -> None:
        if arr is None:
            return
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        w = min(a.shape[1], sl.stop - sl.start) if width is None else width
        dst[:, sl.start : sl.start + w] = a[:, :w]
        mask[sl.start : sl.start + w] = 1.0

    # Arms (7L + 7R)
    _put(act, act_m, ACTION_SLICES["left_arm_7dof"], cols.get("action.left_arm"))
    _put(act, act_m, ACTION_SLICES["right_arm_7dof"], cols.get("action.right_arm"))
    _put(prop, prop_m, ACTION_SLICES["left_arm_7dof"], cols.get("observation.state.left_arm"))
    _put(prop, prop_m, ACTION_SLICES["right_arm_7dof"], cols.get("observation.state.right_arm"))

    # Hands / grippers: either 12-DoF `hand_cmd` (6L+6R Inspire/Brainco) or scalar `left_gripper`/`right_gripper` (Dex1)
    if "action.hand_cmd" in cols:
        hc = np.asarray(cols["action.hand_cmd"], dtype=np.float32)
        half = hc.shape[1] // 2
        _put(act, act_m, ACTION_SLICES["left_hand_or_gripper_7dof"], hc[:, :half], width=min(half, 7))
        _put(act, act_m, ACTION_SLICES["right_hand_or_gripper_7dof"], hc[:, half:], width=min(hc.shape[1] - half, 7))
    else:
        _put(act, act_m, ACTION_SLICES["left_hand_or_gripper_7dof"], cols.get("action.left_gripper"), width=1)
        _put(act, act_m, ACTION_SLICES["right_hand_or_gripper_7dof"], cols.get("action.right_gripper"), width=1)
    if "observation.state.hand_state" in cols:
        hs = np.asarray(cols["observation.state.hand_state"], dtype=np.float32)
        half = hs.shape[1] // 2
        _put(prop, prop_m, ACTION_SLICES["left_hand_or_gripper_7dof"], hs[:, :half], width=min(half, 7))
        _put(prop, prop_m, ACTION_SLICES["right_hand_or_gripper_7dof"], hs[:, half:], width=min(hs.shape[1] - half, 7))
    else:
        _put(prop, prop_m, ACTION_SLICES["left_hand_or_gripper_7dof"], cols.get("observation.state.left_gripper"), width=1)
        _put(prop, prop_m, ACTION_SLICES["right_hand_or_gripper_7dof"], cols.get("observation.state.right_gripper"), width=1)

    # Legs (6L + 6R), Waist (3), Mobile Base Command (4)
    _put(act, act_m, ACTION_SLICES["left_leg_6dof"], cols.get("action.left_leg"))
    _put(act, act_m, ACTION_SLICES["right_leg_6dof"], cols.get("action.right_leg"))
    _put(act, act_m, ACTION_SLICES["waist_3dof"], cols.get("action.waist_action_joint"))
    _put(act, act_m, ACTION_SLICES["mobile_base_twist_4dof"], cols.get("action.base_command"))
    _put(prop, prop_m, ACTION_SLICES["left_leg_6dof"], cols.get("observation.state.left_leg"))
    _put(prop, prop_m, ACTION_SLICES["right_leg_6dof"], cols.get("observation.state.right_leg"))
    _put(prop, prop_m, ACTION_SLICES["waist_3dof"], cols.get("observation.state.waist_state_joint"))

    # Torso IMU unit quaternion (`observation.state.state_base_rot_quat[4]`)
    if "observation.state.state_base_rot_quat" in cols:
        q = np.asarray(cols["observation.state.state_base_rot_quat"], dtype=np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        _put(prop, prop_m, PROPRIO_SLICES["torso_imu_base_orient_4d"], q, width=4)

    assert float(np.max(np.abs(act[:, act_m == 0.0]))) == 0.0
    assert float(np.max(np.abs(prop[:, prop_m == 0.0]))) == 0.0
    return act, act_m, prop, prop_m


# ---------------------------------------------------------------------------
# 3. Strictly Causal Multi-FPS Resampler (25.0 fps / 25.0 Hz Target Clock)
# ---------------------------------------------------------------------------


def _slerp_quat_batch(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Exact spherical linear interpolation (SLERP) for unit quaternion sequences `[T, 4]`."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1_adj = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    a = alpha[:, None]
    small = sin_t < 1e-5
    w0 = np.where(small, 1.0 - a, np.sin((1.0 - a) * theta) / np.where(small, 1.0, sin_t))
    w1 = np.where(small, a, np.sin(a * theta) / np.where(small, 1.0, sin_t))
    out = w0 * q0 + w1 * q1_adj
    return (out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)).astype(np.float32)


def resample_video_and_kinematics_to_25hz(
    *,
    native_fps: float,
    n_src_frames: int,
    action_52: np.ndarray | None = None,
    proprio_68: np.ndarray | None = None,
    proprio_mask: np.ndarray | None = None,
    target_fps: float = 25.0,
) -> dict[str, Any]:
    """Compute strictly causal (`src_ts <= tgt_ts`, `0.000 ms` lookahead) 25 Hz video frame indices & ZOH kinematics.

    Rejects low-FPS corpora (`native_fps < 12.0`, e.g., `5 fps` BridgeData V2 and `10 fps` LIBERO) so they
    can never inject pathological multi-frame static copies into 25 fps Wan 2.1 4-frame video compression.
    Returns `video_group_novel_mask` (`uint8[H]`) flagging groups with `< 2` unique source frames (e.g. for 15 fps).
    """
    if float(native_fps) < 12.0:
        raise ValueError(
            f"Refusing to resample native_fps={native_fps} < 12.0 Hz into 25.0 fps Wan 2.1 temporal compression "
            "(explicit Gate 1 exclusion of 5 fps Bridge & 10 fps LIBERO)."
        )
    if n_src_frames < 5:
        raise ValueError(f"Episode too short ({n_src_frames} frames) for even one 1+4 Wan 2.1 heartbeat.")

    dur_s = (n_src_frames - 1) / float(native_fps)
    # Wan 2.1 `1 + 4*H` framing: find largest `1 + 4*H` fitting strictly inside `dur_s` (zero extrapolation!)
    n_tgt_max = int(math.floor(dur_s * float(target_fps))) + 1
    n_hb = (n_tgt_max - 1) // 4
    if n_hb < 1:
        raise ValueError(f"Episode duration {dur_s:.3f}s yields 0 complete 160 ms heartbeats at {target_fps} fps.")
    n_tgt_frames = 1 + 4 * n_hb  # Frame 0 + H * 4 frames
    tgt_ts = np.arange(n_tgt_frames, dtype=np.float64) / float(target_fps)

    # Strictly causal nearest-past source frame index (`src_ts <= tgt_ts`, 0 ms future lookahead!)
    src_pos = tgt_ts * float(native_fps)
    if abs(native_fps - 2.0 * target_fps) < 1e-6:
        # Exact 2:1 integer decimation for 50 fps -> 25 fps
        vid_src_idx = (np.arange(n_tgt_frames, dtype=np.int64) * 2).clip(0, n_src_frames - 1)
    else:
        vid_src_idx = np.floor(src_pos + 1e-9).astype(np.int64).clip(0, n_src_frames - 1)

    # Verify strict causality: every selected source frame timestamp <= target timestamp!
    assert np.all((vid_src_idx / float(native_fps)) <= tgt_ts + 1e-9)

    # Novelty mask per 4-frame Wan 2.1 group `[4h+1 .. 4h+4]` (for 15 fps DROID duplicate auditing)
    group_novel = np.ones((n_hb,), dtype=np.uint8)
    for h in range(n_hb):
        grp_src = vid_src_idx[1 + 4 * h : 1 + 4 * (h + 1)]
        if len(set(int(x) for x in grp_src)) < 2:
            group_novel[h] = 0

    def _zoh_kin(arr: np.ndarray, normalize_imu: bool) -> np.ndarray:
        """Strictly causal zero-order hold (`src_ts[idx] <= tgt_ts`, 0.000 ms future lookahead!).

        Uses the exact same causal nearest-past index `vid_src_idx` as video framing so kinematics can NEVER
        leak a future sample `idx + 1` (e.g. +40..53 ms on 15 fps DROID or +26.7 ms on 30 fps G1/AgiBot).
        """
        idx_k = np.clip(vid_src_idx, 0, arr.shape[0] - 1)
        out = arr[idx_k].astype(np.float32, copy=True)
        if normalize_imu and proprio_mask is not None:
            imu_sl = PROPRIO_SLICES["torso_imu_base_orient_4d"]
            if np.all(proprio_mask[imu_sl] == 1.0):
                q = out[:, imu_sl]
                out[:, imu_sl] = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        # Drop frame 0 so motor_ticks aligns 1:1 with Heartbeat ticks [t0+0.04 .. H*0.16] (length 4*H)!
        return out[1:].astype(np.float32)

    out_dict: dict[str, Any] = {
        "n_heartbeats": n_hb,
        "n_target_frames_1_plus_4h": n_tgt_frames,
        "video_src_frame_indices": vid_src_idx,
        "kinematic_src_frame_indices": vid_src_idx[1:].copy(),
        "target_timestamps_s": tgt_ts.astype(np.float32),
        "video_group_novel_mask": group_novel,
    }
    if action_52 is not None:
        out_dict["action_25hz"] = _zoh_kin(action_52, normalize_imu=False)
    if proprio_68 is not None:
        out_dict["proprio_25hz"] = _zoh_kin(proprio_68, normalize_imu=True)
    return out_dict


# ---------------------------------------------------------------------------
# 4. Episode-Level Audio Non-Silence / SNR Filter (RH20T & AV Corpora)
# ---------------------------------------------------------------------------


def audit_pcm_nonsilence(
    pcm: np.ndarray,
    sr: int,
    *,
    min_rms_dbfs: float = -55.0,
    min_active_frame_frac: float = 0.05,
) -> dict[str, Any]:
    """Detect silent, DC-stuck, or corrupt microphones without fabricating audio."""
    x = np.asarray(pcm, dtype=np.float64).reshape(-1)
    if x.size < max(1, int(0.16 * sr)) or not np.all(np.isfinite(x)):
        return {"audio_valid": False, "rms_dbfs": -120.0, "peak_abs": 0.0, "active_frame_frac": 0.0, "reason": "short_or_nonfinite"}
    x_ac = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x_ac * x_ac)))
    peak = float(np.max(np.abs(x_ac)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))

    hop = max(1, int(round(0.020 * sr)))
    n_fr = max(1, x_ac.size // hop)
    frames = x_ac[: n_fr * hop].reshape(n_fr, hop)
    fr_rms = np.sqrt(np.mean(frames * frames, axis=-1))
    fr_dbfs = 20.0 * np.log10(np.maximum(fr_rms, 1e-12))
    active_frac = float(np.mean(fr_dbfs >= min_rms_dbfs))

    ok = (rms_dbfs >= min_rms_dbfs) and (peak >= 1e-3) and (active_frac >= min_active_frame_frac)
    return {
        "audio_valid": bool(ok),
        "rms_dbfs": round(rms_dbfs, 2),
        "peak_abs": round(peak, 5),
        "active_frame_frac": round(active_frac, 4),
        "reason": "ok" if ok else "silent_or_dead_mic",
    }


# ---------------------------------------------------------------------------
# 5. Non-Vacuous SHA-256 Split Disjointness & Audio-Only Desync Null Auditors
# ---------------------------------------------------------------------------


def hash_payload_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def audit_v4_splits_non_vacuous(
    records: list[dict[str, Any]],
    *,
    expected_modalities: tuple[str, ...] = ("video", "audio"),
) -> dict[str, Any]:
    """Non-vacuous disjointness certifier across `(train, dev, test)` for BOTH cluster keys AND raw media SHA-256.

    Each record must contain: `split` in `{train, dev, test}`, `cluster_key: str`, and `media_hashes: dict[str, str]`.
    Hard-asserts `n_hashes > 0` for every expected active modality on every populated split (eliminates vacuous passes!).
    """
    clusters_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    hashes_by_mod_split: dict[str, dict[str, set[str]]] = {
        m: {"train": set(), "dev": set(), "test": set()} for m in expected_modalities
    }
    counts_by_split = {"train": 0, "dev": 0, "test": 0}

    for rec in records:
        sp = str(rec["split"])
        if sp not in counts_by_split:
            raise ValueError(f"Unknown split: {sp!r}")
        counts_by_split[sp] += 1
        clusters_by_split[sp].add(str(rec["cluster_key"]))
        mh = rec.get("media_hashes", {})
        for m in expected_modalities:
            if m in mh and mh[m]:
                hashes_by_mod_split[m][sp].add(str(mh[m]))

    # Non-vacuity check: every split must have >0 records and >0 hashes for every expected modality!
    for sp in ("train", "dev", "test"):
        if counts_by_split[sp] == 0:
            raise AssertionError(f"Vacuous split audit: split={sp!r} has 0 records!")
        for m in expected_modalities:
            if len(hashes_by_mod_split[m][sp]) == 0:
                raise AssertionError(f"Vacuous media-hash audit: modality={m!r} on split={sp!r} has 0 hashes!")

    cluster_overlaps = {
        "train_dev": len(clusters_by_split["train"] & clusters_by_split["dev"]),
        "train_test": len(clusters_by_split["train"] & clusters_by_split["test"]),
        "dev_test": len(clusters_by_split["dev"] & clusters_by_split["test"]),
    }
    media_overlaps: dict[str, dict[str, int]] = {}
    total_media_leaks = 0
    for m in expected_modalities:
        ov = {
            "train_dev": len(hashes_by_mod_split[m]["train"] & hashes_by_mod_split[m]["dev"]),
            "train_test": len(hashes_by_mod_split[m]["train"] & hashes_by_mod_split[m]["test"]),
            "dev_test": len(hashes_by_mod_split[m]["dev"] & hashes_by_mod_split[m]["test"]),
        }
        media_overlaps[m] = ov
        total_media_leaks += ov["train_dev"] + ov["train_test"] + ov["dev_test"]

    total_cluster_leaks = sum(cluster_overlaps.values())
    return {
        "counts_by_split": counts_by_split,
        "unique_cluster_counts": {sp: len(v) for sp, v in clusters_by_split.items()},
        "unique_media_hash_counts": {m: {sp: len(v) for sp, v in d.items()} for m, d in hashes_by_mod_split.items()},
        "cluster_overlaps": cluster_overlaps,
        "media_payload_overlaps": media_overlaps,
        "non_vacuous_zero_overlap_pass": bool(total_cluster_leaks == 0 and total_media_leaks == 0),
    }


def _audio_only_seam_features(audio_codes: np.ndarray) -> np.ndarray:
    """Audio-only features specifically designed to catch splice seams, boundary jumps, or distribution shifts."""
    c = audio_codes.astype(np.float64)  # [T, K]
    diffs = np.abs(np.diff(c, axis=0))  # [T-1, K]
    step_l1 = np.mean(diffs, axis=-1)  # mean L1 code jump per tick (spikes sharply at circular splice seams!)
    step_frac = np.mean(diffs > 0.0, axis=-1)
    max_l1 = float(np.max(step_l1)) if step_l1.size else 0.0
    mean_l1 = float(np.mean(step_l1)) if step_l1.size else 0.0
    std_l1 = float(np.std(step_l1)) if step_l1.size else 0.0
    q95_l1 = float(np.quantile(step_l1, 0.95)) if step_l1.size else 0.0
    peak_ratio = max_l1 / max(mean_l1, 1e-6)
    max_frac = float(np.max(step_frac)) if step_frac.size else 0.0
    rvq0_mean = float(np.mean(c[:, 0]))
    rvq0_std = float(np.std(c[:, 0]))
    start_jump = float(np.mean(np.abs(c[0] - c[1]))) if c.shape[0] > 1 else 0.0
    end_jump = float(np.mean(np.abs(c[-1] - c[-2]))) if c.shape[0] > 1 else 0.0
    return np.array([max_l1, mean_l1, std_l1, q95_l1, peak_ratio, max_frac, rvq0_mean, rvq0_std, start_jump, end_jump], dtype=np.float64)


def audit_audio_only_desync_null_auroc(
    clean_audio_list: list[np.ndarray],
    desync_audio_list: list[np.ndarray],
    *,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Verify an audio-only linear/ridge seam detector sits at chance (`0.38 <= AUROC <= 0.62`) on clean vs. desync pairs."""
    n = min(len(clean_audio_list), len(desync_audio_list))
    if n < 20:
        raise ValueError(f"Need >= 20 clean/desync pairs for reliable null AUROC, got {n}")
    x_pos = np.stack([_audio_only_seam_features(a) for a in clean_audio_list[:n]], axis=0)
    x_neg = np.stack([_audio_only_seam_features(a) for a in desync_audio_list[:n]], axis=0)
    X = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([np.ones((n,), dtype=np.int32), np.zeros((n,), dtype=np.int32)], axis=0)

    # Pair-grouped K-fold cross-validation (never splits a clean/desync pair across train and val!)
    scores = np.zeros((2 * n,), dtype=np.float64)
    fold_ids = np.arange(n) % n_folds
    for f in range(n_folds):
        val_p = fold_ids == f
        tr_p = ~val_p
        tr_idx = np.concatenate([np.where(tr_p)[0], n + np.where(tr_p)[0]])
        va_idx = np.concatenate([np.where(val_p)[0], n + np.where(val_p)[0]])
        mu = np.mean(X[tr_idx], axis=0, keepdims=True)
        sd = np.maximum(np.std(X[tr_idx], axis=0, keepdims=True), 1e-6)
        Xtr = (X[tr_idx] - mu) / sd
        Xva = (X[va_idx] - mu) / sd
        # Ridge regression classifier
        w = np.linalg.solve(Xtr.T @ Xtr + 10.0 * np.eye(Xtr.shape[1]), Xtr.T @ (2.0 * y[tr_idx] - 1.0))
        scores[va_idx] = Xva @ w

    # Exact Mann-Whitney U AUROC
    pos_s = scores[:n]
    neg_s = scores[n:]
    auroc = float(np.mean((pos_s[:, None] > neg_s[None, :]) + 0.5 * (pos_s[:, None] == neg_s[None, :])))
    return {
        "n_pairs": n,
        "audio_only_null_auroc": round(auroc, 4),
        "at_chance_zero_splice_shortcut": bool(0.38 <= auroc <= 0.62),
    }
