"""SLWM-v4 Gate 1: Hardened Anti-Knockoff Closed-Loop MuJoCo ALOHA Oracle Ceilings & Counterfactual Certifier.

Certifies BOTH `gym_aloha/AlohaTransferCube-v0` AND `gym_aloha/AlohaInsertion-v0` (14-DoF bimanual ViperX-300s)
at the exact SLWM-v4 25 Hz motor control rate (`dt = 0.04 s` = 2 x 50 Hz = 20 x 0.002 s MuJoCo steps).

Gate 1 Scientific Hardening:
  1. Matched-Perturbation Control Isolation (eliminates disturbance confound):
     Every environment compares 5 decoupled arms across `N = 32` randomized unseen seeds using ONE shared
     controller function and strict arm-joint-only actuation drift (`act_drift` applied ONLY to the 12 radian
     arm joints clipped to `model.jnt_range`, with grippers `[6, 13]` strictly uncorrupted):
       - `oracle_perc_oracle_dyn_closed_loop` (`noise = 0`, `drift = 0.0 rad`)
       - `oracle_perc_oracle_dyn_closed_loop_0p15rad_drift` (`noise = 0`, `drift = 0.15 rad` — matched vs. open-loop!)
       - `noisy_perc_3cm_oracle_dyn_closed_loop` (`noise = 0.03 m`, `drift = 0.0 rad`)
       - `oracle_perc_open_loop_nominal_0p15rad_drift` (pre-computed nominal joint tape + `drift = 0.15 rad`, 0 feedback)
       - `hold_pose_null` (`0.0%` static baseline)
  2. Anti-Knockoff Physical 3-Stage Scoring (`mean_physical_stage_score_3stage` = mean of Stages 1-3 / 3.0):
     Raw `gym_aloha` rewards can fire transiently if an object is knocked off the table edge (`z < 0`).
     - `AlohaTransferCube-v0`:
         * Stage 1 (`touch`): right-finger contact with `box_z >= 0.015 m`.
         * Stage 2 (`sustained_midair_lift`): `>= 15` consecutive 25 Hz ticks (`>= 0.60 s`) with `reward >= 2` and `box_z >= 0.14 m`.
         * Stage 3 (`sustained_goal_transport`): `>= 15` consecutive 25 Hz ticks (`>= 0.60 s`) with `reward >= 2`, `box_z >= 0.16 m`,
           and `||box - [0.0, 0.50, 0.24]|| <= 0.055 m`.
         * Stage 4 (full mid-air bimanual handover `reward == 4`) is deferred to learned-policy / human-demo evaluation and NOT weighted into the analytical 3-stage IK score.
     - `AlohaInsertion-v0`:
         * Stage 1 (`dual_touch`): simultaneous touch (`reward >= 1`) with `peg_z >= 0.006 m` and `soc_z >= 0.015 m`.
         * Stage 2 (`sustained_dual_midair_lift`): `>= 15` consecutive 25 Hz ticks (`>= 0.60 s`) with `reward >= 2`, `peg_z >= 0.14 m`, and `soc_z >= 0.14 m`.
         * Stage 3 (`sustained_bimanual_pre_dock_align`): `>= 15` consecutive 25 Hz ticks (`>= 0.60 s`) holding BOTH peg and socket cleanly in mid-air (`reward >= 2`, `z >= 0.14 m`) within `<= 0.065 m` of pre-docking alignment targets `[+0.095, 0.50, 0.24]` and `[-0.095, 0.50, 0.24]`.
         * Stage 4 (full peg-in-socket `reward == 4` pin insertion) is deferred to learned-policy / human-demo evaluation.
  3. Valid In-Distribution Counterfactual Action Controllability (0.0% joint-limit violations, NEVER OOD `-a`)
     verified on BOTH `AlohaTransferCube-v0` and `AlohaInsertion-v0` with `192 x 320` RGB rendering.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.stats import kendalltau

GRIP_OPEN = 1.0
GRIP_CLOSE = 0.0
TRANSFER_MEET_TARGET = np.array([0.0, 0.50, 0.24], dtype=np.float64)
INSERTION_PEG_ALIGN_TARGET = np.array([0.095, 0.50, 0.24], dtype=np.float64)
INSERTION_SOC_ALIGN_TARGET = np.array([-0.095, 0.50, 0.24], dtype=np.float64)
L_DOFS = list(range(0, 6))
R_DOFS = list(range(8, 14))


def _make_aloha_physics_and_task(env_name: str) -> tuple[Any, Any]:
    from dm_control import mujoco as dm_mujoco
    from gym_aloha.constants import ASSETS_DIR
    from gym_aloha.tasks.sim import InsertionTask, TransferCubeTask

    if env_name == "AlohaTransferCube-v0":
        xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
        return dm_mujoco.Physics.from_xml_path(str(xml_path)), TransferCubeTask()
    if env_name == "AlohaInsertion-v0":
        xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
        return dm_mujoco.Physics.from_xml_path(str(xml_path)), InsertionTask()
    raise ValueError(f"Unsupported ALOHA env: {env_name}")


def _reset_aloha(physics: Any, task: Any, env_name: str, seed: int) -> None:
    from gym_aloha.tasks.sim import BOX_POSE
    from gym_aloha.utils import sample_box_pose, sample_insertion_pose

    task.random.seed(seed)
    if env_name == "AlohaTransferCube-v0":
        BOX_POSE[0] = sample_box_pose(seed)
    else:
        BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))
    with physics.reset_context():
        task.initialize_episode(physics)


def _step_aloha_25hz(physics: Any, task: Any, act: np.ndarray) -> float:
    """Execute one 25 Hz SLWM-v4 motor tick (= 2 x 50 Hz = 20 x 0.002 s MuJoCo steps)."""
    r_max = 0.0
    for _ in range(2):
        task.before_step(act, physics)
        physics.step(10)
        r_max = max(r_max, float(task.get_reward(physics)))
    return r_max


def _pinch_midpoint(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"vx300s_{side}/10_left_gripper_finger")
    g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"vx300s_{side}/10_right_gripper_finger")
    return 0.5 * (data.geom_xpos[g1] + data.geom_xpos[g2])


def _ik_cmd_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_id: int,
    pinch_pos: np.ndarray,
    q_cmd: np.ndarray,
    dofs: list[int],
    target_pos: np.ndarray,
    roll_target: float = 0.0,
    pitch_bias: float | None = None,
    gain: float = 0.60,
    max_dq: float = 0.09,
) -> np.ndarray:
    """Gravity-compensated integral-command damped-least-squares Jacobian IK on exact finger-pad midpoint."""
    err_p = (target_pos - pinch_pos) * gain
    pn = float(np.linalg.norm(err_p))
    if pn > 0.03:
        err_p = err_p * (0.03 / pn)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, pinch_pos, body_id)
    j_pos = jacp[:, dofs]
    j_pinv = j_pos.T @ np.linalg.solve(j_pos @ j_pos.T + (0.025**2) * np.eye(3), np.eye(3))
    dq = j_pinv @ err_p
    if pitch_bias is not None:
        n_null = np.eye(6) - j_pinv @ j_pos
        dq_null = np.zeros((6,), dtype=np.float64)
        dq_null[5] = np.clip(roll_target - float(q_cmd[5]), -0.20, 0.20)
        dq_null[4] = np.clip(pitch_bias - float(q_cmd[4]), -0.20, 0.20)
        dq = np.clip(dq + n_null @ dq_null, -max_dq, max_dq)
        qn = q_cmd.astype(np.float64) + dq
    else:
        dq = np.clip(dq, -max_dq, max_dq)
        qn = q_cmd.astype(np.float64) + dq
        qn[5] = qn[5] + np.clip(roll_target - qn[5], -0.12, 0.12)

    for i, dof in enumerate(dofs):
        lo, hi = model.jnt_range[model.dof_jntid[dof]]
        qn[i] = float(np.clip(qn[i], lo, hi))
    return qn.astype(np.float32)


def _apply_arm_drift_clipped(model: mujoco.MjModel, act: np.ndarray, arm_drift_12: np.ndarray) -> np.ndarray:
    """Apply actuation drift strictly to the 12 radian arm joints clipped to `jnt_range`; grippers untouched!"""
    out = act.astype(np.float64, copy=True)
    for i, dof in enumerate(L_DOFS):
        lo, hi = model.jnt_range[model.dof_jntid[dof]]
        out[i] = float(np.clip(out[i] + float(arm_drift_12[i]), lo, hi))
    for i, dof in enumerate(R_DOFS):
        lo, hi = model.jnt_range[model.dof_jntid[dof]]
        out[7 + i] = float(np.clip(out[7 + i] + float(arm_drift_12[6 + i]), lo, hi))
    out[6] = float(np.clip(out[6], 0.0, 1.0))
    out[13] = float(np.clip(out[13], 0.0, 1.0))
    return out.astype(np.float32)


def _transfer_controller_step(
    t: int,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    *,
    b_left: int,
    b_right: int,
    ql_cmd: np.ndarray,
    qr_cmd: np.ndarray,
    box_obs: np.ndarray,
    init_box_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single shared closed-loop controller for `AlohaTransferCube-v0` (used identically by tape & closed-loop)."""
    pr = _pinch_midpoint(m, d, "right")
    pl = _pinch_midpoint(m, d, "left")
    if t < 28:
        rt, lt = np.array([box_obs[0], box_obs[1], 0.10]), np.array([-0.18, 0.50, 0.24])
        rg, lg = GRIP_OPEN, GRIP_OPEN
        ql_cmd = _ik_cmd_step(m, d, body_id=b_left, pinch_pos=pl, q_cmd=ql_cmd, dofs=L_DOFS, target_pos=lt)
        qr_cmd = _ik_cmd_step(m, d, body_id=b_right, pinch_pos=pr, q_cmd=qr_cmd, dofs=R_DOFS, target_pos=rt)
    elif t < 56:
        rt, lt = np.array([box_obs[0], box_obs[1], 0.018]), np.array([-0.18, 0.50, 0.24])
        rg, lg = GRIP_OPEN, GRIP_OPEN
        ql_cmd = _ik_cmd_step(m, d, body_id=b_left, pinch_pos=pl, q_cmd=ql_cmd, dofs=L_DOFS, target_pos=lt)
        qr_cmd = _ik_cmd_step(m, d, body_id=b_right, pinch_pos=pr, q_cmd=qr_cmd, dofs=R_DOFS, target_pos=rt)
    elif t < 74:
        rg, lg = GRIP_CLOSE, GRIP_OPEN
    elif t < 100:
        rt = np.array([init_box_obs[0], init_box_obs[1], 0.24])
        lt = np.array([-0.15, 0.50, 0.24])
        rg, lg = GRIP_CLOSE, GRIP_OPEN
        ql_cmd = _ik_cmd_step(m, d, body_id=b_left, pinch_pos=pl, q_cmd=ql_cmd, dofs=L_DOFS, target_pos=lt)
        qr_cmd = _ik_cmd_step(m, d, body_id=b_right, pinch_pos=pr, q_cmd=qr_cmd, dofs=R_DOFS, target_pos=rt)
    else:
        rt = TRANSFER_MEET_TARGET.copy()
        lt = TRANSFER_MEET_TARGET + np.array([-0.12, 0.0, 0.0])
        rg, lg = GRIP_CLOSE, GRIP_OPEN
        ql_cmd = _ik_cmd_step(m, d, body_id=b_left, pinch_pos=pl, q_cmd=ql_cmd, dofs=L_DOFS, target_pos=lt)
        qr_cmd = _ik_cmd_step(m, d, body_id=b_right, pinch_pos=pr, q_cmd=qr_cmd, dofs=R_DOFS, target_pos=rt)
    act = np.concatenate([ql_cmd, [lg], qr_cmd, [rg]]).astype(np.float32)
    return ql_cmd, qr_cmd, act


def _insertion_controller_step(
    t: int,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    *,
    b_left: int,
    b_right: int,
    ql_cmd: np.ndarray,
    qr_cmd: np.ndarray,
    peg_obs: np.ndarray,
    soc_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single shared bimanual controller for `AlohaInsertion-v0` (used identically by tape & closed-loop)."""
    pr = _pinch_midpoint(m, d, "right")
    pl = _pinch_midpoint(m, d, "left")
    if t < 35:
        rt = np.array([peg_obs[0] + 0.02, peg_obs[1], 0.13])
        lt = np.array([soc_obs[0] - 0.02, soc_obs[1], 0.13])
        rg, lg = GRIP_OPEN, GRIP_OPEN
        qr_cmd = _ik_cmd_step(m, d, body_id=b_right, pinch_pos=pr, q_cmd=qr_cmd, dofs=R_DOFS, target_pos=rt, pitch_bias=0.3)
        ql_cmd = _ik_cmd_step(m, d, body_id=b_left, pinch_pos=pl, q_cmd=ql_cmd, dofs=L_DOFS, target_pos=lt, pitch_bias=0.3)
    elif t < 68:
        rt = np.array([peg_obs[0] + 0.02, peg_obs[1], 0.005])
        lt = np.array([soc_obs[0] - 0.02, soc_obs[1], 0.035])
        rg, lg = GRIP_OPEN, GRIP_OPEN
        qr_cmd = _ik_cmd_step(m, d, body_id=b_right, pinch_pos=pr, q_cmd=qr_cmd, dofs=R_DOFS, target_pos=rt, pitch_bias=0.3)
        ql_cmd = _ik_cmd_step(m, d, body_id=b_left, pinch_pos=pl, q_cmd=ql_cmd, dofs=L_DOFS, target_pos=lt, pitch_bias=0.3)
    elif t < 86:
        rg, lg = GRIP_CLOSE, GRIP_CLOSE
    else:
        # Target pinch offsets compensating the +-0.02 m grasp grip offset so peg & socket centers align cleanly!
        rt = INSERTION_PEG_ALIGN_TARGET + np.array([0.02, 0.0, 0.015])
        lt = INSERTION_SOC_ALIGN_TARGET + np.array([-0.02, 0.0, 0.015])
        rg, lg = GRIP_CLOSE, GRIP_CLOSE
        qr_cmd = _ik_cmd_step(m, d, body_id=b_right, pinch_pos=pr, q_cmd=qr_cmd, dofs=R_DOFS, target_pos=rt, pitch_bias=0.3)
        ql_cmd = _ik_cmd_step(m, d, body_id=b_left, pinch_pos=pl, q_cmd=ql_cmd, dofs=L_DOFS, target_pos=lt, pitch_bias=0.3)
    act = np.concatenate([ql_cmd, [lg], qr_cmd, [rg]]).astype(np.float32)
    return ql_cmd, qr_cmd, act


def _get_transfer_contact_flags(physics: Any) -> dict[str, bool]:
    pairs = set()
    for i in range(physics.data.ncon):
        g1 = physics.model.id2name(physics.data.contact[i].geom1, "geom")
        g2 = physics.model.id2name(physics.data.contact[i].geom2, "geom")
        pairs.add((g1, g2))
        pairs.add((g2, g1))
    return {
        "touch_right": ("red_box", "vx300s_right/10_right_gripper_finger") in pairs
        or ("red_box", "vx300s_right/10_left_gripper_finger") in pairs,
    }


def run_closed_loop_arm(
    env_name: str,
    arm_name: str,
    *,
    n_episodes: int = 32,
    horizon_25hz: int = 160,
    perc_noise_std_m: float = 0.0,
    arm_drift_rad: float = 0.0,
    seed_base: int = 1000,
) -> dict[str, Any]:
    """Evaluate one decoupled control arm across `n_episodes` unseen seeds on `AlohaTransferCube` or `AlohaInsertion`."""
    phys, task = _make_aloha_physics_and_task(env_name)
    m, d = phys.model.ptr, phys.data.ptr
    b_right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "vx300s_right/gripper_link")
    b_left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "vx300s_left/gripper_link")

    is_transfer = env_name == "AlohaTransferCube-v0"
    if is_transfer:
        b_box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    else:
        b_peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
        b_soc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "socket")

    s1_hits = s2_hits = s3_hits = 0
    ep_scores: list[float] = []
    min_pinch_dists: list[float] = []
    max_obj_zs: list[float] = []

    for ep in range(n_episodes):
        rng = np.random.default_rng(seed_base + ep)
        _reset_aloha(phys, task, env_name, seed=seed_base + ep)
        ql_cmd = d.qpos[0:6].copy().astype(np.float32)
        qr_cmd = d.qpos[8:14].copy().astype(np.float32)
        ql_hold, qr_hold = ql_cmd.copy(), qr_cmd.copy()

        # Matched arm-joint-only disturbance (12 radian joints ONLY, grippers strictly untouched!)
        arm_drift_12 = rng.normal(0.0, arm_drift_rad, size=(12,)).astype(np.float64)
        bias_1 = rng.normal(0.0, perc_noise_std_m, size=(3,))
        bias_2 = rng.normal(0.0, perc_noise_std_m, size=(3,))
        init_box_obs = (d.xpos[b_box].copy() + bias_1) if is_transfer else None

        # Pre-compute true open-loop nominal joint tape on an unperturbed MuJoCo twin using the EXACT same controller!
        open_loop_tape: list[np.ndarray] = []
        if arm_name == "oracle_perc_open_loop_nominal_0p15rad_drift":
            p_nom, t_nom = _make_aloha_physics_and_task(env_name)
            _reset_aloha(p_nom, t_nom, env_name, seed=seed_base + ep)
            mn, dn = p_nom.model.ptr, p_nom.data.ptr
            qln, qrn = dn.qpos[0:6].copy().astype(np.float32), dn.qpos[8:14].copy().astype(np.float32)
            init_b_nom = dn.xpos[b_box].copy() if is_transfer else None
            for tt in range(horizon_25hz):
                if is_transfer:
                    qln, qrn, an = _transfer_controller_step(
                        tt, mn, dn, b_left=b_left, b_right=b_right, ql_cmd=qln, qr_cmd=qrn,
                        box_obs=dn.xpos[b_box].copy(), init_box_obs=init_b_nom,
                    )
                else:
                    qln, qrn, an = _insertion_controller_step(
                        tt, mn, dn, b_left=b_left, b_right=b_right, ql_cmd=qln, qr_cmd=qrn,
                        peg_obs=dn.xpos[b_peg].copy(), soc_obs=dn.xpos[b_soc].copy(),
                    )
                _step_aloha_25hz(p_nom, t_nom, an)
                open_loop_tape.append(an)

        hit_s1 = False
        lift_c = best_lift_c = 0
        align_c = best_align_c = 0
        min_dist = 9.9
        max_z = 0.0

        for t in range(horizon_25hz):
            pr = _pinch_midpoint(m, d, "right")
            pl = _pinch_midpoint(m, d, "left")
            if is_transfer:
                true_box = d.xpos[b_box].copy()
                min_dist = min(min_dist, float(np.linalg.norm(pr - true_box)))
                max_z = max(max_z, float(true_box[2]))
            else:
                true_peg, true_soc = d.xpos[b_peg].copy(), d.xpos[b_soc].copy()
                min_dist = min(
                    min_dist,
                    0.5 * (float(np.linalg.norm(pr - true_peg)) + float(np.linalg.norm(pl - true_soc))),
                )
                max_z = max(max_z, min(float(true_peg[2]), float(true_soc[2])))

            if arm_name == "hold_pose_null":
                act = np.concatenate([ql_hold, [GRIP_OPEN], qr_hold, [GRIP_OPEN]]).astype(np.float32)
            elif arm_name == "oracle_perc_open_loop_nominal_0p15rad_drift":
                act = _apply_arm_drift_clipped(m, open_loop_tape[t], arm_drift_12)
            else:
                if is_transfer:
                    box_obs = true_box + bias_1 + rng.normal(0.0, perc_noise_std_m * 0.2, size=(3,))
                    ql_cmd, qr_cmd, act = _transfer_controller_step(
                        t, m, d, b_left=b_left, b_right=b_right, ql_cmd=ql_cmd, qr_cmd=qr_cmd,
                        box_obs=box_obs, init_box_obs=init_box_obs,
                    )
                else:
                    peg_obs = true_peg + bias_1 + rng.normal(0.0, perc_noise_std_m * 0.2, size=(3,))
                    soc_obs = true_soc + bias_2 + rng.normal(0.0, perc_noise_std_m * 0.2, size=(3,))
                    ql_cmd, qr_cmd, act = _insertion_controller_step(
                        t, m, d, b_left=b_left, b_right=b_right, ql_cmd=ql_cmd, qr_cmd=qr_cmd,
                        peg_obs=peg_obs, soc_obs=soc_obs,
                    )
                if arm_drift_rad > 0.0:
                    act = _apply_arm_drift_clipped(m, act, arm_drift_12)

            r_step = _step_aloha_25hz(phys, task, act)
            if is_transfer:
                cf = _get_transfer_contact_flags(phys)
                cur_box = d.xpos[b_box]
                if cf["touch_right"] and cur_box[2] >= 0.015:
                    hit_s1 = True
                if r_step >= 2.0 and cur_box[2] >= 0.14:
                    lift_c += 1
                    best_lift_c = max(best_lift_c, lift_c)
                else:
                    lift_c = 0
                if (
                    r_step >= 2.0
                    and cur_box[2] >= 0.16
                    and float(np.linalg.norm(cur_box - TRANSFER_MEET_TARGET)) <= 0.055
                ):
                    align_c += 1
                    best_align_c = max(best_align_c, align_c)
                else:
                    align_c = 0
            else:
                cur_peg, cur_soc = d.xpos[b_peg], d.xpos[b_soc]
                if r_step >= 1.0 and cur_peg[2] >= 0.006 and cur_soc[2] >= 0.015:
                    hit_s1 = True
                if r_step >= 2.0 and cur_peg[2] >= 0.14 and cur_soc[2] >= 0.14:
                    lift_c += 1
                    best_lift_c = max(best_lift_c, lift_c)
                else:
                    lift_c = 0
                if (
                    r_step >= 2.0
                    and cur_peg[2] >= 0.14
                    and cur_soc[2] >= 0.14
                    and float(np.linalg.norm(cur_peg - INSERTION_PEG_ALIGN_TARGET)) <= 0.065
                    and float(np.linalg.norm(cur_soc - INSERTION_SOC_ALIGN_TARGET)) <= 0.065
                ):
                    align_c += 1
                    best_align_c = max(best_align_c, align_c)
                else:
                    align_c = 0

        s2_ok = best_lift_c >= 15
        s3_ok = best_align_c >= 15
        s1_hits += int(hit_s1)
        s2_hits += int(s2_ok)
        s3_hits += int(s3_ok)
        ep_scores.append((int(hit_s1) + int(s2_ok) + int(s3_ok)) / 3.0)
        min_pinch_dists.append(min_dist)
        max_obj_zs.append(max_z)

    return {
        "arm": arm_name,
        "n_episodes": n_episodes,
        "horizon_25hz_steps": horizon_25hz,
        "duration_sec": round(horizon_25hz * 0.04, 2),
        "perc_noise_std_m": perc_noise_std_m,
        "arm_drift_rad_12joints_only": arm_drift_rad,
        "stage1_touch_rate": round(s1_hits / n_episodes, 4),
        "stage2_sustained_midair_lift_0p6s_rate": round(s2_hits / n_episodes, 4),
        "stage3_sustained_goal_align_0p6s_rate": round(s3_hits / n_episodes, 4),
        "mean_physical_stage_score_3stage": round(float(np.mean(ep_scores)), 4),
        "mean_min_pinch_to_obj_dist_m": round(float(np.mean(min_pinch_dists)), 4),
        "mean_max_obj_z_m": round(float(np.mean(max_obj_zs)), 4),
    }


def run_in_distribution_counterfactual_audit(env_name: str, n_episodes: int = 8, horizon_25hz: int = 50) -> dict[str, Any]:
    """Audit valid in-distribution counterfactual action controllability (0% joint-limit violation, NEVER OOD `-a`)."""
    pa, ta = _make_aloha_physics_and_task(env_name)
    pb, tb = _make_aloha_physics_and_task(env_name)
    ma, da = pa.model.ptr, pa.data.ptr
    mb, db = pb.model.ptr, pb.data.ptr
    b_right = mujoco.mj_name2id(ma, mujoco.mjtObj.mjOBJ_BODY, "vx300s_right/gripper_link")
    b_left = mujoco.mj_name2id(ma, mujoco.mjtObj.mjOBJ_BODY, "vx300s_left/gripper_link")
    b_obj = mujoco.mj_name2id(ma, mujoco.mjtObj.mjOBJ_BODY, "box" if env_name == "AlohaTransferCube-v0" else "peg")

    ee_divs: list[float] = []
    rgb_l1s: list[float] = []
    joint_limit_violations = 0
    total_steps = 0

    for ep in range(n_episodes):
        _reset_aloha(pa, ta, env_name, seed=2000 + ep)
        _reset_aloha(pb, tb, env_name, seed=2000 + ep)
        ql_a, qr_a = da.qpos[0:6].copy().astype(np.float32), da.qpos[8:14].copy().astype(np.float32)
        ql_b, qr_b = db.qpos[0:6].copy().astype(np.float32), db.qpos[8:14].copy().astype(np.float32)

        for _ in range(horizon_25hz):
            obj_a = da.xpos[b_obj].copy()
            qr_a = _ik_cmd_step(ma, da, body_id=b_right, pinch_pos=_pinch_midpoint(ma, da, "right"), q_cmd=qr_a, dofs=R_DOFS, target_pos=obj_a + np.array([0.0, 0.0, 0.03]))
            act_a = np.concatenate([ql_a, [GRIP_OPEN], qr_a, [GRIP_OPEN]]).astype(np.float32)

            ql_b = _ik_cmd_step(mb, db, body_id=b_left, pinch_pos=_pinch_midpoint(mb, db, "left"), q_cmd=ql_b, dofs=L_DOFS, target_pos=np.array([-0.05, 0.50, 0.16]))
            act_b = np.concatenate([ql_b, [GRIP_OPEN], qr_b, [GRIP_OPEN]]).astype(np.float32)

            for act_v, mod_v in ((act_a, ma), (act_b, mb)):
                for i, dof in enumerate(L_DOFS + R_DOFS):
                    val = act_v[i if i < 6 else i + 1]
                    lo, hi = mod_v.jnt_range[mod_v.dof_jntid[dof]]
                    if val < lo - 1e-4 or val > hi + 1e-4:
                        joint_limit_violations += 1
            total_steps += 1
            _step_aloha_25hz(pa, ta, act_a)
            _step_aloha_25hz(pb, tb, act_b)

        ee_div = float(
            np.linalg.norm(_pinch_midpoint(ma, da, "right") - _pinch_midpoint(mb, db, "right"))
            + np.linalg.norm(_pinch_midpoint(ma, da, "left") - _pinch_midpoint(mb, db, "left"))
        )
        rgb_a = pa.render(height=192, width=320, camera_id="top").astype(np.float32)
        rgb_b = pb.render(height=192, width=320, camera_id="top").astype(np.float32)
        ee_divs.append(ee_div)
        rgb_l1s.append(float(np.mean(np.abs(rgb_a - rgb_b))))

    return {
        "n_episodes": n_episodes,
        "horizon_25hz_steps": horizon_25hz,
        "render_hw": [192, 320],
        "joint_limit_violation_rate": round(joint_limit_violations / max(1, total_steps * 24), 6),
        "mean_bimanual_pinch_divergence_m": round(float(np.mean(ee_divs)), 4),
        "mean_final_rgb_l1_0_255": round(float(np.mean(rgb_l1s)), 3),
        "in_distribution_counterfactual_valid": joint_limit_violations == 0 and float(np.mean(ee_divs)) > 0.25,
    }


def evaluate_environment(env_name: str, n_episodes: int) -> dict[str, Any]:
    arms = [
        run_closed_loop_arm(env_name, "oracle_perc_oracle_dyn_closed_loop", n_episodes=n_episodes),
        run_closed_loop_arm(env_name, "oracle_perc_oracle_dyn_closed_loop_0p15rad_drift", n_episodes=n_episodes, arm_drift_rad=0.15),
        run_closed_loop_arm(env_name, "noisy_perc_3cm_oracle_dyn_closed_loop", n_episodes=n_episodes, perc_noise_std_m=0.03),
        run_closed_loop_arm(env_name, "oracle_perc_open_loop_nominal_0p15rad_drift", n_episodes=n_episodes, arm_drift_rad=0.15),
        run_closed_loop_arm(env_name, "hold_pose_null", n_episodes=n_episodes),
    ]
    cf = run_in_distribution_counterfactual_audit(env_name, n_episodes=8, horizon_25hz=50)
    scores = [a["mean_physical_stage_score_3stage"] for a in arms]
    tau, _ = kendalltau([4, 3, 2, 1, 0], scores)

    # Strict unconfounded matched-perturbation check: closed-loop WITH 0.15 rad drift MUST decisively beat open-loop WITH 0.15 rad drift!
    env_pass = (
        arms[0]["stage2_sustained_midair_lift_0p6s_rate"] >= 0.85
        and arms[0]["stage3_sustained_goal_align_0p6s_rate"] >= 0.75
        and arms[1]["mean_physical_stage_score_3stage"] > arms[3]["mean_physical_stage_score_3stage"] + 0.35
        and arms[0]["mean_physical_stage_score_3stage"] > arms[2]["mean_physical_stage_score_3stage"] + 0.20
        and arms[4]["stage1_touch_rate"] == 0.0
        and cf["in_distribution_counterfactual_valid"]
    )
    return {
        "environment": env_name,
        "scoring_scope": "Anti-knockoff 3-stage analytical IK ceiling (Stages 1-3 / 3.0); full Stage-4 handover/pin-insertion deferred to learned-policy/demo evaluation.",
        "arms": {a["arm"]: a for a in arms},
        "matched_perturbation_0p15rad_feedback_gain": round(
            arms[1]["mean_physical_stage_score_3stage"] - arms[3]["mean_physical_stage_score_3stage"], 4
        ),
        "ranking_kendall_tau_vs_expected": round(float(tau), 4),
        "in_distribution_counterfactual_audit": cf,
        "env_gate1_pass": bool(env_pass),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v4 Gate 1 Closed-Loop MuJoCo ALOHA Oracle Ceilings (TransferCube + Insertion).")
    ap.add_argument("--n-episodes", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("reports/v4/gate1_closed_loop_robotics_receipt.json"))
    args = ap.parse_args()

    res_transfer = evaluate_environment("AlohaTransferCube-v0", args.n_episodes)
    res_insertion = evaluate_environment("AlohaInsertion-v0", args.n_episodes)
    all_pass = bool(res_transfer["env_gate1_pass"] and res_insertion["env_gate1_pass"])

    receipt = {
        "gate": "SLWM-v4 Gate 1 — Hardened Anti-Knockoff Matched-Perturbation Closed-Loop MuJoCo ALOHA Oracle Ceilings (TransferCube + Insertion)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "simulator": "MuJoCo 4.0.1 / gym_aloha 0.1.3 (25 Hz SLWM-v4 motor tick = 2x50Hz sim)",
        "libero_scope_note": "LIBERO closed-loop evaluation is deferred until a dedicated Gate 1 LIBERO oracle ceiling addendum is executed; Gate 4 mandatory closed-loop simulator floors are anchored strictly on the two certified ALOHA-Sim environments below.",
        "environments": {
            "AlohaTransferCube-v0": res_transfer,
            "AlohaInsertion-v0": res_insertion,
        },
        "gate1_closed_loop_robotics_pass": all_pass,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
