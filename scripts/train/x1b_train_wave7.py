"""Compiled JAX/XLA Training & Evaluation Runner for Wave 7 Spatial-Grid Grounding (`SLWM-v2`).

Trains Wave 7 cohorts at exact parameter parity (`453,199,938` strict parameters at `x450m`) under:
  - Strict End-Effector-Only (`ee_only`) proprioception (zero ground-truth object coordinates).
  - Pure-vision + latent 4x4 Spatial Soft-Argmax visual pose readout (`decode_spatial_vis_pose`).
  - Key-addressed 2D symbolic + visual working-memory goal latching and mid-episode overwrite.
  - Multi-shard physical xArm6 + speaker-disjoint real human spoken spectrograms + initial-scene
    contrastive 16-token command-to-displacement grounding, evaluated on `dev` and untouched `vault`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_multimodal import (
    RealMultimodalBank,
    generate_synthetic_world_batch,
)
from data.x1b_unseen_worlds import (
    mask_privileged_proprio,
    render_unseen_world_rgb,
    step_unseen_layout_world,
)
from evals.x1b_eval_wave7 import run_wave7_evaluation
from models.x1b_wave6_core import Wave6Interventions
from models.x1b_wave7_core import (
    encode_goal_query_tokens,
    encode_goal_tokens_2d,
    quantize_goal_2d_val_id,
    world_model_loss_wave7,
)
from models.x1b_world import count_parameters, init_x1b_params, make_x1b_spec
from scripts.train.x1b_train import (
    atomic_write_json,
    load_bf16_weights,
    memory_efficient_adamw_step,
    save_bf16_weights,
    save_portable_int8,
)


class Wave7GroundedRobotBank:
    """Physical xArm6 + speaker-disjoint real human spoken audio + paired counterfactual 16-token bank."""

    def __init__(self, npz_path: str | Path, split: str = "train") -> None:
        self.path = Path(npz_path)
        self.split = split
        with np.load(self.path, allow_pickle=False) as d:
            pfx = f"{split}_"
            self.video = np.asarray(d[pfx + "video"], dtype=np.uint8)
            self.spoken_audio = np.asarray(d[pfx + "spoken_audio"], dtype=np.float32)
            self.ee_xy = np.asarray(d[pfx + "ee_xy"], dtype=np.float32)
            self.action_xy = np.asarray(d[pfx + "action_xy"], dtype=np.float32)
            self.length = np.asarray(d[pfx + "length"], dtype=np.int32)
            self.true_txt16 = np.asarray(d[pfx + "true_text_ids"], dtype=np.int32)
            self.wrong_txt16 = (
                np.asarray(d[pfx + "wrong_text_ids"], dtype=np.int32)
                if (pfx + "wrong_text_ids") in d
                else np.roll(self.true_txt16, shift=7, axis=0)
            )
        self.size = int(self.video.shape[0])

    def sample_batch(self, rng: np.random.Generator, batch_size: int, seq_len: int = 32) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=(batch_size,))
        cf_idx = (idx + 7) % max(1, self.size)
        t = min(seq_len, 32)
        vid = self.video[idx, :t]
        aud = self.spoken_audio[idx, :t]
        wrong_aud = self.spoken_audio[cf_idx, :t]
        ee = self.ee_xy[idx, :t]
        u2 = self.action_xy[idx, :t]
        ln = np.clip(self.length[idx], 2, t)
        tok16 = self.true_txt16[idx]
        wtok16 = self.wrong_txt16[idx]

        txt = np.zeros((batch_size, t, 4), dtype=np.int32)
        wtxt = np.zeros((batch_size, t, 4), dtype=np.int32)
        h44 = tok16.reshape(batch_size, 4, 4)
        wh44 = wtok16.reshape(batch_size, 4, 4)
        transient = rng.random((batch_size,)) < 0.65
        for bi in range(batch_size):
            if transient[bi]:
                txt[bi, : min(4, t)] = h44[bi, : min(4, t)]
                wtxt[bi, : min(4, t)] = wh44[bi, : min(4, t)]
            else:
                reps = (t + 3) // 4
                txt[bi] = np.tile(h44[bi], (reps, 1))[:t]
                wtxt[bi] = np.tile(wh44[bi], (reps, 1))[:t]

        ee_vel = np.zeros_like(ee)
        ee_vel[:, 1:] = ee[:, 1:] - ee[:, :-1]
        moving = (np.linalg.norm(ee_vel, axis=-1) > 0.015).astype(np.float32)
        prop = np.concatenate(
            [
                ee,
                ee,
                ee_vel,
                np.zeros((batch_size, t, 1), dtype=np.float32),
                moving[..., None],
                np.zeros((batch_size, t, 8), dtype=np.float32),
            ],
            axis=-1,
        )
        next_prop = np.concatenate([prop[:, 1:], prop[:, -1:]], axis=1)
        u4 = np.concatenate([u2, np.zeros_like(u2)], axis=-1)
        prev_u4 = np.concatenate([np.zeros((batch_size, 1, 4), dtype=np.float32), u4[:, :-1]], axis=1)

        end_xy = np.array([ee[bi, ln[bi] - 1] for bi in range(batch_size)], dtype=np.float32)
        disp_xy = np.clip((end_xy - ee[:, 0]) * 1.25, -0.95, 0.95).astype(np.float32)
        dist = np.sqrt(np.sum((ee - end_xy[:, None, :]) ** 2, axis=-1))
        val = (-1.35 * dist).astype(np.float32)
        tgoal = np.broadcast_to(disp_xy[:, None, :], (batch_size, t, 2)).copy().astype(np.float32)
        zero_goal = np.zeros((batch_size, 4), dtype=np.float32)
        true_goal = np.concatenate([disp_xy, end_xy], axis=-1).astype(np.float32)
        con = np.stack([moving, np.zeros_like(moving)], axis=-1)
        valid = np.arange(t)[None, :] < ln[:, None]
        active = (valid & (np.linalg.norm(u2, axis=-1) > 1e-3)).astype(np.float32)
        mm = np.ones((batch_size, t, 4), dtype=np.float32)

        return {
            "video": vid,
            "clean_video": vid,
            "audio": aud,
            "wrong_audio": wrong_aud,
            "proprio": prop,
            "target_proprio": prop,
            "target_next_proprio": next_prop,
            "text": txt,
            "wrong_text": wtxt,
            "target_text": np.zeros((batch_size, t), dtype=np.int32),
            "action": prev_u4,
            "next_action": u4,
            "target_action": u4,
            "target_contacts": con,
            "target_value": val,
            "target_commit": np.zeros((batch_size, t), dtype=np.int32),
            "target_goal_xy": tgoal,
            "goal": zero_goal,
            "true_goal": true_goal,
            "modality_mask": mm,
            "full_modality_mask": mm,
            "has_av_pair": np.ones((batch_size,), dtype=np.float32),
            "action_supervision_mask": active,
            "valid": valid,
            "scenario": np.full((batch_size,), -2, dtype=np.int32),
        }


def mix_wave7_batches(*parts: dict[str, Any]) -> dict[str, jax.Array]:
    """Concatenate real multimodal, physical spoken xArm6, and synthetic episodes preserving counterfactual keys."""
    active_parts = [p for p in parts if p is not None]
    base_keys = (
        "video",
        "clean_video",
        "audio",
        "proprio",
        "target_proprio",
        "target_next_proprio",
        "text",
        "target_text",
        "action",
        "next_action",
        "target_action",
        "target_contacts",
        "target_value",
        "target_commit",
        "target_goal_xy",
        "goal",
        "true_goal",
        "modality_mask",
        "full_modality_mask",
        "has_av_pair",
        "action_supervision_mask",
        "valid",
        "scenario",
    )
    out = {k: jnp.concatenate([jnp.asarray(p[k]) for p in active_parts], axis=0) for k in base_keys}
    if any("in_domain_bar_hx" in p for p in active_parts):
        out["in_domain_bar_hx"] = jnp.concatenate(
            [
                jnp.asarray(p["in_domain_bar_hx"], dtype=jnp.float32)
                if "in_domain_bar_hx" in p
                else jnp.zeros((p["video"].shape[0],), dtype=jnp.float32)
                for p in active_parts
            ],
            axis=0,
        )
        out["in_domain_bar_hy"] = jnp.concatenate(
            [
                jnp.asarray(p["in_domain_bar_hy"], dtype=jnp.float32)
                if "in_domain_bar_hy" in p
                else jnp.zeros((p["video"].shape[0],), dtype=jnp.float32)
                for p in active_parts
            ],
            axis=0,
        )
    if any("wrong_text" in p for p in active_parts):
        wt_list = [
            jnp.asarray(p["wrong_text"]) if "wrong_text" in p else jnp.asarray(p["text"])
            for p in active_parts
        ]
        wa_list = [
            jnp.asarray(p["wrong_audio"]) if "wrong_audio" in p else jnp.roll(jnp.asarray(p["audio"]), shift=1, axis=0)
            for p in active_parts
        ]
        out["wrong_text"] = jnp.concatenate(wt_list, axis=0)
        out["wrong_audio"] = jnp.concatenate(wa_list, axis=0)
    return out


def generate_in_domain_deprivileged_sim_batch_v7(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int = 32,
    vocab_size: int = 4096,
) -> dict[str, jax.Array]:
    """Generate strictly in-domain (`open` or single `x=0` bar) episodes + associative recall (`50/50` split).

    Includes:
      - 50% `generate_synthetic_world_batch` (preserving exact Wave 6 associative overwrite recall + commit gating).
      - 50% in-domain `render_unseen_world_rgb` (`open` or single `x=0, cy=0` central bar ONLY) spanning:
          * Goal-Switch + orthogonal distractor key (`scen = 5`)
          * Transient single-goal working memory (`scen = 1`)
          * Independent free-motion `(ee, obj)` sweeps + obstacle collision exploration (`action_supervision_mask = 0`)
            so `decode_spatial_vis_pose` separates amber `ee` from cyan `obj` and `predict_latent_step` learns
            non-contact zero-motion & obstacle stalling without any unseen-layout leakage.
    """
    k_base, k_w, k_mode, k_bar, k_exp = jax.random.split(rng_key, 5)
    base = generate_synthetic_world_batch(k_base, batch_size, seq_len=seq_len, vocab_size=vocab_size)

    n_u = batch_size // 2
    if n_u == 0:
        return base

    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k_w, 7)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (n_u,)), 1.0, -1.0)
    obj0 = jnp.stack([-0.54 * side, jax.random.uniform(k2, (n_u,), minval=-0.38, maxval=0.38)], axis=-1)

    exp_u = jax.random.uniform(k_exp, (n_u,))
    is_free_sweep = exp_u < 0.28
    is_wall_collide = (~is_free_sweep) & (exp_u < 0.55)
    is_exploratory = is_free_sweep | is_wall_collide

    ee_near = jnp.clip(
        obj0 + jnp.stack([-0.16 * side, jax.random.uniform(k3, (n_u,), minval=-0.12, maxval=0.12)], axis=-1),
        -0.88,
        0.88,
    )
    ee_free = jax.random.uniform(k7, (n_u, 2), minval=-0.78, maxval=0.78)
    ee0 = jnp.where(is_free_sweep[:, None], ee_free, ee_near)

    targ_a = jnp.stack([0.54 * side, jax.random.uniform(k4, (n_u,), minval=-0.42, maxval=0.42)], axis=-1)
    targ_b = jnp.stack([0.54 * side, jax.random.uniform(k5, (n_u,), minval=-0.42, maxval=0.42)], axis=-1)
    dist_xy = jax.random.uniform(k6, (n_u, 2), minval=-0.65, maxval=0.65)

    mode_u = jax.random.uniform(k_mode, (n_u,))
    is_switch_ep = (mode_u < 0.48) & (seq_len >= 20)
    is_trans_ep = (~is_switch_ep) & (mode_u < 0.88)
    has_bar = jax.random.bernoulli(k_bar, 0.72, (n_u,))

    # STRICTLY in-domain: ONLY open arena or single central bar at x=0.0, cy=0.0 (varying thickness hx/hy)!
    k_hx, k_hy = jax.random.split(k_bar)
    bar_hx = jax.random.uniform(k_hx, (n_u,), minval=0.022, maxval=0.085)
    bar_hy = jax.random.uniform(k_hy, (n_u,), minval=0.20, maxval=0.38)
    cx = jnp.zeros((n_u, 3), dtype=jnp.float32)
    cy = jnp.zeros((n_u, 3), dtype=jnp.float32)
    hx = jnp.stack([jnp.where(has_bar, bar_hx, 0.0), jnp.zeros((n_u,)), jnp.zeros((n_u,))], axis=-1)
    hy = jnp.stack([jnp.where(has_bar, bar_hy, 0.0), jnp.zeros((n_u,)), jnp.zeros((n_u,))], axis=-1)

    st0 = {
        "ee_xy": ee0,
        "ee_vel": jnp.zeros((n_u, 2), dtype=jnp.float32),
        "obj_xy": obj0,
        "obj_vel": jnp.zeros((n_u, 2), dtype=jnp.float32),
        "box_cx": cx,
        "box_cy": cy,
        "box_hx": hx,
        "box_hy": hy,
        "mass_scale": jnp.ones((n_u,), dtype=jnp.float32),
        "damping": jnp.full((n_u,), 0.72, dtype=jnp.float32),
    }

    tok_ga = encode_goal_tokens_2d(targ_a, is_distractor=False)
    tok_gb = encode_goal_tokens_2d(targ_b, is_distractor=False)
    tok_dist = encode_goal_tokens_2d(dist_xy, is_distractor=True)
    tok_query = encode_goal_query_tokens((n_u,))
    val_ga = quantize_goal_2d_val_id(targ_a)
    val_gb = quantize_goal_2d_val_id(targ_b)

    t_sw = 12
    t_dist = 20
    sweep_ph = jnp.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False, dtype=jnp.float32)

    def sim_step(carry, t_idx):
        st, prev_u, prev_c, rk = carry
        rk, nk1, nk2 = jax.random.split(rk, 3)

        active_targ = jnp.where(is_switch_ep[:, None] & (t_idx >= t_sw), targ_b, targ_a)
        active_val = jnp.where(is_switch_ep & (t_idx >= t_sw), val_gb, val_ga)

        write_a = t_idx < 4
        write_b = is_switch_ep & (t_idx >= t_sw) & (t_idx < t_sw + 4)
        write_d = is_switch_ep & (t_idx >= t_dist) & (t_idx < t_dist + 3)

        show_g = write_a | write_b | ((~is_switch_ep) & (~is_trans_ep))
        txt_t = jnp.where(
            write_a,
            tok_ga,
            jnp.where(write_b[:, None], tok_gb, jnp.where(write_d[:, None], tok_dist, tok_query)),
        )
        is_query_tick = (~write_a) & (~write_b) & (~write_d) & (is_switch_ep | is_trans_ep)
        ttxt_t = jnp.where(is_query_tick, active_val, 0)

        vid_t = render_unseen_world_rgb(st, active_targ, jnp.broadcast_to(show_g, (n_u,)))
        ee_p = mask_privileged_proprio(st["ee_xy"], st["ee_vel"], prev_c)
        full_tprop = ee_p.at[:, 2:4].set(st["obj_xy"]).at[:, 6:8].set(st["obj_vel"])

        # 1. Expert detour push
        det_y = jnp.where(st["obj_xy"][:, 1] >= 0.0, 0.54, -0.54)
        blocked = has_bar & ((st["obj_xy"][:, 0] * active_targ[:, 0]) < 0.0) & (jnp.abs(st["obj_xy"][:, 1]) < 0.39)
        wp = jnp.where(blocked[:, None], jnp.stack([jnp.zeros((n_u,)), det_y], axis=-1), active_targ)
        to_wp = wp - st["obj_xy"]
        u_dir = to_wp * jax.lax.rsqrt(jnp.sum(to_wp * to_wp, axis=-1, keepdims=True) + 1e-6)
        stage = st["obj_xy"] - 0.17 * u_dir
        rel = st["ee_xy"] - st["obj_xy"]
        aligned = jnp.sum(rel * u_dir, axis=-1, keepdims=True) <= -0.08
        aim = jnp.where(aligned, st["obj_xy"] + 0.08 * u_dir, stage)
        u_expert_xy = jnp.clip(3.2 * (aim - st["ee_xy"]) + 0.10 * jax.random.normal(nk1, (n_u, 2)), -1.0, 1.0)

        # 2. Naive straight push into obstacle (teaches stalling & contact_logits[:, 1]!)
        to_direct = active_targ - st["obj_xy"]
        dir_d = to_direct * jax.lax.rsqrt(jnp.sum(to_direct * to_direct, axis=-1, keepdims=True) + 1e-6)
        rel_d = st["ee_xy"] - st["obj_xy"]
        al_d = jnp.sum(rel_d * dir_d, axis=-1, keepdims=True) <= -0.08
        aim_d = jnp.where(al_d, st["obj_xy"] + 0.08 * dir_d, st["obj_xy"] - 0.17 * dir_d)
        u_collide_xy = jnp.clip(3.2 * (aim_d - st["ee_xy"]) + 0.15 * jax.random.normal(nk2, (n_u, 2)), -1.0, 1.0)

        # 3. Free-motion sweep away from object (separates amber EE from cyan Obj & teaches non-contact zero delta!)
        sw_goal = jnp.stack(
            [0.70 * jnp.cos(sweep_ph + 0.27 * t_idx), 0.70 * jnp.sin(sweep_ph + 0.23 * t_idx)],
            axis=-1,
        )
        u_sweep_xy = jnp.clip(3.2 * (sw_goal - st["ee_xy"]), -1.0, 1.0)

        u_exec_xy = jnp.where(
            is_free_sweep[:, None],
            u_sweep_xy,
            jnp.where(is_wall_collide[:, None], u_collide_xy, u_expert_xy),
        )
        u_exec_4 = jnp.concatenate([u_exec_xy, jnp.zeros_like(u_exec_xy)], axis=-1)
        u_exp_4 = jnp.concatenate([u_expert_xy, jnp.zeros_like(u_expert_xy)], axis=-1)

        st_next, c_next = step_unseen_layout_world(st, u_exec_4)
        next_tprop = (
            mask_privileged_proprio(st_next["ee_xy"], st_next["ee_vel"], c_next)
            .at[:, 2:4]
            .set(st_next["obj_xy"])
            .at[:, 6:8]
            .set(st_next["obj_vel"])
        )
        near_box = has_bar & (
            ((jnp.abs(st_next["obj_xy"][:, 0]) < bar_hx + 0.11) & (jnp.abs(st_next["obj_xy"][:, 1]) < bar_hy + 0.06))
            | ((jnp.abs(st_next["ee_xy"][:, 0]) < bar_hx + 0.09) & (jnp.abs(st_next["ee_xy"][:, 1]) < bar_hy + 0.05))
        )
        con_vec = jnp.stack([c_next, near_box.astype(jnp.float32)], axis=-1)
        val_t = -1.35 * jnp.sqrt(jnp.sum((st["obj_xy"] - active_targ) ** 2, axis=-1) + 1e-6) - 0.65 * near_box.astype(
            jnp.float32
        )
        return (st_next, u_exec_4, c_next, rk), (
            vid_t,
            ee_p,
            full_tprop,
            next_tprop,
            txt_t,
            ttxt_t,
            prev_u,
            u_exec_4,
            u_exp_4,
            con_vec,
            val_t,
            active_targ,
        )

    _, (v_s, p_s, tp_s, ntp_s, txt_s, ttxt_s, pu_s, uexec_s, uexp_s, c_s, val_s, tg_s) = jax.lax.scan(
        sim_step,
        (st0, jnp.zeros((n_u, 4)), jnp.zeros((n_u,)), k_bar),
        jnp.arange(seq_len, dtype=jnp.int32),
    )
    mv = lambda a: jnp.swapaxes(a, 0, 1)
    v_bt = mv(v_s)
    g_ext = jnp.where((is_switch_ep | is_trans_ep)[:, None], 0.0, targ_a)
    g4 = jnp.concatenate([jnp.zeros_like(g_ext), g_ext], axis=-1)
    scen_u = jnp.where(is_switch_ep, 5, jnp.where(is_trans_ep, 1, 0)).astype(jnp.int32)
    act_sup = jnp.broadcast_to((~is_exploratory)[:, None].astype(jnp.float32), (n_u, seq_len))

    rep = {
        "video": v_bt,
        "clean_video": v_bt,
        "proprio": mv(p_s),
        "target_proprio": mv(tp_s),
        "target_next_proprio": mv(ntp_s),
        "text": mv(txt_s),
        "target_text": mv(ttxt_s),
        "action": mv(pu_s),
        "next_action": mv(uexec_s),
        "target_action": mv(uexp_s),
        "target_contacts": mv(c_s),
        "target_value": mv(val_s),
        "target_commit": jnp.zeros((n_u, seq_len), dtype=jnp.int32),
        "target_goal_xy": mv(tg_s),
        "goal": g4,
        "true_goal": jnp.concatenate([jnp.zeros_like(targ_a), targ_a], axis=-1),
        "action_supervision_mask": act_sup,
        "scenario": scen_u,
    }
    out = {}
    for k, bv in base.items():
        if k in rep:
            out[k] = jnp.concatenate([rep[k], bv[n_u:]], axis=0)
        else:
            out[k] = bv
    base_scen = base["scenario"][n_u:]
    base_bar = (base_scen >= 0) & (base_scen != 4) & (base["target_contacts"][n_u:, 0, 1] > 0.5)
    out["in_domain_bar_hx"] = jnp.concatenate(
        [jnp.where(has_bar, bar_hx, 0.0), jnp.where(base_bar, 0.025, 0.0)], axis=0
    ).astype(jnp.float32)
    out["in_domain_bar_hy"] = jnp.concatenate(
        [jnp.where(has_bar, bar_hy, 0.0), jnp.where(base_bar, 0.38, 0.0)], axis=0
    ).astype(jnp.float32)
    return out


def train_and_evaluate_wave7(
    *,
    tag: str,
    profile: str,
    arch: str,
    regularizer: str,
    key_shield: bool,
    zerolag_bypass: bool,
    decoupled_heads: bool,
    seed: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    out_dir: Path,
    real_bank_path: str | None = None,
    grounded_robot_path: str | None = None,
    vault_robot_path: str | None = None,
    save_bf16: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one Wave 7 cohort seed and run all 5 Spatial-Grid De-Privileged Grounded Evaluation Panels."""
    job_dir = out_dir / tag
    done_path = job_dir / "done.json"
    if done_path.is_file():
        print(f"[SKIP] {tag} already completed at {done_path}", flush=True)
        return json.loads(done_path.read_text())

    job_dir.mkdir(parents=True, exist_ok=True)
    spec = make_x1b_spec(profile=profile, arch=arch)
    interventions = Wave6Interventions(
        key_shield=key_shield,
        zerolag_bypass=zerolag_bypass,
        decoupled_heads=decoupled_heads,
        regularizer=regularizer,
    )
    peak_lr = {"x_tiny": 9e-4, "x150m": 3.0e-4, "x450m": 2.7e-4, "x1b": 2.4e-4}[profile]

    int8_path = job_dir / "portable_int8.npz"
    bf16_path = job_dir / "weights_bf16.npz"
    resume_bf16_path = job_dir / ".resume_weights_bf16.npz"

    if bf16_path.is_file() or resume_bf16_path.is_file():
        src_ckpt = bf16_path if bf16_path.is_file() else resume_bf16_path
        params = load_bf16_weights(src_ckpt)
        param_counts = count_parameters(params)
        train_wall_s = 0.0
    else:
        params = init_x1b_params(spec, seed=seed)
        param_counts = count_parameters(params)
        ema_params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
        m_state = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)
        v_state = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)

    config_doc = {
        "tag": tag,
        "wave": "wave7_spatial_grid_clincher",
        "spec": asdict(spec),
        "interventions": asdict(interventions),
        "de_privileged_proprio": "ee_only",
        "visual_pose_readout": "pure_vision_spatial_soft_argmax_4x4",
        "planner": "spatial_geometry_free_subgoal_mppi",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "peak_lr": peak_lr,
        "save_bf16": bool(save_bf16),
        "parameter_counts": param_counts,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }
    config_hash = hashlib.sha256(json.dumps(config_doc, sort_keys=True).encode()).hexdigest()
    config_doc["config_sha256"] = config_hash
    atomic_write_json(job_dir / "config.json", config_doc)

    print(
        f"[START W7] {tag} | profile={profile} arch={arch} reg={regularizer} "
        f"interventions={asdict(interventions)} | params={param_counts['strict_total']:,} | backend={jax.default_backend()}",
        flush=True,
    )

    if not (bf16_path.is_file() or resume_bf16_path.is_file()):
        real_train = RealMultimodalBank(real_bank_path, split="train") if real_bank_path else None
        grounded_train = Wave7GroundedRobotBank(grounded_robot_path, split="train") if grounded_robot_path else None
        np_rng = np.random.default_rng(seed + 70013)

        def _step_kernel(p, ema_p, m, v, batch_in, step_idx, rk_loss):
            (loss, metrics), grads = jax.value_and_grad(
                lambda w: world_model_loss_wave7(w, ema_p, spec, batch_in, interventions, rk_loss),
                has_aux=True,
            )(p)
            p_new, m_new, v_new, gnorm, lr_now = memory_efficient_adamw_step(
                p, m, v, grads, step_idx, steps, peak_lr
            )
            ema_decay = 0.992 + 0.006 * (step_idx.astype(jnp.float32) / float(max(1, steps)))
            ema_new = jax.tree.map(
                lambda e, w: (ema_decay * e.astype(jnp.float32) + (1.0 - ema_decay) * w.astype(jnp.float32)).astype(
                    jnp.bfloat16
                ),
                ema_p,
                p_new,
            )
            return p_new, ema_new, m_new, v_new, {**metrics, "grad_norm": gnorm, "lr": lr_now}

        compiled_step = jax.jit(_step_kernel, donate_argnums=(0, 1, 2, 3))
        has_real = (real_train is not None) or (grounded_train is not None)
        synth_n = (batch_size * 2) // 3 if has_real else batch_size
        synth_n = min(max(2, synth_n), batch_size - (1 if has_real else 0))
        rem = batch_size - synth_n
        if (real_train is not None) and (grounded_train is not None):
            rob_n = max(1, (2 * rem + 1) // 3)
            real_n = rem - rob_n
        elif grounded_train is not None:
            rob_n, real_n = rem, 0
        else:
            rob_n, real_n = 0, rem

        jit_sim = jax.jit(
            lambda rk: generate_in_domain_deprivileged_sim_batch_v7(
                rk, synth_n, seq_len=seq_len, vocab_size=spec.vocab_size
            )
        )

        t_start = time.monotonic()
        log_file = job_dir / "train.jsonl"
        for step in range(steps):
            rk_sim, rk_loss = jax.random.split(jax.random.fold_in(jax.random.key(seed), step))
            s_batch = jit_sim(rk_sim)
            r_batch = real_train.sample_batch(np_rng, real_n, seq_len=seq_len) if (real_train and real_n > 0) else None
            g_batch = (
                grounded_train.sample_batch(np_rng, rob_n, seq_len=seq_len) if (grounded_train and rob_n > 0) else None
            )
            batch = mix_wave7_batches(r_batch, g_batch, s_batch) if has_real else s_batch

            params, ema_params, m_state, v_state, metrics = compiled_step(
                params, ema_params, m_state, v_state, batch, jnp.asarray(step, dtype=jnp.int32), rk_loss
            )
            if step == 0 or (step + 1) % 50 == 0 or (step + 1) == steps:
                m_host = {k: float(v) for k, v in jax.device_get(metrics).items()}
                if not np.isfinite(m_host["loss"]):
                    raise RuntimeError(f"Non-finite Wave 7 loss on {tag} at step {step + 1}: {m_host}")
                rec_line = {"step": step + 1, "elapsed_s": round(time.monotonic() - t_start, 2), **m_host}
                with log_file.open("a") as lf:
                    lf.write(json.dumps(rec_line, sort_keys=True) + "\n")
                atomic_write_json(out_dir / f"{tag}-progress.json", rec_line)
                if step == 0 or (step + 1) % 200 == 0 or (step + 1) == steps:
                    print(
                        f"[{tag}] step {step + 1}/{steps} | loss={m_host['loss']:.4f} jepa={m_host['jepa']:.4f} "
                        f"vee={m_host['vis_ee_mse']:.4f} vobj={m_host['vis_obj_mse']:.4f} "
                        f"goal={m_host['goal_mse']:.4f} rank={m_host['rank_margin']:.4f} pr={m_host['pr']:.3f}",
                        flush=True,
                    )

        train_wall_s = time.monotonic() - t_start
        del ema_params, m_state, v_state
        save_bf16_weights(bf16_path if save_bf16 else resume_bf16_path, params)

    int8_sha = save_portable_int8(int8_path, params)
    bf16_sha = save_bf16_weights(bf16_path, params) if (save_bf16 and not bf16_path.is_file()) else None
    if save_bf16 and bf16_path.is_file() and bf16_sha is None:
        from scripts.train.x1b_train import sha256_file
        bf16_sha = sha256_file(bf16_path)

    t_ev0 = time.monotonic()
    eval_res = run_wave7_evaluation(
        params,
        spec,
        interventions,
        seed=seed,
        grounded_npz_path=grounded_robot_path,
        vault_npz_path=vault_robot_path,
        smoke=smoke,
    )
    eval_wall_s = time.monotonic() - t_ev0
    atomic_write_json(job_dir / "eval_metrics.json", eval_res)

    eval_summary = {
        "unseen_mppi_success": eval_res["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_success"],
        "explicit_latch_reactive_success": eval_res["unseen_layout_deprivileged_mpc"]["explicit_latch_reactive_success"],
        "open_loop_policy_success": eval_res["unseen_layout_deprivileged_mpc"]["open_loop_policy_success"],
        "vis_decoded_obj_rmse": eval_res["unseen_layout_deprivileged_mpc"]["vis_decoded_obj_rmse"],
        "vis_decoded_ee_rmse": eval_res["unseen_layout_deprivileged_mpc"]["vis_decoded_ee_rmse"],
        "T160_post_perturb_goal_b_success": eval_res["beyond_window_T160_distractor_switch"][
            "T160_post_perturbation_goal_b_success"
        ],
        "latch_mse_inside_W64": eval_res["beyond_window_T160_distractor_switch"]["latch_mse_inside_W64_t24_64"],
        "latch_mse_beyond_W64": eval_res["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"],
        "symbolic_key_visual_drift": eval_res["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"],
        "dyn_visual_assoc_T1024": eval_res["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"],
        "unseen_ood_corruption_auroc": eval_res["unseen_ood_sensor_corruptions"]["mean_unseen_ood_corruption_auroc"],
        "vault_step4_mm": eval_res["untouched_vault_spoken_robot"].get("step4_mm"),
        "vault_spoken_disc_acc": eval_res["untouched_vault_spoken_robot"].get("spoken_16tok_discrimination_acc"),
    }
    receipt = {
        "tag": tag,
        "wave": "wave7_spatial_grid_clincher",
        "profile": profile,
        "arch": arch,
        "interventions": asdict(interventions),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "backend": jax.default_backend(),
        "parameter_counts": param_counts,
        "config_sha256": config_hash,
        "portable_int8_sha256": int8_sha,
        "portable_int8_bytes": int8_path.stat().st_size,
        "weights_bf16_sha256": bf16_sha,
        "weights_bf16_bytes": bf16_path.stat().st_size if bf16_path.is_file() else 0,
        "train_wall_s": round(train_wall_s, 2),
        "eval_wall_s": round(eval_wall_s, 2),
        "eval_summary": eval_summary,
    }
    atomic_write_json(job_dir / "receipt.json", receipt)
    resume_bf16_path.unlink(missing_ok=True)
    atomic_write_json(done_path, receipt)
    print(f"[DONE W7] {tag} | summary={json.dumps(eval_summary, sort_keys=True)}", flush=True)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--profile", default="x450m")
    ap.add_argument("--arch", default="slwm_x1b")
    ap.add_argument("--regularizer", default="epps_pulley_sigreg")
    ap.add_argument("--no-key-shield", action="store_true")
    ap.add_argument("--no-zerolag-bypass", action="store_true")
    ap.add_argument("--no-decoupled-heads", action="store_true")
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--steps", type=int, default=2200)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--real-bank-path", default=None)
    ap.add_argument("--grounded-robot-path", default=None)
    ap.add_argument("--vault-robot-path", default=None)
    ap.add_argument("--save-bf16", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_and_evaluate_wave7(
        tag=args.tag,
        profile=args.profile,
        arch=args.arch,
        regularizer=args.regularizer,
        key_shield=(not args.no_key_shield),
        zerolag_bypass=(not args.no_zerolag_bypass),
        decoupled_heads=(not args.no_decoupled_heads),
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        out_dir=args.out_dir,
        real_bank_path=args.real_bank_path,
        grounded_robot_path=args.grounded_robot_path,
        vault_robot_path=args.vault_robot_path,
        save_bf16=args.save_bf16,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
