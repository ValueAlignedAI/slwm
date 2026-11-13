"""Compiled JAX/XLA Training & Zero-Compromise Evaluation Runner for Wave 6 (`SLWM-v2`).

Trains all 8 Wave 6 cohorts (single-factor Leave-One-Out causal isolation + Epps-Pulley SIGReg vs VICReg +
bounded-memory O(1) baselines) under strict End-Effector-Only (`ee_only`) proprioception, multi-shard
physical xArm6 + real human spoken spectrograms + untruncated 16-token instructions, and zero geometric
planner priors. Evaluates across all 5 Wave 6 panels including the untouched final vault (`00147`, `00419`).
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
    mix_real_and_synthetic_batches,
)
from data.x1b_unseen_worlds import (
    mask_privileged_proprio,
    render_unseen_world_rgb,
    step_unseen_layout_world,
)
from evals.x1b_eval_wave6 import run_wave6_evaluation
from models.x1b_wave6_core import Wave6Interventions, world_model_loss_wave6
from models.x1b_world import count_parameters, init_x1b_params, make_x1b_spec
from scripts.train.x1b_train import (
    atomic_write_json,
    load_bf16_weights,
    memory_efficient_adamw_step,
    save_bf16_weights,
    save_portable_int8,
)


class Wave6GroundedRobotBank:
    """Multi-shard physical xArm6 + speaker-disjoint real human spoken audio + untruncated 16-token bank."""

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
        self.size = int(self.video.shape[0])

    def sample_batch(self, rng: np.random.Generator, batch_size: int, seq_len: int = 32) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=(batch_size,))
        t = min(seq_len, 32)
        vid = self.video[idx, :t]
        aud = self.spoken_audio[idx, :t]
        ee = self.ee_xy[idx, :t]
        u2 = self.action_xy[idx, :t]
        ln = np.clip(self.length[idx], 2, t)
        tok16 = self.true_txt16[idx]

        # Stream untruncated 16-token instructions across ticks 0..3 (50% transient 0.4s, 50% tiled)
        txt = np.zeros((batch_size, t, 4), dtype=np.int32)
        h44 = tok16.reshape(batch_size, 4, 4)
        transient = rng.random((batch_size,)) < 0.5
        for bi in range(batch_size):
            if transient[bi]:
                txt[bi, : min(4, t)] = h44[bi, : min(4, t)]
            else:
                reps = (t + 3) // 4
                txt[bi] = np.tile(h44[bi], (reps, 1))[:t]

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
        dist = np.sqrt(np.sum((ee - end_xy[:, None, :]) ** 2, axis=-1))
        val = (-1.35 * dist).astype(np.float32)
        tgoal = np.broadcast_to(end_xy[:, None, :], (batch_size, t, 2)).copy().astype(np.float32)
        zero_goal = np.zeros((batch_size, 4), dtype=np.float32)
        true_goal = np.concatenate([end_xy - ee[:, 0], end_xy], axis=-1).astype(np.float32)
        con = np.stack([moving, np.zeros_like(moving)], axis=-1)
        valid = np.arange(t)[None, :] < ln[:, None]
        active = (valid & (np.linalg.norm(u2, axis=-1) > 1e-3)).astype(np.float32)
        mm = np.ones((batch_size, t, 4), dtype=np.float32)

        return {
            "video": vid,
            "clean_video": vid,
            "audio": aud,
            "proprio": prop,
            "target_proprio": prop,
            "target_next_proprio": next_prop,
            "text": txt,
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


def generate_in_domain_deprivileged_sim_batch(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int = 32,
    vocab_size: int = 4096,
) -> dict[str, jax.Array]:
    """Generate in-domain open/single-central-bar episodes + associative recall (ZERO unseen test layouts!).

    Strictly excludes all 4 Wave 6 held-out test layout families (`off_center_wall` at +/-0.24, `dual_chicane`,
    `multi_block_clutter`, and `2.2x` mass / ice `dynamics_shift`), excludes magenta distractors, and excludes
    all 4 unseen OOD camera corruptions.
    """
    k_base, k_w, k_goal, k_bar = jax.random.split(rng_key, 4)
    base = generate_synthetic_world_batch(k_base, batch_size, seq_len=seq_len, vocab_size=vocab_size)

    # Half of sim batch uses the unified RGB box renderer on strictly in-domain open/central-bar scenes
    n_u = batch_size // 2
    if n_u == 0:
        return base

    k1, k2, k3, k4 = jax.random.split(k_w, 4)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (n_u,)), 1.0, -1.0)
    obj0 = jnp.stack([-0.52 * side, jax.random.uniform(k2, (n_u,), minval=-0.32, maxval=0.32)], axis=-1)
    ee0 = jnp.clip(obj0 + jnp.stack([-0.16 * side, jax.random.uniform(k3, (n_u,), minval=-0.10, maxval=0.10)], axis=-1), -0.88, 0.88)
    targ = jnp.stack([0.52 * side, jax.random.uniform(k4, (n_u,), minval=-0.35, maxval=0.35)], axis=-1)
    has_bar = jax.random.bernoulli(k_bar, 0.5, (n_u,))

    # ONLY open arena or single central bar at x=0.0 (zero unseen layouts!)
    cx = jnp.zeros((n_u, 3), dtype=jnp.float32)
    cy = jnp.zeros((n_u, 3), dtype=jnp.float32)
    hx = jnp.stack([jnp.where(has_bar, 0.025, 0.0), jnp.zeros((n_u,)), jnp.zeros((n_u,))], axis=-1)
    hy = jnp.stack([jnp.where(has_bar, 0.34, 0.0), jnp.zeros((n_u,)), jnp.zeros((n_u,))], axis=-1)

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
    transient_ep = jax.random.bernoulli(k_goal, 0.5, (n_u,))

    def sim_step(carry, t_idx):
        st, prev_u, prev_c, rk = carry
        rk, nk = jax.random.split(rk)
        show_g = (~transient_ep) | (t_idx < 4)
        vid_t = render_unseen_world_rgb(st, targ, show_g)
        ee_p = mask_privileged_proprio(st["ee_xy"], st["ee_vel"], prev_c)
        full_tprop = ee_p.at[:, 2:4].set(st["obj_xy"]).at[:, 6:8].set(st["obj_vel"])

        # Simple exploratory push demonstration (noisy alignment toward target or open gap)
        det_y = jnp.where(st["obj_xy"][:, 1] >= 0.0, 0.52, -0.52)
        blocked = has_bar & ((st["obj_xy"][:, 0] * targ[:, 0]) < 0.0) & (jnp.abs(st["obj_xy"][:, 1]) < 0.38)
        wp = jnp.where(blocked[:, None], jnp.stack([jnp.zeros((n_u,)), det_y], axis=-1), targ)
        to_wp = wp - st["obj_xy"]
        u_dir = to_wp * jax.lax.rsqrt(jnp.sum(to_wp * to_wp, axis=-1, keepdims=True) + 1e-6)
        stage = st["obj_xy"] - 0.16 * u_dir
        rel = st["ee_xy"] - st["obj_xy"]
        aligned = (jnp.sum(rel * u_dir, axis=-1, keepdims=True) <= -0.08)
        aim = jnp.where(aligned, st["obj_xy"] + 0.08 * u_dir, stage)
        noise = 0.18 * jax.random.normal(nk, (n_u, 2))
        u_xy = jnp.clip(3.2 * (aim - st["ee_xy"]) + noise, -1.0, 1.0)
        u4 = jnp.concatenate([u_xy, jnp.zeros_like(u_xy)], axis=-1)

        st_next, c_next = step_unseen_layout_world(st, u4)
        next_tprop = mask_privileged_proprio(st_next["ee_xy"], st_next["ee_vel"], c_next).at[:, 2:4].set(
            st_next["obj_xy"]
        ).at[:, 6:8].set(st_next["obj_vel"])
        near_box = has_bar & ((jnp.abs(st_next["obj_xy"][:, 0]) < 0.14) | (jnp.abs(st_next["ee_xy"][:, 0]) < 0.12)) & (
            jnp.abs(st_next["obj_xy"][:, 1]) < 0.38
        )
        con_vec = jnp.stack([c_next, near_box.astype(jnp.float32)], axis=-1)
        val_t = -1.35 * jnp.sqrt(jnp.sum((st["obj_xy"] - targ) ** 2, axis=-1) + 1e-6)
        return (st_next, u4, c_next, rk), (vid_t, ee_p, full_tprop, next_tprop, prev_u, u4, con_vec, val_t)

    _, (v_s, p_s, tp_s, ntp_s, pu_s, u_s, c_s, val_s) = jax.lax.scan(
        sim_step,
        (st0, jnp.zeros((n_u, 4)), jnp.zeros((n_u,)), k_goal),
        jnp.arange(seq_len, dtype=jnp.int32),
    )
    mv = lambda a: jnp.swapaxes(a, 0, 1)
    v_bt = mv(v_s)
    g_ext = jnp.where(transient_ep[:, None], 0.0, targ)
    g4 = jnp.concatenate([jnp.zeros_like(g_ext), g_ext], axis=-1)

    out: dict[str, jax.Array] = {}
    rep = {
        "video": v_bt,
        "clean_video": v_bt,
        "proprio": mv(p_s),
        "target_proprio": mv(tp_s),
        "target_next_proprio": mv(ntp_s),
        "action": mv(pu_s),
        "next_action": mv(u_s),
        "target_action": mv(u_s),
        "target_contacts": mv(c_s),
        "target_value": mv(val_s),
        "target_commit": jnp.zeros((n_u, seq_len), dtype=jnp.int32),
        "target_goal_xy": jnp.broadcast_to(targ[:, None, :], (n_u, seq_len, 2)),
        "goal": g4,
        "true_goal": jnp.concatenate([jnp.zeros_like(targ), targ], axis=-1),
        "scenario": jnp.where(transient_ep, 1, 0).astype(jnp.int32),
    }
    for k, bv in base.items():
        if k in rep:
            out[k] = jnp.concatenate([rep[k], bv[n_u:]], axis=0)
        else:
            out[k] = bv
    return out


def train_and_evaluate_wave6(
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
    """Train one Wave 6 cohort seed and run all 5 Zero-Compromise Grounded Evaluation Panels."""
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
    peak_lr = {"x_tiny": 6e-4, "x150m": 3.0e-4, "x450m": 2.6e-4, "x1b": 2.4e-4}[profile]

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
        "wave": "wave6_zero_compromise",
        "spec": asdict(spec),
        "interventions": asdict(interventions),
        "de_privileged_proprio": "ee_only",
        "planner": "geometry_free_learned_mppi",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "peak_lr": peak_lr,
        "parameter_counts": param_counts,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }
    config_hash = hashlib.sha256(json.dumps(config_doc, sort_keys=True).encode()).hexdigest()
    config_doc["config_sha256"] = config_hash
    atomic_write_json(job_dir / "config.json", config_doc)

    print(
        f"[START W6] {tag} | profile={profile} arch={arch} reg={regularizer} "
        f"interventions={asdict(interventions)} | params={param_counts['strict_total']:,} | backend={jax.default_backend()}",
        flush=True,
    )

    if not (bf16_path.is_file() or resume_bf16_path.is_file()):
        real_train = RealMultimodalBank(real_bank_path, split="train") if real_bank_path else None
        grounded_train = Wave6GroundedRobotBank(grounded_robot_path, split="train") if grounded_robot_path else None
        np_rng = np.random.default_rng(seed + 60013)

        def _step_kernel(p, ema_p, m, v, batch_in, step_idx, rk_loss):
            (loss, metrics), grads = jax.value_and_grad(
                lambda w: world_model_loss_wave6(w, ema_p, spec, batch_in, interventions, rk_loss),
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
        synth_n = batch_size // 2 if has_real else batch_size
        rem = batch_size - synth_n
        if (real_train is not None) and (grounded_train is not None):
            rob_n = max(1, (2 * rem) // 3)
            real_n = rem - rob_n
        elif grounded_train is not None:
            rob_n, real_n = rem, 0
        else:
            rob_n, real_n = 0, rem

        jit_sim = jax.jit(
            lambda rk: generate_in_domain_deprivileged_sim_batch(rk, synth_n, seq_len=seq_len, vocab_size=spec.vocab_size)
        )

        t_start = time.monotonic()
        log_file = job_dir / "train.jsonl"
        for step in range(steps):
            rk_sim, rk_loss = jax.random.split(jax.random.fold_in(jax.random.key(seed), step))
            s_batch = jit_sim(rk_sim)
            if real_train is not None and real_n > 0:
                r_batch = real_train.sample_batch(np_rng, real_n, seq_len=seq_len)
                g_batch = grounded_train.sample_batch(np_rng, rob_n, seq_len=seq_len) if (grounded_train and rob_n > 0) else None
                batch = mix_real_and_synthetic_batches(r_batch, s_batch, g_batch)
            elif grounded_train is not None and rob_n > 0:
                g_batch = grounded_train.sample_batch(np_rng, rob_n, seq_len=seq_len)
                batch = mix_real_and_synthetic_batches(g_batch, s_batch)
            else:
                batch = s_batch

            params, ema_params, m_state, v_state, metrics = compiled_step(
                params, ema_params, m_state, v_state, batch, jnp.asarray(step, dtype=jnp.int32), rk_loss
            )
            if step == 0 or (step + 1) % 50 == 0 or (step + 1) == steps:
                m_host = {k: float(v) for k, v in jax.device_get(metrics).items()}
                if not np.isfinite(m_host["loss"]):
                    raise RuntimeError(f"Non-finite Wave 6 loss on {tag} at step {step + 1}: {m_host}")
                rec_line = {"step": step + 1, "elapsed_s": round(time.monotonic() - t_start, 2), **m_host}
                with log_file.open("a") as lf:
                    lf.write(json.dumps(rec_line, sort_keys=True) + "\n")
                atomic_write_json(out_dir / f"{tag}-progress.json", rec_line)
                if step == 0 or (step + 1) % 200 == 0 or (step + 1) == steps:
                    print(
                        f"[{tag}] step {step + 1}/{steps} | loss={m_host['loss']:.4f} jepa={m_host['jepa']:.4f} "
                        f"sig={m_host['sigreg']:.4f} pr={m_host['pr']:.3f} goal={m_host['goal_mse']:.4f}",
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
    eval_res = run_wave6_evaluation(
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
        "T160_post_perturb_goal_b_success": eval_res["beyond_window_T160_distractor_switch"]["T160_post_perturbation_goal_b_success"],
        "latch_mse_beyond_W64": eval_res["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"],
        "symbolic_key_visual_drift": eval_res["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"],
        "dyn_visual_assoc_T1024": eval_res["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"],
        "unseen_ood_corruption_auroc": eval_res["unseen_ood_sensor_corruptions"]["mean_unseen_ood_corruption_auroc"],
        "vault_step4_mm": eval_res["untouched_vault_spoken_robot"].get("step4_mm"),
        "vault_spoken_disc_acc": eval_res["untouched_vault_spoken_robot"].get("spoken_16tok_discrimination_acc"),
    }
    receipt = {
        "tag": tag,
        "wave": "wave6_zero_compromise",
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
    print(f"[DONE W6] {tag} | summary={json.dumps(eval_summary, sort_keys=True)}", flush=True)
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
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--real-bank-path", default=None)
    ap.add_argument("--grounded-robot-path", default=None)
    ap.add_argument("--vault-robot-path", default=None)
    ap.add_argument("--save-bf16", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_and_evaluate_wave6(
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
