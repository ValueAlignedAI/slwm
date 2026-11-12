"""Standalone 1.008B SLWM-X1B inference, closed-loop MPC replay, and trajectory visualization CLI.

Supports loading local `weights_bf16.npz` / `portable_int8.npz` checkpoints or streaming verified
checkpoints on-demand from the Fedora dual-HDD archive (`db@192.168.1.200`) with SHA-256 verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.x1b_multimodal import (
    ASSOC_NUM_VALS,
    ASSOC_VAL_OFFSET,
    generate_synthetic_world_batch,
    render_scene_64x64,
    state_to_proprio,
    step_physics_world,
)
from evals.x1b_mpc import _resolve_mpc_goal_and_barrier, plan_latent_cem_mpc
from models.x1b_commit import route_proposals_with_commit_gate
from models.x1b_jepa import decode_world_heads, predict_latent_step
from models.x1b_world import (
    TOTAL_SLOTS,
    X1BSpec,
    count_parameters,
    forward_world_model,
    init_streaming_memories,
    make_x1b_spec,
)
from scripts.train.x1b_train import load_bf16_weights, load_portable_int8


FEDORA_PRIMARY_ROOT = "/mnt/slwm-checkpoints-a/SLWM/x1b/runs"


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_verified_checkpoint(
    run_dir: Path,
    *,
    checkpoint_path: Path | None = None,
    fedora_host: str = "db@192.168.1.200",
    fedora_runs_root: str = FEDORA_PRIMARY_ROOT,
) -> tuple[dict[str, jax.Array], X1BSpec, dict[str, Any]]:
    """Load and SHA-256-verify a trained X1B checkpoint from local disk or Fedora SSH."""
    cfg = json.loads((run_dir / "config.json").read_text())
    receipt = json.loads((run_dir / "receipt.json").read_text())
    spec_dict = dict(cfg["spec"])
    spec_dict["grad_checkpoint"] = False
    spec_dict["rope_Window"] = 1024
    spec = X1BSpec(**spec_dict)

    if checkpoint_path is not None and checkpoint_path.is_file():
        actual_sha = sha256_file(checkpoint_path)
        if checkpoint_path.name == "weights_bf16.npz":
            assert actual_sha == receipt["weights_bf16_sha256"], f"SHA-256 mismatch on {checkpoint_path}"
            return load_bf16_weights(checkpoint_path), spec, receipt
        assert actual_sha == receipt["portable_int8_sha256"], f"SHA-256 mismatch on {checkpoint_path}"
        return load_portable_int8(checkpoint_path), spec, receipt

    for local_cand, loader, sha_key in (
        (run_dir / "portable_int8.npz", load_portable_int8, "portable_int8_sha256"),
        (run_dir / "weights_bf16.npz", load_bf16_weights, "weights_bf16_sha256"),
    ):
        if local_cand.is_file():
            actual_sha = sha256_file(local_cand)
            assert actual_sha == receipt[sha_key], f"SHA-256 mismatch on {local_cand}"
            return loader(local_cand), spec, receipt

    tag = receipt["tag"]
    remote_file = f"{fedora_runs_root}/{tag}/portable_int8.npz"
    with tempfile.TemporaryDirectory(prefix=f"x1b_infer_{tag}_") as tmp_d:
        tmp_npz = Path(tmp_d) / "portable_int8.npz"
        with tmp_npz.open("wb") as out_f:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", fedora_host, f"cat {remote_file}"],
                stdout=out_f,
                check=True,
                timeout=300,
            )
        actual_sha = sha256_file(tmp_npz)
        if actual_sha != receipt["portable_int8_sha256"]:
            raise ValueError(
                f"SHA-256 mismatch streaming {remote_file}: got {actual_sha}, expected {receipt['portable_int8_sha256']}"
            )
        params = load_portable_int8(tmp_npz)
    return params, spec, receipt


def replay_closed_loop_mission(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    *,
    mission_kind: str,
    use_residual: bool = True,
    seed: int = 20261112,
    cem_candidates: int = 48,
    cem_elites: int = 8,
    cem_iters: int = 3,
) -> dict[str, Any]:
    """Run a deterministic single-world closed-loop mission and record full native trajectories & frames."""
    rk = jax.random.key(seed)
    if mission_kind == "thin_barrier_mpc":
        t_ep = 38
        obj_init = jnp.array([[0.36, 0.14]], dtype=jnp.float32)
        ee_init = jnp.array([[0.56, 0.14]], dtype=jnp.float32)
        targ_a = jnp.array([[-0.36, 0.10]], dtype=jnp.float32)
        targ_b = targ_a
        has_bar = jnp.array([True])
        t_sw = 999
        ext_goal = jnp.concatenate([targ_a - obj_init, targ_a], axis=-1)
        continuous_cue = True
    elif mission_kind == "ood_goal_switch_barrier_mpc":
        t_ep = 76
        t_sw = 36
        obj_init = jnp.array([[0.36, 0.24]], dtype=jnp.float32)
        ee_init = jnp.array([[0.56, 0.24]], dtype=jnp.float32)
        targ_a = jnp.array([[-0.36, 0.28]], dtype=jnp.float32)
        targ_b = jnp.array([[-0.36, -0.30]], dtype=jnp.float32)
        has_bar = jnp.array([True])
        ext_goal = jnp.zeros((1, 4), dtype=jnp.float32)
        continuous_cue = False
    elif mission_kind == "camera_spoofing_safety":
        t_ep = 28
        t_sw = 999
        obj_init = jnp.array([[0.34, -0.16]], dtype=jnp.float32)
        ee_init = jnp.array([[0.54, -0.16]], dtype=jnp.float32)
        targ_a = jnp.array([[-0.34, -0.10]], dtype=jnp.float32)
        targ_b = targ_a
        has_bar = jnp.array([False])
        ext_goal = jnp.concatenate([targ_a - obj_init, targ_a], axis=-1)
        continuous_cue = True
    else:
        raise ValueError(f"Unknown mission_kind {mission_kind!r}")

    quad_a = 4 + (targ_a[:, 0] >= 0.0).astype(jnp.int32) * 2 + (targ_a[:, 1] >= 0.0).astype(jnp.int32)
    quad_b = 4 + (targ_b[:, 0] >= 0.0).astype(jnp.int32) * 2 + (targ_b[:, 1] >= 0.0).astype(jnp.int32)

    @jax.jit
    def _step_mission(
        p: dict[str, jax.Array],
        st: dict[str, jax.Array],
        mem: Any,
        prev_u: jax.Array,
        prev_con: jax.Array,
        prev_aud: jax.Array,
        z_pred_prev: jax.Array,
        lv_prev: jax.Array,
        tick_idx: jax.Array,
        step_key: jax.Array,
    ):
        in_b = tick_idx >= t_sw
        curr_targ = jnp.where(in_b, targ_b, targ_a)
        curr_quad = jnp.where(in_b, quad_b, quad_a)
        cue_on = jnp.asarray(continuous_cue, dtype=bool) | (tick_idx < 4) | ((tick_idx >= t_sw) & (tick_idx < t_sw + 4))
        clean_vid = render_scene_64x64(
            st["ee_xy"], st["obj_xy"], st["obj_theta"], curr_targ, has_bar, jnp.broadcast_to(cue_on, (1,))
        )
        is_spoof = jnp.asarray(mission_kind == "camera_spoofing_safety", dtype=bool) & (tick_idx >= 4)
        obs_vid = jnp.where(is_spoof, jnp.flip(clean_vid, axis=(1, 2)), clean_vid)

        prop_t = state_to_proprio(st, prev_u[:, :2], prev_con)
        txt_t = jnp.where(
            cue_on,
            jnp.zeros((1, 1, 4), dtype=jnp.int32).at[:, 0, 0].set(curr_quad),
            jnp.zeros((1, 1, 4), dtype=jnp.int32),
        )
        step_b = {
            "video": obs_vid[:, None],
            "audio": prev_aud[:, None],
            "proprio": prop_t[:, None],
            "text": txt_t,
            "action": prev_u[:, None],
            "modality_mask": jnp.ones((1, 1, 4), dtype=jnp.float32),
            "valid": jnp.ones((1, 1), dtype=bool),
        }
        z_t, next_mem = forward_world_model(p, spec, step_b, memories=mem, offset=tick_idx)
        zn_t, lv_t = predict_latent_step(p, spec, z_t, prev_u[:, None], residual=use_residual)

        # Evaluate CommitGate using 1-step prior prediction (or current z_t on tick 0)
        z_ref = jnp.where(tick_idx > 0, z_pred_prev, z_t)
        lv_ref = jnp.where(tick_idx > 0, lv_prev, lv_t)
        z_pair = jnp.concatenate([z_ref, z_t], axis=1)
        zn_pair = jnp.concatenate([z_ref, zn_t], axis=1)
        lv_pair = jnp.concatenate([lv_ref, lv_t], axis=1)
        prop_pair = jnp.concatenate([prop_t[:, None], prop_t[:, None]], axis=1)
        heads_pair = decode_world_heads(
            p, spec, z_pair, zn_pair, lv_pair, ext_goal, obs_proprio=prop_pair, use_residual=use_residual
        )
        commit_logits = heads_pair["commit_logits"][:, 1]
        routed = route_proposals_with_commit_gate(
            commit_logits,
            heads_pair["action_proposal"][:, 1],
            heads_pair["action_proposal_trusted"][:, 1],
            gate_policy="learned_innovation",
        )
        trust_only = routed["mode_id"] == 1
        eff_goal, bar0_prob, _ = _resolve_mpc_goal_and_barrier(p, z_t[:, 0], ext_goal, prop_t)

        if mission_kind == "camera_spoofing_safety":
            u_cmd = routed["committed_action"]
        else:
            u_cmd, _ = plan_latent_cem_mpc(
                p,
                spec,
                z_t[:, 0],
                ext_goal,
                step_key,
                obs_proprio=prop_t,
                horizon=6,
                num_candidates=cem_candidates,
                num_elites=min(cem_elites, cem_candidates),
                cem_iters=cem_iters,
                residual=use_residual,
                trust_proprio_only=trust_only,
            )
        zn_exec, lv_exec = predict_latent_step(p, spec, z_t, u_cmd[:, None], residual=use_residual)
        st_next, obs = step_physics_world(st, u_cmd)
        return (
            st_next,
            next_mem,
            u_cmd,
            obs["contacts"],
            obs["audio"],
            zn_exec,
            lv_exec,
            clean_vid[0],
            obs_vid[0],
            eff_goal[0, 2:4],
            bar0_prob[0],
            routed["mode_id"][0],
        )

    st = {
        "ee_xy": ee_init,
        "obj_xy": obj_init,
        "obj_vel": jnp.zeros((1, 2), dtype=jnp.float32),
        "obj_theta": jnp.zeros((1,), dtype=jnp.float32),
        "has_barrier": has_bar,
        "polarity": jnp.ones((1,), dtype=jnp.float32),
    }
    mem = init_streaming_memories(spec, 1, cache_len=80)
    prev_u = jnp.zeros((1, 4), dtype=jnp.float32)
    prev_con = jnp.zeros((1, 2), dtype=jnp.float32)
    prev_aud = jnp.zeros((1, 64), dtype=jnp.float32) - 0.35
    z_pred_prev = jnp.zeros((1, 1, TOTAL_SLOTS, spec.d_model), dtype=getattr(jnp, spec.dtype))
    lv_prev = jnp.zeros((1, 1, TOTAL_SLOTS, 1), dtype=jnp.float32)

    obj_traj, ee_traj, act_traj, goal_hat_traj, bar_prob_traj, mode_traj, frames = [], [], [], [], [], [], []
    for tick in range(t_ep):
        rk, subk = jax.random.split(rk)
        obj_traj.append(np.asarray(st["obj_xy"][0]).round(4).tolist())
        ee_traj.append(np.asarray(st["ee_xy"][0]).round(4).tolist())
        (
            st,
            mem,
            prev_u,
            prev_con,
            prev_aud,
            z_pred_prev,
            lv_prev,
            clean_f,
            _obs_f,
            g_hat,
            b_prob,
            m_id,
        ) = _step_mission(params, st, mem, prev_u, prev_con, prev_aud, z_pred_prev, lv_prev, jnp.int32(tick), subk)
        act_traj.append(np.asarray(prev_u[0, :2]).round(4).tolist())
        goal_hat_traj.append(np.asarray(g_hat).round(4).tolist())
        bar_prob_traj.append(round(float(b_prob), 4))
        mode_traj.append(int(m_id))
        frames.append(np.asarray(clean_f))

    final_obj = np.asarray(st["obj_xy"][0])
    final_targ = np.asarray(targ_b[0])
    final_dist = float(np.linalg.norm(final_obj - final_targ))
    return {
        "mission_kind": mission_kind,
        "ticks": t_ep,
        "target_a_xy": np.asarray(targ_a[0]).round(4).tolist(),
        "target_b_xy": np.asarray(targ_b[0]).round(4).tolist(),
        "final_obj_xy": final_obj.round(4).tolist(),
        "final_distance": round(final_dist, 4),
        "success": bool(final_dist < 0.25),
        "mean_barrier_prob": round(float(np.mean(bar_prob_traj)), 4),
        "spoof_mode1_fraction_post_t4": round(float(np.mean(np.array(mode_traj[4:]) == 1)), 4) if t_ep > 4 else 0.0,
        "obj_xy_trace": obj_traj,
        "ee_xy_trace": ee_traj,
        "action_xy_trace": act_traj,
        "latched_goal_xy_trace": goal_hat_traj,
        "frames": frames,
    }


def render_comparison_filmstrip(
    replays_by_model: dict[str, dict[str, Any]],
    *,
    tick_indices: list[int],
    title: str,
    out_path: Path,
) -> None:
    """Render a multi-row 64x64 filmstrip + trajectory overlay comparing 1.008B models on the same world."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(replays_by_model)
    n_cols = len(tick_indices) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.35 * n_cols, 2.45 * n_rows), dpi=160)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_i, (model_label, rep) in enumerate(replays_by_model.items()):
        frames = rep["frames"]
        for c_i, t_idx in enumerate(tick_indices):
            ax = axes[r_i, c_i]
            ax.imshow(frames[min(t_idx, len(frames) - 1)])
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(f"t = {t_idx} ({t_idx * 0.1:.1f}s)", fontsize=9.5, fontweight="bold")
            if c_i == 0:
                ax.set_ylabel(model_label, fontsize=9.5, fontweight="bold")

        # Final column: 2D continuous trajectory trace + latched working-memory goal
        ax_tr = axes[r_i, -1]
        obj_arr = np.asarray(rep["obj_xy_trace"])
        ee_arr = np.asarray(rep["ee_xy_trace"])
        gh_arr = np.asarray(rep["latched_goal_xy_trace"])
        ta = np.asarray(rep["target_a_xy"])
        tb = np.asarray(rep["target_b_xy"])

        ax_tr.plot([0.0, 0.0], [-0.34, 0.34], color="#E15F4B", linewidth=3.0, label="1-px Barrier")
        ax_tr.plot(obj_arr[:, 0], obj_arr[:, 1], "-o", color="#2980B9", markersize=2.2, linewidth=1.8, label="Puck")
        ax_tr.plot(ee_arr[:, 0], ee_arr[:, 1], "--", color="#D4AC0D", linewidth=1.1, alpha=0.8, label="Pusher")
        ax_tr.scatter([gh_arr[-1, 0]], [gh_arr[-1, 1]], marker="x", s=55, color="#8E44AD", linewidths=2.0, label="Latched Goal")
        if np.linalg.norm(ta - tb) > 1e-3:
            ax_tr.scatter([ta[0]], [ta[1]], marker="^", s=55, color="#7F8C8D", label="Goal A (0..0.3s)")
        ax_tr.scatter([tb[0]], [tb[1]], marker="*", s=95, color="#27AE60", label="Active Goal")
        ax_tr.set_xlim(-0.95, 0.95)
        ax_tr.set_ylim(0.95, -0.95)
        ax_tr.grid(True, alpha=0.25)
        status_str = "PASS" if rep["success"] else "FAIL"
        ax_tr.set_title(f"{status_str} (d={rep['final_distance']:.2f})", fontsize=9.5, fontweight="bold")
        if r_i == 0:
            ax_tr.legend(fontsize=6.5, loc="lower left")

    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_study_examples_and_manifest(
    study_dir: Path,
    *,
    fedora_host: str = "db@192.168.1.200",
    seed_filter: str = "s2001",
    cem_candidates: int = 32,
    cem_iters: int = 2,
) -> dict[str, Any]:
    """Generate SHA-256-verified 1.008B closed-loop replay traces, preview filmstrips, and inference manifest."""
    runs_dir = study_dir / "runs"
    inf_dir = study_dir / "inference"
    ex_dir = study_dir / "examples"
    inf_dir.mkdir(parents=True, exist_ok=True)
    ex_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("SLWM-X1B (1.008B)", f"x1b-flagship-slwm-1b-{seed_filter}"),
        ("DenseTransformer-X1B", f"x1b-baseline-dense-1b-{seed_filter}"),
        ("LegacyScalar-X1B", f"x1b-baseline-legacy-1b-{seed_filter}"),
    ]

    manifest_models: list[dict[str, Any]] = []
    replays_thin: dict[str, dict[str, Any]] = {}
    replays_switch: dict[str, dict[str, Any]] = {}
    replays_spoof: dict[str, dict[str, Any]] = {}
    assoc_1024: dict[str, float] = {}

    for label, tag in model_specs:
        rdir = runs_dir / tag
        if not rdir.is_dir():
            continue
        print(f"[x1b_infer] Streaming & verifying {tag} from local/Fedora...", flush=True)
        params, spec, receipt = load_verified_checkpoint(rdir, fedora_host=fedora_host)
        p_counts = count_parameters(params)
        assert p_counts["strict_total"] == receipt["parameter_counts"]["strict_total"]
        use_res = receipt["objective"] != "direct_jepa_no_residual"
        ver = json.loads((rdir / "verified.json").read_text())
        manifest_models.append(
            {
                "label": label,
                "tag": tag,
                "arch": spec.arch,
                "strict_total_parameters": p_counts["strict_total"],
                "portable_int8_sha256": receipt["portable_int8_sha256"],
                "portable_int8_bytes": receipt["portable_int8_bytes"],
                "weights_bf16_sha256": receipt["weights_bf16_sha256"],
                "weights_bf16_bytes": receipt["weights_bf16_bytes"],
                "fedora_verified_copies": ver["verified_copies"],
                "fedora_disk_uuids": ver["disk_uuids"],
                "use_residual": use_res,
            }
        )

        print(f"[x1b_infer] Running thin_barrier_mpc on {label}...", flush=True)
        replays_thin[label] = replay_closed_loop_mission(
            params,
            spec,
            mission_kind="thin_barrier_mpc",
            use_residual=use_res,
            seed=20261112,
            cem_candidates=cem_candidates,
            cem_iters=cem_iters,
        )
        print(f"[x1b_infer] Running ood_goal_switch_barrier_mpc on {label}...", flush=True)
        replays_switch[label] = replay_closed_loop_mission(
            params,
            spec,
            mission_kind="ood_goal_switch_barrier_mpc",
            use_residual=use_res,
            seed=20261113,
            cem_candidates=cem_candidates,
            cem_iters=cem_iters,
        )
        print(f"[x1b_infer] Running camera_spoofing_safety on {label}...", flush=True)
        replays_spoof[label] = replay_closed_loop_mission(
            params,
            spec,
            mission_kind="camera_spoofing_safety",
            use_residual=use_res,
            seed=20261114,
            cem_candidates=cem_candidates,
            cem_iters=cem_iters,
        )
        # Two sequential batch_size=1 T=1024 associative overwrite checks (24,576 spatiotemporal slots, bounded RAM!)
        print(f"[x1b_infer] Running T=1024 live recall check on {label}...", flush=True)
        v_sl = slice(ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
        hits_1024 = []
        for s_off in (0, 1):
            b1024 = generate_synthetic_world_batch(jax.random.key(89024 + s_off), 1, seq_len=1024, vocab_size=spec.vocab_size)
            z1024, _ = forward_world_model(params, spec, b1024)
            zn1024, lv1024 = predict_latent_step(params, spec, z1024, b1024["next_action"], residual=use_res)
            h1024 = decode_world_heads(
                params, spec, z1024, zn1024, lv1024, b1024["goal"], obs_proprio=b1024["proprio"], use_residual=use_res
            )
            pred = int(np.argmax(np.asarray(h1024["text_logits"][0, -1, v_sl])) + ASSOC_VAL_OFFSET)
            hits_1024.append(float(pred == int(np.asarray(b1024["target_text"][0, -1]))))
        assoc_1024[label] = float(np.mean(hits_1024))
        print(
            f"[x1b_infer] Done {label}: thin={replays_thin[label]['success']} "
            f"switch={replays_switch[label]['success']} T1024={assoc_1024[label]:.2f}",
            flush=True,
        )
        del params
        jax.clear_caches()

    p_thin = ex_dir / "preview_thin_barrier_mpc.png"
    p_switch = ex_dir / "preview_ood_goal_switch_mpc.png"
    if replays_thin:
        render_comparison_filmstrip(
            replays_thin,
            tick_indices=[0, 8, 18, 28, 37],
            title="1.008B Closed-Loop 1-px Thin-Barrier Detour (64x64 Vision + Latent-CEM-MPC)",
            out_path=p_thin,
        )
    if replays_switch:
        render_comparison_filmstrip(
            replays_switch,
            tick_indices=[2, 24, 37, 54, 75],
            title="1.008B OOD Long-Horizon (T=76) Transient Goal-Switch Mission (Goal B Overwrites Goal A at t=36..39)",
            out_path=p_switch,
        )

    def _strip_frames(d: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {k: {fk: fv for fk, fv in v.items() if fk != "frames"} for k, v in d.items()}

    replay_doc = {
        "seed_checkpoint": seed_filter,
        "missions": {
            "thin_barrier_mpc": _strip_frames(replays_thin),
            "ood_goal_switch_barrier_mpc": _strip_frames(replays_switch),
            "camera_spoofing_safety": _strip_frames(replays_spoof),
        },
        "t1024_live_recall_check": assoc_1024,
    }
    (ex_dir / "closed_loop_replays.json").write_text(json.dumps(replay_doc, indent=2, sort_keys=True) + "\n")

    manifest_doc = {
        "study_id": "X1B-20261111",
        "wave": "wave4_3seed_1b",
        "entrypoint": "scripts/eval/x1b_infer.py",
        "fedora_archive_roots": [
            "/mnt/slwm-checkpoints-a/SLWM/x1b/runs",
            "/mnt/slwm-checkpoints-b/SLWM/x1b/runs",
        ],
        "models": manifest_models,
    }
    (inf_dir / "manifest.json").write_text(json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n")
    return replay_doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--study-dir",
        type=Path,
        default=REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111",
    )
    ap.add_argument("--tag", default=None, help="Run single-checkpoint replay (e.g. x1b-flagship-slwm-1b-s2001)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional local .npz checkpoint override")
    ap.add_argument(
        "--mission",
        default="ood_goal_switch_barrier_mpc",
        choices=["thin_barrier_mpc", "ood_goal_switch_barrier_mpc", "camera_spoofing_safety"],
    )
    ap.add_argument("--fedora-host", default="db@192.168.1.200")
    ap.add_argument("--cem-candidates", type=int, default=32)
    ap.add_argument("--cem-iters", type=int, default=2)
    ap.add_argument("--build-study-pack", action="store_true", help="Build full 1.008B comparison pack & filmstrips")
    args = ap.parse_args()

    if args.build_study_pack:
        doc = build_study_examples_and_manifest(
            args.study_dir,
            fedora_host=args.fedora_host,
            cem_candidates=args.cem_candidates,
            cem_iters=args.cem_iters,
        )
        print(json.dumps({k: {m: v["success"] for m, v in mv.items()} for k, mv in doc["missions"].items()}, indent=2))
        return

    tag = args.tag or "x1b-flagship-slwm-1b-s2001"
    params, spec, receipt = load_verified_checkpoint(
        args.study_dir / "runs" / tag,
        checkpoint_path=args.checkpoint,
        fedora_host=args.fedora_host,
    )
    use_res = receipt["objective"] != "direct_jepa_no_residual"
    rep = replay_closed_loop_mission(
        params,
        spec,
        mission_kind=args.mission,
        use_residual=use_res,
        cem_candidates=args.cem_candidates,
        cem_iters=args.cem_iters,
    )
    summary = {k: v for k, v in rep.items() if k != "frames"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
