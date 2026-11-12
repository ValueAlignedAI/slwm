"""Compiled JAX/XLA training & evaluation runner for SLWM-X1B (~152M, ~455M, and ~1.002B scales).

Runs identically on Google TPU v6e1 (32 GB HBM), NVIDIA H100 (80 GB HBM), and local CPU (smoke).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

# Ensure repo root is importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from data.x1b_multimodal import (
    RealMultimodalBank,
    generate_synthetic_world_batch,
    mix_real_and_synthetic_batches,
)
from evals.x1b_eval_suite import run_full_evaluation
from models.x1b_jepa import VALID_OBJECTIVES, update_ema_params, world_model_loss
from models.x1b_world import count_parameters, init_x1b_params, make_x1b_spec


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def save_bf16_weights(path: Path, params: dict[str, jax.Array]) -> str:
    """Save lossless bfloat16 weights (stored as uint16 bit-patterns for universal NumPy portability)."""
    arrays: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    host_p = jax.device_get(params)
    for k, v in host_p.items():
        if v.ndim >= 2:
            bf = np.asarray(v, dtype=ml_dtypes.bfloat16)
            arrays[k] = bf.view(np.uint16)
            dtypes[k] = "bf16_u16"
        else:
            arrays[k] = np.asarray(v, dtype=np.float32)
            dtypes[k] = "float32"
    arrays["__dtypes_json__"] = np.array(json.dumps(dtypes, sort_keys=True))
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return sha256_file(path)


def load_bf16_weights(path: Path) -> dict[str, jax.Array]:
    """Restore bfloat16/float32 parameter tree saved by save_bf16_weights."""
    with np.load(path, allow_pickle=False) as f:
        dtypes = json.loads(str(f["__dtypes_json__"].item()))
        out: dict[str, jax.Array] = {}
        for k, kind in dtypes.items():
            arr = f[k]
            if kind == "bf16_u16":
                val = arr.view(ml_dtypes.bfloat16).astype(np.float32)
            else:
                val = arr.astype(np.float32)
            out[k] = jnp.asarray(val)
        return out


def save_portable_int8(path: Path, params: dict[str, jax.Array]) -> str:
    """Export symmetric per-channel int8 portable inference package (~1.0 GB at 1.002B params)."""
    arrays: dict[str, np.ndarray] = {}
    scales: dict[str, str] = {}
    host_p = jax.device_get(params)
    for k, v in host_p.items():
        vf = np.asarray(v, dtype=np.float32)
        if vf.ndim >= 2 and vf.size >= 256:
            max_abs = np.max(np.abs(vf), axis=0, keepdims=True)
            sc = np.maximum(max_abs / 127.0, 1e-8).astype(np.float16)
            q = np.clip(np.rint(vf / sc.astype(np.float32)), -127, 127).astype(np.int8)
            arrays[k] = q
            arrays[k + "__scale"] = sc
            scales[k] = "int8_pc"
        else:
            arrays[k] = vf.astype(np.float16)
            scales[k] = "float16"
    arrays["__quant_manifest__"] = np.array(json.dumps(scales, sort_keys=True))
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return sha256_file(path)


def memory_efficient_adamw_step(
    params: dict[str, jax.Array],
    m_bf16: dict[str, jax.Array],
    v_f32_or_bf16: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    step: jax.Array,
    total_steps: int,
    peak_lr: float,
    weight_decay: float = 0.02,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array], jax.Array, jax.Array]:
    """Clipped cosine AdamW with bfloat16 moments so 1.002B params + moments + EMA fit in ~10 GB HBM!"""
    clean_grads = jax.tree.map(
        lambda g: jnp.nan_to_num(g.astype(jnp.float32), nan=0.0, posinf=1e4, neginf=-1e4),
        grads,
    )
    sq_sum = sum(jnp.sum(gf * gf) for gf in jax.tree.leaves(clean_grads))
    grad_norm = jnp.sqrt(sq_sum + 1e-12)
    clip_coef = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-6))

    t = step.astype(jnp.float32) + 1.0
    warmup = float(max(10, min(160, total_steps // 10)))
    prog = jnp.clip((t - warmup) / float(max(1, total_steps - int(warmup))), 0.0, 1.0)
    lr = peak_lr * jnp.minimum(1.0, t / warmup) * (0.1 + 0.45 * (1.0 + jnp.cos(jnp.pi * prog)))

    b1, b2 = 0.9, 0.98
    bc1 = 1.0 - (b1**t)
    bc2 = 1.0 - (b2**t)

    def _upd_m(m: jax.Array, gf_raw: jax.Array) -> jax.Array:
        gf = gf_raw * clip_coef
        return (b1 * m.astype(jnp.float32) + (1.0 - b1) * gf).astype(jnp.bfloat16)

    def _upd_v(v: jax.Array, gf_raw: jax.Array) -> jax.Array:
        gf = gf_raw * clip_coef
        return (b2 * v.astype(jnp.float32) + (1.0 - b2) * (gf * gf)).astype(jnp.bfloat16)

    m_next = jax.tree.map(_upd_m, m_bf16, clean_grads)
    v_next = jax.tree.map(_upd_v, v_f32_or_bf16, clean_grads)

    def _upd_p(k: str, p: jax.Array, m: jax.Array, v: jax.Array) -> jax.Array:
        pf = p.astype(jnp.float32)
        m_hat = m.astype(jnp.float32) / bc1
        v_hat = jnp.maximum(v.astype(jnp.float32) / bc2, 0.0)
        decay = weight_decay if (p.ndim >= 2 and k != "text_emb") else 0.0
        return pf - lr * (m_hat / (jnp.sqrt(v_hat) + 1e-8) + decay * pf)

    p_next = {k: _upd_p(k, params[k], m_next[k], v_next[k]) for k in params}
    return p_next, m_next, v_next, grad_norm, lr


def train_and_evaluate(
    *,
    tag: str,
    profile: str,
    arch: str,
    objective: str,
    seed: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    out_dir: Path,
    real_bank_path: str | None = None,
    pymunk_panels_path: str | None = None,
    peak_lr: float | None = None,
    eval_every: int = 0,
    save_bf16: bool = True,
) -> dict[str, Any]:
    """Run deterministic compiled training + full 7-panel evaluation and emit hash-verified artifacts."""
    job_dir = out_dir / tag
    done_path = job_dir / "done.json"
    if done_path.is_file():
        print(f"[SKIP] {tag} already completed at {done_path}", flush=True)
        return json.loads(done_path.read_text())

    job_dir.mkdir(parents=True, exist_ok=True)
    spec = make_x1b_spec(profile=profile, arch=arch)
    if peak_lr is None:
        peak_lr = {"x_tiny": 6e-4, "x150m": 3.0e-4, "x450m": 2.6e-4, "x1b": 2.4e-4}[profile]

    int8_path = job_dir / "portable_int8.npz"
    bf16_path = job_dir / "weights_bf16.npz"
    resume_bf16_path = job_dir / ".resume_weights_bf16.npz"

    if bf16_path.is_file() or resume_bf16_path.is_file():
        src_ckpt = bf16_path if bf16_path.is_file() else resume_bf16_path
        print(f"[RESUME] Restoring trained weights from {src_ckpt} for evaluation...", flush=True)
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
        "spec": asdict(spec),
        "objective": objective,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "peak_lr": peak_lr,
        "parameter_counts": param_counts,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "real_bank_path": str(real_bank_path) if real_bank_path else None,
    }
    config_hash = hashlib.sha256(json.dumps(config_doc, sort_keys=True).encode()).hexdigest()
    config_doc["config_sha256"] = config_hash
    atomic_write_json(job_dir / "config.json", config_doc)

    print(
        f"[START] {tag} | profile={profile} arch={arch} obj={objective} "
        f"| strict_params={param_counts['strict_total']:,} | backend={jax.default_backend()}",
        flush=True,
    )

    if not (bf16_path.is_file() or resume_bf16_path.is_file()):
        real_train = RealMultimodalBank(real_bank_path, split="train") if real_bank_path else None
        np_rng = np.random.default_rng(seed + 31337)

        def _step_kernel(
            p: dict[str, jax.Array],
            ema_p: dict[str, jax.Array],
            m: dict[str, jax.Array],
            v: dict[str, jax.Array],
            batch_in: dict[str, jax.Array],
            step_idx: jax.Array,
        ):
            (loss, metrics), grads = jax.value_and_grad(
                lambda w: world_model_loss(w, ema_p, spec, batch_in, objective_mode=objective),
                has_aux=True,
            )(p)
            p_new, m_new, v_new, gnorm, lr_now = memory_efficient_adamw_step(
                p, m, v, grads, step_idx, steps, peak_lr
            )
            ema_decay = 0.992 + 0.006 * (step_idx.astype(jnp.float32) / float(max(1, steps)))
            ema_new = jax.tree.map(
                lambda e, w: (ema_decay * e.astype(jnp.float32) + (1.0 - ema_decay) * w.astype(jnp.float32)).astype(jnp.bfloat16),
                ema_p,
                p_new,
            )
            metrics = {**metrics, "grad_norm": gnorm, "lr": lr_now}
            return p_new, ema_new, m_new, v_new, metrics

        compiled_step = jax.jit(_step_kernel, donate_argnums=(0, 1, 2, 3))
        synth_half = batch_size // 2 if real_train is not None else batch_size
        real_half = batch_size - synth_half
        jit_synth = jax.jit(lambda rk: generate_synthetic_world_batch(rk, synth_half, seq_len=seq_len, vocab_size=spec.vocab_size))

        t_start = time.monotonic()
        t_window = t_start
        log_file = job_dir / "train.jsonl"

        for step in range(steps):
            rk = jax.random.fold_in(jax.random.key(seed), step)
            s_batch = jit_synth(rk)
            if real_train is not None and real_half > 0:
                r_batch = real_train.sample_batch(np_rng, real_half, seq_len=seq_len)
                batch = mix_real_and_synthetic_batches(r_batch, s_batch)
            else:
                batch = s_batch

            params, ema_params, m_state, v_state, metrics = compiled_step(
                params, ema_params, m_state, v_state, batch, jnp.int32(step)
            )

            done_step = step + 1
            if done_step == 1 or done_step % 50 == 0 or done_step == steps:
                m_host = {k: float(v) for k, v in jax.device_get(metrics).items()}
                if not np.isfinite(m_host["loss"]):
                    raise FloatingPointError(f"Non-finite loss in {tag} at step {done_step}: {m_host}")
                now = time.monotonic()
                sps = (50.0 / max(now - t_window, 1e-6)) if done_step >= 100 and done_step % 50 == 0 else None
                t_window = now
                row = {
                    "tag": tag,
                    "step": done_step,
                    "goal_steps": steps,
                    "elapsed_s": round(now - t_start, 2),
                    "steps_per_s": round(sps, 3) if sps is not None else None,
                    **{k: round(v, 5) for k, v in m_host.items()},
                }
                with log_file.open("a") as lf:
                    lf.write(json.dumps(row) + "\n")
                atomic_write_json(out_dir / f"{tag}-progress.json", row)
                if done_step == 1 or done_step % 200 == 0 or done_step == steps:
                    print(
                        f"[{tag}] step={done_step}/{steps} loss={row['loss']:.4f} "
                        f"jepa={row['jepa']:.4f} pr={row['pr']:.3f} state={row['state_mse']:.4f} "
                        f"commit={row['commit_ce']:.3f} txt={row['text_ce']:.3f} sps={row['steps_per_s']} ({row['elapsed_s']}s)",
                        flush=True,
                    )

        train_wall_s = time.monotonic() - t_start
        del m_state, v_state, ema_params

    # Persist checkpoints BEFORE running 7-panel evaluation so trained weights are 100% crash-safe!
    int8_sha = sha256_file(int8_path) if int8_path.is_file() else save_portable_int8(int8_path, params)
    bf16_sha = None
    if save_bf16:
        bf16_sha = sha256_file(bf16_path) if bf16_path.is_file() else save_bf16_weights(bf16_path, params)
    elif not resume_bf16_path.is_file():
        save_bf16_weights(resume_bf16_path, params)

    print(f"[EVAL] Running frozen 7-panel evaluation for {tag}...", flush=True)
    t_eval0 = time.monotonic()
    eval_results = run_full_evaluation(
        params,
        spec,
        objective_mode=objective,
        real_bank_path=real_bank_path,
        pymunk_panels_path=pymunk_panels_path,
        include_scaling_bench=(profile in ("x1b", "x450m") or tag.endswith("s2001")),
    )
    eval_wall_s = time.monotonic() - t_eval0
    atomic_write_json(job_dir / "eval_metrics.json", eval_results)
    resume_bf16_path.unlink(missing_ok=True)

    receipt = {
        "tag": tag,
        "profile": profile,
        "arch": arch,
        "objective": objective,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "config_sha256": config_hash,
        "parameter_counts": param_counts,
        "backend": jax.default_backend(),
        "train_wall_s": round(train_wall_s, 2),
        "eval_wall_s": round(eval_wall_s, 2),
        "portable_int8_sha256": int8_sha,
        "portable_int8_bytes": int8_path.stat().st_size,
        "weights_bf16_sha256": bf16_sha,
        "weights_bf16_bytes": (job_dir / "weights_bf16.npz").stat().st_size if bf16_sha else 0,
        "eval_summary": {
            "av_r1": eval_results["real_multimodal_and_text"].get("av_r1", {}).get("mean"),
            "persistence_skill": eval_results["thin_barrier_and_persistence"]["step1_skill_vs_persistence"],
            "barrier_auroc_64": eval_results["thin_barrier_and_persistence"]["barrier_auroc_64x64"],
            "assoc_T512": eval_results["associative_recall_512"]["T512"]["mean"],
            "transient_drop_pp": eval_results["transient_goal_retention"]["transient_drop_pp"]["mean"],
            "mpc_success": eval_results["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"],
            "g8_shooting_success": eval_results["closed_loop_mpc"]["g8_random_shooting_3"]["success_rate"]["mean"],
            "spoof_regret_learned": eval_results["spoofing_and_commit_gate"]["learned_innovation"]["spoofed_action_regret"],
            "spoof_regret_always": eval_results["spoofing_and_commit_gate"]["always_commit"]["spoofed_action_regret"],
        },
    }
    atomic_write_json(job_dir / "receipt.json", receipt)
    atomic_write_json(done_path, receipt)
    print(f"[DONE] {tag} summary: {json.dumps(receipt['eval_summary'])}", flush=True)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--profile", required=True, choices=["x_tiny", "x150m", "x450m", "x1b"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--objective", default="residual_ema_jepa_sigreg", choices=list(VALID_OBJECTIVES))
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--real-bank", default=None)
    ap.add_argument("--pymunk-panels", default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--cache-dir", default="/tmp/x1b-jax-cache")
    args = ap.parse_args()

    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)

    train_and_evaluate(
        tag=args.tag,
        profile=args.profile,
        arch=args.arch,
        objective=args.objective,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        out_dir=args.out,
        real_bank_path=args.real_bank,
        pymunk_panels_path=args.pymunk_panels,
        save_bf16=(not args.no_bf16),
    )


if __name__ == "__main__":
    main()
