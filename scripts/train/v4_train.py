"""SLWM-v4 Config-Driven Single-Pass Multimodal Trainer (TPU v6e-1 / GPU / CPU).

Strictly enforces:
  - Single-pass (`<= 1.0 epoch` per corpus) streaming via `V4SinglePassMixtureLoader(split="train")`.
  - Frozen, deterministic `split="dev"` (`shuffle=False`) evaluation — NEVER touches `split="test"`.
  - Exact base-shape muP (`make_v4_mup_optimizer`) + CompleteP `alpha = 1.0` depth residual scaling.
  - Gradient accumulation inside compiled XLA (`jax.lax.scan`) for high TPU `v6e-1` MXU utilization.
  - Complete Dual-Track accounting, config hash, and early NaN/divergence abort (`AGENTS.md §10, §12`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v4_streaming_loader import DEFAULT_EXCLUSION_MANIFEST, DEFAULT_GATE2_RECEIPT, V4SinglePassMixtureLoader
from models.v4 import (
    SLWMv4Spec,
    V4_S3_DISCOVERY_RUN_IDS,
    compute_v4_training_loss,
    init_v4_core_state,
    init_v4_model_params,
    make_slwm_v4_spec,
    make_v4_mup_optimizer,
    summarize_v4_accounting,
)


def _flatten_pytree_npz(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten_pytree_npz(v, f"{prefix}{k}." if prefix else f"{k}."))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(_flatten_pytree_npz(v, f"{prefix}{i}."))
    else:
        key = prefix[:-1] if prefix.endswith(".") else prefix
        out[key] = np.asarray(tree)
    return out


def _unflatten_pytree_npz(template: Any, flat: dict[str, np.ndarray], prefix: str = "") -> Any:
    if isinstance(template, dict):
        return {k: _unflatten_pytree_npz(v, flat, f"{prefix}{k}." if prefix else f"{k}.") for k, v in template.items()}
    if isinstance(template, list):
        return [_unflatten_pytree_npz(v, flat, f"{prefix}{i}.") for i, v in enumerate(template)]
    if isinstance(template, tuple):
        return tuple(_unflatten_pytree_npz(v, flat, f"{prefix}{i}.") for i, v in enumerate(template))
    key = prefix[:-1] if prefix.endswith(".") else prefix
    return jnp.asarray(flat[key], dtype=template.dtype)


def save_v4_checkpoint(
    path: Path,
    params: Any,
    metadata: dict[str, Any],
    *,
    opt_state: Any | None = None,
) -> None:
    """Persist params, optional exact Optax `opt_state`, and JSON cursor/step/RNG metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = _flatten_pytree_npz(params)
    np.savez(path, **flat)
    if opt_state is not None:
        opt_leaves = jax.tree_util.tree_leaves(opt_state)
        np.savez(path.with_name(path.stem + "_opt.npz"), **{f"arr_{i}": np.asarray(x) for i, x in enumerate(opt_leaves)})
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_v4_checkpoint(path: Path, params_template: Any) -> Any:
    with np.load(path) as z:
        flat = {k: z[k] for k in z.files}
    return _unflatten_pytree_npz(params_template, flat)


def load_v4_opt_state(path: Path, opt_state_template: Any) -> Any:
    opt_path = path.with_name(path.stem + "_opt.npz")
    if not opt_path.is_file():
        raise FileNotFoundError(f"Missing optimizer state file {opt_path} required for lossless single-pass resume.")
    template_leaves, treedef = jax.tree_util.tree_flatten(opt_state_template)
    with np.load(opt_path) as z:
        if len(z.files) != len(template_leaves):
            raise ValueError(f"Optimizer leaf count mismatch: ckpt={len(z.files)} != template={len(template_leaves)}")
        restored = [jnp.asarray(z[f"arr_{i}"], dtype=t.dtype) for i, t in enumerate(template_leaves)]
    return jax.tree_util.tree_unflatten(treedef, restored)


def _serialize_rng_key(key: jax.Array) -> list[int]:
    try:
        arr = np.asarray(jax.random.key_data(key), dtype=np.uint32).reshape(-1)
    except Exception:
        arr = np.asarray(key, dtype=np.uint32).reshape(-1)
    return [int(x) for x in arr.tolist()]


def _deserialize_rng_key(vals: list[int], template_key: jax.Array) -> jax.Array:
    arr = jnp.asarray(vals, dtype=jnp.uint32)
    try:
        return jax.random.wrap_key_data(arr, impl=jax.random.key_impl(template_key))
    except Exception:
        return arr.reshape(template_key.shape)


def _make_prefetched_eval_batches(
    *,
    local_root: Path | None,
    cache_dir: Path | None,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    mixture_weights: dict[str, float] | None,
    gate2_receipt: Path,
    exclusion_manifest: Path,
) -> list[dict[str, np.ndarray]]:
    """Collect a deterministic, unshuffled `split='dev'` batch list (identical across all 17 runs!)."""
    loader = V4SinglePassMixtureLoader(
        split="dev",
        seq_len=seq_len,
        batch_size=batch_size,
        seed=2026,
        shuffle=False,
        local_root=local_root,
        cache_dir=cache_dir,
        delete_cached_after_consume=True,
        mixture_weights=mixture_weights,
        gate2_receipt_path=gate2_receipt,
        exclusion_manifest_path=exclusion_manifest,
    )
    out: list[dict[str, np.ndarray]] = []
    try:
        for _ in range(n_batches):
            try:
                out.append(next(loader))
            except StopIteration:
                break
    finally:
        loader.close()
    return out


def run_v4_training(args: argparse.Namespace) -> dict[str, Any]:
    spec: SLWMv4Spec = make_slwm_v4_spec(profile=args.profile, run_id=args.run_id)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matmul_dtype = jnp.bfloat16 if args.matmul_dtype == "bf16" else jnp.float32
    pos_per_opt_step = int(args.seq_len) * int(args.batch_size) * int(args.grad_accum)
    total_steps = max(1, int(args.max_steps) if args.max_steps > 0 else int(args.target_packets // pos_per_opt_step))
    warmup_steps = max(1, int(round(total_steps * float(args.warmup_frac))))

    rng_root = jax.random.PRNGKey(int(args.seed))
    rng_init, rng_loop = jax.random.split(rng_root)
    params = init_v4_model_params(rng_init, spec)
    acct = summarize_v4_accounting(spec, params, init_v4_core_state(spec, batch_size=1))

    opt = make_v4_mup_optimizer(
        params,
        spec,
        peak_lr=float(args.peak_lr),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
    )
    opt_state = opt.init(params)

    cfg_dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spec": asdict(spec),
        "train_args": {
            "profile": args.profile,
            "run_id": args.run_id,
            "seed": int(args.seed),
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "grad_accum": int(args.grad_accum),
            "pos_per_opt_step": pos_per_opt_step,
            "target_packets": int(args.target_packets),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "peak_lr": float(args.peak_lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "matmul_dtype": args.matmul_dtype,
            "use_remat": bool(args.use_remat),
        },
        "dual_track_accounting": acct,
        "devices": [str(d) for d in jax.devices()],
    }
    # Hash deterministic spec + train_args (excluding wall-clock timestamp so resume checks match identically!)
    cfg_hash = hashlib.sha256(
        json.dumps({"spec": cfg_dict["spec"], "train_args": cfg_dict["train_args"]}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cfg_dict["config_hash"] = cfg_hash
    (out_dir / "run_config.json").write_text(json.dumps(cfg_dict, indent=2))

    mix_weights = json.loads(args.mixture_weights_json) if args.mixture_weights_json else None
    local_root = Path(args.local_data_root) if args.local_data_root else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    train_loader = V4SinglePassMixtureLoader(
        split="train",
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        shuffle=True,
        local_root=local_root,
        cache_dir=cache_dir,
        delete_cached_after_consume=True,
        mixture_weights=mix_weights,
        gate2_receipt_path=Path(args.gate2_receipt),
        exclusion_manifest_path=Path(args.exclusion_manifest),
    )

    start_step = 0
    if args.resume_ckpt:
        r_path = Path(args.resume_ckpt)
        r_meta_path = r_path.with_suffix(".json")
        if not (r_path.is_file() and r_meta_path.is_file()):
            raise FileNotFoundError(f"Resume checkpoint or metadata missing: {r_path} / {r_meta_path}")
        r_meta = json.loads(r_meta_path.read_text())
        if "loader_state" not in r_meta or "step" not in r_meta or "rng_loop" not in r_meta:
            raise ValueError(
                f"Resume checkpoint {r_meta_path} lacks mandatory single-pass cursor/step/rng state; "
                "refusing to silently replay already-consumed shards."
            )
        if r_meta.get("config_hash") and r_meta["config_hash"] != cfg_hash:
            raise ValueError(
                f"Resume config_hash mismatch: ckpt={r_meta.get('config_hash')} != current={cfg_hash}"
            )
        params = load_v4_checkpoint(r_path, params)
        opt_state = load_v4_opt_state(r_path, opt_state)
        rng_loop = _deserialize_rng_key(r_meta["rng_loop"], rng_loop)
        train_loader.load_state(r_meta["loader_state"])
        start_step = int(r_meta["step"])
        print(
            f"[V4_TRAIN RESUME] Restored exact step={start_step}/{total_steps}, opt_state, rng_loop & single-pass cursor "
            f"(packets_seen={train_loader.stats()['total_packets_yielded']:,})",
            flush=True,
        )

    dev_batches: list[dict[str, np.ndarray]] = []
    if int(args.eval_batches) > 0:
        dev_batches = _make_prefetched_eval_batches(
            local_root=local_root,
            cache_dir=cache_dir,
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            n_batches=int(args.eval_batches),
            mixture_weights=mix_weights,
            gate2_receipt=Path(args.gate2_receipt),
            exclusion_manifest=Path(args.exclusion_manifest),
        )

    use_remat = bool(args.use_remat)

    @jax.jit
    def _train_opt_step(p: Any, ost: Any, micro_batches: dict[str, jax.Array], rng_step: jax.Array):
        n_acc = int(args.grad_accum)
        sub_keys = jax.random.split(rng_step, n_acc)

        def _micro_loss(pp: Any, mb: dict[str, jax.Array], k_i: jax.Array):
            return compute_v4_training_loss(
                pp,
                mb,
                spec,
                k_i,
                matmul_dtype=matmul_dtype,
                wy_chunk_size=64,
                gqa_chunk_size=64,
                use_remat=use_remat,
                assume_all_valid=False,
            )

        grad_fn = jax.value_and_grad(_micro_loss, has_aux=True)

        if n_acc == 1:
            mb0 = {k: v[0] for k, v in micro_batches.items()}
            (loss, metrics), grads = grad_fn(p, mb0, sub_keys[0])
        else:
            def _accum_body(carry, xs):
                g_acc, l_acc, m_acc = carry
                mb_i, k_i = xs
                (l_i, m_i), g_i = grad_fn(p, mb_i, k_i)
                g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc, g_i)
                m_acc = jax.tree_util.tree_map(lambda a, b: a + b, m_acc, m_i)
                return (g_acc, l_acc + l_i, m_acc), None

            mb0 = {k: v[0] for k, v in micro_batches.items()}
            (l0, m0), g0 = grad_fn(p, mb0, sub_keys[0])
            rest_mb = {k: v[1:] for k, v in micro_batches.items()}
            (g_sum, l_sum, m_sum), _ = jax.lax.scan(_accum_body, (g0, l0, m0), (rest_mb, sub_keys[1:]))
            inv_n = 1.0 / float(n_acc)
            grads = jax.tree_util.tree_map(lambda x: x * inv_n, g_sum)
            loss = l_sum * inv_n
            metrics = jax.tree_util.tree_map(lambda x: x * inv_n, m_sum)

        grad_norm = jnp.sqrt(
            sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree_util.tree_leaves(grads))
        )
        updates, new_ost = opt.update(grads, ost, p)
        new_p = optax.apply_updates(p, updates)
        metrics["grad_norm"] = grad_norm
        return new_p, new_ost, loss, metrics

    @jax.jit
    def _eval_batch(p: Any, b: dict[str, jax.Array], rng_e: jax.Array):
        return compute_v4_training_loss(
            p, b, spec, rng_e, matmul_dtype=matmul_dtype, wy_chunk_size=64, gqa_chunk_size=64, use_remat=False, assume_all_valid=False
        )

    def _run_dev_eval(p: Any, step_i: int) -> dict[str, float]:
        """Target-count-weighted dev evaluation (zero sparse-batch zero-averaging bias!)."""
        if not dev_batches:
            return {}
        count_map = {
            "text_ce_nat": "n_text_targets",
            "text_bpb": "n_text_targets",
            "audio_rvq_ce_nat": "n_audio_targets",
            "audio_rvq0_ce_nat": "n_audio_targets",
            "video_loss": "n_video_targets",
            "action_huber": "n_action_targets",
            "proprio_huber": "n_proprio_targets",
            "success_bce": "n_success_targets",
            "jepa_sync_loss": "n_jepa_pairs",
        }
        num: dict[str, float] = {}
        den: dict[str, float] = {}
        raw: dict[str, list[float]] = {}
        for bi, db in enumerate(dev_batches):
            _, m = _eval_batch(p, {k: jnp.asarray(v) for k, v in db.items()}, jax.random.PRNGKey(9000 + bi))
            mf = {mk: float(mv) for mk, mv in m.items()}
            for mk, mv in mf.items():
                if mk.startswith("n_"):
                    num[mk] = num.get(mk, 0.0) + mv
                elif mk in count_map:
                    w = mf.get(count_map[mk], 0.0)
                    num[mk] = num.get(mk, 0.0) + mv * w
                    den[mk] = den.get(mk, 0.0) + w
                elif mk == "motor_loss":
                    w = mf.get("n_action_targets", 0.0) + mf.get("n_proprio_targets", 0.0)
                    num[mk] = num.get(mk, 0.0) + mv * w
                    den[mk] = den.get(mk, 0.0) + w
                else:
                    raw.setdefault(mk, []).append(mv)
        out: dict[str, float] = {}
        for mk, vs in raw.items():
            out[f"dev_{mk}"] = round(float(np.mean(vs)), 5)
        for mk, tot in num.items():
            if mk.startswith("n_"):
                out[f"dev_{mk}"] = round(tot, 1)
            else:
                d_w = den.get(mk, 0.0)
                out[f"dev_{mk}"] = round(tot / d_w, 5) if d_w > 0.5 else 0.0
        return out

    metrics_path = out_dir / "train_metrics.jsonl"
    t_start = time.monotonic()
    last_eval: dict[str, float] = {}
    steps_done = start_step
    exhausted_early = False

    steps_this_call = 0
    stop_after = int(getattr(args, "stop_after_steps", 0) or 0)

    try:
        for step_i in range(start_step + 1, total_steps + 1):
            if stop_after > 0 and steps_this_call >= stop_after:
                break
            mbs = []
            for _ in range(int(args.grad_accum)):
                try:
                    mbs.append(next(train_loader))
                except StopIteration:
                    exhausted_early = True
                    break
            if len(mbs) < int(args.grad_accum):
                break
            stacked_mb = {k: jnp.asarray(np.stack([m[k] for m in mbs], axis=0)) for k in mbs[0]}
            rng_loop, k_step = jax.random.split(rng_loop)

            t0 = time.perf_counter()
            params, opt_state, loss_val, m_out = _train_opt_step(params, opt_state, stacked_mb, k_step)
            loss_f = float(loss_val)
            gnorm_f = float(m_out["grad_norm"])
            dt_step = time.perf_counter() - t0
            steps_done = step_i
            steps_this_call += 1

            if not (np.isfinite(loss_f) and np.isfinite(gnorm_f)):
                raise FloatingPointError(f"Non-finite training state at step {step_i}: loss={loss_f}, grad_norm={gnorm_f}")

            if step_i == 1 or step_i % int(args.log_every) == 0 or step_i == total_steps:
                rec = {
                    "step": step_i,
                    "total_steps": total_steps,
                    "packets_seen": int(train_loader.stats()["total_packets_yielded"]),
                    "step_ms": round(dt_step * 1000.0, 2),
                    "pos_per_sec": round(pos_per_opt_step / max(dt_step, 1e-6), 1),
                    **{k: round(float(v), 5) for k, v in m_out.items()},
                }
                if dev_batches and (step_i == 1 or step_i % int(args.eval_every) == 0 or step_i == total_steps):
                    last_eval = _run_dev_eval(params, step_i)
                    rec.update(last_eval)
                with metrics_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(
                    f"[V4_TRAIN {args.run_id}] step={step_i}/{total_steps} pkts={rec['packets_seen']:,} "
                    f"loss={loss_f:.4f} txt_bpb={rec['text_bpb']:.3f} aud={rec['audio_rvq_ce_nat']:.3f} "
                    f"vid={rec['video_loss']:.3f} mot={rec['motor_loss']:.3f} gnorm={gnorm_f:.2f} "
                    f"({rec['pos_per_sec']:,.0f} pos/s)",
                    flush=True,
                )

            if int(args.ckpt_every) > 0 and step_i % int(args.ckpt_every) == 0:
                save_v4_checkpoint(
                    out_dir / "ckpt_latest.npz",
                    params,
                    {
                        "step": step_i,
                        "config_hash": cfg_hash,
                        "rng_loop": _serialize_rng_key(rng_loop),
                        "loader_state": train_loader.get_state(),
                        "loader_stats": train_loader.stats(),
                    },
                    opt_state=opt_state,
                )
    finally:
        loader_stats = train_loader.stats()
        final_loader_state = train_loader.get_state()
        train_loader.close()

    if dev_batches and not last_eval:
        last_eval = _run_dev_eval(params, steps_done)

    final_summary = {
        "run_id": args.run_id,
        "profile": args.profile,
        "config_hash": cfg_hash,
        "seed": int(args.seed),
        "step": steps_done,
        "steps_completed": steps_done,
        "total_steps_planned": total_steps,
        "single_pass_exhausted_early": exhausted_early,
        "wall_time_sec": round(time.monotonic() - t_start, 2),
        "rng_loop": _serialize_rng_key(rng_loop),
        "loader_state": final_loader_state,
        "loader_stats": loader_stats,
        "final_dev_eval": last_eval,
        "dual_track_accounting": acct,
    }
    save_v4_checkpoint(out_dir / "ckpt_final.npz", params, final_summary, opt_state=opt_state)
    (out_dir / "train_summary.json").write_text(json.dumps(final_summary, indent=2))
    return final_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v4 Single-Pass Multimodal Trainer.")
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--run-id", type=str, default="A0_3to1_single_bank", choices=V4_S3_DISCOVERY_RUN_IDS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--target-packets", type=int, default=450_000_000)
    ap.add_argument("--max-steps", type=int, default=0, help="If > 0, overrides target-packets total step count.")
    ap.add_argument("--stop-after-steps", type=int, default=0, help="Optional per-invocation step cap for clean time-sliced resume.")
    ap.add_argument("--peak-lr", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--matmul-dtype", type=str, default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--use-remat", action="store_true", default=True)
    ap.add_argument("--no-remat", action="store_false", dest="use_remat")
    ap.add_argument("--local-data-root", type=str, default="")
    ap.add_argument("--cache-dir", type=str, default="/tmp/slwm_v4_shard_cache")
    ap.add_argument("--mixture-weights-json", type=str, default="")
    ap.add_argument("--gate2-receipt", type=str, default=str(DEFAULT_GATE2_RECEIPT))
    ap.add_argument("--exclusion-manifest", type=str, default=str(DEFAULT_EXCLUSION_MANIFEST))
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--resume-ckpt", type=str, default="")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    args = ap.parse_args()
    run_v4_training(args)


if __name__ == "__main__":
    main()
