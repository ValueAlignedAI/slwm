"""SLWM-v3 TPU v6e-1 / CPU Hard-Capped Validation Pilot Runner (Rev 4).

Implements reproducible, config-driven JAX/XLA training & profiling for SLWM-v3 Wave A:
  - Exact base-shape muP AdamW optimizer (`lr_mult == 1.0` on every leaf at `d == d_base`,
    `1 / width_mult` on width-scaling 2D hidden/readout matrices) with STRICTLY MASKED,
    width-invariant independent weight decay (ZERO decay on `log_tau`, `log_a`, `norm_w`,
    biases, `thresh`, `gate_z0`, `rel_pos_bias`).
  - Compiled step reuse across LR sweeps (zero redundant XLA recompilation per base LR!).
  - Single-pass `eval_step` (zero duplicate `scan_core`), `val_batch_size >= 16` fresh
    100%-observable multimodal validation episodes, full-precision (`float64`) loss reporting,
    and post-warmup feature update RMS (`step3_delta_z_rms`, `end_delta_z_rms`).
  - Full mechanism telemetry (`mean_nowrite`, `mean_eff_write_eta`, `mean_eff_decay`,
    `mean_state_delta_norm`, `mean_mem_out_rms`, `mean_key_norm`, `effective_rank_pr`,
    pre-clip grad L2 norm, non-finite counts, TPU HBM & host RSS).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.v3.core import (
    build_mup_lr_multipliers,
    build_weight_decay_mask,
    init_core_state,
    init_slwm_v3_params,
)
from models.v3.objectives import masked_multi_horizon_predictive_loss
from models.v3.spec import (
    SLWMv3Spec,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _host_rss_mib() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru) / (1024.0 * 1024.0) if ru > 10_000_000 else float(ru) / 1024.0


def _device_hbm_stats_mib() -> dict[str, float]:
    dev = jax.devices()[0]
    try:
        st = dev.memory_stats() or {}
        return {
            "bytes_in_use_mib": round(float(st.get("bytes_in_use", 0)) / (1024.0**2), 2),
            "peak_bytes_in_use_mib": round(float(st.get("peak_bytes_in_use", 0)) / (1024.0**2), 2),
            "bytes_limit_mib": round(float(st.get("bytes_limit", 0)) / (1024.0**2), 2),
        }
    except Exception:
        return {"bytes_in_use_mib": 0.0, "peak_bytes_in_use_mib": 0.0, "bytes_limit_mib": 0.0}


def effective_rank_participation_ratio(z_seq: jax.Array) -> jax.Array:
    """Compute normalized covariance participation ratio (effective rank / D) of `Z_seq: [T, B, D]`."""
    x = z_seq.reshape(-1, z_seq.shape[-1]).astype(jnp.float32)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    s = jnp.linalg.svd(x, compute_uv=False)
    lam = (s * s) / jnp.maximum(float(x.shape[0] - 1), 1.0)
    pr = (jnp.sum(lam) ** 2) / jnp.maximum(jnp.sum(lam * lam), 1e-12)
    return pr / float(x.shape[-1])


def build_optimizer(
    params: dict[str, Any],
    spec: SLWMv3Spec,
    *,
    base_lr: float,
    weight_decay: float = 0.01,
    use_mup_lr_scaling: bool = True,
    grad_clip_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Build exact per-leaf base-shape muP AdamW optimizer with masked width-invariant weight decay.

    Update rule per leaf `p` with muP multiplier `m` (`1.0` everywhere at `d == d_base`) and WD mask `wd_m`:
      `delta_p = -(base_lr * m) * adam(g) - (base_lr * weight_decay) * wd_m * p`
    Guarantees:
      1. Exact SP identity at `d == d_base` (`m == 1.0` on EVERY leaf!).
      2. Width-invariant independent decoupled weight decay (`base_lr * weight_decay`, applied AFTER `_scale_by_mult`).
      3. Strict ZERO weight decay on all 1D/scalar/timescale/norm parameters (`wd_m == False`).
    """
    if use_mup_lr_scaling:
        mults = build_mup_lr_multipliers(params, spec)
    else:
        mults = jax.tree_util.tree_map(lambda _: 1.0, params)
    wd_mask = build_weight_decay_mask(params)

    def _scale_by_mult() -> optax.GradientTransformation:
        def init_fn(_p):
            return optax.EmptyState()

        def update_fn(updates, state, _p=None):
            scaled = jax.tree_util.tree_map(lambda u, m: u * jnp.asarray(m, dtype=u.dtype), updates, mults)
            return scaled, state

        return optax.GradientTransformation(init_fn, update_fn)

    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        _scale_by_mult(),
        optax.add_decayed_weights(weight_decay, mask=wd_mask),
        optax.scale(-base_lr),
    )


def lr_schedule_mult(step_idx: int, total_steps: int, min_mult: float = 0.1) -> float:
    """Deterministic 10% linear warmup + cosine decay multiplier in `[min_mult, 1.0]`."""
    if total_steps <= 1:
        return 1.0
    warmup = max(1, min(total_steps // 5, int(round(0.10 * total_steps))))
    if step_idx < warmup:
        return float(step_idx + 1) / float(warmup)
    denom = max(1, total_steps - warmup)
    progress = min(1.0, float(step_idx - warmup + 1) / float(denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_mult + (1.0 - min_mult) * cosine)


def make_synthetic_multimodal_batch(
    spec: SLWMv3Spec,
    *,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    """Build deterministic causal multimodal batch where 100% of dynamical & overwrite targets are observable.

    Combines:
      1. Smooth coupled damped multimodal latent dynamics `z_dyn[t]` driven by strictly observable
         action impulses (`has_true_action[t - 1] == 1`) and projected across ALL 5 modalities
         (`video_feat`, `audio_codes` & `audio_cont`, `text_byte`, `proprio_feat`, `probe_feat`).
      2. Persistent world factors `(f1, f2)` exposed at `t = 0` and randomized overwrite `f1_new`
         exposed at `t = t_switch`, with `[t_switch, t_switch + max_h)` strictly masked (`tvalid = 0`)
         so NO pre-switch source `t < t_switch` is ever supervised on an unrevealed future `f1_new`.
    """
    rng = np.random.default_rng(seed)
    f_dim = min(16, spec.probe_dim // 2)
    t_lo = max(2, seq_len // 4)
    t_hi = max(t_lo + 1, seq_len // 2)
    t_switch = rng.integers(t_lo, t_hi, size=(batch_size,))

    mod_ids = np.full((seq_len, batch_size), 4, dtype=np.int32)
    for t in range(1, seq_len):
        rem = t % 5
        if rem == 1:
            mod_ids[t, :] = 1
        elif rem == 2:
            mod_ids[t, :] = 2
        elif rem == 3:
            mod_ids[t, :] = 3
        elif rem == 4:
            mod_ids[t, :] = 0
    for b in range(batch_size):
        mod_ids[0, b] = 4
        mod_ids[int(t_switch[b]), b] = 4

    has_act = (mod_ids == 3).astype(np.float32)
    act = rng.normal(0.0, 0.4, size=(seq_len, batch_size, spec.action_dim)).astype(np.float32)
    z_dyn = np.zeros((seq_len, batch_size, f_dim), dtype=np.float32)
    z_dyn[0] = rng.normal(0.0, 0.8, size=(batch_size, f_dim)).astype(np.float32)
    omega = np.linspace(0.08, 0.35, f_dim, dtype=np.float32)[None, :]
    act_dims = min(f_dim, spec.action_dim)
    for t in range(1, seq_len):
        z_dyn[t] = 0.88 * z_dyn[t - 1] + 0.25 * np.sin(omega * float(t) + z_dyn[t - 1])
        z_dyn[t, :, :act_dims] += 0.30 * act[t - 1, :, :act_dims] * has_act[t - 1, :, None]

    f1 = rng.normal(0.0, 0.8, size=(batch_size, f_dim)).astype(np.float32)
    f2 = rng.normal(0.0, 0.8, size=(batch_size, f_dim)).astype(np.float32)
    f1_new = rng.normal(0.0, 0.8, size=(batch_size, f_dim)).astype(np.float32)

    targs = np.zeros((seq_len, batch_size, spec.target_feat_dim), dtype=np.float32)
    probe = np.zeros((seq_len, batch_size, spec.probe_dim), dtype=np.float32)
    vid = np.zeros((seq_len, batch_size, spec.video_patch_dim), dtype=np.float32)
    prop = np.zeros((seq_len, batch_size, spec.proprio_dim), dtype=np.float32)
    aud_cont = np.zeros((seq_len, batch_size, spec.audio_cont_dim), dtype=np.float32)
    tvalid = np.ones((seq_len, batch_size), dtype=np.float32)
    max_h = max(spec.pred_horizons)
    prop_dims = min(f_dim, spec.proprio_dim)
    for b in range(batch_size):
        ts = int(t_switch[b])
        # Mask target validity strictly across the unrevealed overwrite crossing `[ts, ts + max_h)` so
        # NO pre-switch source `t < ts` is supervised on an independently sampled future `f1_new`!
        tvalid[ts : min(seq_len, ts + max_h), b] = 0.0
        for t in range(seq_len):
            cur_f1 = f1_new[b] if t >= ts else f1[b]
            zd = z_dyn[t, b]
            full_state = np.concatenate(
                [
                    zd,
                    cur_f1,
                    f2[b],
                    0.5 * (zd + cur_f1),
                ],
                axis=0,
            )
            if len(full_state) >= spec.target_feat_dim:
                targs[t, b, :] = full_state[: spec.target_feat_dim]
            else:
                reps = (spec.target_feat_dim + len(full_state) - 1) // len(full_state)
                targs[t, b, :] = np.tile(full_state, reps)[: spec.target_feat_dim]

            prop[t, b, :prop_dims] = zd[:prop_dims]
            vid[t, b, :f_dim] = zd
            if spec.video_patch_dim >= 2 * f_dim:
                vid[t, b, f_dim : 2 * f_dim] = np.tanh(zd * 0.7)
            aud_reps = (spec.audio_cont_dim + f_dim - 1) // f_dim
            aud_cont[t, b, :] = np.tile(zd, aud_reps)[: spec.audio_cont_dim]

            # Expose 100% of both factor vectors at t=0 and 100% of overwrite f1_new at t=ts!
            if t == 0:
                probe[t, b, :f_dim] = f1[b]
                probe[t, b, f_dim : 2 * f_dim] = f2[b]
            elif t == ts:
                probe[t, b, :f_dim] = f1_new[b]
                probe[t, b, f_dim : 2 * f_dim] = zd
            else:
                probe[t, b, :f_dim] = zd
                probe[t, b, f_dim : 2 * f_dim] = rng.normal(0.0, 0.15, size=(f_dim,)).astype(np.float32)

    aud_q = np.clip(((z_dyn + 3.0) * 300.0).astype(np.int32), 0, spec.audio_vocab_size - 1)
    cb_reps = (spec.audio_codebooks + f_dim - 1) // f_dim
    aud_codes = np.tile(aud_q, (1, 1, cb_reps))[:, :, : spec.audio_codebooks]
    text_b = np.clip((z_dyn[:, :, 0] * 20.0 + 80.0).astype(np.int32), 32, min(126, spec.text_vocab_size - 1))

    pkts = {
        "modality_id": jnp.asarray(mod_ids, dtype=jnp.int32),
        "dt_s": jnp.full((seq_len, batch_size), 0.08, dtype=jnp.float32),
        "valid_mask": jnp.ones((seq_len, batch_size), dtype=jnp.float32),
        "reliability": jnp.ones((seq_len, batch_size), dtype=jnp.float32),
        "video_feat": jnp.asarray(vid, dtype=jnp.float32),
        "spatial_pos": jnp.zeros((seq_len, batch_size), dtype=jnp.int32),
        "audio_codes": jnp.asarray(aud_codes, dtype=jnp.int32),
        "audio_cont": jnp.asarray(aud_cont, dtype=jnp.float32),
        "text_byte": jnp.asarray(text_b, dtype=jnp.int32),
        "proprio_feat": jnp.asarray(prop, dtype=jnp.float32),
        "action_feat": jnp.asarray(act, dtype=jnp.float32),
        "has_true_action": jnp.asarray(has_act, dtype=jnp.float32),
        "probe_feat": jnp.asarray(probe, dtype=jnp.float32),
    }
    return pkts, jnp.asarray(targs, dtype=jnp.float32), jnp.asarray(tvalid, dtype=jnp.float32)


def compile_pilot_steps(
    params_template: dict[str, Any],
    spec: SLWMv3Spec,
    *,
    batch_size: int,
    val_batch_size: int,
    use_mup_lr_scaling: bool = True,
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
    remat_step: bool = True,
    remat_chunk_size: int = 64,
) -> tuple[optax.GradientTransformation, Any, Any]:
    """Compile `train_step` (accepting dynamic scalar `base_lr`) and single-pass `eval_step` ONCE per shape.

    Eliminates duplicate XLA compilations across multi-LR sweeps (e.g., 5 LRs per width in Phase 2).
    """
    tx_unit = build_optimizer(
        params_template,
        spec,
        base_lr=1.0,
        weight_decay=weight_decay,
        use_mup_lr_scaling=use_mup_lr_scaling,
        grad_clip_norm=grad_clip_norm,
    )

    @jax.jit
    def eval_step(p, val_pkts, val_targs, val_valid):
        st0 = init_core_state(spec, val_batch_size)
        loss, metrics = masked_multi_horizon_predictive_loss(
            p,
            st0,
            val_pkts,
            val_targs,
            val_valid,
            spec,
            remat_step=False,
            return_z_seq=True,
        )
        z_seq = metrics.pop("z_seq")
        pr = effective_rank_participation_ratio(z_seq)
        return loss, metrics, pr, z_seq

    @jax.jit(donate_argnums=(0, 1))
    def train_step(p, o_st, pkts, targs, tvalid, base_lr_arr):
        st0 = init_core_state(spec, batch_size)
        (loss, metrics), grads = jax.value_and_grad(
            lambda pp: masked_multi_horizon_predictive_loss(
                pp,
                st0,
                pkts,
                targs,
                tvalid,
                spec,
                remat_step=remat_step,
                remat_chunk_size=remat_chunk_size,
            ),
            has_aux=True,
        )(p)
        leaves = jax.tree_util.tree_leaves(grads)
        pre_clip_gnorm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in leaves) + 1e-12)
        nan_count = sum(jnp.sum(~jnp.isfinite(g)) for g in leaves)
        unit_updates, new_o_st = tx_unit.update(grads, o_st, p)
        scaled_updates = jax.tree_util.tree_map(lambda u: u * base_lr_arr.astype(u.dtype), unit_updates)
        new_p = optax.apply_updates(p, scaled_updates)
        return new_p, new_o_st, loss, metrics, pre_clip_gnorm, nan_count

    return tx_unit, train_step, eval_step


def run_pilot_train(
    spec: SLWMv3Spec,
    *,
    steps: int = 20,
    seq_len: int = 64,
    batch_size: int = 4,
    val_batch_size: int = 16,
    base_lr: float = 1e-3,
    weight_decay: float = 0.01,
    use_mup_lr_scaling: bool = True,
    remat_step: bool = True,
    remat_chunk_size: int = 64,
    seed: int = 3001,
    compiled_bundle: tuple[optax.GradientTransformation, Any, Any] | None = None,
    val_batch: tuple[dict[str, jax.Array], jax.Array, jax.Array] | None = None,
    return_params: bool = False,
) -> dict[str, Any]:
    """Run JIT-compiled training & profiling loop with full post-warmup diagnostics and memory telemetry."""
    params = init_slwm_v3_params(jax.random.PRNGKey(seed), spec)
    n_params = count_tree_params(params)
    state_bytes = compute_core_streaming_state_bytes(spec, batch_size=batch_size)

    if compiled_bundle is None:
        tx_unit, train_step, eval_step = compile_pilot_steps(
            params,
            spec,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            use_mup_lr_scaling=use_mup_lr_scaling,
            weight_decay=weight_decay,
            remat_step=remat_step,
            remat_chunk_size=remat_chunk_size,
        )
    else:
        tx_unit, train_step, eval_step = compiled_bundle

    opt_state = tx_unit.init(params)

    if val_batch is None:
        val_pkts, val_targs, val_valid = make_synthetic_multimodal_batch(
            spec, seq_len=seq_len, batch_size=val_batch_size, seed=seed + 999_999
        )
    else:
        val_pkts, val_targs, val_valid = val_batch

    val_loss_0, _, pr_0, z_val_0 = eval_step(params, val_pkts, val_targs, val_valid)
    val_loss_0.block_until_ready()

    t_compile_0 = time.perf_counter()
    pkts_0, targs_0, tvalid_0 = make_synthetic_multimodal_batch(spec, seq_len=seq_len, batch_size=batch_size, seed=seed)
    lr_0 = jnp.asarray(base_lr * lr_schedule_mult(0, steps), dtype=jnp.float32)
    params, opt_state, loss_0, m_0, gn_0, nan_0 = train_step(params, opt_state, pkts_0, targs_0, tvalid_0, lr_0)
    loss_0.block_until_ready()
    compile_plus_step0_s = time.perf_counter() - t_compile_0

    warm_times_ms: list[float] = []
    train_losses = [float(loss_0)]
    pre_clip_gnorms = [float(gn_0)]
    total_nan_grads = int(nan_0)
    last_metrics = m_0
    step3_delta_z_rms = 0.0

    for st in range(1, steps):
        pkts, targs, tvalid = make_synthetic_multimodal_batch(spec, seq_len=seq_len, batch_size=batch_size, seed=seed + st)
        step_lr = jnp.asarray(base_lr * lr_schedule_mult(st, steps), dtype=jnp.float32)
        t0 = time.perf_counter()
        params, opt_state, loss, last_metrics, gn, nans = train_step(params, opt_state, pkts, targs, tvalid, step_lr)
        loss.block_until_ready()
        warm_times_ms.append((time.perf_counter() - t0) * 1000.0)
        train_losses.append(float(loss))
        pre_clip_gnorms.append(float(gn))
        total_nan_grads += int(nans)
        if st == 2 or (steps <= 3 and st == steps - 1):
            _, _, _, z_val_3 = eval_step(params, val_pkts, val_targs, val_valid)
            step3_delta_z_rms = float(jnp.sqrt(jnp.mean((z_val_3 - z_val_0) ** 2)))

    val_loss_end, val_m_end, pr_end, z_val_end = eval_step(params, val_pkts, val_targs, val_valid)
    val_loss_end.block_until_ready()
    end_delta_z_rms = float(jnp.sqrt(jnp.mean((z_val_end - z_val_0) ** 2)))
    val_loss_end_exact = float(val_loss_end)
    all_finite = bool(
        np.isfinite(val_loss_end_exact)
        and np.isfinite(train_losses[-1])
        and total_nan_grads == 0
        and np.isfinite(end_delta_z_rms)
    )

    cfg_dict = {
        "spec": asdict(spec),
        "steps": steps,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "base_lr": base_lr,
        "weight_decay": weight_decay,
        "use_mup_lr_scaling": use_mup_lr_scaling,
        "remat_step": remat_step,
        "remat_chunk_size": remat_chunk_size,
        "seed": seed,
    }
    res: dict[str, Any] = {
        "config_hash": _config_hash(cfg_dict),
        "device_kind": str(jax.devices()[0].device_kind),
        "spec_name": spec.name,
        "backbone_type": spec.backbone_type,
        "representation_mode": spec.representation_mode,
        "trainable_core_params": int(n_params),
        "core_streaming_state_bytes": state_bytes,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "val_batch_size": int(val_batch_size),
        "remat_chunk_size": int(remat_chunk_size),
        "compile_plus_step0_s": round(compile_plus_step0_s, 3),
        "warm_step_ms_median": round(float(np.median(warm_times_ms)), 2) if warm_times_ms else 0.0,
        "warm_positions_per_sec": round((seq_len * batch_size) / max(float(np.median(warm_times_ms)) / 1000.0, 1e-6), 1)
        if warm_times_ms
        else 0.0,
        "train_loss_init": round(train_losses[0], 6),
        "train_loss_end": round(train_losses[-1], 6),
        "val_loss_init": round(float(val_loss_0), 6),
        "val_loss_end": round(val_loss_end_exact, 6),
        "val_loss_end_exact": val_loss_end_exact,
        "step3_delta_z_rms": round(step3_delta_z_rms, 6),
        "end_delta_z_rms": round(end_delta_z_rms, 6),
        "val_z_rms_end": round(float(val_m_end["z_rms"]), 5),
        "val_effective_rank_pr_init": round(float(pr_0), 5),
        "val_effective_rank_pr_end": round(float(pr_end), 5),
        "pre_clip_gnorm_max": round(max(pre_clip_gnorms), 4),
        "pre_clip_gnorm_median": round(float(np.median(pre_clip_gnorms)), 4),
        "total_nonfinite_grad_entries": total_nan_grads,
        "all_finite": all_finite,
        "mean_nowrite_end": round(float(last_metrics["mean_nowrite"]), 4),
        "mean_eff_write_eta_end": round(float(last_metrics["mean_eff_write_eta"]), 5),
        "mean_eff_decay_end": round(float(last_metrics["mean_eff_decay"]), 5),
        "mean_state_delta_norm_end": round(float(last_metrics["mean_state_delta_norm"]), 5),
        "mean_mem_out_rms_end": round(float(last_metrics["mean_mem_out_rms"]), 5),
        "mean_key_norm_end": round(float(last_metrics["mean_key_norm"]), 6),
        "host_rss_mib": round(_host_rss_mib(), 2),
        "device_hbm_stats_mib": _device_hbm_stats_mib(),
    }
    if return_params:
        res["_trained_params"] = params
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 TPU v6e-1 / CPU validation pilot runner.")
    ap.add_argument(
        "--profile",
        type=str,
        default="discovery_45m",
        choices=("tiny_unit", "pilot_d256", "discovery_45m", "confirmation_135m"),
    )
    ap.add_argument("--backbone", type=str, default="slwm_v3_routed")
    ap.add_argument("--rep", type=str, default="frozen_codec_edge", choices=("frozen_codec_edge", "scratch_edge"))
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--val-batch-size", type=int, default=16)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--sp-uniform-lr", action="store_true", help="Disable muP LR scaling (SP baseline comparator).")
    ap.add_argument("--no-remat", action="store_true")
    ap.add_argument("--remat-chunk-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3001)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    spec = make_slwm_v3_spec(
        args.profile,
        backbone_type=args.backbone,
        representation_mode=args.rep,
        d_model_override=args.d_model,
        n_layers_override=args.n_layers,
    )
    res = run_pilot_train(
        spec,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        base_lr=args.base_lr,
        use_mup_lr_scaling=not args.sp_uniform_lr,
        remat_step=not args.no_remat,
        remat_chunk_size=args.remat_chunk_size,
        seed=args.seed,
    )
    txt = json.dumps(res, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
