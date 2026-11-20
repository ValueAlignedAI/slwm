"""SLWM-v3 Wave A-Speedup Empirical Hardware Benchmark (`TPU v6e-1` vs `NVIDIA H100`).

Benchmarks our hoisted sequence-parallel `bf16` MXU core (`models/v3/fast_core.py:scan_core_fast`),
Chunkwise Parallel WY Delta-Rule (`use_wy_chunking=True`), and `O(1)` batch-1 online streaming kernel
(`models/v3/core.py:step_core`) across:
  - `discovery_45m` (`~42.4M` core params, `L=12, d=512`)
  - `confirmation_135m` (`~137.9M` core params, `L=16, d=768`)
  - `scale_700m` (`~701M` core params, `L=24, d=1536, H_tot=8, dk=dv=64`)
  - `scale_1b` (`~1.07B` core params, `L=24, d=1920, H_tot=8, dk=dv=64`)

Reports for every configuration:
  - exact trainable core params & streaming state bytes (`MiB`),
  - compile+step0 wall time, warm median step latency (`ms`), warm throughput (`pos/s`),
  - peak device HBM (`MiB`), finite gradient verification,
  - batch-1 online streaming `step_core` p50 and p95 latency (`ms/tick`).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.v3.core import (
    init_core_state,
    init_slwm_v3_params,
    predict_future_latents,
    step_core,
)
from models.v3.fast_core import scan_core_fast
from models.v3.spec import (
    SLWMv3Spec,
    build_wave_a_comparator_specs,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)
from scripts.train.v3_pilot_train import build_optimizer, make_synthetic_multimodal_batch


def make_scale_spec(
    profile: str,
    backbone_type: str = "slwm_v3_routed",
    *,
    surprise_chunk_size: int = 1,
    match_anchor: bool = True,
) -> SLWMv3Spec:
    if match_anchor and profile in ("pilot_d256", "discovery_45m", "confirmation_135m", "scale_700m", "scale_1b"):
        specs = build_wave_a_comparator_specs(profile, match_params_to_anchor=True)
        key = "sliding_window_gqa_eqbyte" if backbone_type == "sliding_window_gqa" else backbone_type
        base = specs[key]
    else:
        base = make_slwm_v3_spec(profile, backbone_type=backbone_type)
    if surprise_chunk_size > 1 and base.use_surprise_router:
        base = replace(base, surprise_chunk_size=int(surprise_chunk_size))
    return base


def _device_hbm_mib() -> dict[str, float]:
    dev = jax.devices()[0]
    try:
        st = dev.memory_stats() or {}
        return {
            "bytes_in_use_mib": round(float(st.get("bytes_in_use", 0)) / (1024.0**2), 2),
            "peak_bytes_in_use_mib": round(float(st.get("peak_bytes_in_use", 0)) / (1024.0**2), 2),
            "bytes_limit_mib": round(float(st.get("bytes_limit", 0)) / (1024.0**2), 2),
        }
    except Exception:
        return {}


def _fast_predictive_loss(
    core_params: dict[str, Any],
    init_state: list[dict[str, Any]],
    packets_seq: dict[str, jax.Array],
    target_feats_seq: jax.Array,
    target_valid_seq: jax.Array,
    spec: SLWMv3Spec,
    *,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> jax.Array:
    z_seq, _, _ = scan_core_fast(
        core_params,
        init_state,
        packets_seq,
        spec,
        remat_layers=True,
        matmul_dtype=matmul_dtype,
        gqa_chunk_size=64,
        assume_all_valid=True,
        use_wy_chunking=use_wy_chunking,
        wy_chunk_size=64,
    )
    t_len, b_sz, _ = z_seq.shape
    act_seq = packets_seq.get("action_feat", jnp.zeros((t_len, b_sz, spec.action_dim), dtype=jnp.float32))
    has_act = packets_seq.get("has_true_action", jnp.zeros((t_len, b_sz), dtype=jnp.float32))
    src_v = packets_seq["valid_mask"].astype(jnp.float32)
    preds = predict_future_latents(core_params, z_seq, act_seq, has_act, spec)

    losses = []
    for hi, h in enumerate(spec.pred_horizons):
        p_h = preds[:-h, :, hi, :]
        y_h = jax.lax.stop_gradient(target_feats_seq[h:, :, :].astype(jnp.float32))
        m_h = src_v[:-h, :] * target_valid_seq[h:, :].astype(jnp.float32)
        d = p_h - y_h
        ad = jnp.abs(d)
        huber = jnp.where(ad <= 1.0, 0.5 * d * d, ad - 0.5)
        losses.append(jnp.sum(jnp.mean(huber, axis=-1) * m_h) / jnp.maximum(jnp.sum(m_h), 1.0))
    return jnp.mean(jnp.stack(losses, axis=0))


def benchmark_config(
    profile: str,
    backbone: str,
    *,
    batch_size: int,
    seq_len: int,
    use_wy_chunking: bool = False,
    surprise_chunk_size: int = 1,
    n_warm_steps: int = 8,
    n_stream_ticks: int = 32,
    use_bf16_params_opt: bool = False,
) -> dict[str, Any]:
    spec = make_scale_spec(profile, backbone_type=backbone, surprise_chunk_size=surprise_chunk_size, match_anchor=True)
    st_bytes = compute_core_streaming_state_bytes(spec, batch_size=1)
    n_params = count_tree_params(jax.eval_shape(lambda k: init_slwm_v3_params(k, spec), jax.random.PRNGKey(0)))

    params = opt_st = pkts = targs = tvalid = st0 = st1 = _train_step = _online_tick = None
    try:
        params = init_slwm_v3_params(jax.random.PRNGKey(701), spec)
        if use_bf16_params_opt:
            # Optional memory-saving bf16 param cast for 1B fit-check on 31 GiB TPU v6e-1
            params = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16) if x.ndim >= 2 else x, params
            )
        tx = build_optimizer(params, spec, base_lr=1e-3, weight_decay=0.01, use_mup_lr_scaling=True)
        opt_st = tx.init(params)
        pkts, targs, tvalid = make_synthetic_multimodal_batch(spec, seq_len=seq_len, batch_size=batch_size, seed=702)
        st0 = init_core_state(spec, batch_size)

        @jax.jit(donate_argnums=(0, 1))
        def _train_step(p, o_st, pk, tg, tv):
            loss, g = jax.value_and_grad(
                lambda pp: _fast_predictive_loss(
                    pp, st0, pk, tg, tv, spec, matmul_dtype=jnp.bfloat16, use_wy_chunking=use_wy_chunking
                )
            )(p)
            upd, new_o = tx.update(g, o_st, p)
            return optax.apply_updates(p, upd), new_o, loss

        t_c0 = time.perf_counter()
        params, opt_st, l0 = _train_step(params, opt_st, pkts, targs, tvalid)
        l0.block_until_ready()
        compile_s = time.perf_counter() - t_c0

        step_ms: list[float] = []
        losses = [float(l0)]
        for _ in range(n_warm_steps):
            t0 = time.perf_counter()
            params, opt_st, lv = _train_step(params, opt_st, pkts, targs, tvalid)
            lv.block_until_ready()
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            losses.append(float(lv))

        hbm = _device_hbm_mib()
        med_ms = float(np.median(step_ms))
        pos_per_s = (batch_size * seq_len) / max(med_ms / 1000.0, 1e-6)

        # Batch-1 online streaming `step_core` p50 & p95 latency (`ms/tick`)
        pkt1 = {k: v[0, :1] for k, v in pkts.items()}
        st1 = init_core_state(spec, 1)

        @jax.jit
        def _online_tick(p, st, pk1):
            return step_core(p, st, pk1, spec)

        z1, st1, _ = _online_tick(params, st1, pkt1)
        z1.block_until_ready()
        tick_ms: list[float] = []
        for _ in range(n_stream_ticks):
            t0 = time.perf_counter()
            z1, st1, _ = _online_tick(params, st1, pkt1)
            z1.block_until_ready()
            tick_ms.append((time.perf_counter() - t0) * 1000.0)

        res = {
            "profile": profile,
            "backbone": backbone,
            "use_wy_chunking": bool(use_wy_chunking),
            "surprise_chunk_size": int(surprise_chunk_size),
            "bf16_2d_params": bool(use_bf16_params_opt),
            "trainable_core_params": int(n_params),
            "core_state_mib_b1": round(st_bytes["total_core_state_bytes"] / (1024.0**2), 4),
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "compile_plus_step0_s": round(compile_s, 2),
            "warm_step_ms_median": round(med_ms, 2),
            "warm_positions_per_sec": round(pos_per_s, 1),
            "loss_init": round(losses[0], 5),
            "loss_end": round(losses[-1], 5),
            "all_finite": bool(all(np.isfinite(x) for x in losses)),
            "device_hbm_mib": hbm,
            "stream_b1_p50_ms": round(float(np.percentile(tick_ms, 50)), 3),
            "stream_b1_p95_ms": round(float(np.percentile(tick_ms, 95)), 3),
            "status": "ok",
        }
    except Exception as e:
        res = {
            "profile": profile,
            "backbone": backbone,
            "use_wy_chunking": bool(use_wy_chunking),
            "surprise_chunk_size": int(surprise_chunk_size),
            "bf16_2d_params": bool(use_bf16_params_opt),
            "trainable_core_params": int(n_params),
            "core_state_mib_b1": round(st_bytes["total_core_state_bytes"] / (1024.0**2), 4),
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:240]}",
        }
    finally:
        params = opt_st = pkts = targs = tvalid = st0 = st1 = _train_step = _online_tick = None
        gc.collect()
        jax.clear_caches()
    print(f"[BENCH] {profile} {backbone} wy={use_wy_chunking} B={batch_size} T={seq_len} -> {res.get('status')} pos/s={res.get('warm_positions_per_sec')} p95={res.get('stream_b1_p95_ms')}ms peakHBM={res.get('device_hbm_mib', {}).get('peak_bytes_in_use_mib')}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SLWM-v3 hoisted bf16 MXU core on TPU v6e-1 or NVIDIA H100.")
    ap.add_argument("--suite", type=str, default="full", choices=("smoke_cpu", "mid", "wy_routed", "full"))
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    dev = jax.devices()[0]
    if args.suite == "smoke_cpu":
        cases = [
            ("tiny_unit", "slwm_v3_routed", 2, 32, False, 1, False),
            ("tiny_unit", "slwm_v3_routed", 2, 32, True, 8, False),
            ("tiny_unit", "single_bank_gated_deltanet", 2, 32, True, 1, False),
        ]
    elif args.suite == "wy_routed":
        # Exact SAME-BACKBONE (`slwm_v3_routed`) fp32-master-weight speedup measurement (`C=1 vs C=16 vs C=32`)!
        cases = [
            ("discovery_45m", "slwm_v3_routed", 16, 512, False, 1, False),
            ("discovery_45m", "slwm_v3_routed", 16, 512, True, 16, False),
            ("discovery_45m", "slwm_v3_routed", 16, 512, True, 32, False),
            ("confirmation_135m", "slwm_v3_routed", 8, 512, False, 1, False),
            ("confirmation_135m", "slwm_v3_routed", 8, 512, True, 16, False),
            ("confirmation_135m", "slwm_v3_routed", 8, 512, True, 32, False),
            ("scale_700m", "slwm_v3_routed", 4, 512, False, 1, False),
            ("scale_700m", "slwm_v3_routed", 4, 512, True, 16, False),
            ("scale_700m", "slwm_v3_routed", 4, 512, True, 32, False),
            ("scale_1b", "slwm_v3_routed", 2, 512, False, 1, False),
            ("scale_1b", "slwm_v3_routed", 2, 512, True, 16, False),
            ("scale_1b", "slwm_v3_routed", 2, 512, True, 32, False),
        ]
    elif args.suite == "mid":
        cases = [
            ("discovery_45m", "slwm_v3_routed", 16, 512, False, 1, False),
            ("discovery_45m", "slwm_v3_routed", 16, 512, True, 16, False),
            ("discovery_45m", "single_bank_gated_deltanet", 16, 512, True, 1, False),
            ("discovery_45m", "sliding_window_gqa", 16, 512, False, 1, False),
            ("confirmation_135m", "slwm_v3_routed", 8, 512, False, 1, False),
            ("confirmation_135m", "slwm_v3_routed", 8, 512, True, 16, False),
            ("confirmation_135m", "single_bank_gated_deltanet", 8, 512, True, 1, False),
            ("confirmation_135m", "sliding_window_gqa", 8, 512, False, 1, False),
        ]
    else:
        cases = [
            ("discovery_45m", "slwm_v3_routed", 16, 512, False, 1, False),
            ("discovery_45m", "slwm_v3_routed", 16, 512, True, 16, False),
            ("discovery_45m", "single_bank_gated_deltanet", 16, 512, True, 1, False),
            ("discovery_45m", "sliding_window_gqa", 16, 512, False, 1, False),
            ("confirmation_135m", "slwm_v3_routed", 8, 512, False, 1, False),
            ("confirmation_135m", "slwm_v3_routed", 8, 512, True, 16, False),
            ("confirmation_135m", "single_bank_gated_deltanet", 8, 512, True, 1, False),
            ("confirmation_135m", "sliding_window_gqa", 8, 512, False, 1, False),
            ("scale_700m", "slwm_v3_routed", 4, 512, False, 1, False),
            ("scale_700m", "slwm_v3_routed", 4, 512, True, 16, False),
            ("scale_700m", "slwm_v3_routed", 4, 512, True, 32, False),
            ("scale_700m", "single_bank_gated_deltanet", 4, 512, True, 1, False),
            ("scale_1b", "slwm_v3_routed", 2, 512, False, 1, False),
            ("scale_1b", "slwm_v3_routed", 2, 512, True, 16, False),
            ("scale_1b", "slwm_v3_routed", 2, 512, True, 32, False),
        ]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for prof, bb, bsz, tlen, wy, sc, bf16p in cases:
        r = benchmark_config(
            prof,
            bb,
            batch_size=bsz,
            seq_len=tlen,
            use_wy_chunking=wy,
            surprise_chunk_size=sc,
            use_bf16_params_opt=bf16p,
            n_warm_steps=3 if args.suite == "smoke_cpu" else 8,
            n_stream_ticks=8 if args.suite == "smoke_cpu" else 32,
        )
        results.append(r)
        report = {
            "jax_version": jax.__version__,
            "device_platform": str(dev.platform),
            "device_kind": str(dev.device_kind),
            "device_count": len(jax.devices()),
            "suite": args.suite,
            "results": results,
        }
        args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[BENCH] Saved -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
