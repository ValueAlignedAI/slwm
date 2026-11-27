"""SLWM-v4 Gate 3 Advisory A1 Closure: True Steady-State `t > 20 s` (`KV=249` Saturated) & 2-Min Flat-HBM Live Certifier.

Runs on a single NVIDIA A100 / Blackwell G4 GPU (`~4 min`, hard-capped <= 4.0 CU, guaranteed verified teardown):
  1. Streams `750` continuous `160 ms` heartbeats (`120.0 s` = `2.0 minutes`) through frozen `Wan 2.1 Causal VAE`
     (`bf16`, `192x320`, 4-frame causal encode + decode with persistent `feat_cache`) AND `Kyutai Mimi 16-RVQ`
     (`use_streaming=True` with persistent `encoder_cache` / `decoder_cache` / `padding_cache` saturated at `KV = 249`
     for `t > 20 s`), logging exact steady-state `t > 20 s` latency and HBM trace at `t = 10, 20, 40, 60, 90, 120 s`.
  2. Benchmarks fused `@jax.jit` `step_v4_live_heartbeat` (`4` motor ticks + `2` `16-RVQ` Depth-Transformer AR ticks +
     `15` Rectified Flow Matching ODE video patches at `nfe=4` & `nfe=6` + `PolicyCommitGate`) on the SAME GPU
     (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) for BOTH `discovery_45m` and `confirmation_150m` (`A0` & `C6`), including
     a full `750`-heartbeat (`120.0 s`) continuous rollout certifying `0.0%` state growth and `0` NaN/Inf.
  3. Produces `reports/v4/gate3_steady_state_latency_receipt.json` closing Gate 2 Peer Review Advisory A1 before Gate 3.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train.v3_colab_any2any_launcher import _run_detached_remote_script_with_polling
from scripts.train.v3_colab_pilot_launcher import (
    _run_remote_kernel_python_with_watchdog,
    allocate_colab_session_safe,
    query_balance,
    run_colab,
    verified_stop_session,
)

REMOTE_SCRIPT = r'''
import gc, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from transformers import MimiModel

sys.path.insert(0, "/content/slwm_v4")

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
gpu_name = torch.cuda.get_device_name(0)
gpu_vram_gib = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
print(f"[G3_LAT] GPU={gpu_name} VRAM={gpu_vram_gib} GiB torch={torch.__version__}", flush=True)

receipt = {
    "gate": "SLWM-v4 Gate 3 — Steady-State (t > 20s, KV=249 Saturated) & 2-Minute Continuous Live Latency/HBM Receipt",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_name": gpu_name,
    "gpu_total_vram_gib": gpu_vram_gib,
    "torch_version": torch.__version__,
}

# ---------------------------------------------------------------------------
# PART 1: 750 Continuous Heartbeats (120.0 s) Frozen Bidirectional Codecs
# ---------------------------------------------------------------------------
print("[PART 1] Loading Wan 2.1 Causal VAE (bf16) + Kyutai Mimi 16-RVQ (bf16 & fp32)...", flush=True)
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
).eval().to(device)
mimi_bf16 = MimiModel.from_pretrained("kyutai/mimi", torch_dtype=torch.bfloat16).eval().to(device)

def _extract_kv_info(cache_obj):
    if cache_obj is None:
        return {"physical_kv_len": 0, "cumulative_seq_len": 0}
    cum_len = -1
    if hasattr(cache_obj, "get_seq_length"):
        try:
            cum_len = int(cache_obj.get_seq_length())
        except Exception:
            pass
    phys_len = -1
    if hasattr(cache_obj, "layers") and cache_obj.layers:
        k0 = getattr(cache_obj.layers[0], "keys", None)
        if k0 is not None:
            phys_len = int(k0.shape[-2])
    elif hasattr(cache_obj, "key_cache") and cache_obj.key_cache:
        phys_len = int(cache_obj.key_cache[0].shape[-2])
    return {"physical_kv_len": phys_len, "cumulative_seq_len": cum_len}

def _run_codec_120s(mimi_model, mimi_dtype_str):
    x_f0 = torch.randn(1, 3, 1, 192, 320, device=device, dtype=torch.bfloat16)
    x_4f = torch.randn(1, 3, 4, 192, 320, device=device, dtype=torch.bfloat16)
    z_1g = torch.randn(1, 16, 1, 24, 40, device=device, dtype=torch.bfloat16)
    m_dtype = torch.bfloat16 if mimi_dtype_str == "bf16" else torch.float32
    pcm_160ms = torch.randn(1, 1, 3840, device=device, dtype=m_dtype)
    codes_160ms = torch.randint(0, 2048, (1, 16, 2), device=device, dtype=torch.long)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    wan_enc_ms, wan_dec_ms, mimi_enc_ms, mimi_dec_ms = [], [], [], []
    hbm_trace = {}
    kv_trace = {}
    mimi_streaming_api_used = True
    mimi_streaming_fallback_reason = None

    with torch.no_grad():
        # Wan 2.1 Frame 0 causal bootstrap
        vae.clear_cache()
        vae._enc_feat_map = [None] * vae._cached_conv_counts["encoder"]
        vae._enc_conv_idx = [0]
        _ = vae.quant_conv(vae.encoder(x_f0, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))
        vae._feat_map = [None] * vae._cached_conv_counts["decoder"]
        vae._conv_idx = [0]
        _ = vae.decoder(vae.post_quant_conv(z_1g), feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True)

        # Verified transformers Mimi streaming API (`encoder_cache` + `padding_cache` + `use_streaming=True` on encode;
        # `decoder_cache` + `use_cache=True` on decode; physical KV tensor saturates at exact `249` by t=10s!)
        enc_cache, enc_pad, dec_cache = None, None, None
        eo0 = mimi_model.encode(pcm_160ms, num_quantizers=16, use_streaming=True)
        enc_cache = getattr(eo0, "encoder_cache", getattr(eo0, "encoder_past_key_values", None))
        enc_pad = getattr(eo0, "padding_cache", None)
        do0 = mimi_model.decode(codes_160ms, use_cache=True)
        dec_cache = getattr(do0, "decoder_cache", getattr(do0, "decoder_past_key_values", None))

        # Run FULL 750 continuous 160 ms heartbeats (120.0 seconds = 2.0 minutes!)
        check_hbs = {62: "10s", 125: "20s", 250: "40s", 375: "60s", 562: "90s", 749: "120s"}
        for h in range(750):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            vae._enc_conv_idx = [0]
            _ = vae.quant_conv(vae.encoder(x_4f, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            vae._conv_idx = [0]
            _ = vae.decoder(vae.post_quant_conv(z_1g), feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=False)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            eo = mimi_model.encode(
                pcm_160ms, num_quantizers=16, encoder_cache=enc_cache, padding_cache=enc_pad, use_streaming=True
            )
            enc_cache = getattr(eo, "encoder_cache", getattr(eo, "encoder_past_key_values", None))
            enc_pad = getattr(eo, "padding_cache", None)
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            do = mimi_model.decode(codes_160ms, decoder_cache=dec_cache, use_cache=True)
            dec_cache = getattr(do, "decoder_cache", getattr(do, "decoder_past_key_values", None))
            torch.cuda.synchronize()
            t4 = time.perf_counter()

            wan_enc_ms.append((t1 - t0) * 1000.0)
            wan_dec_ms.append((t2 - t1) * 1000.0)
            mimi_enc_ms.append((t3 - t2) * 1000.0)
            mimi_dec_ms.append((t4 - t3) * 1000.0)

            if h in check_hbs:
                lbl = check_hbs[h]
                hbm_mib = round(torch.cuda.memory_allocated() / (1024**2), 2)
                hbm_trace[lbl] = hbm_mib
                kv_trace[lbl] = {"enc": _extract_kv_info(enc_cache), "dec": _extract_kv_info(dec_cache)}
                print(
                    f"[CODEC {mimi_dtype_str} t={lbl}] wan_e={wan_enc_ms[-1]:.2f}ms wan_d={wan_dec_ms[-1]:.2f}ms "
                    f"mimi_e={mimi_enc_ms[-1]:.2f}ms mimi_d={mimi_dec_ms[-1]:.2f}ms hbm={hbm_mib:.1f}MiB kv={kv_trace[lbl]}",
                    flush=True,
                )

        vae.clear_cache()

    # Strictly report steady-state t > 20 s (h >= 125, i.e., 625 continuous heartbeats after KV=249 saturation!)
    sl = slice(125, 750)
    def _summ(arr):
        a = np.asarray(arr[sl], dtype=np.float64)
        return {
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.quantile(a, 0.95)), 2),
            "p99_ms": round(float(np.quantile(a, 0.99)), 2),
            "mean_ms": round(float(np.mean(a)), 2),
        }

    we_s = _summ(wan_enc_ms)
    wd_s = _summ(wan_dec_ms)
    me_s = _summ(mimi_enc_ms)
    md_s = _summ(mimi_dec_ms)
    tot_bidir = np.asarray(wan_enc_ms[sl]) + np.asarray(wan_dec_ms[sl]) + np.asarray(mimi_enc_ms[sl]) + np.asarray(mimi_dec_ms[sl])
    tot_gen = np.asarray(wan_dec_ms[sl]) + np.asarray(mimi_dec_ms[sl])
    hbm_drift_20s_to_120s = round(abs(hbm_trace["120s"] - hbm_trace["20s"]), 3)
    kv_249_verified = bool(
        all(
            kv_trace[k]["enc"]["physical_kv_len"] == 249 and kv_trace[k]["dec"]["physical_kv_len"] == 249
            for k in ("20s", "40s", "60s", "90s", "120s")
        )
    )
    return {
        "mimi_dtype": mimi_dtype_str,
        "mimi_streaming_api_used": mimi_streaming_api_used,
        "mimi_streaming_fallback_reason": mimi_streaming_fallback_reason,
        "n_heartbeats_total": 750,
        "steady_state_window_heartbeats": "125..749 (t=20.0s..120.0s, N=625)",
        "wan21_enc_4f_bf16": we_s,
        "wan21_dec_4f_bf16": wd_s,
        "mimi_enc_2tick_kv_sat": me_s,
        "mimi_dec_2tick_kv_sat": md_s,
        "total_bidirectional_codecs_steady_ms": {
            "median_ms": round(float(np.median(tot_bidir)), 2),
            "p95_ms": round(float(np.quantile(tot_bidir, 0.95)), 2),
        },
        "total_decode_only_codecs_steady_ms": {
            "median_ms": round(float(np.median(tot_gen)), 2),
            "p95_ms": round(float(np.quantile(tot_gen, 0.95)), 2),
        },
        "hbm_allocated_mib_trace": hbm_trace,
        "hbm_drift_20s_to_120s_mib": hbm_drift_20s_to_120s,
        "hbm_flat_o1_verified": bool(hbm_drift_20s_to_120s <= 2.0),
        "mimi_kv_249_saturated_20s_to_120s": kv_249_verified,
        "peak_codec_hbm_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "mimi_kv_len_trace": kv_trace,
    }

codec_bf16 = _run_codec_120s(mimi_bf16, "bf16")
receipt["steady_state_codecs_120s_bf16"] = codec_bf16

# Also measure fp32 Mimi (Gate 2 showed fp32 Mimi encode can be faster & cleaner than bf16!)
del mimi_bf16
gc.collect()
torch.cuda.empty_cache()
mimi_fp32 = MimiModel.from_pretrained("kyutai/mimi", torch_dtype=torch.float32).eval().to(device)
codec_fp32 = _run_codec_120s(mimi_fp32, "fp32")
receipt["steady_state_codecs_120s_fp32"] = codec_fp32

best_codec = codec_fp32 if codec_fp32["total_bidirectional_codecs_steady_ms"]["median_ms"] <= codec_bf16["total_bidirectional_codecs_steady_ms"]["median_ms"] else codec_bf16
receipt["selected_best_codec_profile"] = best_codec["mimi_dtype"]

# ---------------------------------------------------------------------------
# PART 2: Co-Resident Fused `@jax.jit` SLWM-v4 Live Heartbeat Latency, Coupled Timing & 2-Min Rollout
# ---------------------------------------------------------------------------
# Keep Wan 2.1 + Mimi co-resident in GPU VRAM (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) to measure
# BOTH true co-resident coupled end-to-end latency (`p95(A+B)`) AND isolated per-component timing!
import jax
import jax.numpy as jnp
from models.v4 import (
    count_state_tree_bytes,
    init_v4_core_state,
    init_v4_model_params,
    make_slwm_v4_spec,
    step_v4_live_heartbeat,
    summarize_v4_accounting,
)

jax_devs = [str(d) for d in jax.devices()]
print(f"[PART 2] JAX version={jax.__version__} devices={jax_devs}", flush=True)
receipt["jax_version"] = jax.__version__
receipt["jax_devices"] = jax_devs

def _jax_hbm_mib():
    try:
        ms = jax.devices()[0].memory_stats()
        if ms:
            return round(float(ms.get("bytes_in_use", 0)) / (1024**2), 2)
    except Exception:
        pass
    return -1.0

# Pre-warm co-resident streaming codecs past KV=249 saturation (130 heartbeats = 20.8 s)
x_f0_c = torch.randn(1, 3, 1, 192, 320, device=device, dtype=torch.bfloat16)
x_4f_c = torch.randn(1, 3, 4, 192, 320, device=device, dtype=torch.bfloat16)
z_1g_c = torch.randn(1, 16, 1, 24, 40, device=device, dtype=torch.bfloat16)
pcm_c = torch.randn(1, 1, 3840, device=device, dtype=torch.float32)
codes_c = torch.randint(0, 2048, (1, 16, 2), device=device, dtype=torch.long)
with torch.no_grad():
    vae.clear_cache()
    vae._enc_feat_map = [None] * vae._cached_conv_counts["encoder"]
    vae._enc_conv_idx = [0]
    _ = vae.quant_conv(vae.encoder(x_f0_c, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))
    vae._feat_map = [None] * vae._cached_conv_counts["decoder"]
    vae._conv_idx = [0]
    _ = vae.decoder(vae.post_quant_conv(z_1g_c), feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True)
    ec_c, ep_c, dc_c = None, None, None
    for _ in range(130):
        eo = mimi_fp32.encode(pcm_c, num_quantizers=16, encoder_cache=ec_c, padding_cache=ep_c, use_streaming=True)
        ec_c = getattr(eo, "encoder_cache", getattr(eo, "encoder_past_key_values", None))
        ep_c = getattr(eo, "padding_cache", None)
        do = mimi_fp32.decode(codes_c, decoder_cache=dc_c, use_cache=True)
        dc_c = getattr(do, "decoder_cache", getattr(do, "decoder_past_key_values", None))

slwm_benchmarks = {}
for profile, run_id in (
    ("discovery_45m", "A0_3to1_single_bank"),
    ("discovery_45m", "C6_routed_4bank"),
    ("confirmation_150m", "A0_3to1_single_bank"),
    ("confirmation_150m", "C6_routed_4bank"),
):
    sp = make_slwm_v4_spec(profile, run_id=run_id)
    params = init_v4_model_params(jax.random.PRNGKey(0), sp)
    st0 = init_v4_core_state(sp, batch_size=1)
    acct = summarize_v4_accounting(sp, params, st0)

    act_m = jnp.ones((1, sp.action_dim), dtype=jnp.float32)
    prop_m = jnp.ones((1, sp.proprio_dim), dtype=jnp.float32)
    z0 = jnp.zeros((1, sp.d_model), dtype=jnp.float32)

    prof_res = {"accounting": acct, "by_nfe": {}}
    for nfe in (4, 6):
        @jax.jit
        def _hb_step(z, st, key, is_first):
            return step_v4_live_heartbeat(
                params, st, z, key, act_m, prop_m, is_first, sp, video_flow_nfe=nfe, matmul_dtype=jnp.bfloat16
            )

        z_c, st_c = z0, st0
        for wi in range(15):
            z_c, st_c, out_c = _hb_step(z_c, st_c, jax.random.PRNGKey(wi), jnp.asarray(wi == 0))
            jax.block_until_ready(out_c["video_patches"])

        # 1. SLWM-v4 isolated heartbeat timing + full 750-heartbeat (120.0s) continuous rollout on confirmation_150m A0 nfe=6
        n_rep = 750 if (profile == "confirmation_150m" and run_id == "A0_3to1_single_bank" and nfe == 6) else 60
        init_b = count_state_tree_bytes(st_c)
        times_ms = []
        jax_hbm_trace = {}
        all_finite = True
        for ri in range(n_rep):
            t0 = time.perf_counter()
            z_c, st_c, out_c = _hb_step(z_c, st_c, jax.random.PRNGKey(1000 + ri), jnp.asarray(False))
            jax.block_until_ready(out_c["policy_probs"])
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            # Check finiteness on EVERY single heartbeat!
            fin_ok = bool(jnp.all(jnp.isfinite(out_c["video_patches"])) and jnp.all(jnp.isfinite(out_c["action_seq"])))
            all_finite = all_finite and fin_ok
            if n_rep >= 750 and ri in (62, 125, 375, 749):
                jax_hbm_trace[f"hb_{ri}"] = _jax_hbm_mib()
        fin_b = count_state_tree_bytes(st_c)

        # 2. True CO-RESIDENT coupled steady-state loop (`Wan Enc -> Mimi Enc -> SLWM-v4 -> Wan Dec -> Mimi Dec`) at KV=249!
        coupled_ms = []
        with torch.no_grad():
            for ci in range(30):
                torch.cuda.synchronize()
                tc0 = time.perf_counter()
                vae._enc_conv_idx = [0]
                _ = vae.quant_conv(vae.encoder(x_4f_c, feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))
                eo = mimi_fp32.encode(pcm_c, num_quantizers=16, encoder_cache=ec_c, padding_cache=ep_c, use_streaming=True)
                ec_c = getattr(eo, "encoder_cache", getattr(eo, "encoder_past_key_values", None))
                ep_c = getattr(eo, "padding_cache", None)
                torch.cuda.synchronize()
                z_c, st_c, out_c = _hb_step(z_c, st_c, jax.random.PRNGKey(5000 + ci), jnp.asarray(False))
                vp_np = np.asarray(out_c["video_patches"])  # Real JAX->host->PyTorch handoff!
                ac_np = np.asarray(out_c["audio_codes"])
                z_dec_in = torch.from_numpy(vp_np.reshape(1, 3, 5, 16, 8, 8).transpose(0, 3, 1, 4, 2, 5).reshape(1, 16, 1, 24, 40)).to(device=device, dtype=torch.bfloat16)
                a_dec_in = torch.from_numpy(ac_np.transpose(0, 2, 1).astype(np.int64)).to(device=device)
                vae._conv_idx = [0]
                _ = vae.decoder(vae.post_quant_conv(z_dec_in), feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=False)
                do = mimi_fp32.decode(a_dec_in, decoder_cache=dc_c, use_cache=True)
                dc_c = getattr(do, "decoder_cache", getattr(do, "decoder_past_key_values", None))
                torch.cuda.synchronize()
                coupled_ms.append((time.perf_counter() - tc0) * 1000.0)

        steady_sl = times_ms[125:] if n_rep >= 750 else times_ms
        a_ms = np.asarray(steady_sl, dtype=np.float64)
        c_ms = np.asarray(coupled_ms[5:], dtype=np.float64)  # drop first 5 warm coupled steps
        med_ms = round(float(np.median(a_ms)), 2)
        p95_ms = round(float(np.quantile(a_ms, 0.95)), 2)
        coupled_med = round(float(np.median(c_ms)), 2)
        coupled_p95 = round(float(np.quantile(c_ms, 0.95)), 2)

        uncoupled_bidir_med = round(best_codec["total_bidirectional_codecs_steady_ms"]["median_ms"] + med_ms, 2)
        uncoupled_bidir_p95_bound = round(best_codec["total_bidirectional_codecs_steady_ms"]["p95_ms"] + p95_ms, 2)
        gen_tot_med = round(best_codec["total_decode_only_codecs_steady_ms"]["median_ms"] + med_ms, 2)

        prof_res["by_nfe"][f"nfe_{nfe}"] = {
            "n_heartbeats_timed": n_rep,
            "slwm_heartbeat_median_ms": med_ms,
            "slwm_heartbeat_p95_ms": p95_ms,
            "coupled_coresident_bidir_median_ms": coupled_med,
            "coupled_coresident_bidir_p95_ms": coupled_p95,
            "uncoupled_additive_bidir_median_ms": uncoupled_bidir_med,
            "uncoupled_additive_bidir_p95_upper_bound_ms": uncoupled_bidir_p95_bound,
            "uncoupled_decode_live_median_ms": gen_tot_med,
            "fits_160ms_coupled_median": bool(coupled_med < 160.0),
            "fits_160ms_coupled_p95": bool(coupled_p95 < 160.0),
            "headroom_coupled_median_ms": round(160.0 - coupled_med, 2),
            "state_bytes_init": init_b,
            "state_bytes_final": fin_b,
            "state_flat_o1": bool(init_b == fin_b),
            "jax_hbm_mib_trace": jax_hbm_trace,
            "outputs_all_finite_every_step": all_finite,
        }
        print(
            f"[SLWM {profile}/{run_id} nfe={nfe}] core+heads={med_ms:.2f}ms | "
            f"COUPLED_BIDIR={coupled_med:.2f}ms (p95={coupled_p95:.2f}ms) / 160ms (headroom={160.0-coupled_med:+.2f}ms) | "
            f"flat_O1={init_b==fin_b} finite_all={all_finite}",
            flush=True,
        )
    slwm_benchmarks[f"{profile}/{run_id}"] = prof_res
    del params, st0, z_c, st_c
    gc.collect()

receipt["slwm_v4_live_heartbeat_benchmarks"] = slwm_benchmarks
c6_45_ratio = (
    slwm_benchmarks["discovery_45m/C6_routed_4bank"]["by_nfe"]["nfe_6"]["slwm_heartbeat_median_ms"]
    / max(slwm_benchmarks["discovery_45m/A0_3to1_single_bank"]["by_nfe"]["nfe_6"]["slwm_heartbeat_median_ms"], 1e-6)
)
c6_150_ratio = (
    slwm_benchmarks["confirmation_150m/C6_routed_4bank"]["by_nfe"]["nfe_6"]["slwm_heartbeat_median_ms"]
    / max(slwm_benchmarks["confirmation_150m/A0_3to1_single_bank"]["by_nfe"]["nfe_6"]["slwm_heartbeat_median_ms"], 1e-6)
)
receipt["c6_vs_a0_latency_ratios_nfe6"] = {"discovery_45m": round(c6_45_ratio, 3), "confirmation_150m": round(c6_150_ratio, 3)}
receipt["advisory_a1_gate3_pass"] = bool(
    best_codec["hbm_flat_o1_verified"]
    and best_codec["mimi_kv_249_saturated_20s_to_120s"]
    and all(
        slwm_benchmarks[k]["by_nfe"][nf]["fits_160ms_coupled_median"]
        and slwm_benchmarks[k]["by_nfe"][nf]["fits_160ms_coupled_p95"]
        and slwm_benchmarks[k]["by_nfe"][nf]["state_flat_o1"]
        and slwm_benchmarks[k]["by_nfe"][nf]["outputs_all_finite_every_step"]
        for k in ("discovery_45m/A0_3to1_single_bank", "discovery_45m/C6_routed_4bank", "confirmation_150m/A0_3to1_single_bank", "confirmation_150m/C6_routed_4bank")
        for nf in ("nfe_4", "nfe_6")
    )
)

Path("/content/out/gate3_steady_state_latency_receipt.json").write_text(json.dumps(receipt, indent=2))
print("GATE3_STEADY_STATE_LATENCY_DONE", flush=True)
'''


def _build_bundle(repo_root: Path) -> bytes:
    buf = io.BytesIO()
    files = (
        "data/__init__.py",
        "data/v4_heartbeat_contract.py",
        "models/__init__.py",
        "models/v4/__init__.py",
        "models/v4/spec.py",
        "models/v4/blocks.py",
        "models/v4/heads.py",
        "models/v4/model.py",
    )
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in files:
            p = repo_root / rel
            data = p.read_bytes() if p.is_file() else b""
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v4 Gate 3 single-GPU steady-state t>20s (KV=249) & 2-min live latency certifier.")
    ap.add_argument("--accel", type=str, default="a100_std", choices=("a100_std", "g4", "a100", "h100", "l4"))
    ap.add_argument("--max-spend-cu", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=Path("reports/v4/gate3_steady_state_latency_receipt.json"))
    args = ap.parse_args()

    cfg_path = _REPO_ROOT / ".ops/colab-config.json"
    log_dir = _REPO_ROOT / ".ops/v4_gate3_latency_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"slwm_v4_g3lat_{int(time.time()) % 100000}"

    bal_start, rate_start = query_balance(cfg_path)
    print(f"[G3_LAT] accel={args.accel} session={session_name} start_bal={bal_start:.2f} CU rate={rate_start:.2f}", flush=True)
    tar_bytes = _build_bundle(_REPO_ROOT)
    created = False
    t_alloc = time.monotonic()

    try:
        used_acc = allocate_colab_session_safe(
            session_name, args.accel, cfg_path, max_attempts=6, sleep_s=10.0, fallback_accels=["a100", "g4", "h100", "l4"]
        )
        created = True
        t_alloc = time.monotonic()
        print(f"[G3_LAT] Allocated {used_acc} session '{session_name}'", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf_tmp:
            tf_tmp.write(tar_bytes)
            tmp_p = Path(tf_tmp.name)
        try:
            up = run_colab("upload", "-s", session_name, str(tmp_p), "/content/v4_g3_bundle.tar.gz", cfg_path=cfg_path, timeout=180)
            if up.returncode != 0:
                raise RuntimeError(f"Upload failed: {up.stderr or up.stdout}")
        finally:
            tmp_p.unlink(missing_ok=True)

        prep_py = (
            "import os, subprocess, sys, tarfile\n"
            "os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'\n"
            "for d in ('/content/slwm_v4', '/content/out'): os.makedirs(d, exist_ok=True)\n"
            "with tarfile.open('/content/v4_g3_bundle.tar.gz', 'r:gz') as tf: tf.extractall('/content/slwm_v4')\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers', 'transformers', 'accelerate', 'optax'], check=True)\n"
            "import torch, jax\n"
            "print('GPU_PREP_OK', torch.cuda.get_device_name(0), 'jax_devs=', jax.devices(), flush=True)\n"
        )
        rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
            session_name,
            prep_py,
            cfg_path=cfg_path,
            log_dir=log_dir,
            log_prefix="00_prep",
            bal_start=bal_start,
            prior_spent_cu=0.0,
            max_spend_cu=args.max_spend_cu,
            t_alloc_start=t_alloc,
            phase_timeout_s=480,
            exec_timeout_s=420,
        )
        print(out, flush=True)
        if rc != 0 or tripped or "GPU_PREP_OK" not in out:
            raise RuntimeError(f"Prep failed rc={rc}: {err[-500:]}")

        rc, tail, tripped = _run_detached_remote_script_with_polling(
            session_name,
            REMOTE_SCRIPT,
            tag="v4_g3_steady_lat",
            cfg_path=cfg_path,
            log_dir=log_dir,
            bal_start=bal_start,
            max_spend_cu=args.max_spend_cu,
            t_alloc_start=t_alloc,
            poll_interval_s=20.0,
            max_wall_s=1500.0,
        )
        print(tail[-3500:], flush=True)
        if rc != 0 or tripped or "GATE3_STEADY_STATE_LATENCY_DONE" not in tail:
            raise RuntimeError(f"Remote latency cert failed rc={rc} tripped={tripped}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        dl = run_colab(
            "download", "-s", session_name, "/content/out/gate3_steady_state_latency_receipt.json", str(args.out), cfg_path=cfg_path, timeout=120
        )
        if dl.returncode != 0:
            raise RuntimeError(f"Download failed: {dl.stderr or dl.stdout}")
        print(f"[G3_LAT] Saved verified receipt to {args.out}", flush=True)
    finally:
        if created:
            verified_stop_session(session_name, cfg_path)


if __name__ == "__main__":
    main()
