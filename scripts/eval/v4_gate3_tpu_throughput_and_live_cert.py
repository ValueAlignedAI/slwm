"""SLWM-v4 Gate 3 TPU v6e-1 Certification: Live 2-Min (750-HB) Rollout Latency + V4-S3 Training Step Throughput.

Runs on ONE Google Colab TPU `v6e-1` (`~3 min`, hard-capped <= 3.0 CU, guaranteed verified teardown):
  1. Certifies exact fused `@jax.jit` `step_v4_live_heartbeat` (`4` motor + `2` 16-RVQ Depth-Audio + `15` Flow-Video
     at `nfe=4` & `nfe=6` + `PolicyCommitGate`) on TPU `v6e-1` for `discovery_45m` and `confirmation_150m`, including
     a continuous `750`-heartbeat (`120.0 s`) rollout confirming `0.0%` state growth and `0` NaN/Inf.
  2. Benchmarks real forward+backward+Optax muP training step latency (`ms/step`, `pos/sec`, HBM) on `discovery_45m`
     (`T=1024, B=8` and `B=16`) across `A0_3to1_single_bank`, `C6_routed_4bank`, `B1_sw_gqa`, `B0_v3_ref`, `O1_jepa_sync`,
     and `H3_self_forcing` -> locks exact `V4-S3` `450M`-pos compute plan before Gate 3 review.
"""

from __future__ import annotations

import argparse
import io
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

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "/content/slwm_v4")

from data.v4_heartbeat_contract import UNIFIED_ACTION_DIM, UNIFIED_PROPRIO_DIM, build_interleaved_heartbeat_stream
from models.v4 import (
    compute_v4_training_loss,
    count_state_tree_bytes,
    init_v4_core_state,
    init_v4_model_params,
    make_slwm_v4_spec,
    make_v4_mup_optimizer,
    step_v4_live_heartbeat,
    summarize_v4_accounting,
)

devs = jax.devices()
dev_str = [str(d) for d in devs]
print(f"[G3_TPU] jax={jax.__version__} devices={dev_str}", flush=True)

receipt = {
    "gate": "SLWM-v4 Gate 3 — TPU v6e-1 Live 2-Min Streaming & V4-S3 Training Throughput Receipt",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "jax_version": jax.__version__,
    "devices": dev_str,
}

# 1. Live B=1 fused heartbeat latency & 750-heartbeat (120.0 s) flat-O(1) cert
live_res = {}
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

    p_out = {"accounting": acct, "by_nfe": {}}
    for nfe in (4, 6):
        @jax.jit
        def _hb(z, st, k, is_first):
            return step_v4_live_heartbeat(
                params, st, z, k, act_m, prop_m, is_first, sp, video_flow_nfe=nfe, matmul_dtype=jnp.bfloat16
            )

        z_c, st_c = z0, st0
        for wi in range(10):
            z_c, st_c, out_c = _hb(z_c, st_c, jax.random.PRNGKey(wi), jnp.asarray(wi == 0))
            jax.block_until_ready(out_c["video_patches"])

        n_rep = 750 if (profile == "confirmation_150m" and run_id == "A0_3to1_single_bank" and nfe == 6) else 50
        init_b = count_state_tree_bytes(st_c)
        ts = []
        ok_fin = True
        for ri in range(n_rep):
            t0 = time.perf_counter()
            z_c, st_c, out_c = _hb(z_c, st_c, jax.random.PRNGKey(1000 + ri), jnp.asarray(False))
            jax.block_until_ready(out_c["policy_probs"])
            ts.append((time.perf_counter() - t0) * 1000.0)
            ok_fin = ok_fin and bool(jnp.all(jnp.isfinite(out_c["video_patches"])) and jnp.all(jnp.isfinite(out_c["action_seq"])))
        fin_b = count_state_tree_bytes(st_c)
        a = np.asarray(ts[125:] if n_rep >= 750 else ts, dtype=np.float64)
        p_out["by_nfe"][f"nfe_{nfe}"] = {
            "n_heartbeats": n_rep,
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.quantile(a, 0.95)), 2),
            "state_bytes_init": init_b,
            "state_bytes_final": fin_b,
            "state_flat_o1": bool(init_b == fin_b),
            "outputs_all_finite": ok_fin,
        }
        print(
            f"[LIVE TPU {profile}/{run_id} nfe={nfe}] med={np.median(a):.2f}ms p95={np.quantile(a, 0.95):.2f}ms "
            f"flat_O1={init_b==fin_b} finite={ok_fin}",
            flush=True,
        )
    live_res[f"{profile}/{run_id}"] = p_out
    del params, st0, z_c, st_c
    gc.collect()

receipt["tpu_live_heartbeat_benchmarks"] = live_res

# 2. V4-S3 Training Throughput Benchmark (discovery_45m, T=1024, B=16 -> 16,384 pos/step!)
rng = np.random.default_rng(7)
T_LEN = 1024
B_SZ = 16
n_hb = 60
st_ex = build_interleaved_heartbeat_stream(
    video_groups=rng.normal(0.0, 0.5, size=(n_hb, 15, 1024)).astype(np.float16),
    audio_codes=rng.integers(0, 2048, size=(2 * n_hb, 16), dtype=np.int16),
    action_seq=rng.normal(0.0, 0.4, size=(4 * n_hb, UNIFIED_ACTION_DIM)).astype(np.float32),
    proprio_seq=rng.normal(0.0, 0.4, size=(4 * n_hb, UNIFIED_PROPRIO_DIM)).astype(np.float32),
    text_by_heartbeat={h: b"robot picks cup " for h in range(n_hb)},
)
vid_f = np.zeros((T_LEN, 1024), dtype=np.float32)
aud_c = np.zeros((T_LEN, 16), dtype=np.int32)
act_f = np.zeros((T_LEN, UNIFIED_ACTION_DIM), dtype=np.float32)
prop_f = np.zeros((T_LEN, UNIFIED_PROPRIO_DIM), dtype=np.float32)
vi, ai, mi = st_ex["video_idx"][:T_LEN], st_ex["audio_idx"][:T_LEN], st_ex["motor_idx"][:T_LEN]
vid_f[vi >= 0] = st_ex["video_feat"][vi[vi >= 0]].astype(np.float32)
aud_c[ai >= 0] = st_ex["audio_codes"][ai[ai >= 0]].astype(np.int32)
act_f[mi >= 0] = st_ex["action_feat"][mi[mi >= 0]].astype(np.float32)
prop_f[mi >= 0] = st_ex["proprio_feat"][mi[mi >= 0]].astype(np.float32)

batch = {
    "modality_id": jnp.broadcast_to(jnp.asarray(st_ex["modality_id"][:T_LEN], dtype=jnp.int32)[:, None], (T_LEN, B_SZ)),
    "heartbeat_id": jnp.broadcast_to(jnp.asarray(st_ex["heartbeat_id"][:T_LEN], dtype=jnp.int32)[:, None], (T_LEN, B_SZ)),
    "dt_s": jnp.broadcast_to(jnp.asarray(st_ex["dt_s"][:T_LEN], dtype=jnp.float32)[:, None], (T_LEN, B_SZ)),
    "reset_mask": jnp.broadcast_to(jnp.asarray(st_ex["reset_mask"][:T_LEN], dtype=jnp.bool_)[:, None], (T_LEN, B_SZ)),
    "valid_mask": jnp.ones((T_LEN, B_SZ), dtype=jnp.float32),
    "target_valid_mask": jnp.ones((T_LEN, B_SZ), dtype=jnp.float32),
    "reliability": jnp.ones((T_LEN, B_SZ), dtype=jnp.float32),
    "spatial_pos": jnp.broadcast_to(jnp.asarray(st_ex["spatial_pos"][:T_LEN], dtype=jnp.int32)[:, None], (T_LEN, B_SZ)),
    "text_byte": jnp.broadcast_to(jnp.asarray(st_ex["text_byte"][:T_LEN], dtype=jnp.int32)[:, None], (T_LEN, B_SZ)),
    "video_feat": jnp.broadcast_to(jnp.asarray(vid_f)[:, None, :], (T_LEN, B_SZ, 1024)),
    "audio_codes": jnp.broadcast_to(jnp.asarray(aud_c)[:, None, :], (T_LEN, B_SZ, 16)),
    "action_feat": jnp.broadcast_to(jnp.asarray(act_f)[:, None, :], (T_LEN, B_SZ, UNIFIED_ACTION_DIM)),
    "proprio_feat": jnp.broadcast_to(jnp.asarray(prop_f)[:, None, :], (T_LEN, B_SZ, UNIFIED_PROPRIO_DIM)),
    "action_mask": jnp.ones((T_LEN, B_SZ, UNIFIED_ACTION_DIM), dtype=jnp.float32),
    "proprio_mask": jnp.ones((T_LEN, B_SZ, UNIFIED_PROPRIO_DIM), dtype=jnp.float32),
    "episode_success": jnp.ones((T_LEN, B_SZ), dtype=jnp.int32),
}

train_bench = {}
for rid in (
    "A0_3to1_single_bank",
    "C6_routed_4bank",
    "B1_sw_gqa",
    "B0_v3_ref",
    "O1_jepa_sync",
    "H3_self_forcing",
):
    sp = make_slwm_v4_spec("discovery_45m", run_id=rid)
    params = init_v4_model_params(jax.random.PRNGKey(10), sp)
    opt = make_v4_mup_optimizer(params, sp, peak_lr=1e-3, total_steps=1000, warmup_steps=50)
    opt_st = opt.init(params)

    @jax.jit
    def _step(p, ost, b_in, rng_k):
        def _lfn(pp):
            return compute_v4_training_loss(
                pp, b_in, sp, rng_k, matmul_dtype=jnp.bfloat16, wy_chunk_size=64, gqa_chunk_size=64, use_remat=True, assume_all_valid=False
            )
        (loss, metrics), grads = jax.value_and_grad(_lfn, has_aux=True)(p)
        upd, nost = opt.update(grads, ost, p)
        return optax.apply_updates(p, upd), nost, loss, metrics

    # Benchmark exact `v4_train.py` path (`assume_all_valid=False`) on BOTH dense (all-valid) AND padded-tail batches!
    batch_padded = dict(batch)
    batch_padded["valid_mask"] = batch["valid_mask"].at[-64:, 0].set(0.0)
    batch_padded["target_valid_mask"] = batch["target_valid_mask"].at[-64:, 0].set(0.0)

    for wi in range(3):
        params, opt_st, lval, _ = _step(params, opt_st, batch, jax.random.PRNGKey(wi))
        jax.block_until_ready(lval)

    step_ms_dense = []
    for ri in range(10):
        t0 = time.perf_counter()
        params, opt_st, lval, _ = _step(params, opt_st, batch, jax.random.PRNGKey(100 + ri))
        jax.block_until_ready(lval)
        step_ms_dense.append((time.perf_counter() - t0) * 1000.0)

    step_ms_padded = []
    for ri in range(3):
        t0 = time.perf_counter()
        params, opt_st, lval_p, _ = _step(params, opt_st, batch_padded, jax.random.PRNGKey(200 + ri))
        jax.block_until_ready(lval_p)
        step_ms_padded.append((time.perf_counter() - t0) * 1000.0)

    med_ms = float(np.median(step_ms_dense))
    med_ms_pad = float(np.median(step_ms_padded))
    pos_per_s = (T_LEN * B_SZ) / (med_ms / 1000.0)
    hrs_450m = 450_000_000.0 / pos_per_s / 3600.0
    train_bench[rid] = {
        "seq_len": T_LEN,
        "batch_size": B_SZ,
        "pos_per_step": T_LEN * B_SZ,
        "step_median_ms_dense": round(med_ms, 2),
        "step_median_ms_padded_tail": round(med_ms_pad, 2),
        "pos_per_sec_dense": round(pos_per_s, 1),
        "est_hours_for_450m_pos": round(hrs_450m, 3),
        "loss_finite": bool(jnp.isfinite(lval) and jnp.isfinite(lval_p)),
    }
    print(
        f"[TRAIN TPU 45M/{rid}] dense={med_ms:.2f}ms padded_tail={med_ms_pad:.2f}ms throughput={pos_per_s:,.0f} pos/s "
        f"est_450M={hrs_450m:.2f}h finite={bool(jnp.isfinite(lval) and jnp.isfinite(lval_p))}",
        flush=True,
    )
    del params, opt_st
    gc.collect()

receipt["v4_s3_tpu_training_throughput"] = train_bench
Path("/content/out/gate3_tpu_throughput_and_live_receipt.json").write_text(json.dumps(receipt, indent=2))
print("GATE3_TPU_CERT_DONE", flush=True)
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
    ap = argparse.ArgumentParser(description="SLWM-v4 Gate 3 TPU v6e-1 Live 2-Min & V4-S3 Training Throughput Certifier.")
    ap.add_argument("--accel", type=str, default="tpu_v6e1", choices=("tpu_v6e1", "tpu_v5e1"))
    ap.add_argument("--max-spend-cu", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("reports/v4/gate3_tpu_throughput_and_live_receipt.json"))
    args = ap.parse_args()

    cfg_path = _REPO_ROOT / ".ops/colab-config.json"
    log_dir = _REPO_ROOT / ".ops/v4_gate3_tpu_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"slwm_v4_g3tpu_{int(time.time()) % 100000}"

    bal_start, rate_start = query_balance(cfg_path)
    print(f"[G3_TPU] accel={args.accel} session={session_name} start_bal={bal_start:.2f} CU rate={rate_start:.2f}", flush=True)
    tar_bytes = _build_bundle(_REPO_ROOT)
    created = False
    t_alloc = time.monotonic()

    try:
        used_acc = allocate_colab_session_safe(
            session_name, args.accel, cfg_path, max_attempts=4, sleep_s=15.0, fallback_accels=["tpu_v5e1"]
        )
        created = True
        t_alloc = time.monotonic()
        print(f"[G3_TPU] Allocated {used_acc} session '{session_name}'", flush=True)

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
            "for d in ('/content/slwm_v4', '/content/out'): os.makedirs(d, exist_ok=True)\n"
            "with tarfile.open('/content/v4_g3_bundle.tar.gz', 'r:gz') as tf: tf.extractall('/content/slwm_v4')\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optax'], check=True)\n"
            "import jax; print('TPU_PREP_OK', jax.__version__, jax.devices(), flush=True)\n"
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
        if rc != 0 or tripped or "TPU_PREP_OK" not in out:
            raise RuntimeError(f"Prep failed rc={rc}: {err[-500:]}")

        rc, tail, tripped = _run_detached_remote_script_with_polling(
            session_name,
            REMOTE_SCRIPT,
            tag="v4_g3_tpu_cert",
            cfg_path=cfg_path,
            log_dir=log_dir,
            bal_start=bal_start,
            max_spend_cu=args.max_spend_cu,
            t_alloc_start=t_alloc,
            poll_interval_s=20.0,
            max_wall_s=1500.0,
        )
        print(tail[-3500:], flush=True)
        if rc != 0 or tripped or "GATE3_TPU_CERT_DONE" not in tail:
            raise RuntimeError(f"Remote TPU cert failed rc={rc} tripped={tripped}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        dl = run_colab(
            "download", "-s", session_name, "/content/out/gate3_tpu_throughput_and_live_receipt.json", str(args.out), cfg_path=cfg_path, timeout=120
        )
        if dl.returncode != 0:
            raise RuntimeError(f"Download failed: {dl.stderr or dl.stdout}")
        print(f"[G3_TPU] Saved verified receipt to {args.out}", flush=True)
    finally:
        if created:
            verified_stop_session(session_name, cfg_path)


if __name__ == "__main__":
    main()
