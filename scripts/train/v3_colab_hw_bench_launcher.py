"""Dual-Accelerator (`TPU v6e-1` vs `NVIDIA H100`) SLWM-v3 Hardware Benchmark Launcher (`<= 12 CU` cap).

Runs `scripts/train/v3_hardware_benchmark.py --suite full` cleanly inside a Colab Jupyter kernel on
either Google TPU `v6e-1` (`--accel tpu_v6e1`) or NVIDIA `H100` (`--accel h100`), measuring:
  - `discovery_45m`, `confirmation_135m`, `scale_700m`, and `scale_1b` throughput (`pos/s`),
  - Peak device HBM (`MiB`),
  - Exact CU cost per `100M` packet positions (`(rate_cu_hr / 3600) * (1e8 / pos_per_s)`),
  - Batch-1 online streaming `step_core` p50 & p95 latency (`ms/tick`),
  - Exact Surprise-Hoisted `slwm_v3_routed` vs Chunkwise Parallel WY `single_bank_gated_deltanet` vs GQA.

Guarantees verified auto-teardown (`rate_after_stop_cu_hr == 0.0`) on exit or error.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train.v3_colab_pilot_launcher import (
    PILOT_BUNDLE_FILES,
    _adopt_single_unnamed_v6e1,
    _run_remote_kernel_python_with_watchdog,
    query_active_sessions,
    query_balance,
    run_colab,
    verified_stop_session,
)

BENCH_BUNDLE_FILES = tuple(dict.fromkeys((*PILOT_BUNDLE_FILES, "scripts/train/v3_hardware_benchmark.py")))


def _build_bench_tar_bytes(repo_root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in BENCH_BUNDLE_FILES:
            fp = repo_root / rel
            if not fp.is_file():
                raise FileNotFoundError(f"Missing benchmark bundle file: {fp}")
            data = fp.read_bytes()
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SLWM-v3 fast_core on TPU v6e-1 or NVIDIA H100.")
    ap.add_argument("--accel", type=str, required=True, choices=("tpu_v6e1", "h100", "a100"))
    ap.add_argument("--suite", type=str, default="full", choices=("mid", "full"))
    ap.add_argument("--max-spend-cu", type=float, default=12.0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/v3/hw_bench"))
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / f"logs_{args.accel}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = repo_root / ".ops/colab-config.json"

    session_name = f"slwm_v3_bench_{args.accel}"
    bal_start, rate_start = query_balance(cfg_path)
    print(f"[HW_BENCH] accel={args.accel} start_balance={bal_start:.2f} CU rate={rate_start:.2f} CU/hr", flush=True)

    tar_bytes = _build_bench_tar_bytes(repo_root)
    t_alloc = time.monotonic()
    created = False
    bench_json: dict[str, Any] = {}
    live_rate_cu_hr = 0.0

    try:
        assigned_ok = False
        last_err = ""
        for attempt in range(1, 5):
            try:
                if args.accel == "tpu_v6e1":
                    print(f"[HW_BENCH] Creating Google TPU V6E1 session '{session_name}' (attempt {attempt}/4)...", flush=True)
                    res = run_colab("new", "-s", session_name, "--tpu", "V6E1", cfg_path=cfg_path, timeout=360)
                elif args.accel == "a100":
                    print(f"[HW_BENCH] Creating NVIDIA A100 High-RAM session '{session_name}' (attempt {attempt}/4)...", flush=True)
                    res = run_colab("new", "-s", session_name, "--gpu", "A100", "--high-mem", cfg_path=cfg_path, timeout=360)
                else:
                    print(f"[HW_BENCH] Creating NVIDIA H100 session '{session_name}' (attempt {attempt}/4)...", flush=True)
                    res = run_colab("new", "-s", session_name, "--gpu", "H100", cfg_path=cfg_path, timeout=360)
                if res.returncode == 0:
                    print(res.stdout.strip(), flush=True)
                    assigned_ok = True
                    break
                last_err = (res.stderr or res.stdout)[-400:]
                print(f"[HW_BENCH WARN] assign attempt {attempt}/4 rc={res.returncode}: {last_err.splitlines()[-1] if last_err.strip() else ''}", flush=True)
            except Exception as e:
                last_err = str(e)
                print(f"[HW_BENCH WARN] assign attempt {attempt}/4 exception: {e}", flush=True)
            active_chk = query_active_sessions(cfg_path) or []
            if any(f"[{session_name}]" in ln for ln in active_chk):
                assigned_ok = True
                break
            if args.accel == "tpu_v6e1" and any("[?]" in ln for ln in active_chk):
                if _adopt_single_unnamed_v6e1(session_name, cfg_path):
                    assigned_ok = True
                    break
            if attempt < 4:
                time.sleep(15.0)
        if not assigned_ok:
            raise RuntimeError(f"Failed to create {args.accel} session after 4 attempts: {last_err}")
        created = True
        t_alloc = time.monotonic()  # Start elapsed-time budget clock ONLY once session is actually allocated!

        # Upload bundle cleanly via `colab upload` (never inline large base64 inside WebSocket frames!)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf_tmp:
            tf_tmp.write(tar_bytes)
            tmp_path = Path(tf_tmp.name)
        try:
            up_res = run_colab(
                "upload", "-s", session_name, str(tmp_path), "/content/slwm_v3_bench.tar.gz", cfg_path=cfg_path, timeout=180
            )
            if up_res.returncode != 0:
                raise RuntimeError(f"Upload failed: {up_res.stderr or up_res.stdout}")
        finally:
            tmp_path.unlink(missing_ok=True)

        _, live_rate_cu_hr = query_balance(cfg_path)
        print(f"[HW_BENCH] Live billing rate on {args.accel}: {live_rate_cu_hr:.2f} CU/hr", flush=True)

        # Prep workspace + ensure JAX CUDA/TPU + optax available inside Colab kernel!
        prep_py = (
            "import os, subprocess, sys, tarfile\n"
            "os.makedirs('/content/slwm_v3', exist_ok=True)\n"
            "os.makedirs('/content/slwm_v3_out', exist_ok=True)\n"
            "os.makedirs('/content/slwm_v3_xla', exist_ok=True)\n"
            "with tarfile.open('/content/slwm_v3_bench.tar.gz', 'r:gz') as tf:\n"
            "    tf.extractall('/content/slwm_v3')\n"
            "try:\n"
            "    import optax\n"
            "except Exception:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optax'], check=True)\n"
            "    import optax\n"
            "import jax\n"
            "print('BENCH_PREP_OK', jax.__version__, optax.__version__, [(d.platform, d.device_kind) for d in jax.devices()], flush=True)\n"
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
            phase_timeout_s=600,
            exec_timeout_s=480,
        )
        print(out, flush=True)
        if rc != 0 or tripped or "BENCH_PREP_OK" not in out:
            raise RuntimeError(f"Prep failed rc={rc} tripped={tripped}: {err[-500:]}")

        run_py = (
            "import os, sys, json\n"
            'os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/slwm_v3_xla"\n'
            'os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"\n'
            'os.chdir("/content/slwm_v3")\n'
            'if "/content/slwm_v3" not in sys.path:\n'
            '    sys.path.insert(0, "/content/slwm_v3")\n'
            f'sys.argv = ["scripts/train/v3_hardware_benchmark.py", "--suite", "{args.suite}", "--out-json", "/content/slwm_v3_out/hw_bench.json"]\n'
            "from scripts.train import v3_hardware_benchmark\n"
            "v3_hardware_benchmark.main()\n"
            "print('BENCH_JSON_BEGIN')\n"
            "print(open('/content/slwm_v3_out/hw_bench.json').read())\n"
            "print('BENCH_JSON_END', flush=True)\n"
        )
        for exec_try in range(1, 3):
            rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                session_name,
                run_py,
                cfg_path=cfg_path,
                log_dir=log_dir,
                log_prefix=f"01_bench_try{exec_try}",
                bal_start=bal_start,
                prior_spent_cu=0.0,
                max_spend_cu=args.max_spend_cu,
                t_alloc_start=t_alloc,
                phase_timeout_s=1800,
                exec_timeout_s=1500,
            )
            if "BENCH_JSON_BEGIN" in out or tripped:
                break
            if "Connection was lost." in err and exec_try < 2:
                print(f"[HW_BENCH WARN] WebSocket dropped on attempt {exec_try}; restarting kernel & retrying...", flush=True)
                run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
                time.sleep(8.0)
            else:
                break
        print(out, flush=True)
        if err.strip():
            print(err[-1000:], flush=True)
        if "BENCH_JSON_BEGIN" in out and "BENCH_JSON_END" in out:
            raw_j = out.split("BENCH_JSON_BEGIN", 1)[1].split("BENCH_JSON_END", 1)[0].strip()
            bench_json = json.loads(raw_j)
        else:
            # Resilient fallback: recover incremental JSON written to remote disk even if WebSocket dropped!
            out_path_dl = args.out_dir / f"hw_bench_{args.accel}_partial.json"
            dl = run_colab("download", "-s", session_name, "/content/slwm_v3_out/hw_bench.json", str(out_path_dl), cfg_path=cfg_path, timeout=120)
            if dl.returncode == 0 and out_path_dl.is_file():
                bench_json = json.loads(out_path_dl.read_text())
            else:
                raise RuntimeError(f"Benchmark JSON missing (rc={rc}, tripped={tripped})")
        _, r_now = query_balance(cfg_path)
        if r_now > 0.0:
            live_rate_cu_hr = r_now
        bench_json["billing_rate_cu_hr"] = round(live_rate_cu_hr, 3)
        for row in bench_json.get("results", []):
            pps = float(row.get("warm_positions_per_sec") or 0.0)
            if pps > 0 and live_rate_cu_hr > 0:
                row["cu_per_100m_positions"] = round((live_rate_cu_hr / 3600.0) * (1e8 / pps), 3)
        out_path = args.out_dir / f"hw_bench_{args.accel}.json"
        out_path.write_text(json.dumps(bench_json, indent=2) + "\n")
        print(f"[HW_BENCH] Saved -> {out_path}", flush=True)
    finally:
        if created:
            ok = verified_stop_session(session_name, cfg_path)
            bal_end, rate_end = query_balance(cfg_path)
            summary = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "accel": args.accel,
                "session_name": session_name,
                "teardown_verified_removed": ok,
                "balance_start_cu": bal_start,
                "balance_end_cu": bal_end,
                "spend_cu": round(max(0.0, bal_start - bal_end), 3),
                "rate_after_stop_cu_hr": rate_end,
            }
            (args.out_dir / f"hw_bench_{args.accel}_teardown.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
