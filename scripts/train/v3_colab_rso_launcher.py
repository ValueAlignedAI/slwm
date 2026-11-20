"""Colab TPU `v6e-1` / GPU Multi-Seed Real-Speech Selective Overwrite Launcher (`Waves A-RSO`).

Uploads ONLY the compact Additive-RF Real-Speech-Overwrite shards (`~45 MiB` for L12 / `~92 MiB` for L16,
excluding sealed `*_test*` unless `--include-test-shards` is set!) and runs paired multi-seed evaluations
across all matched comparators (`±1.0%` params, matched state bytes) with verified auto-teardown!
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

from scripts.train.v3_colab_any2any_launcher import _upload_shards_batch_chunked
from scripts.train.v3_colab_pilot_launcher import (
    PILOT_BUNDLE_FILES,
    _adopt_single_unnamed_v6e1,
    _run_remote_kernel_python_with_watchdog,
    query_active_sessions,
    query_balance,
    run_colab,
    verified_stop_session,
)

RSO_BUNDLE_FILES = tuple(dict.fromkeys((*PILOT_BUNDLE_FILES, "scripts/train/v3_rso_benchmark_runner.py")))


def _build_tar(repo_root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in RSO_BUNDLE_FILES:
            fp = repo_root / rel
            data = fp.read_bytes()
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(data)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
        for init_rel in ("models/__init__.py", "data/__init__.py", "evals/__init__.py", "scripts/__init__.py", "scripts/train/__init__.py"):
            ti = tarfile.TarInfo(name=init_rel)
            ti.size = 0
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(b""))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch multi-seed Real-Speech Selective Overwrite benchmark on Colab.")
    ap.add_argument("--accel", type=str, default="tpu_v6e1", choices=("tpu_v6e1", "a100", "h100"))
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--arms", type=str, default="slwm_v3_routed,slwm_v3_routed_c16,slwm_v3_no_routing,single_bank_gated_deltanet,explicit_bounded_kv,sliding_window_gqa_eqbyte")
    ap.add_argument("--seeds", type=str, default="4201,4202,4203")
    ap.add_argument("--shard-tag", type=str, default="L12_Weq64_additive")
    ap.add_argument("--local-rso-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--include-test-shards", action="store_true")
    ap.add_argument("--max-spend-cu", type=float, default=35.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / ".ops/colab-config.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"slwm_v3_rso_{args.profile}_{args.accel}"

    bal_start, rate_start = query_balance(cfg_path)
    print(f"[RSO_LAUNCH] profile={args.profile} accel={args.accel} start_bal={bal_start:.2f} CU rate={rate_start:.2f}", flush=True)
    code_tar = _build_tar(repo_root)
    created = False
    t_alloc = time.monotonic()

    try:
        assigned_ok = False
        last_err = ""
        for attempt in range(1, 5):
            try:
                if args.accel == "tpu_v6e1":
                    res = run_colab("new", "-s", session_name, "--tpu", "V6E1", cfg_path=cfg_path, timeout=420)
                elif args.accel == "a100":
                    res = run_colab("new", "-s", session_name, "--gpu", "A100", "--high-mem", cfg_path=cfg_path, timeout=360)
                else:
                    res = run_colab("new", "-s", session_name, "--gpu", "H100", cfg_path=cfg_path, timeout=360)
                if res.returncode == 0:
                    print(res.stdout.strip(), flush=True)
                    assigned_ok = True
                    break
                last_err = (res.stderr or res.stdout)[-400:]
            except Exception as e:
                last_err = str(e)
            active = query_active_sessions(cfg_path) or []
            if any(f"[{session_name}]" in ln for ln in active):
                assigned_ok = True
                break
            if args.accel == "tpu_v6e1" and any("[?]" in ln for ln in active):
                if _adopt_single_unnamed_v6e1(session_name, cfg_path):
                    assigned_ok = True
                    break
            if attempt < 4:
                time.sleep(20.0)
        if not assigned_ok:
            raise RuntimeError(f"Failed to allocate {args.accel}: {last_err}")
        created = True
        t_alloc = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf_tmp:
            tf_tmp.write(code_tar)
            tmp_p = Path(tf_tmp.name)
        try:
            up = run_colab("upload", "-s", session_name, str(tmp_p), "/content/slwm_v3_rso.tar.gz", cfg_path=cfg_path, timeout=180)
            if up.returncode != 0:
                raise RuntimeError(f"Code upload failed: {up.stderr or up.stdout}")
        finally:
            tmp_p.unlink(missing_ok=True)

        prep_py = (
            "import os, subprocess, sys, tarfile\n"
            "for d in ('/content/slwm_v3', '/content/shards', '/content/out', '/content/xla'):\n"
            "    os.makedirs(d, exist_ok=True)\n"
            "with tarfile.open('/content/slwm_v3_rso.tar.gz', 'r:gz') as tf:\n"
            "    tf.extractall('/content/slwm_v3')\n"
            "try:\n"
            "    import optax\n"
            "except Exception:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optax'], check=True)\n"
            "import jax\n"
            "print('RSO_PREP_OK', jax.__version__, [(d.platform, d.device_kind) for d in jax.devices()], flush=True)\n"
        )
        rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
            session_name, prep_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix="00_prep",
            bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
            t_alloc_start=t_alloc, phase_timeout_s=600, exec_timeout_s=480,
        )
        print(out, flush=True)
        if rc != 0 or tripped or "RSO_PREP_OK" not in out:
            raise RuntimeError(f"Prep failed rc={rc}: {err[-500:]}")

        splits = ["train", "dev"] + (["test"] if args.include_test_shards else [])
        shard_files = [args.local_rso_dir / f"real_speech_overwrite_{sp}_{args.shard_tag}.npz" for sp in splits]
        _upload_shards_batch_chunked(
            session_name, shard_files, args.local_rso_dir,
            cfg_path=cfg_path, log_dir=log_dir,
            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
        )

        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        tripped = False

        for arm in arms:
            for seed in seeds:
                use_wy = arm.endswith("_wy") or ("_c16" in arm) or ("_c32" in arm)
                sc = 16 if "_c16" in arm else (32 if "_c32" in arm else 1)
                bb = arm.replace("_wy", "").replace("_c16", "").replace("_c32", "")
                tag = f"{args.profile}_{arm}_s{seed}"
                argv = [
                    "scripts/train/v3_rso_benchmark_runner.py",
                    "--profile", args.profile,
                    "--backbone", bb,
                    "--surprise-chunk-size", str(sc),
                    "--rso-dir", "/content/shards",
                    "--shard-tag", args.shard_tag,
                    "--steps", str(args.steps),
                    "--batch-size", str(args.batch_size),
                    "--grad-accum", str(args.grad_accum),
                    "--base-lr", str(args.base_lr),
                    "--seed", str(seed),
                    "--tag", tag,
                    "--out-dir", "/content/out",
                ]
                if use_wy:
                    argv.append("--use-wy-chunking")
                if args.include_test_shards:
                    argv.append("--eval-sealed-test")

                run_py = (
                    "import os, sys\n"
                    'os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/xla"\n'
                    'os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"\n'
                    'os.chdir("/content/slwm_v3")\n'
                    'if "/content/slwm_v3" not in sys.path:\n'
                    '    sys.path.insert(0, "/content/slwm_v3")\n'
                    f"sys.argv = {json.dumps(argv)}\n"
                    "from scripts.train import v3_rso_benchmark_runner\n"
                    "v3_rso_benchmark_runner.main()\n"
                    "print('RSO_ARM_OK', flush=True)\n"
                )
                rc, out, err, tripped = 1, "", "", False
                for attempt in range(1, 3):
                    run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
                    time.sleep(5.0)
                    try:
                        rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                            session_name, run_py, cfg_path=cfg_path, log_dir=log_dir,
                            log_prefix=f"rso_{tag}_try{attempt}",
                            bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                            t_alloc_start=t_alloc, phase_timeout_s=3600, exec_timeout_s=3300,
                        )
                    except TimeoutError as exc:
                        err = repr(exc)
                        rc = 124
                    if rc == 0 or tripped:
                        break
                    if ("Connection was lost" in err) or ("Connection was lost" in out):
                        time.sleep(8.0)
                        continue
                    break
                print(out[-1500:], flush=True)
                if err.strip():
                    print(err[-600:], flush=True)
                rem_rec = f"/content/out/rso_{tag}_receipt.json"
                run_colab("download", "-s", session_name, rem_rec, str(args.out_dir / f"rso_{tag}_receipt.json"), cfg_path=cfg_path, timeout=120)
                if tripped:
                    break
            if tripped:
                break
    finally:
        if created:
            ok = verified_stop_session(session_name, cfg_path)
            bal_end, rate_end = query_balance(cfg_path)
            summary = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "profile": args.profile,
                "accel": args.accel,
                "session_name": session_name,
                "teardown_verified_removed": ok,
                "balance_start_cu": bal_start,
                "balance_end_cu": bal_end,
                "spend_cu": round(max(0.0, bal_start - bal_end), 3),
                "rate_after_stop_cu_hr": rate_end,
            }
            (args.out_dir / "teardown_receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
