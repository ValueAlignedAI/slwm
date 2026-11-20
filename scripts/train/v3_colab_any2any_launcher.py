"""Colab TPU `v6e-1` / GPU Unified Any-to-Any Real-Corpus Training & External SoTA Launcher (`Waves A-D`).

Runs matched-budget, multi-seed real-corpus any-to-any training (`Text/Code + Audio<->Text +
Video<->Audio<->Text + Embodied Robotics`) + strictly frozen-weight external SoTA evaluations
(WikiText-103, LAMBADA, HumanEval/MBPP, ESC-50, LibriSpeech, Clotho, GreatestHits, MSR-VTT,
VGGSound, LeRobot action-intervention, Additive-RF Real-Speech-Overwrite, & Wave B PolicyCommitGate)
on Google TPU `v6e-1` (`5.09 CU/hr`) or NVIDIA `H100`/`A100` with guaranteed verified auto-teardown!
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

TRAIN_BUNDLE_FILES = tuple(
    dict.fromkeys(
        (
            *PILOT_BUNDLE_FILES,
            "data/v3_unified_shards.py",
            "models/v3/any2any_heads.py",
            "evals/v3_sota_benchmarks.py",
            "scripts/train/v3_any2any_trainer.py",
            "scripts/train/v3_wave_b_policy_gate.py",
            "scripts/train/v3_wave_c_decoders.py",
        )
    )
)


def _build_code_tar(repo_root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in TRAIN_BUNDLE_FILES:
            fp = repo_root / rel
            if not fp.is_file():
                raise FileNotFoundError(f"Missing bundle file: {fp}")
            data = fp.read_bytes()
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch Wave A-D SLWM-v3 real-corpus training & SoTA eval on Colab.")
    ap.add_argument("--accel", type=str, default="tpu_v6e1", choices=("tpu_v6e1", "h100", "a100"))
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--arms", type=str, default="slwm_v3_routed,single_bank_gated_deltanet,sliding_window_gqa_eqbyte")
    ap.add_argument("--seeds", type=str, default="4201")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--base-lr", type=float, default=3e-3)
    ap.add_argument("--local-shard-dir", type=Path, required=True, help="Directory with pre-encoded `.npz` shards to upload.")
    ap.add_argument("--run-wave-b", action="store_true", help="Also run frozen-core Wave B PolicyCommitGate after each arm.")
    ap.add_argument("--run-wave-c", action="store_true", help="Also run frozen-core Wave C Decoders & causal audit after each arm.")
    ap.add_argument("--max-spend-cu", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / ".ops/colab-config.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"slwm_v3_wave_{args.profile}_{args.accel}"

    bal_start, rate_start = query_balance(cfg_path)
    print(f"[WAVE_LAUNCH] profile={args.profile} accel={args.accel} start_bal={bal_start:.2f} CU rate={rate_start:.2f}", flush=True)
    code_tar = _build_code_tar(repo_root)
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

        # Upload code bundle
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf_tmp:
            tf_tmp.write(code_tar)
            tmp_p = Path(tf_tmp.name)
        try:
            up = run_colab("upload", "-s", session_name, str(tmp_p), "/content/slwm_v3_code.tar.gz", cfg_path=cfg_path, timeout=180)
            if up.returncode != 0:
                raise RuntimeError(f"Code upload failed: {up.stderr or up.stdout}")
        finally:
            tmp_p.unlink(missing_ok=True)

        # Prep remote workspace & install optax if missing
        prep_py = (
            "import os, subprocess, sys, tarfile\n"
            "for d in ('/content/slwm_v3', '/content/shards/text_code', '/content/shards/audio_text', '/content/shards/av', '/content/shards/robotics', '/content/shards/real_speech_overwrite', '/content/out', '/content/xla'):\n"
            "    os.makedirs(d, exist_ok=True)\n"
            "with tarfile.open('/content/slwm_v3_code.tar.gz', 'r:gz') as tf:\n"
            "    tf.extractall('/content/slwm_v3')\n"
            "try:\n"
            "    import optax\n"
            "except Exception:\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optax'], check=True)\n"
            "    import optax\n"
            "import jax\n"
            "print('WAVE_PREP_OK', jax.__version__, [(d.platform, d.device_kind) for d in jax.devices()], flush=True)\n"
        )
        rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
            session_name, prep_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix="00_prep",
            bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
            t_alloc_start=t_alloc, phase_timeout_s=600, exec_timeout_s=480,
        )
        print(out, flush=True)
        if rc != 0 or tripped or "WAVE_PREP_OK" not in out:
            raise RuntimeError(f"Prep failed rc={rc}: {err[-500:]}")

        # Upload all pre-encoded `.npz` shards preserving relative subdirectory structure
        shard_files = sorted(args.local_shard_dir.rglob("*.npz"))
        if not shard_files:
            raise FileNotFoundError(f"No .npz shards found under {args.local_shard_dir}")
        for sf in shard_files:
            rel = sf.relative_to(args.local_shard_dir).as_posix()
            remote_p = f"/content/shards/{rel}"
            sz_mb = sf.stat().st_size / (1024.0**2)
            print(f"[WAVE_LAUNCH] Uploading shard {rel} ({sz_mb:.1f} MiB)...", flush=True)
            up_s = run_colab("upload", "-s", session_name, str(sf), remote_p, cfg_path=cfg_path, timeout=900)
            if up_s.returncode != 0:
                raise RuntimeError(f"Shard upload failed for {rel}: {up_s.stderr or up_s.stdout}")

        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

        for arm in arms:
            for seed in seeds:
                # Fresh kernel before each arm to keep HBM and Host RAM clean while preserving XLA disk cache!
                run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
                time.sleep(6.0)
                use_wy = (arm == "single_bank_gated_deltanet_wy")
                bb_name = "single_bank_gated_deltanet" if use_wy else arm
                argv = [
                    "scripts/train/v3_any2any_trainer.py",
                    "--profile", args.profile,
                    "--backbone", bb_name,
                    "--match-anchor",
                    "--shard-dir", "/content/shards",
                    "--steps", str(args.steps),
                    "--seq-len", str(args.seq_len),
                    "--batch-size", str(args.batch_size),
                    "--base-lr", str(args.base_lr),
                    "--seed", str(seed),
                    "--out-dir", "/content/out",
                ]
                if use_wy:
                    argv.append("--use-wy-chunking")

                wave_b_block = ""
                if args.run_wave_b:
                    ckpt_rem = f"/content/out/{args.profile}_{bb_name}_s{seed}_ckpt.pkl"
                    wave_b_block = (
                        f"sys.argv = ['scripts/train/v3_wave_b_policy_gate.py', '--ckpt', '{ckpt_rem}', '--shard-dir', '/content/shards', '--steps', '400', '--seed', '{seed + 1000}', '--out-dir', '/content/out']\n"
                        "from scripts.train import v3_wave_b_policy_gate\n"
                        "v3_wave_b_policy_gate.main()\n"
                    )

                wave_c_block = ""
                if args.run_wave_c:
                    ckpt_rem = f"/content/out/{args.profile}_{bb_name}_s{seed}_ckpt.pkl"
                    wave_c_block = (
                        f"sys.argv = ['scripts/train/v3_wave_c_decoders.py', '--ckpt', '{ckpt_rem}', '--shard-dir', '/content/shards', '--steps', '250', '--seq-len', '{args.seq_len}', '--batch-size', '{min(8, args.batch_size)}', '--seed', '{seed + 2000}', '--out-dir', '/content/out']\n"
                        "from scripts.train import v3_wave_c_decoders\n"
                        "v3_wave_c_decoders.main()\n"
                    )

                run_py = (
                    "import os, sys\n"
                    'os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/xla"\n'
                    'os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"\n'
                    'os.chdir("/content/slwm_v3")\n'
                    'if "/content/slwm_v3" not in sys.path:\n'
                    '    sys.path.insert(0, "/content/slwm_v3")\n'
                    f"sys.argv = {json.dumps(argv)}\n"
                    "from scripts.train import v3_any2any_trainer\n"
                    "v3_any2any_trainer.main()\n"
                    + wave_b_block
                    + wave_c_block
                    + "print('ARM_RUN_OK', flush=True)\n"
                )
                rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                    session_name, run_py, cfg_path=cfg_path, log_dir=log_dir,
                    log_prefix=f"arm_{arm}_s{seed}",
                    bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                    t_alloc_start=t_alloc, phase_timeout_s=7200, exec_timeout_s=6600,
                )
                print(out[-2000:], flush=True)
                if err.strip():
                    print(err[-800:], flush=True)
                # Download receipt JSONs & deterministic inference_showcase manifests immediately after each arm!
                rec_fn = f"{args.profile}_{bb_name}_s{seed}_receipt.json"
                out_rec = args.out_dir / f"{args.profile}_{arm}_s{seed}_receipt.json"
                run_colab("download", "-s", session_name, f"/content/out/{rec_fn}", str(out_rec), cfg_path=cfg_path, timeout=120)
                (args.out_dir / "inference_showcase").mkdir(parents=True, exist_ok=True)
                for sh_tag in ("clotho_dev", "msrvtt_dev", "wikitext103_validation"):
                    rem_sc = f"/content/out/inference_showcase/showcase_manifest_dev_{args.profile}_{bb_name}_s{seed}_{sh_tag}.json"
                    loc_sc = args.out_dir / "inference_showcase" / f"showcase_manifest_dev_{args.profile}_{arm}_s{seed}_{sh_tag}.json"
                    run_colab("download", "-s", session_name, rem_sc, str(loc_sc), cfg_path=cfg_path, timeout=60)
                if args.run_wave_b:
                    wb_fn = f"wave_b_policy_gate_{args.profile}_{bb_name}_receipt.json"
                    run_colab("download", "-s", session_name, f"/content/out/{wb_fn}", str(args.out_dir / f"wave_b_{arm}_s{seed}_receipt.json"), cfg_path=cfg_path, timeout=120)
                if args.run_wave_c:
                    wc_fn = f"wave_c_decoders_{args.profile}_{bb_name}_receipt.json"
                    run_colab("download", "-s", session_name, f"/content/out/{wc_fn}", str(args.out_dir / f"wave_c_{arm}_s{seed}_receipt.json"), cfg_path=cfg_path, timeout=120)
                if tripped:
                    print("[WAVE_LAUNCH] Budget watchdog tripped; halting cleanly.", flush=True)
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
