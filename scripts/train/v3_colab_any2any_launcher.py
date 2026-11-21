"""Colab TPU `v6e-1` / GPU Unified Any-to-Any Real-Corpus Training & External SoTA Launcher (`Waves A-D`).

Runs matched-budget, multi-seed real-corpus any-to-any training (`Text/Code + Audio<->Text +
Video<->Audio<->Text + Embodied Robotics`) + strictly frozen-weight external SoTA evaluations
(WikiText-103, LAMBADA, HumanEval/MBPP, ESC-50, LibriSpeech, Clotho, GreatestHits, MSR-VTT,
VGGSound, LeRobot action-intervention, Additive-RF Real-Speech-Overwrite, & Wave B PolicyCommitGate)
on Google TPU `v6e-1` (`5.09 CU/hr`) or NVIDIA `H100`/`A100` with guaranteed verified auto-teardown!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
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
    _run_remote_kernel_python_with_watchdog,
    _unassign_unnamed_assignments,
    allocate_colab_session_safe,
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
            "evals/v3_external_sota_suite.py",
            "evals/v3_sota_benchmarks.py",
            "scripts/train/v3_any2any_trainer.py",
            "scripts/train/v3_wave_b_policy_gate.py",
            "scripts/train/v3_wave_c_decoders.py",
        )
    )
)


def _upload_shards_batch_chunked(
    session_name: str,
    shard_files: list[Path],
    local_root: Path,
    *,
    cfg_path: Path,
    log_dir: Path,
    bal_start: float,
    max_spend_cu: float,
    t_alloc: float,
    chunk_bytes: int = 32 * 1024 * 1024,
) -> None:
    """Stateless 2-stream HTTPS PUT upload of all shards in <=32 MiB parts + ONE retried SHA-256 assembly call.

    Completely immune to WebSocket drops during upload (zero WebSocket traffic while uploading parts!)
    and handles arbitrary shard sizes (`100 MiB .. 4.1 GiB`) cleanly across Jupyter `/api/contents` caps.
    """
    assembly_plan: list[dict[str, Any]] = []

    def _put_one(local_tmp: Path, rem_p: str) -> None:
        last_err = ""
        for attempt in range(1, 5):
            up = run_colab("upload", "-s", session_name, str(local_tmp), rem_p, cfg_path=cfg_path, timeout=300)
            if up.returncode == 0:
                return
            last_err = up.stderr or up.stdout
            time.sleep(3.0 * attempt)
        raise RuntimeError(f"Upload failed for {rem_p}: {last_err}")

    for sf_idx, sf in enumerate(shard_files):
        rel = sf.relative_to(local_root).as_posix()
        remote_p = f"/content/shards/{rel}"
        sz = sf.stat().st_size
        sz_mb = sz / (1024.0**2)
        print(f"[WAVE_LAUNCH] Uploading shard {sf_idx+1}/{len(shard_files)}: {rel} ({sz_mb:.1f} MiB)...", flush=True)
        if sz <= chunk_bytes:
            _put_one(sf, remote_p)
            continue

        # Split locally into <=32 MiB parts and upload 2 parts concurrently via stateless HTTPS PUT
        sha = hashlib.sha256()
        part_jobs: list[tuple[Path, str]] = []
        with sf.open("rb") as fin:
            p_idx = 0
            while True:
                buf = fin.read(chunk_bytes)
                if not buf:
                    break
                sha.update(buf)
                tf = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
                tf.write(buf)
                tf.close()
                part_jobs.append((Path(tf.name), f"/content/_part_s{sf_idx:02d}_{p_idx:03d}.bin"))
                p_idx += 1
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futs = [pool.submit(_put_one, lp, rp) for lp, rp in part_jobs]
                for f in futs:
                    f.result()
        finally:
            for lp, _ in part_jobs:
                lp.unlink(missing_ok=True)
        assembly_plan.append(
            {
                "dst": remote_p,
                "size": int(sz),
                "sha256": sha.hexdigest(),
                "parts": [rp for _, rp in part_jobs],
            }
        )

    if not assembly_plan:
        return

    asm_py = (
        "import hashlib, os\n"
        f"plan = {json.dumps(assembly_plan)}\n"
        "for item in plan:\n"
        "    dst = item['dst']\n"
        "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "    if os.path.isfile(dst) and os.path.getsize(dst) == item['size']:\n"
        "        continue\n"
        "    h = hashlib.sha256()\n"
        "    tot = 0\n"
        "    with open(dst, 'wb') as fout:\n"
        "        for pp in item['parts']:\n"
        "            with open(pp, 'rb') as fin:\n"
        "                b = fin.read()\n"
        "            fout.write(b)\n"
        "            h.update(b)\n"
        "            tot += len(b)\n"
        "            os.remove(pp)\n"
        "    assert tot == item['size'], f'size mismatch on {dst}: {tot} != {item[\"size\"]}'\n"
        "    assert h.hexdigest() == item['sha256'], f'sha256 mismatch on {dst}'\n"
        "print('ALL_SHARDS_ASM_OK', len(plan), flush=True)\n"
    )
    last_err = ""
    for attempt in range(1, 4):
        run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
        time.sleep(5.0)
        try:
            rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                session_name, asm_py, cfg_path=cfg_path, log_dir=log_dir,
                log_prefix=f"asm_all_try{attempt}",
                bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=max_spend_cu,
                t_alloc_start=t_alloc, phase_timeout_s=180, exec_timeout_s=150,
            )
            if rc == 0 and not tripped and "ALL_SHARDS_ASM_OK" in out:
                print(out.strip(), flush=True)
                return
            last_err = f"rc={rc} err={err[-400:]}"
        except Exception as exc:
            last_err = repr(exc)
        print(f"[WAVE_LAUNCH] Shard assembly attempt {attempt}/3 retry after: {last_err}", flush=True)
        time.sleep(8.0)
    raise RuntimeError(f"Batch shard assembly failed after 3 attempts: {last_err}")


def _download_file_chunked_verified(
    session_name: str,
    remote_path: str,
    local_path: Path,
    *,
    cfg_path: Path,
    log_dir: Path,
    bal_start: float,
    max_spend_cu: float,
    t_alloc: float,
    chunk_bytes: int = 32 * 1024 * 1024,
) -> bool:
    """Safely download large checkpoints (`600 MiB - 2.5 GiB`) in `<= 32 MiB` parts with SHA-256 verification."""
    split_py = (
        f"import hashlib, json, os\n"
        f"src = {remote_path!r}\n"
        "if not os.path.isfile(src):\n"
        "    print('DL_MISSING', flush=True)\n"
        "else:\n"
        "    sz = os.path.getsize(src)\n"
        "    h = hashlib.sha256()\n"
        "    parts = []\n"
        "    idx = 0\n"
        f"    cbytes = {chunk_bytes}\n"
        "    with open(src, 'rb') as f:\n"
        "        while True:\n"
        "            b = f.read(cbytes)\n"
        "            if not b:\n"
        "                break\n"
        "            h.update(b)\n"
        "            pp = f'/tmp/_dl_part_{idx:04d}.bin'\n"
        "            with open(pp, 'wb') as fout:\n"
        "                fout.write(b)\n"
        "            parts.append(pp)\n"
        "            idx += 1\n"
        "    print('DL_META_BEGIN' + json.dumps({'size': sz, 'sha256': h.hexdigest(), 'parts': parts}) + 'DL_META_END', flush=True)\n"
    )
    rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
        session_name, split_py, cfg_path=cfg_path, log_dir=log_dir,
        log_prefix=f"dl_split_{local_path.stem}",
        bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=max_spend_cu,
        t_alloc_start=t_alloc, phase_timeout_s=300, exec_timeout_s=240,
    )
    if rc != 0 or tripped or "DL_META_BEGIN" not in out:
        print(f"[CKPT_DL WARN] Could not prepare {remote_path}: {out[-200:]} {err[-200:]}", flush=True)
        return False
    meta = json.loads(out.split("DL_META_BEGIN", 1)[1].split("DL_META_END", 1)[0])
    parts: list[str] = meta["parts"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_local = local_path.with_suffix(local_path.suffix + ".part")
    h_loc = hashlib.sha256()
    tot = 0
    print(f"[CKPT_DL] Downloading {Path(remote_path).name} ({meta['size'] / 1024**2:.1f} MiB in {len(parts)} parts) -> {local_path}...", flush=True)
    with tmp_local.open("wb") as fout:
        for i, rem_p in enumerate(parts):
            p_loc = local_path.parent / f"._{local_path.name}.p{i:04d}"
            ok_p = False
            for att in range(1, 4):
                res = run_colab("download", "-s", session_name, rem_p, str(p_loc), cfg_path=cfg_path, timeout=300)
                if res.returncode == 0 and p_loc.is_file():
                    ok_p = True
                    break
                time.sleep(5.0 * att)
            if not ok_p:
                print(f"[CKPT_DL WARN] Part {i+1}/{len(parts)} failed for {remote_path}", flush=True)
                p_loc.unlink(missing_ok=True)
                tmp_local.unlink(missing_ok=True)
                return False
            raw = p_loc.read_bytes()
            fout.write(raw)
            h_loc.update(raw)
            tot += len(raw)
            p_loc.unlink(missing_ok=True)
    if tot != int(meta["size"]) or h_loc.hexdigest() != str(meta["sha256"]):
        print(f"[CKPT_DL WARN] Integrity mismatch on {local_path}: {tot} vs {meta['size']}", flush=True)
        tmp_local.unlink(missing_ok=True)
        return False
    tmp_local.replace(local_path)
    # Clean up remote temp parts + remote ckpt to keep `/content` disk lean
    clean_py = (
        "import glob, os\n"
        "for p in glob.glob('/tmp/_dl_part_*.bin'):\n"
        "    try: os.remove(p)\n"
        "    except OSError: pass\n"
        f"try: os.remove({remote_path!r})\n"
        "except OSError: pass\n"
    )
    run_colab("exec", "-s", session_name, "-c", clean_py, cfg_path=cfg_path, timeout=60)
    print(f"[CKPT_DL OK] Verified SHA-256 ({h_loc.hexdigest()[:16]}...) -> {local_path}", flush=True)
    return True


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
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
        for init_rel in ("models/__init__.py", "data/__init__.py", "evals/__init__.py", "scripts/__init__.py", "scripts/train/__init__.py"):
            ti = tarfile.TarInfo(name=init_rel)
            ti.size = 0
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(b""))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch Wave A-D SLWM-v3 real-corpus training & SoTA eval on Colab.")
    ap.add_argument("--accel", type=str, default="tpu_v6e1", choices=("tpu_v6e1", "h100", "a100"))
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--representation-mode", type=str, default="frozen_codec_edge", choices=("frozen_codec_edge", "scratch_edge"))
    ap.add_argument("--arms", type=str, default="slwm_v3_routed,single_bank_gated_deltanet,sliding_window_gqa_eqbyte")
    ap.add_argument("--seeds", type=str, default="4201")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=2e-3)
    ap.add_argument("--drop-modality-prob", type=float, default=0.25)
    ap.add_argument("--anchor-transition-prob", type=float, default=0.50)
    ap.add_argument("--multi-horizon-weight", type=float, default=0.15)
    ap.add_argument("--local-shard-dir", type=Path, required=True, help="Directory with pre-encoded `.npz` shards to upload.")
    ap.add_argument("--include-test-shards", action="store_true", help="Upload sealed `*_test*.npz` shards & enable `--eval-sealed-test`.")
    ap.add_argument("--download-ckpts", action=argparse.BooleanOptionalAction, default=True, help="Download trained core+Wave B/C `.pkl` checkpoints with SHA-256 verification before cleanup (default: True).")
    ap.add_argument("--ckpt-dir", type=Path, default=Path(".ops/checkpoints"), help="Local gitignored directory for downloaded `.pkl` checkpoints.")
    ap.add_argument("--run-wave-b", action="store_true", help="Also run frozen-core Wave B PolicyCommitGate after each arm.")
    ap.add_argument("--run-wave-c", action="store_true", help="Also run frozen-core Wave C Decoders & causal audit after each arm.")
    ap.add_argument("--resume", action="store_true", help="Skip arms/seeds whose complete receipt JSONs already exist in `--out-dir`.")
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
        allocate_colab_session_safe(session_name, args.accel, cfg_path, max_attempts=4, sleep_s=15.0)
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
        rc, out, err, tripped = 1, "", "", False
        for p_try in range(1, 4):
            if p_try > 1:
                time.sleep(8.0)
                run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
                time.sleep(4.0)
            rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                session_name, prep_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix=f"00_prep_try{p_try}",
                bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                t_alloc_start=t_alloc, phase_timeout_s=600, exec_timeout_s=480,
            )
            print(out, flush=True)
            if rc == 0 and not tripped and "WAVE_PREP_OK" in out:
                break
        if rc != 0 or tripped or "WAVE_PREP_OK" not in out:
            raise RuntimeError(f"Prep failed rc={rc}: {err[-500:]}")

        # Upload ONLY active pre-encoded `.npz` shards (plus `*_exec.jsonl` test suites when `--include-test-shards`!)
        shard_files = sorted(args.local_shard_dir.rglob("*.npz"))
        if args.include_test_shards:
            shard_files = sorted(shard_files + list(args.local_shard_dir.rglob("*_exec.jsonl")))
        else:
            shard_files = [p for p in shard_files if "_test" not in p.name]
        # Avoid uploading redundant 24m text shard if full train_v1 is present
        has_full_tc = any(p.name == "text_code_train_v1.npz" for p in shard_files)
        if has_full_tc:
            shard_files = [p for p in shard_files if p.name != "text_code_train_24m_balanced.npz"]
        if not shard_files:
            raise FileNotFoundError(f"No .npz shards found under {args.local_shard_dir}")
        _upload_shards_batch_chunked(
            session_name, shard_files, args.local_shard_dir,
            cfg_path=cfg_path, log_dir=log_dir,
            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
        )

        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        tripped = False

        for arm in arms:
            for seed in seeds:
                use_wy = arm.endswith("_wy") or ("_c16" in arm) or ("_c32" in arm)
                surprise_c = 16 if "_c16" in arm else (32 if "_c32" in arm else 1)
                bb_name = arm.replace("_wy", "").replace("_c16", "").replace("_c32", "")
                rep_sfx = "" if args.representation_mode == "frozen_codec_edge" else f"_{args.representation_mode}"
                tag = f"{args.profile}_{arm}{rep_sfx}_s{seed}"
                out_rec = args.out_dir / f"{tag}_receipt.json"
                if args.resume and out_rec.is_file():
                    wb_ok = (not args.run_wave_b) or (args.out_dir / f"wave_b_{tag}_receipt.json").is_file()
                    wc_ok = (not args.run_wave_c) or (args.out_dir / f"wave_c_{tag}_receipt.json").is_file()
                    if wb_ok and wc_ok:
                        print(f"[WAVE_LAUNCH RESUME] Skipping completed {tag}.", flush=True)
                        continue
                argv = [
                    "scripts/train/v3_any2any_trainer.py",
                    "--profile", args.profile,
                    "--backbone", bb_name,
                    "--representation-mode", args.representation_mode,
                    "--match-anchor",
                    "--shard-dir", "/content/shards",
                    "--steps", str(args.steps),
                    "--seq-len", str(args.seq_len),
                    "--batch-size", str(args.batch_size),
                    "--grad-accum", str(args.grad_accum),
                    "--base-lr", str(args.base_lr),
                    "--drop-modality-prob", str(args.drop_modality_prob),
                    "--anchor-transition-prob", str(args.anchor_transition_prob),
                    "--multi-horizon-weight", str(args.multi_horizon_weight),
                    "--seed", str(seed),
                    "--tag", tag,
                    "--out-dir", "/content/out",
                ]
                if surprise_c > 1:
                    argv.extend(["--surprise-chunk-size", str(surprise_c)])
                if use_wy:
                    argv.append("--use-wy-chunking")
                if args.include_test_shards:
                    argv.append("--eval-sealed-test")

                wy_flag = ", '--use-wy-chunking'" if use_wy else ""
                ckpt_rem = f"/content/out/{tag}_ckpt.pkl"
                wave_b_block = ""
                if args.run_wave_b:
                    wave_b_block = (
                        f"sys.argv = ['scripts/train/v3_wave_b_policy_gate.py', '--ckpt', '{ckpt_rem}', '--shard-dir', '/content/shards', '--tag', '{tag}', '--steps', '400', '--seed', '{seed + 1000}', '--out-dir', '/content/out'{wy_flag}]\n"
                        "from scripts.train import v3_wave_b_policy_gate\n"
                        "v3_wave_b_policy_gate.main()\n"
                    )

                wave_c_block = ""
                if args.run_wave_c:
                    wave_c_block = (
                        f"sys.argv = ['scripts/train/v3_wave_c_decoders.py', '--ckpt', '{ckpt_rem}', '--shard-dir', '/content/shards', '--tag', '{tag}', '--steps', '250', '--seq-len', '{args.seq_len}', '--batch-size', '{min(8, args.batch_size)}', '--seed', '{seed + 2000}', '--out-dir', '/content/out'{wy_flag}]\n"
                        "from scripts.train import v3_wave_c_decoders\n"
                        "v3_wave_c_decoders.main()\n"
                    )

                cleanup_ckpt_block = "" if args.download_ckpts else (
                    f"if os.path.exists('{ckpt_rem}'): os.remove('{ckpt_rem}')\n"
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
                    + cleanup_ckpt_block
                    + "print('ARM_RUN_OK', flush=True)\n"
                )
                rc, out, err, tripped = 1, "", "", False
                for attempt in range(1, 3):
                    run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
                    time.sleep(6.0)
                    try:
                        rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                            session_name, run_py, cfg_path=cfg_path, log_dir=log_dir,
                            log_prefix=f"arm_{arm}_s{seed}_try{attempt}",
                            bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                            t_alloc_start=t_alloc, phase_timeout_s=25200, exec_timeout_s=24000,
                        )
                    except TimeoutError as exc:
                        err = repr(exc)
                        rc = 124
                    if rc == 0 or tripped:
                        break
                    if ("Connection was lost" in err) or ("Connection was lost" in out) or ("WebSocket" in err):
                        print(f"[WAVE_LAUNCH] WebSocket drop on {tag} (attempt {attempt}/2); restarting kernel & retrying...", flush=True)
                        time.sleep(10.0)
                        continue
                    break

                print(out[-2000:], flush=True)
                if err.strip():
                    print(err[-800:], flush=True)
                # Download receipt JSONs & deterministic inference_showcase manifests immediately after each arm!
                rec_fn = f"{tag}_receipt.json"
                run_colab("download", "-s", session_name, f"/content/out/{rec_fn}", str(out_rec), cfg_path=cfg_path, timeout=120)
                (args.out_dir / "inference_showcase").mkdir(parents=True, exist_ok=True)
                for sh_tag in ("clotho_dev", "msrvtt_dev", "wikitext103_validation"):
                    rem_sc = f"/content/out/inference_showcase/showcase_manifest_dev_{tag}_{sh_tag}.json"
                    loc_sc = args.out_dir / "inference_showcase" / f"showcase_manifest_dev_{tag}_{sh_tag}.json"
                    run_colab("download", "-s", session_name, rem_sc, str(loc_sc), cfg_path=cfg_path, timeout=60)
                if args.include_test_shards:
                    for sh_tag in ("clotho_test", "msrvtt_test", "wikitext103_test"):
                        rem_sc = f"/content/out/inference_showcase/showcase_manifest_test_{tag}_{sh_tag}.json"
                        loc_sc = args.out_dir / "inference_showcase" / f"showcase_manifest_test_{tag}_{sh_tag}.json"
                        run_colab("download", "-s", session_name, rem_sc, str(loc_sc), cfg_path=cfg_path, timeout=60)
                if args.run_wave_b:
                    wb_fn = f"wave_b_policy_gate_{tag}_receipt.json"
                    run_colab("download", "-s", session_name, f"/content/out/{wb_fn}", str(args.out_dir / f"wave_b_{tag}_receipt.json"), cfg_path=cfg_path, timeout=120)
                    run_colab("download", "-s", session_name, f"/content/out/inference_showcase/showcase_manifest_dev_wave_b_{tag}.json", str(args.out_dir / "inference_showcase" / f"showcase_manifest_dev_wave_b_{tag}.json"), cfg_path=cfg_path, timeout=60)
                if args.run_wave_c:
                    wc_fn = f"wave_c_decoders_{tag}_receipt.json"
                    run_colab("download", "-s", session_name, f"/content/out/{wc_fn}", str(args.out_dir / f"wave_c_{tag}_receipt.json"), cfg_path=cfg_path, timeout=120)
                    for sh_tag in ("clotho_dev", "msrvtt_dev"):
                        run_colab("download", "-s", session_name, f"/content/out/inference_showcase/showcase_manifest_dev_wave_c_{tag}_{sh_tag}.json", str(args.out_dir / "inference_showcase" / f"showcase_manifest_dev_wave_c_{tag}_{sh_tag}.json"), cfg_path=cfg_path, timeout=60)
                if args.download_ckpts:
                    ckpt_dst_dir = args.ckpt_dir / args.profile
                    _download_file_chunked_verified(
                        session_name, ckpt_rem, ckpt_dst_dir / f"{tag}_ckpt.pkl",
                        cfg_path=cfg_path, log_dir=log_dir,
                        bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
                    )
                    if args.run_wave_b:
                        _download_file_chunked_verified(
                            session_name, f"/content/out/wave_b_{tag}_ckpt.pkl", ckpt_dst_dir / f"wave_b_{tag}_ckpt.pkl",
                            cfg_path=cfg_path, log_dir=log_dir,
                            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
                        )
                    if args.run_wave_c:
                        _download_file_chunked_verified(
                            session_name, f"/content/out/wave_c_{tag}_ckpt.pkl", ckpt_dst_dir / f"wave_c_{tag}_ckpt.pkl",
                            cfg_path=cfg_path, log_dir=log_dir,
                            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
                        )
                if tripped:
                    print("[WAVE_LAUNCH] Budget watchdog tripped; halting cleanly.", flush=True)
                    break
            if tripped:
                break
    finally:
        if created:
            ok = verified_stop_session(session_name, cfg_path)
        else:
            _unassign_unnamed_assignments(cfg_path)
            ok = True
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
