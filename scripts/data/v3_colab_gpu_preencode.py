"""Colab GPU (`A100` / `H100` / `L4`) Causal Video+Audio+Robotics Pre-Encoding Launcher (`Wave A-Data`).

Downloads open-domain Video+Audio+Text & Embodied Robotics corpora directly on Google datacenter
bandwidth inside Colab (`MSR-VTT`, `VGGSound` shard 00, `LeRobot aloha_sim / pusht / libero`),
encodes them strictly causally with frozen `Wan 2.1 Causal VAE (bf16)` (`192x320`, `25 fps`,
`15` isotropic `8x8` patches) + `Kyutai Mimi 32-RVQ` (`12.5 Hz`) + `ByT5` UTF-8 bytes into compact
unified sparse `.npz` shards (`data/v3_unified_shards.py`), downloads all shards cleanly to local
disk / `192.168.1.200`, and guarantees verified auto-teardown (`rate_after_stop_cu_hr == 0.0`).
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train.v3_colab_pilot_launcher import (
    _run_remote_kernel_python_with_watchdog,
    query_active_sessions,
    query_balance,
    run_colab,
    verified_stop_session,
)

PREENCODE_BUNDLE_FILES = (
    "data/__init__.py",
    "data/v3_data_contract.py",
    "data/v3_unified_shards.py",
    "scripts/__init__.py",
    "scripts/data/__init__.py",
    "scripts/data/v3_preencode_av_codecs.py",
)


def _build_tar(repo_root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in PREENCODE_BUNDLE_FILES:
            fp = repo_root / rel
            if not fp.is_file():
                # Create empty __init__.py in archive if absent
                if rel.endswith("__init__.py"):
                    data = b""
                else:
                    raise FileNotFoundError(fp)
            else:
                data = fp.read_bytes()
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-encode MSR-VTT, VGGSound, and LeRobot on Colab GPU.")
    ap.add_argument("--accel", type=str, default="a100", choices=("h100", "a100", "l4"))
    ap.add_argument("--corpora", type=str, default="robotics_lerobot,msrvtt,vggsound")
    ap.add_argument("--max-msrvtt", type=int, default=1200)
    ap.add_argument("--max-vggsound", type=int, default=1200)
    ap.add_argument("--max-lerobot-eps", type=int, default=150)
    ap.add_argument("--max-spend-cu", type=float, default=18.0)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/slwm_v3_gpu_preencoded"))
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / ".ops/colab-config.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_name = f"slwm_v3_preencode_{args.accel}"

    bal_start, rate_start = query_balance(cfg_path)
    print(f"[PREENCODE_GPU] accel={args.accel} start_balance={bal_start:.2f} CU rate={rate_start:.2f}", flush=True)
    tar_bytes = _build_tar(repo_root)
    created = False
    t_alloc = time.monotonic()

    try:
        gpu_flag = {"h100": "H100", "a100": "A100", "l4": "L4"}[args.accel]
        assigned_ok = False
        last_err = ""
        for attempt in range(1, 5):
            cmd = ["new", "-s", session_name, "--gpu", gpu_flag]
            if args.accel in ("a100", "l4"):
                cmd.append("--high-mem")
            print(f"[PREENCODE_GPU] Allocating {gpu_flag} session '{session_name}' (attempt {attempt}/4)...", flush=True)
            res = run_colab(*cmd, cfg_path=cfg_path, timeout=360)
            if res.returncode == 0:
                print(res.stdout.strip(), flush=True)
                assigned_ok = True
                break
            last_err = (res.stderr or res.stdout)[-400:]
            print(f"[PREENCODE_GPU WARN] attempt {attempt}/4 rc={res.returncode}: {last_err}", flush=True)
            active = query_active_sessions(cfg_path) or []
            if any(f"[{session_name}]" in ln for ln in active):
                assigned_ok = True
                break
            if attempt < 4:
                time.sleep(15.0)
        if not assigned_ok:
            raise RuntimeError(f"Failed to allocate {gpu_flag}: {last_err}")
        created = True
        t_alloc = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf_tmp:
            tf_tmp.write(tar_bytes)
            tmp_p = Path(tf_tmp.name)
        try:
            up = run_colab("upload", "-s", session_name, str(tmp_p), "/content/preencode_bundle.tar.gz", cfg_path=cfg_path, timeout=180)
            if up.returncode != 0:
                raise RuntimeError(f"Upload failed: {up.stderr or up.stdout}")
        finally:
            tmp_p.unlink(missing_ok=True)

        prep_py = (
            "import os, subprocess, sys, tarfile\n"
            "for d in ('/content/slwm_v3', '/content/raw', '/content/large_av', '/content/out_av', '/content/out_rob'):\n"
            "    os.makedirs(d, exist_ok=True)\n"
            "with tarfile.open('/content/preencode_bundle.tar.gz', 'r:gz') as tf:\n"
            "    tf.extractall('/content/slwm_v3')\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers', 'transformers', 'accelerate', 'huggingface_hub', 'pyarrow', 'scipy'], check=True)\n"
            "import torch\n"
            "print('GPU_PREP_OK', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', flush=True)\n"
        )
        rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
            session_name, prep_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix="00_prep",
            bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
            t_alloc_start=t_alloc, phase_timeout_s=600, exec_timeout_s=480,
        )
        print(out, flush=True)
        if rc != 0 or tripped or "GPU_PREP_OK" not in out:
            raise RuntimeError(f"Prep failed rc={rc}: {err[-500:]}")

        corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
        downloaded_files: list[str] = []

        if "robotics_lerobot" in corpora:
            rob_py = f"""
import os, sys
from huggingface_hub import snapshot_download
os.chdir('/content/slwm_v3')
sys.path.insert(0, '/content/slwm_v3')
for repo_id, dst in (
    ('lerobot/aloha_sim_insertion_human', '/content/raw/lerobot_aloha_sim'),
    ('lerobot/pusht', '/content/raw/lerobot_pusht'),
    ('HuggingFaceVLA/libero', '/content/raw/lerobot_libero'),
):
    print('[DL]', repo_id, flush=True)
    if 'libero' in repo_id:
        snapshot_download(repo_id, repo_type='dataset', local_dir=dst, allow_patterns=['meta/*', 'data/chunk-000/file-00[0-3].parquet', 'videos/observation.images.image/chunk-000/file-00[0-3].mp4'])
    else:
        snapshot_download(repo_id, repo_type='dataset', local_dir=dst)
sys.argv = ['scripts/data/v3_preencode_av_codecs.py', '--corpus', 'robotics_lerobot', '--raw-root', '/content/raw', '--out-dir', '/content/out_rob', '--device', 'cuda', '--max-clips', '{args.max_lerobot_eps}']
from scripts.data import v3_preencode_av_codecs
v3_preencode_av_codecs.main()
print('ROB_DONE', flush=True)
"""
            rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                session_name, rob_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix="01_robotics",
                bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                t_alloc_start=t_alloc, phase_timeout_s=1800, exec_timeout_s=1500,
            )
            print(out[-2000:], flush=True)
            if "ROB_DONE" in out:
                for fn in ("robotics_train.npz", "robotics_dev.npz", "robotics_test.npz", "robotics_lerobot_preencode_receipt.json"):
                    dl = run_colab("download", "-s", session_name, f"/content/out_rob/{fn}", str(args.out_dir / fn), cfg_path=cfg_path, timeout=300)
                    if dl.returncode == 0:
                        downloaded_files.append(fn)

        if "msrvtt" in corpora:
            ms_py = f"""
import os, sys
from huggingface_hub import snapshot_download
os.chdir('/content/slwm_v3')
sys.path.insert(0, '/content/slwm_v3')
print('[DL] MSR-VTT', flush=True)
snapshot_download('friedrichor/MSR-VTT', repo_type='dataset', local_dir='/content/raw/msrvtt', allow_patterns=['MSRVTT_Videos.zip', 'msrvtt_train_9k.json', 'msrvtt_test_1k.json'])
sys.argv = ['scripts/data/v3_preencode_av_codecs.py', '--corpus', 'msrvtt', '--raw-root', '/content/raw', '--out-dir', '/content/out_av', '--device', 'cuda', '--max-clips', '{args.max_msrvtt}']
from scripts.data import v3_preencode_av_codecs
v3_preencode_av_codecs.main()
print('MSRVTT_DONE', flush=True)
"""
            rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                session_name, ms_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix="02_msrvtt",
                bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                t_alloc_start=t_alloc, phase_timeout_s=2400, exec_timeout_s=2100,
            )
            print(out[-2000:], flush=True)
            if "MSRVTT_DONE" in out:
                for fn in ("msrvtt_train.npz", "msrvtt_dev.npz", "msrvtt_test.npz", "msrvtt_preencode_receipt.json"):
                    dl = run_colab("download", "-s", session_name, f"/content/out_av/{fn}", str(args.out_dir / fn), cfg_path=cfg_path, timeout=300)
                    if dl.returncode == 0:
                        downloaded_files.append(fn)

        if "vggsound" in corpora:
            vgg_py = f"""
import os, sys
from huggingface_hub import hf_hub_download
os.chdir('/content/slwm_v3')
sys.path.insert(0, '/content/slwm_v3')
os.makedirs('/content/large_av/vggsound_subset', exist_ok=True)
print('[DL] VGGSound csv + shard 00', flush=True)
hf_hub_download('Loie/VGGSound', 'vggsound.csv', repo_type='dataset', local_dir='/content/large_av/vggsound_subset')
hf_hub_download('Loie/VGGSound', 'vggsound_00.tar.gz', repo_type='dataset', local_dir='/content/large_av/vggsound_subset')
sys.argv = ['scripts/data/v3_preencode_av_codecs.py', '--corpus', 'vggsound', '--large-av-root', '/content/large_av', '--out-dir', '/content/out_av', '--device', 'cuda', '--max-clips', '{args.max_vggsound}']
from scripts.data import v3_preencode_av_codecs
v3_preencode_av_codecs.main()
print('VGG_DONE', flush=True)
"""
            rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                session_name, vgg_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix="03_vggsound",
                bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                t_alloc_start=t_alloc, phase_timeout_s=2400, exec_timeout_s=2100,
            )
            print(out[-2000:], flush=True)
            if "VGG_DONE" in out:
                for fn in ("vggsound_train.npz", "vggsound_dev.npz", "vggsound_test.npz", "vggsound_preencode_receipt.json"):
                    dl = run_colab("download", "-s", session_name, f"/content/out_av/{fn}", str(args.out_dir / fn), cfg_path=cfg_path, timeout=300)
                    if dl.returncode == 0:
                        downloaded_files.append(fn)

        print(f"[PREENCODE_GPU] Downloaded files: {downloaded_files}", flush=True)
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
            (args.out_dir / "gpu_preencode_teardown.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
