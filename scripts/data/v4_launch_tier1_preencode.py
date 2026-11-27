"""SLWM-v4 Gate 2: Colab GPU Parallel Pre-Encoding Launcher (Auto-Upload to Public `valueai-org` Datasets + Verified Auto-Teardown).

Reads the local HuggingFace OAuth token securely from `~/.cache/huggingface/token` (never logged),
stages `data/v4_*.py` + `scripts/data/v4_preencode_tier1.py` onto a Colab `T4` / `L4` / `A100` / `H100` VM,
runs streaming pre-encoding directly into the license-pure public `valueai-org/slwm-v4-*` dataset repos
(via `--hf-repo auto`), and guarantees verified auto-teardown (`0.00 CU/hr`) for newly allocated sessions.
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

BUNDLE_FILES = (
    "data/__init__.py",
    "data/v4_heartbeat_contract.py",
    "data/v4_embodiment_and_splits.py",
    "scripts/__init__.py",
    "scripts/data/__init__.py",
    "scripts/data/v4_valueai_registry.py",
    "scripts/data/v4_preencode_tier1.py",
    "scripts/data/v4_patch_agibot_and_aloha.py",
)


def _build_bundle(repo_root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in BUNDLE_FILES:
            data = b"" if rel.endswith("__init__.py") else (repo_root / rel).read_bytes()
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch Colab GPU SLWM-v4 Tier-1 pre-encoding worker.")
    ap.add_argument("--accel", type=str, default="l4", choices=("t4", "l4", "g4", "a100", "a100_std", "h100"))
    ap.add_argument("--corpora", type=str, required=True, help="Comma-separated list of corpus=hours (e.g. vggsound_full_sync=65,epic_kitchens_100_long_ego_av=35)")
    ap.add_argument("--max-episodes", type=int, default=0, help="If >0, cap episodes per corpus (for fast smoke test).")
    ap.add_argument("--split-filter", type=str, default="", choices=("", "train", "dev", "test"), help="Optional single-split recovery filter.")
    ap.add_argument("--hf-repo", type=str, default="auto")
    ap.add_argument("--migrate-from-repo", type=str, default="")
    ap.add_argument("--flush-mb", type=float, default=650.0)
    ap.add_argument("--session-tag", type=str, default="worker")
    ap.add_argument("--reuse-session", type=str, default="", help="Optional existing active Colab session name to reuse (still auto-stops on exit unless --keep-alive).")
    ap.add_argument("--attach-only", action="store_true", help="Attach to an already-running detached job on --reuse-session without killing or re-prepping.")
    ap.add_argument("--keep-alive", action="store_true", help="Do not stop session on exit (only for rapid smoke debugging).")
    ap.add_argument("--max-spend-cu", type=float, default=60.0)
    ap.add_argument("--max-wall-s", type=float, default=21600.0)
    args = ap.parse_args()

    hf_tok_path = Path.home() / ".cache/huggingface/token"
    if not hf_tok_path.is_file():
        raise FileNotFoundError(f"Missing HF token at {hf_tok_path}; run `hf auth login` first.")
    hf_token = hf_tok_path.read_text().strip()
    if len(hf_token) < 10:
        raise ValueError("HF token file appears empty/invalid.")

    jobs: list[tuple[str, float]] = []
    for item in args.corpora.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            c_name, val = item.split("=", 1)
            jobs.append((c_name.strip(), float(val.strip())))
        else:
            jobs.append((item, 1.0))

    cfg_path = _REPO_ROOT / ".ops/colab-config.json"
    log_dir = _REPO_ROOT / ".ops/v4_preencode_logs" / args.session_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    session_name = args.reuse_session.strip() or f"slwm_v4_enc_{args.session_tag}_{args.accel}_{int(time.time()) % 100000}"

    bal_start, rate_start = 0.0, 0.0
    for _att in range(1, 5):
        try:
            bal_start, rate_start = query_balance(cfg_path)
            break
        except Exception as _exc:
            if _att == 4:
                raise
            print(f"[V4_ENC WARN {_att}/4] query_balance transient error: {_exc}; retrying in 5s...", flush=True)
            time.sleep(5.0)
    print(f"[V4_ENC] accel={args.accel} session={session_name} reuse={bool(args.reuse_session.strip())} jobs={jobs} bal={bal_start:.2f} CU rate={rate_start:.2f}", flush=True)
    tar_bytes = _build_bundle(_REPO_ROOT)
    created = bool(args.reuse_session.strip())
    t_alloc = time.monotonic()

    try:
        if not args.reuse_session.strip():
            allocate_colab_session_safe(session_name, args.accel, cfg_path, max_attempts=5, sleep_s=15.0)
            created = True
            t_alloc = time.monotonic()

        if not args.attach_only:
            # Upload code bundle + private HF token file (never echoed to stdout!)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf_tmp:
                tf_tmp.write(tar_bytes)
                tmp_bundle = Path(tf_tmp.name)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".tok", delete=False) as tf_tok:
                tf_tok.write(hf_token)
                tmp_tok = Path(tf_tok.name)
            try:
                up1 = run_colab("upload", "-s", session_name, str(tmp_bundle), "/content/v4_enc_bundle.tar.gz", cfg_path=cfg_path, timeout=180)
                up2 = run_colab("upload", "-s", session_name, str(tmp_tok), "/content/.hf_token", cfg_path=cfg_path, timeout=120)
                if up1.returncode != 0 or up2.returncode != 0:
                    raise RuntimeError(f"Upload failed: {up1.stderr or up2.stderr}")
            finally:
                tmp_bundle.unlink(missing_ok=True)
                tmp_tok.unlink(missing_ok=True)

            prep_py = (
                "import os, subprocess, sys, tarfile\n"
                + ("subprocess.run(['pkill', '-9', '-f', 'v4_preencode_tier1.py'])\n" if args.reuse_session.strip() else "")
                + "for d in ('/content/slwm_v4', '/content/slwm_v3', '/content/out', '/content/v4_shards'): os.makedirs(d, exist_ok=True)\n"
                "os.chmod('/content/.hf_token', 0o600)\n"
                "with tarfile.open('/content/v4_enc_bundle.tar.gz', 'r:gz') as tf: tf.extractall('/content/slwm_v4')\n"
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers', 'transformers', 'accelerate', 'huggingface_hub', 'hf_transfer', 'pyarrow', 'scipy', 'soundfile'], check=True)\n"
                "import torch, shutil; du = shutil.disk_usage('/content')\n"
                "print('ENC_PREP_OK', torch.cuda.get_device_name(0), f'free_gb={du.free/1e9:.1f}', flush=True)\n"
            )
            rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                session_name, prep_py, cfg_path=cfg_path, log_dir=log_dir, log_prefix="00_prep",
                bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=args.max_spend_cu,
                t_alloc_start=t_alloc, phase_timeout_s=480, exec_timeout_s=420,
            )
            print(out, flush=True)
            if rc != 0 or tripped or "ENC_PREP_OK" not in out:
                raise RuntimeError(f"Prep failed rc={rc}: {err[-500:]}")

        remote_runner = f"""
import os, subprocess, sys
os.environ['HF_TOKEN'] = open('/content/.hf_token').read().strip()
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir('/content/slwm_v4')
jobs = {json.dumps(jobs)}
for corpus, val in jobs:
    if corpus == 'patch_agibot_and_aloha':
        cmd = [
            sys.executable, 'scripts/data/v4_patch_agibot_and_aloha.py',
            '--work-dir', '/content/v4_agibot_rebuilt',
            '--flush-mb', str({args.flush_mb}),
        ]
    else:
        cmd = [
            sys.executable, 'scripts/data/v4_preencode_tier1.py',
            '--corpus', corpus,
            '--hf-repo', {args.hf_repo!r},
            '--migrate-from-repo', {args.migrate_from_repo!r},
            '--out-dir', '/content/v4_shards',
            '--flush-mb', str({args.flush_mb}),
            '--max-episodes', str({args.max_episodes}),
            '--split-filter', {args.split_filter!r},
        ]
        if corpus == 'fineweb_edu_and_stack_code':
            cmd += ['--target-bytes-m', str(val)]
        else:
            cmd += ['--target-hours', str(val)]
    print('[WORKER RUN]', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd)
    print('[WORKER RC]', corpus, r.returncode, flush=True)
    if r.returncode != 0:
        raise RuntimeError(f'Corpus {{corpus}} failed rc={{r.returncode}}')
print('V4_ENC_WORKER_DONE', flush=True)
"""
        rc, tail, tripped = _run_detached_remote_script_with_polling(
            session_name, remote_runner, tag=f"v4_enc_{args.session_tag}", cfg_path=cfg_path, log_dir=log_dir,
            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc_start=t_alloc,
            poll_interval_s=30.0, max_wall_s=args.max_wall_s,
        )
        print(tail[-3000:], flush=True)
        if rc != 0 or tripped or "V4_ENC_WORKER_DONE" not in tail:
            raise RuntimeError(f"Worker {args.session_tag} failed rc={rc} tripped={tripped}")
        print(f"[V4_ENC OK] Worker {args.session_tag} finished cleanly and uploaded to {args.hf_repo}", flush=True)
    finally:
        if created and not args.keep_alive:
            verified_stop_session(session_name, cfg_path)


if __name__ == "__main__":
    main()
