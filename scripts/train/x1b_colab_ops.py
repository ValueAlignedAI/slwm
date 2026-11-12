"""Isolated Colab TPU v6e1 / GPU H100 Orchestrator & Direct Fedora Archiver for Study X1B.

Guarantees:
    - Uses isolated session state `.ops/colab_sessions.json` so concurrent G11 is never touched.
    - Prefers `--tpu v6e1` (~4.50 credits/hr, 32 GB HBM); falls back to `--gpu H100` (80 GB HBM) only if TPU capacity is unavailable.
    - Enforces hard compute-unit cap (`<= 500.0` spent credits).
    - Streams code/datasets and dual-HDD checkpoints over authenticated SSH pipes (zero Mac disk bloat, zero Google Drive contention).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = REPO_ROOT / ".ops"
SESSIONS_CONFIG = OPS_DIR / "colab_sessions.json"
BUDGET_LEDGER = OPS_DIR / "budget.json"
COLAB_BIN = REPO_ROOT / ".venv/bin/colab"
PRIVATE_COLAB = Path.home() / ".config/slwm-colab"
SSH_KEY = PRIVATE_COLAB / "session_ed25519"
KNOWN_HOSTS = OPS_DIR / "known_hosts"
FEDORA_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "db@192.168.1.200"]

BALANCE_RE = re.compile(r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)")
RATE_RE = re.compile(r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)")


def colab_env() -> dict[str, str]:
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "CLOUDSDK_CONFIG": str(PRIVATE_COLAB / "gcloud"),
        "COLAB_CLI_REQUEST_TIMEOUT": "360",
        "COLAB_CLI_ASSIGN_RETRIES": "1",
        "COLAB_CLI_RETRY_BACKOFF": "8",
    }


def session_config_path(session_name: str | None = None) -> Path:
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    if session_name:
        return OPS_DIR / f"session_{session_name}.json"
    return SESSIONS_CONFIG


@contextmanager
def ops_lock():
    """Serialize mutations to .ops/budget.json and Fedora helper scripts across parallel lanes."""
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = OPS_DIR / "colab_ops.lock"
    with lock_file.open("a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def run_colab_cli(*args: str, session_name: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cfg = session_config_path(session_name)
    cmd = [str(COLAB_BIN), "--config", str(cfg), "--auth", "adc", *args]
    return subprocess.run(cmd, env=colab_env(), text=True, capture_output=True, timeout=timeout, check=False)


def check_usage_and_budget(max_spend: float = 500.0) -> dict[str, Any]:
    with ops_lock():
        res = run_colab_cli("usage", timeout=60)
        if res.returncode != 0:
            raise RuntimeError(f"colab usage failed: {res.stderr or res.stdout}")
        txt = res.stdout + "\n" + res.stderr
        bal_m = BALANCE_RE.search(txt)
        rate_m = RATE_RE.search(txt)
        balance = float(bal_m.group(1)) if bal_m else 0.0
        rate = float(rate_m.group(1)) if rate_m else 0.0

        if BUDGET_LEDGER.is_file():
            ledger = json.loads(BUDGET_LEDGER.read_text())
        else:
            ledger = {
                "initial_balance": balance,
                "max_spend_units": max_spend,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sessions": {},
            }
        spent = round(float(ledger["initial_balance"]) - balance, 4)
        ledger["current_balance"] = balance
        ledger["current_rate_per_hr"] = rate
        ledger["estimated_total_account_delta"] = spent
        ledger["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        BUDGET_LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    if spent > max_spend:
        stop_all_x1b_sessions()
        raise RuntimeError(f"HARD BUDGET STOP: account delta {spent:.2f} > {max_spend:.2f} credits!")
    return ledger


def get_session_endpoint(session_name: str) -> str | None:
    res = run_colab_cli("sessions", session_name=session_name, timeout=60)
    for line in (res.stdout + "\n" + res.stderr).splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == f"[{session_name}]":
            return parts[1]
    return None


def allocate_accelerator(
    session_name: str,
    *,
    prefer_tpu: bool = True,
    tpu_retries: int = 2,
    allow_h100_fallback: bool = True,
    tpu_only: bool = False,
) -> dict[str, str]:
    """Allocate a named session preferring TPU v6e1 (~5.09 credits/hr), with H100/A100 fallback or TPU queue."""
    if not session_name.startswith("slwm-x1b-"):
        raise ValueError(f"Refusing non-isolated session name {session_name!r}")
    check_usage_and_budget()

    ep = get_session_endpoint(session_name)
    if ep:
        _record_session(session_name, "reused", ep)
        return {"session": session_name, "endpoint": ep, "status": "reused"}

    for wave in range(1, 200):  # Up to ~150 min slot-wait when queuing onto single-TPU quota slot
        check_usage_and_budget()
        refused_cap = False
        if prefer_tpu or tpu_only:
            attempts = 1 if tpu_only else tpu_retries
            for attempt in range(1, attempts + 1):
                print(f"[ALLOC] Requesting TPU v6e1 for {session_name} (wave {wave}, attempt {attempt}/{attempts})...", flush=True)
                res = run_colab_cli("new", "-s", session_name, "--tpu", "v6e1", session_name=session_name, timeout=780)
                if res.returncode == 0:
                    ep = get_session_endpoint(session_name) or "assigned"
                    _record_session(session_name, "tpu:v6e1", ep)
                    return {"session": session_name, "accelerator": "tpu:v6e1", "endpoint": ep}
                msg = (res.stderr or res.stdout).strip()
                print(f"[WARN] v6e1 request failed ({res.returncode}): {msg[-240:]}", flush=True)
                if any(k in msg.lower() for k in ("allocation refused", "precondition failed", "service unavailable")):
                    refused_cap = True
                    break
                if attempt < attempts:
                    time.sleep(6)

        if allow_h100_fallback and not tpu_only:
            alloc_lock = OPS_DIR / "colab_new_rpc.lock"
            with alloc_lock.open("a") as alf:
                fcntl.flock(alf, fcntl.LOCK_EX)
                try:
                    for gpu_kind in ("H100", "A100"):
                        print(f"[ALLOC] Requesting GPU {gpu_kind} for {session_name} (wave {wave})...", flush=True)
                        res = run_colab_cli("new", "-s", session_name, "--gpu", gpu_kind, session_name=session_name, timeout=540)
                        if res.returncode == 0:
                            ep = get_session_endpoint(session_name) or f"gpu-{gpu_kind.lower()}"
                            _record_session(session_name, f"gpu:{gpu_kind}", ep)
                            return {"session": session_name, "accelerator": f"gpu:{gpu_kind}", "endpoint": ep}
                        gmsg = (res.stderr or res.stdout).strip()
                        print(f"[WARN] GPU {gpu_kind} request failed ({res.returncode}): {gmsg[-240:]}", flush=True)
                        if any(k in gmsg.lower() for k in ("allocation refused", "precondition failed", "service unavailable", "aborted", "exhausted", "already")):
                            refused_cap = True
                finally:
                    fcntl.flock(alf, fcntl.LOCK_UN)

        if tpu_only or refused_cap or allow_h100_fallback:
            wait_s = 45 if tpu_only else 25
            print(f"[QUEUE] {session_name} waiting {wait_s}s before next allocation wave...", flush=True)
            time.sleep(wait_s)
            continue
        break

    raise RuntimeError(f"Failed to allocate accelerator for {session_name}")


def _record_session(name: str, accel: str, endpoint: str) -> None:
    ledger = check_usage_and_budget()
    with ops_lock():
        ledger = json.loads(BUDGET_LEDGER.read_text())
        ledger.setdefault("sessions", {})[name] = {
            "accelerator": accel,
            "endpoint": endpoint,
            "allocated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": "active",
        }
        BUDGET_LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def ssh_cmd_base(session_name: str) -> list[str]:
    endpoint = get_session_endpoint(session_name)
    if not endpoint:
        raise RuntimeError(f"Session {session_name} has no active Colab assignment")
    cfg_p = session_config_path(session_name)
    return [
        str(COLAB_BIN),
        "--config",
        str(cfg_p),
        "--auth",
        "adc",
        "ssh",
        "-s",
        session_name,
        "-i",
        str(SSH_KEY),
        "--",
        "bash",
        "-lc",
    ]


def exec_python_in_kernel(session_name: str, code: str, *, timeout: int = 5400) -> subprocess.CompletedProcess[str]:
    """Execute Python directly inside the Colab session kernel (which holds the warmed-up v6e-1 TPU)."""
    cfg_p = session_config_path(session_name)
    cmd = [
        str(COLAB_BIN),
        "--config",
        str(cfg_p),
        "--auth",
        "adc",
        "exec",
        "-s",
        session_name,
        "--timeout",
        "3600",
    ]
    return subprocess.run(cmd, input=code, env=colab_env(), text=True, timeout=timeout, check=True)


def run_kernel_job_resilient(session_name: str, tag: str, code: str, *, timeout: int = 5400) -> None:
    """Execute a training/eval job inside the Colab kernel with automatic recovery from WebSocket drops."""
    for attempt in range(1, 4):
        try:
            exec_python_in_kernel(session_name, code, timeout=timeout)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"[RECOVER {session_name}] colab exec interrupted on attempt {attempt}/3 ({exc}); checking remote kernel state...", flush=True)
            base = ssh_cmd_base(session_name)
            # If kernel continues executing after WebSocket drop, monitor progress until done.json appears or stalls!
            for _ in range(45):
                chk = subprocess.run(
                    [
                        *base,
                        f"if [ -f /content/x1b-out/{shlex.quote(tag)}/done.json ]; then echo DONE; "
                        f"elif [ -f /content/x1b-out/{shlex.quote(tag)}-progress.json ]; then "
                        f"echo PROG:$(( $(date +%s) - $(stat -c %Y /content/x1b-out/{shlex.quote(tag)}-progress.json) )); "
                        f"else echo NONE; fi",
                    ],
                    env=colab_env(),
                    text=True,
                    capture_output=True,
                    timeout=45,
                )
                st_txt = chk.stdout.strip()
                if st_txt == "DONE":
                    print(f"[RECOVER {session_name}] {tag} finished cleanly on remote kernel!", flush=True)
                    return
                if st_txt.startswith("PROG:"):
                    age = int(st_txt.split(":", 1)[1])
                    if age < 600:
                        time.sleep(20)
                        continue
                break
            if attempt < 3:
                print(f"[RECOVER {session_name}] Restarting kernel & re-invoking {tag} (resumes eval if weights saved)...", flush=True)
                restart_session_kernel(session_name)
                time.sleep(5)
            else:
                raise


def restart_session_kernel(session_name: str) -> None:
    """Restart the Colab session kernel between jobs to release 100% of TPU/GPU HBM cleanly."""
    run_colab_cli("restart-kernel", "-s", session_name, session_name=session_name, timeout=120)


def stage_code_and_data(session_name: str) -> None:
    """Bundle X1B Python modules + configs and stream Fedora's verified real dataset bank to Colab."""
    # 1. Verify real TPU/GPU JAX backend directly inside the Colab kernel & create remote dirs
    verify_py = """
import os, pathlib, jax, ml_dtypes
for d in ('/content/x1b-code', '/content/x1b-data', '/content/x1b-out', '/content/x1b-cache'):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
b = jax.default_backend()
assert b in ('tpu', 'gpu'), f'Expected tpu/gpu backend, got {b}'
devs = [(d.device_kind, d.memory_stats().get('bytes_limit', 0) if d.memory_stats() else 0) for d in jax.devices()]
print('REMOTE_KERNEL_JAX_OK', jax.__version__, b, devs, flush=True)
"""
    exec_python_in_kernel(session_name, verify_py, timeout=180)
    base = ssh_cmd_base(session_name)

    # 2. Upload code tarball (< 200 KB)
    buf_path = OPS_DIR / f".code-{session_name}.tar.gz"
    include_paths = [
        "configs/x1b",
        "models/x1b_world.py",
        "models/x1b_jepa.py",
        "models/x1b_commit.py",
        "data/x1b_multimodal.py",
        "evals/x1b_mpc.py",
        "evals/x1b_eval_suite.py",
        "scripts/train/x1b_train.py",
        "scripts/train/x1b_storage.py",
    ]
    with tarfile.open(buf_path, "w:gz") as tf:
        for rel in include_paths:
            tf.add(REPO_ROOT / rel, arcname=rel)
    for up_try in range(1, 7):
        try:
            with buf_path.open("rb") as f:
                subprocess.run([*base, "tar -xzf - -C /content/x1b-code"], env=colab_env(), stdin=f, check=True, timeout=90)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if up_try == 6:
                buf_path.unlink(missing_ok=True)
                raise
            print(f"[WARN] Code upload retry {up_try}/6 ({exc})", flush=True)
            time.sleep(1.5 * up_try)
    buf_path.unlink(missing_ok=True)
    time.sleep(0.8)

    # 3. Stream Fedora's real multimodal bank + PyMunk 64x64 panels directly Fedora -> Colab SSH (zero Mac disk write!)
    card_raw = subprocess.check_output(
        [*FEDORA_SSH, "cat /mnt/slwm-data/SLWM/x1b/datasets/x1b_real_bank_v1.card.json"],
        text=True,
        timeout=30,
    )
    card = json.loads(card_raw)
    for fname, exp_sha in (
        ("x1b_real_bank_v1.npz", card["sha256"]),
        ("x1b_pymunk_panels_64.npz", card["pymunk_panels_sha256"]),
    ):
        chk = subprocess.run(
            [*base, f"test -f /content/x1b-data/{fname} && echo $(sha256sum /content/x1b-data/{fname} | awk '{{print $1}}')"],
            env=colab_env(),
            text=True,
            capture_output=True,
            timeout=45,
        )
        time.sleep(0.8)
        if chk.returncode == 0 and chk.stdout.strip() == exp_sha:
            print(f"[DATA {session_name}] {fname} already verified ({exp_sha[:12]})", flush=True)
            continue
        for ds_try in range(1, 7):
            print(f"[DATA {session_name}] Streaming {fname} from Fedora -> Colab (try {ds_try}/6)...", flush=True)
            fed_p = subprocess.Popen(
                [*FEDORA_SSH, f"cat /mnt/slwm-data/SLWM/x1b/datasets/{fname}"],
                stdout=subprocess.PIPE,
            )
            col_p = subprocess.Popen(
                [
                    *base,
                    f"cat > /content/x1b-data/{fname}.tmp && "
                    f"echo '{exp_sha}  /content/x1b-data/{fname}.tmp' | sha256sum -c - && "
                    f"mv /content/x1b-data/{fname}.tmp /content/x1b-data/{fname}",
                ],
                env=colab_env(),
                stdin=fed_p.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert fed_p.stdout is not None
            fed_p.stdout.close()
            out_s, err_s = col_p.communicate(timeout=300)
            fed_rc = fed_p.wait(timeout=30)
            time.sleep(0.8)
            if fed_rc == 0 and col_p.returncode == 0:
                print(f"[DATA {session_name}] Verified {fname} ({exp_sha[:12]})", flush=True)
                break
            if ds_try == 6:
                raise RuntimeError(f"Dataset stream failed for {fname}: {err_s or out_s}")
            print(f"[WARN] Dataset stream retry {ds_try}/6 for {fname}: {(err_s or out_s).strip()}", flush=True)
            time.sleep(1.8 * ds_try)


def archive_run_to_fedora(session_name: str, tag: str) -> dict[str, Any]:
    """Stream completed run artifacts from Colab -> Fedora primary HDD -> mirror & verify on secondary HDD."""
    # Ensure latest x1b_storage.py is installed atomically on Fedora under ops_lock
    with ops_lock():
        subprocess.run(
            [
                *FEDORA_SSH,
                "cat > /home/db/dev/slwm-x1b/.x1b_storage.py.tmp && mv /home/db/dev/slwm-x1b/.x1b_storage.py.tmp /home/db/dev/slwm-x1b/x1b_storage.py",
            ],
            input=(REPO_ROOT / "scripts/train/x1b_storage.py").read_text(),
            text=True,
            check=True,
            timeout=30,
        )
    base = ssh_cmd_base(session_name)
    rec_raw = subprocess.check_output(
        [*base, f"cat /content/x1b-out/{shlex.quote(tag)}/receipt.json"],
        env=colab_env(),
        text=True,
        timeout=60,
    )
    receipt = json.loads(rec_raw)

    # Pull lightweight JSON/JSONL manifests + portable_int8.npz (+ weights_bf16.npz if present)
    file_list_py = f"""
import hashlib, json, pathlib
d = pathlib.Path('/content/x1b-out/{tag}')
items = []
for name in ['receipt.json', 'eval_metrics.json', 'config.json', 'train.jsonl', 'portable_int8.npz', 'weights_bf16.npz']:
    p = d / name
    if p.is_file():
        with p.open('rb') as f:
            h = hashlib.file_digest(f, 'sha256').hexdigest()
        items.append([name, h, p.stat().st_size])
print(json.dumps(items))
"""
    manifest_raw = subprocess.check_output(
        [*base, f"python -c {shlex.quote(file_list_py)}"],
        env=colab_env(),
        text=True,
        timeout=120,
    )
    files_meta: list[list[Any]] = json.loads(manifest_raw.strip().splitlines()[-1])

    for fname, sha_hex, n_bytes in files_meta:
        for arch_try in range(1, 7):
            if arch_try > 1:
                time.sleep(2.5 * arch_try)
            else:
                time.sleep(0.6)
            print(
                f"[ARCHIVE {session_name}] Streaming {tag}/{fname} ({n_bytes / 1e6:.2f} MB) -> Fedora (try {arch_try}/6)...",
                flush=True,
            )
            src_p = subprocess.Popen(
                [*base, f"cat /content/x1b-out/{shlex.quote(tag)}/{shlex.quote(fname)}"],
                env=colab_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            dst_p = subprocess.Popen(
                [
                    *FEDORA_SSH,
                    f"/home/db/dev/slwm-x1b/.venv/bin/python /home/db/dev/slwm-x1b/x1b_storage.py receive "
                    f"{shlex.quote(tag)} {shlex.quote(fname)} {shlex.quote(sha_hex)} {int(n_bytes)}",
                ],
                stdin=src_p.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert src_p.stdout is not None
            src_p.stdout.close()
            out_s, err_s = dst_p.communicate(timeout=900)
            src_err_b = src_p.stderr.read() if src_p.stderr is not None else b""
            src_rc = src_p.wait(timeout=30)
            if src_rc == 0 and dst_p.returncode == 0:
                break
            src_err_s = src_err_b.decode("utf-8", errors="replace").strip()
            print(
                f"[WARN] Archive stream attempt {arch_try}/6 failed on {tag}/{fname}: src_rc={src_rc} ({src_err_s[-180:]}) dst={err_s[-180:]}",
                flush=True,
            )
            if arch_try == 6:
                raise RuntimeError(f"Archive stream failed on {tag}/{fname}: {src_err_s or err_s or out_s}")

    mir_out = subprocess.check_output(
        [
            *FEDORA_SSH,
            f"/home/db/dev/slwm-x1b/.venv/bin/python /home/db/dev/slwm-x1b/x1b_storage.py mirror {shlex.quote(tag)}",
        ],
        text=True,
        timeout=600,
    )
    print(f"[MIRROR {tag}] {mir_out.strip()}", flush=True)

    # Also save small JSON metrics locally in experiments/multimodal/x1b/results/<tag>/ for plotting & git!
    local_res_dir = REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111/runs" / tag
    local_res_dir.mkdir(parents=True, exist_ok=True)
    for small in ("receipt.json", "eval_metrics.json", "config.json", "train.jsonl"):
        txt = subprocess.check_output([*FEDORA_SSH, f"cat /mnt/slwm-checkpoints-a/SLWM/x1b/runs/{shlex.quote(tag)}/{small}"], text=True, timeout=60)
        (local_res_dir / small).write_text(txt)
    ver_txt = subprocess.check_output([*FEDORA_SSH, f"cat /mnt/slwm-checkpoints-a/SLWM/x1b/runs/{shlex.quote(tag)}/verified.json"], text=True, timeout=60)
    (local_res_dir / "verified.json").write_text(ver_txt)
    return receipt


def stop_session(session_name: str) -> None:
    if not session_name.startswith("slwm-x1b-"):
        raise ValueError(f"Refusing to stop non-X1B session {session_name!r}")
    res = run_colab_cli("stop", "-s", session_name, session_name=session_name, timeout=90)
    with ops_lock():
        if BUDGET_LEDGER.is_file():
            ledger = json.loads(BUDGET_LEDGER.read_text())
            if session_name in ledger.get("sessions", {}):
                ledger["sessions"][session_name]["state"] = "stopped"
                ledger["sessions"][session_name]["stopped_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                BUDGET_LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"[STOP {session_name}] rc={res.returncode} {(res.stdout or res.stderr).strip()}", flush=True)


def stop_all_x1b_sessions() -> None:
    seen: set[str] = set()
    for cfg in sorted(OPS_DIR.glob("session_slwm-x1b-*.json")):
        sname = cfg.stem.removeprefix("session_")
        if sname.startswith("slwm-x1b-"):
            seen.add(sname)
            if get_session_endpoint(sname):
                stop_session(sname)
    if BUDGET_LEDGER.is_file():
        ledger = json.loads(BUDGET_LEDGER.read_text())
        for sname in ledger.get("sessions", {}):
            if sname.startswith("slwm-x1b-") and sname not in seen and get_session_endpoint(sname):
                stop_session(sname)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("budget")
    p_alloc = sub.add_parser("alloc")
    p_alloc.add_argument("session")
    p_alloc.add_argument("--gpu-only", action="store_true")

    p_stage = sub.add_parser("stage")
    p_stage.add_argument("session")

    p_arch = sub.add_parser("archive")
    p_arch.add_argument("session")
    p_arch.add_argument("tag")

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("session")

    sub.add_parser("stop-all")

    args = ap.parse_args()
    if args.cmd == "budget":
        print(json.dumps(check_usage_and_budget(), indent=2))
    elif args.cmd == "alloc":
        print(json.dumps(allocate_accelerator(args.session, prefer_tpu=(not args.gpu_only)), indent=2))
    elif args.cmd == "stage":
        stage_code_and_data(args.session)
    elif args.cmd == "archive":
        print(json.dumps(archive_run_to_fedora(args.session, args.tag), indent=2))
    elif args.cmd == "stop":
        stop_session(args.session)
    elif args.cmd == "stop-all":
        stop_all_x1b_sessions()


if __name__ == "__main__":
    main()
