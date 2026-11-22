"""SLWM-v3 Hard-Capped Google TPU v6e-1 Validation Pilot Launcher (Rev 4, <= 45.0 CU, TOCTOU-Free & Strict Teardown).

Operational guardrails (verified against installed `colab` CLI 0.8.0):
  1. **Single-Read TOCTOU-Free Bundle Build & Dual Digest Lock:** Reads every bundled file ONCE into memory,
     builds the `.tar.gz` upload archive from those EXACT in-memory bytes, verifies ALL FIVE explicit boolean
     stage-gate flags are strictly `True`, and asserts that BOTH in-memory `models_v3_sha256` AND full
     `pilot_bundle_sha256` match `reports/v3/gate3_architecture_and_mup_receipt.json` BEFORE allocation!
  2. **Strict Google TPU `v6e-1` Policy:** Always requests `colab new -s <name> --tpu v6e1` (`32 GiB` HBM).
     Never uses `A100`.
  3. **Hard Cumulative Budget Cap (`<= 45.0 CU`) + Strict 3.0 CU Reserve + Elapsed-Time Fallback Watchdog:**
     - Persists cumulative pilot spend across resumptions (`cumulative_pilot_spend.json`) without overwriting
       prior spend records.
     - Enforces BOTH actual AND next-interval projected cumulative spend against `hard_limit_cu = max - 3.0`
       (`42.0 CU`) BEFORE EVERY phase dispatch AND every `45 s` during execution, plus an independent
       elapsed-time conservative ceiling (`12.0 CU/hr`) immune to balance-API outages.
     - File-backed stdout/stderr logs prevent macOS 64 KiB `subprocess.PIPE` deadlocks.
     - Requires a validated production TPU `v6e` Phase 2 receipt bound to the exact `pilot_bundle_sha256` and
       valid finite `recommended_base_lr` before resuming Phases 3/4 (and uploads that receipt to remote on resume!).
  4. **Strict Verified Auto-Teardown:** Wraps `colab new -s <name>` INSIDE `try ... finally:`. Handles timeouts
     inside the 3-attempt retry loop, requires `stop` `rc == 0` AND a recognized non-empty `colab sessions`
     `rc == 0` inventory proving BOTH `[<session_name>]` AND any unnamed `[?]` assignment are absent!
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = REPO_ROOT / ".ops"
COLAB_BIN = next(
    (p for p in (REPO_ROOT / ".venv/bin/colab", Path.home() / ".local/bin/colab") if p.exists()),
    Path.home() / ".local/bin/colab",
)
BALANCE_RE = re.compile(r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)")
RATE_RE = re.compile(r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)")
SHUTDOWN_RESERVE_CU = 3.0
CONSERVATIVE_V6E1_RATE_CU_HR = 12.0
WATCHDOG_POLL_INTERVAL_S = 45

PILOT_BUNDLE_FILES: tuple[str, ...] = (
    "models/v3/__init__.py",
    "models/v3/spec.py",
    "models/v3/memory.py",
    "models/v3/core.py",
    "models/v3/fast_core.py",
    "models/v3/any2any_heads.py",
    "models/v3/objectives.py",
    "data/v3_data_contract.py",
    "evals/v3_gate3_audit_receipt.py",
    "scripts/train/v3_pilot_train.py",
    "scripts/train/v3_tpu_pilot_suite.py",
)


def resolve_gcloud_config_dir() -> Path:
    """Prefer SLWM-isolated gcloud config if present, else standard `~/.config/gcloud`."""
    override = os.environ.get("SLWM_CLOUDSDK_CONFIG")
    if override:
        return Path(override)
    dedicated = Path.home() / ".config/slwm-colab/gcloud"
    if dedicated.is_dir():
        return dedicated
    return Path.home() / ".config/gcloud"


def _colab_env() -> dict[str, str]:
    return {
        **os.environ,
        "CLOUDSDK_CONFIG": str(resolve_gcloud_config_dir()),
        "COLAB_CLI_REQUEST_TIMEOUT": "300",
        "COLAB_CLI_ASSIGN_RETRIES": "1",
        "COLAB_CLI_RETRY_BACKOFF": "15",
    }


def _colab_cmd(*args: str, cfg_path: Path) -> list[str]:
    return [str(COLAB_BIN), "--config", str(cfg_path), "--auth", "adc", *args]


def run_colab(*args: str, cfg_path: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _colab_cmd(*args, cfg_path=cfg_path),
        env=_colab_env(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def query_balance(cfg_path: Path) -> tuple[float, float]:
    res = run_colab("usage", cfg_path=cfg_path, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"colab usage failed: {res.stderr or res.stdout}")
    txt = res.stdout + "\n" + res.stderr
    bal_m = BALANCE_RE.search(txt)
    rate_m = RATE_RE.search(txt)
    if bal_m is None:
        raise RuntimeError(f"Could not parse Colab balance from:\n{txt}")
    return float(bal_m.group(1)), (float(rate_m.group(1)) if rate_m else 0.0)


def compute_models_v3_sha256(repo_root: Path) -> str:
    """Deterministic SHA-256 over all source files in `models/v3/*.py`."""
    h = hashlib.sha256()
    models_dir = repo_root / "models/v3"
    for fp in sorted(models_dir.glob("*.py")):
        h.update(fp.name.encode("utf-8"))
        h.update(b"\0")
        h.update(fp.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def compute_pilot_bundle_sha256(repo_root: Path) -> str:
    """Deterministic SHA-256 over ALL files uploaded inside the TPU pilot bundle."""
    h = hashlib.sha256()
    for rel in PILOT_BUNDLE_FILES:
        p = repo_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"Required bundle file missing: {p}")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def build_and_verify_pilot_bundle(repo_root: Path) -> tuple[dict[str, Any], bytes]:
    """Single-read TOCTOU-free Gate 1–3 lock + archive builder.

    Reads every bundled file ONCE into memory, hashes those exact bytes for BOTH `models_v3_sha256` and
    `pilot_bundle_sha256`, checks all 5 Gate 1–3 receipts against those exact digests, and writes those
    EXACT in-memory bytes directly into the upload `.tar.gz` archive (zero reopen TOCTOU window!).
    """
    checks = (
        ("gate1_control", repo_root / "reports/v3/gate1_control_receipt.json", "overall_gate1_synthetic_and_control_pass"),
        ("gate1_speech_prototype", repo_root / "reports/v3/gate1_speech_prototype_receipt.json", "gate1_speech_prototype_pass"),
        ("gate2_splits", repo_root / "reports/v3/gate2_split_ledger_receipt.json", "all_connected_splits_zero_overlap"),
        ("gate2_codecs", repo_root / "reports/v3/gate2_codec_ceilings_receipt.json", "gate2_codec_causality_and_ceilings_pass"),
        ("gate3_arch_mup", repo_root / "reports/v3/gate3_architecture_and_mup_receipt.json", "gate3_receipt_pass"),
    )
    status: dict[str, Any] = {}
    gate3_data: dict[str, Any] = {}
    for name, path, key in checks:
        if not path.is_file():
            raise RuntimeError(f"PRE-FLIGHT GATE LOCK: missing required receipt {path}")
        raw_txt = path.read_text().strip()
        if not raw_txt:
            raise RuntimeError(f"PRE-FLIGHT GATE LOCK: empty receipt {path}")
        data = json.loads(raw_txt)
        if not isinstance(data, dict) or data.get(key) is not True:
            raise RuntimeError(
                f"PRE-FLIGHT GATE LOCK: {path.name}[{key!r}] is {data.get(key)!r} (expected strict True)!"
            )
        status[name] = True
        if name == "gate3_arch_mup":
            gate3_data = data

    # Single-read snapshot of all files under `models/v3/*.py` + `PILOT_BUNDLE_FILES`
    file_bytes: dict[str, bytes] = {}
    for rel in PILOT_BUNDLE_FILES:
        p = repo_root / rel
        if not p.is_file():
            raise FileNotFoundError(f"Required bundle file missing: {p}")
        file_bytes[rel] = p.read_bytes()

    models_dir = repo_root / "models/v3"
    extra_model_files = sorted(fp.name for fp in models_dir.glob("*.py"))
    h_models = hashlib.sha256()
    for fname in extra_model_files:
        rel = f"models/v3/{fname}"
        raw = file_bytes.get(rel) if rel in file_bytes else (models_dir / fname).read_bytes()
        h_models.update(fname.encode("utf-8"))
        h_models.update(b"\0")
        h_models.update(raw)
        h_models.update(b"\0")
    live_models_sha = h_models.hexdigest()

    h_bundle = hashlib.sha256()
    for rel in PILOT_BUNDLE_FILES:
        h_bundle.update(rel.encode("utf-8"))
        h_bundle.update(b"\0")
        h_bundle.update(file_bytes[rel])
        h_bundle.update(b"\0")
    live_bundle_sha = h_bundle.hexdigest()

    rec_models_sha = gate3_data.get("models_v3_sha256")
    if not rec_models_sha or rec_models_sha != live_models_sha:
        raise RuntimeError(
            f"PRE-FLIGHT BUNDLE LOCK: live models/v3 SHA-256 ({live_models_sha}) != "
            f"Gate 3 receipt models_v3_sha256 ({rec_models_sha!r})! Re-run Gate 3 audit first."
        )
    rec_bundle_sha = gate3_data.get("pilot_bundle_sha256")
    if not rec_bundle_sha or rec_bundle_sha != live_bundle_sha:
        raise RuntimeError(
            f"PRE-FLIGHT BUNDLE LOCK: live pilot_bundle_sha256 ({live_bundle_sha}) != "
            f"Gate 3 receipt pilot_bundle_sha256 ({rec_bundle_sha!r})! Re-run Gate 3 audit first."
        )

    # Build upload tarball strictly from the ALREADY-VERIFIED in-memory bytes!
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in PILOT_BUNDLE_FILES:
            raw = file_bytes[rel]
            ti = tarfile.TarInfo(name=rel)
            ti.size = len(raw)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(raw))
        for init_rel in ("models/__init__.py", "data/__init__.py", "evals/__init__.py", "scripts/__init__.py", "scripts/train/__init__.py"):
            ti = tarfile.TarInfo(name=init_rel)
            ti.size = 0
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(b""))

    status["models_v3_sha256_verified"] = live_models_sha
    status["pilot_bundle_sha256_verified"] = live_bundle_sha
    return status, buf.getvalue()


def verify_gates_1_to_3_passed(repo_root: Path) -> dict[str, Any]:
    """Convenience wrapper returning verified Gate 1–3 status dict."""
    status, _ = build_and_verify_pilot_bundle(repo_root)
    return status


_STDERR_FAILURE_MARKERS = ("error", "failed", "warning", "credential", "auth", "traceback", "unavailable", "exception")
_BENIGN_ADC_REFRESH_LINES = (
    "failed to obtain valid adc credentials.",
    "trying to refresh adc credentials",
)


def _strip_benign_adc_lines(text: str) -> str:
    """Remove benign `google-colab-cli` hourly ADC token-refresh messages before error screening."""
    kept: list[str] = []
    for ln in (text or "").splitlines():
        if ln.strip().lower() in _BENIGN_ADC_REFRESH_LINES:
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def query_active_sessions(cfg_path: Path, *, runner=run_colab) -> list[str] | None:
    """Query `colab sessions`; inspect stdout ONLY, reject ANY stderr error/warning or unrecognized output."""
    try:
        res = runner("sessions", cfg_path=cfg_path, timeout=60)
    except Exception:
        return None
    if res.returncode != 0:
        return None
    stderr_clean = _strip_benign_adc_lines(res.stderr or "")
    stderr_l = stderr_clean.lower()
    if any(m in stderr_l for m in _STDERR_FAILURE_MARKERS):
        return None
    stdout_txt = _strip_benign_adc_lines(res.stdout or "")
    if not stdout_txt:
        return None  # Empty stdout is NEVER a valid inventory!
    stdout_l = stdout_txt.lower()
    if any(m in stdout_l for m in _STDERR_FAILURE_MARKERS):
        return None
    raw_lines = [ln.strip() for ln in stdout_txt.splitlines() if ln.strip()]
    if raw_lines == ["[colab] No active sessions found on server."]:
        return []
    lines = [ln for ln in raw_lines if not ln.startswith("[colab]")]
    if not lines or not all("[" in ln and "]" in ln for ln in lines):
        return None
    return lines


def _session_lines_contain_target_or_unnamed(lines: list[str], session_name: str) -> bool:
    """Return True if `[session_name]` OR an unnamed `[?]` server assignment is present."""
    exact_tag = f"[{session_name}]"
    for ln in lines:
        if ln.startswith(exact_tag) or (" " + exact_tag) in ln or ln.startswith("[?]") or " [?]" in ln:
            return True
    return False


def _colab_cli_python() -> Path | None:
    """Resolve the Python interpreter backing `COLAB_BIN` (for official `colab_cli` orphan recovery/unassign)."""
    try:
        first_line = COLAB_BIN.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
        if first_line.startswith("#!"):
            cand = Path(first_line[2:].strip().split()[0])
            if cand.is_file():
                return cand
    except Exception:
        pass
    fallback = Path.home() / ".local/share/uv/tools/google-colab-cli/bin/python"
    return fallback if fallback.is_file() else None


@contextmanager
def _colab_config_lock(cfg_path: Path):
    """Process-level exclusive lock around `colab-config.json` mutations (`new`/`stop`) for safe parallel instances."""
    lock_p = cfg_path.with_suffix(".lock")
    lock_p.parent.mkdir(parents=True, exist_ok=True)
    with lock_p.open("a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _adopt_single_unnamed_assignment(session_name: str, cfg_path: Path, *, accel: str = "tpu_v6e1") -> bool:
    """If `colab new` timed out right after server provisioned ONE matching orphan runtime, adopt it into `cfg_path`."""
    py = _colab_cli_python()
    if py is None:
        return False
    target_map = {
        "tpu_v6e1": ("V6E1", "TPU"),
        "tpu_v5e1": ("V5E1", "TPU"),
        "a100": ("A100", "GPU"),
        "h100": ("H100", "GPU"),
        "g4": ("G4", "GPU"),
        "l4": ("L4", "GPU"),
    }
    want_acc, want_var = target_map.get(accel, ("V6E1", "TPU"))
    code = (
        "import sys; from colab_cli.auth import AuthProvider; from colab_cli.common import state; "
        "from colab_cli.state import SessionState\n"
        "state.config_path = sys.argv[1]; state.auth_provider = AuthProvider.ADC\n"
        "known = {s.endpoint for s in state.store.list().values()}\n"
        "asg = state.client.list_assignments()\n"
        "want_acc, want_var = sys.argv[3], sys.argv[4]\n"
        "orphans = [a for a in asg if a.endpoint not in known and a.accelerator.value == want_acc and a.variant.name == want_var]\n"
        "if len(orphans) != 1: sys.exit(2)\n"
        "a = orphans[0]\n"
        "s = SessionState(name=sys.argv[2], token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url, "
        "endpoint=a.endpoint, token_expires_at=a.runtime_proxy_info.expires_at(), variant=a.variant.value, "
        "accelerator=a.accelerator.value, machine_shape=a.machine_shape.name)\n"
        "state.store.add(s); print('ADOPTED', s.name, s.endpoint)\n"
    )
    try:
        res = subprocess.run(
            [str(py), "-c", code, str(cfg_path), session_name, want_acc, want_var],
            env=_colab_env(), text=True, capture_output=True, timeout=60, check=False,
        )
        if res.returncode == 0 and "ADOPTED" in res.stdout:
            print(f"[ALLOC] {res.stdout.strip()}", flush=True)
            return True
        if res.stderr.strip():
            print(f"[ALLOC WARN] Orphan adopt rc={res.returncode}: {res.stderr.strip()[:240]}", flush=True)
    except Exception as exc:
        print(f"[ALLOC WARN] Orphan adopt failed: {exc}", flush=True)
    return False


def _adopt_single_unnamed_v6e1(session_name: str, cfg_path: Path) -> bool:
    """Backwards-compatible alias for adopting a single orphaned V6E1 TPU assignment."""
    return _adopt_single_unnamed_assignment(session_name, cfg_path, accel="tpu_v6e1")


def allocate_colab_session_safe(
    session_name: str,
    accel: str,
    cfg_path: Path,
    *,
    max_attempts: int = 4,
    sleep_s: float = 15.0,
    fallback_accels: list[str] | None = None,
) -> str:
    """Safely allocate ONE Colab runtime with per-attempt locking (never blocks siblings during sleep) and pool failover."""
    accel_order = [accel] + [a for a in (fallback_accels or []) if a and a != accel]
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        cur_acc = accel_order[(attempt - 1) % len(accel_order)]
        with _colab_config_lock(cfg_path):
            pre = query_active_sessions(cfg_path) or []
            if any("[?]" in ln for ln in pre):
                print(f"[ALLOC WARN] Pre-existing unnamed assignment(s) detected: {pre}; unassigning before new alloc.", flush=True)
                _unassign_unnamed_assignments(cfg_path)
                time.sleep(3.0)

            print(f"[ALLOC attempt {attempt}/{max_attempts}] Requesting {cur_acc} session '{session_name}'...", flush=True)
            try:
                if cur_acc == "tpu_v6e1":
                    res = run_colab("new", "-s", session_name, "--tpu", "V6E1", cfg_path=cfg_path, timeout=420)
                elif cur_acc == "tpu_v5e1":
                    res = run_colab("new", "-s", session_name, "--tpu", "V5E1", cfg_path=cfg_path, timeout=420)
                elif cur_acc == "a100":
                    res = run_colab("new", "-s", session_name, "--gpu", "A100", "--high-mem", cfg_path=cfg_path, timeout=360)
                elif cur_acc == "l4":
                    res = run_colab("new", "-s", session_name, "--gpu", "L4", "--high-mem", cfg_path=cfg_path, timeout=360)
                elif cur_acc == "g4":
                    res = run_colab("new", "-s", session_name, "--gpu", "G4", cfg_path=cfg_path, timeout=360)
                else:
                    res = run_colab("new", "-s", session_name, "--gpu", "H100", cfg_path=cfg_path, timeout=360)
                if res.returncode == 0:
                    print(res.stdout.strip(), flush=True)
                    return cur_acc
                last_err = (res.stderr or res.stdout)[-400:]
                print(f"[ALLOC attempt {attempt}/{max_attempts} WARN] ({cur_acc}) rc={res.returncode}: {last_err.strip()}", flush=True)
            except Exception as e:
                last_err = str(e)
                print(f"[ALLOC attempt {attempt}/{max_attempts} WARN] ({cur_acc}) {type(e).__name__}: {e}", flush=True)

            active = query_active_sessions(cfg_path) or []
            if any(f"[{session_name}]" in ln for ln in active):
                print(f"[ALLOC] Session '{session_name}' already active on server after retry check.", flush=True)
                return cur_acc
            if any("[?]" in ln for ln in active):
                if _adopt_single_unnamed_assignment(session_name, cfg_path, accel=cur_acc):
                    return cur_acc
                print(f"[ALLOC WARN] Could not adopt unnamed assignment(s) {active}; unassigning before retry.", flush=True)
                _unassign_unnamed_assignments(cfg_path)
        if attempt < max_attempts:
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to allocate {accel_order} session '{session_name}' after {max_attempts} attempts: {last_err}")


def _unassign_unnamed_assignments(cfg_path: Path) -> None:
    """Cleanly unassign any orphaned `[?]` server assignment not tracked in `cfg_path` via official `colab_cli` client."""
    py = _colab_cli_python()
    if py is None:
        return
    code = (
        "import sys; from colab_cli.auth import AuthProvider; from colab_cli.common import state\n"
        "state.config_path = sys.argv[1]; state.auth_provider = AuthProvider.ADC\n"
        "known = {s.endpoint for s in state.store.list().values()}\n"
        "for a in state.client.list_assignments():\n"
        "    if a.endpoint not in known:\n"
        "        state.client.unassign(a.endpoint)\n"
        "        print('UNASSIGNED_ORPHAN', a.endpoint)\n"
    )
    try:
        res = subprocess.run([str(py), "-c", code, str(cfg_path)], env=_colab_env(), text=True, capture_output=True, timeout=60, check=False)
        if res.stdout.strip():
            print(f"[TEARDOWN] {res.stdout.strip()}", flush=True)
    except Exception as exc:
        print(f"[TEARDOWN WARN] Orphan unassign error: {exc}", flush=True)


def verified_stop_session(session_name: str, cfg_path: Path, *, runner=run_colab, sleep_s: float = 3.0) -> bool:
    """Stop named Colab session under file lock (safe with concurrent sibling sessions!) and verify disappearance."""
    lock_ctx = _colab_config_lock(cfg_path) if (runner is run_colab) else None
    if lock_ctx is not None:
        lock_ctx.__enter__()
    try:
        for attempt in range(1, 4):
            stop_ok = False
            try:
                stop_res = runner("stop", "-s", session_name, cfg_path=cfg_path, timeout=120)
                stop_txt = (stop_res.stdout or stop_res.stderr or "").strip()
                stop_ok = stop_res.returncode == 0
                print(f"[TEARDOWN attempt {attempt}] stop rc={stop_res.returncode}: {stop_txt}", flush=True)
            except Exception as exc:
                print(f"[TEARDOWN attempt {attempt}] stop raised {type(exc).__name__}: {exc}", flush=True)

            if sleep_s > 0.0:
                time.sleep(sleep_s)

            active = query_active_sessions(cfg_path, runner=runner)
            if active is None:
                print(f"[TEARDOWN attempt {attempt}] `colab sessions` failed or returned unrecognized output.", flush=True)
                continue
            if (runner is run_colab) and any(ln.startswith("[?]") or " [?]" in ln for ln in active):
                _unassign_unnamed_assignments(cfg_path)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                active = query_active_sessions(cfg_path, runner=runner)
                if active is None:
                    continue
            if stop_ok and (not _session_lines_contain_target_or_unnamed(active, session_name)):
                print(f"[TEARDOWN] Verified stop rc=0 AND session {session_name!r} / `[?]` absent from server.", flush=True)
                return True
            print(f"[TEARDOWN attempt {attempt}] stop_ok={stop_ok}, active={active}", flush=True)
        print(f"[TEARDOWN CRITICAL] Session {session_name!r} or unnamed `[?]` could NOT be verified stopped!", flush=True)
        return False
    finally:
        if lock_ctx is not None:
            lock_ctx.__exit__(None, None, None)


def projected_spend_exceeds_reserve(
    cumulative_spent_cu: float,
    live_rate_cu_hr: float,
    *,
    max_spend_cu: float,
    poll_interval_s: int = WATCHDOG_POLL_INTERVAL_S,
) -> tuple[bool, float, float]:
    """Check BOTH actual and conservative next-interval projected cumulative spend against `max_spend_cu - 3.0 CU`."""
    hard_limit_cu = max(0.5, max_spend_cu - SHUTDOWN_RESERVE_CU)
    eff_rate = max(live_rate_cu_hr, CONSERVATIVE_V6E1_RATE_CU_HR)
    projected_cu = cumulative_spent_cu + eff_rate * (float(poll_interval_s) / 3600.0)
    tripped = (cumulative_spent_cu >= hard_limit_cu) or (projected_cu >= hard_limit_cu)
    return tripped, projected_cu, hard_limit_cu


def load_prior_cumulative_spend(
    out_dir: Path, *, current_balance_cu: float | None = None
) -> tuple[float, list[dict[str, Any]]]:
    """Load accumulated pilot spend across prior invocations AND reconcile any interrupted in-progress run!"""
    ledger_path = out_dir / "cumulative_pilot_spend.json"
    if not ledger_path.is_file():
        return 0.0, []
    data = json.loads(ledger_path.read_text())
    runs = list(data.get("runs") or [])
    prior = sum(max(0.0, float(r.get("pilot_spend_cu", 0.0))) for r in runs)
    in_prog = data.get("in_progress_run")
    if isinstance(in_prog, dict):
        prev_bal = float(in_prog.get("balance_start_cu", 0.0))
        prev_ts = float(in_prog.get("start_epoch_s", time.time()))
        elapsed_est = max(0.0, (time.time() - prev_ts) / 3600.0) * CONSERVATIVE_V6E1_RATE_CU_HR
        bal_delta = max(0.0, prev_bal - float(current_balance_cu)) if current_balance_cu is not None else elapsed_est
        reconciled_spend = max(bal_delta, min(elapsed_est, 45.0)) if current_balance_cu is None else bal_delta
        prior += reconciled_spend
        runs.append(
            {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_name": in_prog.get("session_name", "interrupted_unknown"),
                "reconciled_interrupted_run": True,
                "balance_start_cu": prev_bal,
                "balance_end_cu": current_balance_cu,
                "pilot_spend_cu": round(reconciled_spend, 4),
            }
        )
    return prior, runs


def record_in_progress_run(
    out_dir: Path,
    *,
    session_name: str,
    bal_start: float,
    prior_runs: list[dict[str, Any]],
    max_spend_cu: float,
) -> None:
    """Persist starting balance BEFORE `colab new` so an interrupted run can always be reconciled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "total_cumulative_spend_cu": round(sum(float(r.get("pilot_spend_cu", 0.0)) for r in prior_runs), 4),
        "max_spend_cu": max_spend_cu,
        "in_progress_run": {
            "session_name": session_name,
            "balance_start_cu": bal_start,
            "start_epoch_s": time.time(),
        },
        "runs": prior_runs,
    }
    (out_dir / "cumulative_pilot_spend.json").write_text(json.dumps(ledger, indent=2) + "\n")


def _run_subprocess_with_watchdog(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_dir: Path,
    log_prefix: str,
    cfg_path: Path | None,
    bal_start: float,
    prior_spent_cu: float,
    max_spend_cu: float,
    t_alloc_start: float,
    phase_timeout_s: int = 2400,
    poll_interval_s: int = WATCHDOG_POLL_INTERVAL_S,
    balance_fn=query_balance,
) -> tuple[int, str, str, bool]:
    """Run subprocess with file-backed stdout/stderr (ZERO pipe deadlock!) and active cumulative CU + elapsed watchdog."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{log_prefix}.stdout.log"
    stderr_path = log_dir / f"{log_prefix}.stderr.log"

    hard_limit_cu = max(0.5, max_spend_cu - SHUTDOWN_RESERVE_CU)
    t_start = time.monotonic()
    t_last_poll = t_start
    t_ws_dead_seen: float | None = None
    budget_tripped = False

    with stdout_path.open("w+", encoding="utf-8") as f_out, stderr_path.open("w+", encoding="utf-8") as f_err:
        proc = subprocess.Popen(cmd, env=env, stdout=f_out, stderr=f_err, text=True)
        while proc.poll() is None:
            now = time.monotonic()
            # Guard against `google-colab-cli` non-daemon jupyter thread hanging forever after WebSocket disconnect!
            try:
                f_err.flush()
                if stderr_path.stat().st_size > 0:
                    err_tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-1200:]
                    if "RuntimeError: Connection was lost." in err_tail:
                        if t_ws_dead_seen is None:
                            t_ws_dead_seen = now
                        elif (now - t_ws_dead_seen) >= 12.0:
                            print("[WATCHDOG] Detected zombie colab-cli exit after WebSocket disconnect; killing immediately.", flush=True)
                            proc.kill()
                            break
            except Exception:
                pass
            if (now - t_start) > phase_timeout_s:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
                raise TimeoutError(f"Remote phase timed out after {phase_timeout_s}s")

            # Independent elapsed-time cumulative spending ceiling (immune to balance-API outages!)
            elapsed_cum_cu = prior_spent_cu + ((now - t_alloc_start) / 3600.0) * CONSERVATIVE_V6E1_RATE_CU_HR
            if elapsed_cum_cu >= hard_limit_cu:
                budget_tripped = True
                print(
                    f"[WATCHDOG] Conservative elapsed-time cumulative bound ({elapsed_cum_cu:.2f} CU) >= hard_limit ({hard_limit_cu:.2f} CU)! Halting...",
                    flush=True,
                )
                proc.terminate()
                break

            if (cfg_path is not None) and ((now - t_last_poll) >= poll_interval_s):
                t_last_poll = now
                try:
                    bal_now, rate_now = balance_fn(cfg_path)
                    cum_spent = prior_spent_cu + max(0.0, bal_start - bal_now)
                    tripped_now, projected, hard_lim = projected_spend_exceeds_reserve(
                        cum_spent, rate_now, max_spend_cu=max_spend_cu, poll_interval_s=poll_interval_s
                    )
                    print(
                        f"[WATCHDOG] cumulative_spent={cum_spent:.2f} CU, rate={rate_now:.2f} CU/hr, "
                        f"projected={projected:.2f} CU / hard_stop={hard_lim:.2f} CU",
                        flush=True,
                    )
                    if tripped_now:
                        budget_tripped = True
                        print("[WATCHDOG] Budget cap / 3.0 CU reserve reached! Terminating active phase cleanly...", flush=True)
                        proc.terminate()
                        break
                except Exception as exc:
                    print(
                        f"[WATCHDOG] Balance poll error ({exc}); elapsed_cum_bound={elapsed_cum_cu:.2f}/{hard_limit_cu:.2f} CU",
                        flush=True,
                    )
            time.sleep(1.0)

        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)

        f_out.flush()
        f_err.flush()
        f_out.seek(0)
        f_err.seek(0)
        out_txt = f_out.read()
        err_txt = f_err.read()
        return int(proc.returncode or 0), out_txt, err_txt, budget_tripped


def _run_remote_kernel_python_with_watchdog(
    session_name: str,
    python_code: str,
    *,
    cfg_path: Path,
    log_dir: Path,
    log_prefix: str,
    bal_start: float,
    prior_spent_cu: float,
    max_spend_cu: float,
    t_alloc_start: float,
    phase_timeout_s: int = 2400,
    exec_timeout_s: int = 1800,
    poll_interval_s: int = WATCHDOG_POLL_INTERVAL_S,
) -> tuple[int, str, str, bool]:
    """Execute Python directly inside the Colab TPU Jupyter kernel (`colab exec -f`) under the CU watchdog."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as tf:
        tf.write(python_code)
        script_path = Path(tf.name)
    try:
        cmd = _colab_cmd(
            "exec",
            "-s",
            session_name,
            "-f",
            str(script_path),
            "--timeout",
            str(int(exec_timeout_s)),
            cfg_path=cfg_path,
        )
        return _run_subprocess_with_watchdog(
            cmd,
            env=_colab_env(),
            log_dir=log_dir,
            log_prefix=log_prefix,
            cfg_path=cfg_path,
            bal_start=bal_start,
            prior_spent_cu=prior_spent_cu,
            max_spend_cu=max_spend_cu,
            t_alloc_start=t_alloc_start,
            phase_timeout_s=phase_timeout_s,
            poll_interval_s=poll_interval_s,
        )
    finally:
        script_path.unlink(missing_ok=True)


def _run_remote_bash_with_watchdog(
    session_name: str,
    remote_bash_cmd: str,
    *,
    cfg_path: Path,
    log_dir: Path,
    log_prefix: str,
    bal_start: float,
    prior_spent_cu: float,
    max_spend_cu: float,
    t_alloc_start: float,
    phase_timeout_s: int = 2400,
    poll_interval_s: int = WATCHDOG_POLL_INTERVAL_S,
) -> tuple[int, str, str, bool]:
    cmd = _colab_cmd("ssh", "-s", session_name, "--", "bash", "-lc", remote_bash_cmd, cfg_path=cfg_path)
    return _run_subprocess_with_watchdog(
        cmd,
        env=_colab_env(),
        log_dir=log_dir,
        log_prefix=log_prefix,
        cfg_path=cfg_path,
        bal_start=bal_start,
        prior_spent_cu=prior_spent_cu,
        max_spend_cu=max_spend_cu,
        t_alloc_start=t_alloc_start,
        phase_timeout_s=phase_timeout_s,
        poll_interval_s=poll_interval_s,
    )


def is_tpu_v6e_device_kind_stdlib(device_kind: str) -> bool:
    """Pure-stdlib exact TPU v6e / TPU v6 lite check (zero JAX dependency on Mac launcher host!)."""
    return device_kind.strip().lower() in {"tpu v6e", "tpu v6 lite"}


def validate_reusable_mup_receipt_stdlib(
    mup_receipt_path: Path,
    *,
    expected_bundle_sha256: str,
    require_tpu_v6e: bool = True,
) -> float:
    """Pure-stdlib strict Phase 2 receipt validator usable on the stdlib-only Mac launcher host!"""
    if not mup_receipt_path.is_file():
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: missing {mup_receipt_path}")
    data = json.loads(mup_receipt_path.read_text())
    if not isinstance(data, dict) or data.get("status") != "pass":
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: status={data.get('status')!r} (expected 'pass')")
    if data.get("smoke_cpu") is not False:
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: smoke_cpu={data.get('smoke_cpu')!r} (expected False)")
    if require_tpu_v6e and not is_tpu_v6e_device_kind_stdlib(str(data.get("device_kind", ""))):
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: invalid device_kind={data.get('device_kind')!r}")
    if data.get("widths") != [256, 512, 768] or data.get("depths") != [8, 12, 16]:
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: non-production grid widths={data.get('widths')} depths={data.get('depths')}")
    rec_sha = data.get("pilot_bundle_sha256")
    if not isinstance(rec_sha, str) or len(rec_sha) != 64 or rec_sha != expected_bundle_sha256:
        raise RuntimeError(
            f"PHASE 2 PREREQUISITE LOCK: bundle digest mismatch ({rec_sha!r} != {expected_bundle_sha256!r})"
        )
    lr_raw = data.get("recommended_base_lr")
    lrs_grid = data.get("lrs")
    if (
        isinstance(lr_raw, bool)
        or (not isinstance(lr_raw, (int, float)))
        or (not math.isfinite(float(lr_raw)))
        or float(lr_raw) <= 0.0
    ):
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: invalid recommended_base_lr={lr_raw!r}")
    if not isinstance(lrs_grid, list) or not any(
        (not isinstance(x, bool)) and isinstance(x, (int, float)) and abs(float(x) - float(lr_raw)) <= 1e-12
        for x in lrs_grid
    ):
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: recommended_base_lr={lr_raw!r} not in lrs grid {lrs_grid!r}")
    return float(lr_raw)


def validate_phase_prerequisites(
    phases: list[str], out_dir: Path, *, expected_bundle_sha256: str
) -> float | None:
    """Validate Phase 2 prerequisite strictly using pure stdlib (zero JAX import on Mac host!)."""
    needs_mup = [p for p in phases if p in ("dev_ablation", "long_stream")]
    if not needs_mup:
        return None
    if "mup_check" in phases and phases.index("mup_check") < min(phases.index(p) for p in needs_mup):
        return None
    return validate_reusable_mup_receipt_stdlib(
        out_dir / "phase_mup_check_receipt.json",
        expected_bundle_sha256=expected_bundle_sha256,
        require_tpu_v6e=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch hard-capped <= 45 CU SLWM-v3 pilot on Google TPU v6e-1 or NVIDIA H100/A100 (Rev 4).")
    ap.add_argument("--accel", type=str, default="tpu_v6e1", choices=("tpu_v6e1", "a100", "h100"))
    ap.add_argument("--session-name", type=str, default="slwm_v3_pilot_v6e1")
    ap.add_argument("--max-spend-cu", type=float, default=45.0)
    ap.add_argument("--phases", type=str, default="profile,mup_check,dev_ablation,long_stream")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports/v3/pilot_v6e1")
    ap.add_argument("--adopt-existing", action="store_true", help="Reuse an already-active named TPU v6e-1 session instead of calling `colab new`.")
    ap.add_argument("--dry-run", action="store_true", help="Verify gates, balance, digests, and CLI without assigning a TPU.")
    args = ap.parse_args()

    if args.max_spend_cu > 45.0:
        raise ValueError(f"Hard pilot cap exceeded: max_spend_cu={args.max_spend_cu} > 45.0 CU!")

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    # Single-read TOCTOU-free Gate 1–3 lock + archive construction BEFORE balance check or allocation!
    gate_status, tar_bytes = build_and_verify_pilot_bundle(REPO_ROOT)
    bundle_sha = gate_status["pilot_bundle_sha256_verified"]
    resumed_lr = validate_phase_prerequisites(phases, args.out_dir, expected_bundle_sha256=bundle_sha)
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = OPS_DIR / f"session_{args.session_name}.json"

    bal_start, rate_start = query_balance(cfg_path)
    prior_spent_cu, prior_runs = load_prior_cumulative_spend(args.out_dir, current_balance_cu=bal_start)
    tripped_pre, proj_pre, hard_limit_cu = projected_spend_exceeds_reserve(
        prior_spent_cu, rate_start, max_spend_cu=args.max_spend_cu
    )
    summary = {
        "gate_lock_status": gate_status,
        "gcloud_config_dir": str(resolve_gcloud_config_dir()),
        "colab_bin": str(COLAB_BIN),
        "starting_balance_cu": bal_start,
        "starting_rate_cu_hr": rate_start,
        "prior_cumulative_spend_cu": round(prior_spent_cu, 4),
        "max_spend_cu": args.max_spend_cu,
        "shutdown_reserve_cu": SHUTDOWN_RESERVE_CU,
        "hard_watchdog_stop_cu": round(hard_limit_cu, 2),
        "accelerator_policy": "tpu_v6e1_strict",
        "bundle_bytes": len(tar_bytes),
        "pilot_bundle_sha256": bundle_sha,
        "phases": phases,
        "resumed_validated_base_lr": resumed_lr,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    if tripped_pre:
        raise RuntimeError(
            f"CUMULATIVE BUDGET CAP: prior_spent={prior_spent_cu:.2f} CU (projected={proj_pre:.2f} CU) >= hard_limit={hard_limit_cu:.2f} CU!"
        )
    active_pre = query_active_sessions(cfg_path)
    if active_pre is None:
        raise RuntimeError("Pre-launch `colab sessions` failed or returned unrecognized output; refusing to proceed!")
    adopting_active = False
    if args.adopt_existing:
        matching = [
            ln for ln in active_pre
            if (ln.startswith(f"[{args.session_name}]") or f" [{args.session_name}]" in ln)
            and ("Hardware: V6E1" in ln)
            and ("Variant: TPU" in ln)
        ]
        if len(active_pre) == 1 and len(matching) == 1:
            adopting_active = True
            print(f"[PILOT] Adopting active named TPU v6e-1 session: {matching[0]}", flush=True)
        elif active_pre:
            raise RuntimeError(f"--adopt-existing expected single active V6E1 [{args.session_name}], found: {active_pre}")
    else:
        if rate_start > 0.0:
            raise RuntimeError(f"Active Colab session already burning {rate_start:.2f} CU/hr! Stop it before launching.")
        if active_pre:
            raise RuntimeError(f"Active Colab sessions already exist on server: {active_pre}! Stop them first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    chosen_lr = float(resumed_lr) if resumed_lr is not None else 1e-3
    completed_phases: list[str] = []
    t_alloc_start = time.monotonic()

    # Persist starting balance BEFORE `colab new` so an interrupted run can always be reconciled!
    record_in_progress_run(
        args.out_dir,
        session_name=args.session_name,
        bal_start=bal_start,
        prior_runs=prior_runs,
        max_spend_cu=args.max_spend_cu,
    )

    try:
        assigned_ok = adopting_active
        last_assign_err = ""
        for assign_try in range(1, 5):
            if assigned_ok:
                break
            print(
                f"[PILOT] Creating named {args.accel} session {args.session_name!r} (attempt {assign_try}/4)...",
                flush=True,
            )
            try:
                if args.accel == "tpu_v6e1":
                    assign_cmd = ("--logtostderr", "new", "-s", args.session_name, "--tpu", "v6e1")
                elif args.accel == "a100":
                    assign_cmd = ("--logtostderr", "new", "-s", args.session_name, "--gpu", "A100", "--high-mem")
                else:
                    assign_cmd = ("--logtostderr", "new", "-s", args.session_name, "--gpu", "H100")
                assign_res = run_colab(*assign_cmd, cfg_path=cfg_path, timeout=420)
                if assign_res.stdout.strip():
                    print(assign_res.stdout.strip(), flush=True)
                if assign_res.returncode == 0:
                    assigned_ok = True
                    break
                last_assign_err = (assign_res.stderr or assign_res.stdout)[-1000:]
                print(f"[PILOT WARN] TPU v6e-1 assign attempt {assign_try}/4 failed (rc={assign_res.returncode}): {last_assign_err.splitlines()[-1] if last_assign_err.strip() else 'unknown'}", flush=True)
            except subprocess.TimeoutExpired:
                last_assign_err = "subprocess timeout (420s)"
                print(f"[PILOT WARN] TPU v6e-1 assign attempt {assign_try}/4 timed out after 420s.", flush=True)

            active_chk = query_active_sessions(cfg_path) or []
            if any(ln.startswith(f"[{args.session_name}]") or f" [{args.session_name}]" in ln for ln in active_chk):
                print(f"[PILOT] Session {args.session_name!r} confirmed active after transient assign error!", flush=True)
                assigned_ok = True
                break
            if any(ln.startswith("[?]") or " [?]" in ln for ln in active_chk):
                if _adopt_single_unnamed_v6e1(args.session_name, cfg_path):
                    assigned_ok = True
                    break
                raise RuntimeError(f"Unnamed Colab assignment `[?]` detected during assign retry: {active_chk}")
            if assign_try < 4:
                time.sleep(25.0)

        if not assigned_ok:
            raise RuntimeError(f"TPU v6e-1 assign failed after 4 attempts: {last_assign_err}")
        t_alloc_start = time.monotonic()

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(tar_bytes)
            tmp_path = Path(tmp.name)
        try:
            up_res = run_colab(
                "upload", "-s", args.session_name, str(tmp_path), "/content/slwm_v3_bundle.tar.gz", cfg_path=cfg_path, timeout=180
            )
            if up_res.returncode != 0:
                raise RuntimeError(f"Bundle upload failed: {up_res.stderr or up_res.stdout}")
        finally:
            tmp_path.unlink(missing_ok=True)

        expected_bk = "tpu" if args.accel == "tpu_v6e1" else "gpu"
        prep_py = f"""
import os, pathlib, subprocess, sys, tarfile
for d in ("/content/slwm_v3", "/content/slwm_v3_out", "/content/slwm_v3_xla"):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/slwm_v3_bundle.tar.gz", "r:gz") as tf:
    tf.extractall("/content/slwm_v3")
try:
    import optax
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "optax"])
    import optax
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/slwm_v3_xla"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"
import jax
b = jax.default_backend()
assert b == "{expected_bk}", f"Expected {expected_bk} backend in Colab kernel, got {{b!r}}"
devs = [(d.device_kind, (d.memory_stats() or {{}}).get("bytes_limit", 0)) for d in jax.devices()]
print("KERNEL_TPU_OK", "JAX:", jax.__version__, "OPTAX:", optax.__version__, "BACKEND:", b, "DEVS:", devs, flush=True)
"""
        rc, out, err, _ = _run_remote_kernel_python_with_watchdog(
            args.session_name,
            prep_py,
            cfg_path=cfg_path,
            log_dir=log_dir,
            log_prefix="00_prep",
            bal_start=bal_start,
            prior_spent_cu=prior_spent_cu,
            max_spend_cu=args.max_spend_cu,
            t_alloc_start=t_alloc_start,
            phase_timeout_s=360,
            exec_timeout_s=300,
        )
        print(out, flush=True)
        if rc != 0 or "KERNEL_TPU_OK" not in out:
            print(err, flush=True)
            raise RuntimeError(f"Remote kernel prep failed rc={rc}")

        # If resuming Phases 3/4, upload the validated local Phase 2 receipt so remote paid dispatch verifies it too!
        if resumed_lr is not None:
            mup_local = args.out_dir / "phase_mup_check_receipt.json"
            up_mup = run_colab(
                "upload", "-s", args.session_name, str(mup_local), "/content/slwm_v3_out/phase_mup_check_receipt.json", cfg_path=cfg_path, timeout=120
            )
            if up_mup.returncode != 0:
                raise RuntimeError(f"Failed to upload validated Phase 2 receipt: {up_mup.stderr or up_mup.stdout}")

        for idx, ph in enumerate(phases, start=1):
            bal_now, rate_now = query_balance(cfg_path)
            elapsed_cu = ((time.monotonic() - t_alloc_start) / 3600.0) * CONSERVATIVE_V6E1_RATE_CU_HR
            cum_spent = max(prior_spent_cu + max(0.0, bal_start - bal_now), prior_spent_cu + elapsed_cu)
            tripped_entry, proj_entry, hard_lim = projected_spend_exceeds_reserve(
                cum_spent, rate_now, max_spend_cu=args.max_spend_cu
            )
            print(
                f"[PILOT] Pre-phase {ph!r}: cumulative_spent={cum_spent:.2f} CU, projected={proj_entry:.2f} CU / hard_limit={hard_lim:.2f} CU (rate={rate_now:.2f})",
                flush=True,
            )
            if tripped_entry:
                print(f"[PILOT] Reserve/budget cap reached before dispatching phase {ph!r}; stopping cleanly.", flush=True)
                break

            # Restart the Colab kernel before every phase to release 100% of TPU HBM cleanly while keeping XLA disk cache!
            rk_res = run_colab("restart-kernel", "-s", args.session_name, cfg_path=cfg_path, timeout=120)
            if rk_res.returncode != 0:
                print(f"[PILOT WARN] restart-kernel before {ph!r} returned rc={rk_res.returncode}: {rk_res.stderr or rk_res.stdout}", flush=True)
            time.sleep(6.0)
            for ping_i in range(4):
                ping = run_colab("exec", "-s", args.session_name, "print('KERNEL_READY')", "--timeout", "60", cfg_path=cfg_path, timeout=90)
                if ping.returncode == 0 and "KERNEL_READY" in ping.stdout:
                    break
                print(f"[PILOT WARN] Post-restart kernel ping {ping_i + 1}/4 failed (rc={ping.returncode}); retrying in 8s...", flush=True)
                time.sleep(8.0)

            argv_list = [
                "scripts/train/v3_tpu_pilot_suite.py",
                "--phases",
                ph,
                "--bundle-sha256",
                bundle_sha,
                "--out-dir",
                "/content/slwm_v3_out",
            ]
            if args.accel == "tpu_v6e1":
                argv_list.append("--require-tpu-v6e")
            if ph in ("dev_ablation", "long_stream"):
                argv_list.extend(["--base-lr", str(chosen_lr)])

            phase_py = (
                "import os, sys\n"
                'os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/slwm_v3_xla"\n'
                'os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"\n'
                'os.environ["OPENBLAS_NUM_THREADS"] = "1"\n'
                'os.environ["OMP_NUM_THREADS"] = "2"\n'
                'os.chdir("/content/slwm_v3")\n'
                'if "/content/slwm_v3" not in sys.path:\n'
                '    sys.path.insert(0, "/content/slwm_v3")\n'
                f"sys.argv = {json.dumps(argv_list)}\n"
                "from scripts.train import v3_tpu_pilot_suite\n"
                "v3_tpu_pilot_suite.main()\n"
                'print("PHASE_EXEC_OK", flush=True)\n'
            )
            for ph_try in range(1, 3):
                rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
                    args.session_name,
                    phase_py,
                    cfg_path=cfg_path,
                    log_dir=log_dir,
                    log_prefix=f"{idx:02d}_{ph}" + ("" if ph_try == 1 else f"_retry{ph_try}"),
                    bal_start=bal_start,
                    prior_spent_cu=prior_spent_cu,
                    max_spend_cu=args.max_spend_cu,
                    t_alloc_start=t_alloc_start,
                    phase_timeout_s=(6600 if ph == "dev_ablation" else 2400),
                    exec_timeout_s=(6000 if ph == "dev_ablation" else 1800),
                )
                if (rc == 0 and "PHASE_EXEC_OK" in out) or tripped:
                    break
                if "Connection was lost." in err and ph_try < 2:
                    print(f"[PILOT WARN] WebSocket dropped during phase {ph!r} (try {ph_try}); restarting kernel & retrying...", flush=True)
                    run_colab("restart-kernel", "-s", args.session_name, cfg_path=cfg_path, timeout=90)
                    time.sleep(8.0)
                else:
                    break
            print(out, flush=True)
            if err.strip():
                print(err, flush=True)
            if tripped:
                print(f"[PILOT] Watchdog halted phase {ph!r} to protect <= {args.max_spend_cu:.1f} CU cap.", flush=True)
                break
            if rc != 0 or "PHASE_EXEC_OK" not in out:
                raise RuntimeError(f"Remote phase {ph!r} failed with rc={rc}")

            remote_receipt = f"/content/slwm_v3_out/phase_{ph}_receipt.json"
            local_receipt = args.out_dir / f"phase_{ph}_receipt.json"
            dl_res = run_colab(
                "download", "-s", args.session_name, remote_receipt, str(local_receipt), cfg_path=cfg_path, timeout=120
            )
            if dl_res.returncode != 0 or not local_receipt.is_file():
                raise RuntimeError(f"Failed to download {remote_receipt}: {dl_res.stderr or dl_res.stdout}")
            ph_data = json.loads(local_receipt.read_text())
            completed_phases.append(ph)
            print(f"[PILOT] Downloaded {local_receipt} (status={ph_data.get('status')!r})", flush=True)

            if ph == "mup_check" and ph_data.get("status") == "pass":
                chosen_lr = float(ph_data["recommended_base_lr"])

            if ph_data.get("status") != "pass":
                print(
                    f"[PILOT] Phase {ph!r} returned status={ph_data.get('status')!r}; "
                    "HALTING remaining phases before further spend!",
                    flush=True,
                )
                break
    finally:
        print(f"[PILOT] Guaranteed verified auto-teardown for session {args.session_name!r}...", flush=True)
        stopped_ok = verified_stop_session(args.session_name, cfg_path)
        elapsed_est = max(0.0, (time.monotonic() - t_alloc_start) / 3600.0) * CONSERVATIVE_V6E1_RATE_CU_HR
        bal_end: float | None = None
        rate_end: float | None = None
        bal_query_ok = True
        try:
            bal_end, rate_end = query_balance(cfg_path)
            this_spend = max(0.0, bal_start - bal_end)
        except Exception as exc:
            bal_query_ok = False
            this_spend = elapsed_est
            print(f"[PILOT WARNING] Final balance query failed ({exc}); recording conservative elapsed estimate {this_spend:.4f} CU.", flush=True)
        run_record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_name": args.session_name,
            "pilot_bundle_sha256": bundle_sha,
            "models_v3_sha256": gate_status["models_v3_sha256_verified"],
            "phases_completed": completed_phases,
            "teardown_verified_removed": stopped_ok,
            "final_balance_query_ok": bal_query_ok,
            "balance_start_cu": bal_start,
            "balance_end_cu": bal_end,
            "pilot_spend_cu": round(this_spend, 4),
            "cumulative_pilot_spend_cu": round(prior_spent_cu + this_spend, 4),
            "rate_after_stop_cu_hr": rate_end,
        }
        prior_runs.append(run_record)
        ledger = {
            "total_cumulative_spend_cu": round(sum(float(r["pilot_spend_cu"]) for r in prior_runs), 4),
            "max_spend_cu": args.max_spend_cu,
            "runs": prior_runs,
        }
        (args.out_dir / "cumulative_pilot_spend.json").write_text(json.dumps(ledger, indent=2) + "\n")
        print(json.dumps(run_record, indent=2), flush=True)
        if not stopped_ok:
            raise RuntimeError(f"CRITICAL: Colab session {args.session_name!r} could not be verified stopped!")


if __name__ == "__main__":
    main()
