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
import subprocess
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
            "scripts/eval/v3_live_any2any_inference.py",
        )
    )
)


def _exec_short_py(session_name: str, code: str, *, cfg_path: Path, timeout: int = 180) -> subprocess.CompletedProcess:
    """Execute a short Python snippet on a Colab session via temporary `-f` file (since `colab exec` uses `-f`, not `-c`)."""
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tf.write(code)
    tf.close()
    tf_p = Path(tf.name)
    try:
        return run_colab("exec", "-s", session_name, "-f", str(tf_p), "--timeout", str(timeout), cfg_path=cfg_path, timeout=timeout + 30)
    finally:
        tf_p.unlink(missing_ok=True)


def _run_detached_remote_script_with_polling(
    session_name: str,
    script_body: str,
    *,
    tag: str,
    cfg_path: Path,
    log_dir: Path,
    bal_start: float,
    max_spend_cu: float,
    t_alloc_start: float,
    poll_interval_s: float = 45.0,
    max_wall_s: float = 39600.0,
) -> tuple[int, str, bool]:
    """Run a long GPU script completely detached from the Colab WebSocket and poll via short 3s heartbeats!

    Eliminates 90-120 min WebSocket timeouts AND idle-VM reclaims: if a single poll hits a transient
    network glitch, the remote GPU process keeps training uninterrupted and polling resumes on the next tick.
    """
    rem_script = f"/content/out/_run_{tag}.py"
    rem_log = f"/content/out/_run_{tag}.log"
    rem_rc = f"/content/out/_run_{tag}.rc"
    rem_pid = f"/content/out/_run_{tag}.pid"
    loc_log = log_dir / f"detached_{tag}.log"

    # Wrap script so exit code is always written atomically to `rem_rc`
    wrapped = (
        "import os, sys, traceback\n"
        f"rc_path = {rem_rc!r}\n"
        "rc = 0\n"
        "try:\n"
        + "\n".join("    " + ln for ln in script_body.splitlines())
        + "\nexcept SystemExit as e:\n"
        "    rc = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    rc = 1\n"
        "finally:\n"
        "    with open(rc_path, 'w') as _f:\n"
        "        _f.write(str(rc))\n"
        "    sys.stdout.flush(); sys.stderr.flush()\n"
    )
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tf.write(wrapped)
    tf.close()
    tf_p = Path(tf.name)
    try:
        up = run_colab("upload", "-s", session_name, str(tf_p), rem_script, cfg_path=cfg_path, timeout=120)
        if up.returncode != 0:
            raise RuntimeError(f"Failed uploading detached script {rem_script}: {up.stderr or up.stdout}")
    finally:
        tf_p.unlink(missing_ok=True)

    # Launch completely detached (`start_new_session=True`, clean GPU memory) in a 3-second kernel call!
    launch_py = (
        "import os, subprocess, sys\n"
        f"for p in ({rem_rc!r}, {rem_log!r}, {rem_pid!r}):\n"
        "    if os.path.exists(p): os.remove(p)\n"
        "env = dict(os.environ)\n"
        "env['PYTHONUNBUFFERED'] = '1'\n"
        "env['JAX_COMPILATION_CACHE_DIR'] = '/content/xla'\n"
        "env['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '1'\n"
        f"lf = open({rem_log!r}, 'wb')\n"
        f"p = subprocess.Popen([sys.executable, {rem_script!r}], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd='/content/slwm_v3')\n"
        f"with open({rem_pid!r}, 'w') as pf: pf.write(str(p.pid))\n"
        "print('DETACHED_LAUNCH_OK', p.pid, flush=True)\n"
    )
    l_res = None
    last_l_err = ""
    for l_try in range(1, 4):
        try:
            l_res = _exec_short_py(session_name, launch_py, cfg_path=cfg_path, timeout=90)
            if l_res.returncode == 0 and "DETACHED_LAUNCH_OK" in l_res.stdout:
                break
            last_l_err = (l_res.stderr or l_res.stdout)[-300:]
        except subprocess.TimeoutExpired:
            last_l_err = "TimeoutExpired(90s)"
        print(f"[DETACHED LAUNCH WARN {l_try}/3] {last_l_err}", flush=True)
        run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
        time.sleep(5.0)
    if l_res is None or l_res.returncode != 0 or "DETACHED_LAUNCH_OK" not in l_res.stdout:
        raise RuntimeError(f"Detached launch failed for {tag}: {last_l_err}")
    print(f"[DETACHED] {l_res.stdout.strip()} (polling every {int(poll_interval_s)}s)", flush=True)

    offset = 0
    full_tail = ""
    t0 = time.monotonic()
    consec_errs = 0
    poll_n = 0
    tripped = False
    loc_log.write_text("", encoding="utf-8")

    while True:
        time.sleep(poll_interval_s)
        poll_n += 1
        poll_py = (
            "import json, os\n"
            f"lp, rp, pp = {rem_log!r}, {rem_rc!r}, {rem_pid!r}\n"
            f"off = {offset}\n"
            "chunk = ''\n"
            "new_off = off\n"
            "if os.path.isfile(lp):\n"
            "    with open(lp, 'rb') as f:\n"
            "        f.seek(off)\n"
            "        b = f.read(65536)\n"
            "        new_off = off + len(b)\n"
            "        chunk = b.decode('utf-8', errors='replace')\n"
            "rc_val = None\n"
            "if os.path.isfile(rp):\n"
            "    try: rc_val = int(open(rp).read().strip())\n"
            "    except Exception: rc_val = 1\n"
            "alive = False\n"
            "if os.path.isfile(pp):\n"
            "    try:\n"
            "        pid = int(open(pp).read().strip())\n"
            "        os.kill(pid, 0)\n"
            "        alive = True\n"
            "    except Exception:\n"
            "        alive = False\n"
            "print('POLL_JSON_BEGIN' + json.dumps({'off': new_off, 'rc': rc_val, 'alive': alive, 'chunk': chunk}) + 'POLL_JSON_END', flush=True)\n"
        )
        try:
            pr = _exec_short_py(session_name, poll_py, cfg_path=cfg_path, timeout=90)
        except Exception as exc:
            consec_errs += 1
            print(f"[DETACHED POLL WARN {consec_errs}/12] {exc!r}", flush=True)
            if consec_errs >= 12:
                return 125, full_tail, False
            continue

        if pr.returncode == 0 and "POLL_JSON_BEGIN" in pr.stdout:
            consec_errs = 0
            try:
                raw_j = pr.stdout.split("POLL_JSON_BEGIN", 1)[1].split("POLL_JSON_END", 1)[0]
                st = json.loads(raw_j)
                offset = int(st["off"])
                chunk = str(st["chunk"])
                if chunk:
                    with loc_log.open("a", encoding="utf-8") as lf_loc:
                        lf_loc.write(chunk)
                    full_tail = (full_tail + chunk)[-16000:]
                    for ln in chunk.splitlines():
                        if any(k in ln for k in ("[TRAIN", "[CKPT", "[WAVE_", "[INIT_CKPT", "[PREENCODE", "[DL]", "ARM_RUN_OK", "DONE", "Error", "Traceback")):
                            print(ln, flush=True)
                if st["rc"] is not None:
                    return int(st["rc"]), full_tail, False
                if not st["alive"] and poll_n >= 2:
                    print(f"[DETACHED WARN] Remote PID exited without rc file for {tag}.", flush=True)
                    return 1, full_tail, False
            except Exception as exc:
                print(f"[DETACHED PARSE WARN] {exc!r}", flush=True)
        else:
            consec_errs += 1
            msg = (pr.stderr or pr.stdout).strip()[-200:]
            print(f"[DETACHED POLL WARN {consec_errs}/12] rc={pr.returncode} {msg}", flush=True)
            if "not found" in msg.lower():
                return 126, full_tail, False
            if consec_errs >= 12:
                return 125, full_tail, False

        # Budget watchdog check every 4 polls (~3 minutes)
        if poll_n % 4 == 0:
            try:
                bal_now, rate_now = query_balance(cfg_path)
                spent = max(0.0, bal_start - bal_now)
                proj = spent + (rate_now * (poll_interval_s * 4.0) / 3600.0)
                hard_stop = max(0.5, max_spend_cu - 3.0)
                print(f"[WATCHDOG] spent={spent:.2f} CU, rate={rate_now:.2f} CU/hr, proj={proj:.2f} / hard_stop={hard_stop:.2f} CU (elapsed={(time.monotonic()-t0)/60:.1f}m)", flush=True)
                if proj >= hard_stop:
                    tripped = True
                    return 124, full_tail, True
            except Exception:
                pass
        if (time.monotonic() - t0) >= max_wall_s:
            print(f"[DETACHED WARN] Max wall time {max_wall_s:.0f}s exceeded for {tag}.", flush=True)
            return 124, full_tail, False


def _mirror_path_to_fedora(local_path: Path, *, fedora_host: str, fedora_dir: str, profile: str) -> None:
    """Best-effort permanent archive mirror of downloaded checkpoints/media to Fedora LAN storage."""
    if not fedora_host or not fedora_dir or not local_path.exists():
        return
    rem_dir = f"{fedora_dir.rstrip('/')}/{profile}"
    try:
        mk = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", fedora_host, f"mkdir -p {rem_dir}"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if mk.returncode != 0:
            print(f"[FEDORA MIRROR SKIP] Host {fedora_host} unreachable ({mk.stderr.strip()[:120]})", flush=True)
            return
        cmd = ["scp", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
        if local_path.is_dir():
            cmd.append("-r")
        cmd.extend([str(local_path), f"{fedora_host}:{rem_dir}/"])
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        if cp.returncode == 0:
            print(f"[FEDORA MIRROR OK] {local_path.name} -> {fedora_host}:{rem_dir}/", flush=True)
        else:
            print(f"[FEDORA MIRROR WARN] scp failed for {local_path.name}: {cp.stderr.strip()[:160]}", flush=True)
    except Exception as exc:
        print(f"[FEDORA MIRROR WARN] {exc!r}", flush=True)


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
    chunk_bytes: int = 16 * 1024 * 1024,
) -> None:
    """Stateless 4-stream HTTPS PUT upload of shards in <=16 MiB parts with per-shard SHA-256 assembly + heartbeats.

    Assembles and removes temporary `_part_*` files right after each large shard (sending a live kernel
    heartbeat every ~1-2 min so Colab's idle-kernel watchdog NEVER reclaims the VM during slow Wi-Fi uploads!).
    """

    def _put_one(local_tmp: Path, rem_p: str) -> None:
        last_err = ""
        for attempt in range(1, 6):
            try:
                up = run_colab("upload", "-s", session_name, str(local_tmp), rem_p, cfg_path=cfg_path, timeout=480)
                if up.returncode == 0:
                    return
                last_err = up.stderr or up.stdout
            except subprocess.TimeoutExpired as exc:
                last_err = f"TimeoutExpired({exc.timeout}s)"
            time.sleep(4.0 * attempt)
        raise RuntimeError(f"Upload failed for {rem_p}: {last_err}")

    def _assemble_one(item: dict[str, Any]) -> None:
        asm_py = (
            "import hashlib, os\n"
            f"item = {json.dumps(item)}\n"
            "dst = item['dst']\n"
            "os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
            "if not (os.path.isfile(dst) and os.path.getsize(dst) == item['size']):\n"
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
            "print('SHARD_ASM_OK', dst, item['size'], flush=True)\n"
        )
        last_err = ""
        for attempt in range(1, 4):
            try:
                res = _exec_short_py(session_name, asm_py, cfg_path=cfg_path, timeout=180)
                if res.returncode == 0 and "SHARD_ASM_OK" in res.stdout:
                    return
                last_err = (res.stderr or res.stdout)[-300:]
            except subprocess.TimeoutExpired:
                last_err = "TimeoutExpired(180s)"
            run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
            time.sleep(4.0)
        raise RuntimeError(f"Shard assembly failed for {item['dst']}: {last_err}")

    last_hb = time.monotonic()
    for sf_idx, sf in enumerate(shard_files):
        rel = sf.relative_to(local_root).as_posix()
        remote_p = f"/content/shards/{rel}"
        sz = sf.stat().st_size
        sz_mb = sz / (1024.0**2)
        print(f"[WAVE_LAUNCH] Uploading shard {sf_idx+1}/{len(shard_files)}: {rel} ({sz_mb:.1f} MiB)...", flush=True)
        if sz <= chunk_bytes:
            _put_one(sf, remote_p)
            if time.monotonic() - last_hb >= 180.0:
                try:
                    _exec_short_py(session_name, "print('UPLOAD_HB_OK', flush=True)\n", cfg_path=cfg_path, timeout=60)
                except Exception:
                    pass
                last_hb = time.monotonic()
            continue

        # Split locally into <=16 MiB parts and upload 4 parts concurrently via stateless HTTPS PUT
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
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(_put_one, lp, rp) for lp, rp in part_jobs]
                for f in futs:
                    f.result()
        finally:
            for lp, _ in part_jobs:
                lp.unlink(missing_ok=True)
        _assemble_one(
            {
                "dst": remote_p,
                "size": int(sz),
                "sha256": sha.hexdigest(),
                "parts": [rp for _, rp in part_jobs],
            }
        )
        last_hb = time.monotonic()
    print(f"ALL_SHARDS_ASM_OK {len(shard_files)}", flush=True)


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
        "            pp = f'/content/_dl_part_{idx:04d}.bin'\n"
        "            with open(pp, 'wb') as fout:\n"
        "                fout.write(b)\n"
        "            parts.append(pp)\n"
        "            idx += 1\n"
        "    print('DL_META_BEGIN' + json.dumps({'size': sz, 'sha256': h.hexdigest(), 'parts': parts}) + 'DL_META_END', flush=True)\n"
    )
    rc, out, err, tripped = 1, "", "", False
    for sp_try in range(1, 5):
        if sp_try > 1:
            print(f"[CKPT_DL] Retrying remote split {sp_try}/4 for {remote_path}...", flush=True)
            try:
                run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
            except Exception:
                pass
            time.sleep(5.0)
        rc, out, err, tripped = _run_remote_kernel_python_with_watchdog(
            session_name, split_py, cfg_path=cfg_path, log_dir=log_dir,
            log_prefix=f"dl_split_{local_path.stem}_try{sp_try}",
            bal_start=bal_start, prior_spent_cu=0.0, max_spend_cu=max_spend_cu,
            t_alloc_start=t_alloc, phase_timeout_s=360, exec_timeout_s=240,
        )
        if rc == 0 and (not tripped) and ("DL_META_BEGIN" in out or "DL_MISSING" in out):
            break
    if rc != 0 or tripped or "DL_META_BEGIN" not in out:
        print(f"[CKPT_DL WARN] Could not prepare {remote_path}: {out[-200:]} {err[-200:]}", flush=True)
        return False
    meta = json.loads(out.split("DL_META_BEGIN", 1)[1].split("DL_META_END", 1)[0])
    parts: list[str] = meta["parts"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_local = local_path.with_suffix(local_path.suffix + ".part")
    h_loc = hashlib.sha256()
    tot = 0
    print(f"[CKPT_DL] Downloading {Path(remote_path).name} ({meta['size'] / 1024**2:.1f} MiB in {len(parts)} parts, 4 streams) -> {local_path}...", flush=True)

    def _dl_one(i_part: int, rem_p: str) -> Path:
        p_loc = local_path.parent / f"._{local_path.name}.p{i_part:04d}"
        for att in range(1, 6):
            try:
                res = run_colab("download", "-s", session_name, rem_p, str(p_loc), cfg_path=cfg_path, timeout=480)
                if res.returncode == 0 and p_loc.is_file():
                    return p_loc
            except subprocess.TimeoutExpired:
                pass
            time.sleep(4.0 * att)
        raise RuntimeError(f"Part {i_part+1}/{len(parts)} failed for {remote_path}")

    part_paths: list[Path | None] = [None] * len(parts)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_dl_one, i, rp): i for i, rp in enumerate(parts)}
            done_cnt = 0
            for fut in futs:
                idx = futs[fut]
                part_paths[idx] = fut.result()
                done_cnt += 1
                if done_cnt % 16 == 0:
                    try:
                        _exec_short_py(session_name, "print('DL_HB_OK', flush=True)\n", cfg_path=cfg_path, timeout=45)
                    except Exception:
                        pass
        with tmp_local.open("wb") as fout:
            for p_loc in part_paths:
                assert p_loc is not None
                raw = p_loc.read_bytes()
                fout.write(raw)
                h_loc.update(raw)
                tot += len(raw)
                p_loc.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[CKPT_DL WARN] Download aborted for {remote_path}: {exc!r}", flush=True)
        for p_loc in part_paths:
            if p_loc is not None:
                p_loc.unlink(missing_ok=True)
        tmp_local.unlink(missing_ok=True)
        return False
    if tot != int(meta["size"]) or h_loc.hexdigest() != str(meta["sha256"]):
        print(f"[CKPT_DL WARN] Integrity mismatch on {local_path}: {tot} vs {meta['size']}", flush=True)
        tmp_local.unlink(missing_ok=True)
        return False
    tmp_local.replace(local_path)
    # Clean up remote temp parts + remote ckpt to keep `/content` disk lean
    clean_py = (
        "import glob, os\n"
        "for p in glob.glob('/content/_dl_part_*.bin'):\n"
        "    try: os.remove(p)\n"
        "    except OSError: pass\n"
        f"try: os.remove({remote_path!r})\n"
        "except OSError: pass\n"
    )
    try:
        _exec_short_py(session_name, clean_py, cfg_path=cfg_path, timeout=60)
    except Exception:
        pass
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
        for init_rel in (
            "models/__init__.py",
            "data/__init__.py",
            "evals/__init__.py",
            "scripts/__init__.py",
            "scripts/train/__init__.py",
            "scripts/eval/__init__.py",
        ):
            ti = tarfile.TarInfo(name=init_rel)
            ti.size = 0
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(b""))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch Wave A-D SLWM-v3 real-corpus training & SoTA eval on Colab.")
    ap.add_argument("--accel", type=str, default="tpu_v6e1", choices=("tpu_v6e1", "h100", "a100", "l4"))
    ap.add_argument("--session-name", type=str, default="", help="Optional unique Colab session name (required when launching parallel instances!).")
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--representation-mode", type=str, default="frozen_codec_edge", choices=("frozen_codec_edge", "scratch_edge"))
    ap.add_argument("--arms", type=str, default="slwm_v3_routed,single_bank_gated_deltanet,sliding_window_gqa_eqbyte")
    ap.add_argument("--seeds", type=str, default="4201")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--start-step", type=int, default=0, help="Global start step for multi-leg cosine continuation.")
    ap.add_argument("--total-steps", type=int, default=0, help="Global total steps for cosine schedule (0 => start_step + steps).")
    ap.add_argument("--init-ckpt-local", type=Path, default=None, help="Optional local `.pkl` checkpoint to upload & resume from.")
    ap.add_argument("--tag-suffix", type=str, default="", help="Optional suffix appended to run tag (e.g. `_leg2` or `_300m`).")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=2e-3)
    ap.add_argument("--drop-modality-prob", type=float, default=0.25)
    ap.add_argument("--anchor-transition-prob", type=float, default=0.50)
    ap.add_argument("--multi-horizon-weight", type=float, default=0.15)
    ap.add_argument("--sampling-mode", type=str, default="temp_sqrt", choices=("temp_sqrt", "fixed"), help="Domain mixing: `temp_sqrt` enforces epoch caps (default: temp_sqrt).")
    ap.add_argument("--max-epochs-per-domain", type=float, default=4.0, help="Max planned epochs per domain under `temp_sqrt` sampling (default: 4.0).")
    ap.add_argument("--local-shard-dir", type=Path, required=True, help="Directory with pre-encoded `.npz` shards to upload.")
    ap.add_argument("--include-test-shards", action="store_true", help="Upload sealed `*_test*.npz` shards & enable `--eval-sealed-test`.")
    ap.add_argument("--download-ckpts", action=argparse.BooleanOptionalAction, default=True, help="Download trained core+Wave B/C `.pkl` checkpoints with SHA-256 verification before cleanup (default: True).")
    ap.add_argument("--ckpt-dir", type=Path, default=Path(".ops/checkpoints"), help="Local gitignored directory for downloaded `.pkl` checkpoints.")
    ap.add_argument("--mirror-fedora-host", type=str, default="db@192.168.1.200", help="SSH host for automatic checkpoint archival (empty to disable).")
    ap.add_argument("--mirror-fedora-dir", type=str, default="/mnt/slwm-data/SLWM/v3_checkpoints", help="Remote directory on Fedora for permanent checkpoint storage.")
    ap.add_argument("--run-wave-b", action="store_true", help="Also run frozen-core Wave B PolicyCommitGate after each arm.")
    ap.add_argument("--run-wave-c", action="store_true", help="Also run frozen-core Wave C Decoders & causal audit after each arm.")
    ap.add_argument("--wave-c-steps", type=int, default=1500, help="Frozen-core Wave C multimodal decoder steps (default: 1500).")
    ap.add_argument("--render-media-gallery", action="store_true", help="Also render real `.wav` audio & `.png`/`.gif`/`.mp4` video any-to-any media gallery on the Colab GPU before teardown.")
    ap.add_argument("--resume", action="store_true", help="Skip arms/seeds whose complete receipt JSONs already exist in `--out-dir`.")
    ap.add_argument("--max-spend-cu", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / ".ops/colab-config.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session_name = args.session_name.strip() or f"slwm_v3_{args.profile}_{args.accel}_{int(time.time()) % 100000}"
    log_dir = args.out_dir / "logs" / session_name
    log_dir.mkdir(parents=True, exist_ok=True)

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
        # Avoid uploading redundant 24m text shard, smaller libri100 shard when libri360 exists, or RSO clipbank!
        has_full_tc = any(p.name == "text_code_train_v1.npz" for p in shard_files)
        if has_full_tc:
            shard_files = [p for p in shard_files if p.name != "text_code_train_24m_balanced.npz"]
        has_libri360 = any(p.name == "librispeech_train_clean_360.npz" for p in shard_files)
        if has_libri360:
            shard_files = [p for p in shard_files if p.name != "librispeech_train_clean_100.npz"]
        shard_files = [p for p in shard_files if not p.name.startswith("rso_train_clipbank_")]
        if not shard_files:
            raise FileNotFoundError(f"No .npz shards found under {args.local_shard_dir}")
        _upload_shards_batch_chunked(
            session_name, shard_files, args.local_shard_dir,
            cfg_path=cfg_path, log_dir=log_dir,
            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
        )
        remote_init_ckpt = None
        if args.init_ckpt_local is not None and args.init_ckpt_local.is_file():
            print(f"[WAVE_LAUNCH] Uploading continuation checkpoint {args.init_ckpt_local}...", flush=True)
            _upload_shards_batch_chunked(
                session_name, [args.init_ckpt_local], args.init_ckpt_local.parent,
                cfg_path=cfg_path, log_dir=log_dir,
                bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
            )
            remote_init_ckpt = f"/content/shards/{args.init_ckpt_local.name}"

        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        tripped = False

        for arm in arms:
            for seed in seeds:
                use_wy = arm.endswith("_wy") or ("_c16" in arm) or ("_c32" in arm)
                surprise_c = 16 if "_c16" in arm else (32 if "_c32" in arm else 1)
                bb_name = arm.replace("_wy", "").replace("_c16", "").replace("_c32", "")
                rep_sfx = "" if args.representation_mode == "frozen_codec_edge" else f"_{args.representation_mode}"
                tag = f"{args.profile}_{arm}{rep_sfx}_s{seed}{args.tag_suffix}"
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
                    "--start-step", str(args.start_step),
                    "--total-steps", str(args.total_steps),
                    "--seq-len", str(args.seq_len),
                    "--batch-size", str(args.batch_size),
                    "--grad-accum", str(args.grad_accum),
                    "--base-lr", str(args.base_lr),
                    "--drop-modality-prob", str(args.drop_modality_prob),
                    "--anchor-transition-prob", str(args.anchor_transition_prob),
                    "--multi-horizon-weight", str(args.multi_horizon_weight),
                    "--sampling-mode", str(args.sampling_mode),
                    "--max-epochs-per-domain", str(args.max_epochs_per_domain),
                    "--seed", str(seed),
                    "--tag", tag,
                    "--out-dir", "/content/out",
                ]
                if remote_init_ckpt is not None:
                    argv.extend(["--init-ckpt", remote_init_ckpt])
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
                        "try:\n"
                        f"    sys.argv = ['scripts/train/v3_wave_b_policy_gate.py', '--ckpt', '{ckpt_rem}', '--shard-dir', '/content/shards', '--tag', '{tag}', '--steps', '600', '--seed', '{seed + 1000}', '--out-dir', '/content/out'{wy_flag}]\n"
                        "    from scripts.train import v3_wave_b_policy_gate\n"
                        "    v3_wave_b_policy_gate.main()\n"
                        "except Exception as _wb_exc:\n"
                        "    import traceback; traceback.print_exc()\n"
                        "    print('WAVE_B_WARN', repr(_wb_exc), flush=True)\n"
                    )

                wave_c_block = ""
                if args.run_wave_c:
                    wave_c_block = (
                        "try:\n"
                        f"    sys.argv = ['scripts/train/v3_wave_c_decoders.py', '--ckpt', '{ckpt_rem}', '--shard-dir', '/content/shards', '--tag', '{tag}', '--steps', '{int(args.wave_c_steps)}', '--seq-len', '{args.seq_len}', '--batch-size', '{min(8, args.batch_size)}', '--seed', '{seed + 2000}', '--out-dir', '/content/out'{wy_flag}]\n"
                        "    from scripts.train import v3_wave_c_decoders\n"
                        "    v3_wave_c_decoders.main()\n"
                        "except Exception as _wc_exc:\n"
                        "    import traceback; traceback.print_exc()\n"
                        "    print('WAVE_C_WARN', repr(_wc_exc), flush=True)\n"
                    )

                cleanup_ckpt_block = "" if (args.download_ckpts or args.render_media_gallery) else (
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
                # Restart kernel cleanly to guarantee 100% free GPU HBM, then run COMPLETELY DETACHED with 45s polling!
                run_colab("restart-kernel", "-s", session_name, cfg_path=cfg_path, timeout=90)
                time.sleep(5.0)
                rc, out, tripped = _run_detached_remote_script_with_polling(
                    session_name, run_py, tag=tag, cfg_path=cfg_path, log_dir=log_dir,
                    bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc_start=t_alloc,
                    poll_interval_s=45.0, max_wall_s=39600.0,
                )
                print(f"[WAVE_LAUNCH] Finished {tag} rc={rc} tripped={tripped}", flush=True)
                print(out[-2000:], flush=True)

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
                if args.render_media_gallery and out_rec.is_file() and not tripped:
                    # Render real `.wav` audio (Mimi) + `.png`/`.gif`/`.mp4` video (Wan 2.1 VAE) detached with shared GPU VRAM!
                    wb_arg = f", '--wave-b-ckpt', '/content/out/wave_b_{tag}_ckpt.pkl'" if (args.run_wave_b and (args.out_dir / f"wave_b_{tag}_receipt.json").is_file()) else ""
                    wc_arg = f", '--wave-c-ckpt', '/content/out/wave_c_{tag}_ckpt.pkl'" if (args.run_wave_c and (args.out_dir / f"wave_c_{tag}_receipt.json").is_file()) else ""
                    media_py = (
                        "import os, subprocess, sys, tarfile\n"
                        "env = dict(os.environ)\n"
                        "env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'\n"
                        "env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.45'\n"
                        "env['PYTHONPATH'] = '/content/slwm_v3'\n"
                        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers', 'transformers', 'accelerate', 'scipy', 'pillow'], check=False)\n"
                        f"out_m = '/content/out/live_media_showcase_{tag}'\n"
                        f"cmd = [sys.executable, '/content/slwm_v3/scripts/eval/v3_live_any2any_inference.py', '--ckpt', '{ckpt_rem}'{wb_arg}{wc_arg}, '--mode', 'full_gallery', '--shard-dir', '/content/shards', '--out-dir', out_m]\n"
                        "r = subprocess.run(cmd, env=env, cwd='/content/slwm_v3')\n"
                        "print('MEDIA_GALLERY_RC', r.returncode, flush=True)\n"
                        f"tar_p = '/content/out/live_media_showcase_{tag}.tar.gz'\n"
                        "if os.path.isdir(out_m):\n"
                        "    with tarfile.open(tar_p, 'w:gz') as tf:\n"
                        "        tf.add(out_m, arcname='.')\n"
                        "    print('MEDIA_TAR_BYTES', os.path.getsize(tar_p), flush=True)\n"
                    )
                    m_rc, m_out, _ = _run_detached_remote_script_with_polling(
                        session_name, media_py, tag=f"media_{tag}", cfg_path=cfg_path, log_dir=log_dir,
                        bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc_start=t_alloc,
                        poll_interval_s=20.0, max_wall_s=1200.0,
                    )
                    print(m_out[-1200:], flush=True)
                    loc_tar = args.out_dir / f"live_media_showcase_{tag}.tar.gz"
                    dl_m = run_colab(
                        "download", "-s", session_name, f"/content/out/live_media_showcase_{tag}.tar.gz",
                        str(loc_tar), cfg_path=cfg_path, timeout=240,
                    )
                    if dl_m.returncode == 0 and loc_tar.is_file():
                        dst_m = args.out_dir / f"live_media_showcase_{tag}"
                        dst_m.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(loc_tar, "r:gz") as tf:
                            tf.extractall(dst_m)
                        loc_tar.unlink(missing_ok=True)
                        print(f"[MEDIA_GALLERY OK] Extracted real audio/video gallery -> {dst_m}", flush=True)
                        _mirror_path_to_fedora(dst_m, fedora_host=args.mirror_fedora_host, fedora_dir=args.mirror_fedora_dir, profile=args.profile)

                if args.download_ckpts:
                    ckpt_dst_dir = args.ckpt_dir / args.profile
                    p_core = ckpt_dst_dir / f"{tag}_ckpt.pkl"
                    if _download_file_chunked_verified(
                        session_name, ckpt_rem, p_core,
                        cfg_path=cfg_path, log_dir=log_dir,
                        bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
                    ):
                        _mirror_path_to_fedora(p_core, fedora_host=args.mirror_fedora_host, fedora_dir=args.mirror_fedora_dir, profile=args.profile)
                    if args.run_wave_b:
                        p_wb = ckpt_dst_dir / f"wave_b_{tag}_ckpt.pkl"
                        if _download_file_chunked_verified(
                            session_name, f"/content/out/wave_b_{tag}_ckpt.pkl", p_wb,
                            cfg_path=cfg_path, log_dir=log_dir,
                            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
                        ):
                            _mirror_path_to_fedora(p_wb, fedora_host=args.mirror_fedora_host, fedora_dir=args.mirror_fedora_dir, profile=args.profile)
                    if args.run_wave_c:
                        p_wc = ckpt_dst_dir / f"wave_c_{tag}_ckpt.pkl"
                        if _download_file_chunked_verified(
                            session_name, f"/content/out/wave_c_{tag}_ckpt.pkl", p_wc,
                            cfg_path=cfg_path, log_dir=log_dir,
                            bal_start=bal_start, max_spend_cu=args.max_spend_cu, t_alloc=t_alloc,
                        ):
                            _mirror_path_to_fedora(p_wc, fedora_host=args.mirror_fedora_host, fedora_dir=args.mirror_fedora_dir, profile=args.profile)
                if tripped:
                    print("[WAVE_LAUNCH] Budget watchdog tripped; halting cleanly.", flush=True)
                    break
            if tripped:
                break
    finally:
        if created:
            ok = verified_stop_session(session_name, cfg_path)
        else:
            ok = True
        try:
            bal_end, rate_end = query_balance(cfg_path)
        except Exception as exc:
            print(f"[TEARDOWN WARN] query_balance failed during cleanup: {exc!r}", flush=True)
            bal_end, rate_end = bal_start, -1.0
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
        (args.out_dir / f"teardown_receipt_{session_name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
