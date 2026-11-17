"""Study X3 (`SLWM-1B-480p`) Autonomous Colab Accelerator Lane Runner (`TPU v6e-1 -> H100 -> A100`).

Executes balanced lanes of the 18-run (`6` cohorts x `3` paired seeds, `5,600` steps/run) Study X3 matrix:
  - Prioritizes `TPU v6e-1` (`prefer_tpu=True`) with automatic `H100` -> `A100` fallback,
  - Verifies dataset SHA-256 provenance, binding SHA-256, source-code SHA-256, and Fedora dual-copy archival fail-closed,
  - Restarts the remote kernel between jobs (and rotates VMs after 65m) to release 100% of HBM,
  - Tears down Colab VMs immediately on lane completion or error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.x3_480p_core import make_x3_spec
from scripts.train.x1b_colab_ops import (
    FEDORA_SSH,
    allocate_accelerator,
    archive_run_to_fedora,
    check_usage_and_budget,
    restart_session_kernel,
    run_kernel_job_resilient,
    stage_code_and_data,
    stop_session,
)
from scripts.train.x1b_train import sha256_file
from scripts.train.x3_train_480p import compute_x3_binding_sha256, compute_x3_source_code_sha256


def _tag_for(cohort: str, seed: int) -> str:
    slug = cohort.replace("_", "-")
    return f"{slug}-s{seed}"


def load_x3_lane_jobs(
    lane_idx: int,
    num_lanes: int = 6,
    *,
    config_rel: str = "configs/x3/study_x3_1b_480p.json",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = json.loads((REPO_ROOT / config_rel).read_text())
    sched = cfg["training_schedule"]
    all_jobs: list[dict[str, Any]] = []
    for c in cfg["cohorts"]:
        for s in c["seeds"]:
            s_int = int(s)
            tag = _tag_for(c["cohort"], s_int)
            save_bf16 = bool(c["cohort"] == "x3_slwm_1b_480p" and s_int == 3001)
            all_jobs.append(
                {
                    "tag": tag,
                    "cohort": c["cohort"],
                    "profile": c["profile"],
                    "arch": c["arch"],
                    "resolution_mode": c["resolution_mode"],
                    "seed": s_int,
                    "steps": int(c.get("steps", sched["steps"])),
                    "batch_size": int(sched["batch_size"]),
                    "seq_len": int(sched["seq_len"]),
                    "save_bf16": save_bf16,
                    "export_showcase": True,
                }
            )
    # Interleave across seeds/cohorts so every lane balances 140M/473.5M and 1.022B runs evenly
    lane_jobs = [j for idx, j in enumerate(all_jobs) if (idx % num_lanes) == lane_idx]
    return cfg, lane_jobs


def verify_x3_receipt_data_provenance(rec_doc: dict[str, Any], reg_cards: dict[str, Any]) -> bool:
    dp = rec_doc.get("data_provenance")
    if not isinstance(dp, dict) or dp.get("registered_cards_verified") is not True:
        return False
    role_map = (
        ("x3_train_480p_sha256", "x3_multimodal_480p_train"),
        ("x3_dev_480p_sha256", "x3_multimodal_480p_dev"),
        ("x3_vault_480p_sha256", "x3_multimodal_480p_vault"),
        ("grounded_robot_sha256", "x1b_wave8a_grounded_v2"),
        ("vault_robot_sha256", "x1b_wave8a_vault_v2"),
    )
    for dp_key, card_key in role_map:
        if card_key not in reg_cards or dp.get(dp_key) != reg_cards[card_key].get("sha256"):
            return False
    return True


def _try_hydrate_from_fedora_dual_copy(runs_subdir: str, tag: str, local_dir: Path) -> bool:
    """If Fedora already holds a dual-copy verified run (e.g. interrupted after mirror), pull local files cleanly."""
    chk = subprocess.run(
        [
            *FEDORA_SSH,
            f"test -f /mnt/slwm-checkpoints-a/SLWM/x1b/{shlex.quote(runs_subdir)}/{shlex.quote(tag)}/verified.json && "
            f"test -f /mnt/slwm-checkpoints-b/SLWM/x1b/{shlex.quote(runs_subdir)}/{shlex.quote(tag)}/verified.json",
        ],
        capture_output=True,
        timeout=30,
    )
    if chk.returncode != 0:
        return False
    local_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("receipt.json", "eval_metrics.json", "config.json", "train.jsonl", "verified.json"):
        txt = subprocess.check_output(
            [*FEDORA_SSH, f"cat /mnt/slwm-checkpoints-a/SLWM/x1b/{shlex.quote(runs_subdir)}/{shlex.quote(tag)}/{fn}"],
            text=True,
            timeout=60,
        )
        (local_dir / fn).write_text(txt)
    rec = json.loads((local_dir / "receipt.json").read_text())
    if rec.get("showcase_bundle_sha256"):
        sc_b = subprocess.check_output(
            [*FEDORA_SSH, f"cat /mnt/slwm-checkpoints-a/SLWM/x1b/{shlex.quote(runs_subdir)}/{shlex.quote(tag)}/showcase_bundle.npz"],
            timeout=120,
        )
        if hashlib.sha256(sc_b).hexdigest() != rec["showcase_bundle_sha256"]:
            raise RuntimeError(f"Hydrated showcase_bundle.npz SHA mismatch for {tag}")
        (local_dir / "showcase_bundle.npz").write_bytes(sc_b)
    return True


def is_verified_x3_local_job(
    job: dict[str, Any],
    *,
    runs_subdir: str,
    source_code_sha: str,
    reg_cards: dict[str, Any],
    runs_base_dir: Path,
) -> bool:
    """Fail-closed provenance verification of local + Fedora dual-copy run completion."""
    local_dir = runs_base_dir / str(job["tag"])
    ver_p = local_dir / "verified.json"
    rec_p = local_dir / "receipt.json"
    if not local_dir.is_dir() or (not ver_p.is_file() and not rec_p.is_file()):
        if not _try_hydrate_from_fedora_dual_copy(runs_subdir, str(job["tag"]), local_dir):
            return False
    if not ver_p.is_file() or not rec_p.is_file():
        raise RuntimeError(
            f"Incomplete local run directory {local_dir} for {job['tag']!r}; refusing unverified skip or silent overwrite."
        )

    ver_doc = json.loads(ver_p.read_text())
    rec_doc = json.loads(rec_p.read_text())
    spec = make_x3_spec(profile=job["profile"], arch=job["arch"], resolution_mode=job["resolution_mode"])
    exp_binding_sha = compute_x3_binding_sha256(
        tag=str(job["tag"]),
        cohort=str(job["cohort"]),
        spec=spec,
        seed=int(job["seed"]),
        steps=int(job["steps"]),
        batch_size=int(job["batch_size"]),
        seq_len=int(job["seq_len"]),
        smoke=False,
        save_bf16=bool(job["save_bf16"]),
    )
    if (
        rec_doc.get("tag") != job["tag"]
        or ver_doc.get("tag") != job["tag"]
        or int(ver_doc.get("verified_copies", 0)) < 2
        or rec_doc.get("smoke") is not False
        or int(rec_doc.get("steps", -1)) != int(job["steps"])
        or rec_doc.get("binding_sha256") != exp_binding_sha
        or rec_doc.get("source_code_sha256") != source_code_sha
        or not verify_x3_receipt_data_provenance(rec_doc, reg_cards)
    ):
        raise RuntimeError(
            f"Provenance mismatch in local verified run {local_dir} for {job['tag']!r}: "
            "binding/source/smoke/steps/dataset/dual-copy check failed."
        )

    ver_digests = ver_doc.get("sha256")
    if not isinstance(ver_digests, dict):
        raise RuntimeError(f"Malformed sha256 digest table in {ver_p} for {job['tag']!r}")

    local_files = [
        ("receipt.json", ver_digests.get("receipt.json")),
        ("eval_metrics.json", rec_doc.get("eval_metrics_sha256") or ver_digests.get("eval_metrics.json")),
        ("config.json", rec_doc.get("config_file_sha256") or ver_digests.get("config.json")),
        ("train.jsonl", rec_doc.get("train_log_sha256") or ver_digests.get("train.jsonl")),
    ]
    if job["export_showcase"]:
        local_files.append(("showcase_bundle.npz", rec_doc.get("showcase_bundle_sha256")))

    for fname, exp_sha in local_files:
        fp = local_dir / fname
        if not fp.is_file() or not isinstance(exp_sha, str) or len(exp_sha) != 64:
            raise RuntimeError(f"Missing or unhashed local file {fp} in {local_dir} for {job['tag']!r}")
        got_sha = sha256_file(fp)
        if got_sha != exp_sha or ver_digests.get(fname) != got_sha:
            raise RuntimeError(f"Digest mismatch on {fp} for {job['tag']!r}: got={got_sha}, expected={exp_sha}")

    int8_sha = rec_doc.get("portable_int8_sha256")
    if not isinstance(int8_sha, str) or len(int8_sha) != 64 or ver_digests.get("portable_int8.npz") != int8_sha:
        raise RuntimeError(f"Unverified portable_int8.npz digest in {local_dir} for {job['tag']!r}")
    if job["save_bf16"]:
        bf16_sha = rec_doc.get("weights_bf16_sha256")
        if not isinstance(bf16_sha, str) or len(bf16_sha) != 64 or ver_digests.get("weights_bf16.npz") != bf16_sha:
            raise RuntimeError(f"Unverified weights_bf16.npz digest in {local_dir} for {job['tag']!r}")
    return True


def run_lane(
    lane_idx: int,
    num_lanes: int = 6,
    *,
    config_rel: str = "configs/x3/study_x3_1b_480p.json",
    prefer_tpu: bool = True,
    tpu_only: bool = False,
    max_spend: float = 2495.0,
    min_balance_reserve: float = 5.0,
) -> None:
    cfg, jobs = load_x3_lane_jobs(lane_idx, num_lanes=num_lanes, config_rel=config_rel)
    reg_cards = json.loads((REPO_ROOT / "configs/x3/x3_dataset_cards.json").read_text())
    source_sha = compute_x3_source_code_sha256()
    runs_subdir = str(cfg["runs_subdir"])
    local_runs_root = REPO_ROOT / str(cfg["exp_dir"]) / runs_subdir
    local_runs_root.mkdir(parents=True, exist_ok=True)
    session_name = f"slwm-x1b-x3l{lane_idx + 1}"

    def _is_verified(job_spec: dict[str, Any]) -> bool:
        return is_verified_x3_local_job(
            job_spec,
            runs_subdir=runs_subdir,
            source_code_sha=source_sha,
            reg_cards=reg_cards,
            runs_base_dir=local_runs_root,
        )

    if all(_is_verified(j) for j in jobs):
        print(f"[X3 LANE {lane_idx}] All {len(jobs)} jobs already provenance-verified; exiting.", flush=True)
        return

    check_usage_and_budget(max_spend=max_spend, min_balance_reserve=min_balance_reserve)

    def _alloc_and_stage() -> float:
        for boot_try in range(1, 4):
            alloc = allocate_accelerator(
                session_name,
                prefer_tpu=prefer_tpu,
                tpu_retries=2,
                allow_h100_fallback=(not tpu_only),
                tpu_only=tpu_only,
            )
            t0 = time.monotonic()
            print(f"[X3 LANE {lane_idx}] Allocated {session_name}: {alloc}", flush=True)
            try:
                stage_code_and_data(session_name, include_x3_480p=True)
                return t0
            except Exception as exc:
                if boot_try == 3:
                    raise
                print(f"[WARN X3 LANE {lane_idx}] Staging failed (try {boot_try}/3: {exc}); recycling VM...", flush=True)
                stop_session(session_name)
                time.sleep(8)
        raise RuntimeError("Unreachable")

    session_started_at = _alloc_and_stage()
    try:
        jobs_on_session = 0
        for ji, job in enumerate(jobs, start=1):
            check_usage_and_budget(max_spend=max_spend, min_balance_reserve=min_balance_reserve)
            if _is_verified(job):
                print(f"[X3 LANE {lane_idx}] ({ji}/{len(jobs)}) {job['tag']} already provenance-verified, skipping.", flush=True)
                continue

            age_s = time.monotonic() - session_started_at
            if jobs_on_session >= 1 and age_s >= 3900:
                print(f"[ROTATE X3 LANE {lane_idx}] Session age={age_s / 60:.1f}m >= 65m; rotating VM cleanly...", flush=True)
                stop_session(session_name)
                time.sleep(4)
                session_started_at = _alloc_and_stage()
                jobs_on_session = 0
            elif jobs_on_session >= 1:
                print(f"[X3 LANE {lane_idx}] Restarting kernel on {session_name} before {job['tag']} to clear HBM...", flush=True)
                restart_session_kernel(session_name)
                time.sleep(4)

            py_job = f"""
import gc, glob, os, pathlib, sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.92'
for _old_npz in glob.glob('/content/x1b-out/*/*.npz') + glob.glob('/content/x1b-out/*/.*.npz'):
    if pathlib.Path(_old_npz).parent.name != {job['tag']!r}:
        try:
            os.remove(_old_npz)
        except OSError:
            pass
if '/content/x1b-code' not in sys.path:
    sys.path.insert(0, '/content/x1b-code')
import jax
jax.config.update('jax_compilation_cache_dir', '/content/x1b-cache')
from scripts.train.x3_train_480p import train_and_evaluate_x3

train_and_evaluate_x3(
    tag={job['tag']!r},
    cohort={job['cohort']!r},
    profile={job['profile']!r},
    arch={job['arch']!r},
    resolution_mode={job['resolution_mode']!r},
    seed={int(job['seed'])},
    steps={int(job['steps'])},
    batch_size={int(job['batch_size'])},
    seq_len={int(job['seq_len'])},
    out_dir=pathlib.Path('/content/x1b-out'),
    train_480p_path='/content/x1b-data/x3_multimodal_480p_train.npz',
    dev_480p_path='/content/x1b-data/x3_multimodal_480p_dev.npz',
    vault_480p_path='/content/x1b-data/x3_multimodal_480p_vault.npz',
    grounded_robot_path='/content/x1b-data/x1b_wave8a_grounded_v2.npz',
    vault_robot_path='/content/x1b-data/x1b_wave8a_vault_v2.npz',
    vocab_path='/content/x1b-data/vocab_4096_ext.json',
    save_bf16={bool(job['save_bf16'])},
    export_showcase={bool(job['export_showcase'])},
    smoke=False,
)
gc.collect()
"""
            print(f"[X3 LANE {lane_idx}] ({ji}/{len(jobs)}) Training {job['tag']} on {session_name}...", flush=True)
            run_kernel_job_resilient(session_name, job["tag"], py_job, timeout=14400)
            archive_run_to_fedora(
                session_name,
                job["tag"],
                runs_subdir=runs_subdir,
                local_runs_base_dir=local_runs_root,
            )
            if not _is_verified(job):
                raise RuntimeError(f"Post-archive provenance check failed for {job['tag']!r}")
            jobs_on_session += 1

        fin = check_usage_and_budget(max_spend=max_spend, min_balance_reserve=min_balance_reserve)
        print(f"[X3 LANE {lane_idx}] ALL {len(jobs)} JOBS COMPLETED. Balance={fin['current_balance']}", flush=True)
    finally:
        print(f"[X3 LANE {lane_idx}] Stopping Colab session {session_name} immediately...", flush=True)
        try:
            stop_session(session_name)
        except Exception as exc:
            print(f"[WARN] stop_session({session_name}) cleanup error: {exc}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--num-lanes", type=int, default=6)
    ap.add_argument("--config", type=str, default="configs/x3/study_x3_1b_480p.json")
    ap.add_argument("--gpu-fallback-first", action="store_true")
    ap.add_argument("--tpu-only", action="store_true")
    args = ap.parse_args()
    run_lane(
        args.lane,
        num_lanes=args.num_lanes,
        config_rel=args.config,
        prefer_tpu=(not args.gpu_fallback_first),
        tpu_only=args.tpu_only,
    )


if __name__ == "__main__":
    main()
