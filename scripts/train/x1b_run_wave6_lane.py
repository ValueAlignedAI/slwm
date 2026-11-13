"""Autonomous Colab TPU v6e1 / GPU Lane Runner for Wave 6 Zero-Compromise Grounded Study (`SLWM-v2`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train.x1b_colab_ops import (
    allocate_accelerator,
    archive_run_to_fedora,
    check_usage_and_budget,
    restart_session_kernel,
    run_kernel_job_resilient,
    stage_code_and_data,
    stop_session,
)


def build_wave6_lanes(study_cfg: dict[str, Any]) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    """Partition the 8 Wave 6 cohorts (24 total 453M runs) evenly across 3 parallel Colab lanes (8 runs/lane)."""
    cohorts = study_cfg["cohorts_453m_3seed"]
    all_jobs: list[dict[str, Any]] = []
    for c in cohorts:
        iv = c["interventions"]
        for seed in c["seeds"]:
            tag = f"x1b-{c['cohort']}-s{seed}"
            all_jobs.append(
                {
                    "tag": tag,
                    "cohort": c["cohort"],
                    "profile": c["profile"],
                    "arch": c["arch"],
                    "regularizer": c["regularizer"],
                    "key_shield": bool(iv["key_shield"]),
                    "zerolag_bypass": bool(iv["zerolag_bypass"]),
                    "decoupled_heads": bool(iv["decoupled_heads"]),
                    "seed": int(seed),
                    "steps": int(c.get("steps", 2600)),
                    "batch_size": 12,
                    "seq_len": 32,
                    "save_bf16": bool(c["cohort"] == "w6_slwm_450m_full_epps_sigreg" and int(seed) == 2001),
                }
            )

    lanes: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for li in range(3):
        lname = f"w6_lane{li + 1}"
        sname = f"slwm-x1b-w6l{li + 1}"
        lanes[lname] = (sname, all_jobs[li::3])
    return lanes


def run_wave6_lane(
    lane_key: str,
    *,
    config_rel: str = "configs/x1b/study_x1b_wave6.json",
    prefer_tpu: bool = True,
    tpu_only: bool = False,
) -> None:
    study_cfg = json.loads((REPO_ROOT / config_rel).read_text())
    runs_subdir = str(study_cfg.get("runs_subdir", "runs_wave6"))
    lanes = build_wave6_lanes(study_cfg)
    if lane_key not in lanes:
        raise ValueError(f"Unknown Wave 6 lane {lane_key!r}; expected one of {sorted(lanes)}")
    session_name, jobs = lanes[lane_key]

    def _is_verified(tag: str) -> bool:
        return (
            REPO_ROOT
            / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111"
            / runs_subdir
            / tag
            / "verified.json"
        ).is_file()

    if all(_is_verified(j["tag"]) for j in jobs):
        print(f"[W6 {lane_key}] All {len(jobs)} jobs already verified locally; exiting.", flush=True)
        return

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
            print(f"[W6 {lane_key}] Allocated: {alloc}", flush=True)
            try:
                stage_code_and_data(session_name, include_real_robot=False, include_wave6_grounded=True)
                return t0
            except Exception as exc:
                if boot_try == 3:
                    raise
                print(f"[WARN W6 {lane_key}] Staging failed (try {boot_try}/3: {exc}); recycling VM...", flush=True)
                stop_session(session_name)
                time.sleep(8)
        raise RuntimeError("Unreachable")

    session_started_at = _alloc_and_stage()
    try:
        jobs_on_session = 0
        for idx, job in enumerate(jobs, start=1):
            check_usage_and_budget(max_spend=float(study_cfg["budget"]["max_compute_units"]))
            if _is_verified(job["tag"]):
                print(f"[W6 {lane_key}] ({idx}/{len(jobs)}) {job['tag']} already verified, skipping.", flush=True)
                continue

            age_s = time.monotonic() - session_started_at
            if jobs_on_session >= 1 and age_s >= 3900:
                print(f"[ROTATE W6 {lane_key}] Session age={age_s / 60:.1f}m >= 65m; rotating VM cleanly...", flush=True)
                stop_session(session_name)
                time.sleep(4)
                session_started_at = _alloc_and_stage()
                jobs_on_session = 0
            elif jobs_on_session >= 1:
                restart_session_kernel(session_name)

            print(f"[W6 {lane_key}] ({idx}/{len(jobs)}) Training {job['tag']} on {session_name}...", flush=True)
            py_job = f"""
import gc, glob, os, pathlib, sys
for _old_npz in glob.glob('/content/x1b-out/*/*.npz'):
    try:
        os.remove(_old_npz)
    except OSError:
        pass
if '/content/x1b-code' not in sys.path:
    sys.path.insert(0, '/content/x1b-code')
import jax
jax.config.update('jax_compilation_cache_dir', '/content/x1b-cache')
from scripts.train.x1b_train_wave6 import train_and_evaluate_wave6
train_and_evaluate_wave6(
    tag={job['tag']!r},
    profile={job['profile']!r},
    arch={job['arch']!r},
    regularizer={job['regularizer']!r},
    key_shield={bool(job['key_shield'])},
    zerolag_bypass={bool(job['zerolag_bypass'])},
    decoupled_heads={bool(job['decoupled_heads'])},
    seed={int(job['seed'])},
    steps={int(job['steps'])},
    batch_size={int(job['batch_size'])},
    seq_len={int(job['seq_len'])},
    out_dir=pathlib.Path('/content/x1b-out'),
    real_bank_path='/content/x1b-data/x1b_real_bank_v1.npz',
    grounded_robot_path='/content/x1b-data/x1b_wave6_grounded_v1.npz',
    vault_robot_path='/content/x1b-data/x1b_wave6_vault_test.npz',
    save_bf16={bool(job['save_bf16'])},
)
gc.collect()
"""
            run_kernel_job_resilient(session_name, job["tag"], py_job, timeout=7200)
            archive_run_to_fedora(session_name, job["tag"], runs_subdir=runs_subdir)
            jobs_on_session += 1

        fin = check_usage_and_budget(max_spend=float(study_cfg["budget"]["max_compute_units"]))
        print(f"[W6 {lane_key}] ALL {len(jobs)} JOBS COMPLETED. Balance={fin['current_balance']}", flush=True)
    finally:
        stop_session(session_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lane")
    ap.add_argument("--config", default="configs/x1b/study_x1b_wave6.json")
    ap.add_argument("--tpu-only", action="store_true")
    args = ap.parse_args()
    run_wave6_lane(args.lane, config_rel=args.config, tpu_only=args.tpu_only)


if __name__ == "__main__":
    main()
