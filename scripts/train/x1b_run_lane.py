"""End-to-end autonomous Colab TPU v6e1 / GPU lane runner with dual-HDD archival & auto-stop."""

from __future__ import annotations

import argparse
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

from scripts.train.x1b_colab_ops import (
    allocate_accelerator,
    archive_run_to_fedora,
    check_usage_and_budget,
    colab_env,
    restart_session_kernel,
    run_kernel_job_resilient,
    ssh_cmd_base,
    stage_code_and_data,
    stop_session,
)


def expand_lane_jobs(study_cfg: dict[str, Any], lane_key: str) -> tuple[str, list[dict[str, Any]]]:
    lane = study_cfg["lanes"][lane_key]
    session_name = lane["session"]
    if "runs" in lane:
        jobs = []
        for r in lane["runs"]:
            jobs.append(
                {
                    "tag": r["tag"],
                    "profile": r["profile"],
                    "arch": r["arch"],
                    "objective": r["objective"],
                    "seed": int(r["seed"]),
                    "steps": int(r["steps"]),
                    "batch_size": int(r["batch_size"]),
                    "seq_len": int(r["seq_len"]),
                    "save_bf16": bool(r.get("save_bf16", int(r["seed"]) == 2001)),
                }
            )
        return session_name, jobs

    jobs = []
    for var in lane["variants"]:
        for seed in lane["seeds"]:
            tag = f"x1b-abl-{var['name']}-s{seed}"
            jobs.append(
                {
                    "tag": tag,
                    "profile": lane["profile"],
                    "arch": var["arch"],
                    "objective": var["objective"],
                    "seed": int(seed),
                    "steps": int(lane["steps"]),
                    "batch_size": int(lane["batch_size"]),
                    "seq_len": int(lane["seq_len"]),
                    "save_bf16": (int(seed) == 2001),
                }
            )
    return session_name, jobs


def run_lanes(
    lane_keys: list[str],
    *,
    config_rel: str = "configs/x1b/study_x1b.json",
    prefer_tpu: bool = True,
    tpu_only: bool = False,
    smoke: bool = False,
) -> None:
    cfg_path = REPO_ROOT / config_rel
    study_cfg = json.loads(cfg_path.read_text())
    runs_subdir = str(study_cfg.get("runs_subdir", "runs"))
    use_real_robot = bool(study_cfg.get("use_real_robot", False))

    session_name: str | None = None
    jobs: list[dict[str, Any]] = []
    for lk in lane_keys:
        if lk not in study_cfg["lanes"]:
            raise ValueError(f"Unknown lane {lk!r} in {config_rel}; available: {sorted(study_cfg['lanes'])}")
        s_name, l_jobs = expand_lane_jobs(study_cfg, lk)
        if session_name is None:
            session_name = s_name
        jobs.extend(l_jobs)
    assert session_name is not None

    lane_title = "+".join(lane_keys)
    if smoke:
        session_name = f"{session_name}-smoke"
        jobs = [{**jobs[0], "tag": f"x1b-smoke-{lane_keys[0]}", "profile": "x_tiny", "steps": 6, "batch_size": 4}]

    def _is_verified(tag: str) -> bool:
        return (
            REPO_ROOT
            / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111"
            / runs_subdir
            / tag
            / "verified.json"
        ).is_file()

    if not smoke and all(_is_verified(j["tag"]) for j in jobs):
        print(f"[LANE {lane_title}] All {len(jobs)} jobs already archived & verified locally; exiting.", flush=True)
        return

    use_wave6_grounded = bool(study_cfg.get("use_wave6_grounded", False))

    def _alloc_and_stage() -> float:
        for boot_try in range(1, 4):
            alloc = allocate_accelerator(
                session_name,
                prefer_tpu=prefer_tpu,
                tpu_retries=2,
                allow_h100_fallback=(not tpu_only),
                tpu_only=tpu_only,
            )
            t_alloc = time.monotonic()
            print(f"[LANE {lane_title}] Allocated: {alloc}", flush=True)
            try:
                stage_code_and_data(
                    session_name,
                    include_real_robot=use_real_robot,
                    include_wave6_grounded=use_wave6_grounded,
                )
                return t_alloc
            except Exception as exc:
                if boot_try == 3:
                    raise
                print(
                    f"[WARN {lane_title}] Staging failed on fresh session {session_name} (try {boot_try}/3: {exc}); recycling VM...",
                    flush=True,
                )
                stop_session(session_name)
                time.sleep(8)
        raise RuntimeError(f"Unreachable boot loop for {session_name}")

    session_started_at = _alloc_and_stage()

    try:
        prev_compile_sig: tuple[Any, ...] | None = None
        jobs_run_on_session = 0
        robot_arg = "'/content/x1b-data/x1b_real_robot_v2.npz'" if use_real_robot else "None"

        for idx, job in enumerate(jobs, start=1):
            check_usage_and_budget(max_spend=float(study_cfg["budget"]["max_compute_units"]))
            if not smoke and _is_verified(job["tag"]):
                print(f"[LANE {lane_title}] ({idx}/{len(jobs)}) {job['tag']} already archived & verified, skipping.", flush=True)
                continue

            # Proactively rotate Colab VM lease before starting a new job if current lease is >= 65 min old!
            # Prevents hitting Colab's ~3.2h VM reclamation boundary and upgrades GPU fallback lanes to freed TPUs.
            session_age_s = time.monotonic() - session_started_at
            if jobs_run_on_session >= 1 and session_age_s >= 3900:
                print(
                    f"[ROTATE {lane_title}] Session {session_name} age={session_age_s / 60:.1f}m >= 65m; "
                    f"rotating lease cleanly before {job['tag']}...",
                    flush=True,
                )
                stop_session(session_name)
                time.sleep(4)
                session_started_at = _alloc_and_stage()
                prev_compile_sig = None
                jobs_run_on_session = 0

            print(f"[LANE {lane_title}] ({idx}/{len(jobs)}) Training {job['tag']} on {session_name}...", flush=True)
            cur_sig = (job["profile"], job["arch"], job["objective"], job["batch_size"], job["seq_len"])
            if prev_compile_sig is not None and (cur_sig != prev_compile_sig or job["profile"] in ("x1b", "x450m")):
                restart_session_kernel(session_name)
            prev_compile_sig = cur_sig
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
from scripts.train.x1b_train import train_and_evaluate
train_and_evaluate(
    tag={job['tag']!r},
    profile={job['profile']!r},
    arch={job['arch']!r},
    objective={job['objective']!r},
    seed={int(job['seed'])},
    steps={int(job['steps'])},
    batch_size={int(job['batch_size'])},
    seq_len={int(job['seq_len'])},
    out_dir=pathlib.Path('/content/x1b-out'),
    real_bank_path='/content/x1b-data/x1b_real_bank_v1.npz',
    real_robot_path={robot_arg},
    pymunk_panels_path='/content/x1b-data/x1b_pymunk_panels_64.npz',
    save_bf16={bool(job['save_bf16'])},
)
gc.collect()
"""
            run_kernel_job_resilient(session_name, job["tag"], py_job, timeout=9000)
            archive_run_to_fedora(session_name, job["tag"], runs_subdir=runs_subdir)
            jobs_run_on_session += 1

        final_ledger = check_usage_and_budget(max_spend=float(study_cfg["budget"]["max_compute_units"]))
        print(
            f"[LANE {lane_title}] ALL {len(jobs)} JOBS COMPLETED & ARCHIVED. Balance={final_ledger['current_balance']}",
            flush=True,
        )
    finally:
        print(f"[LANE {lane_title}] Releasing Colab session {session_name}...", flush=True)
        stop_session(session_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/x1b/study_x1b.json")
    ap.add_argument("lanes", nargs="+")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--tpu-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_lanes(
        args.lanes,
        config_rel=args.config,
        prefer_tpu=(not args.gpu_only),
        tpu_only=args.tpu_only,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
