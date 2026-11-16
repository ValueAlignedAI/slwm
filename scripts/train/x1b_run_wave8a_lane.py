"""Autonomous Colab Accelerator Lane Runner for Study X2 Wave 8A (`SLWM-X2`)."""

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

from models.x1b_wave8a_core import Wave8AInterventions
from scripts.eval.x1b_analyze_wave8a import verify_receipt_data_provenance
from scripts.train.x1b_colab_ops import (
    allocate_accelerator,
    archive_run_to_fedora,
    check_usage_and_budget,
    restart_session_kernel,
    run_kernel_job_resilient,
    stage_code_and_data,
    stop_session,
)
from scripts.train.x1b_train import sha256_file
from scripts.train.x1b_train_wave8a import compute_wave8a_binding_sha256, compute_wave8a_source_code_sha256


def build_wave8a_lanes(
    study_cfg: dict[str, Any],
    *,
    num_lanes: int = 3,
) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    """Partition the 5 Wave 8A cohorts (15 total 453M runs) evenly across parallel Colab lanes."""
    cohorts = study_cfg["cohorts_453m_3seed"]
    sched = study_cfg.get("training_schedule", {})
    def_steps = int(sched.get("steps", 1800))
    def_bs = int(sched.get("batch_size", 8))
    def_seq = int(sched.get("seq_len", 64))
    all_jobs: list[dict[str, Any]] = []
    for c in cohorts:
        iv = c["interventions"]
        for seed in c["seeds"]:
            s_int = int(seed)
            tag = f"x1b-{c['cohort']}-s{s_int}"
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
                    "zmem_isolation": bool(iv["zmem_isolation"]),
                    "enable_assoc_loss": bool(iv["enable_assoc_loss"]),
                    "seed": s_int,
                    "steps": int(c.get("steps", def_steps)),
                    "batch_size": int(c.get("batch_size", def_bs)),
                    "seq_len": int(c.get("seq_len", def_seq)),
                    "save_bf16": bool(c["cohort"] == "w8a_slwm_x2_isolated_450m" and s_int == 2001),
                    "export_showcase": True,
                }
            )

    lanes: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for li in range(num_lanes):
        lname = f"w8a_lane{li + 1}"
        sname = f"slwm-x1b-w8al{li + 1}"
        lanes[lname] = (sname, all_jobs[li::num_lanes])
    return lanes


def is_verified_wave8a_local_job(
    job: dict[str, Any],
    *,
    runs_subdir: str,
    source_code_sha: str,
    reg_cards: dict[str, Any],
    runs_base_dir: Path | None = None,
) -> bool:
    """Verify local completion & Fedora dual-copy archival provenance fail-closed (never skip on bare verified.json)."""
    base_d = (
        runs_base_dir
        if runs_base_dir is not None
        else (REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111" / runs_subdir)
    )
    local_dir = base_d / str(job["tag"])
    ver_p = local_dir / "verified.json"
    rec_p = local_dir / "receipt.json"
    if not local_dir.is_dir() or (not ver_p.is_file() and not rec_p.is_file()):
        return False
    if not ver_p.is_file() or not rec_p.is_file():
        raise RuntimeError(
            f"Incomplete local run directory {local_dir} for {job['tag']!r} (partial verified.json/receipt.json); "
            "refusing unverified skip or silent overwrite."
        )

    ver_doc = json.loads(ver_p.read_text())
    rec_doc = json.loads(rec_p.read_text())
    iv = Wave8AInterventions(
        key_shield=bool(job["key_shield"]),
        zerolag_bypass=bool(job["zerolag_bypass"]),
        decoupled_heads=bool(job["decoupled_heads"]),
        zmem_isolation=bool(job["zmem_isolation"]),
        enable_assoc_loss=bool(job["enable_assoc_loss"]),
        regularizer=str(job["regularizer"]),
    )
    exp_binding_sha = compute_wave8a_binding_sha256(
        tag=str(job["tag"]),
        cohort=str(job["cohort"]),
        profile=str(job["profile"]),
        arch=str(job["arch"]),
        interventions=iv,
        seed=int(job["seed"]),
        steps=int(job["steps"]),
        batch_size=int(job["batch_size"]),
        seq_len=int(job["seq_len"]),
        smoke=False,
        save_bf16=bool(job["save_bf16"]),
        export_showcase=bool(job["export_showcase"]),
    )
    if (
        rec_doc.get("tag") != job["tag"]
        or ver_doc.get("tag") != job["tag"]
        or int(ver_doc.get("verified_copies", 0)) < 2
        or rec_doc.get("smoke") is not False
        or int(rec_doc.get("steps", -1)) != int(job["steps"])
        or rec_doc.get("binding_sha256") != exp_binding_sha
        or rec_doc.get("source_code_sha256") != source_code_sha
        or not verify_receipt_data_provenance(rec_doc, reg_cards)
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


def run_wave8a_lane(
    lane_key: str,
    *,
    config_rel: str = "configs/x1b/study_x2_wave8a.json",
    num_lanes: int = 3,
    prefer_tpu: bool = True,
    tpu_only: bool = False,
) -> None:
    study_cfg = json.loads((REPO_ROOT / config_rel).read_text())
    reg_cards = json.loads((REPO_ROOT / "configs/x1b/wave8a_dataset_cards.json").read_text())
    source_code_sha = compute_wave8a_source_code_sha256()
    runs_subdir = str(study_cfg.get("runs_subdir", "runs_wave8a"))
    max_spend = float(study_cfg["budget"]["max_compute_units"])
    min_res = float(study_cfg["budget"].get("min_balance_reserve", 3.0))

    lanes = build_wave8a_lanes(study_cfg, num_lanes=num_lanes)
    if lane_key not in lanes:
        raise ValueError(f"Unknown Wave 8A lane {lane_key!r}; expected one of {sorted(lanes)}")
    session_name, jobs = lanes[lane_key]

    def _is_verified(job_spec: dict[str, Any]) -> bool:
        return is_verified_wave8a_local_job(
            job_spec,
            runs_subdir=runs_subdir,
            source_code_sha=source_code_sha,
            reg_cards=reg_cards,
        )

    if all(_is_verified(j) for j in jobs):
        print(f"[W8A {lane_key}] All {len(jobs)} jobs already provenance-verified locally; exiting.", flush=True)
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
            print(f"[W8A {lane_key}] Allocated: {alloc}", flush=True)
            try:
                stage_code_and_data(
                    session_name,
                    include_real_robot=False,
                    include_wave6_grounded=False,
                    include_wave8a_grounded=True,
                    real_bank_name=str(study_cfg.get("real_bank_name", "x1b_real_bank_v2")),
                )
                return t0
            except Exception as exc:
                if boot_try == 3:
                    raise
                print(f"[WARN W8A {lane_key}] Staging failed (try {boot_try}/3: {exc}); recycling VM...", flush=True)
                stop_session(session_name)
                time.sleep(8)
        raise RuntimeError("Unreachable")

    session_started_at = _alloc_and_stage()
    try:
        jobs_on_session = 0
        for idx, job in enumerate(jobs, start=1):
            check_usage_and_budget(max_spend=max_spend, min_balance_reserve=min_res)
            if _is_verified(job):
                print(f"[W8A {lane_key}] ({idx}/{len(jobs)}) {job['tag']} already provenance-verified, skipping.", flush=True)
                continue

            age_s = time.monotonic() - session_started_at
            if jobs_on_session >= 1 and age_s >= 3900:
                print(f"[ROTATE W8A {lane_key}] Session age={age_s / 60:.1f}m >= 65m; rotating VM cleanly...", flush=True)
                stop_session(session_name)
                time.sleep(4)
                session_started_at = _alloc_and_stage()
                jobs_on_session = 0
            elif jobs_on_session >= 1:
                restart_session_kernel(session_name)

            print(f"[W8A {lane_key}] ({idx}/{len(jobs)}) Training {job['tag']} on {session_name}...", flush=True)
            py_job = f"""
import gc, glob, os, pathlib, sys
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
from scripts.train.x1b_train_wave8a import train_and_evaluate_wave8a
train_and_evaluate_wave8a(
    tag={job['tag']!r},
    cohort={job['cohort']!r},
    profile={job['profile']!r},
    arch={job['arch']!r},
    regularizer={job['regularizer']!r},
    key_shield={bool(job['key_shield'])},
    zerolag_bypass={bool(job['zerolag_bypass'])},
    decoupled_heads={bool(job['decoupled_heads'])},
    zmem_isolation={bool(job['zmem_isolation'])},
    enable_assoc_loss={bool(job['enable_assoc_loss'])},
    seed={int(job['seed'])},
    steps={int(job['steps'])},
    batch_size={int(job['batch_size'])},
    seq_len={int(job['seq_len'])},
    out_dir=pathlib.Path('/content/x1b-out'),
    real_bank_path='/content/x1b-data/x1b_real_bank_v2.npz',
    grounded_robot_path='/content/x1b-data/x1b_wave8a_grounded_v2.npz',
    vault_robot_path='/content/x1b-data/x1b_wave8a_vault_v2.npz',
    save_bf16={bool(job['save_bf16'])},
    export_showcase={bool(job['export_showcase'])},
    allow_verified_resume=True,
)
gc.collect()
"""
            run_kernel_job_resilient(session_name, job["tag"], py_job, timeout=8100)
            archive_run_to_fedora(session_name, job["tag"], runs_subdir=runs_subdir)
            jobs_on_session += 1

        fin = check_usage_and_budget(max_spend=max_spend, min_balance_reserve=min_res)
        print(f"[W8A {lane_key}] ALL {len(jobs)} JOBS COMPLETED. Balance={fin['current_balance']}", flush=True)
    finally:
        stop_session(session_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lane")
    ap.add_argument("--config", default="configs/x1b/study_x2_wave8a.json")
    ap.add_argument("--num-lanes", type=int, default=3)
    ap.add_argument("--tpu-only", action="store_true")
    args = ap.parse_args()
    run_wave8a_lane(args.lane, config_rel=args.config, num_lanes=args.num_lanes, tpu_only=args.tpu_only)


if __name__ == "__main__":
    main()
