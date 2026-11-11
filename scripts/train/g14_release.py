"""Assemble the closed-loop release from measured telemetry, model artifacts and verified disk reads."""

import json
from pathlib import Path
import shutil
import subprocess
from scripts.train.g9_storage import write, sha, disks

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def read(p):
    return json.loads(p.read_bytes())


def copy(p, q):
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_suffix(q.suffix + ".new")
    shutil.copyfile(p, tmp)
    if sha(p) != sha(tmp):
        raise ValueError("Published artifact checksum changed")
    tmp.replace(q)


def mission_assets():
    from scripts.train.g14_stages import await_files

    work = ROOT / "work"
    out = ROOT / "release"
    ex = out / "examples"
    ex.mkdir(exist_ok=True, parents=True)
    await_files([work / "missions/summary.json"])
    (out / "metrics").mkdir(exist_ok=True)
    copy(work / "missions/summary.json", out / "metrics/missions.json")
    movies = {}
    for i in (0, 8):
        n = f"mission-{i:02d}"
        p = work / "missions" / (n + ".mp4")
        copy(p, ex / (n + ".mp4"))
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-ss",
                "25",
                "-i",
                str(p),
                "-frames:v",
                "1",
                "-update",
                "1",
                str(ex / (n + "-preview.png")),
            ],
            check=True,
        )
        copy(work / "missions" / (n + ".npz"), ex / (n + ".npz"))
        d = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type,codec_name,width,height",
                    "-of",
                    "json",
                    str(p),
                ]
            )
        )
        expected = read(work / "missions" / (n + ".json"))["sequences"][
            "spatial-carry"
        ]["duration_seconds"]
        if (
            len(d["streams"]) != 2
            or abs(float(d["format"]["duration"]) - expected) > 0.12
            or expected <= 30
        ):
            raise ValueError("Incomplete continuous audiovisual artifact")
        movies[n] = dict(
            seconds=float(d["format"]["duration"]),
            scheduled_seconds=expected,
            sha256=sha(p),
            streams=d["streams"],
        )
    raw = [read(p) for p in sorted((work / "missions").glob("mission-*.json"))]
    write(ex / "mission-outcomes.json", raw)
    write(ex / "mission-videos.json", movies)
    return movies


def assemble():
    from scripts.train.g14_stages import await_files

    await_files(
        [
            ROOT / "work/throughput-post.json",
            ROOT / "work/training-transfer.json",
            ROOT / "work/source-sensitivity.json",
            ROOT / "release/metrics/held-prediction.json",
        ]
    )
    out = ROOT / "release"
    m = out / "metrics"
    ex = out / "examples"
    ex.mkdir(exist_ok=True)
    work = ROOT / "work"
    for source, name in (
        ("missions/summary.json", "missions.json"),
        ("audit-traces/audit.json", "trajectory-replay.json"),
        ("field-replay.json", "field-replay.json"),
        ("speech-replay.json", "speech-replay.json"),
        ("telemetry.json", "telemetry.json"),
        ("throughput-post.json", "throughput.json"),
        ("throughput.json", "throughput-active.json"),
        ("isolation-replay.json", "isolation-replay.json"),
        ("reset-replay.json", "reset-replay.json"),
        ("training-label-audit.json", "training-label-audit.json"),
        ("isolated-label-audit.json", "isolated-label-audit.json"),
        ("training-fidelity-decision.json", "ordinary-label-decision.json"),
        ("final-fidelity-decision.json", "final-fidelity-decision.json"),
        ("released-lock.json", "published-panel.json"),
        ("compute.json", "compute.json"),
        ("drive-probe.json", "drive-probe.json"),
        ("acoustic-abstention.json", "acoustic-abstention.json"),
        ("routing-audit.json", "routing-audit.json"),
        ("training-transfer.json", "training-transfer.json"),
        ("partition-audit.json", "partition-audit.json"),
        ("source-sensitivity.json", "source-sensitivity.json"),
    ):
        copy(work / source, m / name)
    copy(work / "audit-traces/native-traces.npz", ex / "native-traces.npz")
    movies = mission_assets()
    cost = read(work / "compute.json")
    stats = read(m / "physical-results.json")
    storage = read(m / "custody.json")
    audio = read(work / "speech-results.json")
    replay = read(m / "trajectory-replay.json")
    if replay["same_selective_scores"] != 48 or replay["identical_digests"] != 48:
        raise ValueError("Independent isolated trajectory replay disagrees")
    if storage["generations"] != 225 or storage["copies"] != 2:
        raise ValueError("Incomplete matched training inventory")
    pkg = read(out / "inference/manifest.json")
    for name, v in pkg["files"].items():
        if sha(out / "inference" / name) != v["sha256"]:
            raise ValueError("Export check failed: " + name)
    fresh = stats["cohorts"]["fresh"]
    older = stats["cohorts"]["held"]
    if stats["shared_processor_superiority"]:
        nxt = "Test the contact controller prospectively on new object geometry and a second physical task, with the fixed-field and geometric controls retained."
    else:
        nxt = "Train a per-object motion probability with an explicit stationary outcome and a conditional displacement head. Use predicted collateral motion to rank pushes, then test against the current geometry, dense and fixed-field controls on new object shapes and materials. Calibrate the actuator on separate real recordings before a physical placement trial."
    base = tuple(p.parent.parent / "g14" for p in disks())
    state = (
        read(base[0] / "drive-state.json")
        if (base[0] / "drive-state.json").is_file()
        else {"phase": "queued", "verified_files": 0}
    )
    receipt = dict(
        complete=True,
        fits=75,
        speech_fits=15,
        physical_fits=60,
        generations=storage["generations"],
        counts=storage["groups"],
        published_physical_seeds=6,
        field_coordinates=128,
        physical_model_parameters={
            k: v["parameters"] for k, v in pkg["models"].items()
        },
        shared_pretrained_speech_parameters=audio["encoder"]["encoder_parameters"],
        total_held_worlds=older["starts"] + fresh["starts"],
        primary_acknowledgment_to_finish_minutes=(
            read(work / "held-finished.json")["unix"]
            - (work / "released-lock.json").stat().st_mtime
        )
        / 60,
        additional_original_recipe_branches=sum(
            x["additional_executions"]
            for x in read(work / "training-transfer.json")["held"].values()
        ),
        additional_held_fork_candidates=read(m / "held-prediction.json")[
            "macro_alternatives"
        ],
        total_held_branches=older["branches"] + fresh["branches"],
        total_held_native_ticks=older["native_ticks"] + fresh["native_ticks"],
        goals_presented=older["commands"] + fresh["commands"],
        front_end="Frozen Whisper-small.en plus one supervised 182,535-parameter speech head shared by both world processors",
        owned_credits=cost["owned_units"],
        total_owned_credits=cost["cumulative_owned_units"],
        remaining_credits=cost["remaining_units"],
        movies=movies,
        drive_at_release={
            k: state[k]
            for k in ("phase", "reason", "pending", "verified_files")
            if k in state
        },
        next=nxt,
    )
    write(m / "release.json", receipt)
    print(
        "Complete G14 science package",
        receipt["total_held_branches"],
        receipt["movies"].keys(),
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    a = argparse.ArgumentParser()
    a.add_argument("--missions", action="store_true")
    args = a.parse_args()
    (mission_assets if args.missions else assemble)()
