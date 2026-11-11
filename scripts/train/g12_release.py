"""Read back every G12 generation and data object, and publish source and custody receipts."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from g12_custody import ROOT, vaults
from g9_storage import write, sha


def verify_disk(base):
    index = {}
    for group, steps, count in (
        ("runs", (256, 512, 1024), 48),
        ("scale", (256, 512, 1024), 16),
        ("early", (32, 96, 192, 384), 48),
    ):
        folds = sorted(p for p in (base / group).iterdir() if p.is_dir())
        if len(folds) != count:
            raise ValueError(f"Missing {group} fit")
        for model in folds:
            for step in steps:
                folder = model / f"step-{step:05d}"
                receipt = json.loads((folder / "verified.json").read_bytes())
                if receipt["step"] != step or receipt["tag"] != model.name:
                    raise ValueError("Invalid generation identity")
                for p, h in receipt["sha256"].items():
                    if sha(folder / p) != h:
                        raise ValueError("Changed training artifact")
                index[str(folder.relative_to(base))] = receipt["sha256"]
    for group in ("inputs", "inference"):
        manifests = json.loads((base / group / "verified.json").read_bytes())
        for n, r in manifests.items():
            if sha(base / group / n) != r["sha256"]:
                raise ValueError("Changed released input or model " + n)
    return index


def main():
    out = ROOT / "release/metrics"
    root = ROOT / "data"
    with ThreadPoolExecutor(2) as pool:
        a, b = pool.map(verify_disk, vaults())
    if a != b or len(a) != 384:
        raise ValueError("Two-disk custody differs")
    write(out / "checkpoints.json", a)
    meta = json.loads((root / "groups.json").read_bytes())
    sources = {}
    banks = {}
    for fold, roles in meta.items():
        for role in roles:
            file = root / "banks" / f"{fold}-{role}.json"
            r = json.loads(file.read_bytes())
            bank = root / "banks" / f"{fold}-{role}.npz"
            if sha(bank) != r["sha256"]:
                raise ValueError("Changed frozen data")
            banks[fold + "-" + role] = dict(
                sha256=r["sha256"],
                examples=len(r["rows"]),
                cued=r["cued"],
                recordings=len(r["names"]),
                represented_recordings=len({v["recording"] for v in r["rows"]}),
                vocabulary=r["vocabulary"],
            )
        for n in roles["confirm"]:
            shard = root / "fresh/shards" / (n + ".npz")
            av = root / "fresh/media" / (n + "-av.mp4")
            note = json.loads(shard.with_suffix(".json").read_bytes())
            if sha(shard) != note["sha256"]:
                raise ValueError("Fresh AV/IMU features changed")
            sources[n] = dict(
                av_sha256=sha(av),
                features_sha256=note["sha256"],
                ticks=note["ticks"],
                stale_imu_ticks=note["stale_imu_ticks"],
            )
    old = Path("/mnt/slwm-data/SLWM/g10/data/holo")
    write(
        out / "data.json",
        dict(
            groups=meta,
            banks=banks,
            fresh=sources,
            source_label_sha256=sha(old / "labels.json"),
            source_split_sha256=sha(old / "split.zip"),
            vision=json.loads((root / "vision-nv/receipt.json").read_bytes()),
            speech_revision="e8727524f962ee844a7319d92be39ac1bd25655a",
            device_labels_sha256=sha(root / "state-labels.json"),
            fields=[
                "ImageNet visual features, pooled RGB",
                "speech-masked acoustic spectra",
                "synchronized accelerometer and gyroscope",
                "spoken small.en encoder frames",
                "recognized word features",
            ],
            frozen_visual_backbone_parameters=11176512,
        ),
    )
    base = vaults()[0]
    drive = json.loads((base / "drive-state.json").read_bytes())
    compute = json.loads((out / "compute.json").read_bytes())
    if not compute["all_owned_released"]:
        raise ValueError("Incomplete native compute accounting")
    summaries = [
        json.loads((out / f).read_bytes())
        for f in ("comparisons.json", "width.json", "early.json")
    ]
    counts = [len(v["runs"]) for v in summaries]
    inf = json.loads((ROOT / "release/inference/manifest.json").read_bytes())["models"]
    checks = json.loads((out / "inference-checks.json").read_bytes())
    live = json.loads((out / "live.json").read_bytes())
    calibration = json.loads((out / "calibration.json").read_bytes())["models"]
    if (
        counts != [48, 16, 48]
        or len(inf) != 24
        or set(inf) != set(checks)
        or len(live) != 48
        or len(calibration) != 112
    ):
        raise ValueError("Missing registered model, diagnostic or inference package")
    write(
        out / "release.json",
        dict(
            complete=True,
            fits=sum(counts),
            primary_fits=48,
            wide_fits=16,
            early_fits=48,
            generations=len(a),
            verified_hdds=2,
            prepared_objects=len(
                json.loads((base / "inputs/verified.json").read_bytes())
            ),
            portable_models=len(inf),
            live_replays=len(live),
            fresh_recordings=len(sources),
            total_owned_units=compute["owned_units"],
            remaining_credits=compute["remaining_credits"],
            drive_at_release={k: v for k, v in drive.items() if k != "retry_unix"},
        ),
    )
    print(
        "Verified both disks",
        len(a),
        "generations and",
        len(inf),
        "portable models",
        flush=True,
    )


if __name__ == "__main__":
    main()
