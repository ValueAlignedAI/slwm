"""Verify the published G11 comparisons, four-device evaluation and audible inference artifacts."""

import gzip
import hashlib
import json
from pathlib import Path
import wave
import zipfile
import numpy as np

ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments/multimodal/t11/instructions-physical-effects-20261111"
)


def read(name):
    return json.loads((ROOT / name).read_text())


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def test_public_study_and_device_groups():
    r = read("metrics/release.json")
    p = read("metrics/primary.json")
    s = read("metrics/scale.json")
    assert (
        r["complete"]
        and r["all_owned_released"]
        and r["authorized_project_allowance"] == 600
    )
    assert (
        abs(r["prior_owned_units"] + r["owned_units"] - r["cumulative_owned_units"])
        < 1e-8
    )
    assert (r["primary_fits"], r["scale_adapter_fits"], r["new_recurrent_parents"]) == (
        54,
        24,
        4,
    )
    assert r["verified_generations"] == 250 and (p["fits"], s["fits"]) == (54, 24)
    assert (
        len(read("metrics/primary-artifacts.json")) == 162
        and len(read("metrics/scale-artifacts.json")) == 72
    )
    assert len(s["size_comparisons"]) == 12
    d = read("metrics/data.json")
    fresh = d["groups"]["fresh"]
    assert all(len(v) == 24 for v in fresh.values())
    assert len({n for values in fresh.values() for n in values}) == 96
    for fold, roles in d["groups"]["folds"].items():
        assert len(roles["train"]) == 256 and set(roles["held"]).isdisjoint(
            roles["train"]
        )
    with gzip.open(ROOT / "metrics/held-decisions.jsonl.gz", "rt") as f:
        events = [json.loads(x) for x in f]
    assert len(events) == 838
    assert all(
        e["other_interval"][1] <= e["cue_interval"][0] - 0.1 + 1e-6
        and e["cue_interval"][1] <= e["cut_end_s"] + 1e-7
        and e["cut_end_s"] + 0.2 <= e["action_start_s"] + 1e-7
        for e in events
    )
    assert sum(read("metrics/device-audit.json")["counts"].values()) == 259
    for name in ("primary-predictions", "scale-predictions"):
        with zipfile.ZipFile(ROOT / "metrics" / (name + ".npz")) as z:
            assert z.testzip() is None


def test_public_model_identity_and_examples():
    m = read("inference/portable.json")
    ledger = read("metrics/primary-artifacts.json")
    assert len(m["backbones"]) == 3 and len(m["adapters"]) == 6
    for c, b in m["backbones"].items():
        file = ROOT / "inference" / b["file"]
        assert file.stat().st_size == b["bytes"] and checksum(file) == b["sha256"]
        for fold in ("A", "B"):
            head = m["adapters"][fold + "-" + c]
            scientific = ledger[f"g11-{fold}-{c}-paired-s1001/1024"]
            assert (
                head["source_sha256"] == scientific["inference_sha256"]
                and head["parent_sha256"] == b["source_sha256"]
            )
            assert checksum(ROOT / "inference" / head["file"]) == head["sha256"]
    for family, v in read("examples/outcomes.json").items():
        file = ROOT / "examples" / (family + ".npz")
        assert checksum(file) == v["sha256"]
        with np.load(file, allow_pickle=False) as x:
            assert (
                x["a"].shape == (32, 64)
                and x["v"].shape == (32, 768)
                and x["ac"].shape == (16, 128)
            )
            assert "verb" not in x and "effect" not in x
        for mode in ("correct", "wrong"):
            with wave.open(
                str(ROOT / "examples" / (family + "-" + mode + ".wav"))
            ) as sound:
                assert (
                    sound.getframerate() == 16000
                    and 3000 <= sound.getnframes() <= 28 * 16000
                )
