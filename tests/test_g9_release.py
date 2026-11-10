"""The published G9 scores and compact weights must refer to the same verified training generations."""
import hashlib
import json
from pathlib import Path
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "experiments/multimodal/t9/concurrent-memory-20261110"


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def test_complete_science_inventory_and_cost():
    release = read("metrics/release.json")
    results = read("metrics/results.json")
    manifest = read("metrics/artifacts.json")
    assert release["complete"] and release["all_owned_released"]
    assert release["total_fits"] == 42 and len(manifest) == 58
    assert results["registered_stages_scored"] == 92
    assert all(not row["active"] for row in release["owned_endpoints"])
    assert abs(sum(row["units"] for row in release["owned_endpoints"]) - release["owned_units"]) < 1e-8
    panels = results["followups"]
    assert all(len(panels[key]) == n for key, n in (
        ("source_retrieval", 8), ("history_window", 8), ("uncapped_contact", 8),
        ("decision_adaptation", 4), ("backend_check", 16), ("independent_episode_dose", 4),
    ))
    with zipfile.ZipFile(ROOT / "metrics/paired-observations.npz") as tape:
        assert len(tape.namelist()) > 1000 and tape.testzip() is None


def test_published_inference_generations():
    ledger = read("metrics/artifacts.json")
    compact = read("inference/portable.json")
    assert len(compact) == 2
    for tag, entry in compact.items():
        model = ROOT / "inference" / (tag + ".npz")
        assert model.stat().st_size == entry["bytes"]
        assert digest(model) == entry["sha256"]
        assert ledger[tag + "/4096"]["inference_sha256"] == entry["parent_sha256"]
    aux = ROOT / "metrics/followups"
    copied = read("metrics/followups/verified.json")
    assert len(copied) == 20
    for name, expected in copied.items():
        assert digest(aux / name) == expected
    for file in (aux / "heads").glob("*/posttraining.json"):
        record = json.loads(file.read_text())
        assert record["parent_sha256"] == ledger[record["tag"] + "/4096"]["inference_sha256"]
    media = ROOT / "examples"
    assert len(list(media.glob("*.mp4"))) == 5
    for file in media.glob("*.wav"):
        with wave.open(str(file)) as sound:
            assert sound.getframerate() == 16000 and sound.getnframes() == 640000
    assert len(read("examples/live-trace.json")["actions"]["hybrid"]) == 512
