"""Cross-check public G10 models, held panels, readouts and named compute receipts."""

import hashlib
import json
from pathlib import Path
import wave
import zipfile

import numpy as np

ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments/multimodal/t10/context-alignment-20261111"
)


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(file):
    digest = hashlib.sha256()
    with file.open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def test_released_factorials_and_custody():
    release = read("metrics/release.json")
    assert release["complete"] and release["all_owned_released"]
    assert (
        release["context_fits"],
        release["alignment_fits"],
        release["instruction_fits"],
    ) == (27, 24, 12)
    assert release["total_fits"] == 63 and release["verified_generations"] == 90
    assert (
        abs(sum(row["units"] for row in release["waves"]) - release["owned_units"])
        < 1e-8
    )
    assert all(not row["active"] for row in release["waves"])
    cards = read("metrics/data.json")["holo"]
    cohorts = [
        {r["name"] for r in cards[k]["session_list"]}
        for k in ("train", "dev", "familiar", "novel")
    ]
    assert list(map(len, cohorts)) == [256, 32, 64, 64]
    assert len(set.union(*cohorts)) == 416
    assert set(cards["train"]["tasks"]).isdisjoint(cards["novel"]["tasks"])
    assert read("metrics/context.json")["positive_hybrid_seeds"] == 1
    grounding = read("metrics/grounding.json")
    assert (
        len(grounding["rows"]) == 24
        and len(read("metrics/forecasts.json")["rows"]) == 12
    )
    assert all(
        len(row["panels"]["novel"]["retrieval"]) == 12
        for row in grounding["rows"].values()
    )
    for name in (
        "context-paired",
        "instruction-paired",
        "grounding-paired",
        "risk-predictions",
    ):
        with zipfile.ZipFile(ROOT / "metrics" / (name + ".npz")) as archive:
            assert archive.testzip() is None


def test_portable_models_and_exercised_decoders():
    stages = {}
    for name in ("context", "instruction", "grounding"):
        stages.update(read("metrics/" + name + "-artifacts.json"))
    assert len(stages) == 90
    compact = read("inference/portable.json")
    assert len(compact) == 3
    for tag, item in compact.items():
        model = ROOT / "inference" / (tag + ".npz")
        assert sha(model) == item["sha256"] and model.stat().st_size == item["bytes"]
        key = tag + ("/1536" if tag.startswith("align-") else "/5120")
        assert stages[key]["inference_sha256"] == item["parent_sha256"]
        assert len(item["control"]["portable_continuous"]) == 128
    readouts = read("inference/risk-heads/receipts.json")
    assert len(readouts) == 2
    for name, meta in readouts.items():
        assert sha(ROOT / "inference/risk-heads" / name) == meta["sha256"]
        assert (
            meta["parent_sha256"]
            == compact[name.removesuffix("-risk-heads.npz")]["parent_sha256"]
        )
    with np.load(ROOT / "examples/held-public-input.npz", allow_pickle=False) as f:
        assert f["a"].shape == (600, 64) and f["v"].shape == (600, 768)
        assert f["x"].shape == (600, 128) and np.any(f["x"] != 0)
    assert len(list((ROOT / "examples").glob("*.mp4"))) == 4
    for file in (ROOT / "examples").glob("*.wav"):
        with wave.open(str(file)) as sound:
            assert sound.getframerate() == 16000 and sound.getnframes() == 960000
