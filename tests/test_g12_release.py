"""Check release integrity and independently reproduce an ordered-action and physical score."""

import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments/multimodal/t12/instruction-sequences-20261111"
)


def data(name):
    return json.loads((ROOT / name).read_bytes())


def test_g12_complete_disjoint_and_clock_causal():
    summary = data("metrics/release.json")
    cost = data("metrics/compute.json")
    groups = data("metrics/data.json")["groups"]
    assert (summary["fits"], summary["generations"], summary["portable_models"]) == (
        112,
        384,
        24,
    )
    assert len(data("metrics/checkpoints.json")) == 384 and cost["all_owned_released"]
    assert abs(cost["cumulative_owned_units"] + cost["remaining_credits"] - 600) < 1e-7
    fresh = set(groups["A"]["confirm"] + groups["B"]["confirm"])
    assert len(fresh) == 48
    for roles in groups.values():
        assert fresh.isdisjoint(roles["train"] + roles["dev"] + roles["replica"])
    with gzip.open(ROOT / "metrics/decisions.jsonl.gz", "rt") as f:
        rows = [json.loads(s) for s in f]
    assert len(rows) == 1057 and sum(r["role"] == "confirm" for r in rows) == 373
    for r in rows:
        t = (r["cut_tick"] + 1) / 10
        assert (
            r["foil_span"][1] <= r["speech_span"][0] - 0.1 + 1e-6
            and r["speech_span"][1] <= t + 1e-6
        )
        starts = r["target_onsets"]
        assert (
            t + 0.2 <= starts[0] + 1e-6
            and starts[2] <= starts[0] + 12
            and min(np.diff(starts)) >= 0.1 - 1e-6
        )


def test_g12_public_predictions_reproduce_source_group_means():
    report = data("metrics/comparisons.json")
    with gzip.open(ROOT / "metrics/decisions.jsonl.gz", "rt") as f:
        rows = [
            r
            for line in f
            if (r := json.loads(line))["role"] == "confirm" and r["fold"] == "B"
        ]
    families = np.array([r["family"] for r in rows])
    groups = np.array([r["recording"] for r in rows])

    def macro(obs):
        parts = []
        for fam in np.unique(families):
            mask = (families == fam) & np.isfinite(obs)
            parts.append(
                np.mean(
                    [obs[mask & (groups == g)].mean() for g in np.unique(groups[mask])]
                )
            )
        return float(np.mean(parts))

    with np.load(ROOT / "metrics/paired-predictions.npz") as file:
        seq = file["B-gpt2-paired-1201_confirm_correct_top"]
        y = file["B_confirm_y"]
        p = file["B-gpt2-paired-1201_confirm_correct_state"]
        state = file["B_confirm_state"]
    expected = report["runs"]["B-gpt2-paired-1201"]["by_role"]["confirm"]["input"][
        "correct"
    ]
    assert abs(macro((seq == y).mean(1)) - expected["ordered"]["value"]) < 1e-7
    assert (
        abs(
            macro(np.where(state >= 0, (state - p) ** 2, np.nan))
            - expected["brier"]["value"]
        )
        < 1e-6
    )


def test_g12_blind_labels_and_exact_portable_live_outputs():
    labels = data("metrics/visual-labels.json")
    models = data("inference/manifest.json")["models"]
    assert len(labels["items"]) == 688 and labels["repeats"] == 57
    assert (
        sum(r["role"] == "confirm" and r["label"] >= 0 for r in labels["items"]) == 77
    )
    with zipfile.ZipFile(ROOT / "audit/frames.zip") as z:
        assert (
            len([n for n in z.namelist() if n.endswith(".jpg")]) == 47
            and z.testzip() is None
        )
    checks = data("metrics/inference-checks.json")
    assert set(models) == set(checks)
    for k, v in models.items():
        with (ROOT / "inference" / v["file"]).open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == v["sha256"]
        assert all(
            r["all_argmax_identical"]
            and max(r["max_probability_error"], r["max_state_error"]) == 0
            for r in checks[k]
        )
    live = data("metrics/single-clock.json")
    assert (
        sum(x["cycles"] for x in live) == 642
        and sum(x["source_seconds"] for x in live) == 603
    )
    assert len(data("metrics/raw-audio.json")) == 8
    for row in data("examples/streams.json"):
        with np.load(ROOT / "examples" / (row["family"] + "-stream.npz")) as z:
            assert set(z.files) == {
                "scene",
                "ac",
                "wc",
                "words",
                "wrong",
                "human",
                "wrong_human",
                "seconds",
            }
            assert np.all(np.diff(z["seconds"]) > 0)
