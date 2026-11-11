"""Verify independent cohorts, auditable output utilities, deployment artifacts and voice portability."""

import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp

ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments/multimodal/t13/grounded-transfer-20261111"
)


def read(name):
    return json.loads((ROOT / name).read_bytes())


def test_g13_release_has_complete_matched_generations_and_disjoint_external_sources():
    release = read("metrics/release.json")
    cost = read("metrics/compute.json")
    assert release["complete"] and (
        release["neural_fits"],
        release["generations"],
        release["portable_models"],
    ) == (182, 594, 27)
    assert len(read("metrics/checkpoints.json")) == 594 and cost["all_owned_released"]
    assert abs(cost["cumulative_owned_units"] + cost["remaining_units"] - 600) < 1e-7
    groups = {}
    source_ids = set()
    with gzip.open(ROOT / "metrics/robot-source-index.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            groups.setdefault(row["acquisition"], set()).add(row["group"])
            assert row["source_id"] not in source_ids
            source_ids.add(row["source_id"])
            if row["role"] == "train":
                assert not row["held_color_pair"]
    sets = list(groups.values())
    assert len(sets) == 4
    for i, s in enumerate(sets):
        assert all(s.isdisjoint(x) for x in sets[:i])


def test_direct_readout_improvement_recomputes_from_fourth_acquisition():
    result = read("metrics/direct-action.json")
    with np.load(ROOT / "metrics/direct-actions.npz") as z:
        y = z["truth"]
        groups = z["group"]
        raw = np.linalg.norm(z["raw"] - y, axis=1) * 1000
        losses = np.mean(
            [
                1000 * np.linalg.norm(z[f"transport-linear-{s}-protected"] - y, axis=1)
                for s in (1301, 1302, 1303, 1311, 1312)
            ],
            axis=0,
        )
        gain = np.mean([(raw - losses)[groups == g].mean() for g in np.unique(groups)])
    assert (
        result["groups"] == 210
        and result["episodes"] == 3535
        and abs(gain - result["five_seed_over_raw"]["value"]) < 1e-4
    )
    assert (
        result["five_seed_over_raw"]["lo99"] > 0
        and read("metrics/readout-controls.json")["post_hoc"]
    )


def test_explicit_waveform_patch_matches_an_independent_asymmetric_padding_reference():
    from models.g13_speech import patch_convolution

    rng = np.random.default_rng(32)
    x = rng.standard_normal((3, 10, 8, 2)).astype("f4")
    k = rng.standard_normal((3, 3, 2, 3)).astype("f4")
    padded = np.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
    ref = np.zeros((3, 5, 4, 3))
    for i in range(5):
        for j in range(4):
            ref[:, i, j] = np.einsum(
                "bhwc,hwcd->bd", padded[:, i * 2 : i * 2 + 3, j * 2 : j * 2 + 3], k
            )
    np.testing.assert_allclose(
        patch_convolution(jnp.asarray(x), jnp.asarray(k)), ref, atol=5e-5, rtol=2e-5
    )
    evidence = read("metrics/speech-parity.json")
    assert (
        evidence["portable"]["same_top"] == 1639
        and evidence["portable"]["gpu_single_delta"] < 0.003
    )


def test_portable_fields_and_all_raw_waveform_routes_are_present():
    index = read("inference/manifest.json")
    assert (
        len(index["worlds"]) == 10
        and len(index["components"]) == 16
        and len(index["nouns"]) == 165
    )
    for name, entry in index["files"].items():
        with (ROOT / "inference" / name).open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == entry["sha256"]
    assert (
        len(read("metrics/voice-waveforms.json")) == 32
        and len(read("metrics/component-inference.json")) == 48
    )
    assert all(
        x["plan_to_passive_error"] == 0 and x["plan_to_object_error"] == 0
        for x in read("metrics/world-parity.json").values()
    )
    live = read("metrics/live.json")
    assert (
        len(live) == 6
        and sum(x["clocks"] for x in live if x["model"] == "transport-linear") == 842
    )
    for s in read("examples/robot-streams.json"):
        with np.load(ROOT / "examples" / (s["stream"] + ".npz")) as z:
            assert set(z.files) == {"px", "geo", "kin", "plan", "tick"} and np.all(
                np.diff(z["tick"]) == 1
            )
