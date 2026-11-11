"""Independently rederive the primary G14 result and the measured native replay outcomes."""

import json
from pathlib import Path
import numpy as np
from scripts.train.g14_execute import tag, score

BASE = (
    Path(__file__).resolve().parents[1]
    / "experiments/multimodal/t14/spoken-closed-loop-20261111"
)


def load(file):
    return json.loads((BASE / file).read_bytes())


def test_matched_physical_gain_recomputes_from_untouched_worlds():
    doc = load("metrics/physical-results.json")
    lock = load("final-control-lock.json")
    count = 0
    for cohort, worlds in (("held", 256), ("fresh", 64)):
        r = doc["cohorts"][cohort]
        with np.load(
            BASE / "metrics" / (cohort + "-scene-results.npz"), allow_pickle=False
        ) as d:
            labels = d["models"].tolist()
            metric = d["metrics"].tolist().index("success")
            c = d["conditions"].tolist().index("actual")
            assert len(np.unique(d["start"])) == worlds and d["values"].shape[:3] == (
                22,
                worlds,
                4,
            )
            values = d["values"][:, :, c, metric]
            f = np.column_stack((d["speaker"], d["shift"]))

            def m(a):
                return np.mean(
                    [np.mean(a[(f == v).all(1)]) for v in np.unique(f, axis=0)]
                )

            diffs = []
            for seed in (1411, 1412, 1413):

                def i(core):
                    return labels.index(
                        tag(
                            dict(
                                core=core,
                                stage=lock["stage"],
                                seed=seed,
                                arm="protected",
                                routed=lock["routed"],
                                contact_only=lock["contact_only"],
                            )
                        )
                    )

                v = values[i("transport")] - values[i("gpt2")]
                diffs.append(v)
                reported = r["spatial_over_dense_by_seed"][str(seed)]["success"]
                assert abs(m(v) - reported["value"]) < 1e-6
            assert (
                abs(
                    m(np.mean(diffs, axis=0))
                    - r["paired"]["transport_over_gpt2"]["success"]["value"]
                )
                < 1e-6
            )
        count += r["branches"]
    release = load("metrics/release.json")
    assert (
        count == release["total_held_branches"] == 27200
        and release["total_held_native_ticks"] == count * 72
    )


def test_native_replays_and_training_fidelity_are_reproducible():
    audit = load("metrics/trajectory-replay.json")
    assert (
        audit["same_selective_scores"]
        == audit["identical_digests"]
        == audit["replays"]
        == 48
    )
    with np.load(BASE / "examples/native-traces.npz", allow_pickle=False) as d:
        assert d["objects"].shape == (48, 72, 4, 2)
        actual = [
            score(y, b, int(c), t)["success"]
            for y, b, c, t in zip(d["objects"], d["initial"], d["color"], d["target"])
        ]
        np.testing.assert_array_equal(actual, d["original_score"])
    valid = load("metrics/final-fidelity-decision.json")
    assert valid["usable"] and all(valid["passed"].values())
    labels = load("metrics/isolated-label-audit.json")
    assert (
        labels["candidates"] == 960 and labels["long_return_difference_mm"]["max"] == 0
    )


def test_release_tracks_live_missions_and_modality_paths():
    d = load("metrics/release.json")
    m = load("metrics/missions.json")
    f = load("metrics/field-replay.json")
    assert (
        d["fits"] == 75
        and d["generations"] == 225
        and len(load("metrics/checkpoint-inventory.json")) == 225
    )
    assert d["owned_credits"] == 0 and d["published_physical_seeds"] == 6
    assert len(m["durations"]) == 16 and min(m["durations"]) > 30
    for key, v in f["models"].items():
        assert (
            v["absent"]["route"] == "internal_only"
            and v["actual"]["field_coordinates"] == 128
        )
        assert f["same_prefix_contrasts"][key]["matching-text"]["learned_field_l2"] == 0
    assert all(
        len(v["streams"]) == 2 and v["seconds"] > 30 for v in d["movies"].values()
    )


def test_prediction_error_recomputes_from_native_displacement_measurements():
    d = load("metrics/held-prediction.json")
    assert d["worlds"] == 64 and d["states"] == 1024 and d["macro_alternatives"] == 8192
    with np.load(
        BASE / "metrics/held-physics-predictions.npz", allow_pickle=False
    ) as z:
        names = z["models"].tolist()
        world = z["world"]
        truth = z["actual_mm"]
        color = z["color"] + 1
        actual_request = truth[np.arange(len(world)), :, color]
        mask = np.linalg.norm(actual_request, axis=-1) >= 5
        assert mask.sum() == d["moving_requested_object_alternatives"]
        for name in ("raw", "transport-1411", "gpt2-1411"):
            prediction = z["predicted_mm"][names.index(name)][
                np.arange(len(world)), :, color
            ]
            err = np.linalg.norm(prediction - actual_request, axis=-1)
            vals = [
                (i, err[world == i][mask[world == i]].mean())
                for i in np.unique(world)
                if mask[world == i].any()
            ]
            check = np.mean(
                [np.mean([v for i, v in vals if i % 2 == shift]) for shift in (0, 1)]
            )
            assert (
                abs(check - d["scores"][name]["moving_requested_block_mm"]["value"])
                < 1e-4
            )
