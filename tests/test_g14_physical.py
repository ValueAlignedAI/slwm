"""Science invariants for the prospective spoken-goal and physically forked studies."""

from pathlib import Path
import json
import numpy as np
from data.g14_policy import choose
from data.g14_table import evaluate_path
from scripts.data.g14_cases import decode
from models.g14_world import initialize, core_mix


def test_causal_mixing_excludes_every_future_frame():
    rng = np.random.default_rng(814)
    q, k, v = [rng.normal(size=(2, 4, 5, 16)).astype("f4") for _ in range(3)]
    pos = rng.normal(size=(2, 4, 5, 2)).astype("f4")
    altered = v.copy()
    altered[:, 2:] += 30
    for core in ("gpt2", "transport", "gated", "dense_spatial"):
        a = core_mix(q, k, v, pos, core, np)
        b = core_mix(q, k, altered, pos, core, np)
        np.testing.assert_array_equal(a[:, :2], b[:, :2])
        assert np.linalg.norm(a[:, 2:] - b[:, 2:]) > 1


def test_a_substituted_address_changes_the_destination_in_the_public_map():
    color = [0.02, 0.96, 0.01, 0.01]
    letter = np.eye(25)
    digit = np.eye(10)
    addresses = [(8, 6), (5, 3), (8, 3), (5, 6)]
    for i, (l, d) in enumerate(addresses):
        c, s, prob = decode(color, letter[l], digit[d], addresses)
        assert (c, s) == (1, i) and prob > 0.95


def test_absence_and_already_satisfied_goal_do_not_need_a_model():
    obs = {"q": np.zeros((4, 2))}
    target = np.zeros(2)
    no = np.zeros(2)
    for core in ("gpt2", "transport", "geometry"):
        for c, expected in ((-1, "internal_only"), (1, "no_op")):
            action, detail = choose(None, obs, c, target, {"core": core}, 0)
            np.testing.assert_array_equal(action, no)
            assert detail["route"] == expected


def test_independent_pose_measurement_detects_collateral():
    start = np.array([[0.3, 0], [0.4, 0.1], [0.48, -0.1], [0.26, -0.1]])
    traj = np.tile(start, (3, 1, 1))
    goal = np.array([0.35, 0])
    traj[:, 0] = goal
    traj[:, 3, 1] += 0.05
    distance, other = evaluate_path(traj, start, 0, goal)
    assert np.all(distance == 0) and np.all(other > 0.035)


def test_speech_and_telemetry_receipts_reconcile_measured_sources():
    root = (
        Path(__file__).resolve().parents[1]
        / "experiments/multimodal/t14/spoken-closed-loop-20261111"
    )
    s = json.loads((root / "metrics/speech.json").read_bytes())
    row = s["results"]["cue-semantic"]["cases"]["fresh"]
    assert (
        row["correct"] == row["valid_scheduled"] == 256
        and row["accepted"] + row["rejected_correct"] == 256
    )
    assert set(s["dataset"]["people"]["train"]).isdisjoint(
        s["dataset"]["people"]["fresh"]
    )
    t = json.loads((root / "metrics/telemetry.json").read_bytes())
    assert t["sim_worlds"] == 192 and t["cohorts"]["robot-readout"]["parents"] == 210
    for c in t["cohorts"].values():
        assert (
            c["calibration"]["4"]["sim_calibrated"][-1]["value"]
            < c["baseline"]["sim"][-1]["value"]
        )


def test_field_replacement_preserves_raw_coordinates_and_bypasses_only_learned_context():
    from models.g14_world import predict

    rng = np.random.default_rng(8141)
    norm = lambda shape: rng.standard_normal(shape).astype("f4")
    p = initialize((19, 24, 8), 4, width=16)
    row = dict(
        slots=norm((2, 4, 5, 19)),
        image=norm((2, 4, 24)),
        location=norm((2, 4, 5, 2)),
        focus=np.eye(5, dtype="f4")[[1, 2]],
        phi=norm((2, 8, 8)),
        prior=norm((2, 8, 11)),
        raw_action=norm((2, 2)),
    )
    constant = norm((126,))
    for core in ("gpt2", "transport"):
        y, _, z = predict(p, row, core, "protected", np)
        a, _, abl = predict(p, row, core, "protected", np, mean_field=constant)
        np.testing.assert_array_equal(abl[:, :2], row["raw_action"])
        np.testing.assert_array_equal(abl[:, 2:], np.broadcast_to(constant, (2, 126)))
        assert np.max(abs(y - a)) > 1e-4
        _, _, same = predict(
            p,
            {**row, "phi": row["phi"] * 3, "prior": row["prior"] * 0},
            core,
            "protected",
            np,
        )
        np.testing.assert_array_equal(same, z)


def test_portable_contact_field_replays_exact_saved_decision():
    from scripts.infer.g14_live import BASE, read, check, infer

    man = json.loads((BASE / "manifest.json").read_bytes())
    scene = read(check(BASE, man, "scene.npz"))
    replay = json.loads((BASE.parent / "metrics/field-replay.json").read_bytes())
    for core in ("gpt2", "transport"):
        for seed in (1411, 1412, 1413):
            saved = replay["models"][f"{core}-{seed}"]["actual"]
            y = infer(scene, BASE, man, saved["goal"], core, seed, True)
            assert y["candidate"] == saved["candidate"] and len(y["field"]) == 128
            np.testing.assert_allclose(y["command_m"], saved["command_m"], atol=1e-7)
            np.testing.assert_allclose(
                y["forecast_effector_and_four_blocks_mm"],
                saved["forecast_effector_and_four_blocks_mm"],
                atol=2e-4,
            )
