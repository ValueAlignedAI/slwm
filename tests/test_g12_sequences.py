"""Prevent temporal leakage, mismatched field controls and uncalibrated external outputs."""

from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "scripts/data"


def test_triplet_rejects_ties_future_limits_and_insufficient_lead(monkeypatch):
    monkeypatch.syspath_prepend(str(DATA))
    from g12_events import triplet

    def act(t):
        return {"start": t, "end": t + 0.4}

    seq = [act(t) for t in (2, 4, 6, 9)]
    assert triplet(seq, 0, 1.8, 20) == seq[:3]
    assert triplet(seq, 0, 1.81, 20) is None
    assert triplet([*seq[:2], act(14.1)], 0, 1, 20) is None
    assert triplet([*seq[:3], act(6.05)], 0, 1, 20) is None
    assert triplet(seq, 2, 1, 20) is None


def test_four_causal_processors_cannot_read_a_later_signal():
    from models.g12_world import CORES, mix
    import jax.numpy as jnp

    q, k, v = (
        jnp.asarray(x, np.float32)
        for x in np.random.default_rng(52).normal(size=(3, 2, 7, 16))
    )
    for core in CORES:
        short = mix(core, q[:, :4], k[:, :4], v[:, :4], 4)
        full = mix(core, q, k, v, 4)
        np.testing.assert_allclose(short, full[:, :4], atol=2e-5, rtol=2e-5)


def test_proposal_gate_has_separate_decoders_and_internal_action():
    from models.g12_world import route

    logits = np.array([[2, 1], [0, 5], [0.2, 0.1]])
    voc = ["open", "insert"]
    assert route(logits, 0.8, [0.1, 0.1], [0.5, 0.4], voc) == {"route": "internal_only"}
    emitted = route(logits, 0.8, [0.6, 0.3], [0.5, 0.4], voc)
    assert emitted == {"actions": ["open", "insert", "open"], "route": "actions"}
    state = route(logits, 0.8, [0.1, 0.8], [0.5, 0.4], voc)
    assert (
        state["route"] == "device_change"
        and state["device_change"]
        and "actions" not in state
    )


def test_blind_duplicate_disagreement_is_excluded(monkeypatch):
    monkeypatch.syspath_prepend(str(DATA))
    from g12_states import bind

    case = {"key": "test", "recording": "r", "id": 7}
    index = [{"number": 1, "key": "test"}, {"number": 2, "key": "test"}]
    ratings = [
        {"number": 1, "code": "S"},
        {"number": 2, "code": "A", "component": "latched cover"},
    ]
    result = bind([case], index, ratings)
    assert (
        result["items"][0]["label"] == -1
        and result["repeats"] == 1
        and result["readable_agreement"] == 0
    )


def test_real_foil_and_silence_do_not_modify_scene():
    from models.g12_world import inputs

    row = dict(
        scene=np.ones((2, 8, 633)),
        ac=np.ones((2, 4, 128)),
        wc=np.full((2, 4, 128), 2),
        words=np.ones((2, 4, 147)),
        wrong=np.full((2, 4, 147), 3),
    )
    for mode, aval, wval in (("correct", 1, 1), ("wrong", 2, 3), ("none", 0, 0)):
        scene, ac, words = inputs(row, mode)
        np.testing.assert_array_equal(scene, row["scene"])
        assert np.all(np.asarray(ac) == aval) and np.all(np.asarray(words) == wval)
