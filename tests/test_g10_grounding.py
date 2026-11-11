"""Isolation and negative-control checks for the public signal and risk interfaces."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from data.g10_language import timeline
from models.g10_alignment import add_edges, encode_record
from models.g9_shared import Spec, parameters


def test_transcripts_enter_only_after_completion_and_expire():
    words = timeline(
        [{"end": 1.45, "text": "please turn the knob"}, {"end": 3.0, "text": "stop"}],
        120,
    )
    assert not words[:14].any() and words[14].any()
    assert not np.array_equal(words[28], words[29])
    assert not words[109:].any()


def test_hidden_new_edges_cannot_change_the_audio_only_state():
    s = Spec("gpt2", 128, 1, 128, "gelu", "float32")
    p = add_edges(parameters(s, 17), s, 17)
    b, t = 2, 8
    r = {
        "a": jnp.zeros((b, t, 64), jnp.uint8),
        "v": jnp.zeros((b, t, 768), jnp.uint8),
        "n": jnp.zeros((b, t, 12)),
        "w": jnp.zeros((b, t, 8), jnp.uint8),
        "u": jnp.zeros((b, t), jnp.int32),
        "m": jnp.ones((b, t, 256)),
        "x": jnp.ones((b, t, 128), jnp.int8),
    }
    mask = jnp.broadcast_to(jnp.array([1, 0, 0, 0], bool), (b, 4))
    before = encode_record(p, s, r, mask)
    changed = encode_record(
        p,
        s,
        {**r, "m": r["m"] * 7, "x": r["x"] * 85, "w": r["w"] + 13, "n": r["n"] + 2},
        mask,
    )
    np.testing.assert_allclose(before, changed, atol=0, rtol=0)
