"""Causal input, segmentation and sanity checks for context-length comparisons."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from models.g10_context import brief_visibility, segment, simulation
from evals.g10_control import delayed
from models.g9_shared import Spec, parameters


def test_segmentation_preserves_every_training_position_and_source_mask():
    row = simulation(jnp.arange(3), 101)
    row["keep"] = jnp.ones((3, 4), bool)
    row["is_sim"] = jnp.ones(3, bool)
    for length in (32, 64, 128, 256, 512):
        parts = segment(row, length)
        for name in ("u", "policy", "next_n", "w"):
            np.testing.assert_array_equal(parts[name].reshape(row[name].shape), row[name])
        np.testing.assert_array_equal(parts["keep"][::512//length], row["keep"])
    np.testing.assert_array_equal(row["u"][:, 1:], row["plan"][:, :-1])


def test_brief_words_are_visible_only_after_each_observed_change():
    goals = jnp.array([[1]*9 + [2]*8 + [1]*5])
    visible = np.flatnonzero(np.asarray(brief_visibility(goals))[0])
    assert list(visible) == [0,1,2,3,9,10,11,12,17,18,19,20]


def test_sensor_identical_terminal_targets_are_opposite():
    spec = Spec("gpt2",128,1,128,"gelu","float32")
    result = delayed(parameters(spec,20),spec,661,b=4)
    assert result["delays"][-1]==480 and len(result["accuracy"][-1])==4
