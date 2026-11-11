"""Independently reproduce G13's prospective effect and check causal field controls."""

import json
from pathlib import Path
import numpy as np

BASE = (
    Path(__file__).resolve().parents[1]
    / "experiments/multimodal/t13/grounded-transfer-20261111/metrics"
)


def test_frozen_five_seed_gain_recomputes_from_measured_commands():
    d = json.loads((BASE / "prospective.json").read_bytes())
    with np.load(BASE / "prospective-actions.npz") as z:
        y = z["truth"]
        g = z["group"]
        diff = []
        for seed in (1301, 1302, 1303, 1311, 1312):
            a = z[f"transport-linear-{seed}"]
            b = z[f"gpt2-linear-{seed}"]
            diff.append(
                1000 * (np.linalg.norm(b - y, axis=1) - np.linalg.norm(a - y, axis=1))
            )
        v = np.mean(diff, axis=0)
        estimate = np.mean([v[g == i].mean() for i in np.unique(g)])
    assert d["passed"] and d["parent_groups"] == 196 and d["episodes"] == 3368
    assert (
        abs(estimate - d["five_seed_gain"]["value"]) < 1e-4
        and d["five_seed_gain"]["lo99"] > 0
    )


def test_proposed_controls_do_not_enter_action_or_object_proposals():
    import jax.numpy as jnp
    from models.g13_world import Spec, initial, propose

    spec = Spec(width=16, layers=1, field=16, heads=4)
    p = initial(spec, 88)
    z = jnp.zeros
    r = dict(
        px=z((2, 4, 480)),
        geo=z((2, 4, 4, 2, 6)),
        kin=z((2, 4, 6)),
        goal=z((2, 128)),
        wrong=z((2, 128)),
        bind=z((2, 4, 4)),
        wb=z((2, 4, 4)),
        plan=z((2, 4, 2)),
        alt_plan=jnp.ones((2, 4, 2)) * 0.04,
    )
    a = propose(p, spec, "transport", "bound", r, "correct")
    b = propose(p, spec, "transport", "bound", r, "wrong_plan")
    for key in ("field", "action", "passive", "mover"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.allclose(a["active"], b["active"])


def test_future_control_cannot_modify_prior_horizon_in_identification():
    from training.g13_physics import design

    kin = np.ones((3, 4, 6))
    plan = np.zeros((3, 4, 2))
    changed = plan.copy()
    changed[:, 2:] = 1.0
    np.testing.assert_array_equal(design(kin, plan, 2), design(kin, changed, 2))
