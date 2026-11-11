"""Verify exported weights, chronological action decisions and the 128-coordinate field against JAX."""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
from data.g14_policy import loaded, choose, candidates, features
from data.g14_signals import attach, raw_predict
from models.g14_world import predict
from scripts.infer.g14_live import read, check, infer, active_goal
from scripts.train.g9_storage import write

ROOT = Path("/mnt/slwm-data/SLWM/g14")
pkg = ROOT / "release/inference"


def main():
    man = json.loads((pkg / "manifest.json").read_bytes())
    scene = read(check(pkg, man, "scene.npz"))
    hist = [
        {k: scene[k][i] for k in ("p", "q", "seen", "u", "motor", "px")}
        for i in range(4)
    ]
    table = SimpleNamespace(history=hist)
    goals = {
        name: active_goal(scene, pkg, man, feature_file=(pkg / file if file else None))
        for name, file in (
            ("actual", None),
            ("other-address", "other-address-features.npz"),
            ("other-block", "other-block-features.npz"),
        )
    }
    silence = active_goal(scene, pkg, man, "absent")
    results = {}
    for core in ("gpt2", "transport"):
        for seed in (1411, 1412, 1413):
            key = f"{core}-{seed}"
            result = {}
            policy = dict(
                core=core,
                seed=seed,
                stage=man["stage"],
                arm="protected",
                routed=man["routed"],
                contact_only=man["contact_only"],
            )
            for name, goal in {**goals, "absent": silence}.items():
                val = infer(scene, pkg, man, goal, core, seed, True)
                c = goal["color"] if goal["accepted"] else -1
                target = scene["stations"][goal["station"]]
                action, info = choose(
                    table, hist[-1], c, target, policy, man["variant"]
                )
                if (
                    not np.allclose(action, val["command_m"], atol=1e-7)
                    or info["chosen"] != val["candidate"]
                ):
                    raise ValueError("Standalone decision mismatch")
                out = {k: v for k, v in val.items() if k != "field"}
                if c >= 0:
                    opts = candidates(
                        hist[-1], c, target, man["variant"], man["routed"]
                    )
                    r = features(table, c, target, opts)
                    ref, _, model, _ = loaded(man["stage"], core, "protected", seed)
                    attach(r, raw_predict(r, ref))
                    y, _, field = predict(
                        {k: jnp.asarray(v) for k, v in model.items()},
                        {k: jnp.asarray(v) for k, v in r.items()},
                        core,
                        "protected",
                        jnp,
                    )
                    a = np.max(np.abs(np.asarray(field[0]) - val["field"]))
                    yn = predict(model, r, core, "protected", np)[0]
                    b = np.max(np.abs(np.asarray(y) - yn))
                    out["jax_numpy_max_field_difference"] = float(a)
                    out["jax_numpy_max_effect_difference"] = float(b)
                    if a > 0.0001 or b > 0.0001:
                        raise ValueError("Backend mismatch in released fields")
                result[name] = out
            results[key] = result
    write(
        ROOT / "work/field-replay.json",
        dict(
            models=results,
            voices=man["same_speaker_alternatives"],
            initial=man["example"],
            prefix_macros=3,
        ),
    )
    print("Verified six deployed fields and four cue paths each", flush=True)


if __name__ == "__main__":
    main()
