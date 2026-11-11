"""Closed-loop execution, actuator recalibration and strictly sensor-identical delayed decisions."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_live import initial_memory, raw
from models.g9_shared import Spec, encode
from models.g9_tabletop import ACTIONS, advance, camera, expert, initial, target
from models.g10_context import TICKS, schedule, sensed, brief_visibility
from scripts.train.g9_fit import numpy_weights, save_json

DELAYS = (8, 32, 64, 128, 256, 480)


@lru_cache(maxsize=90)
def live_kernel(spec, mode, changes, batch, instruction):
    limit = 512 if mode in ("full", "wrong") else 64

    def run(p, seed):
        ids = jnp.arange(batch)
        st, goals, signs, notes, _ = schedule(ids, seed, changes)
        image = camera(st["stations"], st["tint"])
        template = (
            initial_memory(spec, batch, limit) if mode not in ("raw", "idle") else ()
        )
        visible = (
            brief_visibility(goals)
            if instruction in ("brief", "latched")
            else jnp.full(goals.shape, instruction == "continuous")
        )
        keep = jnp.ones((batch, 4), bool)

        def tick(c, item):
            state, mem, estimated, seen, remembered = c
            t, goal, sign, notice, show = item
            state = {
                **state,
                "goal": goal,
                "sign": sign,
                "known": (state["known"] & ~notice) | (state["previous"] >= 5),
            }
            obs = sensed(state, image, notice)
            remembered = jnp.where(show[:, None], obs["w"], remembered)
            ref, estimated, seen = raw(
                {**obs, "w": remembered}, estimated, seen & ~notice
            )
            ref = jnp.where(remembered[:, 2] > 0, ref, 0)
            obs = {
                **obs,
                "w": remembered
                if instruction == "latched"
                else jnp.where(show[:, None], obs["w"], jnp.zeros_like(obs["w"])),
            }
            if mode == "raw":
                action = ref
            elif mode == "idle":
                action = jnp.zeros(batch, jnp.int32)
            else:
                if mode == "block32":
                    mem = jax.tree.map(
                        lambda cur, blank: jnp.where(t % 32 == 0, blank, cur),
                        mem,
                        template,
                    )
                if mode == "wrong":
                    obs = {
                        **obs,
                        "u": jnp.where(obs["u"] >= 5, 11 - obs["u"], obs["u"]),
                    }
                z, mem = encode(
                    p,
                    spec,
                    {k: v[:, None] for k, v in obs.items()},
                    keep,
                    memory=mem,
                    window=limit,
                    offset=t,
                )
                action = (z[:, 0] @ p["read"][:, -ACTIONS:].astype(z.dtype)).argmax(-1)
            nxt = advance(state, action, t, True)
            reached = (nxt["pos"] == target(state)).all(1)
            measures = jnp.stack(
                (
                    reached,
                    action == 0,
                    action >= 5,
                    (action == 0) & ~reached,
                    ref == expert(state),
                    action,
                    goal,
                    notice,
                ),
                -1,
            )
            return (nxt, mem, estimated, seen, remembered), measures

        carry = (
            st,
            template,
            jnp.zeros((batch, 2), jnp.int32),
            jnp.zeros(batch, bool),
            jnp.zeros((batch, 8), jnp.uint8),
        )
        return jax.lax.scan(
            tick,
            carry,
            (jnp.arange(TICKS), goals.T, signs.swapaxes(0, 1), notes.T, visible.T),
        )[1]

    return jax.jit(run)


def mission(p, spec, mode, changes, seed, b=256, instruction="continuous"):
    data = np.asarray(
        live_kernel(spec, mode, changes, b, instruction)(p, jnp.int32(seed))
    )
    if instruction != "blank" and not data[:, :, 4].all():
        raise ValueError("Sensor-only reference disagrees with causal expert")
    success = []
    recovery = []
    probe = []
    for i in range(b):
        boundaries = np.r_[0, np.flatnonzero(np.diff(data[:, i, 6])) + 1, TICKS]
        success.append(
            float(
                np.mean(
                    [data[l:h, i, 0].any() for l, h in zip(boundaries, boundaries[1:])]
                )
            )
        )
        marks = np.flatnonzero(data[:, i, 7])
        if len(marks):
            recovery.append(
                float(np.mean([data[t : t + 32, i, 0].mean() for t in marks]))
            )
            probe.append(float(np.mean([data[t : t + 6, i, 2].any() for t in marks])))
    return dict(
        occupancy=data[:, :, 0].mean(0).tolist(),
        subgoals=success,
        hold=data[:, :, 1].mean(0).tolist(),
        tap=data[:, :, 2].mean(0).tolist(),
        false_hold=data[:, :, 3].mean(0).tolist(),
        post_change_occupancy=recovery,
        post_change_probe=probe,
        first_actions=data[:, 0, 5].tolist(),
    )


@lru_cache(maxsize=30)
def diagnostic(spec, b, limit):
    def test(p, seed):
        st = initial(jnp.arange(b), seed)
        dest = target(st)
        st = {**st, "pos": st["pos"].at[:, 0].set((dest[:, 0] + 1) % 4)}
        image = camera(st["stations"], st["tint"])
        ids = jnp.arange(TICKS)

        def trace(flip):
            u = jnp.where(ids == 2, 5 if flip == 1 else 6, 0).astype(jnp.int32)

            def at(t, prev):
                state = {
                    **st,
                    "sign": st["sign"] * flip,
                    "previous": jnp.full(b, prev),
                    "known": jnp.full(b, t >= 2),
                }
                return sensed(state, image, jnp.zeros(b, bool)), expert(state)

            r, labels = jax.vmap(at)(ids, u)
            return jax.tree.map(lambda v: v.swapaxes(0, 1), r), labels[-1]

        left, lt = trace(1)
        right, rt = trace(-1)
        same = jnp.all(
            jnp.stack(
                [jnp.array_equal(left[k], right[k]) for k in ("a", "v", "n", "w")]
            )
        )

        def predict(rows):
            z, _ = encode(p, spec, rows, jnp.ones((b, 4), bool), window=limit)
            return (
                z[:, jnp.array(DELAYS) + 2] @ p["read"][:, -7:].astype(z.dtype)
            ).argmax(-1)

        l, r = predict(left), predict(right)
        return (
            same,
            jnp.all(lt != rt),
            0.5
            * (
                (l == lt[:, None]).astype(jnp.float32)
                + (r == rt[:, None]).astype(jnp.float32)
            ),
            (l == lt[:, None]) & (r == rt[:, None]),
        )

    return jax.jit(test)


def delayed(p, spec, seed, b=128, limit=512):
    same, opposite, score, both = diagnostic(spec, b, limit)(p, jnp.int32(seed))
    if not bool(same) or not bool(opposite):
        raise ValueError("Broken sensor-alias contrast")
    return dict(
        delays=DELAYS,
        accuracy=np.asarray(score).T.tolist(),
        both=np.asarray(both).T.tolist(),
        pairs=b,
    )


def load_stage(stage):
    d = json.loads((stage / "receipt.json").read_text())
    with (stage / "weights.npz").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != d["inference_sha256"]:
            raise ValueError("Inference digest mismatch")
    return (
        d,
        jax.tree.map(jnp.asarray, numpy_weights(stage / "weights.npz")),
        Spec(**d["spec"]),
    )


def evaluate(stage, dest):
    d, p, s = load_stage(stage)
    dest.mkdir(exist_ok=True, parents=True)
    file = dest / f"{d['tag']}-{d['step']}.json"
    if file.exists():
        return
    terminal = d["step"] == d["goal_steps"]
    t = time.monotonic()
    out = dict(
        tag=d["tag"],
        step=d["step"],
        sha256=d["inference_sha256"],
        backend=jax.default_backend(),
    )
    if not terminal:
        out["development"] = mission(p, s, "full", 0, 117225, 64)
    else:
        for mode in ("full", "carry", "block32"):
            out[mode] = mission(p, s, mode, 0, 810537)
        out["changes"] = mission(p, s, "full", 2, 960133)
        out["wrong"] = mission(p, s, "wrong", 0, 810537)
        out["delayed"] = delayed(p, s, 730117)
    out["seconds"] = time.monotonic() - t
    save_json(file, out)
    print(
        d["tag"],
        d["step"],
        round(
            float(np.mean(out["full" if terminal else "development"]["occupancy"])), 4
        ),
        round(out["seconds"], 1),
        flush=True,
    )


def controls(path):
    s = Spec("gpt2", 128, 1, 128, "gelu")
    ref = {
        f"{name}-{changes}": mission(
            None, s, name, changes, 810537 if changes == 0 else 960133
        )
        for name in ("raw", "idle")
        for changes in (0, 2)
    }
    if min(np.mean(ref[f"raw-{n}"]["subgoals"]) for n in (0, 2)) < 0.99:
        raise ValueError("Physical positive control failed")
    path.parent.mkdir(exist_ok=True, parents=True)
    save_json(path, ref)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=Path, nargs="?")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    controls(a.dest / "reference.json") if a.raw else evaluate(a.stage, a.dest)
