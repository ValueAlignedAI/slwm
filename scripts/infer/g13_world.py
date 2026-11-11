"""Use a released G13 field to propose physical outcomes and route only accepted predictions."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import wave
import jax
import jax.numpy as jnp
import numpy as np
from data.g10_language import text_features
from data.g13_goals import COLORS, DIRECTIONS, semantics
from data.g13_speech_features import features
from models.g13_world import Spec, propose
from models.g13_speech import predict as word_predict
from training.g13_physics import reference
from training.g13_adaptive import response, apply_prior

BASE = (
    Path(__file__).resolve().parents[2]
    / "experiments/multimodal/t13/grounded-transfer-20261111/inference"
)


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def artifact(pkg, manifest, name):
    record = manifest["files"][name]
    file = pkg / name
    if checksum(file) != record["sha256"]:
        raise ValueError("Inference artifact failed verification: " + name)
    return read(file)


def causal_rows(scene, linear, adaptive, kind):
    r = {k: np.asarray(scene[k]) for k in ("px", "geo", "kin", "plan", "goal", "bind")}
    zero = np.zeros_like(r["plan"])
    r["alt_plan"] = zero
    r["wrong"] = np.zeros_like(r["goal"])
    r["wb"] = np.zeros_like(r["bind"])
    reset = np.broadcast_to(r["kin"][:, -1:], r["kin"].shape).copy()
    reset[:, :, 2:] = 0
    for key, controls in (("active", r["plan"]), ("wrong", zero), ("zero", zero)):
        r["ref_" + key] = reference(r["kin"], controls, linear, True)
        r["adapt_" + key] = response(r["kin"], controls, kind, adaptive)
    r["ref_reset"] = reference(reset, r["plan"], linear, True)
    r["ref_pass_reset"] = reference(reset, zero, linear, False)
    r["adapt_reset"] = response(reset, r["plan"], kind, adaptive)
    r["ref_passive"] = reference(r["kin"], zero, linear, False)
    r["p_goal"] = apply_prior(r["bind"], adaptive["mass"])
    r["p_wrong"] = r["p_none"] = apply_prior(r["wb"], adaptive["mass"])
    return r


@lru_cache(maxsize=12)
def kernel(core, recipe):
    @jax.jit
    def run(p, r):
        o = propose(p, Spec(), core, recipe, r, "correct")
        return dict(
            mover=jax.nn.softmax(o["mover"]),
            confidence=jax.nn.sigmoid(o["certainty"]),
            active=o["active"],
            passive=o["passive"],
            action=o["action"],
            field=o["field"],
        )

    return run


def model(package, which):
    index = json.loads((package / "manifest.json").read_bytes())
    m = index["worlds"][which]
    p = {k: jnp.asarray(x) for k, x in artifact(package, index, m["file"]).items()}
    return (
        index,
        m,
        p,
        artifact(package, index, "system-id.npz"),
        artifact(package, index, "adaptive.npz"),
    )


def encoded(scene, instruction):
    b = len(scene["px"])
    _, _, role, _ = semantics(instruction)
    return dict(
        **{k: scene[k] for k in ("px", "geo", "kin", "plan")},
        goal=np.broadcast_to(text_features(instruction), (b, 128)),
        bind=np.broadcast_to(role, (b, 4, 4)),
    )


def forecast(scene, net):
    index, m, p, linear, adaptive = net
    r = causal_rows(scene, linear, adaptive, index["adaptive_kind"])
    o = {
        k: np.asarray(x)
        for k, x in kernel(m["core"], m["recipe"])(
            p, {k: jnp.asarray(x) for k, x in r.items()}
        ).items()
    }
    nominal = (
        r["adapt_active"]
        if m["recipe"] in ("adaptive", "anchored")
        else r["ref_active"]
    )
    o["calibrated"] = nominal + m["alpha"] * (o["active"] - nominal)
    return o, r


def commit(o, m, i=0):
    prob = o["mover"][i]
    conf = o["confidence"][i]
    result = {}
    if 0.5 * (prob.max() + conf[0]) >= m["thresholds"][0]:
        result["object"] = ("no_visible_change", *COLORS)[int(prob.argmax())]
        result["object_probability"] = float(prob.max())
    if conf[1] >= m["thresholds"][1]:
        result["predicted_xy_change_m"] = o["calibrated"][i, -1].tolist()
    return dict(route="forecast" if result else "internal_only", **result)


def direct_action(scene, net, package):
    from data.g13_design import action_design

    index, m, _, _, _ = net
    if m["recipe"] != "linear":
        raise ValueError("Direct action readouts use the linear-reference field")
    raw = artifact(package, index, "direct-raw.npz")

    def project(x, f):
        return (x - f["mu"]) / f["sd"] @ f["beta"] + f["base"]

    prior = project(action_design(scene), raw)
    o, _ = forecast(scene, net)
    z = 0.5 * (o["field"][:, 0] + o["field"][:, 1:].mean(1))
    fit = artifact(package, index, "direct-" + m["core"] + ".npz")
    proposal = prior + float(fit["alpha"]) * project(
        np.column_stack((prior / 0.08, z[:, :126])), fit
    )
    return proposal, prior


def select_plan(scene, net, goal):
    # Offline candidates share the same observed prefix. The score margin is a development distance.
    commands = np.asarray(
        [[0, 0], [-0.014, 0], [0.014, 0], [0, -0.014], [0, 0.014]], "f4"
    )
    many = {k: np.repeat(v, 5, axis=0) for k, v in scene.items()}
    many["plan"] = np.repeat(commands[:, None, :], 4, 1)
    o, _ = forecast(many, net)
    cost = np.linalg.norm(o["calibrated"][:, -1] - goal, axis=1)
    best = int(np.argmin(cost))
    gain = float(cost[0] - cost[best])
    margin = net[1]["dev_calibrated_mm"] / 1000.0
    return dict(
        route="plan" if best and gain > margin else "no_op",
        delta_setpoint_m=(
            commands[best].tolist() if best and gain > margin else [0.0, 0.0]
        ),
        advantage_m=gain,
    )


def acoustic(package, index, path):
    with wave.open(str(path), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected a mono 16 kHz PCM16 word")
        a = np.frombuffer(f.readframes(f.getnframes()), "<i2").astype("f4") / 32768
    p = {
        k: jnp.asarray(v)
        for k, v in artifact(package, index, "direction-codec.npz").items()
    }
    return np.asarray(
        jax.nn.softmax(
            word_predict(p, jnp.asarray(features(a)[None]), portable=True)[1]
        )
    )[0]


def voice(scene, net, package, path, phrase, arm="protected", target="effector"):
    index, m, _, _, _ = net
    if m["recipe"] != "anchored":
        raise ValueError("Use an anchored model for the direction readout")
    q = acoustic(package, index, path)
    clean = " ".join(
        re.sub(
            r"\b(leftwards?|rightwards?|left|right|upwards?|downwards?|up|down|upper|lower|top|bottom)\b",
            " ",
            phrase,
            flags=re.I,
        ).split()
    )
    row = encoded(scene, clean)
    row["goal"] = (
        q
        @ np.stack([text_features(clean + " " + word) for word in DIRECTIONS]).astype(
            "f4"
        )
    )[None]
    o, _ = forecast(row, net)
    z = o["field"]
    bind = row["bind"][:, :, 0]
    source = (z[:, 1:] * bind[:, :, None]).sum(1)
    where = np.where(bind.sum(1, keepdims=True) > 0, source, z[:, 1:].mean(1))
    x = 0.5 * (z[:, 0] + where)
    if arm == "protected":
        x = np.column_stack((x[:, :124], q[None]))
    f = artifact(package, index, f"voice-{m['core']}-{target}-{arm}.npz")
    score = (((x - f["mean"]) / f["std"]) @ f["beta"] + f["base"]) * float(
        f["temperature"]
    )
    prob = np.exp(score - score.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    return dict(
        heard=DIRECTIONS[int(q.argmax())],
        forecast=DIRECTIONS[int(prob.argmax())],
        probability=float(prob.max()),
        target=target,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("--package", type=Path, default=BASE)
    p.add_argument("--model", default="transport-linear")
    p.add_argument("--instruction", default="")
    p.add_argument("--event", type=int)
    p.add_argument("--wav", type=Path)
    p.add_argument(
        "--direction-target", choices=("effector", "object"), default="effector"
    )
    p.add_argument("--direct-action", action="store_true")
    p.add_argument("--target", type=float, nargs=2, metavar=("DX", "DY"))
    p.add_argument("--plan", help="Four proposed planar offsets as JSON")
    a = p.parse_args()
    s = read(a.scene)
    net = model(a.package, a.model)
    n = len(s["px"])
    for i in (range(n) if a.event is None else [a.event]):
        if i < 0 or i >= n:
            raise ValueError("Requested clock is outside the stream")
        scene = {k: s[k][i : i + 1] for k in ("px", "geo", "kin", "plan")}
        if a.plan:
            cmds = np.asarray(json.loads(a.plan), "f4")
            if cmds.shape != (4, 2) or not np.isfinite(cmds).all():
                raise ValueError("Expected four finite 2D controls")
            scene["plan"] = cmds[None]
        if a.wav:
            if a.event is None:
                raise ValueError("Select one event for a spoken-goal intervention")
            result = voice(
                scene, net, a.package, a.wav, a.instruction, target=a.direction_target
            )
        else:
            row = encoded(scene, a.instruction)
            if a.direct_action:
                forecasted, raw = direct_action(row, net, a.package)
                result = dict(
                    route="forecast",
                    four_control_sum_m=forecasted[0].tolist(),
                    raw_reference_m=raw[0].tolist(),
                )
            else:
                result = (
                    select_plan(row, net, np.asarray(a.target))
                    if a.target is not None
                    else commit(forecast(row, net)[0], net[1])
                )
        print(json.dumps(dict(clock=int(s["tick"][i]), **result)), flush=True)


if __name__ == "__main__":
    main()
