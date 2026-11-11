"""Run a released spoken goal and a camera/telemetry field on a chronological tabletop prefix."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave
import numpy as np
from data.g14_grid import SLOTS, T, labels
from data.g14_policy import features, candidates, choose
from data.g14_signals import raw_predict, attach
from data.g14_table import COLORS
from models.g14_world import predict
from scripts.data.g14_cases import decode

BASE = (
    Path(__file__).resolve().parents[2]
    / "experiments/multimodal/t14/spoken-closed-loop-20261111/inference"
)


def read(file):
    with np.load(file, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check(pkg, man, name):
    p = pkg / name
    with p.open("rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    if got != man["files"][name]["sha256"]:
        raise ValueError("Inference checksum mismatch: " + name)
    return p


def encode_wav(path, root, pkg, man, device):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getframerate(), w.getsampwidth()) != (1, 16000, 2):
            raise ValueError("Expected a mono 16 kHz signed 16-bit WAV")
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype("f4") / 32768.0
    if not len(x):
        raise ValueError("An empty recording has no completed cue")
    cfg = man["encoder"]
    src = str(root or cfg["edge"])
    kw = dict(local_files_only=True) if root else dict(revision=cfg["revision"])
    dev = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(src, **kw)
    enc = (
        WhisperForConditionalGeneration.from_pretrained(src, dtype=dtype, **kw)
        .model.encoder.to(dev)
        .eval()
    )
    normal = read(check(pkg, man, "normalizer.npz"))
    step = np.rint((np.arange(T) * 0.032 + 0.016) / 0.02).astype(int)
    with torch.inference_mode():
        inp = proc.feature_extractor(
            [x], sampling_rate=16000, return_tensors="pt", device=dev
        ).input_features.to(dev, dtype=dtype)
        vec = (
            (
                enc(inp, return_dict=True).last_hidden_state[:, step].float()
                @ torch.tensor(normal["basis"], device=dev)
            )
            .cpu()
            .numpy()[0]
        )
    n = min(T, int(np.ceil(len(x) / 512)))
    return dict(
        a=(
            np.clip(0.5 + (vec - normal["mean"]) / (6 * normal["std"]), 0, 1)
            * (np.arange(T) < n)[:, None]
        ).astype("f2"),
        length=np.array(n),
        l=np.zeros((T, 96), "u1"),
    )


def speech(row, pkg, man, addresses):
    import jax
    import jax.numpy as jnp
    from models.g14_cue import forward

    p = {k: jnp.asarray(v) for k, v in read(check(pkg, man, "speech.npz")).items()}
    p0 = forward(
        p,
        jnp.asarray(row["a"])[None],
        jnp.asarray(row["l"])[None],
        jnp.asarray(row["length"])[None],
        man["speech_arm"],
    )[0]
    p0 = [np.asarray(jax.nn.softmax(v, axis=-1))[0] for v in p0]
    c, st, prob = decode(*p0, addresses)
    ok = bool(c < 3 and prob >= man["threshold"])
    return dict(
        color=c,
        station=st,
        confidence=prob,
        accepted=ok,
        unconstrained=[SLOTS[i][int(v.argmax())] for i, v in enumerate(p0)],
    )


def active_goal(
    scene,
    pkg,
    man,
    mode="prepared",
    utterance=None,
    encoder=None,
    device="auto",
    feature_file=None,
):
    if mode == "absent":
        return dict(
            color=-1, station=0, confidence=0.0, accepted=False, unconstrained=[]
        )
    if mode == "text":
        word = labels(utterance)
        addrs = [tuple(a) for a in scene["addresses"].tolist()]
        address = tuple(word[1:])
        known = address in addrs and word[0] < 3
        return dict(
            color=int(word[0]),
            station=addrs.index(address) if known else 0,
            confidence=1.0,
            accepted=bool(known),
            unconstrained=[SLOTS[i][int(v)] for i, v in enumerate(word)],
        )
    row = (
        read(feature_file or check(pkg, man, "cue-features.npz"))
        if mode == "prepared"
        else encode_wav(utterance, encoder, pkg, man, device)
    )
    return speech(row, pkg, man, scene["addresses"])


def infer(scene, pkg, man, goal, core, seed, inspect=False):
    history = [
        {k: scene[k][i] for k in ("q", "p", "u", "seen", "motor", "px")}
        for i in range(len(scene["p"]))
    ]
    state = SimpleNamespace(history=history)
    c = goal["color"] if goal["accepted"] else -1
    target = scene["stations"][goal["station"]]

    def loader(stage, kind, arm, seed):
        key = f"{kind}-{seed}"
        m = man["models"][key] if kind != "raw" else man["raw"]
        base = read(check(pkg, man, m["reference"]))
        long = (
            read(check(pkg, man, m["long_reference"]))
            if m.get("long_reference")
            else None
        )
        model = read(check(pkg, man, m["file"])) if kind != "raw" else None
        return base, long, model, m

    pol = dict(
        core=core,
        stage=man["stage"],
        routed=man["routed"],
        contact_only=man["contact_only"] and core in ("gpt2", "transport", "raw"),
        arm="protected",
        seed=seed,
    )
    action, info = choose(
        state, history[-1], c, target, pol, man["variant"], loader=loader
    )
    result = dict(
        goal=goal,
        core=core,
        seed=seed,
        route=info["route"],
        candidate=info["chosen"],
        command_m=action.astype(float).tolist(),
    )
    if inspect and c >= 0 and core in ("gpt2", "transport"):
        opts = candidates(history[-1], c, target, man["variant"], man["routed"])
        r = features(state, c, target, opts)
        ref, long, p, meta = loader(man["stage"], core, "protected", seed)
        attach(r, raw_predict(r, ref))
        y, risk, field = predict(p, r, core, "protected", np)
        i = info["chosen"]
        rotation = r["basis"][0]
        physical = (y[0, i, :10].reshape(5, 2) @ rotation.T) * 50
        result.update(
            field_coordinates=int(field.shape[-1]),
            field=field[0].tolist(),
            forecast_effector_and_four_blocks_mm=physical.tolist(),
            estimated_large_utility_error=float(risk[0, i]),
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, default=BASE)
    p.add_argument("--scene", type=Path)
    p.add_argument("--features", type=Path)
    m = p.add_mutually_exclusive_group()
    m.add_argument("--wav", type=Path)
    m.add_argument("--text")
    m.add_argument("--absent", action="store_true")
    p.add_argument("--encoder", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument(
        "--core",
        choices=("gpt2", "transport", "raw", "geometry", "contact_speed"),
        default="transport",
    )
    p.add_argument("--seed", type=int, choices=(1411, 1412, 1413), default=1411)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--speech-only", action="store_true")
    a = p.parse_args()
    man = json.loads((a.package / "manifest.json").read_bytes())
    scene = read(
        a.scene
        or check(
            a.package, man, "scene.npz" if not a.speech_only else "cue-features.npz"
        )
    )
    mode = (
        "wav" if a.wav else "text" if a.text else "absent" if a.absent else "prepared"
    )
    goal = active_goal(
        scene, a.package, man, mode, a.wav or a.text, a.encoder, a.device, a.features
    )
    print(
        json.dumps(
            (
                goal
                if a.speech_only
                else infer(scene, a.package, man, goal, a.core, a.seed, a.inspect)
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
