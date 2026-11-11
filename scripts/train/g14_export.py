"""Export small runnable speech and field weights, original waveforms, a causal prefix and verified hashes."""

import argparse
import json
from pathlib import Path
import shutil
import wave
import numpy as np
from data.g14_wave import pcm
from scripts.train.g9_storage import write, sha, disks

ROOT = Path("/mnt/slwm-data/SLWM/g14")
PKG = ROOT / "release/inference"


def load(p):
    return json.loads(p.read_bytes())


def put(src, dest, expected=None):
    if expected and sha(src) != expected:
        raise ValueError("Changed selected checkpoint")
    tmp = dest.with_name("." + dest.name + ".new")
    shutil.copyfile(src, tmp)
    if sha(src) != sha(tmp):
        raise ValueError("Incomplete export")
    tmp.replace(dest)


def manifest(doc):
    files = {
        p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(PKG.iterdir())
        if p.is_file()
        and p.suffix != ""
        and p.name != "manifest.json"
        and not p.name.startswith(".")
    }
    write(
        PKG / "manifest.json",
        dict(**{k: v for k, v in doc.items() if k != "files"}, files=files),
    )


def export_speech():
    PKG.mkdir(parents=True, exist_ok=True)
    bases = tuple(p.parent.parent / "g14" for p in disks())
    choice = load(ROOT / "work/speech-choice.json")
    if choice["directory"] != "cue-semantic":
        raise ValueError("Unknown selected speech package")
    src = load(ROOT / "work/cue-semantic/choices.json")[choice["tag"]]
    loc = (
        Path("cue-semantic")
        / choice["tag"]
        / f"step-{choice['step']:05d}"
        / "weights.npz"
    )
    if sha(bases[1] / loc) != choice["weights"]:
        raise ValueError("Unmatched speech mirror")
    put(bases[0] / loc, PKG / "speech.npz", choice["weights"])
    put(ROOT / "data/semantic-utterances/normalizer.npz", PKG / "normalizer.npz")
    case = load(ROOT / "data/cases/fresh.json")[0]
    clip = case["commands"][0]["clip"]
    row = load(ROOT / "data/utterances/media.json")[clip]
    with np.load(ROOT / "data/semantic-utterances/fresh.npz") as z:
        i = int(np.flatnonzero(z["global_index"] == clip)[0])
        d = {k: z[k][i] for k in ("a", "l", "length")}
    np.savez_compressed(PKG / "cue-features.npz", **d, addresses=case["addresses"])
    with wave.open(str(PKG / "spoken-goal.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(
            np.clip(np.rint(pcm(row) * 32768), -32768, 32767).astype("<i2").tobytes()
        )
    doc = dict(
        encoder=load(ROOT / "data/semantic-utterances/complete.json"),
        speech_arm=src["arm"],
        threshold=load(ROOT / "data/cases/gate.json")["threshold"],
        speech_checkpoint=choice["weights"],
        example=dict(
            person=row["person"],
            clip=row["key"],
            transcript=row["text"],
            source=row["source"].split("/")[-1],
            seed=case["seed"],
        ),
    )
    manifest(doc)
    print("Speech artifact", doc["example"], flush=True)


def export_fields():
    from data.g14_table import Table
    from data.g14_policy import choose
    from scripts.data.g14_collect import obs_pack

    doc = load(PKG / "manifest.json")
    lock = load(ROOT / "work/final-control-lock.json")
    stage = lock["stage"]
    work = ROOT / "work" / stage
    bases = tuple(p.parent.parent / "g14/fields" for p in disks())
    names = dict(
        reference="reference.npz",
        long_reference=(
            "long-reference.npz" if (work / "long-reference.npz").is_file() else None
        ),
    )
    for file in names.values():
        if file:
            put(work / file, PKG / file)
    models = {}
    curves = load(work / "lock.json")
    for c in ("gpt2", "transport"):
        for seed in (1411, 1412, 1413):
            name = f"{c}-{seed}"
            tag = f"{stage}-{c}-protected-{seed}"
            entry = curves[tag]
            path = Path(tag) / f"step-{entry['selected']:05d}" / "weights.npz"
            if sha(bases[1] / path) != entry["sha256"]:
                raise ValueError("Unmatched field mirror")
            put(bases[0] / path, PKG / (name + ".npz"), entry["sha256"])
            models[name] = dict(
                **names,
                file=name + ".npz",
                seed=seed,
                parameters=entry["config"]["parameters"],
                step=entry["selected"],
                alpha=entry["alpha"],
                beta=entry.get("beta", 0.0),
                margin=entry["margin"],
            )
    case = load(ROOT / "data/cases/fresh.json")[0]
    cue = case["commands"][0]
    table = Table()
    table.cal = np.load(ROOT / "data/camera.npz")["H"]
    table.reset(case["seed"], case["shift"])
    stations = table.stations(case["seed"])
    variant = load(ROOT / "work/geometry-choice.json")["variant"]
    samples = {}
    media = load(ROOT / "data/utterances/media.json")
    with np.load(
        ROOT / "data/semantic-utterances/fresh.npz", allow_pickle=False
    ) as bank:
        for offset, name in ((1, "other-address"), (2, "other-block")):
            clip = case["commands"][offset]["clip"]
            r = media[clip]
            idx = int(np.flatnonzero(bank["global_index"] == clip)[0])
            np.savez_compressed(
                PKG / (name + "-features.npz"),
                **{k: bank[k][idx] for k in ("a", "l", "length")},
                addresses=case["addresses"],
            )
            with wave.open(str(PKG / (name + ".wav")), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(
                    np.clip(np.rint(pcm(r) * 32768), -32768, 32767)
                    .astype("<i2")
                    .tobytes()
                )
            samples[name] = dict(
                person=r["person"], clip=r["key"], transcript=r["text"]
            )
    doc["same_speaker_alternatives"] = samples
    # Advance a fixed causal prefix with deployed spoken geometry. Use solely camera estimates, observations and past controls at inference.
    for i in range(4):
        obs = table.observation()
        if i < 3:
            table.execute(
                choose(
                    table,
                    obs,
                    cue["heard_color"] if cue["accept"] else -1,
                    stations[cue["heard_station"]],
                    {"core": "geometry"},
                    variant,
                )[0]
            )
    np.savez_compressed(
        PKG / "scene.npz",
        **obs_pack(table),
        stations=stations,
        addresses=case["addresses"],
    )
    gate = load(work / "reference.json").get(
        "gate", dict(margin=0.04, alpha=0.0, beta=0.0)
    )
    doc.update(
        models=models,
        stage=stage,
        routed=lock["routed"],
        contact_only=lock["contact_only"],
        variant=variant,
        raw=dict(**names, **gate),
        physical_prefix_macros=3,
    )
    manifest(doc)
    print("Six matched physical fields", stage, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("speech", "fields"))
    a = p.parse_args()
    (export_speech if a.mode == "speech" else export_fields)()
