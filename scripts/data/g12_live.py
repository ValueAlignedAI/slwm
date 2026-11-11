"""Prepare a fixed, clock-driven replay with every completed prospective instruction and genuine idle ticks."""

import argparse
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from g11_events import decoy, G10
from g12_events import ROOT, key, write
from g12_media import source
from g12_banks import packed_record


def selections(root):
    groups = json.loads((root / "groups.json").read_bytes())
    cat = json.loads((root / "annotations.json").read_bytes())
    choice = {}
    for fold, roles in groups.items():
        for family in sorted({cat[n]["family"] for n in roles["confirm"]}):
            pool = [
                n
                for n in roles["confirm"]
                if cat[n]["family"] == family
                and cat[n]["duration"] >= 90
                and sum(s["end"] <= 180 for s in cat[n]["speech"]) >= 3
            ]
            choice[family] = (fold, min(pool, key=lambda n: key("live/" + n)))
    return choice, cat


def prepare(root):
    picks, cat = selections(root)
    dst = root.parent / "live-data"
    dst.mkdir(exist_ok=True)
    (dst / "fresh").mkdir(exist_ok=True)
    for path, target in (
        (dst / "fresh/media", root / "fresh/media"),
        (dst / "whisper-small", root / "whisper-small"),
    ):
        if not path.exists():
            path.symlink_to(target, target_is_directory=True)
    instructions = {}
    for fold, name in picks.values():
        doc = cat[name]
        speech = [s for s in doc["speech"] if s["end"] <= 180]
        instructions[name] = dict(
            events=[dict(cue=s["id"], wrong=s["id"]) for s in speech], speech=speech
        )
    write(dst / "annotations.json", instructions)
    print(
        "All prospectively completed directives",
        [(n, len(v["speech"])) for n, v in instructions.items()],
        flush=True,
    )


def audio(file, spoken, out):
    import wave

    data = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(file),
            "-t",
            "600",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        timeout=150,
    )
    pcm = np.frombuffer(data, "<f4")
    left = int(np.ceil(spoken["start"] * 16000))
    right = int(np.floor(spoken["end"] * 16000))
    raw = np.clip(np.rint(pcm[left:right] * 32768), -32768, 32767).astype("<i2")
    with wave.open(str(out), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(raw.tobytes())


def pack(root):
    picks, catalog = selections(root)
    dest = root.parent / "release/examples"
    dest.mkdir(parents=True, exist_ok=True)
    source_actions = {
        r["video_name"]: r["events"]
        for r in json.loads((G10 / "labels.json").read_bytes())
        if r["video_name"] in [v[1] for v in picks.values()]
    }
    public = []
    for fam, (fold, name) in picks.items():
        cat = catalog[name]
        stop = min(180, math.floor(cat["duration"]) - 1)
        span = {i * 10 - 1: [] for i in range(1, stop + 1)}
        spoken = sorted(
            [s for s in cat["speech"] if s["end"] <= stop],
            key=lambda r: (r["end"], r["id"]),
        )
        by_id = {s["id"]: s for s in spoken}
        for i, cur in enumerate(spoken):
            old, _ = decoy(cur, spoken[:i])
            tick = math.ceil(cur["end"] * 10) - 1
            span.setdefault(tick, []).append((cur["id"], old["id"] if old else -1))
        seq = []
        labels = []
        for tick, utterances in sorted(span.items()):
            cue, wrong = utterances[-1] if utterances else (-1, -1)
            t = (tick + 1) / 10
            blank = dict(
                id=tick,
                tick=tick,
                cue=cue,
                wrong=wrong,
                verbs=[],
                previous="",
                starts=[],
                ends=[],
                conflict=False,
            )
            seq.append(blank)
            # Used for offline scoring only. No quiet/future flag enters the feature archive.
            quiet = not any(
                e["label"] == "Fine grained action"
                and e["end"] > t
                and e["start"] < t + 8
                for e in source_actions[name]
            )
            labels.append(
                dict(
                    tick=tick,
                    instruction=cue,
                    foil=wrong,
                    quiet_next_eight_seconds=quiet,
                )
            )
        rows = packed_record(
            root,
            name,
            dict(rows=seq, family=fam),
            cat,
            {},
            root.parent / "live-data/speech-small",
        )
        valid = {r["cut_tick"] for _, r in rows}
        labels = [r for r in labels if r["tick"] in valid]
        signal = {
            k: np.stack([x[k] for x, _ in rows])
            for k in ("scene", "ac", "wc", "words", "wrong", "human", "wrong_human")
        }
        signal["seconds"] = np.array([(r["cut_tick"] + 1) / 10 for _, r in rows])
        p = dest / (fam + "-stream.npz")
        np.savez_compressed(p, **signal)
        chose = min(
            (r for r in labels if r["foil"] >= 0),
            key=lambda r: key("listen/" + name + "/" + str(r["tick"])),
            default=None,
        )
        example = None
        if chose:
            file = source(root, name, "media", "-av.mp4")
            for kind, ident in (
                ("true", chose["instruction"]),
                ("foil", chose["foil"]),
            ):
                audio(file, by_id[ident], dest / (fam + "-" + kind + ".wav"))
            example = dict(
                second=(chose["tick"] + 1) / 10,
                index=next(
                    i for i, l in enumerate(labels) if l["tick"] == chose["tick"]
                ),
                instruction=by_id[chose["instruction"]]["text"],
                foil=by_id[chose["foil"]]["text"],
            )
        write(dest / (fam + "-observations.json"), labels)
        public.append(
            dict(
                family=fam,
                fold=fold,
                recording=name,
                seconds=stop,
                cycles=len(rows),
                spoken=sum(r["instruction"] >= 0 for r in labels),
                quiet=sum(r["quiet_next_eight_seconds"] for r in labels),
                audio_example=example,
            )
        )
    write(dest / "streams.json", public)
    print("Continuous public replay", public, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "pack"))
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    (prepare if a.stage == "prepare" else pack)(a.root)
