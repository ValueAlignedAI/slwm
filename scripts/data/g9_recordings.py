"""Prepare synchronized 10 Hz sensor banks, preserving whole-recording boundaries."""
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf

SPLITS = {"train": "dev-train-tau", "dev": "dev-test-tau", "test": "dev-test-sony"}
EDGES = np.geomspace(60.0, 7900.0, 17)


def spectrum(wave, rate):
    """Four 16-band channel spectra per 100 ms; mono sources explicitly repeat a channel."""
    length = rate // 10
    x = wave[:len(wave) // length * length].reshape(-1, length, wave.shape[-1])
    if x.shape[-1] < 4:
        x = np.repeat(x[:, :, :1], 4, axis=2)
    fft = np.fft.rfft(x[:, :, :4] * np.hanning(length)[None, :, None], axis=1)
    power = np.abs(fft / length) ** 2
    hz = np.fft.rfftfreq(length, 1 / rate)
    bands = np.stack([power[:, (hz >= low) & (hz < high)].sum(1)
                      for low, high in zip(EDGES[:-1], EDGES[1:])], axis=-1)
    db = 10 * np.log10(np.maximum(bands, 1e-12))
    return np.clip((db + 90) * 255 / 80, 0, 255).astype(np.uint8).reshape(len(x), 64)


def starss_one(root, role, stem):
    with zipfile.ZipFile(root / "foa_dev.zip") as z:
        with z.open(f"foa_dev/{role}/{stem}.wav") as stream:
            wave, rate = sf.read(io.BytesIO(stream.read()), dtype="float32", always_2d=True)
    a = spectrum(wave, rate)
    with zipfile.ZipFile(root / "video_dev.zip") as z:
        video = z.read(f"video_dev/{role}/{stem}.mp4")
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=root.parent.parent / "g9/data") as tmp:
        tmp.write(video); tmp.flush()
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-threads", "1", "-i", tmp.name,
                              "-vf", "fps=10,scale=16:16:flags=area", "-threads", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], check=True, capture_output=True)
    v = np.frombuffer(raw.stdout, np.uint8).reshape(-1, 768)
    ticks = min(len(v), len(a))
    if ticks < 33 or abs(len(v)-len(a)) > 3:
        raise ValueError(f"Invalid paired clock {stem}: {len(v)} video, {len(a)} audio bins")
    with zipfile.ZipFile(root / "metadata_dev.zip") as z:
        rows = csv.reader(io.StringIO(z.read(f"metadata_dev/{role}/{stem}.csv").decode()))
        entries = [[] for _ in range(ticks)]
        for row in rows:
            tick, kind, _, azimuth, elevation, distance = map(int, row[:6])
            if 0 <= tick < ticks:
                az, el = np.deg2rad([azimuth, elevation])
                xyz = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)]) * distance/600
                entries[tick].append((kind, xyz))
    n = np.zeros((ticks, 12), np.float32)
    w = np.zeros((ticks, 8), np.uint8)
    for i, events in enumerate(entries):
        w[i, 0] = 24 if events else 25
        if events:
            kinds = sorted({k for k, _ in events})[:7]
            w[i, 1:1+len(kinds)] = 32 + np.array(kinds)
            n[i, :3] = np.clip(np.mean([xyz for _, xyz in events], axis=0), -1, 1)
            n[i, 3] = min(1, len(events)/4)
    return dict(a=a[:ticks], v=v[:ticks], n=n, w=w)


def pack(records, destination):
    names = [name for name, _ in records]
    lengths = np.array([len(d["a"]) for _, d in records], np.int32)
    arrays = {k: np.concatenate([d[k] for _, d in records]) for k in ("a","v","n","w")}
    tmp = destination.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays, lengths=lengths, names=np.array(names))
    os.replace(tmp, destination)
    return {"recordings": names, "ticks": int(lengths.sum()), "sha256": hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()}


def main(root, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root/"video_dev.zip") as z:
        members = z.namelist()
    receipt = {}
    for split, role in SPLITS.items():
        names = sorted(Path(n).stem for n in members if f"/{role}/" in n and n.endswith(".mp4"))
        with ThreadPoolExecutor(3) as pool:
            records = list(zip(names, pool.map(lambda stem: starss_one(root, role, stem), names)))
        receipt[split] = pack(records, dest/f"starss-{split}.npz")
        print(split, len(names), receipt[split]["ticks"], flush=True)
    (dest/"starss.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    main(args.root, args.dest)
