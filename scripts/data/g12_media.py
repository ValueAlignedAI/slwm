"""Extract timestamped frozen image features and blinded before/after device observations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

G10 = Path("/mnt/slwm-data/SLWM/g10/data/holo")
G11 = Path("/mnt/slwm-data/SLWM/g11/data")
ROOT = Path("/mnt/slwm-data/SLWM/g12/data")


def source(root, name, kind, extension):
    paths = [
        G10 / kind / (name + extension),
        G11 / "fresh" / kind / (name + extension),
        root / "fresh" / kind / (name + extension),
    ]
    return next((p for p in paths if p.is_file()), None)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def frame_at(file, when):
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-ss",
            f"{when:.4f}",
            "-i",
            str(file),
            "-frames:v",
            "1",
            "-vf",
            "scale=320:180:flags=area",
            "-threads",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout=40,
    )
    return Image.frombytes("RGB", (320, 180), data)


def panel_images(root):
    panel = json.loads((root / "panel.json").read_bytes())
    out = root / "sheets"
    cache = out / "tiles"
    cache.mkdir(parents=True, exist_ok=True)

    def extract(row):
        target = cache / (row["key"] + ".jpg")
        if target.exists():
            return
        file = source(root, row["recording"], "media", "-av.mp4")
        if file is None:
            return
        canvas = Image.new("RGB", (1280, 180))
        for j, t in enumerate(row["times"]):
            canvas.paste(frame_at(file, t), (j * 320, 0))
        temp = target.with_suffix(".new.jpg")
        canvas.save(temp, quality=91)
        temp.replace(target)

    while True:
        remaining = [
            row for row in panel if not (cache / (row["key"] + ".jpg")).exists()
        ]
        if not remaining:
            break
        ready = [
            r for r in remaining if source(root, r["recording"], "media", "-av.mp4")
        ]
        if not ready:
            time.sleep(4)
            continue
        with ThreadPoolExecutor(7) as pool:
            list(pool.map(extract, ready))
        print("Blinded pairs remaining", len(remaining) - len(ready), flush=True)
    order = [(digest("G12/panel/first/" + r["key"]), r) for r in panel]
    repeat = sorted(panel, key=lambda r: digest("G12/blind-repeat/" + r["key"]))[
        : max(1, len(panel) // 12)
    ]
    order.extend((digest("G12/panel/duplicate/" + r["key"]), r) for r in repeat)
    order.sort(key=lambda t: t[0])
    index = []
    for i, (opaque, row) in enumerate(order):
        if i % 16 == 0:
            sheet = Image.new("RGB", (1936, 1240), "#eeeeee")
        tile = Image.open(cache / (row["key"] + ".jpg"))
        flip = int(opaque[-1], 16) % 2 == 1
        groups = (2, 3, 0, 1) if flip else (0, 1, 2, 3)
        x = (i % 2) * 968
        y = ((i % 16) // 2) * 155
        for col, k in enumerate(groups):
            crop = tile.crop((320 * k, 0, 320 * (k + 1), 180)).resize(
                (237, 133), Image.Resampling.LANCZOS
            )
            sheet.paste(crop, (x + col * 239 + (6 if col >= 2 else 0), y + 20))
        ImageDraw.Draw(sheet).text((x + 3, y + 2), f"{i+1:04d}    X", fill="black")
        ImageDraw.Draw(sheet).text((x + 486, y + 2), "Y", fill="black")
        index.append(dict(number=i + 1, key=row["key"], reversed=flip))
        if (i + 1) % 16 == 0 or i + 1 == len(order):
            sheet.save(out / f"sheet-{i//16+1:02d}.jpg", quality=92)
    (out / "index.json").write_text(json.dumps(index))
    (out / "complete").touch()


def vision(root):
    import torch
    from torchvision.models import resnet18

    weights = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    with weights.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if not sha.startswith("f37072fd"):
        raise ValueError("Unexpected frozen visual encoder")
    model = resnet18()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model = model.eval().cuda().half().to(memory_format=torch.channels_last)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").half()[
        None, :, None, None
    ]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").half()[None, :, None, None]
    names = sorted(json.loads((root / "records.json").read_bytes()))
    import PyNvVideoCodec as nv
    import torch.nn.functional as F

    dest = root / "vision-nv"
    dest.mkdir(exist_ok=True)

    def frames(file):
        demux = nv.CreateDemuxer(filename=str(file))
        decoder = nv.CreateDecoder(
            gpuid=0,
            codec=demux.GetNvCodecId(),
            usedevicememory=True,
            outputColorType=nv.OutputColorType.RGB,
        )
        unit = demux.GetTimebaseNum() / demux.GetTimebaseDen()
        previous = -2.0
        times = []
        batch = []
        parts = []

        def encode():
            rgb = torch.stack(batch).half() / 255
            features = (
                model((rgb - mean) / std).float().cpu().numpy().astype(np.float16)
            )
            parts.append(features)
            batch.clear()

        for packet in demux:
            for frame in decoder.Decode(packet):
                t = float(frame.getPTS()) * unit
                if t >= 600:
                    break
                if t < previous + 0.98:
                    continue
                rgb = torch.from_dlpack(frame).permute(2, 0, 1)[None].float()
                resized = (
                    F.interpolate(rgb, size=(108, 192), mode="area")[0].round().byte()
                )
                torch.cuda.current_stream().synchronize()
                batch.append(resized)
                times.append(t)
                previous = t
                if len(batch) == 96:
                    encode()
            else:
                continue
            break
        if batch:
            encode()
        if not times or np.any(np.diff(times) <= 0):
            raise ValueError("Invalid actual video frame clock")
        return np.asarray(times, np.float64), np.concatenate(parts)

    with torch.inference_mode():
        while True:
            behind = [n for n in names if not (dest / (n + ".npz")).exists()]
            if not behind:
                break
            ready = [n for n in behind if source(root, n, "media", "-av.mp4")][:8]
            if not ready:
                time.sleep(4)
                continue
            for name in ready:
                times, features = frames(source(root, name, "media", "-av.mp4"))
                path = dest / (name + ".npz")
                tmp = path.with_suffix(".new.npz")
                np.savez_compressed(tmp, time=times, x=features)
                tmp.replace(path)
            print(
                "Timestamped NVDEC feature streams remaining",
                len(behind) - len(ready),
                flush=True,
            )
    (dest / "receipt.json").write_text(
        json.dumps(
            dict(
                sha256=sha,
                dimensions=512,
                weights="ResNet-18 ImageNet V1",
                recordings=len(names),
            )
        )
    )
    (dest / "complete").touch()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("vision", "pairs"))
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    (vision if a.stage == "vision" else panel_images)(a.root)
