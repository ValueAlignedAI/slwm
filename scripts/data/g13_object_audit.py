"""Make blinded, source-balanced before/after evidence for independent block-motion readback."""

import json
import struct
import cv2
import numpy as np
from PIL import Image, ImageDraw
from g13_robot import R, rank, episode
from g13_acquire import write


def frame_pair(rec, t):
    path = R / "raw" / ("language_table-train.tfrecord-" + rec["shard"])
    with path.open("rb") as f:
        f.seek(rec["offset"])
        n = struct.unpack("<Q", f.read(8))[0]
        f.read(4)
        _, _, _, frames, _, _ = episode(f.read(n))
    return [
        Image.fromarray(
            cv2.imdecode(
                np.frombuffer(frames[j], np.uint8), cv2.IMREAD_REDUCED_COLOR_2
            )[:, :, ::-1]
        )
        for j in (t, -1)
    ]


def main():
    out = R / "audit"
    out.mkdir(exist_ok=True)
    ep = json.loads((R / "banks/episodes.json").read_bytes())
    picked = []
    for role in ("dev", "test"):
        with np.load(R / "banks" / (role + ".npz")) as z:
            m = z["mover"]
            ids = z["episode"]
            first = z["first"]
            tick = z["tick"]
        for label in range(-1, 5):
            ix = sorted(
                np.flatnonzero(first & (m == label)),
                key=lambda i: rank("blind/" + ep[ids[i]]["episode"]),
            )
            groups = set()
            for i in ix:
                src = ep[ids[i]]
                group = src["group"]
                if group in groups:
                    continue
                groups.add(group)
                picked.append(
                    dict(
                        role=role,
                        episode=int(ids[i]),
                        tick=int(tick[i]),
                        detector=label,
                    )
                )
                if len(groups) == (15 if label < 0 else 10):
                    break
    items = [(rank("order/" + str(x)), x) for x in picked]
    repeat = sorted(picked, key=lambda x: rank("repeat/" + str(x)))[:12]
    items.extend((rank("again/" + str(x)), x) for x in repeat)
    items.sort()
    index = []
    for i, (opaque, item) in enumerate(items):
        frames = frame_pair(ep[item["episode"]], item["tick"])
        reverse = int(opaque[-1], 16) % 2
        if reverse:
            frames.reverse()
        if i % 12 == 0:
            sheet = Image.new("RGB", (1310, 1190), "#eeeeee")
        x = i % 2 * 655
        y = i % 12 // 2 * 198
        ImageDraw.Draw(sheet).text((x + 3, y + 1), f"{i+1:03d}", fill="black")
        sheet.paste(frames[0], (x, y + 15))
        sheet.paste(frames[1], (x + 324, y + 15))
        index.append(dict(number=i + 1, **item, reversed=bool(reverse)))
        if i % 12 == 11 or i + 1 == len(items):
            sheet.save(out / f"blind-{i//12+1:02d}.jpg", quality=90)
    write(out / "index.json", index)
    print("Registered visual audit", len(picked), len(repeat), flush=True)


if __name__ == "__main__":
    main()
