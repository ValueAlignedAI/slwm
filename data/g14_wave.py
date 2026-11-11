"""Decode the original mono recording on the GRID video clock, using the verified local corpus."""

import json
from pathlib import Path
import subprocess
import numpy as np

OLD = Path("/mnt/slwm-data/SLWM/data/grid_content_v1")


def pcm(row):
    path = OLD / "decoded" / f"s{row['person']:02d}" / (row["key"] + ".npz")
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            return z["pcm"].astype("f4") / 32768.0
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        row["raw"],
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    x = np.frombuffer(subprocess.check_output(args, timeout=50), "<f4").copy()
    meta = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,start_time",
                "-of",
                "json",
                row["raw"],
            ],
            timeout=30,
        )
    )
    start = {q["codec_type"]: float(q.get("start_time", 0)) for q in meta["streams"]}
    trim = round((start["video"] - start["audio"]) * 16000)
    return x[trim:] if trim >= 0 else np.pad(x, (-trim, 0))
