"""Portable per-output-channel 4-bit storage for a frozen G9 field."""

import json
import numpy as np


def pack(path, arrays, spec, parent_sha):
    entries = {}
    blob = {}
    for name, original in arrays.items():
        a = np.asarray(original, dtype=np.float32)
        if a.ndim == 2 and a.size >= 65536:
            scale = np.maximum(np.abs(a).max(0) / 7, 1e-12).astype(np.float32)
            codes = (
                (np.rint(a / scale).clip(-7, 7).astype(np.int8) + 7)
                .astype(np.uint8)
                .ravel()
            )
            if len(codes) % 2:
                codes = np.pad(codes, (0, 1))
            blob[name + ".q"] = codes[::2] | (codes[1::2] << 4)
            blob[name + ".scale"] = scale
            entries[name] = {"shape": list(a.shape), "format": "int4-per-output"}
        else:
            blob[name + ".f"] = a.astype(np.float16)
            entries[name] = {"shape": list(a.shape), "format": "float16"}
    blob["manifest"] = np.array(
        json.dumps(
            dict(version=1, spec=spec, parent_sha256=parent_sha, tensors=entries),
            sort_keys=True,
        )
    )
    np.savez_compressed(path, **blob)


def load(path):
    with np.load(path, allow_pickle=False) as db:
        meta = json.loads(str(db["manifest"]))
        if meta["version"] != 1:
            raise ValueError("Unsupported portable G9 format")
        p = {}
        for name, entry in meta["tensors"].items():
            shape = tuple(entry["shape"])
            if entry["format"] == "int4-per-output":
                packed = db[name + ".q"]
                codes = np.empty(2 * len(packed), np.uint8)
                codes[::2] = packed & 15
                codes[1::2] = packed >> 4
                p[name] = (
                    codes[: int(np.prod(shape))].reshape(shape).astype(np.float32) - 7
                ) * db[name + ".scale"]
            elif entry["format"] == "float16":
                p[name] = db[name + ".f"].astype(np.float32)
            else:
                raise ValueError("Unknown parameter format")
            if p[name].shape != shape:
                raise ValueError("Truncated portable parameter")
    return meta, p
