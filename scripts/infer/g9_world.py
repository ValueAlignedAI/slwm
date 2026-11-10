"""Run sustained decisions or condition the trained field on an aligned sensor file."""

import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_live import initial_memory, live
from models.g9_portable import load
from models.g9_shared import Spec, encode, decode, decision_scores
from scripts.train.g9_fit import numpy_weights

SOURCES = ("audio", "vision", "physics", "words")
SHAPES = {"a": 64, "v": 768, "n": 12, "w": 8}


def checkpoint(path):
    if path.is_file():
        meta, params = load(path)
        source = meta["parent_sha256"]
        receipt = path.parent / "portable.json"
        if receipt.is_file():
            known = [
                v["sha256"]
                for v in json.loads(receipt.read_text()).values()
                if v["parent_sha256"] == source
            ]
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if known and digest not in known:
                raise ValueError("Portable artifact checksum mismatch")
    else:
        meta = json.loads((path / "receipt.json").read_text())
        file = path / "weights.npz"
        with file.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != meta["inference_sha256"]:
                raise ValueError("Weight receipt mismatch")
        params = numpy_weights(file)
        source = meta["inference_sha256"]
    return Spec(**meta["spec"]), jax.tree.map(jnp.asarray, params), source


def adapted(path, parent):
    if path is None:
        return None
    record = json.loads((path.parent / "posttraining.json").read_text())
    name = path.stem.removesuffix("-head")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if (
        record["parent_sha256"] != parent
        or name not in record["arms"]
        or record["arms"][name]["sha256"] != digest
    ):
        raise ValueError("Decision head belongs to a different frozen backbone")
    return jax.tree.map(jnp.asarray, numpy_weights(path))


def condition(params, spec, file, sources, head=None):
    flags = np.array([k in sources for k in SOURCES])
    keep = jnp.asarray(flags[None])
    with np.load(file, allow_pickle=False) as db:
        if "u" not in db:
            raise ValueError("Input needs the stream of previously issued actions in u")
        u = np.asarray(db["u"])
        length = len(u)
        if (
            u.ndim != 1
            or not length
            or not np.issubdtype(u.dtype, np.integer)
            or (u < 0).any()
            or (u >= 7).any()
        ):
            raise ValueError("Invalid issued action stream")
        row = {"u": u.astype(np.int32)}
        for (key, width), wanted in zip(SHAPES.items(), flags):
            if wanted and key not in db:
                raise ValueError("Selected source missing: " + key)
            vals = (
                np.asarray(db[key])
                if wanted
                else np.zeros(
                    (length, width), dtype=np.float32 if key == "n" else np.uint8
                )
            )
            if vals.shape != (length, width):
                raise ValueError("Wrong clock or channel shape: " + key)
            if key != "n" and not np.issubdtype(vals.dtype, np.integer):
                raise ValueError(
                    "Quantized sensor and word inputs must use integer values"
                )
            if key == "w" and ((vals < 0).any() or (vals >= 48).any()):
                raise ValueError("Unknown word token")
            if key in ("a", "v") and ((vals < 0).any() or (vals > 255).any()):
                raise ValueError("Signal features outside quantized range")
            if not np.all(np.isfinite(vals)):
                raise ValueError("Nonfinite input")
            row[key] = vals.astype(np.float32 if key == "n" else np.uint8)
        plan = np.asarray(db["plan"]) if "plan" in db else np.zeros(length, np.int32)
        if (
            plan.shape != (length,)
            or not np.issubdtype(plan.dtype, np.integer)
            or (plan < 0).any()
            or (plan >= 7).any()
        ):
            raise ValueError("Invalid candidate action stream")
    mem = initial_memory(spec, 1)
    outputs = []

    @jax.jit
    def predict(p, x, query, cache, t):
        z, new = encode(p, spec, x, keep, memory=cache, offset=t)
        pred = decode(p, z, query)
        if head is not None:
            pred["policy"] = decision_scores(z, p["read"][:, -7:], head)
        return {"z": z, **pred}, new

    for start in range(0, length, 16):
        end = min(start + 16, length)
        x = {k: jnp.asarray(v[None, start:end]) for k, v in row.items()}
        result, mem = predict(
            params, x, jnp.asarray(plan[None, start:end]), mem, jnp.int32(start)
        )
        outputs.append(
            {k: np.asarray(v[0], dtype=np.float32) for k, v in result.items()}
        )
    return {k: np.concatenate([o[k] for o in outputs]) for k in outputs[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--mission", action="store_true")
    modes.add_argument("--input", type=Path)
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mission-seed", type=int, default=180013)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument(
        "--head",
        type=Path,
        help="Optional verified decision head for this exact parent",
    )
    a = ap.parse_args()
    spec, p, source = checkpoint(a.weights)
    head = adapted(a.head, source)
    if a.mission:
        if not 1 <= a.worlds <= 256:
            raise ValueError("World batch must be between 1 and 256")
        signal = live(p, spec, "carry", b=a.worlds, seed=a.mission_seed, head=head)
        doc = dict(
            parent_sha256=source,
            mission_seed=a.mission_seed,
            hz=10,
            steps=512,
            issued_actions=signal[:, :, 5].T.tolist(),
            goal_occupancy=signal[:, :, 0].mean(0).tolist(),
            subgoal_completion=signal[:, :, 0]
            .reshape(8, 64, -1)
            .any(1)
            .mean(0)
            .tolist(),
        )
        a.output.write_text(json.dumps(doc, indent=2) + "\n")
    else:
        pred = condition(p, spec, a.input, a.sources, head)
        np.savez_compressed(a.output, **pred, parent_sha256=np.array(source))


if __name__ == "__main__":
    main()
