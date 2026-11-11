"""Run one native validation or post-training stage after its registered parent is durable locally."""

import argparse
import time
from pathlib import Path
import jax
from evals.g10_control import evaluate
from scripts.train.g10_align_fit import real_bank, fit


def parents(runs, seed):
    return [
        runs / f"context-{c}-{mode}-s{seed}" / "step-05120"
        for mode in ("variable", "short", "long")
        for c in ("hybrid", "gpt2", "gated")
    ]


def wait_file(path):
    until = time.monotonic() + 8 * 3600
    while not path.is_file() and time.monotonic() < until:
        time.sleep(8)
    if not path.is_file():
        raise TimeoutError(str(path))


def main(op, runs, data, out, seed):
    if op == "context-eval":
        for p in parents(runs, seed):
            wait_file(p / "receipt.json")
            for stage in (p, p.parent / "step-02560"):
                evaluate(stage, out / "control")
    elif op in ("rooms", "mixed"):
        star = data / "starss-train.npz"
        holo = data / "holo-train.npz"
        if op == "mixed":
            wait_file(holo)
        bank, names = real_bank(star, holo, rooms_only=op == "rooms")
        for mode in (
            ("rooms",) if op == "rooms" else ("paired", "reconstruct", "shifted")
        ):
            for c in ("hybrid", "gpt2"):
                fit(runs, runs, bank, names, c, seed, mode)
    elif op == "align-eval":
        for mode in ("rooms", "paired", "reconstruct", "shifted"):
            for c in ("hybrid", "gpt2"):
                stage = runs / f"align-{c}-{mode}-s{seed}" / "step-01536"
                wait_file(stage / "receipt.json")
                evaluate(stage, out / "control")
    else:
        raise ValueError(op)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("op")
    p.add_argument("runs", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("seed", type=int)
    p.add_argument("--cache")
    a = p.parse_args()
    if a.cache:
        jax.config.update("jax_compilation_cache_dir", a.cache)
    main(a.op, a.runs, a.data, a.out, a.seed)
