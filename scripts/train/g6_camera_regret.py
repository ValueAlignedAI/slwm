"""Post hoc camera-substitution interaction for the G6 frozen three-plan Push-T decision probe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("correct", "delayed", "erased", "wrong")
BLOCK_LENGTHS = (1, 8, 16, 32)
DRAWS = 10000


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bootstrap_rows(donor, length):
    # Adjacent elements of the donor cycle share a scene as source or recipient.
    n = len(donor)
    np.testing.assert_array_equal(np.sort(donor), np.arange(n))
    order = [0]
    for _ in range(n - 1):
        order.append(int(donor[order[-1]]))
    if len(set(order)) != n or donor[order[-1]] != 0 or n % length:
        raise ValueError("Expected the registered single-cycle donor map and whole circular blocks")
    rng = np.random.default_rng(27141 + length)
    starts = rng.integers(0, n, (DRAWS, n // length))
    offsets = (starts[:, :, None] + np.arange(length)) % n
    return np.asarray(order)[offsets.reshape(DRAWS, n)]


def interval(values, rows, tail=.025):
    means = values[rows].mean(axis=1)
    return np.quantile(means, [tail, 1 - tail]).tolist()


def read_arm(folder, tag, scene, scene_sha):
    receipt = json.loads((folder / "scene-controls.json").read_bytes())
    path = folder / "scene-rows.npz"
    if (receipt["run"] != tag or receipt["scenes"]["sha256"] != scene_sha
            or receipt["paired_rows_sha256"] != digest(path)):
        raise ValueError(f"Different evaluation inputs for {tag}")
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["scene"], scene)
        values = {c: np.array(arrays[f"video_{c}_regret_px"]) for c in CONDITIONS}
    for condition, array in values.items():
        if array.shape != scene.shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid scene regret array for {tag}/{condition}")
        np.testing.assert_allclose(array.mean(), receipt["conditions"]["video_" + condition]["regret_px"])
    return values, receipt


def analyze(terminal, scenes):
    scene_sha = digest(scenes / "scenes.npz")
    if json.loads((scenes / "card.json").read_bytes())["sha256"] != scene_sha:
        raise ValueError("Scene pack differs from its published card")
    with np.load(scenes / "scenes.npz", allow_pickle=False) as pack:
        scene, donor = pack["group"], pack["wrong"]
    np.testing.assert_array_equal(scene, np.arange(600000, 600512))
    draws = {length: bootstrap_rows(donor, length) for length in BLOCK_LENGTHS}
    result = {
        "role": "Post hoc publication assessment on existing unseen-scene outcomes",
        "scene_sha256": scene_sha,
        "scenes": len(scene), "candidate_plans": 3, "action_horizon": 16,
        "scoring": "Excess final distance to the goal in pixels over the best available plan",
        "input": "Video history and the prospective action plan with numerical state withheld",
        "bootstrap": {"replicates": DRAWS, "seed_rule": "27141 + block length",
                      "method": "paired circular blocks along the fixed wrong-scene donor cycle",
                      "reported_block_length": 8,
                      "scope": "conditional on these fitted weights, task, proposal bank and camera permutation"},
        "comparisons": {},
    }
    for scale in ("anchor", "larger"):
        for seed in (511, 512):
            key = f"v5e-{scale}-{seed}"
            h, h_card = read_arm(terminal / key / "scene-hybrid", f"g6-v5e-{scale}-hybrid-{seed}", scene, scene_sha)
            d, d_card = read_arm(terminal / key / "scene-gpt2", f"g6-v5e-{scale}-gpt2-{seed}", scene, scene_sha)
            cells = {}
            for condition in CONDITIONS:
                gain = d[condition] - h[condition]
                cells[condition] = {
                    "hybrid_regret_px": float(h[condition].mean()), "dense_regret_px": float(d[condition].mean()),
                    "dense_minus_hybrid_px": float(gain.mean()), "paired_ci95_px": interval(gain, draws[8]),
                }
            effects = {}
            for condition in ("wrong", "delayed", "erased"):
                effect = (h[condition] - h["correct"]) - (d[condition] - d["correct"])
                effects[condition + "_vs_correct"] = {
                    "excess_hybrid_penalty_px": float(effect.mean()), "paired_ci95_px": interval(effect, draws[8]),
                }
                if condition == "wrong":
                    effects["wrong_vs_correct"].update({
                        "ci95_by_block_length": {str(k): interval(effect, rows) for k, rows in draws.items()},
                        "four_cell_bonferroni_interval_px": interval(effect, draws[8], .025 / 4),
                    })
            result["comparisons"][key] = {
                "weights_sha256": {"hybrid": h_card["inference_sha256"], "dense": d_card["inference_sha256"]},
                "row_sha256": {"hybrid": h_card["paired_rows_sha256"], "dense": d_card["paired_rows_sha256"]},
                "conditions": cells, "interactions": effects,
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.terminal, args.scenes), sort_keys=True, indent=2, allow_nan=False))
