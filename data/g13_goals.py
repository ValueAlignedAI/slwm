"""Source/destination roles for completed natural language at the signal edge."""

import re
import numpy as np

COLORS = ("red", "blue", "green", "yellow")
DIRECTIONS = ("left", "right", "up", "down")


def semantics(goal):
    tokens = re.findall(r"[a-z]+", goal.lower())
    colours = [COLORS.index(t) for t in tokens if t in COLORS]
    ordered = list(dict.fromkeys(colours))
    source = ordered[0] if ordered else -1
    destination = ordered[1] if len(ordered) > 1 else -1
    mentions = [t in tokens for t in COLORS]
    dirs = []
    for j, words in enumerate(
        (
            {"left", "leftward", "leftwards"},
            {"right", "rightward", "rightwards"},
            {"up", "upward", "upwards", "upper", "top"},
            {"down", "downward", "downwards", "bottom", "lower"},
        )
    ):
        if words.intersection(tokens):
            dirs.append(j)
    role = np.zeros((4, 4), np.float32)
    for c in range(4):
        role[c] = [
            c == source,
            c == destination,
            mentions[c],
            float("arm" in tokens or "effector" in tokens),
        ]
    return source, destination, role, (dirs[0] if len(dirs) == 1 else -1)
