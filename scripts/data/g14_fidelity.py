"""Apply the recorded tolerances to the independent audit of training fork labels."""

import json
from pathlib import Path
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main(separate=False):
    file = ROOT / "work/training-label-audit.json"
    d = json.loads(file.read_bytes())
    measured = ROOT / "work/isolated-label-audit.json"
    if separate:
        isolated = json.loads(measured.read_bytes())
        for k in ("long_return_difference_mm", "changed_best_action_regret_mm"):
            d[k] = isolated[k]
    levels = {
        "one_step_effect_difference_mm": 0.25,
        "long_return_difference_mm": 1.0,
        "changed_best_action_regret_mm": 0.5,
    }
    decisions = {key: d[key]["mean"] <= value for key, value in levels.items()}
    write(
        ROOT
        / (
            "work/final-fidelity-decision.json"
            if separate
            else "work/training-fidelity-decision.json"
        ),
        dict(
            usable=all(decisions.values()),
            criteria=levels,
            passed=decisions,
            examined_sha=sha(file),
            isolated_label_sha=sha(measured) if separate else None,
        ),
    )
    print("Measured candidate label fidelity", decisions, flush=True)


if __name__ == "__main__":
    import sys

    main("--isolated" in sys.argv)
