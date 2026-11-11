"""Apply the recorded tolerances to the independent audit of training fork labels."""

import json
from pathlib import Path
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    file = ROOT / "work/training-label-audit.json"
    d = json.loads(file.read_bytes())
    levels = {
        "one_step_effect_difference_mm": 0.25,
        "long_return_difference_mm": 1.0,
        "changed_best_action_regret_mm": 0.5,
    }
    decisions = {key: d[key]["mean"] <= value for key, value in levels.items()}
    write(
        ROOT / "work/training-fidelity-decision.json",
        dict(
            usable=all(decisions.values()),
            criteria=levels,
            passed=decisions,
            examined_sha=sha(file),
        ),
    )
    print("Measured candidate label fidelity", decisions, flush=True)


if __name__ == "__main__":
    main()
