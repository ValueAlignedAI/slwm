"""Run the public interface one timestamp at a time and check the batched replay against it."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main(release):
    cases = json.loads((release / "examples/streams.json").read_bytes())
    rows = []
    for case in cases:
        family = case["family"]
        start = time.monotonic()
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.infer.g12_live",
                str(release / "examples" / (family + "-stream.npz")),
                "--fold",
                case["fold"],
                "--core",
                "associative",
                "--package",
                str(release / "inference"),
                "--batch-size",
                "1",
            ],
            text=True,
            check=True,
            capture_output=True,
            timeout=180,
        )
        observed = [json.loads(line) for line in run.stdout.splitlines()]
        labels = json.loads(
            (release / "examples" / (family + "-observations.json")).read_bytes()
        )
        expected = [
            json.loads(s)
            for s in (release / "examples" / (family + "-decisions.jsonl"))
            .read_text()
            .splitlines()
        ]
        if [v["second"] for v in observed] != [v["second"] for v in expected] or len(
            labels
        ) != len(observed):
            raise ValueError("Clock mismatch")
        same = sum(a["route"] == b["route"] for a, b in zip(expected, observed))
        action = sum("actions" in a for a in observed)
        rows.append(
            dict(
                family=family,
                source_seconds=case["seconds"],
                cycles=len(observed),
                seconds_including_startup=time.monotonic() - start,
                same_routing_as_batch=same,
                action_commits=action,
                state_commits=sum("device_change" in a for a in observed),
                internal_only=sum(a["route"] == "internal_only" for a in observed),
                quiet_clocks=sum(r["quiet_next_eight_seconds"] for r in labels),
                action_on_quiet=sum(
                    "actions" in a and b["quiet_next_eight_seconds"]
                    for a, b in zip(observed, labels)
                ),
            )
        )
    (release / "metrics/single-clock.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(rows, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("release", type=Path)
    main(p.parse_args().release)
