"""Record stage counts and a measured ETA without changing a running recipe or emitting notifications."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from scripts.train.g9_storage import write

ROOT = Path("/mnt/slwm-data/SLWM/g14")
WORK = ROOT / "work"


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def status():
    now = time.time()
    marker = WORK / "released-lock.json"
    if not marker.exists():
        return dict(phase="development", updated_utc=utc(now))
    start = marker.stat().st_mtime
    roles = {}
    for role, total, workers in (("held", 256, 11), ("fresh", 64, 3)):
        folder = WORK / "executed/final" / role
        times = sorted(f.stat().st_mtime for f in folder.glob("[0-9]*.json"))
        n = len(times)
        data = dict(completed=n, total=total, workers=workers)
        if times:
            data.update(
                latest_completion_utc=utc(times[-1]),
                seconds_since_completion=round(now - times[-1]),
            )
        if n >= workers * 2:
            # Smooth full worker waves; individual case timestamps are clustered when workers finish together.
            cutoff = max(start, now - 1800)
            recent = sum(t > cutoff for t in times)
            elapsed = max(1, min(now, times[-1]) - start)
            full = n / elapsed
            window = recent / max(1, now - cutoff)
            rate = (
                0.65 * window + 0.35 * full
                if now - cutoff >= 600 and recent >= workers
                else full
            )
            if rate > 0:
                data.update(
                    worlds_per_minute=round(60 * rate, 3),
                    estimated_finish_utc=(
                        utc(now + (total - n) / rate) if n < total else utc(times[-1])
                    ),
                )
        roles[role] = data
    complete = all(x["completed"] == x["total"] for x in roles.values())
    total = sum(x["completed"] for x in roles.values())
    return dict(
        phase="physical_execution" if not complete else "publication",
        updated_utc=utc(now),
        frozen_utc=utc(start),
        completed_worlds=total,
        total_worlds=320,
        primary_branches=total * 85,
        roles=roles,
        mission_worlds=len(list((WORK / "missions").glob("mission-*.json"))),
        root_free_gib=round(shutil.disk_usage("/").free / 2**30, 1),
        research_ssd_free_gib=round(shutil.disk_usage(ROOT).free / 2**30, 1),
    )


def main():
    while True:
        card = status()
        write(WORK / "progress.json", card)
        if (ROOT / "release/metrics/release.json").exists():
            return
        time.sleep(55)


if __name__ == "__main__":
    main()
