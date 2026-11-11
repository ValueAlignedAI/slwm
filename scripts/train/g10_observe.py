"""Collect finite remote evaluation panels, transfer real banks directly, and track owned work."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
from g7_inputs import FED, stream
from g8_file_transfer import file_data
from g9_cloud_archive import notebook
from g9_storage import write

ROOT = "/mnt/slwm-data/SLWM/g10"
NAMES = {
    "slwm-g10-context-a": 1002,
    "slwm-g10-context-b": 1001,
    "slwm-g10-ground-a": 1001,
    "slwm-g10-ground-b": 1002,
}


def send_bank(name, ctl):
    routing = ctl / "routing.json"
    if routing.exists() and name not in json.loads(routing.read_text())["bank_targets"]:
        return
    if (ctl / (name + "-bank.json")).exists():
        return
    source = ROOT + "/data/holo/holo-train.npz"
    check = subprocess.run(
        [*FED, "cat " + ROOT + "/data/holo/holo-train.json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check.returncode:
        return
    info = json.loads(check.stdout)
    digest = info["sha256"]
    stream(
        name,
        [*FED, "cat " + source],
        "set -eu; cd /content/g10-data; cat > .holo.partial; echo '"
        + digest
        + "  .holo.partial' | sha256sum -c -; mv .holo.partial holo-train.npz; touch holo-ready",
    )
    write(ctl / (name + "-bank.json"), info)


def collect(name, ctl):
    marker = ctl / (name + "-eval.json")
    old = json.loads(marker.read_text()) if marker.exists() else {"files": []}
    code = """import pathlib,tarfile,hashlib,json
r=pathlib.Path('/content/g10-eval/control')
files=sorted(f for f in r.glob('*.json') if f.name not in %r)
logs={}
for p in pathlib.Path('/content/g10-out').glob('*.log'):
 if p.stat().st_size:
  with p.open('rb') as inp:inp.seek(max(0,p.stat().st_size-280));logs[p.name]=inp.read().decode(errors='replace')
if files:
 path=pathlib.Path('/content/g10-scores-snapshot.tgz')
 with tarfile.open(path,'w:gz') as t:
  for f in files:t.add(f,arcname=f.name,recursive=False)
 print('G10_SCORES '+json.dumps(dict(files=[f.name for f in files],size=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),tail=logs)))
else:print('G10_SCORES '+json.dumps(dict(files=[],tail=logs)))""" % old["files"]
    output = notebook(name, code, 75)
    msg = json.loads(
        next(s[11:] for s in output.splitlines() if s.startswith("G10_SCORES "))
    )
    write(ctl / (name + "-health.json"), dict(tail=msg["tail"], checked=time.time()))
    if not msg["files"]:
        return
    if msg["size"] > 15 << 20:
        raise ValueError("Unexpected score bundle")
    data = file_data(name, "/content/g10-scores-snapshot.tgz", msg["size"])
    if hashlib.sha256(data).hexdigest() != msg["sha256"]:
        raise ValueError("Score transfer mismatch")
    decoded = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for item in tar.getmembers():
            if (
                not item.isfile()
                or item.name != Path(item.name).name
                or item.size > 2 << 20
            ):
                raise ValueError("Unknown score member")
            d = json.loads(tar.extractfile(item).read())
            name_key = d["tag"] + "-" + str(d["step"]) + ".json"
            routing = ctl / "routing.json"
            seeds = (
                json.loads(routing.read_text())["seeds"][name]
                if routing.exists()
                else [NAMES[name]]
            )
            if name_key != item.name or not any(
                d["tag"].endswith("-s" + str(s)) for s in seeds
            ):
                raise ValueError("Score identity mismatch")
            decoded[item.name] = d
    if set(decoded) != set(msg["files"]):
        raise ValueError("Incomplete published scores")
    script = (
        "import pathlib,json,sys,os;p=pathlib.Path('/mnt/slwm-data/SLWM/g10/eval/control');p.mkdir(parents=True,exist_ok=True);d=json.load(sys.stdin);"
        + "[(lambda dest,v:(dest.with_suffix('.incoming').write_text(json.dumps(v,sort_keys=True)),dest.with_suffix('.incoming').replace(dest)))(p/n,v) for n,v in d.items()]"
    )
    subprocess.run(
        [*FED, "python3 -c " + shlex.quote(script)],
        input=json.dumps(decoded).encode(),
        timeout=90,
        check=True,
    )
    write(
        marker,
        dict(files=sorted(set(old["files"]) | set(decoded)), checked=time.time()),
    )


def watch(name, ctl):
    until = json.loads((ctl / "bounds.json").read_text())["deadline"]
    while time.time() < until and not (ctl / "finalized").exists():
        budget = json.loads((ctl / "budget.json").read_text())
        if budget["phase"] == "complete" or (
            (ctl / (name + "-release")).exists() and name not in budget["active"]
        ):
            return
        try:
            send_bank(name, ctl)
            collect(name, ctl)
        except (
            OSError,
            ValueError,
            RuntimeError,
            StopIteration,
            subprocess.SubprocessError,
        ) as exc:
            write(
                ctl / (name + "-observe-retry.json"),
                dict(time=time.time(), error=str(exc)[:200]),
            )
        time.sleep(30)


def journal(ctl):
    home = Path.home() / ".local/state/slwm"
    while not (ctl / "finalized").exists():
        try:
            budget = json.loads((ctl / "budget.json").read_text())
            stages = {
                name: len(
                    json.loads((ctl / (name + "-runtime.json")).read_text())["stages"]
                )
                for name in NAMES
                if (ctl / (name + "-runtime.json")).exists()
            }
            with (home / "research-state.lock").open("a") as fd:
                fcntl.flock(fd, fcntl.LOCK_EX)
                if (ctl / "finalized").exists():
                    return
                f = home / "research-state.json"
                doc = json.loads(f.read_text())
                g = doc["studies"]["G10"]
                spent = budget["project_spend"]
                g.update(
                    phase="training_and_validation",
                    owned_session_names=budget["active"],
                    owned_credits_this_study=round(spent, 5),
                    cumulative_owned_credits=round(g["prior_project_units"] + spent, 5),
                    staged_native_generations=stages,
                    cloud_to_two_disk_receipts=len(
                        list((ctl / "receipts").glob("*.json"))
                    ),
                    supplemental_room_only_fits=6,
                )
                doc["updated_utc"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                write(f, doc)
                os.chmod(f, 0o600)
        except (OSError, ValueError):
            pass
        time.sleep(40)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("control", type=Path)
    a = p.parse_args()
    with ThreadPoolExecutor(3) as pool:
        jobs = [
            pool.submit(watch, n, a.control)
            for n in json.loads((a.control / "bounds.json").read_text())["sessions"]
        ]
        if a.control.name != "recovery":
            jobs.append(pool.submit(journal, a.control))
        for job in jobs:
            job.result()
