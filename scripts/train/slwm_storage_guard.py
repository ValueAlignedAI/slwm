"""Pause only identified SLWM producers while storage is unavailable or during a cutover."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return fields[19], fields[0]


def producers(repos):
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid != os.getuid() or (proc / "cwd").resolve(strict=True) not in repos:
                continue
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            module = args[args.index("-m")+1] if "-m" in args else ""
            shell_runner = any(Path(a).name.startswith("run_joint_") and a.endswith(".sh") for a in args)
            if module.startswith("training.") or shell_runner:
                yield int(proc.name)
        except (OSError, ValueError, UnicodeError, IndexError):
            continue


class Held:
    def __init__(self, state):
        self.path = state
        state.parent.mkdir(parents=True, exist_ok=True)
        self.rows = json.loads(state.read_text()) if state.exists() else {}

    def save(self):
        temp = self.path.with_suffix(".new")
        temp.write_text(json.dumps(self.rows) + "\n")
        temp.replace(self.path)

    def pause(self, repos):
        # Stop supervisors and workers, then catch children created during the scan.
        for _ in range(3):
            changed = False
            for pid in producers(repos):
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    start, state = identity(pid)
                    if state in ("T", "t", "Z") or self.rows.get(str(pid)) == start:
                        continue
                    self.rows[str(pid)] = start
                    self.save()
                    signal.pidfd_send_signal(fd, signal.SIGSTOP)
                    changed = True
                except (FileNotFoundError, ProcessLookupError):
                    continue
                finally:
                    os.close(fd)
            if not changed:
                break
            time.sleep(.15)

    def resume(self):
        for key, start in list(self.rows.items()):
            try:
                fd = os.pidfd_open(int(key))
            except ProcessLookupError:
                fd = None
            try:
                if fd is not None and identity(key)[0] == start:
                    signal.pidfd_send_signal(fd, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError):
                pass
            finally:
                if fd is not None:
                    os.close(fd)
            del self.rows[key]
        self.save()


def space_ok(paths, volumes, minimum):
    for path, uuid in volumes:
        result = subprocess.run(["findmnt", "-nro", "UUID", "-T", path], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != uuid:
            return False, f"required filesystem missing at {path}"
    for path in paths:
        free = shutil.disk_usage(path).free / 1024**3
        if free < minimum:
            return False, f"{path} has {free:.1f} GiB free (threshold {minimum:g})"
    return True, "storage available"


def anchor_open_files(tree, directory, names):
    """Keep long-lived file descriptions and flock ownership across a volume move."""
    if not names:
        return set()
    if tree is None or directory is None:
        raise ValueError("Anchored log/lock files need both a tree and an anchor directory")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    inodes = set()
    for name in names:
        if Path(name).name != name or Path(name).suffix not in (".log", ".lock"):
            raise ValueError("Only explicitly named log or lock files may be anchored")
        original, anchor = tree/name, directory/name
        current = original.stat()
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f"Not a regular file: {original}")
        if not anchor.exists():
            os.link(original,anchor)
        if not original.samefile(anchor):
            raise ValueError(f"Unexpected existing anchor for {original}")
        if not original.is_symlink():
            temp = original.with_name("." + name + ".slwm-link")
            temp.symlink_to(anchor.absolute())
            temp.replace(original)
        if original.resolve() != anchor.resolve():
            raise ValueError(f"Anchor did not resolve correctly: {original}")
        inodes.add((current.st_dev, current.st_ino))
    return inodes


def writable_handles(rows, tree, anchored=frozenset()):
    """A paused optimizer must not keep an open output file on the original filesystem."""
    if tree is None:
        return []
    busy = []
    for pid in rows:
        for fd in Path(f"/proc/{pid}/fd").glob("*"):
            try:
                path = Path(os.readlink(fd))
                if not path.is_absolute() or not path.is_relative_to(tree):
                    continue
                description = fd.stat()
                if (description.st_dev, description.st_ino) in anchored:
                    continue
                info = Path(f"/proc/{pid}/fdinfo/{fd.name}").read_text().splitlines()
                flags = int(next(line.split()[1] for line in info if line.startswith("flags:")), 8)
                if flags & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                    busy.append(str(path))
            except FileNotFoundError:
                continue
    return busy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["watch", "quiesce"])
    p.add_argument("--repo", action="append", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--path", action="append", type=Path, default=[])
    p.add_argument("--volume", nargs=2, action="append", default=[], metavar=("PATH","UUID"))
    p.add_argument("--tree", type=Path, help="Quiesce between writes to this tree")
    p.add_argument("--anchor-dir", type=Path)
    p.add_argument("--anchor", action="append", default=[])
    p.add_argument("--low-gib", type=float, default=16)
    p.add_argument("--resume-gib", type=float, default=24)
    p.add_argument("--exec", nargs=argparse.REMAINDER, dest="command")
    args = p.parse_args()
    repos = {x.resolve() for x in args.repo}
    held = Held(args.state)
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "quiesce":
            if not args.command:
                p.error("quiesce requires --exec COMMAND")
            def terminate(*_):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, terminate)
            try:
                anchored = anchor_open_files(args.tree,args.anchor_dir,args.anchor)
                deadline = time.monotonic()+45
                while True:
                    held.pause(repos)
                    waiting = [pid for pid in held.rows if Path(f"/proc/{pid}").exists() and identity(pid)[1] not in ("T","t","Z")]
                    if not waiting and not writable_handles(held.rows, args.tree, anchored):
                        break
                    held.resume()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("No quiet checkpoint boundary; producers resumed")
                    time.sleep(.75)
                print(f"quiesced {len(held.rows)} SLWM processes", flush=True)
                subprocess.run(args.command, check=True)
            finally:
                held.resume()
                print("resumed SLWM producers", flush=True)
            return
        if not args.path or args.resume_gib <= args.low_gib:
            p.error("watch needs a filesystem path and a higher resume threshold")
        previous = None
        while True:
            ok, reason = space_ok(args.path, args.volume, args.resume_gib if held.rows else args.low_gib)
            if ok and held.rows:
                held.resume()
            elif not ok:
                held.pause(repos)
            if ok != previous:
                print(f"{'running' if ok else 'holding'}: {reason}; held={len(held.rows)}", flush=True)
            previous = ok
            time.sleep(3)


if __name__ == "__main__":
    main()
