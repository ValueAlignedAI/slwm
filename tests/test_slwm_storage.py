import json
import os
import subprocess
import sys
import time

import pytest

from scripts.train import relocate_slwm_tree as mover
from scripts.train import slwm_storage_guard as guard
from scripts.train import slwm_storage_mounts as mounts


def test_missing_volume_keeps_original(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pt").write_bytes(b"state")
    monkeypatch.setattr(mover, "filesystem", lambda _: "root")
    with pytest.raises(OSError, match="wrong UUID"):
        mover.migrate(source, tmp_path/"unmounted"/"target", "disk")
    assert (source/"model.pt").read_bytes() == b"state"
    assert not (tmp_path/"unmounted").exists()


def test_storage_mount_config_rejects_unrelated_targets(tmp_path):
    config = tmp_path / "mounts.json"
    config.write_text(json.dumps({"volumes": {"/etc": "00000000-0000-4000-8000-000000000001"}, "binds": {}}))
    with pytest.raises(ValueError):
        mounts.load(config)


def test_live_log_and_lock_keep_their_open_file_descriptions(tmp_path):
    tree = tmp_path/"run"
    tree.mkdir()
    original = tree/"study.log"
    anchor = tmp_path/"state"
    with original.open("w") as out:
        before = os.fstat(out.fileno())
        inodes = guard.anchor_open_files(tree,anchor,["study.log"])
        assert (before.st_dev,before.st_ino) in inodes
        out.write("still running\n")
    assert original.is_symlink() and original.samefile(anchor/"study.log")
    assert original.read_text() == "still running\n"


@pytest.mark.skipif(not hasattr(os,"pidfd_open"), reason="Linux process control")
def test_quiesce_and_resume_only_owned_producer(tmp_path):
    repo = tmp_path/"repo"
    (repo/"training").mkdir(parents=True)
    (repo/"training"/"disk_probe.py").write_text("import time\ntime.sleep(30)\n")
    worker = subprocess.Popen([sys.executable,"-m","training.disk_probe"],cwd=repo)
    unrelated = subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],cwd=repo)
    held = guard.Held(tmp_path/"held.json")
    try:
        time.sleep(.2)
        held.pause({repo})
        assert guard.identity(worker.pid)[1] == "T"
        assert guard.identity(unrelated.pid)[1] != "T"
        guard.Held(tmp_path/"held.json").resume()
        time.sleep(.1)
        assert guard.identity(worker.pid)[1] != "T"
    finally:
        for proc in (worker, unrelated):
            proc.kill()
            proc.wait()
