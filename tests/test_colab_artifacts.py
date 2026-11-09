"""Crash boundaries and restore selection using the real rclone local backend."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.train import colab_artifacts as vault


pytestmark = pytest.mark.skipif(shutil.which("rclone") is None, reason="requires rclone")


def stage(root, step, value):
    source = root / "resume.pt"
    temp = root / "resume.tmp"
    temp.write_bytes(value)
    temp.replace(source)
    vault.capture(root / "spool", "model", step, {"resume.pt": source})
    return source


def test_async_capture_pins_atomic_source_and_restore_refuses_overwrite(tmp_path):
    remote = str(tmp_path / "remote")
    source = stage(tmp_path, 1, b"first")
    assert source.samefile(tmp_path / "spool/pending/model/000000000001/resume.pt")
    stage(tmp_path, 2, b"second")
    vault.sync(tmp_path / "spool", remote)
    assert (Path(remote) / "objects" / hashlib.sha256(b"first").hexdigest()).read_bytes() == b"first"
    out = tmp_path / "fresh"
    assert vault.restore(remote, "model", out)["step"] == 2
    assert (out / "resume.pt").read_bytes() == b"second"
    with pytest.raises(FileExistsError):
        vault.restore(remote, "model", out)


def test_interrupted_transfer_never_commits_partial_state_then_retries(tmp_path, monkeypatch):
    remote = str(tmp_path / "remote")
    stage(tmp_path, 1, b"first")
    vault.sync(tmp_path / "spool", remote)
    stage(tmp_path, 2, b"second")
    real_put = vault.put_verified

    def disconnect(source, dest, digest):
        if dest.endswith(".json"):
            raise subprocess.CalledProcessError(5, ["rclone"])
        return real_put(source, dest, digest)

    with monkeypatch.context() as m:
        m.setattr(vault, "put_verified", disconnect)
        with pytest.raises(subprocess.CalledProcessError):
            vault.sync(tmp_path / "spool", remote)
    assert vault.restore(remote, "model", tmp_path / "after-interruption")["step"] == 1
    assert (tmp_path / "spool/pending/model/000000000002").exists()
    vault.sync(tmp_path / "spool", remote)
    assert vault.restore(remote, "model", tmp_path / "after-retry")["step"] == 2


@pytest.mark.parametrize("damage", ["corrupt", "missing", "bad-manifest"])
def test_restore_falls_back_to_previous_complete_generation(tmp_path, damage):
    remote = tmp_path / "remote"
    for step in (1, 2):
        stage(tmp_path, step, str(step).encode())
    vault.sync(tmp_path / "spool", str(remote))
    blob = remote / "objects" / hashlib.sha256(b"2").hexdigest()
    if damage == "missing":
        blob.unlink()
    elif damage == "corrupt":
        blob.write_bytes(b"x")
    else:
        next((remote / "runs/model").glob("000000000002-*")).write_text("{}")
    out = tmp_path / "fresh"
    assert vault.restore(str(remote), "model", out)["step"] == 1
    assert (out / "resume.pt").read_bytes() == b"1"


def test_conflicting_step_and_path_traversal_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        vault.safe_name("../outside")
    remote = tmp_path / "remote"
    stage(tmp_path, 1, b"first")
    vault.sync(tmp_path / "spool", str(remote))
    folder = remote / "runs/model"
    marker = next(folder.iterdir())
    body = marker.read_bytes() + b" "
    (folder / ("000000000001-" + hashlib.sha256(body).hexdigest() + ".json")).write_bytes(body)
    with pytest.raises(vault.IntegrityError, match="Conflicting manifests"):
        vault.restore(str(remote), "model", tmp_path / "out")
    assert not (tmp_path / "out").exists()
