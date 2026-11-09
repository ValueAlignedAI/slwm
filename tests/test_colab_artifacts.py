"""Crash boundaries and restore selection using the real rclone local backend."""
import hashlib
import json
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


def test_independent_replica_can_restore_before_drive_recovers(tmp_path, monkeypatch):
    local, cloud = str(tmp_path/"hdd"), str(tmp_path/"drive")
    stage(tmp_path, 3, b"real optimizer")
    original = vault.put_verified
    def outage(source, dest, sha):
        if dest.startswith(cloud + "/"):
            raise subprocess.CalledProcessError(5, ["rclone"])
        original(source, dest, sha)
    with monkeypatch.context() as m:
        m.setattr(vault, "put_verified", outage)
        with pytest.raises(subprocess.CalledProcessError):
            vault.sync(tmp_path/"spool", cloud, local)
    assert vault.restore(local, "model", tmp_path/"offline")["step"] == 3
    vault.sync(tmp_path/"spool", cloud, local)
    assert vault.restore(cloud, "model", tmp_path/"online")["step"] == 3


def test_prepared_inventory_follows_original_robot_links_and_checks_their_targets(tmp_path):
    from scripts.train import colab_pack_joint_data as pack
    g2, cache = tmp_path/pack.G2, tmp_path/pack.CACHE
    g2.mkdir(parents=True)
    for name in ("normalizers.npz", "dataset_contract.json"):
        (g2/name).write_bytes(b"fixture")
    (g2/"robot_rows.json").write_text(json.dumps([{"id":"robot-1"}]))
    for family in ("coco","speech","ami","clotho","pamap2"):
        (cache/family).mkdir(parents=True)
        (cache/family/"rows.json").write_text("[]")
    original = cache/"droid_expanded"/"robot-1"
    alias = cache/"droid_g2"/"robot-1"
    original.mkdir(parents=True)
    alias.mkdir(parents=True)
    for name in ("trajectory.npz","rgb.npy"):
        (original/name).write_bytes(b"data")
        (alias/name).symlink_to(original/name)
    files = pack.inventory(tmp_path)
    assert files[str((alias/"rgb.npy").relative_to(tmp_path))] == 4
    (alias/"rgb.npy").unlink()
    other = tmp_path/"unrelated"
    other.write_bytes(b"data")
    (alias/"rgb.npy").symlink_to(other)
    with pytest.raises(ValueError, match="regular prepared file"):
        pack.inventory(tmp_path)
