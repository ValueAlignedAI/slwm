import hashlib
import json
import shutil

import pytest

from scripts.train import deliver_joint_transfer as publisher
from scripts.train.deliver_joint_transfer import archive_files, verified_size


@pytest.mark.skipif(shutil.which("rclone") is None, reason="requires rclone")
def test_drive_publication_needs_matching_receipt_without_local_weights(tmp_path):
    digest = "ab" * 32
    record = {"schema": 1, "run": "g3-main-exports", "step": 8000,
              "files": {"gpt2-81.pt": {"sha256": digest, "size": 52}}}
    data = json.dumps(record).encode()
    folder = tmp_path / "runs/g3-main-exports"
    folder.mkdir(parents=True)
    marker = folder / f"000000008000-{hashlib.sha256(data).hexdigest()}.json"
    marker.write_bytes(data)
    files = archive_files(str(tmp_path), "main", 8000)
    assert verified_size(tmp_path / "missing", "gpt2-81", digest, files) == 52
    with pytest.raises(ValueError):
        verified_size(tmp_path / "missing", "gpt2-81", "ef" * 32, files)
    marker.write_bytes(data + b" ")
    with pytest.raises(ValueError, match="checksum"):
        archive_files(str(tmp_path), "main", 8000)


@pytest.mark.skipif(shutil.which("rclone") is None, reason="requires rclone")
def test_pending_primary_does_not_block_verified_companions_or_finish_early(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(publisher,"REL",tmp_path/"study")
    monkeypatch.setattr(publisher,"OPS",tmp_path/"delivery")
    real_run=publisher.subprocess.run
    monkeypatch.setattr(publisher.subprocess,"run",lambda cmd,*a,**k: None if cmd[0]=="rsync" else real_run(cmd,*a,**k))
    monkeypatch.setattr(publisher,"git",lambda *args:"0" if args[0]=="rev-list" else "head")
    commits=[]
    def record(paths,message):
        assert all(p.is_file() for p in paths)
        commits.append(message)
    monkeypatch.setattr(publisher,"commit_files",record)
    remote=tmp_path/"remote"
    digest="ab"*32
    def marker(group):
        folder=remote/"runs"/f"g3-{group}-exports"
        body=json.dumps({"schema":1,"run":f"g3-{group}-exports","step":8000,"files":{"model.pt":{"size":52,"sha256":digest}}}).encode()
        (folder/f"000000008000-{hashlib.sha256(body).hexdigest()}.json").write_bytes(body)
    for group,prefix in (("main","summary"),("depth6","depth"),("structured","structured")):
        d=publisher.REL if group=="main" else publisher.REL/group
        (d/"model").mkdir(parents=True)
        (d/f"{prefix}-8000-test.json").write_text('{"rows":{"model":{}}}')
        (d/"model/step-8000.json").write_text(json.dumps({"delta_sha256":digest,"parent_sha256":"parent","exposure_sha256":"data"}))
        for name in ("training_lock.json","confirmation_lock.json","config.json"): (d/name).write_text("{}")
        (remote/"runs"/f"g3-{group}-exports").mkdir(parents=True)
        if group=="main": (d/"summary-8000-test.md").write_text("results")
        else: marker(group)
    for role in ("screen","scale"):
        d=publisher.REL/"pusht"/role; d.mkdir(parents=True)
        for name in ("lock.json","results.json"): (d/name).write_text("{}")
    assert not publisher.collect(str(remote))
    assert publisher.read(publisher.OPS/"latest.json")["pending_drive_generations"]==["main/8000"]
    assert not (publisher.REL/"verified-8000.json").exists()
    assert len(commits)==4
    assert (publisher.REL/"depth6/verified-8000.json").is_file() and (publisher.REL/"structured/verified-8000.json").is_file()
    marker("main")
    assert publisher.collect(str(remote))
    assert publisher.read(publisher.OPS/"latest.json")["pending_drive_generations"]==[]


@pytest.mark.skipif(shutil.which("rclone") is None, reason="requires rclone")
def test_conflicting_receipts_are_an_integrity_error(tmp_path):
    d=tmp_path/"runs/g3-main-exports"; d.mkdir(parents=True)
    for suffix in ("a","b"): (d/f"000000008000-{suffix}.json").write_text("{}")
    with pytest.raises(ValueError,match="found 2") as caught:
        archive_files(str(tmp_path),"main",8000)
    assert not isinstance(caught.value,publisher.PendingArchive)
