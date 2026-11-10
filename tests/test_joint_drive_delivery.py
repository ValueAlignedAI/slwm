import hashlib
import json
import shutil

import pytest

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
