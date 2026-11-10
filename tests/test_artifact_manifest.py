import json
import pytest
from training.artifact_manifest import create_manifest,verify_manifest


def test_manifest_detects_changed_bytes_and_excludes_itself(tmp_path):
    folder=tmp_path/"experiment"; folder.mkdir(); (folder/"model.pt").write_bytes(b"model"); (folder/"report.json").write_text("{}")
    path=folder/"manifest.json"; manifest=create_manifest(tmp_path,["experiment"],path)
    assert manifest["file_count"]==2 and manifest["bytes"]==7
    assert verify_manifest(tmp_path,json.loads(path.read_text()))["all_recorded_files_match"]
    assert create_manifest(tmp_path,["experiment"],path)["file_count"]==2
    (folder/"model.pt").write_bytes(b"other")
    with pytest.raises(ValueError,match="bytes differ"): verify_manifest(tmp_path,manifest)


def test_manifest_verification_rejects_parent_path_escape(tmp_path):
    with pytest.raises(ValueError,match="escapes"): verify_manifest(tmp_path,{"files":{"../outside":{}}})
