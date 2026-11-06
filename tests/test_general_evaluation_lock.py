import pytest

from training import general_evaluation_lock as lock


def test_registered_evaluation_rejects_source_changes_and_keeps_original_snapshot(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=tmp_path/"metric.py";source.write_text("fixed prospective metric\n")
    root=tmp_path/"study";root.mkdir()
    monkeypatch.setattr(lock,"SOURCES",("metric.py",))
    first=lock.register({"root":str(root)})
    assert lock.register({"root":str(root)})==first
    source.write_text("changed after registration\n")
    with pytest.raises(AssertionError,match="changed after registration"):
        lock.register({"root":str(root)})
    assert (root/"evaluation_source_snapshot/metric.py").read_text()=="fixed prospective metric\n"
    assert not (root/"evaluation_lock.json").exists()
