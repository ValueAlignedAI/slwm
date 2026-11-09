import json
from subprocess import CompletedProcess

from scripts.train import g4_watch_research as watcher


def test_exited_supervisor_is_visible_even_with_an_old_evaluate_status(tmp_path,monkeypatch):
    folder=tmp_path/"depth6"; folder.mkdir()
    path=folder/"status.json"; path.write_text(json.dumps({"phase":"evaluate"}))
    monkeypatch.setattr(watcher.subprocess,"run",lambda *args,**kwargs:CompletedProcess([],0,"failed\n",""))
    assert watcher.companion_health(tmp_path,"depth6")=={"service":"failed","phase":"evaluate","needs_attention":True}
    path.write_text(json.dumps({"phase":"complete"}))
    assert not watcher.companion_health(tmp_path,"depth6")["needs_attention"]
