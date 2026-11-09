import json
from subprocess import CompletedProcess

from scripts.train import g4_watch_research as watcher


def test_exited_supervisor_is_visible_even_with_an_old_evaluate_status(tmp_path,monkeypatch):
    folder=tmp_path/"depth6"; folder.mkdir()
    path=folder/"status.json"; path.write_text(json.dumps({"phase":"evaluate"}))
    monkeypatch.setattr(watcher.subprocess,"run",lambda *args,**kwargs:CompletedProcess([],0,"failed\n",""))
    assert watcher.companion_health(tmp_path,"depth6")=={"service":"failed","phase":"evaluate","needs_attention":True,
                                                         "matched_endpoints":{},"unequal_endpoints":[]}
    path.write_text(json.dumps({"phase":"complete"}))
    assert not watcher.companion_health(tmp_path,"depth6")["needs_attention"]


def test_comparison_hashes_checked_as_soon_as_each_endpoint_exists(tmp_path,monkeypatch):
    monkeypatch.setattr(watcher.subprocess,"run",lambda *args,**kwargs:CompletedProcess([],0,"active\n",""))
    for name in ("signal-dense-81","structured/signal-dense-81"):
        p=tmp_path/name; p.mkdir(parents=True); (p/"step-100.json").write_text(json.dumps({"exposure_sha256":"ab"*32}))
    assert watcher.companion_health(tmp_path,"structured")["matched_endpoints"]=={"100":1}
    (tmp_path/"structured/signal-dense-81/step-100.json").write_text(json.dumps({"exposure_sha256":"cd"*32}))
    health=watcher.companion_health(tmp_path,"structured")
    assert health["needs_attention"] and health["unequal_endpoints"]==["signal-dense-81/step-100"]
