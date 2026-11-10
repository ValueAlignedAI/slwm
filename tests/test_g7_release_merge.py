import importlib
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def release(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts/train"))
    return importlib.import_module("g7_integrate_release")


def github(monkeypatch,release,responses,*,merged=False,base="base",head="head"):
    calls=[]
    def request(args,**kwargs):
        calls.append(args)
        output,error=responses[len(calls)-1]
        return subprocess.CompletedProcess(args,bool(error),output,error)
    monkeypatch.setattr(release.subprocess,"run",request)
    monkeypatch.setattr(release,"gh",lambda *args:json.dumps({"merged":merged,"head":{"sha":head},"base":{"sha":base}}))
    monkeypatch.setattr(release.time,"sleep",lambda seconds:None)
    return calls


def test_retries_transient_error_at_the_same_reviewed_head(release,monkeypatch):
    calls=github(monkeypatch,release,[("","HTTP 502"),('{"merged":true}',"")])
    assert release.merge_checked(17,"head","base","Study")
    assert len(calls)==2 and calls[0]==calls[1]
    assert "sha=head" in calls[0] and "merge_method=merge" in calls[0]


def test_acknowledgement_lost_after_success_does_not_repeat_merge(release,monkeypatch):
    calls=github(monkeypatch,release,[("","connection reset")],merged=True)
    assert release.merge_checked(17,"head","base","Study")
    assert len(calls)==1


def test_new_base_returns_to_exact_tree_review(release,monkeypatch):
    calls=github(monkeypatch,release,[("","HTTP 502")],base="changed")
    assert not release.merge_checked(17,"head","base","Study")
    assert len(calls)==1


@pytest.mark.parametrize("diagnostic,head,error",[
    ("HTTP 403","head",RuntimeError),
    ("HTTP 502","changed",ValueError),
])
def test_respects_policy_and_reviewed_head_changes(release,monkeypatch,diagnostic,head,error):
    calls=github(monkeypatch,release,[("",diagnostic)],head=head)
    with pytest.raises(error):
        release.merge_checked(17,"head","base","Study")
    assert len(calls)==1
