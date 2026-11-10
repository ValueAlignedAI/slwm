import hashlib
import json

import pytest

from scripts.train.verify_g3_milestone_mirrors import milestone, verify_group


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_milestone_binds_all_heads_parent_and_test_lock(tmp_path):
    root=tmp_path/"study"; conf=root/"config.json"; name="signal-dense-81"
    parents={"slwm-81":["parent","ef"*32]}
    dump(conf,{"profiles":{"signal-dense":{"core":"slwm"}},"seeds":[81],"parents":parents})
    train_sha=dump(root/"training_lock.json",{"input":"fixed"})
    report=root/"summary-8000-test.json"; dump(report,{"rows":{name:{}}})
    out=root/name; out.mkdir(); (out/"step-8000.pt").write_bytes(b"weights")
    delta=hashlib.sha256(b"weights").hexdigest()
    dump(out/"step-8000.json",{"profile":"signal-dense","step":8000,"seed":81,"delta_sha256":delta,
                               "lock_sha256":train_sha,"parent_sha256":"ef"*32,"exposure_sha256":"ab"*32})
    code=tmp_path/"evaluation.py";code.write_text("frozen")
    lock={"endpoints":{name:delta},"parents":parents,"training_lock_sha256":train_sha,
          "evaluation":{str(code):hashlib.sha256(code.read_bytes()).hexdigest()}}
    dump(root/"confirmation_lock.json",lock)
    files,details=milestone(root,"",conf,report,8000,True)
    assert details["fitted_32shot_heads"]==3
    assert all(name+"-"+task+"-32shot-head.npz" in files for task in ("fsdd","fashion","banking"))
    code.write_text("changed")
    with pytest.raises(ValueError,match="Confirmation"):
        milestone(root,"",conf,report,8000,True)


def test_exact_manifest_refuses_to_omit_expected_files(tmp_path,monkeypatch):
    name="g3-main-exports"; step=8000
    body=json.dumps({"schema":1,"run":name,"step":step,"files":{"model.pt":{}}}).encode()
    folder=tmp_path/"receipts"/name;folder.mkdir(parents=True)
    (folder/f"{step:012d}-{hashlib.sha256(body).hexdigest()}.json").write_bytes(body)
    monkeypatch.setattr("scripts.train.verify_g3_milestone_mirrors.subprocess.check_output",lambda *args,**kwargs:body)
    with pytest.raises(ValueError,match="inventory"):
        verify_group(name,step,{"model.pt":"unused","required-head.npz":"unused"},tmp_path,tmp_path,"drive",{},False,True)
