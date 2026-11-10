import json
from pathlib import Path
import pytest
pytest.importorskip("torch")
pytest.importorskip("liger_kernel")
from training.extend_text_train import learning_rate as initial_lr
from training.extend_text_train_v2 import learning_rate


def test_larger_runtime_preserves_initial_lr_and_exposure():
    cfg=json.loads(Path("configs/t1/fineweb_extend_v2.json").read_text())
    old=json.loads(Path("configs/t1/fineweb_extend.json").read_text())
    for step in (0,100,127,128,255,511,512):
        assert abs(learning_rate(step,cfg)-initial_lr(step,old))<1e-12
    assert (1024+cfg["additional_updates"])*32*512==150994944
    assert 1024+512+(8192-512)//2==5376
    assert (1024+512)*32==1024*32+512*32
    assert abs(learning_rate(8192,cfg)-cfg["lr_floor"])<1e-12
