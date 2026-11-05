"""Continuation must not reuse the original training/test windows."""
import json
from pathlib import Path
import numpy as np
from training.scale_text_train import UniqueWindows
from training.extend_text_train import learning_rate


def test_extension_consumes_disjoint_windows(tmp_path):
    path=tmp_path/"tokens.bin"; np.arange(60000,dtype="<u2").tofile(path)
    stream=UniqueWindows(path,32,81)
    initial,_=stream.batch(0,128,"cpu"); extended,_=stream.batch(128,128,"cpu")
    assert set(initial[:,0].tolist()).isdisjoint(extended[:,0].tolist())
    repeated,_=UniqueWindows(path,32,81).batch(128,128,"cpu")
    assert np.array_equal(extended.numpy(),repeated.numpy())


def test_registered_exposure_and_restart():
    cfg=json.loads(Path("configs/t1/fineweb_extend.json").read_text())
    assert (1024+cfg["additional_updates"])*32*512==83886080
    assert cfg["test_window_start"]>=256
    assert 6e-5<learning_rate(0,cfg)<cfg["lr"]
    assert learning_rate(cfg["warmup"]-1,cfg)==cfg["lr"]
    assert abs(learning_rate(cfg["additional_updates"],cfg)-cfg["lr_floor"])<1e-12
