import json
from pathlib import Path
import numpy as np
import pytest
from training.text_500m import learning_rate,window_index
from training.scale_text_train import UniqueWindows


def test_counter_budget_and_learning_rate_boundaries():
    cfg=json.loads(Path("configs/t1/fineweb_500m.json").read_text())
    assert cfg["target_tokens"]>=500000000 and cfg["target_tokens"]-500000000<32768
    assert cfg["parent_tokens"]+cfg["additional_updates"]*32768==cfg["target_tokens"]
    assert learning_rate(0,cfg)==pytest.approx(.00002)
    assert learning_rate(cfg["warmup_updates"]-1,cfg)==pytest.approx(.0001)
    assert learning_rate(cfg["additional_updates"]-1,cfg)==pytest.approx(.00001)
    assert window_index(cfg,256)==window_index(cfg,255)+64
    assert window_index(cfg,cfg["additional_updates"])*512==cfg["target_tokens"]


def test_logical_windows_continue_without_repeating_parent_or_slice(tmp_path):
    data=np.arange(512*400+1,dtype=np.uint16); path=tmp_path/"tokens.bin"; data.tofile(path)
    stream=UniqueWindows(path,512,81); cfg={"parent_tokens":64*512,"sequence_length":512,"global_batch":64}
    old=[int((stream.offset+stream.stride*i)%stream.blocks) for i in range(64)]
    new=[int((stream.offset+stream.stride*i)%stream.blocks) for i in range(window_index(cfg,0),window_index(cfg,2))]
    assert len(set(old+new))==192
    first,_=stream.batch(window_index(cfg,0),64,"cpu"); second,_=stream.batch(window_index(cfg,1),64,"cpu")
    assert first.shape==second.shape==(64,512)
    with pytest.raises(ValueError,match="repeat"): stream.batch(stream.blocks-32,64,"cpu")
