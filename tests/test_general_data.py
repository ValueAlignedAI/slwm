"""Physical contracts and corruption checks for heterogeneous source preparation."""
import struct
import numpy as np
import pytest
from training.general_data import SIGNALS,signal_array,pose_delta,native_crc,checked_records
from training.action_data_probe import crc32c


def test_cross_source_signal_units_and_missing_channels():
    raw=np.zeros((3,24)); raw[:,8:11]=180; raw[:,17:20]=90; raw[:,23]=4
    values,labels,times=signal_array(raw,"mhealth")
    assert values.shape==(3,33) and np.all(labels==4)
    np.testing.assert_allclose(values[:,SIGNALS.index("ankle.gyro.x")],np.pi)
    np.testing.assert_allclose(values[:,SIGNALS.index("wrist.gyro.x")],np.pi/2)
    assert np.isnan(values[:,SIGNALS.index("chest.gyro.x")]).all()
    assert np.isnan(values[:,SIGNALS.index("heart_rate")]).all()
    np.testing.assert_allclose(times,[0,.02,.04])
    pamap=np.zeros((2,54)); pamap[:,0]=[0,.01]; pamap[:,16:20]=999
    converted,_,_=signal_array(pamap,"pamap2")
    assert np.nanmax(converted)==0  # Invalid quaternion fields never enter input.


def test_action_rotation_wrap_is_a_small_physical_motion():
    observed=np.array([[0,0,0,0,0,np.pi-.01]])
    commanded=np.array([[0,1,0,0,0,-np.pi+.01]])
    delta=pose_delta(observed,commanded)
    assert delta.shape==(1,6)
    assert np.linalg.norm(delta[0,3:])==pytest.approx(.02,abs=1e-6)
    np.testing.assert_allclose(delta[0,:3],[np.sin(.01),-np.cos(.01),0],atol=1e-6)


def test_tfrecord_checksum_and_truncation(tmp_path):
    crc=native_crc(); payload=bytes(range(256))*9
    assert crc(payload)==crc32c(payload)
    def masked(x):
        value=crc(x); return (((value>>15)|(value<<17))+0xa282ead8)&0xffffffff
    length=struct.pack("<Q",len(payload)); data=length+struct.pack("<I",masked(length))+payload+struct.pack("<I",masked(payload))
    path=tmp_path/"record"; path.write_bytes(data)
    assert list(checked_records(path,crc))==[payload]
    path.write_bytes(data[:-2])
    with pytest.raises(ValueError,match="payload CRC"): list(checked_records(path,crc))
    corrupt=bytearray(data); corrupt[20]^=1; path.write_bytes(corrupt)
    with pytest.raises(ValueError,match="payload CRC"): list(checked_records(path,crc))
