import struct
import pytest
pytest.importorskip("google.protobuf")
pytest.importorskip("PIL")
pytest.importorskip("torch")
from training.action_data_probe import crc32c,masked_crc,records,example_type


def test_standard_crc_and_corrupted_record_detection(tmp_path):
    assert crc32c(b"123456789")==0xE3069283
    payload=b"record data"; size=struct.pack("<Q",len(payload)); data=size+struct.pack("<I",masked_crc(size))+payload+struct.pack("<I",masked_crc(payload)); path=tmp_path/"data"
    path.write_bytes(data); assert list(records(path))==[payload]
    path.write_bytes(data[:-5]+bytes([data[-5]^1])+data[-4:])
    with pytest.raises(ValueError,match="payload CRC"): list(records(path))


def test_example_wire_descriptor_preserves_each_feature_kind():
    Example=example_type(); value=Example(); value.features.feature["words"].bytes_list.value.append(b"move"); value.features.feature["action"].float_list.value.extend([1.,2.]); value.features.feature["last"].int64_list.value.append(1); restored=Example.FromString(value.SerializeToString())
    assert list(restored.features.feature["words"].bytes_list.value)==[b"move"]
    assert list(restored.features.feature["action"].float_list.value)==[1.,2.]
    assert list(restored.features.feature["last"].int64_list.value)==[1]
