"""Bounded official DROID source/schema probe; no TensorFlow, GPU or model fitting."""
import base64
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import struct

import numpy as np
from google.protobuf import descriptor_pb2,descriptor_pool,message_factory
from PIL import Image

from training.research_sources import fetch,file_sha
from training.t2_real_av import write_json

ROOT=Path("artifacts/robotics/droid_debug_probe_v1")
OUTPUT=Path("experiments/policy/droid-source-probe-20261105")
BASE="https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/"
OBJECTS=(
    ("dataset_info.json","1710527613369658",760,"JOcTGdwePw+iymeLhrqavg=="),
    ("features.json","1710517501027287",18665,"AUayWu2IHU/GMy61byaQpg=="),
    ("r2d2_faceblur-train.tfrecord-00000-of-00031","1710528095074666",27488816,"IQoCwa04u58UpSOzKsS/6g=="),
)


def example_type():
    """Wire-compatible Example descriptor; definitions verified against TF2.20."""
    schema=descriptor_pb2.FileDescriptorProto(name="slwm_probe_example.proto",package="slwm_probe",syntax="proto3")
    for name,kind in (("BytesList",12),("FloatList",2),("Int64List",3)):
        msg=schema.message_type.add(name=name); msg.field.add(name="value",number=1,label=3,type=kind)
    feature=schema.message_type.add(name="Feature"); feature.oneof_decl.add(name="kind")
    for i,name in enumerate(("BytesList","FloatList","Int64List"),1):
        field=feature.field.add(name=("bytes_list","float_list","int64_list")[i-1],number=i,label=1,type=11,type_name=".slwm_probe."+name); field.oneof_index=0
    features=schema.message_type.add(name="Features"); entry=features.nested_type.add(name="FeatureEntry"); entry.options.map_entry=True
    entry.field.add(name="key",number=1,label=1,type=9); entry.field.add(name="value",number=2,label=1,type=11,type_name=".slwm_probe.Feature")
    features.field.add(name="feature",number=1,label=3,type=11,type_name=".slwm_probe.Features.FeatureEntry")
    msg=schema.message_type.add(name="Example"); msg.field.add(name="features",number=1,label=1,type=11,type_name=".slwm_probe.Features")
    pool=descriptor_pool.DescriptorPool(); descriptor=pool.Add(schema)
    return message_factory.GetMessageClass(descriptor.message_types_by_name["Example"])


def crc32c(data):
    table=[]
    for value in range(256):
        for _ in range(8): value=(value>>1)^(0x82F63B78 if value&1 else 0)
        table.append(value)
    crc=0xFFFFFFFF
    for value in data: crc=(crc>>8)^table[(crc^value)&255]
    return crc^0xFFFFFFFF


def masked_crc(data):
    value=crc32c(data)
    return (((value>>15)|(value<<17))+0xA282EAD8)&0xFFFFFFFF


def records(path):
    with Path(path).open("rb") as stream:
        while True:
            header=stream.read(12)
            if not header: return
            if len(header)!=12: raise ValueError("Truncated TFRecord header")
            length,checksum=struct.unpack("<QI",header)
            if checksum!=masked_crc(header[:8]): raise ValueError("TFRecord length CRC failed")
            if length>64*1024**2: raise ValueError("Probe record exceeds64MiB bound")
            value=stream.read(length); tail=stream.read(4)
            if len(value)!=length or len(tail)!=4: raise ValueError("Truncated TFRecord payload")
            if struct.unpack("<I",tail)[0]!=masked_crc(value): raise ValueError("TFRecord payload CRC failed")
            yield value


def array(feature,columns,steps):
    kind=feature.WhichOneof("kind")
    if kind=="float_list": values=np.asarray(feature.float_list.value,np.float64)
    elif kind=="int64_list": values=np.asarray(feature.int64_list.value,np.int64)
    else:
        payload=b"".join(feature.bytes_list.value)
        if len(payload)!=steps*columns*8: raise ValueError("Unexpected float64 tensor serialization")
        values=np.frombuffer(payload,dtype="<f8")
    return values.reshape(steps,columns)


def main():
    ROOT.mkdir(parents=True,exist_ok=True); OUTPUT.mkdir(parents=True,exist_ok=True)
    assert crc32c(b"123456789")==0xE3069283
    write_json(OUTPUT/"registered.json",{"scope":"Only first official debug shard, expected2 episodes; parser/schema check, no model fitting or primary test data", "source_sha256":file_sha(Path(__file__)),"max_download_bytes":sum(r[2] for r in OBJECTS),"tensorflow_used":False,"gpu_used":False,"official_guide":"https://droid-dataset.github.io/droid/the-droid-dataset","wire_spec":"https://www.tensorflow.org/tutorials/load_data/tfrecord","protobuf_version":importlib.metadata.version("protobuf")})
    sources=[]
    for name,generation,size,md5 in OBJECTS:
        path=ROOT/name; metadata=fetch(BASE+name+"?generation="+generation,path)
        assert path.stat().st_size==size
        assert base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()==md5
        sources.append({"file":name,"generation":generation,"bytes":size,"gcs_md5_base64":md5,"sha256":metadata["sha256"]})
    info=json.loads((ROOT/"dataset_info.json").read_text()); Example=example_type(); rows=[]
    for i,payload in enumerate(records(ROOT/OBJECTS[-1][0])):
        example=Example.FromString(payload); features=example.features.feature; keys=sorted(features)
        schema={key:{"kind":features[key].WhichOneof("kind"),"values":len(getattr(features[key],features[key].WhichOneof("kind")).value)} for key in keys}
        first=np.asarray(features["steps/is_first"].int64_list.value); last=np.asarray(features["steps/is_last"].int64_list.value); n=len(first)
        assert n>1 and first.sum()==last.sum()==1 and first[0] and last[-1]
        values={"action":array(features["steps/action"],7,n)}
        for family,fields in (("action_dict",(("cartesian_velocity",6),("joint_velocity",7),("gripper_position",1))),("observation",(("cartesian_position",6),("joint_position",7),("gripper_position",1)))):
            for name,width in fields: values[family+"_"+name]=array(features["steps/"+family+"/"+name],width,n)
        assert all(np.isfinite(v).all() for v in values.values())
        values.update(is_first=first,is_last=last)
        out=OUTPUT/("episode"+str(i+1)); out.mkdir(exist_ok=True); np.savez_compressed(out/"transitions.npz",**values)
        instructions={name:sorted({v.decode("utf-8") for v in features["steps/"+name].bytes_list.value}) for name in ("language_instruction","language_instruction_2","language_instruction_3")}
        images={}
        for name in ("wrist_image_left","exterior_image_1_left","exterior_image_2_left"):
            frames=features["steps/observation/"+name].bytes_list.value; assert len(frames)==n
            with Image.open(io.BytesIO(frames[0])) as image:
                image.load(); assert image.size==(320,180); image.convert("RGB").save(out/(name+"_first.png"))
            images[name]={"frames":len(frames),"source_dimensions":[180,320,3],"first_image_decoded":True}
        cart=values["action_dict_cartesian_velocity"]; joint=values["action_dict_joint_velocity"]; action=values["action"]
        row={"id":"DROID-DEBUG-"+str(i+1),"steps":n,"features":schema,"language":instructions,"images":images,"metadata":{key:features[key].bytes_list.value[0].decode("utf-8") for key in ("episode_metadata/file_path","episode_metadata/recording_folderpath")},"action_first6_vs_cartesian_velocity_max_error":float(np.abs(action[:,:6]-cart).max()),"action_first6_vs_first6_joint_velocity_max_error":float(np.abs(action[:,:6]-joint[:,:6]).max()),"action_gripper_vs_commanded_position_max_error":float(np.abs(action[:,6:]-values["action_dict_gripper_position"]).max()),"physical_timestamps_available":any("timestamp" in key for key in keys),"fitted_model":False}; write_json(out/"schema.json",row); rows.append(row)
    assert len(rows)==int(info["splits"][0]["shardLengths"][0])==2
    write_json(OUTPUT/"report.json",{"sources":sources,"source_episodes":rows,"verified_tfrecord_crcs":True,"whole_debug_dataset_episodes":sum(map(int,info["splits"][0]["shardLengths"])),"whole_debug_dataset_bytes":int(info["splits"][0]["numBytes"]),"downloaded_only_first_shard":True,"model_fitted":False,"purpose":"Validate next action-conditioned data interface, not physical-control capability or a suitable scale-training split"})
    print(json.dumps({"action_data_probe_complete":True,"episodes":len(rows),"steps":[r["steps"] for r in rows],"cartesian_action_errors":[r["action_first6_vs_cartesian_velocity_max_error"] for r in rows],"timestamps":[r["physical_timestamps_available"] for r in rows]}),flush=True)


if __name__=="__main__": main()
