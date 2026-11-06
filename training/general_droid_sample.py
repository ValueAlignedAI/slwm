"""Bounded deterministic episode ranges from full official DROID1.0.1.

One complete first TFRecord from each hash-selected shard. This is not a uniform
episode sample or a whole-shard checksum verification. Generation pins, exact
HTTP ranges, complete record CRCs and local SHA256 identify every acquired byte.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import io
import json
from pathlib import Path
import re
import shutil
import struct
import urllib.parse
import urllib.request
import numpy as np
from scipy.spatial.transform import Rotation
from training.action_data_probe import example_type,array
from training.general_data import ROOT,CACHE,order,native_crc,checked_records,rgb,pose_delta
from training.general_sources import write
from training.research_sources import file_sha

FAMILY="droid_expanded"


def pin():
    root=ROOT/FAMILY; root.mkdir(parents=True,exist_ok=True); path=root/"source_plan.json"
    if path.exists(): return json.loads(path.read_text())
    items=[]; cursor=None
    while True:
        params={"prefix":"robotics/droid/1.0.1/","maxResults":1000}
        if cursor: params["pageToken"]=cursor
        url="https://storage.googleapis.com/storage/v1/b/gresearch/o?"+urllib.parse.urlencode(params)
        with urllib.request.urlopen(url,timeout=45) as response: page=json.load(response)
        items+=page.get("items",[]); cursor=page.get("nextPageToken")
        if not cursor: break
    write(root/"objects.json",items)
    shards=sorted([i for i in items if re.search(r"\.tfrecord-\d+-of-\d+$",i["name"])],key=lambda i:order(i["name"]))
    assert len(shards)==2048,len(shards)
    plan={"objects":shards[:384],"episodes_requested":384,"maximum_record_bytes":256*1024**2,
          "maximum_total_record_bytes":16*1024**3,"source_sha256":file_sha(__file__),"method":"First complete record per deterministic hash-selected shard; not uniform episode sampling. Keep duplicates/failures explicit; do not substitute according to model scores.","published_whole_shard_checksums_verified":False}
    write(path,plan); return plan


def get_range(item,a,b):
    url="https://storage.googleapis.com/gresearch/"+urllib.parse.quote(item["name"],safe="/")+"?generation="+item["generation"]
    request=urllib.request.Request(url,headers={"Range":f"bytes={a}-{b}"})
    with urllib.request.urlopen(request,timeout=90) as response:
        assert response.status==206 and response.headers["Content-Range"]==f"bytes {a}-{b}/{item['size']}"
        value=response.read(); assert len(value)==b-a+1
        return value


def acquire(item,limit,header=None):
    root=ROOT/FAMILY; path=root/"sources"/(item["name"].rsplit("/",1)[-1]+".first_record"); path.parent.mkdir(parents=True,exist_ok=True)
    card=path.with_suffix(path.suffix+".verified.json"); crc=native_crc()
    if path.exists() and card.exists():
        saved=json.loads(card.read_text()); assert file_sha(path)==saved["sha256"]; return saved
    if shutil.disk_usage(root).free<18*1024**3: raise OSError("Archive verified task-owned transports before further robot acquisition")
    header=get_range(item,0,11) if header is None else header; length,checksum=struct.unpack("<QI",header); check=crc(header[:8]); masked=(((check>>15)|(check<<17))+0xa282ead8)&0xffffffff
    assert checksum==masked and length<=limit
    payload=get_range(item,12,length+15); temporary=path.with_suffix(path.suffix+".part"); temporary.write_bytes(header+payload)
    values=list(checked_records(temporary,crc)); assert len(values)==1; temporary.replace(path)
    saved={"file":str(path),"bytes":path.stat().st_size,"sha256":file_sha(path),"source_object":item,"range":[0,length+15],"whole_shard_checksum_verified":False,"complete_record_crc_verified":True}
    write(card,saved); return saved


def download():
    plan=pin(); results=[]; errors=[]
    # Check the whole sample's exact payload budget before issuing large ranges.
    with ThreadPoolExecutor(max_workers=4) as pool:
        headers=list(pool.map(lambda item:get_range(item,0,11),plan["objects"]))
    lengths=[struct.unpack("<QI",h)[0]+16 for h in headers]
    assert all(n<=plan["maximum_record_bytes"]+16 for n in lengths)
    assert sum(lengths)<=plan["maximum_total_record_bytes"],"Pinned sample exceeds archive budget; no bulk ranges requested"
    write(ROOT/FAMILY/"record_header_plan.json",{"records":[{"object":item["name"],"generation":item["generation"],"header_hex":header.hex(),"bytes":length} for item,header,length in zip(plan["objects"],headers,lengths)],"total_bytes":sum(lengths)})
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(acquire,item,plan["maximum_record_bytes"],header):item for item,header in zip(plan["objects"],headers)}
        for future in as_completed(pending):
            item=pending[future]
            try: results.append(future.result())
            except Exception as exc: errors.append({"object":item,"error_type":type(exc).__name__,"error":str(exc)})
            write(ROOT/FAMILY/"acquisition_progress.json",{"verified":results,"errors":errors,"requested":len(pending)})
            print(json.dumps({"full_droid_records":len(results),"failed":len(errors),"verified_bytes":sum(r["bytes"] for r in results)}),flush=True)
    assert sum(r["bytes"] for r in results)<=plan["maximum_total_record_bytes"],"Sample exceeded registered archive budget; do not silently fit"
    write(ROOT/FAMILY/"acquisition_complete.json",{"verified":results,"errors":errors,"all_requested_verified":not errors})
    if errors: raise RuntimeError("Retain explicit range acquisition failures before any data-role revision")


def prepare():
    folder=CACHE/FAMILY; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"schema_manifest.json").exists(): return
    plan=pin(); Example=example_type(); crc=native_crc(); rows=[]; seen=set(); duplicates=[]
    for item in plan["objects"]:
        card=acquire(item,plan["maximum_record_bytes"]); payload=next(checked_records(card["file"],crc)); features=Example.FromString(payload).features.feature; n=len(features["steps/is_first"].int64_list.value)
        assert features["steps/is_first"].int64_list.value[0] and features["steps/is_last"].int64_list.value[-1]
        raw_path=features["episode_metadata/file_path"].bytes_list.value[0].decode()
        if raw_path in seen: duplicates.append(raw_path); continue
        seen.add(raw_path); match=re.search(r"/([^/]+)/(?:success|failure)/(\d{4}-\d{2}-\d{2})/",raw_path)
        assert match,raw_path; collector,date=match.groups(); key="DROID-"+order(raw_path)[:20]; dest=folder/key; dest.mkdir(exist_ok=True)
        observed=array(features["steps/observation/cartesian_position"],6,n); commanded=array(features["steps/action_dict/cartesian_position"],6,n)
        gripper=array(features["steps/observation/gripper_position"],1,n); commanded_gripper=array(features["steps/action_dict/gripper_position"],1,n); default=array(features["steps/action"],7,n)
        assert all(np.isfinite(v).all() for v in (observed,commanded,gripper,commanded_gripper))
        rotation=Rotation.from_euler("xyz",observed[:,3:]).as_matrix()[:,:,:2].reshape(n,6)
        state=np.concatenate([observed[:,:3],rotation,gripper],1).astype(np.float32); actions=np.concatenate([pose_delta(observed,commanded),commanded_gripper],1)
        np.savez(dest/"trajectory.npz",observed=observed.astype(np.float32),state=state,commanded=commanded.astype(np.float32),action=actions,gripper=gripper.astype(np.float32))
        pixels=np.lib.format.open_memmap(dest/"rgb.npy",mode="w+",dtype=np.uint8,shape=(n,3,64,64,3))
        for view,name in enumerate(("wrist_image_left","exterior_image_1_left","exterior_image_2_left")):
            frames=features["steps/observation/"+name].bytes_list.value; assert len(frames)==n
            for tick,frame in enumerate(frames): pixels[tick,view]=rgb(frame)
        pixels.flush(); del pixels
        captions=sorted({v.decode() for name in ("language_instruction","language_instruction_2","language_instruction_3") for v in features["steps/"+name].bytes_list.value if v.decode().strip()})
        row={"id":key,"group":collector+"-"+date,"collector":collector,"session_date":date,"raw_path":raw_path,"steps":n,"captions":captions,"physical_timestamps":False,"environment_id_verified":False,"source_payload_sha256":card["sha256"],"default_action_equals_commanded_pose_gripper":bool(np.array_equal(default,np.concatenate([commanded,commanded_gripper],1))),"files":{p.name:file_sha(p) for p in dest.iterdir()}}
        rows.append(row); write(folder/"rows.json",rows)
        if len(rows)%16==0: print(json.dumps({"prepared_full_droid_episodes":len(rows),"steps":sum(r["steps"] for r in rows)}),flush=True)
    write(folder/"schema_manifest.json",{"records":len(rows),"steps":sum(r["steps"] for r in rows),"duplicates":duplicates,"all_jpegs_decoded":True,"rows_sha256":file_sha(folder/"rows.json"),"source_plan_sha256":file_sha(ROOT/FAMILY/"source_plan.json"),"source_sha256":file_sha(__file__),"state":"World XYZ + first two rotation-matrix columns + observed gripper (10 dimensions), continuous across Euler wrap","model_action":"Explicit commanded Cartesian pose converted to observed-body-frame SE3 delta + commanded gripper; no default-action-schema assumption","time":"Ordinal RLDS steps; no physical cadence claim"})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("pin","download","prepare"),required=True);args=parser.parse_args()
    if args.stage=="pin": print(json.dumps({"pinned_records":len(pin()["objects"])}),flush=True)
    elif args.stage=="download": download()
    else: prepare()


if __name__=="__main__":main()
