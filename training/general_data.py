"""Audited G1 caches: real pairs, group splits, physical units, explicit missingness.

No model is fitted here. Every record keeps its original source/group identity.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import ctypes
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation

from training.general_sources import ROOT, write
from training.research_sources import file_sha

CACHE=ROOT/"prepared"
ROLES=("train","selection","calibration","test")
SIGNALS=[f"{site}.{kind}.{axis}" for site in ("wrist","chest","ankle") for kind in ("accel","gyro","mag") for axis in ("x","y","z")]
SIGNALS += [f"{site}.temperature" for site in ("wrist","chest","ankle")]+["heart_rate","chest.ecg.1","chest.ecg.2"]
UNITS=["m/s2" if ".accel." in s else "rad/s" if ".gyro." in s else "source_magnetic" if ".mag." in s else "C" if "temperature" in s else "bpm" if s=="heart_rate" else "mV" for s in SIGNALS]


def order(value): return hashlib.sha256(("G1-20261106:"+str(value)).encode()).hexdigest()


def bytes_sha(value): return hashlib.sha256(value).hexdigest()


def rows_write(folder,rows,**extra):
    write(folder/"rows.json",rows)
    card={"records":len(rows),"roles":dict(Counter(r["role"] for r in rows)),
          "groups":{role:sorted({r["group"] for r in rows if r["role"]==role}) for role in ROLES},
          "preparer_sha256":file_sha(__file__),"rows_sha256":file_sha(folder/"rows.json"),**extra}
    sets=[set(card["groups"][r]) for r in ROLES]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    write(folder/"manifest.json",card)
    print(json.dumps({"prepared":str(folder),"records":len(rows),"roles":card["roles"]}),flush=True)
    return card


def source(family,name):
    path=ROOT/family/"sources"/name
    verification=json.loads(path.with_suffix(path.suffix+".verified.json").read_text())
    assert verification["verified"] and verification["source"]["sha256"]==file_sha(path),str(path)
    return path


def rgb(value,size=64):
    """Encoded RGB/JPEG -> uint8[H,W,3]; fit preserves aspect before center crop."""
    with Image.open(io.BytesIO(value)) as image:
        image.load()
        return np.asarray(ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"),(size,size),method=Image.Resampling.BOX)).copy()


def prepare_coco():
    folder=CACHE/"coco"; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"manifest.json").exists(): return
    rows=[]; licenses={}; seen={}; duplicates=[]
    with zipfile.ZipFile(source("coco","annotations_trainval2017.zip")) as annotations:
        # Highest-priority held roles first; any exact pixel/byte overlap is purged
        # from all lower-priority roles before fitting. This is not near-dup removal.
        for split in ("val2017","train2017"):
            data=json.loads(annotations.read("annotations/captions_"+split+".json")); licenses[split]=data["licenses"]
            captions=defaultdict(list)
            for item in data["annotations"]: captions[item["image_id"]].append(item["caption"])
            images=sorted(data["images"],key=lambda r:order(r["id"]))
            if split=="val2017":
                for i,item in enumerate(images): item["role"]="test" if i<2000 else "calibration" if i<3000 else "selection"
            else:
                for item in images: item["role"]="train"
            images.sort(key=lambda r:(-ROLES.index(r["role"]),order(r["id"])))
            output=np.lib.format.open_memmap(folder/(split+"_rgb.npy"),mode="w+",dtype=np.uint8,shape=(len(images),64,64,3))
            with zipfile.ZipFile(source("coco",split+".zip")) as archive:
                for i,item in enumerate(images):
                    raw=archive.read(split+"/"+item["file_name"]); pixels=rgb(raw); output[i]=pixels
                    digest=bytes_sha(raw); pixel_digest=bytes_sha(pixels.tobytes()); matches=[seen[k] for k in (digest,pixel_digest) if k in seen]
                    if matches:
                        duplicates.append({"id":item["id"],"role":item["role"],"kept":matches[0],"reason":"exact source bytes or exact64x64 RGB"}); continue
                    seen[digest]=seen[pixel_digest]=item["id"]
                    rows.append({"id":"COCO-"+str(item["id"]),"group":"COCO-"+str(item["id"]),"role":item["role"],"index":i,"array":split+"_rgb.npy","captions":captions[item["id"]],"license":item["license"],"original_url":item.get("flickr_url"),"raw_sha256":digest,"pixel_sha256":pixel_digest})
                    if i%10000==0: print(json.dumps({"coco_images":i,"split":split}),flush=True)
            output.flush(); del output
    write(folder/"licenses.json",licenses); write(folder/"deduplication.json",duplicates)
    rows_write(folder,rows,caption_count=sum(len(r["captions"]) for r in rows),pixel_shape=[64,64,3],
               near_duplicate_removal=False,files={p.name:file_sha(p) for p in folder.glob("*.npy")})


def signal_array(data,family):
    """Official table[N,54/24] -> SI-valued float32[N,33], activity[N], time[N].

    Invalid PAMAP2 orientation fields are never ingested. NaN stays unobserved.
    MHEALTH gyros are converted from degrees/s. Its magnetic units are local,
    so those channels cannot support a physically matched cross-source claim.
    """
    result=np.full((len(data),len(SIGNALS)),np.nan,np.float32)
    if family=="pamap2":
        assert data.shape[1]==54
        for site,start in (("wrist",3),("chest",20),("ankle",37)):
            result[:,SIGNALS.index(site+".temperature")]=data[:,start]
            for kind,offset in (("accel",1),("gyro",7),("mag",10)):
                for k,axis in enumerate("xyz"): result[:,SIGNALS.index(site+"."+kind+"."+axis)]=data[:,start+offset+k]
        result[:,SIGNALS.index("heart_rate")]=data[:,2]
        labels=data[:,1].astype(np.int16); clock=data[:,0]
    else:
        assert data.shape[1]==24
        for site,kind,start in (("chest","accel",0),("ankle","accel",5),("ankle","gyro",8),("ankle","mag",11),("wrist","accel",14),("wrist","gyro",17),("wrist","mag",20)):
            for k,axis in enumerate("xyz"): result[:,SIGNALS.index(site+"."+kind+"."+axis)]=data[:,start+k]*(np.pi/180 if kind=="gyro" else 1)
        result[:,SIGNALS.index("chest.ecg.1")]=data[:,3]; result[:,SIGNALS.index("chest.ecg.2")]=data[:,4]
        labels=data[:,23].astype(np.int16); clock=np.arange(len(data))/50
    return result,labels,np.asarray(clock,np.float64)


def prepare_signals(family):
    folder=CACHE/family; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"manifest.json").exists(): return
    path=source(family,family+".zip"); outer=zipfile.ZipFile(path)
    if family=="pamap2":
        nested=next((n for n in outer.namelist() if n.lower().endswith(".zip")),None)
        archive=zipfile.ZipFile(io.BytesIO(outer.read(nested))) if nested else outer
        names=sorted(n for n in archive.namelist() if n.endswith(".dat") and "/Protocol/" in n)
    else: archive=outer; names=sorted(n for n in archive.namelist() if re.search(r"mHealth_subject\d+\.log$",n))
    rows=[]
    for name in names:
        number=int(re.search(r"subject(\d+)",name).group(1)); number=number-100 if family=="pamap2" else number
        # Source-separated transfer: only PAMAP2 supplies generic pretraining.
        # MHEALTH people1-3 may fit readouts;4-5 select them;6-10 are final only.
        role=("train" if number<=6 else "selection" if number==7 else "calibration" if number==8 else "test") if family=="pamap2" else ("train" if number<=3 else "selection" if number<=5 else "test")
        table=np.loadtxt(io.BytesIO(archive.read(name)),dtype=np.float64); values,labels,clock=signal_array(table,family)
        if family=="pamap2":
            # Keep native100Hz samples. The loader performs clock-aware windows;
            # no interpolation can read held future samples into observed input.
            assert np.all(np.diff(clock)>0)
        key=family.upper()+"-"+str(number); np.save(folder/(key+"_signals.npy"),values); np.save(folder/(key+"_activity.npy"),labels); np.save(folder/(key+"_time.npy"),clock)
        rows.append({"id":key,"group":key,"role":role,"steps":len(values),"rate_hz":100 if family=="pamap2" else 50,"duration_seconds":len(values)/(100 if family=="pamap2" else 50),"file":key+"_signals.npy","activity_file":key+"_activity.npy","time_file":key+"_time.npy","source_member":name,"source_table_sha256":bytes_sha(archive.read(name)),"missing_fraction":np.mean(~np.isfinite(values),axis=0).tolist()})
    assert len(rows)==(9 if family=="pamap2" else 10),(family,len(rows))
    write(folder/"channels.json",{"names":SIGNALS,"units":UNITS,"invalid_orientation_excluded":True,"unobserved":"NaN with explicit Boolean presence at model input","cross_source_common_channels":[i for i,s in enumerate(SIGNALS) if ".accel." in s or ".gyro." in s],"cross_source_limit":"Different body-side/axis placement remains real domain shift; no axis equivalence asserted"})
    rows_write(folder,rows,hours=sum(r["duration_seconds"] for r in rows)/3600,files={p.name:file_sha(p) for p in folder.glob("*.npy")},mhealth_use="Held source during shared pretraining; people1-3 for new readouts only" if family=="mhealth" else None)


def native_crc():
    """Compile a small reentrant CPU checksum; no environment package changes."""
    code=Path("training/tfrecord_crc32c.c"); library=Path("/tmp/slwm_crc32c_"+file_sha(code)[:16]+".so")
    if not library.exists(): subprocess.run(["cc","-O3","-shared","-fPIC",str(code),"-o",str(library)],check=True)
    module=ctypes.CDLL(str(library)); function=module.slwm_crc32c
    function.argtypes=[ctypes.c_char_p,ctypes.c_size_t]; function.restype=ctypes.c_uint32
    def crc(value): return int(function(value,len(value)))
    assert crc(b"123456789")==0xe3069283
    return crc


def checked_records(path,crc):
    """TFRecord stream -> independently CRC-verified episode payloads, <=256MiB."""
    def masked(value):
        check=crc(value); return (((check>>15)|(check<<17))+0xa282ead8)&0xffffffff
    with Path(path).open("rb") as stream:
        while True:
            header=stream.read(12)
            if not header: return
            if len(header)!=12: raise ValueError("Truncated record header")
            length,check=struct.unpack("<QI",header)
            if length>256*1024**2 or check!=masked(header[:8]): raise ValueError("Record size/length CRC failed")
            value=stream.read(length); tail=stream.read(4)
            if len(value)!=length or len(tail)!=4 or struct.unpack("<I",tail)[0]!=masked(value): raise ValueError("Record payload CRC failed")
            yield value


def pose_delta(observed,commanded):
    """World XYZ+EulerXYZ[N,6] -> body-frame translation+rotation-vector[N,6]."""
    initial=Rotation.from_euler("xyz",observed[:,3:]); target=Rotation.from_euler("xyz",commanded[:,3:])
    return np.concatenate([initial.inv().apply(commanded[:,:3]-observed[:,:3]),(initial.inv()*target).as_rotvec()],axis=-1).astype(np.float32)


def prepare_droid():
    from training.action_data_probe import example_type,array
    folder=CACHE/"droid"; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"manifest.json").exists(): return
    Example=example_type(); crc=native_crc(); rows=[]
    paths=sorted((ROOT/"droid/sources").glob("*.tfrecord-*")); paths=[p for p in paths if not p.name.endswith((".json",".part"))]
    assert len(paths)==31,len(paths)
    for path in paths:
        source("droid",path.name)
        for value in checked_records(path,crc):
            features=Example.FromString(value).features.feature; n=len(features["steps/is_first"].int64_list.value)
            assert features["steps/is_first"].int64_list.value[0] and features["steps/is_last"].int64_list.value[-1]
            raw_path=features["episode_metadata/file_path"].bytes_list.value[0].decode(); group_match=re.search(r"/([^/]+)/(?:success|failure)/(\d{4}-\d{2}-\d{2})/",raw_path)
            if not group_match: raise ValueError("Unreviewed collection/session path: "+raw_path)
            collector,date=group_match.groups(); group=collector+"-"+date
            key="DROID-"+order(raw_path)[:20]; dest=folder/key; dest.mkdir(exist_ok=True)
            observed=array(features["steps/observation/cartesian_position"],6,n); commanded=array(features["steps/action_dict/cartesian_position"],6,n)
            gripper=array(features["steps/observation/gripper_position"],1,n); commanded_gripper=array(features["steps/action_dict/gripper_position"],1,n)
            action=array(features["steps/action"],7,n)
            assert np.array_equal(action,np.concatenate([commanded,commanded_gripper],axis=1)),raw_path
            assert all(np.isfinite(v).all() for v in (observed,commanded,gripper,commanded_gripper))
            positions=np.concatenate([observed[:,:3],Rotation.from_euler("xyz",observed[:,3:]).as_rotvec(),gripper],axis=1).astype(np.float32)
            delta=np.concatenate([pose_delta(observed,commanded),commanded_gripper],axis=1)
            np.savez(dest/"trajectory.npz",observed=observed.astype(np.float32),state=positions,commanded=commanded.astype(np.float32),action=delta,gripper=gripper.astype(np.float32))
            pixels=np.lib.format.open_memmap(dest/"rgb.npy",mode="w+",dtype=np.uint8,shape=(n,3,64,64,3))
            for view,name in enumerate(("wrist_image_left","exterior_image_1_left","exterior_image_2_left")):
                frames=features["steps/observation/"+name].bytes_list.value; assert len(frames)==n
                for tick,frame in enumerate(frames): pixels[tick,view]=rgb(frame)
            pixels.flush(); del pixels
            captions=sorted({v.decode() for name in ("language_instruction","language_instruction_2","language_instruction_3") for v in features["steps/"+name].bytes_list.value if v.decode().strip()})
            rows.append({"id":key,"group":group,"collector":collector,"session_date":date,"raw_path":raw_path,"steps":n,"captions":captions,"physical_timestamps":False,"environment_id_verified":False,"source_payload_sha256":bytes_sha(value),"files":{p.name:file_sha(p) for p in dest.iterdir()}})
            print(json.dumps({"droid_episode":key,"steps":n,"group":group,"episodes":len(rows)}),flush=True)
    assert len(rows)==100,len(rows)
    groups=sorted({r["group"] for r in rows},key=order)
    if len(groups)<4: raise ValueError("Fewer than4 collector-day groups; source cannot support registered4-way role split")
    n=len(groups); group_roles={g:"train" if i<max(1,int(n*.65)) else "selection" if i<max(2,int(n*.8)) else "calibration" if i<max(3,int(n*.9)) else "test" for i,g in enumerate(groups)}
    for row in rows: row["role"]=group_roles[row["group"]]
    rows_write(folder,rows,total_steps=sum(r["steps"] for r in rows),all_jpegs_decoded=True,
        action_contract="Default7D action exactly equals commanded Cartesian poseXYZ/EulerXYZ plus gripper in all100 episodes; model targets are observed-body-frame SE3 delta plus gripper. Prediction horizons are ordinal steps.",
        environment_limit="Collector-day disjoint; unseen buildings/scenes NOT established without raw metadata",crc_source_sha256=file_sha("training/tfrecord_crc32c.c"))


def safe_7z_extract(path,dest):
    listing=subprocess.check_output(["7z","l","-slt",str(path)]).decode(errors="replace")
    records=listing.split("----------",1)[1]
    for line in records.splitlines():
        if line.startswith("Path = "):
            item=Path(line[7:]); assert not item.is_absolute() and ".." not in item.parts
        assert not line.startswith(("Symbolic Link = ","Hard Link = "))
    dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(["7z","x","-aos","-y","-o"+str(dest),str(path)],check=True,stdout=subprocess.DEVNULL)


def prepare_clotho():
    import soundfile as sf
    import torch
    import torchaudio
    from vocos.feature_extractors import MelSpectrogramFeatures
    folder=CACHE/"clotho"; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"manifest.json").exists(): return
    # Group the same Freesound uploader across every role, including official
    # cross-split overlap. Held precedence purges development examples first.
    rows=[]; role_rank={"development":0,"validation":1,"evaluation":3}; owners=defaultdict(int)
    for split in role_rank:
        captions={r["file_name"]:[r["caption_"+str(i)] for i in range(1,6)] for r in csv.DictReader(source("clotho","clotho_captions_"+split+".csv").open())}
        metadata=list(csv.DictReader(source("clotho","clotho_metadata_"+split+".csv").open()))
        for item in metadata:
            owner=item["manufacturer"]; assert owner; owners[owner]=max(owners[owner],role_rank[split])
            rows.append({"id":"CLOTHO-"+item["sound_id"],"file_name":item["file_name"],"group":"CLOTHO-UPLOADER-"+owner,"owner":owner,"official_split":split,"captions":captions[item["file_name"]],"original_url":item["sound_link"],"original_license":item["license"]})
    kept=[]; purged=[]; extractor=MelSpectrogramFeatures(); torch.set_num_threads(1)
    for split in role_rank:
        archive=source("clotho","clotho_audio_"+split+".7z"); raw=folder/("decoded_"+split); safe_7z_extract(archive,raw)
        filenames=defaultdict(list)
        for path in raw.rglob("*.wav"): filenames[path.name].append(path)
        for row in [r for r in rows if r["official_split"]==split]:
            if owners[row["owner"]]>role_rank[split]: purged.append(row); continue
            row["role"]="train" if split=="development" else "test" if split=="evaluation" else ("selection" if int(order(row["owner"])[:8],16)%2 else "calibration")
            matches=filenames[row["file_name"]]; assert len(matches)==1,row["file_name"]
            path=matches[0]; audio,sr=sf.read(path,dtype="float32",always_2d=True); tensor=torch.from_numpy(audio.mean(1))[None]
            if sr!=24000: tensor=torchaudio.functional.resample(tensor,sr,24000)
            mel=extractor(tensor)[0].T.numpy(); length=len(mel)//4; mel=mel[:length*4].reshape(length,400).astype(np.float16)
            name=row["id"]+"_mel.npy"; np.save(folder/name,mel); row.update(file=name,steps=length,duration_seconds=len(audio)/sr,raw_sha256=file_sha(path),mel_sha256=file_sha(folder/name)); kept.append(row)
            if len(kept)%128==0: print(json.dumps({"clotho_prepared":len(kept),"split":split}),flush=True)
    write(folder/"purged_uploader_overlap.json",purged)
    rows_write(folder,kept,hours=sum(r["duration_seconds"] for r in kept)/3600,original_records=len(rows),purged_overlap=len(purged),frame_seconds=1024/24000,caption_license_file=str(source("clotho","LICENSE")),codec="Fixed Vocos log-mel transform, no learned audio teacher; waveform rendering external and separately counted")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--family",choices=("coco","clotho","pamap2","mhealth","droid"),required=True); args=parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    if args.family=="coco": prepare_coco()
    elif args.family=="clotho": prepare_clotho()
    elif args.family=="droid": prepare_droid()
    else: prepare_signals(args.family)


if __name__=="__main__": main()
