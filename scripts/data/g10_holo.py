"""Prepare causal ten-hertz AV, IMU and completed-speech windows from public HoloAssist."""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from g10_holo_index import ARCHIVES, BASE, fetch
from g9_recordings import spectrum

LIMITS = dict(train=256, dev=32, familiar=64, novel=64)
SOURCE_HASH = "cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c"
SPLIT_HASH = "e10674e7ac32957386d5e88afa60acf3335251b92ba2f8cfb97a736dc85e1621"
LOCAL = threading.local()


def rank(s):
    return hashlib.sha256(("G10/HoloAssist/" + s).encode()).hexdigest()


def session_group(s):
    return re.sub(r"[-_]part[-_]?\d+.*$", "", s.casefold())


def choose(pool, count, used):
    by_task = defaultdict(list)
    for record in pool:
        by_task[record["taskType"]].append(record)
    for items in by_task.values():
        items.sort(key=lambda r: rank(r["video_name"]), reverse=True)
    order = sorted(by_task, key=rank)
    chosen = []
    while len(chosen) < count:
        before = len(chosen)
        for task in order:
            while by_task[task]:
                item = by_task[task].pop()
                group = session_group(item["video_name"])
                if group not in used:
                    used.add(group)
                    chosen.append(item)
                    break
            if len(chosen) == count:
                break
        if before == len(chosen):
            raise ValueError(f"Too few disjoint eligible recordings for {count}")
    return chosen


def text_features(sentence):
    tokens = re.findall(r"[\w']+", sentence.casefold())
    grams = ["w:" + s for s in tokens] + ["b:" + a + " " + b for a,b in zip(tokens,tokens[1:])]
    chars = " "+" ".join(tokens)+" "
    grams += ["c:"+chars[i:i+3] for i in range(len(chars)-2)]
    out = np.zeros(128, np.float32)
    for gram in grams:
        h = int.from_bytes(hashlib.blake2s(gram.encode(),digest_size=4).digest(),"little")
        out[h%128] += 1 if h & 128 else -1
    return np.rint(127*out/max(1.,np.linalg.norm(out))).astype(np.int8)


def role_lists(root, labels, sources):
    with zipfile.ZipFile(root/"split.zip") as z:
        official = {role:{n.decode().strip().casefold() for n in z.read(role+"-v1_2.txt").splitlines()} for role in ("train","val","test")}
    pools = defaultdict(list)
    for r in labels:
        n=r["video_name"].casefold()
        if n not in sources or float(r["videoMetadata"]["duration"]["seconds"]) < 38:
            continue
        novel = any(s in r["taskType"].casefold() for s in ("coffee","computer"))
        if n in official["test"]:
            continue
        if novel and (n in official["train"] or n in official["val"]):
            pools["novel"].append(r)
        elif n in official["train"]:
            pools["train"].append(r)
        elif n in official["val"]:
            pools["familiar"].append(r)
    used=set()
    return {role:choose(pools["train" if role=="dev" else role],count,used) for role,count in LIMITS.items()}


def payload(index, archive, path):
    if not hasattr(LOCAL,"http"):
        LOCAL.http=requests.Session()
    off,n=index[archive][path]
    return fetch(LOCAL.http, BASE+archive+".tar",off,off+n,ARCHIVES[archive][0])


def decode(file, media):
    common=["ffmpeg","-nostdin","-v","error","-threads","1","-i",str(file),"-t","600"]
    if media=="video":
        cmd=common+["-map","0:v:0","-vf","fps=10,scale=16:16:flags=area","-threads","1","-an","-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
        return np.frombuffer(subprocess.check_output(cmd,timeout=300),np.uint8).reshape(-1,768)
    cmd=common+["-map","0:a:0","-ac","1","-ar","16000","-vn","-f","f32le","pipe:1"]
    pcm=np.frombuffer(subprocess.check_output(cmd,timeout=150),"<f4")
    chunks = [spectrum(pcm[i:i+1600*128,None],16000) for i in range(0,len(pcm)-1599,1600*128)]
    return np.concatenate(chunks,axis=0)


def one(root, row, index, files):
    name=row["video_name"]
    if not re.fullmatch(r"[A-Za-z0-9_. -]+",name):
        raise ValueError("Unknown session name")
    shard=root/"shards"/(name+".npz")
    note=shard.with_suffix(".json")
    if shard.exists() and note.exists():
        return json.loads(note.read_text())
    src=files[name.casefold()]
    video=root/"media"/(name+".mp4")
    if not video.exists():
        tmp=video.with_suffix(".tmp")
        tmp.write_bytes(payload(index,"video_compress",src["video"]))
        tmp.replace(video)
    streams=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=codec_type,start_time","-of","json",str(video)],timeout=30))["streams"]
    origin={v["codec_type"]:float(v.get("start_time",0)) for v in streams if v["codec_type"] in ("video","audio")}
    if set(origin)!={"audio","video"} or abs(origin["video"]-origin["audio"])>.10:
        raise ValueError(f"Unsupported audio/video clocks: {origin}")
    v=decode(video,"video"); a=decode(video,"audio")
    ticks=min(len(a),len(v),6000)
    if ticks<320 or abs(min(len(a),6000)-min(len(v),6000))>12:
        raise ValueError(f"Unexpected 10 Hz AV lengths {len(a)} and {len(v)}")
    ends=(np.arange(ticks)+1)/10
    n=np.zeros((ticks,12),np.float32)
    stale=np.zeros(ticks,bool)
    for start,key,scale in ((0,"acc",12.),(3,"gyro",4.)):
        raw=payload(index,"imu",src[key])
        arr=np.loadtxt(io.BytesIO(raw),usecols=(0,2,3,4),ndmin=2)
        if np.any(np.diff(arr[:,0])<0):raise ValueError("IMU clock regressed")
        pos=np.searchsorted(arr[:,0],ends,side="right")-1
        bad=(pos<0)|(ends-arr[np.maximum(0,pos),0]>.18)
        stale|=bad
        n[:,start:start+3]=np.where(bad[:,None],0,np.clip(arr[np.maximum(0,pos),1:]/scale,-4,4))
    n[:,6]=~stale
    natural=np.zeros((ticks,128),np.int8)
    words=np.zeros((ticks,8),np.uint8)
    utterances=[]; outcomes=[]
    for ev in sorted(row["events"],key=lambda x:float(x.get("end",0))):
        att=ev["attributes"]
        if ev["label"]=="Conversation":
            sentence=str(att.get("Transcription","")).strip()
            if not sentence or sentence.casefold() in ("none","*unintelligible*","[unintelligible]") or "low-confidence" in str(att.get("Transcription Confidence","")).lower():continue
            first=max(0,int(np.ceil(float(ev["end"])*10))-1)
            if first<ticks:
                natural[first:min(ticks,first+80)]=text_features(sentence)
                utterances.append([first,len(sentence.split())])
        if ev["label"]=="Fine grained action":
            label=att.get("Action Correctness","")
            if label!="Correct Action" and not label.startswith("Wrong Action"):continue
            # A full 200 ms guard separates the usable input from the beginning of the target action.
            last=int(np.floor(float(ev["start"])*10))-3
            if 31<=last<ticks:outcomes.append((last,int(label!="Correct Action")))
    tmp=shard.with_suffix(".partial.npz")
    np.savez_compressed(tmp,a=a[:ticks],v=v[:ticks],n=n,w=words,x=natural,events=np.asarray(outcomes,np.int32).reshape(-1,2))
    tmp.replace(shard)
    with shard.open("rb") as f:digest=hashlib.file_digest(f,"sha256").hexdigest()
    record={"name":name,"task":row["taskType"],"ticks":ticks,"sha256":digest,"completed_utterances":len(utterances),"transcribed_words":sum(i[1] for i in utterances),
            "text_present_ticks":int((natural!=0).any(1).sum()),"stale_imu_ticks":int(stale.sum()),"action_examples":len(outcomes),"mistakes":sum(l for _,l in outcomes)}
    nt=note.with_suffix(".new");nt.write_text(json.dumps(record));nt.replace(note)
    return record


def index_sources(root):
    index={}
    for k in ARCHIVES:
        p=root/(k+"-index.json")
        while not p.exists():time.sleep(5)
        index[k]=json.loads(p.read_bytes())["members"]
    sources=defaultdict(dict)
    for archive,d in index.items():
        for path in d:
            name=path.split("/")[0].casefold()
            leaf=path.split("/")[-1].casefold()
            kind={"video_compress.mp4":"video","accelerometer_sync.txt":"acc","gyroscope_sync.txt":"gyro"}.get(leaf)
            if kind:sources[name][kind]=path
    return index,{k:v for k,v in sources.items() if len(v)==3}


def main(root):
    for name,digest in (("labels.json",SOURCE_HASH),("split.zip",SPLIT_HASH)):
        with (root/name).open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=digest:raise ValueError("Source hash mismatch")
    for k in ("media","shards"):(root/k).mkdir(exist_ok=True)
    idx,files=index_sources(root)
    groups=role_lists(root,json.loads((root/"labels.json").read_bytes()),files)
    (root/"selection.json").write_text(json.dumps({k:[r["video_name"] for r in rs] for k,rs in groups.items()},indent=2))
    receipt={}
    for role in ("train","dev","familiar","novel"):
        with ThreadPoolExecutor(8) as pool:
            rows=list(pool.map(lambda r:one(root,r,idx,files),groups[role]))
        receipt[role]=rows
        tmp=root/"banks.new";tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(root/"banks.json")
        print(role,len(rows),sum(r["ticks"] for r in rows),sum(r["mistakes"] for r in rows),flush=True)


if __name__=="__main__":
    import sys
    main(Path(sys.argv[1]))
