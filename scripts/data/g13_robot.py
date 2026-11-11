"""Decode real Language-Table sessions, group overlapping footage and construct causal measurement banks."""
import argparse
import base64
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import itertools
import json
import re
import struct
import sys
from pathlib import Path
import numpy as np
import cv2
from g13_acquire import ROOT,write,file_hash
from data.g10_language import text_features

COLORS=("red","blue","green","yellow")
DIRECTIONS=("left","right","up","down")
NEW_PAIRS={(0,1),(1,2),(2,3),(3,0)}
R=ROOT/"robot"


def rank(text):
    return hashlib.sha256(("G13/"+str(text)).encode()).hexdigest()


def get_schema():
    from example_pb2 import Example
    return Example


def unpack_source(raw):
    data=base64.b64decode(raw+b"="*((-len(raw))%4),altchars=b"-_",validate=True)
    mark=re.search(rb"\x2a\x20([a-f0-9]{32})$",data)
    if not mark:raise ValueError("Missing parent ID in the released episode key")
    return mark[1].decode()


def episode(buf):
    f=get_schema().FromString(buf).features.feature
    raw=f["episode_id"].bytes_list.value[0]
    img=f["steps/observation/rgb"].bytes_list.value;n=len(img)
    words=np.asarray(f["steps/observation/instruction"].int64_list.value,np.int32).reshape(n,512)
    if not np.array_equal(words,np.broadcast_to(words[:1],words.shape)):raise ValueError("Instruction changed during subtrajectory")
    text=bytes(words[0].astype(np.uint8)).split(b"\0")[0].decode("utf8").strip()
    def numeric(name):
        return np.asarray(f["steps/"+name].float_list.value,np.float32).reshape(n,2)
    state=numeric("observation/effector_translation");u=numeric("action")
    return raw.decode(),unpack_source(raw),text,img,state,u


def source_records(path):
    import ctypes
    checksum=ctypes.CDLL("libcrc32c.so.1").crc32c_value
    checksum.argtypes=[ctypes.c_char_p,ctypes.c_size_t];checksum.restype=ctypes.c_uint32
    def masked(buf):
        crc=checksum(buf,len(buf));return (((crc>>15)|(crc<<17))+0xa282ead8)&0xffffffff
    with path.open("rb") as fh:
        while True:
            offset=fh.tell();header=fh.read(12)
            if not header:return
            if len(header)!=12 or masked(header[:8])!=struct.unpack_from("<I",header,8)[0]:raise ValueError("TFRecord header integrity")
            n=struct.unpack_from("<Q",header)[0]
            if not 0<n<150<<20:raise ValueError("Unexpected episode size")
            buf=fh.read(n);crc=fh.read(4)
            if len(crc)!=4 or len(buf)!=n or masked(buf)!=struct.unpack("<I",crc)[0]:raise ValueError("TFRecord payload integrity")
            yield offset,buf


def visual(bgr):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv);height,width=h.shape
    roi=np.zeros(h.shape,np.uint8)
    poly=np.array([[.14,.035],[.84,.035],[.995,.90],[.005,.90]])*np.array([width,height])
    cv2.fillPoly(roi,[poly.astype(np.int32)],1)
    base=(s>82)&(v>54)&(roi>0)
    hues=((h<=10)|(h>=168),(h>=94)&(h<=133),(h>=38)&(h<=92),(h>=13)&(h<=37))
    out=np.zeros((4,2,6),np.float32)
    for c,hue in enumerate(hues):
        mask=cv2.medianBlur((base&hue).astype(np.uint8),3)
        _,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        candidates=[]
        for (x,y,w,hh,area),(cx,cy) in zip(stats[1:],centers[1:]):
            if not (28<=area<=1450 and 5<=w<=54 and 5<=hh<=56 and area/(w*hh)>.38):continue
            candidates.append((area,cx/width,cy/height,w/width,hh/height))
        candidates=sorted(sorted(candidates,reverse=True)[:2],key=lambda item:item[1])
        for j,(area,x,y,w,hh) in enumerate(candidates):out[c,j]=(x,y,w,hh,area/(height*width),1.)
    tiny=cv2.resize(bgr,(16,10),interpolation=cv2.INTER_AREA)[:,:,::-1].reshape(480)
    return tiny,out


def decode_shard(arg):
    path,folder=arg;tag=path.name.split("-")[-1];file=folder/(tag+".npz");meta=file.with_suffix(".json")
    if file.is_file() and meta.is_file():return tag
    cv2.setNumThreads(1);arrays=defaultdict(list);rows=[];start=0
    for off,buf in source_records(path):
        ident,parent,goal,frames,state,u=episode(buf)
        if not goal or len(frames)<5 or not np.isfinite(state).all() or not np.isfinite(u).all():continue
        tiny=[];geom=[];hashes=[]
        for frame in frames:
            im=cv2.imdecode(np.frombuffer(frame,np.uint8),cv2.IMREAD_REDUCED_COLOR_2)
            if im is None or im.shape[:2]!=(180,320):raise ValueError("Changed RGB format")
            px,g=visual(im);tiny.append(px);geom.append(g);hashes.append(hashlib.blake2b(frame,digest_size=12).digest().hex())
        n=len(tiny);rows.append(dict(episode=ident,parent=parent,goal=goal,shard=tag,offset=off,start=start,length=n))
        for k,value in dict(px=np.stack(tiny),geom=np.asarray(geom,np.float16),frame=np.asarray(hashes,"S24"),x=state,u=u).items():arrays[k].append(value)
        start+=n
    temp=file.with_suffix(".part.npz");np.savez_compressed(temp,**{k:np.concatenate(v) for k,v in arrays.items()});temp.replace(file)
    write(meta,rows);print("Decoded source",tag,len(rows),start,flush=True);return tag


def semantics(goal):
    tokens=re.findall(r"[a-z]+",goal.lower())
    colours=[COLORS.index(t) for t in tokens if t in COLORS]
    ordered=list(dict.fromkeys(colours));source=ordered[0] if ordered else -1
    destination=ordered[1] if len(ordered)>1 else -1
    mentions=[t in tokens for t in COLORS]
    dirs=[]
    for j,words in enumerate(({"left","leftward","leftwards"},{"right","rightward","rightwards"},{"up","upward","upwards","upper","top"},{"down","downward","downwards","bottom","lower"})):
        if words.intersection(tokens):dirs.append(j)
    role=np.zeros((4,4),np.float32)
    for c in range(4):role[c]=[c==source,c==destination,mentions[c],float("arm" in tokens or "effector" in tokens)]
    return source,destination,role,(dirs[0] if len(dirs)==1 else -1)


def tracked(start,end):
    # Solve the two-instance correspondence within color. Unmatched or shape-inconsistent components are unmeasurable.
    displacement=np.zeros((4,2),np.float32);valid=np.zeros(4,bool)
    for c in range(4):
        a=start[c][start[c,:,5]>.5].astype(np.float32);b=end[c][end[c,:,5]>.5].astype(np.float32)
        if not len(a) or len(a)!=len(b):continue
        def cost(order):
            z=b[list(order)];area=np.abs(np.log(z[:,4]/a[:,4]))
            return float((np.linalg.norm((z[:,:2]-a[:,:2])*[1,.5625],axis=1)+.18*area).sum())
        idx=min(itertools.permutations(range(len(b))),key=cost);z=b[list(idx)]
        if np.any(np.abs(np.log(z[:,4]/a[:,4]))>.60):continue
        diff=(z[:,:2]-a[:,:2])*[320.,180.]
        dist=np.linalg.norm(diff,axis=1)
        if dist.max()>140:continue
        valid[c]=True;displacement[c]=diff[int(np.argmax(dist))]
    return displacement,valid


def outcome(geo,t):
    # Independent geometry must persist on each of the last two images.
    a,m=tracked(geo[t],geo[-1]);b,k=tracked(geo[t],geo[-2]);mask=m&k
    dist=np.linalg.norm(a,axis=1);other=np.linalg.norm(b,axis=1)
    y=-1
    if mask.sum()>=2:
        order=np.argsort(np.where(mask,dist,-1));top=order[-1]
        if dist[top]>=6 and other[top]>=5 and dist[order[-2]]<=.62*dist[top] and np.linalg.norm(a[top]-b[top])<max(5,.4*dist[top]):
            y=int(top+1)
        if mask.all() and max(dist.max(),other.max())<2.5:y=0
    return y,a,mask


def split_groups(parts):
    parent={}
    def find(a):
        parent.setdefault(a,a)
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    image_owner={}
    for file in parts:
        records=json.loads(file.with_suffix(".json").read_bytes())
        with np.load(file) as z:frames=z["frame"]
        for r in records:
            p=r["parent"];find(p)
            for ind in np.unique(frames[r["start"]:r["start"]+r["length"]]):
                old=image_owner.setdefault(bytes(ind),p)
                if old!=p:union(old,p)
    grouping={p:find(p) for p in parent}
    roles={p:("train" if (n:=int(rank("split/"+p)[:12],16)%100)<70 else "dev" if n<85 else "test") for p in set(grouping.values())}
    return grouping,roles,len(image_owner)


def build(root,external=False):
    raw=root/"raw";folder=root/"decoded";folder.mkdir(exist_ok=True,parents=True)
    paths=[raw/r["name"].split("/")[-1] for r in json.loads((raw/"selection.json").read_bytes())]
    with ProcessPoolExecutor(6) as pool:list(pool.map(decode_shard,[(p,folder) for p in paths]))
    parts=sorted(folder.glob("*.npz"));grouping,roles,nframe=split_groups(parts)
    excluded_groups=set()
    if external:
        prior=sorted((R/"decoded").glob("*.npz"))
        if root.name=="robot-prospective":prior+=sorted((R.parent/"robot-extra/decoded").glob("*.npz"))
        old_keys=set();old_frames=set()
        for old in prior:
            old_keys.update(r["parent"] for r in json.loads(old.with_suffix(".json").read_bytes()))
            with np.load(old) as z:old_frames.update(bytes(k) for k in z["frame"])
        for p in parts:
            records=json.loads(p.with_suffix(".json").read_bytes())
            with np.load(p) as z:fps=z["frame"]
            for r in records:
                if r["parent"] in old_keys or any(bytes(k) in old_frames for k in fps[r["start"]:r["start"]+r["length"]]):
                    excluded_groups.add(grouping[r["parent"]])
        roles={g:"test" for g in roles}
    groups=sorted(roles);gidx={k:i for i,k in enumerate(groups)}
    episodes=[];features={role:defaultdict(list) for role in ("train","dev","test")};seen=set();excluded=Counter()
    by_parent=defaultdict(list)
    for file in parts:
        for r in json.loads(file.with_suffix(".json").read_bytes()):by_parent[grouping[r["parent"]]].append(r)
    for file in parts:
        with np.load(file) as z:cache={k:z[k] for k in ("px","geom","x","u","frame")}
        for r in json.loads(file.with_suffix(".json").read_bytes()):
            parent=grouping[r["parent"]];split=roles[parent];start=r["start"];n=r["length"]
            if parent in excluded_groups:excluded["overlapping_external_group"]+=1;continue
            x=cache["x"][start:start+n];u=cache["u"][start:start+n];g=cache["geom"][start:start+n]
            if n<7 or n>250:excluded["duration"]+=1;continue
            goal=r["goal"];src,dst,bind,dr=semantics(goal);novel=(src,dst) in NEW_PAIRS
            if novel and split=="train":excluded["reserved_composition_training"]+=1;continue
            unique=(cache["frame"][start], cache["frame"][start+n-1],n,goal)
            if unique in seen:excluded["identical_episode_and_goal"]+=1;continue
            seen.add(unique)
            first=2;present=g[first,:,:,5].max(1)>.5
            pool=[]
            for other in by_parent[parent]:
                so,do,_,_=semantics(other["goal"])
                if other["goal"]==goal or src<0 or so<0 or so==src or not (present[src] and present[so]):continue
                if split=="train" and (so,do) in NEW_PAIRS:continue
                pool.append((abs(len(other["goal"].split())-len(goal.split())),rank(r["episode"]+"/"+other["episode"]),other["goal"]))
            wrong=min(pool)[2] if pool else "";_,_,wbind,wdir=semantics(wrong)
            eid=len(episodes);episodes.append(dict(**r,group=parent,role=split,novel=novel,wrong=wrong,source=src,destination=dst,direction=dr,wrong_direction=wdir))
            arr=features[split];clock=[first]+list(range(4,n-4,3));clock=clock[:28]
            for t in clock:
                ix=np.maximum(0,np.arange(t-3,t+1));prev=ix-1
                past=np.where((prev>=0)[:,None],u[np.maximum(prev,0)],0)
                cur=x[ix];delta=np.diff(cur,axis=0,prepend=cur[:1])
                geo=g[ix];y,flow,vis=outcome(g,t) if t==first else (-1,np.zeros((4,2)),np.zeros(4,bool))
                plan=u[t:t+4];future=x[t+1:t+5]-x[t]
                if len(plan)!=4 or len(future)!=4:raise ValueError("Short physical forecast")
                row=dict(px=cache["px"][start+ix],geo=geo,kin=np.concatenate((cur,delta,past),1),plan=plan,future=future,
                        goal=text_features(goal),wrong=text_features(wrong) if wrong else np.zeros(128,np.int8),
                        bind=bind,wb=wbind,direction=dr,wrong_direction=wdir,mover=y,flow=flow,visible=vis,first=t==first,
                        paired=bool(wrong),novel=novel,episode=eid,tick=t,group=gidx[parent],endpoint=x[-1]-x[t])
                for k,v in row.items():arr[k].append(v)
    bank=root/"banks";bank.mkdir(exist_ok=True)
    counts={}
    for role,vals in features.items():
        if not vals:continue
        d={k:np.asarray(v) for k,v in vals.items()}
        for k in ("kin","plan","future","flow","endpoint","bind","wb"):d[k]=d[k].astype(np.float32)
        for k in ("group","episode","tick","mover","direction","wrong_direction"):d[k]=d[k].astype(np.int32)
        file=bank/(role+".npz");tmp=file.with_suffix(".new.npz");np.savez_compressed(tmp,**d);tmp.replace(file)
        counts[role]=dict(rows=len(d["tick"]),episodes=int(d["first"].sum()),groups=len(np.unique(d["group"])),sha256=file_hash(file),
            paired_goals=int((d["paired"]&d["first"]).sum()),measurable=int((d["mover"]>=0).sum()),novel=int((d["novel"]&d["first"]).sum()))
    write(bank/"episodes.json",episodes);write(bank/"data.json",dict(groups=groups,roles=roles,merged_parent_ids=len(grouping)-len(groups),unique_frames=nframe,excluded_groups=sorted(excluded_groups),excluded=excluded,counts=counts))
    print("Frozen robot banks",counts,flush=True);(bank/"ready").touch()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=R);p.add_argument("--external",action="store_true");a=p.parse_args();build(a.root,a.external)
