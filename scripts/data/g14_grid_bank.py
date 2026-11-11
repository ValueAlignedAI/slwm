"""Prepare independently partitioned full utterances, aligned training targets and crossed speaker quartets."""
import hashlib
import itertools
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile
import threading
import numpy as np
from data.g14_grid import acoustic,mouth_field,labels,SLOTS,T
from g14_runtime import ROOT,json_file
OLD=Path("/mnt/slwm-data/SLWM/data/grid_content_v1")
DATA=ROOT/"data/utterances"
thread=threading.local()

def rank(t):return hashlib.sha256(("G14/GRID/"+str(t)).encode()).hexdigest()

def parse_archive(path):
    result={}
    with tarfile.open(path) as a:
        for m in a:
            if not m.isfile() or not m.name.endswith(".align"):continue
            spans=[(int(x)/25000,int(y)/25000,w) for x,y,w in [l.split() for l in a.extractfile(m).read().decode("ascii").splitlines()] if w not in ("sil","sp")]
            if len(spans)!=6:continue
            txt=" ".join(w for _,_,w in spans);result[Path(m.name).stem]=(txt,spans,int(hashlib.sha256(txt.encode()).hexdigest()[:12],16)%10)
    return result

def decode_mp(path):
    cmd=["ffmpeg","-nostdin","-v","error","-threads","1","-i",str(path)]
    pcm=np.frombuffer(subprocess.check_output(cmd+["-map","0:a:0","-vn","-ac","1","-ar","16000","-f","s16le","pipe:1"],timeout=45),"<i2")
    vf="setpts=PTS-STARTPTS,fps=25,crop=iw*0.34:ih*0.28:iw*0.31:ih*0.60,scale=96:64"
    raw=subprocess.check_output(cmd+["-an","-vf",vf,"-frames:v","80","-pix_fmt","gray","-threads","1","-f","rawvideo","pipe:1"],timeout=45)
    pix=np.frombuffer(raw,"u1").reshape(-1,64,96)
    clocks=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=codec_type,start_time","-of","json",str(path)],timeout=20))
    starts={v["codec_type"]:float(v.get("start_time",0.0)) for v in clocks["streams"]};trim=round((starts["video"]-starts["audio"])*16000)
    pcm=pcm[trim:] if trim>=0 else np.pad(pcm,(-trim,0))
    return pcm,pix

def one(item):
    person,k,text,spans,role,source,crc=item;cache=OLD/"decoded"/f"s{person:02d}"/(k+".npz")
    if cache.is_file():
        with np.load(cache,allow_pickle=False) as f:pcm,pix=f["pcm"],f["mouth"]
        raw=OLD/"raw"/f"s{person:02d}"/(k+".mpg")
    else:
        raw=DATA/"raw"/f"s{person:02d}"/(k+".mpg");raw.parent.mkdir(parents=True,exist_ok=True)
        if not raw.exists():
            if not hasattr(thread,"zips"):thread.zips={}
            if source not in thread.zips:thread.zips[source]=zipfile.ZipFile(source)
            z=thread.zips[source];members={Path(m.filename).stem.split("_")[-1]:m for m in z.infolist() if m.filename.endswith(".mpg")}
            member=members[k];assert member.CRC==crc
            partial=raw.with_suffix(".part");partial.write_bytes(z.read(member));partial.replace(raw)
        pcm,pix=decode_mp(raw)
    if len(pix)<50 or len(pcm)<28000:return None
    sound,n=acoustic(pcm);lip=mouth_field(pix)
    spoken=[v for v in spans if v[2] not in ("sil","sp")]
    times=np.arange(T)*.032+.016
    aligns=np.stack([(times>=spoken[h][0])&(times<spoken[h][1]) for h in (1,3,4)])
    if not aligns.any(1).all():return None
    return dict(a=sound,l=lip,length=n,alignment=aligns,y=labels(text),speaker=person),dict(person=person,key=k,text=text,raw=str(raw),source=str(source),crc=crc,role=role)

def manifest():
    result=[];block=set()
    for old,role in (("train","train"),("validation","dev"),("calibration","cal")):
        data=json.loads((OLD/(old+".json")).read_bytes())
        for r in data:
            n=int(r["actor"][-2:]);k=r["sample_id"].split("-")[-1];txt=r["sentence"]
            path=OLD/"sources"/f"s{n:02d}"/f"s{n}.mpg_vcd.zip"
            result.append((n,k,txt,r["spans"],role,path,r["zip_member_crc32"]));block.add(txt)
    for n in range(25,35):
        base=OLD/"sources"/f"s{n:02d}" if n<=32 else ROOT/"data/grid-new"
        path=base/f"s{n}.mpg_vcd.zip";align=parse_archive(base/f"s{n}.tar")
        with zipfile.ZipFile(path) as z:
            members={Path(m.filename).stem.split("_")[-1]:m for m in z.infolist() if m.filename.endswith(".mpg")}
            for k,(txt,spans,bucket) in sorted(align.items()):
                if bucket not in (8,9) or txt in block or k not in members:continue
                result.append((n,k,txt,spans,"held" if n<=32 else "fresh",path,members[k].CRC))
    return result

def quartets(rows,role):
    by={}
    for i,r in enumerate(rows):
        if r["role"]!=role:continue
        c,l,d=labels(r["text"]);c,l,d=map(int,(c,l,d))
        if c>=3:continue
        by.setdefault(r["person"],{}).setdefault((c,l,d),[]).append(i)
    out={}
    for p,lookup in by.items():
        opts=[]
        for c1,c2 in itertools.combinations(range(3),2):
            ac={(l,d) for c,l,d in lookup if c==c1}&{(l,d) for c,l,d in lookup if c==c2}
            for a,b in itertools.combinations(sorted(ac),2):
                if a[0]==b[0] or a[1]==b[1]:continue
                ids=[min(lookup[(c,*k)],key=lambda i:rank(rows[i]["key"])) for c,k in ((c1,a),(c1,b),(c2,a),(c2,b))]
                opts.append(dict(speaker=p,colors=[c1,c2],addresses=[a,b],clips=ids))
        out[str(p)]=sorted(opts,key=lambda v:rank(str(v)))[:32]
    return out

def main():
    DATA.mkdir(exist_ok=True,parents=True);all_items=manifest();meta=[]
    for role in ("train","dev","cal","held","fresh"):
        items=[v for v in all_items if v[4]==role];arrays={};start=len(meta)
        with ThreadPoolExecutor(8) as pool:
            for record in pool.map(one,items):
                if record is None:continue
                data,r=record;meta.append(r)
                for k,v in data.items():arrays.setdefault(k,[]).append(v)
        f=DATA/(role+".npz");np.savez_compressed(f,**{k:np.asarray(v) for k,v in arrays.items()},global_index=np.arange(start,len(meta),dtype="i4"))
        print("Prepared GRID",role,len(meta)-start,flush=True)
    qs={s:quartets(meta,s) for s in ("dev","cal","held","fresh")}
    json_file(DATA/"media.json",meta);json_file(DATA/"quartets.json",qs)
    groups={r:{q["person"] for q in meta if q["role"]==r} for r in ("train","dev","cal","held","fresh")}
    assert all(a.isdisjoint(b) for a,b in itertools.combinations(groups.values(),2))
    tr={r["text"] for r in meta if r["role"] in ("train","dev","cal")}
    assert tr.isdisjoint(r["text"] for r in meta if r["role"] in ("held","fresh"))
    out=dict(clips={role:sum(r["role"]==role for r in meta) for role in groups},people={k:sorted(v) for k,v in groups.items()},
        quartets={k:{p:len(v) for p,v in d.items()} for k,d in qs.items()},source_manifest=hashlib.sha256((DATA/"media.json").read_bytes()).hexdigest())
    json_file(DATA/"data.json",out);print("Qualified within-speaker quartets",out["quartets"],flush=True)

if __name__=="__main__":main()
