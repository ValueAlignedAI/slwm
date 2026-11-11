"""Reserve a new natural speech cohort, and assemble independent language-only demonstration supervision."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import zipfile
import numpy as np
from g10_holo import one,session_group
from g11_events import G10,G11,FOLDS,family,annotate
from g12_events import make_rows,balanced,write
from g12_banks import packed_record

PRIOR=Path("/mnt/slwm-data/SLWM/g12/data")
ROOT=Path("/mnt/slwm-data/SLWM/g13/data/holo")


def sources(root,component=False):
    root.mkdir(parents=True,exist_ok=True)
    for sub in ("fresh/media","fresh/shards","vision-nv","banks"):(root/sub).mkdir(parents=True,exist_ok=True)
    if not (root/"whisper-small").exists():(root/"whisper-small").symlink_to(G11/"whisper-small",target_is_directory=True)
    full={r["video_name"]:r for r in json.loads((G10/"labels.json").read_bytes())}
    used={session_group(n) for p in (G11/"annotations.json",PRIOR/"annotations.json") for n in json.loads(p.read_bytes())}
    if component:used.update(session_group(n) for n in json.loads((ROOT/"annotations.json").read_bytes()))
    used|={session_group(v["name"]) for rows in json.loads((G10/"banks.json").read_bytes()).values() for v in rows}
    with zipfile.ZipFile(G10/"split.zip") as z:
        allowed={v.decode().strip().casefold() for role in ("train","val") for v in z.read(role+"-v1_2.txt").splitlines()}
    idx={k:json.loads((G10/(k+"-index.json")).read_bytes())["members"] for k in ("video_pitch_shifted","imu")}
    files=defaultdict(dict)
    for items in idx.values():
        for p in items:
            k={"video_pitchshift.mp4":"av","accelerometer_sync.txt":"acc","gyroscope_sync.txt":"gyro"}.get(p.split("/")[-1].casefold())
            if k:files[p.split("/")[0].casefold()][k]=p
    fresh={}
    fold_def={"A":("coffee","switch"),"B":("printer","camera")} if component else FOLDS
    for fam in (f for fs in fold_def.values() for f in fs):
        pool=[r for n,r in full.items() if n.casefold() in allowed and session_group(n) not in used and family(r["taskType"])==fam
              and len(files[n.casefold()])==3 and r["videoMetadata"]["duration"]["seconds"]>=38]
        fresh[fam]=balanced(pool,min(16 if component else 12,len({session_group(v["video_name"]) for v in pool})),used)
    names=[n for ns in fresh.values() for n in ns]
    cat={n:annotate(full[n]) for n in names}
    rec={n:dict(family=cat[n]["family"],duration=cat[n]["duration"],task=cat[n]["task"],rows=make_rows(full[n],cat[n])) for n in names}
    write(root/"groups.json",{f:[n for fam in fams for n in fresh[fam]] for f,fams in fold_def.items()})
    write(root/"annotations.json",cat);write(root/"records.json",rec)
    with ThreadPoolExecutor(7) as pool:
        for i,_ in enumerate(pool.map(lambda n:one(root/"fresh",full[n],idx,files),names),1):
            if i%8==0:print("Fresh HoloAssist",i,flush=True)
    write(root/"fresh/complete.json",fresh)
    if component:return
    more={}
    for n,r in full.items():
        if n.casefold() not in allowed or session_group(n) in used:continue
        ann=annotate(r);lookup={x["id"]:x["text"] for x in ann["speech"]}
        examples=[dict(text=lookup[a["cue"]],verbs=a["verbs"]) for a in make_rows(r,ann) if a["cue"]>=0]
        if examples:more[n]=dict(family=ann["family"],examples=examples)
    write(root/"extra-language.json",more);print("Additional independent transcript recordings",len(more),flush=True)


def build(root):
    groups=json.loads((root/"groups.json").read_bytes());rec=json.loads((root/"records.json").read_bytes());cat=json.loads((root/"annotations.json").read_bytes())
    for fold,names in groups.items():
        vocab=json.loads((PRIOR/"banks"/(fold+"-train.json")).read_bytes())["vocabulary"];ids={v:i for i,v in enumerate(vocab)}
        data=[];index=[]
        for i,n in enumerate(names):
            for row,meta in packed_record(root,n,rec[n],cat[n],ids,root/"speech-small"):
                if row["cued"]:row["group"]=i;data.append(row);index.append(meta)
        p=root/"banks"/(fold+"-test.npz");arrays={k:np.stack([v[k] for v in data]) for k in data[0]}
        np.savez_compressed(p,**arrays)
        from g12_banks import sha
        write(p.with_suffix(".json"),dict(rows=index,names=names,vocabulary=vocab,sha256=sha(p),examples=len(data)))
        print("Fresh fold",fold,len(names),len(data),flush=True)
    (root/"banks/ready").touch()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("sources","build"));p.add_argument("--component",action="store_true");a=p.parse_args()
    root=ROOT.parent/"components" if a.component else ROOT
    sources(root,a.component) if a.stage=="sources" else build(root)
