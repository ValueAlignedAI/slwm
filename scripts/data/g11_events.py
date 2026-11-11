"""Lock source groups, prospective spoken decisions and earlier same-recording decoys."""
import argparse
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile
import numpy as np
from g10_holo import one,session_group

G10=Path("/mnt/slwm-data/SLWM/g10/data/holo")
G11=Path("/mnt/slwm-data/SLWM/g11/data")
FOLDS={"A":("coffee","computer"),"B":("printer","camera")}
KEYWORDS={"grab":("grab","grasp","take","pick"),"place":("place","put","set","sit"),"insert":("insert","plug","slot","push in"),
          "withdraw":("remove","unplug","withdraw","take out"),"screw":("screw","tighten","fasten"),"unscrew":("unscrew","loosen"),
          "rotate":("rotate","twist","turn"),"open":("open",),"close":("close","shut"),"press":("press","click"),"lift":("lift","raise"),
          "slide":("slide",),"pull":("pull",),"push":("push",),"align":("align","line up"),"flip":("flip",),"lock":("lock",),"unlock":("unlock",),"pour":("pour",)}
STOP=set("the a an you your that this there then now it is of to and on in with just can so yeah okay like gonna".split())

def rank(s):return hashlib.sha256(("G11/"+str(s)).encode()).hexdigest()

def family(task):
    for k in ("coffee","computer","printer"):
        if k in task:return k
    if "camera" in task or "gopro" in task:return "camera"
    return task.replace("disassemble ","").replace("assemble ","").replace("setup ","").replace("change ","").replace("fix ","")

def words(text):return re.findall(r"[a-z0-9']+",str(text).casefold())

def speech_events(record,limit):
    speech=[];all_ranges=[]
    for e in sorted(record["events"],key=lambda x:(float(x["end"]),int(x["id"]))):
        if e["label"]!="Conversation":continue
        start,end=float(e["start"]),float(e["end"])
        if not 0<=start<end:continue
        all_ranges.append([start,end])
        a=e["attributes"];purpose=str(a.get("Conversation Purpose","")).casefold()
        if not purpose.startswith("instructor-") or not any(k in purpose for k in ("instruction","correct the wrong action")):continue
        text=str(a.get("Transcription","")).strip();tokens=words(text)
        if not text or len(tokens)<2 or "unintelligible" in text.casefold() or "low-confidence" in str(a.get("Transcription Confidence","")).casefold():continue
        if end>limit or end-start>.0+28 or end-start<.20:continue
        content=set(tokens)-STOP
        if not content:continue
        explicit=sorted(k for k,aliases in KEYWORDS.items() if any(re.search(r"\b"+re.escape(x)+r"\b"," ".join(tokens)) for x in aliases))
        speech.append(dict(id=int(e["id"]),start=start,end=end,text=text,purpose=purpose.split("_")[-1],tokens=tokens,content=sorted(content),explicit=explicit))
    return speech,all_ranges

def decoy(cur,previous):
    eligible=[];a=set(cur["content"])
    for old in previous:
        if old["end"]>cur["start"]-.1:continue
        b=set(old["content"]);j=len(a&b)/max(1,len(a|b))
        ratio=len(cur["tokens"])/len(old["tokens"])
        if j>=.8 or not .4<=ratio<=2.5:continue
        conflict=bool(cur["explicit"] and old["explicit"] and set(cur["explicit"]).isdisjoint(old["explicit"]))
        cost=(old["purpose"]!=cur["purpose"],not conflict,abs(math.log(ratio))-.22*j,rank(str(cur["id"])+"/"+str(old["id"])))
        eligible.append((cost,old,conflict))
    if not eligible:return None,False
    _,choice,flag=min(eligible,key=lambda x:x[0])
    return choice,flag

def annotate(rec):
    duration=min(600,float(rec["videoMetadata"]["duration"]["seconds"]))
    speech,ranges=speech_events(rec,duration)
    acts=[]
    for e in rec["events"]:
        if e["label"]!="Fine grained action":continue
        a=e["attributes"];v=str(a.get("Verb","")).strip().casefold();noun=str(a.get("Noun","")).strip().casefold()
        st,en=float(e["start"]),float(e["end"])
        if v in ("","none") or not 0<=st<en<=duration:continue
        acts.append(dict(id=int(e["id"]),start=st,end=en,verb=v,noun=noun or "<unknown>"))
    acts.sort(key=lambda a:(a["start"],a["id"]));starts=[a["start"] for a in acts]
    picked={};exclusions=Counter()
    for i,ins in enumerate(speech):
        cut_tick=int(math.ceil(ins["end"]*10))-1;cut=(cut_tick+1)/10
        j=bisect_left(starts,ins["end"])
        if cut_tick<31 or j==len(acts):exclusions["prefix_or_no_following_action"]+=1;continue
        nxt=acts[j]
        if not cut+.20-1e-8<=nxt["start"]<=cut+8:exclusions["guard_or_gap"]+=1;continue
        if j+1<len(acts) and acts[j+1]["start"]-nxt["start"]<.1:exclusions["ambiguous_onset"]+=1;continue
        if any(nxt["start"]>q["end"]>ins["end"] for q in speech[i+1:]):exclusions["superseded"]+=1;continue
        alt,conflict=decoy(ins,speech[:i])
        if alt is None:exclusions["no_prior_distinct_decoy"]+=1;continue
        # The next onset determines the outcome. A previous completed annotation is reserved for the persistence reference.
        before=[a for a in acts if a["end"]<=cut]
        prior=max(before,key=lambda a:a["end"]) if before else None
        ev=dict(**nxt,cut=cut_tick,cue=ins["id"],wrong=alt["id"],conflict=conflict,last_verb=prior["verb"] if prior else "<none>",
                last_noun=prior["noun"] if prior else "<none>",gap=nxt["start"]-cut,
                isolated=not any(a["id"]!=nxt["id"] and a["start"]<nxt["end"]+.2 and a["end"]>nxt["start"]+.15 for a in acts))
        if nxt["id"] in picked:exclusions["duplicate_action"]+=1
        picked[nxt["id"]]=ev
    events=sorted(picked.values(),key=lambda a:a["start"])
    for a in events:a["physical"]=False
    pool=[a for a in events if a["end"]+.50<duration and a["start"]-.7>0]
    for a in sorted(pool,key=lambda a:rank(rec["video_name"]+"/depth/"+str(a["id"])))[:8]:a["physical"]=True
    background=[]
    for a in sorted(acts,key=lambda a:rank(rec["video_name"]+"/background/"+str(a["id"]))):
        tick=int(math.floor(a["start"]*10))-3
        if tick<31 or a["id"] in picked:continue
        background.append({**a,"cut":tick,"cue":-1,"wrong":-1,"physical":False})
        if len(background)==64:break
    return dict(name=rec["video_name"],task=rec["taskType"],family=family(rec["taskType"]),duration=duration,
                speech=speech,conversation=ranges,events=events,background=background,excluded=dict(exclusions))

def main(root,fresh):
    root.mkdir(exist_ok=True,parents=True);(root/"fresh/media").mkdir(exist_ok=True,parents=True);(root/"fresh/shards").mkdir(exist_ok=True,parents=True)
    data=json.loads((G10/"labels.json").read_text());base=json.loads((G10/"banks.json").read_text())
    by_name={r["video_name"]:r for r in data}
    available={x["name"] for rows in base.values() for x in rows};used={session_group(n) for n in available}
    with zipfile.ZipFile(G10/"split.zip") as z:allowed={v.decode().strip().casefold() for role in ("train","val") for v in z.read(role+"-v1_2.txt").splitlines()}
    idx={k:json.loads((G10/(k+"-index.json")).read_text())["members"] for k in ("video_pitch_shifted","imu")}
    files={}
    for k,d in idx.items():
        for path in d:
            leaf=path.split("/")[-1].casefold()
            key={"video_pitchshift.mp4":"av","accelerometer_sync.txt":"acc","gyroscope_sync.txt":"gyro"}.get(leaf)
            if key:files.setdefault(path.split("/")[0].casefold(),{})[key]=path
    chosen={}
    for fam in ("coffee","computer","printer","camera"):
        eligible=[r for r in data if family(r["taskType"])==fam and session_group(r["video_name"]) not in used and r["video_name"].casefold() in allowed and len(files.get(r["video_name"].casefold(),{}))==3 and r["videoMetadata"]["duration"]["seconds"]>=38]
        tasks=sorted({r["taskType"] for r in eligible});pools={t:sorted([r for r in eligible if r["taskType"]==t],key=lambda r:rank(r["video_name"]),reverse=True) for t in tasks};names=[]
        while len(names)<24:
            count=len(names)
            for t in tasks:
                while pools[t]:
                    n=pools[t].pop()["video_name"];g=session_group(n)
                    if g not in used:used.add(g);names.append(n);break
                if len(names)==24:break
            if len(names)==count:raise ValueError("Incomplete unseen family: "+fam)
        chosen[fam]=names
    src={k:[x["name"] for x in rows] for k,rows in base.items()}
    groups={}
    for fold,held in FOLDS.items():
        train=[n for n in src["train"]+src["novel"] if family(by_name[n]["taskType"]) not in held]
        train=sorted(train,key=rank)[:256]
        groups[fold]=dict(train=train,dev=[n for n in src["dev"] if family(by_name[n]["taskType"]) not in held],
            familiar=[n for n in src["familiar"] if family(by_name[n]["taskType"]) not in ("coffee","computer","printer","camera")],
            held=sorted([n for fam in held for n in chosen[fam]],key=rank))
        if len(train)!=256 or len(groups[fold]["held"])!=48:raise ValueError("Incorrect fold cardinality")
    names=sorted(set(n for r in groups.values() for l in r.values() for n in l))
    catalog={n:annotate(by_name[n]) for n in names}
    (root/"groups.json").write_text(json.dumps(dict(folds=groups,fresh=chosen),sort_keys=True,indent=2)+"\n")
    (root/"annotations.json").write_text(json.dumps(catalog,separators=(",",":"))+"\n")
    print("locked source groups",len(names),"fresh",sum(map(len,chosen.values())),"cued",sum(len(x["events"]) for x in catalog.values()),flush=True)
    if fresh:
        new=[n for l in chosen.values() for n in l]
        with ThreadPoolExecutor(8) as pool:notes=list(pool.map(lambda n:one(root/"fresh",by_name[n],idx,files),new))
        (root/"fresh/complete.json").write_text(json.dumps(notes,sort_keys=True)+"\n")
        print("new recordings",len(notes),sum(r["ticks"] for r in notes),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--fresh",action="store_true");a=p.parse_args();main(a.root,a.fresh)
