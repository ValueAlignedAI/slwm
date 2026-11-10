"""Test fixed payload assignment and conditional coefficient forecasts on newly reserved recordings."""
from pathlib import Path
import numpy as np
import torch
from training.g7_align_cache import mask_for
from training.g7_alignment_audit import Synthesis,words,REV
from training.g8_acoustic_codec import OUT,codebook,encode,decode
from training.g8_rate_codec import HELD,OUT_RATE,BUDGETS,load_book,coefficients,quantize,mel,verify_clip_payload
from training.g8_rate_heads import ContinuousHead,HEADS
from training.g8_speech_score import TAGS,result,summarize
from training.g8_models import read
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha,lock
from training.g7_units import put

RESULT=OUT.parent/"rate-audit"


def freeze():
    pin=sha(OUT_RATE/"weights.npz");heads={k:sha(HEADS/k/"weights.pt") for k in TAGS}
    for k,h in heads.items():
        d=read(HEADS/k/"complete.json")
        if d["weights"]!=h or d["codec"]!=pin or d["updates"]!=2048:raise ValueError("Incorrect rate-head license to score")
    doc={"allocator":pin,"predictors":heads,"product":sha(OUT/"codec.npz"),"recognizer":REV,
         "sources":{f:sha(HELD/"confirm"/f/"card.json") for f in ("libri","crema")},
         "fields":{t+"/"+f:sha(HELD/"fields"/t/f/"card.json") for t in TAGS for f in ("libri","crema")},
         "code":{f:sha(Path(f)) for f in ("training/g8_rate_codec.py","training/g8_rate_score.py","training/g8_rate_heads.py","training/g8_speech_fields.py")}}
    for disk in roots("g8"):
        p=disk/"rate-audit";p.mkdir(exist_ok=True);lock(p/"lock.json",doc)
    return doc


def interpolate(coarse,truth,spans,lengths):
    out=truth.copy()
    for i,(blank,n) in enumerate(zip(spans,lengths)):
        ix=np.flatnonzero(blank[:n]);known=np.flatnonzero(~blank[:n])
        if not len(ix) or not len(known):continue
        ticks=(ix[:,None]*4+np.arange(4)).ravel()
        for band in range(100):
            out[i,ticks,band]=np.interp(ticks,known*4+1.5,coarse[i,known,band])
    return out


def evaluate():
    doc=freeze();b=load_book();product=codebook();synth=Synthesis(Path("/home/db/dev/slwm-g3"),roots("g7")[0]);torch.set_num_threads(3)
    with torch.inference_mode():
        for family in ("libri","crema"):
            folder=HELD/"confirm"/family;src={k:np.load(folder/(k+".npy"),mmap_mode="r") for k in read(folder/"card.json")["sha256"]}
            n=len(src["uid"]);refs=src["phrase"].tolist();groups=np.array(src["group"]);length=src["length"].astype(int)
            counts=np.array([len(words(s)) for s in refs]);pc=coefficients(src["fine"],b)
            truth=src["fine"].reshape(n,256,100).astype(np.float32)/16-12
            mask=mask_for(src["uid"],length);gap=np.repeat(mask,4,axis=1)[...,None]
            ref=RESULT/family/"oracles";ref.mkdir(parents=True,exist_ok=True)
            if not (ref/"scores.json").is_file():
                scores={"original":result(synth,[np.array(x[:round(int(t)*1024/24000*16000)]) for x,t in zip(src["pcm"],length)],refs,groups,ref,"original")}
                def measure(name,values):
                    scores[name]=result(synth,synth.waves(values,length),refs,groups,ref,name)
                measure("fine",truth);measure("continuous-pca",mel(pc,b,n));measure("product48",decode(encode(src["fine"],product),product,8))
                measure("raw-gap-interpolation",interpolate(src["a"].astype(np.float32)/16-12,truth,mask,length))
                payload_sizes={}
                for budget in BUDGETS:
                    for recipe in ("equal","allocated"):
                        tag=recipe+str(budget);alloc=b[tag];codes,rest=quantize(pc,b,alloc)
                        # The first actual clip also traverses the byte-level stream interface.
                        count=2*int(length[0]);blob,actual=verify_clip_payload(codes[:count],b,alloc)
                        np.testing.assert_array_equal(actual,rest[:count]);(ref/(tag+".bin")).write_bytes(blob)
                        payload_sizes[tag]={"bytes":len(blob),"frame_pairs":count,"payload_bits":count*budget,"padding_bits":8*len(blob)-count*budget}
                        measure(tag,mel(rest,b,n))
                def diff(k,l):return summarize(np.asarray(scores[k]["per_clip_edits"])-scores[l]["per_clip_edits"],counts,groups)
                effects={str(k):diff("allocated"+str(k),"equal"+str(k)) for k in BUDGETS}
                put(ref/"scores.json",{"conditions":scores,"allocated_minus_equal":effects,"allocated48_minus_product":diff("allocated48","product48"),
                    "primary_passed_this_corpus":effects["48"]["speaker_ci95"][1]<0,"payloads":payload_sizes,"source":doc["sources"][family],"ids":src["id"].tolist(),"references":refs})
            baseline=read(ref/"scores.json")["conditions"]["raw-gap-interpolation"]["per_clip_edits"]
            for tag in TAGS:
                target=RESULT/family/tag;target.mkdir(parents=True,exist_ok=True)
                if (target/"scores.json").exists():continue
                net=ContinuousHead().cuda().eval();net.load_state_dict(torch.load(HEADS/tag/"weights.pt",weights_only=True,map_location="cuda"))
                features=HELD/"fields"/tag/family;cv=read(features/"card.json");wrong=np.load(features/"donor.npy");scores={};effects={}
                for mode in cv["views"]:
                    kind={0:"masked-audio",1:"video",3:"text"}[mode]
                    for cond in ("correct","wrong","erased"):
                        f=features/f"z{mode}-{cond}.npy"
                        if sha(f)!=cv["sha256"][f.name]:raise ValueError("Changed frozen observation")
                        z=np.load(f,mmap_mode="r")
                        pred=np.concatenate([net(torch.tensor(np.array(z[i:i+32],np.float32),device="cuda")).cpu().numpy() for i in range(0,n,32)]).reshape(-1,64)*b["std"]
                        _,picked=quantize(pred,b,b["allocated48"]);y=mel(picked,b,n);key=kind+"-"+cond
                        scores[key]=result(synth,synth.waves(y,length),refs,groups,target,key)
                        if mode==0:scores["repair-"+cond]=result(synth,synth.waves(np.where(gap,y,truth),length),refs,groups,target,"repair-"+cond)
                    for name in ([kind,"repair"] if mode==0 else [kind]):
                        for ctrl in ("wrong","erased"):
                            e=np.asarray(scores[name+"-correct"]["per_clip_edits"])-scores[name+"-"+ctrl]["per_clip_edits"]
                            effects[name+"-minus-"+ctrl]=summarize(e,counts,groups,wrong>=0 if ctrl=="wrong" else None)
                effects["repair-minus-raw-interpolation"]=summarize(np.asarray(scores["repair-correct"]["per_clip_edits"])-baseline,counts,groups)
                put(target/"scores.json",{"conditions":scores,"paired_wer_effects":effects,"head":doc["predictors"][tag],"ids":src["id"].tolist()})
                print("Held bit-allocation/conditioning result",family,tag,flush=True)


if __name__=="__main__":evaluate()
