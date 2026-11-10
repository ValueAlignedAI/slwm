"""Score decoded waveforms on a sealed cohort with speaker-paired controls and an actual codec ceiling."""
import argparse
from pathlib import Path
import numpy as np
import torch
from training.g7_alignment_audit import Synthesis,words,edits,dump_wave,REV
from training.g7_units import put
from training.g8_acoustic_codec import DATA,OLD,OUT,codebook,encode,decode
from training.g8_acoustic_head import HEADS,Head
from training.g8_models import read
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import lock,sha

TAGS=[f"{c}-{s}" for s in (711,712) for c in ("hybrid","gpt2")]
SCORES=OUT.parent/"speech-audit"


def freeze():
    d={"codec":sha(OUT/"codec.npz"),"heads":{},"features":{},"panels":{},"recognizer":REV,"source_code":{}}
    for tag in TAGS:
        complete=read(HEADS/tag/"complete.json");w=sha(HEADS/tag/"weights.pt")
        if complete["weights"]!=w or complete["codec"]!=d["codec"] or complete["updates"]!=2048:raise ValueError("Unregistered acoustic head")
        d["heads"][tag]=w
        for family in ("crema","libri"):
            panel=DATA/"confirm"/family;src=DATA/"fields"/tag/family;card=read(src/"card.json")
            if card["parent"]!=complete["parent"] or card["source"]!=sha(panel/"card.json"):raise ValueError("Wrong held feature sources")
            d["features"][tag+"/"+family]=sha(src/"card.json")
            d["panels"][family]=sha(panel/"card.json")
    for f in ("training/g8_speech_score.py","training/g8_speech_fields.py","training/g8_acoustic_head.py","training/g8_acoustic_codec.py"):d["source_code"][f]=sha(Path(f))
    for disk in roots("g8"):
        root=disk/"speech-audit";root.mkdir(exist_ok=True);lock(root/"lock.json",d)
    return d


def summarize(by_clip,counts,groups,valid=None):
    if valid is not None:by_clip,counts,groups=by_clip[valid],counts[valid],groups[valid]
    people,ix=np.unique(groups,return_inverse=True);e=np.bincount(ix,weights=by_clip);n=np.bincount(ix,weights=counts)
    picks=np.random.default_rng(81293).integers(len(people),size=(4000,len(people)))
    values=e[picks].sum(1)/n[picks].sum(1)
    return {"word_error":float(e.sum()/n.sum()),"words":int(n.sum()),"edits":int(e.sum()),"clips":len(ix),"speakers":len(people),
            "speaker_ci95":np.quantile(values,[.025,.975]).tolist()}


def result(synth,rec,refs,groups,folder,condition):
    hyp=synth.predict(rec);err=np.array([edits(words(r),words(h)) for r,h in zip(refs,hyp)]);num=np.array([len(words(r)) for r in refs])
    dump_wave(folder/(condition+".wav"),rec[0])
    return {**summarize(err,num,groups),"per_clip_edits":err.tolist(),"hypotheses":hyp}


def semantic_codes(hidden):
    with np.load(OLD/"units/codebook.npz") as f:
        x=(hidden.astype(np.float32)-f["mean"])/f["scale"];centers=f["centers"]
    return (np.sum(centers**2,axis=1)-2*x.reshape(-1,32)@centers.T).argmin(1).reshape(len(x),64,2)


def evaluate():
    pins=freeze();SCORES.mkdir(exist_ok=True);book=codebook()
    synth=Synthesis(Path("/home/db/dev/slwm-g3"),roots("g7")[0]);torch.set_num_threads(3)
    with torch.inference_mode():
        for family in ("libri","crema"):
            path=DATA/"confirm"/family
            src={k:np.load(path/(k+".npy"),mmap_mode="r") for k in read(path/"card.json")["sha256"]}
            refs=src["phrase"].tolist();groups=np.asarray(src["group"]);length=src["length"].astype(int);n=len(refs)
            words_n=np.array([len(words(r)) for r in refs])
            truth=src["fine"].reshape(n,256,100).astype(np.float32)/16-12;oracle=encode(src["fine"],book)
            constant=np.broadcast_to(book["counts"].argmax(-1),oracle.shape);from training.g7_align_cache import mask_for
            mask=mask_for(src["uid"],length);gap=np.repeat(mask,4,axis=1)[...,None]
            refdir=SCORES/family/"oracles";refdir.mkdir(parents=True,exist_ok=True)
            if not (refdir/"scores.json").exists():
                scores={}
                pcm=[np.asarray(v[:round(int(k)*1024/24000*16000)]) for v,k in zip(src["pcm"],length)]
                scores["original"]=result(synth,pcm,refs,groups,refdir,"original")
                sem=semantic_codes(src["h"]);mel=synth.synth(sem,synth.style)
                conds={"fine":truth,"semantic-oracle":mel,"constant":decode(constant,book),"known-audio-constant-gap":np.where(gap,decode(constant,book),truth)}
                conds.update({f"acoustic-oracle-{k}":decode(oracle,book,k) for k in (1,4,8)})
                for name,y in conds.items():scores[name]=result(synth,synth.waves(y,length),refs,groups,refdir,name)
                semerr=np.asarray(scores["semantic-oracle"]["per_clip_edits"])
                contrast={str(k):summarize(np.asarray(scores[f"acoustic-oracle-{k}"]["per_clip_edits"])-semerr,words_n,groups) for k in (1,4,8)}
                put(refdir/"scores.json",{"conditions":scores,"acoustic_minus_semantic_wer":contrast,"refs":refs,"ids":src["id"].tolist(),
                    "speakers":list(map(str,groups)),"codec_sha256":pins["codec"],"recognizer":REV,"word_errors_include_insertions":True})
            for tag in TAGS:
                folder=SCORES/family/tag;folder.mkdir(parents=True,exist_ok=True)
                if (folder/"scores.json").exists():continue
                net=Head().cuda().eval();net.load_state_dict(torch.load(HEADS/tag/"weights.pt",map_location="cuda",weights_only=True))
                field=DATA/"fields"/tag/family;card=read(field/"card.json");wrong=np.load(field/"donor.npy");scores={};effects={};units={}
                for mode in card["views"]:
                    name={0:"masked-audio",1:"video",3:"text"}[mode]
                    for condition in ("correct","wrong","erased"):
                        f=field/f"z{mode}-{condition}.npy"
                        if sha(f)!=card["sha256"][f.name]:raise ValueError("Held state file changed")
                        z=np.load(f,mmap_mode="r");codes=np.concatenate([net(torch.as_tensor(np.array(z[j:j+32],np.float32),device="cuda")).argmax(-1).cpu().numpy() for j in range(0,n,32)])
                        m=decode(codes,book);label=name+"-"+condition
                        keep=np.arange(64)[None]<length[:,None]
                        if mode==0:keep&=mask
                        units[label]=(codes==oracle)[keep].mean((0,1)).tolist()
                        scores[label]=result(synth,synth.waves(m,length),refs,groups,folder,label)
                        if mode==0:
                            key="repair-"+condition;scores[key]=result(synth,synth.waves(np.where(gap,m,truth),length),refs,groups,folder,key)
                    for trial in ([name,"repair"] if mode==0 else [name]):
                        for ctrl in ("wrong","erased"):
                            delta=np.asarray(scores[trial+"-correct"]["per_clip_edits"])-np.asarray(scores[trial+"-"+ctrl]["per_clip_edits"])
                            effects[trial+"-minus-"+ctrl]=summarize(delta,words_n,groups,wrong>=0 if ctrl=="wrong" else None)
                put(folder/"scores.json",{"head":pins["heads"][tag],"conditions":scores,"paired_wer_effects":effects,"per_group_code_top1":units,
                      "ids":src["id"].tolist(),"codec_sha256":pins["codec"]})
                print("Confirmed waveform panel",family,tag,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("lock","score"));a=p.parse_args()
    if a.phase=="lock":freeze()
    else:evaluate()
