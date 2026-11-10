"""Extract synchronized mel bands, CTC-timed words and frozen acoustic features from authentic recordings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from training.g4_data import number,quant,tokens,digest

DT=1024/24000
ROLE={"train":"train","validation":"dev","test":"test"}
KEYS=("a","v","x","group","uid","length")


def records(root,repo,family):
    result={k:[] for k in ("train","dev","test")}
    if family=="libri":
        with gzip.open(root/"libri/rows.jsonl.gz","rt") as f:
            for line in f:
                row=json.loads(line); path=root/"libri/audio"/row["speaker"]/(row["id"]+".flac")
                result[ROLE[row["role"]]].append((row["id"],row["speaker"],row["transcript"],str(path),""))
    else:
        base=repo/"artifacts/t2_audio_visual/cremad_motion_v3"
        for split in ROLE:
            for row in json.loads((base/(split+".json")).read_bytes()):
                result[ROLE[split]].append((row["sample_id"],row["actor"],row["transcript"],str(repo/row["raw_path"]),str(base/"decoded"/(row["sample_id"]+".npz"))))
    return {k:sorted(v,key=lambda r:(Path(r[3]).stat().st_size//16384,r[0])) for k,v in result.items()}


def read_clip(row):
    uid,person,label,path,vis=row
    if not vis:
        import soundfile as sf
        wave,sr=sf.read(path,dtype="float32")
        if sr!=16000 or wave.ndim!=1:raise ValueError("Unexpected speech source schema")
        return wave,None
    clock=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=codec_type,start_time","-of","json",path]))
    starts={s["codec_type"]:float(s.get("start_time",0)) for s in clock["streams"]}
    raw=subprocess.check_output(["ffmpeg","-v","error","-threads","1","-i",path,"-vn","-ac","1","-ar","16000","-f","f32le","pipe:1"])
    wave=np.frombuffer(raw,"<f4").copy(); gap=round((starts.get("audio",0)-starts.get("video",0))*16000)
    if gap>0:wave=np.pad(wave,(gap,0))
    elif gap<0:wave=wave[-gap:]
    with np.load(vis,allow_pickle=False) as f:
        v=f["mouth"];n,h,w=v.shape
        frames=np.rint(v.reshape(n,16,h//16,16,w//16).mean((2,4))).astype(np.uint8).reshape(n,256)
    return wave,frames


def aligned(logits,text,vocab,fa):
    import torch
    words=re.findall("[A-Z']+",text.upper()); coded="|".join(words)
    if not coded or any(ch not in vocab for ch in coded):raise ValueError("Unsupported transcript")
    ids=[vocab[ch] for ch in coded]
    path,score=fa(logits[None].float().log_softmax(-1).cpu(),torch.tensor([ids]),blank=0)
    arr=path[0].numpy(); starts=np.r_[0,np.flatnonzero(arr[1:]!=arr[:-1])+1]; ends=np.r_[starts[1:],len(arr)]
    emitted=[(int(arr[l]),int(l),int(h)) for l,h in zip(starts,ends) if arr[l]!=0]
    if [j for j,_,_ in emitted]!=ids:raise ValueError("CTC path did not reproduce transcript")
    spans=[]; at=0
    for word in words:
        spans.append((emitted[at][1]*.02,emitted[at+len(word)-1][2]*.02,word));at+=len(word)+1
    return spans


def assemble(folder,target,fresh,role,family):
    parts=sorted(folder.glob("*.npz")); shapes={}; total=0
    for file in parts:
        with np.load(file,allow_pickle=False) as f:
            if len(f["uid"]) and not shapes:shapes={k:(v.dtype,v.shape[1:]) for k,v in f.items()}
            total+=len(f["uid"])
    if total<16:raise ValueError("Insufficient valid recordings")
    target.parent.mkdir(exist_ok=True,parents=True)
    if not (target/"card.json").exists():
        with tempfile.TemporaryDirectory(dir=target.parent,prefix=".speech-") as t:
            dest=Path(t)/"pack";dest.mkdir()
            arrays={k:np.lib.format.open_memmap(dest/(k+".npy"),mode="w+",dtype=typ,shape=(total,*shape)) for k,(typ,shape) in shapes.items()}
            pos=0
            for file in parts:
                with np.load(file,allow_pickle=False) as f:
                    n=len(f["uid"])
                    if n:
                        for k in shapes:arrays[k][pos:pos+n]=f[k]
                    pos+=n
            for a in arrays.values():a.flush()
            if len(np.unique(arrays["uid"]))!=total:raise ValueError("Duplicate recording identities")
            record={"family":family,"role":role,"rows":total,"people":len(np.unique(arrays["group"])),"sha256":{k:digest(dest/(k+".npy")) for k in shapes}}
            (dest/"card.json").write_text(json.dumps(record,sort_keys=True,indent=2)+"\n");dest.rename(target)
    original=json.loads((target/"card.json").read_bytes())
    fresh.mkdir(exist_ok=True,parents=True)
    for k in KEYS:
        s=target/(k+".npy");d=fresh/(k+".npy")
        if s.is_file() and not d.exists(): os.link(s,d)
    original["sha256"]={k:v for k,v in original["sha256"].items() if k in KEYS}
    tmp=fresh/"card.partial";tmp.write_text(json.dumps(original,sort_keys=True,indent=2)+"\n");tmp.replace(fresh/"card.json")
    print("Packed",family,role,total,flush=True)


def run(root,repo):
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC
    from vocos.feature_extractors import MelSpectrogramFeatures
    torch.set_num_threads(2); torch.manual_seed(711)
    teacher=json.loads((root/"teacher.json").read_bytes())
    if teacher["revision"]!="22aad52d435eb6dbaf354bdad9b0da84ce7d6156":raise ValueError("Acoustic teacher changed")
    net=Wav2Vec2ForCTC.from_pretrained(teacher["snapshot"],local_files_only=True).eval().half().cuda()
    vocab=json.loads((Path(teacher["snapshot"])/"vocab.json").read_bytes())
    feat=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100).eval().cuda()
    proj=torch.tensor(np.random.default_rng(711).standard_normal((768,32)).astype(np.float32)/np.sqrt(768),dtype=torch.float32,device="cuda")
    # Train roles are available immediately for the continuation. Held roles are never used to fit units.
    corpora={family:records(root,repo,family) for family in ("libri","crema")}
    for family,role in (("libri","train"),("crema","train"),("libri","dev"),("crema","dev"),("libri","test"),("crema","test")):
        rows=corpora[family][role]; target=root/"speech"/family/role; fresh=root/"fresh"/family/role
        if (fresh/"card.json").is_file():continue
        chunks=root/"speech-shards"/family/role;chunks.mkdir(exist_ok=True,parents=True)
        for first in range(0,len(rows),96):
            file=chunks/f"{first:07d}.npz"
            if file.exists():continue
            group=rows[first:first+96]; kept=[]; failures=[]
            with ThreadPoolExecutor(max_workers=4) as pool: streams=list(pool.map(read_clip,group))
            for j in range(0,len(group),8):
                part=group[j:j+8]; media=streams[j:j+8]
                waves=[torch.from_numpy(v[0]) for v in media]
                length=torch.tensor([len(x) for x in waves],device="cuda")
                pcm=torch.nn.utils.rnn.pad_sequence(waves,batch_first=True).cuda()
                mask=torch.arange(pcm.shape[1],device="cuda")[None]<length[:,None]
                mu=pcm.sum(1)/length; var=(((pcm-mu[:,None])**2)*mask).sum(1)/length
                norm=(pcm-mu[:,None])*(var[:,None]+1e-7).rsqrt()*mask
                with torch.inference_mode():
                    out=net(norm.half(),attention_mask=mask.long(),output_hidden_states=True)
                    projected=torch.nn.functional.layer_norm(out.hidden_states[6].float(),(768,))@proj
                    hidden=projected.cpu().numpy(); log=out.logits.float().cpu(); lens=net._get_feat_extract_output_lengths(length).cpu().tolist()
                    mel=feat(torchaudio.functional.resample(pcm,16000,24000)).transpose(1,2).cpu().numpy()
                for k,(uid,person,txt,path,vis) in enumerate(part):
                    try:spans=aligned(log[k,:lens[k]],txt,vocab,torchaudio.functional.forced_align)
                    except (RuntimeError,ValueError) as e:failures.append([uid,str(e)[:120]]);continue
                    duration=len(media[k][0])/16000
                    if vis:duration=min(duration,len(media[k][1])/25)
                    ticks=int(duration/DT); size=min(64,ticks)
                    if size<8:failures.append([uid,"short source"]);continue
                    maxstart=ticks-size
                    choice=[max(0,int(l/DT)) for l,h,w in spans if 0<=int(l/DT)<=maxstart]
                    start=choice[number("g7-speech/"+uid)%len(choice)] if choice else 0
                    l=start*DT;h=(start+size)*DT
                    phrase=" ".join(w for s,e,w in spans if s>=l-.015 and e<=h+.015)
                    if not phrase:failures.append([uid,"no enclosed annotation"]);continue
                    fine=np.zeros((64,4,100),np.uint8);a=np.zeros((64,100),np.uint8);z=np.zeros((64,2,32),np.float16)
                    acoustic=mel[k,start*4:(start+size)*4].reshape(size,4,100)
                    fine[:size]=quant(acoustic); a[:size]=quant(acoustic.mean(1))
                    centers=(start+np.arange(size)[:,None]+np.array([.25,.75])[None])*DT
                    indices=np.clip(np.rint((centers-.0125)/.02).astype(int),0,lens[k]-1)
                    z[:size]=hidden[k,indices].astype(np.float16)
                    x=np.zeros(64,np.uint16); ids=tokens(phrase);x[:len(ids)]=ids
                    entry={"a":a,"fine":fine,"h":z,"x":x,"uid":np.uint64(number(family+"/"+uid)),"group":np.uint64(number(family+"/"+person)),"length":np.uint8(size),
                           "start":np.int32(start),"id":np.asarray(uid,dtype="<U64"),"phrase":np.asarray(phrase,dtype="<U320")}
                    if vis:
                        vid=np.zeros((64,256),np.uint8);at=np.minimum(((start+np.arange(size)+.5)*DT*25).astype(int),len(media[k][1])-1);vid[:size]=media[k][1][at];entry["v"]=vid
                    kept.append(entry)
            raw={k:np.stack([r[k] for r in kept]) for k in kept[0]} if kept else {"uid":np.empty(0,np.uint64)}
            with file.with_suffix(".part").open("wb") as f:np.savez(f,**raw)
            file.with_suffix(".part").replace(file)
            if failures:file.with_suffix(".excluded.json").write_text(json.dumps(failures))
            if first%960==0:print(family,role,first,len(rows),len(kept),flush=True)
        assemble(chunks,target,fresh,role,family)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--repo",type=Path,required=True);a=p.parse_args();run(a.root,a.repo)
