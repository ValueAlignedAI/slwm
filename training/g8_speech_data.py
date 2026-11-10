"""Prepare predetermined fresh-speaker waveform cases with the original G7 physical clock and teacher."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np
from training.g4_data import digest,number,quant,tokens
from training.g7_speech import DT,read_clip,aligned
from training.g7_units import put
from training.t2_prepare_cremad import SENTENCES

G7=Path("/mnt/slwm-data/SLWM/g7");ROOT=Path("/mnt/slwm-data/SLWM/g8/speech")
MD5="fb5a50374b501bb3bac4815ee91d3135"
SALT="g8-fixed-speech-81211/"


def sources():
    folder=ROOT/"sources";folder.mkdir(parents=True,exist_ok=True);archive=folder/"test-other.tar.gz"
    if not archive.exists():
        tmp=archive.with_suffix(".partial")
        subprocess.run(["curl","-fLsS","--retry","3","--max-time","1000","https://www.openslr.org/resources/12/test-other.tar.gz","-o",str(tmp)],check=True)
        with tmp.open("rb") as f:actual=hashlib.file_digest(f,"md5").hexdigest()
        if actual!=MD5:raise ValueError("Official LibriSpeech MD5 mismatch")
        tmp.replace(archive)
    with archive.open("rb") as f:
        if hashlib.file_digest(f,"md5").hexdigest()!=MD5:raise ValueError("Held source archive changed")
    if not (folder/"extracted").exists():
        with tarfile.open(archive,"r:gz") as t:
            for m in t:
                parts=Path(m.name).parts
                if parts[:2]!=("LibriSpeech","test-other"):continue
                if ".." in parts or not (m.isdir() or m.isfile()):raise ValueError("Unexpected held source archive member")
                t.extract(m,folder,filter="data")
        (folder/"extracted").touch()
    libri=[]
    for index in sorted((folder/"LibriSpeech/test-other").glob("*/*/*.trans.txt")):
        for line in index.read_text().splitlines():
            uid,text=line.split(" ",1);libri.append((uid,uid.split("-")[0],text,str(index.parent/(uid+".flac")),""))
    raw=G7/"physical-time/crema/files";crema=[]
    for path in sorted(raw.glob("*.flv")):
        actor,phrase=path.stem.split("_")[:2]
        crema.append((path.stem,actor,SENTENCES[phrase],str(path),str(raw/(path.stem+".mouth.npz"))))
    out={}
    for family,items in (("libri",libri),("crema",crema)):
        group=sorted({r[1] for r in items});prior=set(np.load(G7/"speech"/family/"train/group.npy").tolist())
        if any(number(family+"/"+p) in prior for p in group):raise ValueError("Speaker overlaps the G7 fitting population")
        selected=[r for g in group for r in sorted([v for v in items if v[1]==g],key=lambda x:number(SALT+x[0]))[:4]]
        if len(selected)!=4*len(group):raise ValueError("Four predetermined clips per speaker required")
        out[family]=selected
    put(folder/"selection.json",{"salt":SALT,"archive_md5":MD5,"archive_sha256":digest(archive),"license":"LibriSpeech CC BY 4.0; CREMA-D ODbL / DbCL",
        "clips":{k:[list(row[:3]) for row in vals] for k,vals in out.items()}})
    return out


def prepare():
    import torch
    import torchaudio
    from transformers import Wav2Vec2ForCTC
    from vocos.feature_extractors import MelSpectrogramFeatures
    sets=sources();teacher=json.loads((G7/"teacher.json").read_bytes())
    if teacher["revision"]!="22aad52d435eb6dbaf354bdad9b0da84ce7d6156":raise ValueError("Acoustic clock teacher changed")
    net=Wav2Vec2ForCTC.from_pretrained(teacher["snapshot"],local_files_only=True).cuda().half().eval()
    vocab=json.loads((Path(teacher["snapshot"])/"vocab.json").read_bytes())
    feat=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100).eval().cuda()
    proj=torch.tensor(np.random.default_rng(711).standard_normal((768,32)).astype(np.float32)/np.sqrt(768),dtype=torch.float32,device="cuda")
    for family,rows in sets.items():
        folder=ROOT/"confirm"/family;folder.mkdir(parents=True,exist_ok=True)
        if (folder/"card.json").exists():continue
        arrays={};excluded=[]
        for row in rows:
            uid,person,transcript,*_=row;wave,video=read_clip(row)
            t=torch.from_numpy(wave).cuda();norm=(t-t.mean())*torch.rsqrt(t.var(unbiased=False)+1e-7)
            with torch.inference_mode():
                y=net(norm[None].half(),output_hidden_states=True);h=(torch.nn.functional.layer_norm(y.hidden_states[6].float(),(768,))@proj)[0].cpu().numpy()
                mel=feat(torchaudio.functional.resample(t[None],16000,24000))[0].T.cpu().numpy()
            try:words=aligned(y.logits[0].float().cpu(),transcript,vocab,torchaudio.functional.forced_align)
            except (ValueError,RuntimeError) as e:excluded.append([uid,str(e)[:140]]);continue
            ticks=min(int(len(wave)/16000/DT),len(mel)//4)
            if video is not None:ticks=min(ticks,int(len(video)/25/DT))
            length=min(64,ticks);latest=ticks-length
            if length<8:excluded.append([uid,"short physical window"]);continue
            choices=[max(0,int(t0/DT)) for t0,_,_ in words if 0<=int(t0/DT)<=latest]
            start=choices[number("g7-speech/"+uid)%len(choices)] if choices else 0
            low=start*DT;high=(start+length)*DT
            phrase=" ".join(w for t0,t1,w in words if t0>=low-.015 and t1<=high+.015)
            if not phrase:excluded.append([uid,"no enclosed words"]);continue
            fine=np.zeros((64,4,100),np.uint8);a=np.zeros((64,100),np.uint8);hidden=np.zeros((64,2,32),np.float16);x=np.zeros(64,np.uint16)
            chunk=mel[start*4:(start+length)*4].reshape(length,4,100)
            fine[:length]=quant(chunk);a[:length]=quant(chunk.mean(1));code=tokens(phrase);x[:len(code)]=code
            centers=(start+np.arange(length)[:,None]+np.array([.25,.75])[None])*DT
            hidden[:length]=h[np.clip(np.rint((centers-.0125)/.02).astype(int),0,len(h)-1)].astype(np.float16)
            pcm=np.zeros(43691,np.float32);crop=wave[round(low*16000):round(high*16000)];pcm[:min(len(crop),len(pcm))]=crop[:len(pcm)]
            one={"a":a,"fine":fine,"h":hidden,"x":x,"pcm":pcm,"length":np.uint8(length),"id":uid,"phrase":phrase,
                 "uid":np.uint64(number(family+"/"+uid)),"group":np.uint64(number(family+"/"+person))}
            if video is not None:
                v=np.zeros((64,256),np.uint8);v[:length]=video[np.minimum(((start+np.arange(length)+.5)*DT*25).astype(int),len(video)-1)];one["v"]=v
            for k,value in one.items():arrays.setdefault(k,[]).append(value)
        for k,val in arrays.items():np.save(folder/(k+".npy"),np.stack(val),allow_pickle=False)
        put(folder/"card.json",{"family":family,"source_rows":len(rows),"eligible":len(arrays["id"]),"people":len(set(arrays["group"])),"excluded":excluded,
            "selection":digest(ROOT/"sources/selection.json"),"teacher_revision":teacher["revision"],"sha256":{k:digest(folder/(k+".npy")) for k in arrays}})
        print("Speech confirmation cases",family,len(arrays["id"]),len(set(arrays["group"])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("sources","prepare"));a=p.parse_args()
    if a.phase=="sources":sources()
    else:prepare()
