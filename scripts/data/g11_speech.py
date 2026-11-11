"""Extract instruction-only acoustics and blind automatic transcripts from the released microphone track."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from data.g10_language import text_features

REV="87c7102498dcde7456f24cfd30239ca606ed9063"
G10=Path("/mnt/slwm-data/SLWM/g10/data/holo")

def tokens(s):return re.findall(r"[a-z0-9']+",s.casefold())

def edits(source,guess):
    old=list(range(len(guess)+1))
    for i,a in enumerate(source):
        row=[i+1]
        for j,b in enumerate(guess):row.append(min(row[-1]+1,old[j+1]+1,old[j]+int(a!=b)))
        old=row
    return old[-1]

def audio(root,name,rec):
    path=G10/"media"/(name+"-av.mp4")
    if not path.exists():path=root/"fresh/media"/(name+"-av.mp4")
    buf=subprocess.check_output(["ffmpeg","-nostdin","-v","error","-threads","1","-i",str(path),"-t","600","-map","0:a:0","-ac","1","-ar","16000","-f","f32le","pipe:1"],timeout=180)
    pcm=np.frombuffer(buf,"<f4")
    ids={v[key] for v in rec["events"] for key in ("cue","wrong")}
    output=[]
    for cue in rec["speech"]:
        if cue["id"] not in ids:continue
        l=max(0,int(np.ceil(cue["start"]*16000)));r=min(len(pcm),int(np.floor(cue["end"]*16000)))
        if r-l<1600:raise ValueError("Missing recorded instruction segment")
        output.append((cue,pcm[l:r].copy()))
    return name,output

def pool_time(x,count=16):
    # The fixed slot count prevents duration or absolute scene clocks from identifying the intervention.
    edges=np.linspace(0,len(x),count+1).astype(int)
    return np.stack([x[min(a,len(x)-1):max(min(a,len(x)-1)+1,b)].mean(0) for a,b in zip(edges[:-1],edges[1:])])

def main(root,recognizer="tiny"):
    revision=REV if recognizer=="tiny" else "e8727524f962ee844a7319d92be39ac1bd25655a"
    target=root/("speech" if recognizer=="tiny" else "speech-small");target.mkdir(exist_ok=True)
    catalog=json.loads((root/"annotations.json").read_text());source=Path.home()/".cache/huggingface/hub/models--openai--whisper-tiny.en/snapshots"/REV if recognizer=="tiny" else root/"whisper-small"
    proc=WhisperProcessor.from_pretrained(str(source),local_files_only=True)
    net=WhisperForConditionalGeneration.from_pretrained(str(source),local_files_only=True,dtype=torch.float16).to("cuda").eval()
    projection=np.linalg.qr(np.random.default_rng(4173).standard_normal((net.config.d_model,128)))[0].astype(np.float32)
    wanted={n for n,v in catalog.items() if v["events"]}
    with ThreadPoolExecutor(4) as workers,torch.inference_mode():
        while wanted:
            present=sorted(n for n in wanted if (target/(n+".json")).exists() or (G10/"media"/(n+"-av.mp4")).exists() or (root/"fresh/media"/(n+"-av.mp4")).exists())
            if not present:time.sleep(5);continue
            ready=[n for n in present if not (target/(n+".json")).exists()][:12]
            for n in present:
                if (target/(n+".json")).exists():wanted.remove(n)
            if not ready:continue
            items=list(workers.map(lambda n:audio(root,n,catalog[n]),ready))
            clips=[(n,cue,pcm) for n,rows in items for cue,pcm in rows];outputs={n:[] for n in ready}
            batch=32 if recognizer=="tiny" else 16
            for pos in range(0,len(clips),batch):
                chunk=clips[pos:pos+batch];wave=[v[2] for v in chunk]
                inp=proc.feature_extractor(wave,sampling_rate=16000,return_attention_mask=True,return_tensors="pt",device="cuda")
                features=inp.input_features.to("cuda",dtype=torch.float16);mask=inp.attention_mask.to("cuda")
                encoded=net.model.encoder(features,return_dict=True)
                hidden=encoded.last_hidden_state.float().cpu().numpy()
                generated=net.generate(input_features=features,encoder_outputs=encoded,attention_mask=mask,max_new_tokens=96,num_beams=1,do_sample=False)
                hypotheses=proc.batch_decode(generated,skip_special_tokens=True)
                for j,(n,cue,pcm) in enumerate(chunk):
                    h=hidden[j,:max(1,min(1500,int(np.ceil(len(pcm)/320))))]
                    vec=pool_time(h)@projection
                    # Fixed numerical scale, no held-set normalization.
                    vec=np.clip(vec,-20,20).astype(np.float16)
                    hyp=hypotheses[j].strip();ref=cue["text"]
                    outputs[n].append((cue["id"],vec,text_features(hyp),text_features(ref),hyp,edits(tokens(ref),tokens(hyp)),len(tokens(ref))))
            for n in ready:
                rows=outputs[n];stamp=target/(n+".npz");tmp=target/(n+".tmp.npz")
                np.savez_compressed(tmp,ids=np.asarray([v[0] for v in rows],np.int32),acoustic=np.stack([v[1] for v in rows]),
                    asr=np.stack([v[2] for v in rows]),oracle=np.stack([v[3] for v in rows]))
                tmp.replace(stamp);note=dict(revision=revision,sha256=hashlib.sha256(stamp.read_bytes()).hexdigest(),utterances={str(v[0]):dict(hypothesis=v[4],edits=v[5],words=v[6]) for v in rows})
                f=target/(n+".json");temp=f.with_suffix(".new");temp.write_text(json.dumps(note));temp.replace(f);wanted.remove(n)
            print("speech recordings",len(list(target.glob("*.npz"))),"remaining",len(wanted),flush=True)
    (target/"complete").touch()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--recognizer",choices=("tiny","small"),default="tiny");a=p.parse_args();main(a.root,a.recognizer)
