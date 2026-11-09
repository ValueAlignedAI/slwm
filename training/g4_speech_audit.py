"""Fixed 32-clip, eight-speaker diagnostic of speech content in the generated waveform."""
import argparse
import json
from pathlib import Path

import numpy as np


def predict(data,out,root):
    import jax
    import jax.numpy as jnp
    from models.g4_shared import encode
    from training.g4_eval import weights
    ok=np.load(data/"test/ok.npy",mmap_mode="r"); groups=np.load(data/"test/group.npy")
    fam=np.load(data/"test/family.npy"); uid=np.load(data/"test/uid.npy"); rows=np.flatnonzero((fam==0)&(ok.sum(1)==64))
    selected=np.concatenate([g[np.argsort(uid[g])[:4]] for name in np.unique(groups[rows]) for g in [rows[groups[rows]==name]]])
    if len(selected)!=32: raise ValueError("Expected four recordings from each of eight held-out people")
    batch={k:np.array(np.load(data/f"test/{k}.npy",mmap_mode="r")[selected]) for k in ("a","v","n","x","ok","family")}
    mels={"codec_reference":batch["a"].astype(np.float32)/16-12}
    tags=["g4-v5e-pilot-gpt2-cross-81","g4-v5e-pilot-signal-cross-81",
          "g4-v5e-fair-all-medium-gpt2-cross-81","g4-v5e-fair-all-medium-gpt2-motion-81"]
    for tag in tags:
        _,_,spec,p,stats=weights(root/tag)
        @jax.jit
        def fn(params,b,source):
            z,_=encode(params,spec,b,jnp.full((len(selected),),source,jnp.int32),stats)
            return (z@params["readout"].astype(z.dtype))[:,:,:100].astype(jnp.float32)*3-5
        for name,source in (("text",3),("video",1)):
            mels[tag+"__"+name]=np.clip(np.array(fn(p,batch,np.int32(source))),-12,4)
    np.savez_compressed(out/"speech-predictions.npz",uid=uid[selected],**mels)


def score(repo,out):
    import hashlib
    import torch
    import torch.nn.functional as F
    import torchaudio
    from evals.motion_audio_intelligibility import word_error
    from training.stream_codec import load_vocoder
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    torch.set_num_threads(4)
    rev="87c7102498dcde7456f24cfd30239ca606ed9063"; name="openai/whisper-tiny.en"
    proc=WhisperProcessor.from_pretrained(name,revision=rev)
    model=WhisperForConditionalGeneration.from_pretrained(name,revision=rev).eval()
    codec=load_vocoder("cpu")
    raw=json.loads((repo/"artifacts/t2_audio_visual/grid_content_v1/test.json").read_bytes())
    index={int.from_bytes(hashlib.sha256(r["sample_id"].encode()).digest()[:8],"little"):r for r in raw}
    with np.load(out/"speech-predictions.npz") as data:
        rows=[index[int(k)] for k in data["uid"]]; refs=[r["sentence"] for r in rows]
        waveforms={}
        clean=[]
        for r in rows:
            with np.load(repo/r["decoded_path"]) as source: pcm=source["pcm"].astype(np.float32)/32768
            duration=min(r["frames"]/25,len(pcm)/16000)
            start=int(max(0,int((duration/(1024/24000)-64)/2))*(1024/24000)*16000)
            clean.append(pcm[start:start+round(64*1024/24000*16000)])
        waveforms["original_pcm"]=clean
        with torch.inference_mode():
            for key in data.files:
                if key=="uid":continue
                records=[]
                for i in range(0,len(rows),4):
                    coarse=torch.from_numpy(data[key][i:i+4].astype(np.float32)).transpose(1,2)
                    fine=F.interpolate(coarse,scale_factor=4,mode="linear",align_corners=False)
                    signal=codec.decode(torch.cat([fine,fine[:,:,-1:]],axis=2)).clamp(-1,1)
                    if signal.shape[1]!=64*1024: raise ValueError("External renderer clock mismatch")
                    records.extend(torchaudio.functional.resample(signal,24000,16000).numpy())
                waveforms[key]=records
    results={}
    for cond,records in waveforms.items():
        guesses=[]
        for i in range(0,len(rows),8):
            inputs=proc(records[i:i+8],sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():
                ids=model.generate(**inputs,max_new_tokens=32,num_beams=1,do_sample=False)
            guesses.extend(proc.batch_decode(ids,skip_special_tokens=True))
        scores=[word_error(r,g) for r,g in zip(refs,guesses)]
        results[cond]={"mean_clip_wer":float(np.mean(scores)),
                       "by_speaker":{actor:float(np.mean([v for v,r in zip(scores,rows) if r["actor"]==actor])) for actor in sorted({r["actor"] for r in rows})}}
        print(cond,results[cond]["mean_clip_wer"],flush=True)
    (out/"speech-content.json").write_text(json.dumps({"clips":len(rows),"speakers":8,"words_per_clip":6,"recognizer":name,
        "revision":rev,"source_selection":"four lowest fixed sample hashes per held-out actor with a complete 64-tick window",
        "conditions":results},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["predict","score"]); p.add_argument("--data",type=Path)
    p.add_argument("--repo",type=Path); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="predict":predict(a.data,a.out,a.out)
    else:score(a.repo,a.out)
