"""Source substitution tests on real paired development records and fixed-clip speech recognition."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def group_ci(values,groups):
    _,ix=np.unique(groups,return_inverse=True)
    k=ix.max()+1
    sums=np.bincount(ix,weights=values)
    counts=np.bincount(ix)
    rng=np.random.default_rng(5124)
    drawn=rng.integers(0,k,(1000,k))
    avg=sums[drawn].sum(1)/counts[drawn].sum(1)
    return np.quantile(avg,[.025,.975]).tolist()


def predict(root,data_dir,out):
    import jax
    from models.g4_shared import Spec
    from models.g5_residual import world
    from training.g4_train import write,sha_file
    from training.g5_batches import JointData
    out.mkdir(parents=True,exist_ok=True)
    saved={}
    metrics={}
    for core in ("hybrid","gpt2"):
        run=root/f"g5-medium-{core}-511-residual-gpu"
        spec=Spec(**json.loads((run/"config.json").read_bytes())["spec"])
        if sha_file(run/"inference.npz")!=json.loads((run/"complete.json").read_bytes())["sha256"]["inference.npz"]:
            raise ValueError("Generation weights changed")
        with np.load(run/"inference.npz") as f:
            stats=np.array(f["stats"])
            p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
            cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")}
        data=JointData(data_dir,"dev",stats=stats)
        @jax.jit
        def infer(b):
            shared,_,_=world(p,spec,b,b["source"],cal)
            return shared[:,:,:356]
        def gather(family,indices,src):
            b=data.gather(np.full(len(indices),family),indices,np.zeros(len(indices),int))
            b["source"][:,src]=1
            return b
        scores={}
        cases=((7,0,1),(7,1,0),(0,0,1),(0,1,0),(0,3,0),(0,3,1),(1,3,0),(2,3,1))
        for family,src,target in cases:
            arr=data.packs[1 if family==7 else 0]
            ids=data.pools[family]
            if family!=7:
                ids=ids[(arr["ok"][ids].sum(1)==64)&arr["has"][ids,src].astype(bool)&arr["has"][ids,target].astype(bool)]
            groups=arr["group"][ids]
            identity=arr["recording"][ids] if family==7 else (arr["cap"][ids] if src==3 else arr["uid"][ids])
            rng=np.random.default_rng(5151+10*family+src)
            valid=[]; wrong=[]
            for k,ix in enumerate(ids):
                allowed=np.flatnonzero((identity!=identity[k])&((groups==groups[k]) if family!=2 else True))
                if len(allowed):
                    valid.append(k); wrong.append(ids[rng.choice(allowed)])
            ids=ids[valid]; groups=groups[valid]
            if not len(ids):
                continue
            wrong=np.asarray(wrong)
            conds=["correct","different","erased"]+(["delayed"] if src<2 else [])
            names={0:"audio",1:"video",3:"text"}
            family_name={7:"impact",0:"grid",1:"libri",2:"coco"}[family]
            key=f"{family_name}-{names[src]}-to-{names[target]}"
            rows={cond:[[],[]] for cond in conds}
            for j in range(0,len(ids),32):
                take=ids[j:j+32]; size=len(take)
                pad=lambda x:np.pad(x,(0,32-len(x)),mode="edge")
                original=gather(family,pad(take),src)
                other=gather(family,pad(wrong[j:j+32]),src)
                sensor={0:"a",1:"v",3:"input_x"}[src]
                target_map=(original["a"].astype(np.float32)/48-7/3) if target==0 else (original["v"].astype(np.float32)/127.5-1)
                for cond in conds:
                    b=original.copy()
                    if cond=="different":
                        b[sensor]=other[sensor]
                    if cond=="erased":
                        b["sense"]=np.zeros_like(b["sense"])
                    if cond=="delayed":
                        val=original[sensor]
                        b[sensor]=np.concatenate((np.repeat(val[:,:1],8,axis=1),val[:,:-8]),1)
                    pred=np.asarray(infer(b))
                    generated=pred[:,:,:100] if target==0 else pred[:,:,100:356]
                    err=generated[:size,8:]-target_map[:size,8:]
                    rows[cond][0].extend((err**2).mean((1,2)))
                    centered=err-err.mean(1,keepdims=True)
                    rows[cond][1].extend((centered**2).mean((1,2)))
            scores[key]={"examples":len(ids),"groups":len(np.unique(groups)),"mse":{},"centered_mse":{}}
            for cond,(a,b) in rows.items():
                for ix,(metric,v) in enumerate((("mse",a),("centered_mse",b))):
                    diff=np.asarray(v)-rows["correct"][ix]
                    scores[key][metric][cond]={"mean":float(np.mean(v)),"error_with_condition_minus_correct":float(diff.mean()),
                                                "paired_group_ci95":group_ci(diff,groups)}
        # The fixed speech set is separately stratified so each development speaker contributes equally.
        base=data.packs[0]; grid=data.pools[0]; grid=grid[base["ok"][grid].sum(1)==64]
        fixed=np.concatenate([rows[np.argsort(base["uid"][rows])[:8]] for g in np.unique(base["group"][grid])
                              for rows in [grid[base["group"][grid]==g]]])
        saved["uid"]=np.array(base["uid"][fixed]); saved["codec"]=base["a"][fixed].astype(np.float32)/16-12
        for key,src in (("video",1),("text",3)):
            saved[core+"-"+key]=np.clip(np.asarray(infer(gather(0,fixed,src)))[:,:,:100]*3-5,-12,4)
        metrics[core]=scores
        write(out/"generation-dev.json",{"complete":len(metrics)==2,"models":metrics})
        print(core,"cross-modal cases",list(scores),flush=True)
    np.savez_compressed(out/"speech-mels.npz",**saved)


def speech(repo,out):
    import torch
    import torchaudio
    from evals.motion_audio_intelligibility import words,word_error
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    from vocos import Vocos
    torch.set_num_threads(4)
    rev="87c7102498dcde7456f24cfd30239ca606ed9063"; model_name="openai/whisper-tiny.en"
    processor=WhisperProcessor.from_pretrained(model_name,revision=rev,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(model_name,revision=rev,local_files_only=True).eval()
    directory=repo/"artifacts/decoders/vocos-mel-24khz"
    voc=Vocos.from_hparams(str(directory/"config.yaml"))
    voc.load_state_dict(torch.load(directory/"pytorch_model.bin",map_location="cpu",weights_only=True),strict=True)
    voc.eval()
    metadata=json.loads((repo/"artifacts/t2_audio_visual/grid_content_v1/validation.json").read_bytes())
    hashed=lambda s:int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],"little")
    lookup={hashed(r["sample_id"]):r for r in metadata}
    with np.load(out/"speech-mels.npz") as mels,torch.inference_mode():
        rows=[lookup[int(uid)] for uid in mels["uid"]]
        wave={}
        clean=[]
        for row in rows:
            with np.load(repo/row["decoded_path"]) as f:
                pcm=f["pcm"][:48000].astype(np.float32)/32768
            length=min(row["frames"]/25,len(pcm)/16000)
            start=max(0,int((length/(1024/24000)-64)/2))
            offset=round(start*(1024/24000)*16000)
            clean.append(pcm[offset:offset+round(64*1024/24000*16000)])
        wave["original"]=clean
        for k in mels.files:
            if k=="uid":
                continue
            feat=np.repeat(mels[k],4,axis=1).transpose(0,2,1).copy()
            values=[]
            for j in range(0,len(feat),8):
                samples=voc.decode(torch.from_numpy(feat[j:j+8]))
                values.extend(torchaudio.functional.resample(samples,24000,16000).numpy())
            wave[k]=values
        results={}
        refs=[r["sentence"] for r in rows]
        denom=sum(len(words(t)) for t in refs)
        for key,records in wave.items():
            guesses=[]
            for j in range(0,len(records),8):
                batch=processor(records[j:j+8],sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
                ids=model.generate(**batch,max_new_tokens=32,num_beams=1,do_sample=False)
                guesses.extend(processor.batch_decode(ids,skip_special_tokens=True))
            edits=sum(round(word_error(r,g)*len(words(r))) for r,g in zip(refs,guesses))
            results[key]={"word_edits":edits,"words":denom,"wer":edits/denom}
            print(key,results[key],flush=True)
    (out/"speech-dev.json").write_text(json.dumps({"clips":len(rows),"speakers":len(set(r["actor"] for r in rows)),
        "role":"development","recognizer":model_name,"revision":rev,"results":results},sort_keys=True,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("phase",choices=["predict","speech"])
    p.add_argument("--root",type=Path); p.add_argument("--data",type=Path); p.add_argument("--repo",type=Path)
    p.add_argument("--out",required=True,type=Path); a=p.parse_args()
    if a.phase=="predict":
        predict(a.root,a.data,a.out)
    else:
        speech(a.repo,a.out)
