"""Challenge the frozen confidence gate with acoustically absent and out-of-grammar inputs."""
from pathlib import Path
import json
import numpy as np
import torch
import jax
import jax.numpy as jnp
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from data.g14_grid import T,labels
from data.g14_wave import pcm
from scripts.data.g14_cases import decode
from models.g14_cue import forward
from scripts.train.g9_storage import write
ROOT=Path("/mnt/slwm-data/SLWM/g14")
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def load(p):return json.loads(p.read_bytes())

def main():
    pkg=ROOT/"release/inference";man=load(pkg/"manifest.json");p={k:jnp.asarray(v) for k,v in read(pkg/"speech.npz").items()}
    normal=read(pkg/"normalizer.npz");snapshot=ROOT.parent/"g11/data/whisper-small"
    proc=WhisperProcessor.from_pretrained(str(snapshot),local_files_only=True)
    enc=WhisperForConditionalGeneration.from_pretrained(str(snapshot),local_files_only=True,dtype=torch.float16).model.encoder.cuda().eval()
    basis=torch.tensor(normal["basis"],device="cuda");at=np.rint((np.arange(T)*.032+.016)/.02).astype(int)
    @jax.jit
    def head(a,length):
        return tuple(jax.nn.softmax(v,axis=-1) for v in forward(p,a,jnp.zeros((len(a),T,96)),length,man["speech_arm"])[0])
    def probabilities(waves):
        rows=[[] for _ in range(3)]
        with torch.inference_mode():
            for j in range(0,len(waves),24):
                vals=waves[j:j+24];length=np.asarray([min(T,int(np.ceil(len(v)/512))) for v in vals])
                x=proc.feature_extractor(vals,sampling_rate=16000,return_tensors="pt",device="cuda").input_features.to("cuda",dtype=torch.float16)
                h=(enc(x,return_dict=True).last_hidden_state[:,at].float()@basis).cpu().numpy()
                a=(np.clip(.5+(h-normal["mean"])/(6*normal["std"]),0,1)*(np.arange(T)[None,:]<length[:,None])[:,:,None]).astype("f2")
                for dest,pp in zip(rows,head(jnp.asarray(a),jnp.asarray(length))):dest.extend(np.asarray(pp))
        return [np.stack(x) for x in rows]
    cases={r:load(ROOT/"data/cases"/(r+".json")) for r in ("held","fresh")}
    meta=load(ROOT/"data/utterances/media.json");silent=probabilities([np.zeros(48000,"f4")]);out={}
    def inspect(prob,index,maps):
        decisions=[decode(*(v[i] for v in prob),m) for i,m in zip(index,maps)]
        conf=np.asarray([x[2] for x in decisions]);accepted=sum(c<3 and h>=man["threshold"] for c,s,h in decisions)
        return dict(evaluations=len(maps),commits=int(accepted),fraction=float(accepted/len(maps)),peak_confidence=float(conf.max()),median_confidence=float(np.median(conf)))
    with np.load(ROOT/"work/cue-semantic/posterior.npz") as saved:
        for role,items in cases.items():
            others=[item for item in items];rng=np.random.default_rng(14531+(role=="fresh"))
            other_ids={k:i for i,k in enumerate(saved[role+"_indices"])}
            probs=[saved[f"{role}_normal_{k}"] for k in range(3)];white={s:[i for i in other_ids if meta[i]["person"]==s and labels(meta[i]["text"])[0]==3] for s in {r["speaker"] for r in others}}
            ids=[rng.choice(white[r["speaker"]]) for r in others];maps=[r["addresses"] for r in others]
            selected=rng.choice(len(others),32,replace=False);sample=[others[i] for i in selected]
            actual=[pcm(meta[r["commands"][0]["clip"]]) for r in sample];noise=[rng.normal(0,max(.001,float(np.sqrt(np.mean(x*x)))),len(x)).astype("f4") for x in actual]
            result={"zero_wave":inspect(silent,np.zeros(len(others),int),maps),"unsupported_white_command":inspect(probs,[other_ids[i] for i in ids],maps),
                    "reversed_recording":inspect(probabilities([x[::-1].copy() for x in actual]),range(len(sample)),[r["addresses"] for r in sample]),
                    "matched_rms_noise":inspect(probabilities(noise),range(len(sample)),[r["addresses"] for r in sample])}
            result["zero_wave"]["unique_waveforms"]=1;result["unsupported_white_command"]["unique_waveforms"]=len(set(ids))
            result["reversed_recording"]["unique_waveforms"]=len({r["commands"][0]["clip"] for r in sample});result["matched_rms_noise"]["unique_waveforms"]=len(noise)
            out[role]=result
    write(ROOT/"work/acoustic-abstention.json",dict(threshold=man["threshold"],held=out,design="Frozen deployed audio encoder and calibration gate. Identical public station maps. No new threshold fitting or physical actuation."))
    print("Null-input commit audit",{r:{k:v["commits"] for k,v in d.items()} for r,d in out.items()},flush=True)
if __name__=="__main__":main()
