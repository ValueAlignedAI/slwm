"""Actual frozen speech-repair outputs, global corruption and ASR diagnostics."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from models.slwm_restoration import RestorationSLWM
from training.restoration_train import load_data,load_codec,fixed_holes
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,cluster_interval
from exploration.generation_outputs import invert100
from exploration.motion_outputs import save_wave
from evals.motion_audio_intelligibility import read_wave,word_error


def full_visibility(row: dict,length: int,device: torch.device) -> torch.Tensor:
    """One deterministic global packet mask; neighboring windows never reveal it."""
    mask=fixed_holes([row],device).repeat(1,math.ceil(length/32))[:,:length].clone(); mask[:,0]=True; mask[:,-1]=True
    return mask


def restore_sequence(model: RestorationSLWM,observation: dict,video: torch.Tensor,video_present: bool) -> np.ndarray:
    """Observed [1,T,...] only -> overlap-averaged predicted [4T,64]."""
    length=observation["audio"].shape[1]; starts=list(range(0,length-31,16))
    if starts[-1]!=length-32: starts.append(length-32)
    total=np.zeros((4*length,64),np.float32); count=np.zeros((4*length,1),np.float32); device=video.device
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        for s in starts:
            a=observation["audio"][:,s:s+32]; seen=observation["visible"][:,s:s+32]; base=observation["base"][:,4*s:4*(s+32)]
            vm=torch.full_like(seen,video_present)
            pred=model(a,video[:,s:s+32],seen,vm,base)[0].float().cpu().numpy()
            total[4*s:4*(s+32)]+=pred; count[4*s:4*(s+32)]+=1
    return total/count


def selection(data: dict,split: str) -> list[int]:
    if split=="libri_test": return list(range(len(data["rows"])))
    ids=[]
    for actor in sorted({r["actor"] for r in data["rows"]}):
        pool=sorted([i for i,r in enumerate(data["rows"]) if r["actor"]==actor],key=lambda i:data["rows"][i]["sample_id"])
        if split=="ravdess_test": pool=[i for i in pool if data["rows"][i]["sample_id"].split("-")[2]=="01"]
        ids.extend(pool[i] for i in np.linspace(0,len(pool)-1,2).round().astype(int))
    return ids


def render_case(cfg,data,split,index,folder,models,codec,new_codec,plot=False):
    row=data["rows"][index]; length=int(data["lengths"][index]); y=data["mel100"][index:index+1,:length].float(); v=data["video"][index:index+1,:length].float()
    present=bool(data["video_present"][index]); visible=full_visibility(row,length,y.device); obs=codec.observe(y,visible)
    folder.mkdir(parents=True,exist_ok=True); predictions={"codec100":obs["target"][0].cpu().numpy(),"interpolation":obs["base"][0].cpu().numpy()}
    silence=((-18.42068074-codec.mean100)/codec.scale100).repeat(4)[None,None]
    corrupted=torch.where(visible[...,None],y,silence).reshape(1,4*length,64)
    predictions["silence_gaps"]=corrupted[0].cpu().numpy()
    for key,model in models.items(): predictions[key]=restore_sequence(model,obs,v,present)
    wi=int(data["wrong"][index]); wl=int(data["lengths"][wi]); wv=data["video"][wi,:wl].float()
    wrong=F.interpolate(wv.T[None],size=length,mode="linear",align_corners=False).transpose(1,2)
    predictions["wrong_video_501"]=restore_sequence(models["EXP-T2-501"],obs,wrong,present)
    clean=codec.observe(y,torch.ones_like(visible)); predictions["clean_audio_501"]=restore_sequence(models["EXP-T2-501"],clean,torch.zeros_like(v),False)
    for name,mel in predictions.items(): save_wave(folder/(name+".wav"),invert100(mel,new_codec))
    subprocess.run(["ffmpeg","-v","error","-y","-i",row["raw_path"],"-vn","-ac","1","-ar","16000","-t",str(length/25),str(folder/"reference.wav")],check=True,timeout=45)
    np.save(folder/"visible_mask.npy",visible[0].cpu().numpy()); np.savez_compressed(folder/"predicted_mels.npz",**predictions)
    if plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names=("codec100","interpolation","EXP-T2-507","EXP-T2-501","wrong_video_501")
        fig,axes=plt.subplots(5,1,figsize=(11,8),sharex=True,layout="constrained"); limits=np.quantile(predictions["codec100"],[.02,.98])
        for axis,name in zip(axes,names):
            axis.imshow(predictions[name].T,origin="lower",aspect="auto",extent=(0,length/25,0,64),vmin=limits[0],vmax=limits[1],cmap="magma")
            for t in np.where(~visible[0].cpu().numpy())[0]: axis.axvspan(t/25,(t+1)/25,color="cyan",alpha=.12)
            axis.set_ylabel(name,fontsize=8)
        axes[-1].set_xlabel("Seconds; cyan marks hidden mel packets"); fig.suptitle(row["sample_id"]+": actual predicted sound features")
        fig.savefig(folder/"comparison.png",dpi=150); plt.close(fig)
    return {"sample_id":row["sample_id"],"actor":row["actor"],"split":split,"transcript":row["transcript"],"folder":str(folder),
        "conditions":["reference"]+list(predictions),"video_present":present,"missing_packet_fraction":float((~visible).float().mean()),
        "scope":"Withheld centered-mel packets, not sample-exact waveform loss; waveform phase newly estimated, no source samples substituted"}


def score(cfg,root,records):
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error(); name="openai/whisper-tiny.en"; revision="87c7102498dcde7456f24cfd30239ca606ed9063"
    processor=WhisperProcessor.from_pretrained(name,revision=revision); recognizer=WhisperForConditionalGeneration.from_pretrained(name,revision=revision).cuda().half().eval()
    results=[]
    for record in records:
        scores={}
        for condition in record["conditions"]:
            path=Path(record["folder"])/(condition+".wav"); inputs=processor(read_wave(path),sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():
                output=recognizer.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=64,num_beams=1,do_sample=False)
            text=processor.batch_decode(output,skip_special_tokens=True)[0].strip(); scores[condition]={"recognizer_text":text,"wer":word_error(record["transcript"],text),"wave_sha256":sha256(path)}
        results.append({**record,"scores":scores}); write_json(root/"intelligibility_progress.json",results)
        print(json.dumps({"recognized":len(results),"sample":record["sample_id"]}),flush=True)
    summary={}
    for split in ("crema_test","libri_test","ravdess_test"):
        subset=[r for r in results if r["split"]==split]; conditions=records[0]["conditions"]
        means={c:float(np.mean([r["scores"][c]["wer"] for r in subset])) for c in conditions}; changes={}
        for model in ("EXP-T2-501","EXP-T2-502","EXP-T2-508","EXP-T2-507"):
            delta=np.array([r["scores"]["interpolation"]["wer"]-r["scores"][model]["wer"] for r in subset])
            changes[model]={"absolute_wer_reduction":float(delta.mean()),"relative_wer_reduction":float(delta.mean()/max(means["interpolation"],1e-9)),
                "person_cluster_absolute_reduction_95_interval":cluster_interval(delta,[r["actor"] for r in subset])}
        summary[split]={"samples":len(subset),"people":len({r["actor"] for r in subset}),"mean_clip_wer":means,"paired_changes":changes}
    write_json(root/"intelligibility.json",{"scope":"Fallible recognizer diagnostic, never SLWM-generated text or human intelligibility judgment", "recognizer":name,"revision":revision,
        "evaluation_only_weights":sum(p.numel() for p in recognizer.parameters()),"summary":summary,"cases":results})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--score-only",action="store_true"); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"])/"exploration"; root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    if not args.score_only:
        codec=load_codec(cfg); models={}
        for run in cfg["runs"]:
            if run["id"] not in {"EXP-T2-501","EXP-T2-502","EXP-T2-503","EXP-T2-507","EXP-T2-508","EXP-T2-509"}: continue
            checkpoint=torch.load(Path(cfg["artifacts"])/run["id"]/"best.pt",map_location="cpu",weights_only=False)
            model=RestorationSLWM(cfg,run).cuda().eval(); model.load_state_dict(checkpoint["model"]); models[run["id"]]=model
        with np.load(Path(cfg["generation_corpus"])/"codec100.npz",allow_pickle=False) as d: new_codec={k:d[k] for k in d.files}
        records=[]
        for split in ("crema_test","libri_test","ravdess_test"):
            data=load_data(cfg,split)
            for j,index in enumerate(selection(data,split)):
                folder=root/(split+"_"+data["rows"][index]["sample_id"])
                records.append(render_case(cfg,data,split,index,folder,models,codec,new_codec,plot=j==0)); write_json(root/"cases.json",records)
                print(json.dumps({"rendered":len(records),"sample":data["rows"][index]["sample_id"]}),flush=True)
            del data; torch.cuda.empty_cache()
        del models,codec; torch.cuda.empty_cache()
    records=json.loads((root/"cases.json").read_text()); score(cfg,root,records)


if __name__=="__main__": main()
