"""Render actual T2-G1 predictions, codec ceilings and falsification controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from models.slwm_motion import MotionSLWM
from exploration.motion_outputs import invert_mel,save_wave,save_video,predict_sequence
from training.t2_generation_train import load_data,make_model,predict
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json


def invert100(features: np.ndarray,codec: dict,iterations: int=48) -> np.ndarray:
    """100 Hz predicted mel [S,64] -> new-phase 16 kHz waveform [N]."""
    mel=np.exp(np.clip(features*codec["scale"]+codec["mean"],-18,15)).astype(np.float32)
    banks=torchaudio.functional.melscale_fbanks(513,0,8000,64,16000)
    power=(torch.from_numpy(mel)@torch.linalg.pinv(banks)).clamp_min(1e-9).T
    torch.manual_seed(17)
    griffin=torchaudio.transforms.GriffinLim(n_fft=1024,n_iter=iterations,hop_length=160,power=2.,momentum=.99)
    with torch.inference_mode(): wave=griffin(power).numpy()
    wave=np.nan_to_num(wave); return (wave/max(float(np.abs(wave).max()),1e-5)*.8).astype(np.float32)


def predict_audio(model: torch.nn.Module,x: torch.Tensor) -> np.ndarray:
    """Observed input [T,D] -> overlap-averaged 100 Hz [4T,64], no targets."""
    length=len(x); starts=list(range(0,length-31,16))
    if starts[-1]!=length-32: starts.append(length-32)
    total=np.zeros((4*length,64),np.float32); count=np.zeros((4*length,1),np.float32)
    with torch.inference_mode(),torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.device.type=="cuda"):
        for s in starts:
            y=model(x[None,s:s+32])[0].float().cpu().numpy()
            total[4*s:4*(s+32)]+=y; count[4*s:4*(s+32)]+=1
    return total/count


def resample_wrong(x: torch.Tensor,length: int) -> torch.Tensor:
    """Wrong full clip [S,D] -> same duration [T,D], for listening controls only."""
    return F.interpolate(x.float().T[None],size=length,mode="linear",align_corners=False)[0].T


def load_models(cfg: dict) -> dict:
    models={}
    for run in cfg["runs"]:
        checkpoint=torch.load(Path(cfg["artifacts"])/run["id"]/"best.pt",map_location="cpu",weights_only=False)
        model=make_model(run).cuda().eval(); model.load_state_dict(checkpoint["model"]); models[run["id"]]=model
    return models


def audio_case(cfg: dict,data: dict,split: str,index: int,folder: Path,models: dict,old_model: MotionSLWM,codec: dict,codec100: dict) -> dict:
    """One fixed held-out clip -> actual waveforms for every registered arm."""
    row=data["rows"][index]; length=int(data["lengths"][index]); folder.mkdir(parents=True,exist_ok=True)
    a=data["audio_coarse"][index,:length].float(); v=data["video_coarse"][index,:length].float()
    outputs={}
    for mode in ("audio","video"):
        outputs["old_"+mode]=predict_sequence(old_model,a,v,mode)["audio"][:,:64]
    outputs["codec25"]=a[:,:64].cpu().numpy()
    for name,features in outputs.items(): save_wave(folder/(name+".wav"),invert_mel(features,codec))
    target=data["mel100"][index,:length].float().cpu().numpy().reshape(4*length,64)
    save_wave(folder/"codec100.wav",invert100(target,codec100))
    for run in (r for r in cfg["runs"] if r["task"]=="audio"):
        x=data[run["input"]][index,:length].float(); model=models[run["id"]]
        features=predict_audio(model,x); save_wave(folder/(run["id"]+".wav"),invert100(features,codec100))
        if run["id"]=="EXP-T2-402":
            wrong_index=int(data["wrong"][index]); wrong_length=int(data["lengths"][wrong_index])
            wrong=resample_wrong(data[run["input"]][wrong_index,:wrong_length],length)
            for label,xx in (("wrong_video",wrong),("null_video",torch.zeros_like(x))):
                save_wave(folder/(label+".wav"),invert100(predict_audio(model,xx),codec100))
    subprocess.run(["ffmpeg","-v","error","-y","-i",row["raw_path"],"-vn","-ac","1","-ar","16000","-t",str(length/25),str(folder/"reference.wav")],check=True,timeout=45)
    return {"sample_id":row["sample_id"],"actor":row["actor"],"split":split,"transcript":row["transcript"],"folder":str(folder),
        "conditions":["reference","codec25","codec100","old_audio","old_video"]+[r["id"] for r in cfg["runs"] if r["task"]=="audio"]+["wrong_video","null_video"],
        "waveforms_are_actual_predictions":True,"peak_normalized_generated_audio":True,"wrong_control":"same person different clip; duration linearly resampled"}


def visual_case(cfg: dict,data: dict,split: str,index: int,folder: Path,models: dict) -> dict:
    """Middle 32 frames -> one real seed, actual model motions, all controls."""
    row=data["rows"][index]; start=(int(data["lengths"][index])-32)//2; ids=np.array([index]); starts=np.array([start])
    folder.mkdir(parents=True,exist_ok=True); frames={}
    for run in (r for r in cfg["runs"] if r["task"]=="visual"):
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            pred,target=predict(models[run["id"]],run,data,ids,starts,32)
        frames[run["id"]]=pred[0].float().cpu().numpy()
    frames["reference"]=target[0].float().cpu().numpy(); frames["seed_only"]=np.repeat(frames["reference"][:1],32,axis=0)
    full=next(r for r in cfg["runs"] if r["id"]=="EXP-T2-411")
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        pred,_=predict(models["EXP-T2-411"],full,data,ids,starts,32,"wrong")
    frames["wrong_audio"]=pred[0].float().cpu().numpy()
    for name,sequence in frames.items():
        label={"reference":"REFERENCE recorded frames","seed_only":"CONTROL copied seed","wrong_audio":"CONTROL wrong audio + real seed"}.get(name,name+": audio + one real seed")
        if name=="EXP-T2-413": label="CONTROL time + one real seed"
        save_video(folder/(name+".gif"),np.clip(sequence,0,1),label)
    # A compact scientific contact sheet at fixed timepoints, with all values
    # from the actual decoder. It complements the complete GIF/WebM artifacts.
    from PIL import Image,ImageDraw
    Image.fromarray((frames["reference"][0]*255).round().astype(np.uint8)).save(folder/"seed.png")
    names=("reference","seed_only","EXP-T2-411","EXP-T2-412","EXP-T2-413","wrong_audio")
    canvas=Image.new("RGB",(160+8*144,len(names)*108),"#14232d"); draw=ImageDraw.Draw(canvas)
    for j,name in enumerate(names):
        draw.text((6,j*108+40),name,fill="white")
        for k,t in enumerate(np.linspace(0,31,8).round().astype(int)):
            frame=Image.fromarray((np.clip(frames[name][t],0,1)*255).astype(np.uint8)).resize((144,96))
            canvas.paste(frame,(160+k*144,j*108))
    canvas.save(folder/"comparison.png")
    np.savez_compressed(folder/"predictions.npz",**frames)
    return {"sample_id":row["sample_id"],"actor":row["actor"],"split":split,"folder":str(folder),"start_frame":start,"frames":32,
        "conditioning":"entire simultaneous audio window plus ONE observed initial frame; no other frame is input","transcript":row["transcript"]}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); torch.set_num_threads(4)
    out=Path(cfg["artifacts"])/"exploration"; out.mkdir(parents=True,exist_ok=True)
    snapshot=out/"source_snapshot"; hashes={}
    for name in ("exploration/generation_outputs.py","exploration/motion_outputs.py","evals/generation_intelligibility.py"):
        path=snapshot/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(name,path); hashes[name]=sha256(Path(name))
    write_json(out/"source_hashes.json",hashes)
    models=load_models(cfg)
    checkpoint=torch.load(cfg["checkpoint"],map_location="cpu",weights_only=False)
    old_model=MotionSLWM(checkpoint["config"]["run"]["model"]).cuda().eval(); old_model.load_state_dict(checkpoint["model"])
    with np.load(Path(cfg["source_corpus"])/"codec.npz",allow_pickle=False) as d: codec={k:d[k] for k in d.files}
    with np.load(Path(cfg["corpus"])/"codec100.npz",allow_pickle=False) as d: codec100={k:d[k] for k in d.files}
    gallery=[]; audio=[]
    for split in ("validation","test","external"):
        data=load_data(Path(cfg["corpus"]),split)
        if split=="validation":
            results=json.loads((Path(cfg["artifacts"])/"EXP-T2-411/validation_metrics.json").read_text())["per_clip"]
            order=np.argsort([r["correct_roi_mse"] for r in results]); visual_ids=[int(order[i]) for i in (0,len(order)//2,len(order)-1)]
        elif split=="test": visual_ids=[0,len(data["rows"])//2,len(data["rows"])-1]
        else: visual_ids=[next(i for i,r in enumerate(data["rows"]) if r["actor"]==actor) for actor in sorted({r["actor"] for r in data["rows"]})]
        for index in visual_ids:
            folder=out/"visual"/(split+"_"+data["rows"][index]["sample_id"])
            gallery.append(visual_case(cfg,data,split,index,folder,models))
        if split!="validation":
            audio_ids=[]
            for actor in sorted({r["actor"] for r in data["rows"]}):
                candidates=sorted((i for i,r in enumerate(data["rows"]) if r["actor"]==actor),key=lambda i:data["rows"][i]["sample_id"])
                audio_ids.extend(candidates[:4])
            for index in audio_ids:
                folder=out/"audio"/(split+"_"+data["rows"][index]["sample_id"])
                audio.append(audio_case(cfg,data,split,index,folder,models,old_model,codec,codec100))
                write_json(out/"audio_cases.json",audio)
                print(json.dumps({"rendered_audio":len(audio),"sample":data["rows"][index]["sample_id"]}),flush=True)
        del data; torch.cuda.empty_cache()
    write_json(out/"gallery.json",{"visual_examples":gallery,"audio_examples":audio,"scope":"diagnostic outputs, never policy commits",
        "selection":"validation low/median/high visual errors + fixed test/external visual; first four sample IDs per test person for audio"})
    files={str(p.relative_to(out)):sha256(p) for p in out.rglob("*") if p.is_file() and "source_snapshot" not in str(p) and p.name!="artifact_hashes.json"}
    write_json(out/"artifact_hashes.json",files)


if __name__=="__main__": main()
