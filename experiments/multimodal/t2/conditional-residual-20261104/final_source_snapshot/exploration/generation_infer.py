"""Standalone inference for actual trained T2-G1/G2 heads; no weight updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from models.slwm_motion import MotionSLWM
from models.slwm_generation_residual import SignalResidualAudioHead
from training.t2_generation_train import make_model
from training.t2_prepare_generation import high_mel,packetize
from training.t2_prepare_motion import encode
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json
from exploration.generation_outputs import predict_audio,invert100
from exploration.generation_residual_outputs import predict_residual
from exploration.motion_outputs import save_video,save_wave


def observed_field(core: MotionSLWM,features: torch.Tensor,modality: str) -> torch.Tensor:
    """One observed [T,128/160] signal -> frozen [T,256], other signal absent."""
    length=len(features); starts=list(range(0,length-31,16))
    if starts[-1]!=length-32: starts.append(length-32)
    total=torch.zeros((length,256),device=features.device); count=torch.zeros((length,1),device=features.device)
    with torch.inference_mode(),torch.autocast(features.device.type,dtype=torch.bfloat16,enabled=features.device.type=="cuda"):
        for s in starts:
            x=features[None,s:s+32]; one=torch.ones((1,32),device=x.device,dtype=torch.bool); zero=torch.zeros_like(one)
            if modality=="audio": a,v,am,vm=x,x.new_zeros((1,32,160)),one,zero
            else: a,v,am,vm=x.new_zeros((1,32,128)),x,zero,one
            z=core.encode(a,v,am,vm)[0][0].float(); total[s:s+32]+=z; count[s:s+32]+=1
    return total/count


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True)
    source=parser.add_mutually_exclusive_group(required=True); source.add_argument("--audio"); source.add_argument("--video")
    parser.add_argument("--seed-image",help="One pre-cropped grayscale mouth image for visual generation, resized to 48x32")
    parser.add_argument("--start-frame",type=int,default=0,help="Start of the 32-frame visual generation window, at 25 Hz")
    parser.add_argument("--video-domain",choices=("cremad","ravdess"),default="cremad")
    parser.add_argument("--output",required=True); parser.add_argument("--device",choices=("cpu","cuda"),default="cpu"); args=parser.parse_args()
    torch.set_num_threads(4); checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    residual=checkpoint["config"].get("sprint")=="T2-G2"
    if residual:
        cfg=json.loads(Path(checkpoint["config"]["g1_config"]).read_text()); run={**checkpoint["run"],"task":"audio"}
    else: cfg=checkpoint["config"]["matrix"]; run=checkpoint["config"]["run"]
    device=torch.device(args.device)
    if run["task"]=="visual" and not args.seed_image: parser.error("Visual generation requires one observed --seed-image")
    kind=run["input"]; modality="audio" if args.audio else "video"
    required="video" if kind in {"field_video","video_coarse"} else "audio"
    if kind!="time_only" and modality!=required: parser.error(f"This head requires {required} conditioning")
    with np.load(Path(cfg["source_corpus"])/"codec.npz",allow_pickle=False) as d: codec={k:d[k] for k in d.files}
    with np.load(Path(cfg["corpus"])/"codec100.npz",allow_pickle=False) as d: codec100={k:d[k] for k in d.files}
    if args.audio:
        mel=high_mel(Path(args.audio)); length=min(128,len(mel)//4)
        if length<32: parser.error("Input must contain at least 1.28 seconds")
        packed,_=packetize(mel,length); coarse=mel[::4][:length]
        a=(coarse-codec["audio_mean"])/codec["audio_scale"]
        delta=(np.diff(coarse,axis=0,prepend=coarse[:1])-codec["audio_delta_mean"])/codec["audio_delta_scale"]
        features=np.concatenate([a,delta],-1)
        high=((packed.reshape(-1,64)-codec100["mean"])/codec100["scale"]).reshape(length,256)
    else:
        crop=cfg["external_crop"] if args.video_domain=="ravdess" else "fps=25,crop=iw*0.30:ih*0.24:iw*0.35:ih*0.42,scale=48:32"
        result=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",args.video,"-an","-vf",crop,"-frames:v","128","-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45)
        mouth=np.frombuffer(result.stdout,np.uint8).reshape(-1,32,48).copy(); length=len(mouth)
        if length<32: parser.error("Video must contain at least 32 frames at 25 Hz")
        _,features=encode(mouth,np.zeros((length,64),np.float32),codec)
    x=torch.from_numpy(features.astype(np.float16)).to(device).float()
    observed=x
    if kind.startswith("field_"):
        if sha256(Path(cfg["checkpoint"]))!=cfg["checkpoint_sha256"]: raise ValueError("Frozen core changed")
        old=torch.load(cfg["checkpoint"],map_location="cpu",weights_only=False)
        core=MotionSLWM(old["config"]["run"]["model"]).to(device).eval(); core.load_state_dict(old["model"])
        x=observed_field(core,x,modality).half().float()
    elif kind=="mel100": x=torch.from_numpy(high.astype(np.float16)).to(device).float()
    elif kind=="time_only": x=torch.zeros((length,256),device=device)
    model=(SignalResidualAudioHead(codec,codec100) if residual else make_model(run)).to(device).eval(); model.load_state_dict(checkpoint["model"])
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if run["task"]=="audio":
        mel=predict_residual(model,x,observed) if residual else predict_audio(model,x)
        save_wave(out/"generated_audio.wav",invert100(mel,codec100)); np.save(out/"predicted_mel.npy",mel)
    else:
        if args.start_frame<0 or args.start_frame+32>length: parser.error("Requested visual window is outside the observed audio")
        seed=np.asarray(Image.open(args.seed_image).convert("L").resize((48,32)),np.float32)/255.
        with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            frames=model(x[None,args.start_frame:args.start_frame+32],torch.from_numpy(seed)[None].to(device))[0].float().cpu().numpy()
        save_video(out/"generated_motion.gif",frames,"MODEL: observed audio + ONE real seed"); np.save(out/"predicted_frames.npy",frames)
    write_json(out/"inference.json",{"run":run,"checkpoint_sha256":sha256(Path(args.checkpoint)),"source_modality":modality,
        "source_sha256":sha256(Path(args.audio or args.video)),"seed_sha256":sha256(Path(args.seed_image)) if args.seed_image else None,
        "start_frame":args.start_frame,"max_observed_seconds":length/25,"output":"actual frozen decoder proposal; no policy commit", "device":args.device})


if __name__=="__main__": main()
