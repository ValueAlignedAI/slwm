"""Use a frozen T2-R1 checkpoint on ordinary partial audio and optional video."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from models.slwm_restoration import RestorationSLWM
from models.slwm_content import ContentRestorationSLWM
from training.restoration_train import load_codec
from training.t2_prepare_generation import high_mel,packetize
from training.t2_prepare_motion import encode
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json
from exploration.restoration_outputs import restore_sequence
from exploration.generation_outputs import invert100
from exploration.motion_outputs import save_wave


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True); parser.add_argument("--audio",required=True); parser.add_argument("--video")
    parser.add_argument("--video-domain",choices=("cremad","ravdess"),default="cremad"); parser.add_argument("--visibility",help="Boolean NPY, one observed/missing flag per 25Hz packet")
    parser.add_argument("--gaps",default="",help="Missing intervals in seconds, e.g. 0.4:0.56,1.2:1.36; overrides input evidence")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu"); parser.add_argument("--output",required=True); args=parser.parse_args()
    torch.set_num_threads(4); checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False); cfg=checkpoint["config"]; run=checkpoint["run"]; device=torch.device(args.device)
    mel=high_mel(Path(args.audio)); length=min(128,len(mel)//4)
    if length<32: parser.error("At least 1.28 seconds of audio is required")
    codec=load_codec(cfg,args.device); packed,_=packetize(mel,length)
    with np.load(Path(cfg["generation_corpus"])/"codec100.npz",allow_pickle=False) as d: new={k:d[k] for k in d.files}
    y=((packed.reshape(-1,64)-new["mean"])/new["scale"]).reshape(length,256).astype(np.float16)
    video=np.zeros((length,160),np.float32)
    if args.video:
        crop=cfg["external_crop"] if args.video_domain=="ravdess" else "fps=25,crop=iw*0.30:ih*0.24:iw*0.35:ih*0.42,scale=48:32"
        result=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",args.video,"-an","-vf",crop,"-frames:v","128","-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45)
        mouth=np.frombuffer(result.stdout,np.uint8).reshape(-1,32,48).copy(); length=min(length,len(mouth))
        if length<32: parser.error("Video too short")
        with np.load(Path(cfg["old_corpus"])/"codec.npz",allow_pickle=False) as d: old={k:d[k] for k in d.files}
        _,video=encode(mouth[:length],np.zeros((length,64),np.float32),old)
    visible=np.ones(length,bool)
    if args.visibility:
        mask=np.load(args.visibility,allow_pickle=False)
        if mask.dtype!=bool or len(mask)<length: parser.error("Visibility must be a boolean vector covering the observed clip")
        visible=mask[:length].copy()
    for span in filter(None,args.gaps.split(",")):
        start,end=map(float,span.split(":"))
        if start<0 or end<=start: parser.error("Invalid missing interval")
        visible[max(0,int(round(start*25))):min(length,int(round(end*25)))]=False
    if not visible.any(): parser.error("This restoration model requires some observed sound")
    obs=codec.observe(torch.from_numpy(y[:length])[None].to(device).float(),torch.from_numpy(visible)[None].to(device))
    model=(ContentRestorationSLWM if cfg["sprint"]=="T2-R2" else RestorationSLWM)(cfg,run).to(device).eval(); model.load_state_dict(checkpoint["model"],strict=True)
    predicted=restore_sequence(model,obs,torch.from_numpy(video.astype(np.float16))[None].to(device).float(),bool(args.video))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); save_wave(out/"restored_audio.wav",invert100(predicted,new)); np.save(out/"predicted_mel.npy",predicted); np.save(out/"visible_mask.npy",visible)
    write_json(out/"inference.json",{"run":run,"checkpoint_sha256":sha256(Path(args.checkpoint)),"audio_sha256":sha256(Path(args.audio)),"video_sha256":sha256(Path(args.video)) if args.video else None,
        "missing_packet_fraction":float((~visible).mean()),"task":"simultaneous feature-packet restoration, not causal future prediction", "device":args.device,
        "note":"Centered 64ms mel windows can overlap packet boundaries. This does not establish sample-exact waveform inpainting. No source waveform samples are inserted into generated audio."})


if __name__=="__main__": main()
