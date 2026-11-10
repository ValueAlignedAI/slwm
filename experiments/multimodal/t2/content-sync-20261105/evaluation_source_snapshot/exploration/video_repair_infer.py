"""True waveform speech repair with observed-only features and phase.

Inference on ordinary files; no corpus targets, transcripts or recognizer needed.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from models.slwm_restoration import RestorationCodec
from training.video_repair_train import load_model
from training.t2_prepare_motion import encode
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json
from exploration.video_repair_audio import decode_pcm,write_pcm,mel100,constrained_audio


def load_codecs(cfg,device="cpu",directory=None):
    parent=json.loads(Path(cfg["restoration_config"]).read_text()) if directory is None else None
    old_path=Path(directory)/"codec.npz" if directory else Path(parent["old_corpus"])/"codec.npz"
    new_path=Path(directory)/"codec100.npz" if directory else Path(parent["generation_corpus"])/"codec100.npz"
    with np.load(old_path,allow_pickle=False) as f: old={k:f[k] for k in f.files}
    with np.load(new_path,allow_pickle=False) as f: new={k:f[k] for k in f.files}
    return RestorationCodec(old,new).to(device),old,new


def support_masks(missing: np.ndarray,packets: int):
    prefix=np.r_[0,np.cumsum(missing.astype(np.int64))]
    def intersects(centers):
        left=np.clip(centers-512,0,len(missing)); right=np.clip(centers+512,0,len(missing))
        return (prefix[right]-prefix[left])>0
    return ~intersects(np.arange(packets)*640),intersects(np.arange(packets*4)*160)


def observed_features(recorded,missing,new,device="cpu"):
    # Erase before the first Fourier transform. Hidden source values are never
    # used even temporarily to initialize interpolation or reconstruction phase.
    observed=np.asarray(recorded,np.float32).copy(); observed[missing]=0
    physical=mel100(torch.as_tensor(observed,device=device)).cpu().numpy()
    return ((physical-new["mean"])/new["scale"]).astype(np.float32)


def pad_subframes(features,packets):
    """Match fixed video packets; at most three absent terminal fine frames.

    This is the existing packetize contract. Padding repeats only the already
    erased recording's last analyzed frame, never an original hidden sample.
    """
    missing=max(0,packets*4-len(features))
    if missing>3: raise ValueError("Audio/video duration differs by more than30ms")
    return np.pad(features,((0,missing),(0,0)),mode="edge") if missing else features


def repair_windows(unknown,packets):
    positions=np.where(unknown.reshape(packets,4).any(1))[0]
    if not len(positions): return []
    groups=np.split(positions,np.where(np.diff(positions)>1)[0]+1); starts=[]
    for group in groups:
        if len(group)>28: raise ValueError("A repair gap exceeds the supported short-gap context")
        starts.append(int(np.clip((int(group[0])+int(group[-1])+1)//2-16,0,packets-32)))
    return sorted(set(starts))


def predict_mel(model,codec,new,recorded,missing,mouth,video=None,present=True,device="cpu"):
    """Full observed waveform -> [S,64], using one centered window per gap.

    All windows share a globally erased input; overlapping gaps cannot leak
    their originals through another window. Longer ordinary clips are limited
    to 5.12s by the input CLI, matching the trained feature contract.
    """
    features=observed_features(recorded,missing,new,device)
    packets=min(128,(len(features)+3)//4,len(mouth)); features=pad_subframes(features,packets)
    if packets<32: raise ValueError("At least 1.28 seconds is required")
    visible,unknown=support_masks(missing,packets); starts=repair_windows(unknown,packets)
    if not starts: return features,{"windows":[],"visible":visible,"unknown":unknown,"gates":[]}
    totals=np.zeros((packets*4,64),np.float32); weight=np.zeros(packets*4,np.float32); gates=[]
    if video is None: video=np.zeros((packets,160),np.float32)
    amp=torch.autocast("cuda",dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode(),amp:
        for start in starts:
            stop=start+32
            y=torch.as_tensor(features[start*4:stop*4].reshape(1,32,256),device=device)
            seen=torch.as_tensor(visible[None,start:stop],device=device)
            obs=codec.observe(y,seen)
            result=model(obs,torch.as_tensor(mouth[None,start:stop],device=device),torch.as_tensor(video[None,start:stop],device=device).float(),
                torch.as_tensor(unknown[None,start*4:stop*4],device=device),torch.tensor([present],device=device))
            totals[start*4:stop*4]+=result["prediction"][0].float().cpu().numpy(); weight[start*4:stop*4]+=1
            gates.append(result["gate"][0].float().cpu().numpy().tolist())
    predicted=features.copy(); use=unknown&(weight>0); predicted[:packets*4][use]=totals[use]/weight[use,None]
    if (unknown&(weight==0)).any(): raise ValueError("Gap is outside supported repair windows")
    return predicted,{"windows":starts,"visible":visible,"unknown":unknown,"gates":gates}


def decode_mouth(path,cfg,domain="cremad"):
    filt=cfg["ravdess_crop"] if domain=="ravdess" else cfg["crema_crop"]
    raw=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-an","-vf",filt,"-frames:v","128",
        "-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45).stdout
    return np.frombuffer(raw,np.uint8).reshape(-1,64,96).copy()


def pca_video(mouth,old):
    from PIL import Image
    small=np.stack([np.asarray(Image.fromarray(p).resize((48,32))) for p in mouth])
    return encode(small,np.zeros((len(small),64),np.float32),old)[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--audio",required=True); p.add_argument("--video")
    p.add_argument("--video-domain",choices=("cremad","ravdess"),default="cremad"); p.add_argument("--gaps",required=True,help="Seconds, e.g. 1.20:1.52; empty string returns unchanged audio")
    p.add_argument("--codec-dir"); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--iterations",type=int,default=64); p.add_argument("--output",required=True)
    args=p.parse_args(); torch.set_num_threads(4); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    model,ck=load_model(args.checkpoint,args.device); codec,old,new=load_codecs(ck["config"],args.device,args.codec_dir)
    pcm=decode_pcm(Path(args.audio)); sound=pcm.astype(np.float32)/32768.; missing=np.zeros(len(sound),bool)
    for span in filter(None,args.gaps.split(",")):
        begin,end=map(float,span.split(":"))
        if begin<0 or end<=begin or end-begin>.481 or end*16000>len(sound): p.error("Each gap must be within the recording and at most 480ms")
        missing[int(round(begin*16000)):int(round(end*16000))]=True
    mouth=decode_mouth(args.video,ck["config"],args.video_domain) if args.video else np.zeros((min(128,(len(sound)+639)//640),64,96),np.uint8)
    if args.video and len(mouth)*640<len(sound)-640: p.error("Video does not cover the recording")
    video=pca_video(mouth,old) if ck["run"].get("arm")=="pca_video" else None
    predicted,audit=predict_mel(model,codec,new,sound,missing,mouth,video,bool(args.video),args.device)
    restored=constrained_audio(predicted,sound,missing,new,args.iterations,args.device)
    write_pcm(out/"repaired.wav",restored); np.save(out/"predicted_mel.npy",predicted); np.save(out/"missing_samples.npy",missing)
    recovered=decode_pcm(out/"repaired.wav"); exact=bool(np.array_equal(recovered[~missing],pcm[~missing]))
    if not exact: raise ValueError("Observed waveform preservation failed")
    write_json(out/"inference.json",{"run":ck["run"],"checkpoint_sha256":sha256(Path(args.checkpoint)),"input_sha256":sha256(Path(args.audio)),
        "video_sha256":sha256(Path(args.video)) if args.video else None,"observed_pcm_exact":exact,"hidden_samples":int(missing.sum()),
        "task":"offline true waveform short-gap repair; 64ms analysis support guard; observed samples passed through exactly",
        "hidden_original_used":False,"recognizer_used":False,"phase_iterations":args.iterations,"windows":audit["windows"],"device":args.device})
    print(json.dumps({"output":str(out/"repaired.wav"),"observed_pcm_exact":exact}))


if __name__=="__main__": main()
