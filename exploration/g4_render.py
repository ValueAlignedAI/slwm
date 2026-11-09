"""Render the actual live decoder outputs. The frozen Vocos output codec is accounted for separately."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import torch
from vocos import Vocos
from exploration.stream_media import synthesize


def pcm(codec, coarse):
    # Interpolate the model's packet-rate log-mels onto Vocos's 256-sample hop.
    n=len(coarse)*4
    x=(np.arange(n)+.5)/4-.5
    fine=np.stack([np.interp(x,np.arange(len(coarse)),band) for band in coarse.T],axis=1).astype(np.float32)
    identity={"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)}
    signal,_=synthesize(fine.reshape(-1,400),identity,codec,device="cpu",context=48,chunk_frames=512)
    return np.rint(np.clip(signal,-1,32767/32768)*32768).astype("<i2")


def wav(path,samples):
    with wave.open(str(path),"wb") as out:
        out.setparams((1,2,24000,0,"NONE","not compressed")); out.writeframes(samples.tobytes())


def video(path,frames,audio=None):
    cmd=["ffmpeg","-y","-v","error","-f","rawvideo","-pixel_format","gray","-video_size","16x16",
         "-framerate","375/16","-i","pipe:0"]
    if audio: cmd+=["-i",str(audio),"-c:a","aac","-b:a","64k"]
    else: cmd+=["-an"]
    encoders=subprocess.check_output(["ffmpeg","-hide_banner","-encoders"],stderr=subprocess.DEVNULL,text=True)
    coding=["-c:v","libx264","-preset","fast","-crf","22"] if " libx264 " in encoders else ["-c:v","mpeg4","-q:v","3"]
    cmd+=["-vf","scale=256:256:flags=neighbor","-pix_fmt","yuv420p",*coding,"-threads","2","-movflags","+faststart",str(path)]
    subprocess.run(cmd,input=frames.tobytes(),check=True)


def render(root,codec_dir):
    d=np.load(root/"generated.npz",allow_pickle=False)
    torch.set_num_threads(3)
    model=Vocos.from_hparams(str(codec_dir/"config.yaml"))
    model.load_state_dict(torch.load(codec_dir/"pytorch_model.bin",map_location="cpu",weights_only=True))
    model.eval().requires_grad_(False)
    video(root/"audio-to-video.mp4",d["audio_to_video"])
    for key in ("video","text"):
        wav(root/(key+"-to-audio.wav"),pcm(model,d[key+"_to_logmel"].astype(np.float32)))
    video(root/"text-to-audio-video.mp4",d["text_to_video"],root/"text-to-audio.wav")
    subprocess.run(["ffmpeg","-y","-v","error","-i",root/"video-to-audio.wav","-c:a","aac","-b:a","64k",root/"video-to-audio.m4a"],check=True)
    files={}
    for name in ("audio-to-video.mp4","video-to-audio.wav","video-to-audio.m4a","text-to-audio.wav","text-to-audio-video.mp4"):
        path=root/name
        duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",path],text=True))
        with path.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
        files[name]={"duration_s":duration,"bytes":path.stat().st_size,"sha256":sha}
    result={"codec":"fixed Vocos mel-24khz, 4 interpolated log-mel frames per model tick",
            "video_native_hw":[16,16],"files":files}
    (root/"media.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(files),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--live",type=Path,required=True); p.add_argument("--codec",type=Path,required=True)
    a=p.parse_args(); render(a.live,a.codec)
