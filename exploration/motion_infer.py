"""Diagnostic cross-modal inference on a supplied audio or silent-video file."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from models.slwm_motion import MotionSLWM
from training.t2_prepare_cremad import sha256
from training.t2_prepare_motion import encode
from training.t2_real_av import write_json
from exploration.motion_outputs import predict_sequence,inverse_appearance,inverse_motion,invert_mel,save_video,save_wave


def audio_features(path: Path,codec: dict) -> np.ndarray:
    """Read only audio -> physical-time [T,128]; never reads video."""
    from torchaudio.transforms import MelSpectrogram
    out=subprocess.run(["ffmpeg","-v","error","-i",str(path),"-vn","-ac","1","-ar","16000","-t","5.12","-f","f32le","-"],capture_output=True,check=True,timeout=30)
    wave=np.frombuffer(out.stdout,np.float32).copy()
    mel=MelSpectrogram(sample_rate=16000,n_fft=1024,hop_length=640,n_mels=64,center=True,power=2)
    with torch.no_grad(): m=mel(torch.from_numpy(wave)).clamp_min(1e-8).log().T.numpy()[:128]
    dummy=np.zeros((len(m),32,48),np.uint8)
    return encode(dummy,m,codec)[0]


def video_features(path: Path,codec: dict) -> np.ndarray:
    """Read only video -> physical-time [T,160]; never reads its audio track."""
    vf="fps=25,crop=iw*0.30:ih*0.24:iw*0.35:ih*0.42,scale=48:32"
    out=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-an","-vf",vf,"-frames:v","128",
        "-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=30)
    mouth=np.frombuffer(out.stdout,np.uint8).reshape(-1,32,48).copy()
    return encode(mouth,np.broadcast_to(codec["audio_mean"],(len(mouth),64)).copy(),codec)[1]


def infer(checkpoint_path: str,input_path: str,mode: str,output: str,head: str="cross",device_name: str="cuda") -> dict:
    device=torch.device(device_name); torch.set_num_threads(4)
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    cfg=checkpoint["config"]["matrix"]; model_cfg=checkpoint["config"]["run"]["model"]
    corpus=Path(cfg["corpus"]); card_path=corpus/"dataset_card.json"
    if sha256(card_path)!=checkpoint["provenance"]["dataset_card_sha256"]: raise ValueError("Checkpoint/codec corpus mismatch")
    card=json.loads(card_path.read_text())
    if sha256(corpus/"codec.npz")!=card["files"]["codec.npz"]: raise ValueError("Codec changed")
    with np.load(corpus/"codec.npz",allow_pickle=False) as archive: codec={k:archive[k] for k in archive.files}
    path=Path(input_path)
    if not path.is_file(): raise FileNotFoundError(path)
    if mode=="audio":
        a=audio_features(path,codec); v=np.zeros((len(a),160),np.float32)
    else:
        v=video_features(path,codec); a=np.zeros((len(v),128),np.float32)
    if len(a)<32: raise ValueError("This prototype requires at least 1.28 seconds of input")
    model=MotionSLWM(model_cfg).to(device).eval(); model.load_state_dict(checkpoint["model"])
    prediction=predict_sequence(model,torch.from_numpy(a).to(device),torch.from_numpy(v).to(device),mode)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"predictions.npz",**prediction)
    selected=("video" if mode=="audio" else "audio") if head=="cross" else head
    if selected in {"audio","all"}: save_wave(out/"generated_audio.wav",invert_mel(prediction["audio"][:,:64],codec))
    if selected in {"video","all"}:
        save_video(out/"generated_mouth.gif",inverse_appearance(prediction["video"][:,:96],codec),f"MODEL: {mode} -> mouth appearance (diagnostic)")
        save_video(out/"generated_motion.gif",inverse_motion(prediction["video"][:,96:],codec),f"MODEL: {mode} -> motion on mean-face prior")
    report={"input":str(path),"input_sha256":sha256(path),"input_modality":mode,"selected_head":selected,
        "checkpoint":checkpoint_path,"checkpoint_sha256":sha256(Path(checkpoint_path)),"frames":len(a),"seconds":len(a)/25,
        "output_tag":"uncommitted_diagnostic_prediction","outputs":sorted(p.name for p in out.iterdir()),
        "limitations":"Fixed lower-face crop; training-mean appearance ambiguity; inverse PCA / estimated phase; no general language or realistic-media claim"}
    write_json(out/"inference.json",report); return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True)
    group=parser.add_mutually_exclusive_group(required=True); group.add_argument("--audio"); group.add_argument("--video")
    parser.add_argument("--output",required=True); parser.add_argument("--head",choices=("cross","audio","video","all"),default="cross")
    parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    print(json.dumps(infer(args.checkpoint,args.audio or args.video,"audio" if args.audio else "video",args.output,args.head,args.device),indent=2))
