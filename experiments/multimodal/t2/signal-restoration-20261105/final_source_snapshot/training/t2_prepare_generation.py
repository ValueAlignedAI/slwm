"""Immutable high-rate targets and independently frozen fields, T2-G1.

All fitting uses CREMA training people. External media only uses fixed geometry
and existing codecs. Every field encode sees exactly one observed modality.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import zlib

import numpy as np
import torch
import torchaudio

from models.slwm_motion import MotionSLWM
from training.t2_prepare_cremad import sha256
from training.t2_prepare_motion import encode
from training.t2_real_av import write_json, stable_hash


def high_mel(path: Path) -> np.ndarray:
    """Original media -> unnormalized 100 Hz log-power mel [4T,64]."""
    sound=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-vn","-ac","1","-ar","16000",
        "-t","5.12","-f","f32le","-"],capture_output=True,check=True,timeout=45)
    wave=torch.from_numpy(np.frombuffer(sound.stdout,np.float32).copy())
    transform=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=1024,hop_length=160,n_mels=64,center=True,power=2)
    with torch.no_grad(): return transform(wave).clamp_min(1e-8).log().T.numpy()


def packetize(mel: np.ndarray,length: int) -> tuple[np.ndarray,int]:
    """100 Hz [S,64] -> four subframes per physical-time packet [T,256]."""
    missing=max(0,4*length-len(mel))
    if missing>3: raise ValueError("Audio/video duration differs by more than 30 ms")
    if missing: mel=np.pad(mel,((0,missing),(0,0)),mode="edge")
    return mel[:4*length].reshape(length,256),missing


def frozen_fields(model: MotionSLWM,audio: np.ndarray,video: np.ndarray,lengths: np.ndarray) -> dict:
    """[N,128,A/V] -> blind overlap-averaged fields [N,128,256]."""
    outputs={k:np.zeros((*audio.shape[:2],256),np.float32) for k in ("field_audio","field_video")}
    weights=np.zeros((*audio.shape[:2],1),np.float32); windows=[]
    for i,length in enumerate(lengths):
        starts=list(range(0,int(length)-31,16))
        if starts[-1]!=length-32: starts.append(int(length)-32)
        windows.extend((i,s) for s in starts)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for begin in range(0,len(windows),96):
            pairs=windows[begin:begin+96]
            a=torch.from_numpy(np.stack([audio[i,s:s+32] for i,s in pairs])).cuda().float()
            v=torch.from_numpy(np.stack([video[i,s:s+32] for i,s in pairs])).cuda().float()
            one=torch.ones(a.shape[:2],device=a.device,dtype=torch.bool); zero=torch.zeros_like(one)
            for key,aa,vv,am,vm in (("field_audio",a,torch.zeros_like(v),one,zero),("field_video",torch.zeros_like(a),v,zero,one)):
                z=model.encode(aa,vv,am,vm)[0].float().cpu().numpy()
                for j,(i,s) in enumerate(pairs): outputs[key][i,s:s+32]+=z[j]
            for i,s in pairs: weights[i,s:s+32]+=1
    return {k:(v/weights.clip(1)).astype(np.float16) for k,v in outputs.items()}


def prepare(config: str,external: bool=False):
    cfg=json.loads(Path(config).read_text()); root=Path(cfg["corpus"]); root.mkdir(parents=True,exist_ok=True)
    old=Path(cfg["source_corpus"]); torch.set_num_threads(1)
    if sha256(Path(cfg["checkpoint"]))!=cfg["checkpoint_sha256"]: raise ValueError("Frozen core hash mismatch")
    checkpoint=torch.load(cfg["checkpoint"],map_location="cpu",weights_only=False)
    model=MotionSLWM(checkpoint["config"]["run"]["model"]).cuda().eval().requires_grad_(False)
    with np.load(old/"codec.npz",allow_pickle=False) as d: codec={k:d[k] for k in d.files}
    if external:
        source=Path(cfg["external_corpus"]); card=json.loads((source/"dataset_card.json").read_text())
        if card["usable"]!=64 or card["failures"]: raise ValueError("Incomplete external subset")
        rows=[]
        for r in card["records"]:
            path=Path(r["raw_path"])
            if sha256(path)!=r["sha256"] or zlib.crc32(path.read_bytes())!=r["zip_member_crc32"]: raise ValueError("External media integrity failure")
            rows.append({**r,"sentence":"RAVDESS-"+r["statement"],"split":"external"})
        groups={"external":rows}
    else:
        card=json.loads((old/"dataset_card.json").read_text())
        for name,digest in card["files"].items():
            if sha256(old/name)!=digest: raise ValueError("Source corpus changed")
        groups={s:json.loads((old/(s+".json")).read_text()) for s in ("train","validation","test")}
    pending=[]
    for split,rows in groups.items():
        manifest=root/(split+"_manifest.json")
        if manifest.exists():
            existing=json.loads(manifest.read_text())
            if existing["config_hash"]!=stable_hash(cfg): raise ValueError("Use new corpus for changed configuration")
            for name,digest in existing["files"].items():
                if sha256(root/name)!=digest: raise ValueError("Prepared corpus changed")
            continue
        pending.append((split,rows))
    if not pending: return
    cache=root/"decoded"; cache.mkdir(exist_ok=True)
    def decode(row):
        path=cache/(row["sample_id"]+".npz")
        if path.exists():
            with np.load(path,allow_pickle=False) as d: return d["mouth"],d["mel"],int(d["padded"])
        mel=high_mel(Path(row["raw_path"]))
        if external:
            visual=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",row["raw_path"],"-an","-vf",cfg["external_crop"],
                "-frames:v","128","-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45)
            mouth=np.frombuffer(visual.stdout,np.uint8).reshape(-1,32,48).copy()
            length=min(len(mouth),len(mel)//4,128); mouth=mouth[:length]
        else:
            with np.load(old/"decoded"/(row["sample_id"]+".npz"),allow_pickle=False) as d: mouth=d["mouth"][:row["frames"]]
            length=len(mouth)
        if length<32: raise ValueError("Short clip")
        mel,padded=packetize(mel,length)
        temporary=path.with_suffix(".part.npz"); np.savez_compressed(temporary,mouth=mouth,mel=mel,padded=np.array(padded)); temporary.replace(path)
        return mouth,mel,padded
    for split,rows in pending:
        with ThreadPoolExecutor(max_workers=4) as pool:
            decoded=[]
            for i,result in enumerate(pool.map(decode,rows)):
                decoded.append(result)
                if (i+1)%500==0: print(json.dumps({"decoded":i+1,"split":split}),flush=True)
        if split=="train":
            m=np.concatenate([r[1].reshape(-1,64) for r in decoded])
            mean=m.mean(0); scale=m.std(0).clip(.1)
            np.savez_compressed(root/"codec100.npz",mean=mean,scale=scale)
            del m
        else:
            with np.load(root/"codec100.npz",allow_pickle=False) as d: mean,scale=d["mean"],d["scale"]
        n=len(rows); lengths=np.array([len(r[0]) for r in decoded],np.int32)
        arrays={"mouth":np.zeros((n,128,32,48),np.uint8),"mel100":np.zeros((n,128,256),np.float16),
                "audio_coarse":np.zeros((n,128,128),np.float16),"video_coarse":np.zeros((n,128,160),np.float16)}
        if not external:
            with np.load(old/(split+".npz"),allow_pickle=False) as d:
                arrays["audio_coarse"]=d["audio"].astype(np.float16); arrays["video_coarse"]=d["video"].astype(np.float16)
        for i,(mouth,mel,padded) in enumerate(decoded):
            length=len(mouth); arrays["mouth"][i,:length]=mouth
            arrays["mel100"][i,:length]=((mel.reshape(-1,64)-mean)/scale).reshape(length,256)
            if external:
                a,v=encode(mouth,mel.reshape(-1,64)[::4],codec)
                arrays["audio_coarse"][i,:length]=a; arrays["video_coarse"][i,:length]=v
            rows[i]={**rows[i],"frames":length,"audio_edge_padded_subframes":padded}
        del decoded
        arrays.update(frozen_fields(model,arrays["audio_coarse"],arrays["video_coarse"],lengths))
        files={}
        for name,value in {**arrays,"lengths":lengths}.items():
            path=root/f"{split}_{name}.npy"; np.save(path,value); files[path.name]=sha256(path)
        write_json(root/(split+".json"),rows); files[split+".json"]=sha256(root/(split+".json"))
        files["codec100.npz"]=sha256(root/"codec100.npz")
        write_json(root/(split+"_manifest.json"),{"sprint":"T2-G1","config_hash":stable_hash(cfg),"config":cfg,"files":files,
            "samples":len(rows),"people":sorted({r["actor"] for r in rows}),"core_sha256":cfg["checkpoint_sha256"],
            "source_card_sha256":sha256((Path(cfg["external_corpus"]) if external else old)/"dataset_card.json"),
            "codec_fit_split":"train","external_never_fit":external,"frozen_neural_parameters":model.parameter_counts()["total"],
            "field_protocol":"Each modality alone; 32-frame windows stride 16; overlap average; absent channel zero and masked"})
        print(json.dumps({"prepared":split,"samples":n,"padded_audio_subframes":sum(r["audio_edge_padded_subframes"] for r in rows)}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--external",action="store_true")
    args=parser.parse_args(); prepare(args.config,args.external)
