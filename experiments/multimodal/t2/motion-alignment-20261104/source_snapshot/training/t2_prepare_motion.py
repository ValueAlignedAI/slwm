"""Physical-time, decodable speech/audio motion features for T2-M1.

audio [N,128,128], video [N,128,160], lengths [N]. Appearance/motion PCAs
and all statistics fit only training people. No pretrained neural encoder.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import torch

from training.t2_prepare_cremad import REPO,EMOTIONS,SENTENCES,download,sha256


def decode(path: Path,max_frames: int=128) -> tuple[np.ndarray,np.ndarray]:
    """Recorded FLV -> grayscale [T,32,48], 25-Hz log-mel [T,64]."""
    from torchaudio.transforms import MelSpectrogram
    vf="fps=25,crop=iw*0.30:ih*0.24:iw*0.35:ih*0.42,scale=48:32"
    visual=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-an","-vf",vf,
        "-frames:v",str(max_frames),"-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45)
    mouth=np.frombuffer(visual.stdout,np.uint8).reshape(-1,32,48).copy()
    sound=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-vn","-ac","1","-ar","16000",
        "-t",str(max_frames/25),"-f","f32le","-"],capture_output=True,check=True,timeout=30)
    waveform=np.frombuffer(sound.stdout,np.float32).copy()
    if len(mouth)<36 or len(waveform)<36*640 or waveform.std()<1e-6:
        raise ValueError("Clip is too short or lacks valid audio for fixed-time alignment")
    # Centered 64ms windows are allowed for simultaneous translation; this is
    # not a future-prediction dataset. Mel centers are exactly 40ms apart.
    mel=MelSpectrogram(sample_rate=16000,n_fft=1024,hop_length=640,n_mels=64,center=True,power=2)
    with torch.no_grad():
        m=mel(torch.from_numpy(waveform)).clamp_min(1e-8).log().T.numpy()
    length=min(len(m),len(mouth),max_frames)
    return mouth[:length],m[:length].astype(np.float32)


def fit_pca(frames: np.ndarray,dim: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Training [N,P] -> mean [P], orthogonal basis [P,D], scale [D]."""
    x=frames.astype(np.float64)
    mean=x.mean(0); centered=x-mean
    values,vectors=np.linalg.eigh(centered.T@centered/max(1,len(x)-1))
    basis=vectors[:,-dim:][:,::-1]
    scale=np.sqrt(values[-dim:][::-1].clip(1e-6))
    return mean.astype(np.float32),basis.astype(np.float32),scale.astype(np.float32)


def encode(mouth: np.ndarray,mel: np.ndarray,codec: dict) -> tuple[np.ndarray,np.ndarray]:
    """Pixels [T,32,48]/mel [T,64] -> audio [T,128], video [T,160]."""
    pixels=mouth.reshape(len(mouth),-1).astype(np.float32)/255.
    delta=np.diff(pixels,axis=0,prepend=pixels[:1])
    appearance=((pixels-codec["appearance_mean"])@codec["appearance_basis"])/codec["appearance_scale"]
    motion=((delta-codec["motion_mean"])@codec["motion_basis"])/codec["motion_scale"]
    audio=(mel-codec["audio_mean"])/codec["audio_scale"]
    audio_delta=np.diff(mel,axis=0,prepend=mel[:1])
    audio_delta=(audio_delta-codec["audio_delta_mean"])/codec["audio_delta_scale"]
    return np.concatenate([audio,audio_delta],-1),np.concatenate([appearance,motion],-1)


def prepare(path: str) -> dict:
    cfg=json.loads(Path(path).read_text()); out=Path(cfg["output_dir"])
    out.mkdir(parents=True,exist_ok=True)
    if (out/"dataset_card.json").exists():
        card=json.loads((out/"dataset_card.json").read_text())
        if card["config"]!=cfg: raise ValueError("Use a new corpus version for changed configs")
        for name,digest in card["files"].items():
            if sha256(out/name)!=digest: raise ValueError("Prepared motion corpus hash changed")
        print(json.dumps({"reused":str(out)}),flush=True); return card
    revision=cfg["source_revision"]
    for name in ("SentenceFilenames.csv","README.md","LICENSE.txt"):
        (out/name).write_bytes(urllib.request.urlopen(f"https://raw.githubusercontent.com/{REPO}/{revision}/{name}",timeout=30).read())
    names=[r["Filename"] for r in csv.DictReader(open(out/"SentenceFilenames.csv"))]
    splits=cfg["actor_splits"]; groups=[set(v) for v in splits.values()]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)): raise ValueError("People overlap")
    rng=np.random.default_rng(cfg["seed"])
    rows=[]
    for split,actors in splits.items():
        for actor in actors:
            for emotion in EMOTIONS:
                pool=sorted(n for n in names if n.startswith(actor+"_") and n.split("_")[2]==emotion)
                rng.shuffle(pool)
                limit=len(pool) if split=="train" else cfg["evaluation_clips_per_emotion"]
                for name in pool[:limit]:
                    rows.append({"sample_id":name,"actor":actor,"emotion":emotion,"sentence":name.split("_")[1],
                                 "transcript":SENTENCES[name.split("_")[1]],"split":split})
    (out/"selection.json").write_text(json.dumps(rows,indent=2))
    def fetch(row):
        target=out/"raw"/(row["sample_id"]+".flv"); target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            for old in cfg["reuse_raw_dirs"]:
                source=Path(old)/target.name
                if source.is_file():
                    shutil.copy2(source,target); break
        download(f"https://media.githubusercontent.com/media/{REPO}/{revision}/VideoFlash/{target.name}",target)
        return {**row,"raw_path":str(target),"raw_sha256":sha256(target)}
    good,failures=[],[]
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as pool:
        futures={pool.submit(fetch,r):r for r in rows}
        for i,f in enumerate(as_completed(futures)):
            try: good.append(f.result())
            except Exception as exc: failures.append({"sample_id":futures[f]["sample_id"],"stage":"download","error":str(exc)})
            if (i+1)%250==0: print(json.dumps({"downloaded":len(good),"failures":len(failures),"selected":len(rows)}),flush=True)
    good.sort(key=lambda r:r["sample_id"])
    torch.set_num_threads(1)
    kept,mouths,mels=[],[],[]
    def extract(row):
        cache=out/"decoded"/(row["sample_id"]+".npz")
        if cache.exists():
            with np.load(cache,allow_pickle=False) as d: return row,d["mouth"],d["mel"]
        mouth,mel=decode(Path(row["raw_path"]),cfg["max_frames"])
        cache.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(cache,mouth=mouth,mel=mel)
        return row,mouth,mel
    with ThreadPoolExecutor(max_workers=cfg["decode_workers"]) as pool:
        futures={pool.submit(extract,r):r for r in good}
        for i,f in enumerate(as_completed(futures)):
            try:
                row,mouth,mel=f.result(); kept.append({**row,"frames":len(mouth),"seconds":len(mouth)/25})
                mouths.append(mouth); mels.append(mel)
            except Exception as exc: failures.append({"sample_id":futures[f]["sample_id"],"stage":"decode","error":str(exc)})
            if (i+1)%250==0: print(json.dumps({"decoded":len(kept),"failures":len(failures),"total":len(good)}),flush=True)
    (out/"failures.json").write_text(json.dumps(failures,indent=2))
    if len(kept)<.85*len(rows): raise RuntimeError("More than 15% media failed; inspect data before training")
    order=sorted(range(len(kept)),key=lambda i:kept[i]["sample_id"])
    kept=[kept[i] for i in order]; mouths=[mouths[i] for i in order]; mels=[mels[i] for i in order]
    train=[i for i,r in enumerate(kept) if r["split"]=="train"]
    # Fit PCA on reproducibly sampled training frames, never validation/test.
    samples=[]; motion_samples=[]
    for i in train:
        x=mouths[i].reshape(len(mouths[i]),-1).astype(np.float32)/255.
        indices=np.linspace(1,len(x)-1,min(5,len(x)-1)).round().astype(int)
        samples.append(x[indices]); motion_samples.append(x[indices]-x[indices-1])
    codec={}
    for label,frames,dim in (("appearance",samples,96),("motion",motion_samples,64)):
        mean,basis,scale=fit_pca(np.concatenate(frames),dim)
        codec.update({label+"_mean":mean,label+"_basis":basis,label+"_scale":scale.clip(.02 if label=="appearance" else .003)})
    m=np.concatenate([mels[i] for i in train]); dm=np.concatenate([np.diff(mels[i],axis=0,prepend=mels[i][:1]) for i in train])
    codec.update({"audio_mean":m.mean(0),"audio_scale":m.std(0).clip(.1),"audio_delta_mean":dm.mean(0),"audio_delta_scale":dm.std(0).clip(.1)})
    np.savez_compressed(out/"codec.npz",**codec)
    for split in splits:
        indices=[i for i,r in enumerate(kept) if r["split"]==split]
        audio=np.zeros((len(indices),cfg["max_frames"],128),np.float32)
        video=np.zeros((len(indices),cfg["max_frames"],160),np.float32)
        for j,i in enumerate(indices):
            a,v=encode(mouths[i],mels[i],codec); audio[j,:len(a)]=a; video[j,:len(v)]=v
        np.savez_compressed(out/f"{split}.npz",audio=audio,video=video,lengths=np.array([kept[i]["frames"] for i in indices],np.int32))
        (out/f"{split}.json").write_text(json.dumps([kept[i] for i in indices],indent=2))
    card={"sprint":"T2-M1","dataset":"CREMA-D","source_revision":revision,"config":cfg,"actor_splits":splits,
          "selected":len(rows),"usable":len(kept),"failures":failures,
          "split_counts":{s:sum(r["split"]==s for r in kept) for s in splits},
          "hours":{s:sum(r["seconds"] for r in kept if r["split"]==s)/3600 for s in splits},
          "feature_spec":{"fps":25,"sample_rate":16000,"audio_dim":128,"video_dim":160,"max_frames":cfg["max_frames"],
                          "appearance_pca_dim":96,"motion_pca_dim":64,"crop":"x=.35w y=.42h width=.30w height=.24h -> 48x32 gray",
                          "frozen_neural_parameters":0,"fitted_codec_values":sum(v.size for v in codec.values()),
                          "fit_split":"train","task":"simultaneous missing-modality prediction; not causal future prediction"},
          "license":"ODbL-1.0 database / DbCL-1.0 contents; source README/LICENSE retained",
          "files":{p.name:sha256(p) for p in [out/(s+ext) for s in splits for ext in (".npz",".json")]+[out/"codec.npz"]}}
    (out/"dataset_card.json").write_text(json.dumps(card,indent=2)+"\n")
    print(json.dumps({"prepared":str(out),"counts":card["split_counts"],"failures":len(failures)}),flush=True)
    return card


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    prepare(parser.parse_args().config)
