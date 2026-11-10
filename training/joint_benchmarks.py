"""Reproducible few-shot supports, separate development and locked final inputs."""
import csv
import gzip
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.nn import functional as F

from training.joint_data import G2, frames, image_patches, text_rows


def normalized(text):
    return " ".join(text.casefold().split())


def stratified(labels, task, group):
    """Fixed nested support, with disjoint development rows from official train."""
    indices = []
    ndev = 100 if task=="fashion" else 24
    for label in np.unique(labels):
        ids = np.flatnonzero(labels==label)
        np.random.default_rng(14311+int(label)).shuffle(ids)
        if len(ids)<33 and task!="fsdd":
            raise ValueError("Class too small for the registered support")
        take = ids[:32] if group=="fit" else ids[32:32+ndev]
        indices.extend(take.tolist())
    return np.asarray(indices)


def fashion(root, role):
    split = "t10k" if role=="test" else "train"
    with gzip.open(root/f"{split}-labels-idx1-ubyte.gz") as stream:
        labels = np.frombuffer(stream.read()[8:],np.uint8).astype(np.int64)
    with gzip.open(root/f"{split}-images-idx3-ubyte.gz") as stream:
        pixels = np.frombuffer(stream.read()[16:],np.uint8).reshape(-1,28,28)
    idx = np.arange(len(labels)) if role=="test" else stratified(labels,"fashion",role)
    images = torch.tensor(pixels[idx],dtype=torch.float32)[:,None]/127.5-1
    resized = F.interpolate(images,size=(32,32),mode="bilinear",align_corners=False).expand(-1,3,-1,-1)
    return {"image":image_patches(resized.numpy()),"labels":labels[idx],"groups":labels[idx],"ids":idx}


def banking(root, role):
    def load(split):
        with (root/f"banking_{split}.csv").open(newline="",encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        return list(dict.fromkeys((normalized(r["text"]),r["category"]) for r in rows))
    train = load("train")
    categories = sorted({label for _,label in train})
    lookup = {label:i for i,label in enumerate(categories)}
    train_text = {text for text,_ in train}
    rows = [(text,label) for text,label in load("test") if text not in train_text] if role=="test" else train
    # Exact text with two different labels is ambiguous and excluded prospectively.
    seen, ambiguous = {},set()
    for text,label in rows:
        if text in seen and seen[text]!=label:
            ambiguous.add(text)
        seen[text] = label
    rows = [(text,label) for text,label in rows if text not in ambiguous]
    labels = np.array([lookup[label] for _,label in rows])
    idx = np.arange(len(rows)) if role=="test" else stratified(labels,"banking",role)
    ids,lengths = text_rows([rows[i][0] for i in idx])
    return {"text":ids, "lengths":lengths,"labels":labels[idx],"groups":labels[idx],"ids":idx}


def fsdd(root, role, cfg):
    from vocos.feature_extractors import MelSpectrogramFeatures
    mel = MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100)
    with np.load(G2/"normalizers.npz",allow_pickle=False) as stats:
        mean,scale = stats["audio_mean"],stats["audio_scale"]
    speakers = cfg["fsdd"]["train" if role=="fit" else role]
    with zipfile.ZipFile(root/"fsdd.zip") as archive:
        files = []
        for name in sorted(archive.namelist()):
            if "/recordings/" in name and name.endswith(".wav"):
                digit,speaker,_ = Path(name).stem.split("_")
                if speaker in speakers:
                    files.append((name,int(digit),speaker))
        labels = np.array([y for _,y,_ in files])
        idx = stratified(labels,"fsdd","fit") if role=="fit" else np.arange(len(files))
        audio,groups,durations = [],[],[]
        with torch.inference_mode():
            for i in idx:
                name,_,speaker = files[i]
                rate,wave = wavfile.read(io.BytesIO(archive.read(name)))
                if rate!=8000 or wave.dtype!=np.int16 or wave.ndim!=1:
                    raise ValueError(name)
                sound = resample_poly(wave.astype(np.float32)/32768,3,1)
                spec = mel(torch.from_numpy(sound)[None])[0].T.numpy()
                values = (frames(spec,cfg["signal_steps"])-mean)/scale
                audio.append(values.clip(-10,10))
                groups.append(speaker)
                durations.append(len(wave)/rate)
    return {"audio":np.stack(audio),"labels":labels[idx],"groups":np.array(groups),"ids":idx,
            "duration":np.array(durations,np.float32)}


def benchmark(cfg, task, role):
    study = Path(cfg["output"])
    if role=="test" and not (study/"confirmation_lock.json").exists():
        raise RuntimeError("Confirmation requires all ten frozen endpoints and evaluation sources")
    root = Path(cfg["data"])
    cache = root/"bench"/f"{task}-{role}.npz"
    if not cache.exists():
        cache.parent.mkdir(parents=True,exist_ok=True)
        source = root/"sources"
        if task=="banking":
            data = banking(source,role)
        elif task=="fashion":
            data = fashion(source,role)
        else:
            data = fsdd(source,role,cfg)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp,**data)
        tmp.replace(cache)
    with np.load(cache,allow_pickle=False) as data:
        return dict(data)
