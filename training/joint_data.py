"""Deterministic G3 exposure stream. Only train-role rows enter joint updates."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import tiktoken

from training.general_batch import STEP
from training.general_data import CACHE
from training.general_readout_data import observed_robot_history
from training.scale_text_train import UniqueWindows

G2 = Path("artifacts/predictive_world/g2")
CORPUS = Path("artifacts/text/fineweb_edu_scale_v1")
WIDTHS = {"image": 192, "audio": 100, "numeric": 64}


@lru_cache(maxsize=96)
def mapped(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def read(path):
    return json.loads(Path(path).read_text())


def image_patches(images):
    """RGB[B,3,32,32] in [-1,1] -> 16 raster patches[B,16,192]."""
    return images.reshape(-1, 3, 4, 8, 4, 8).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 16, 192)


def frames(values, steps):
    """Variable-length [frames,100] -> [steps,100] equal-duration bin means."""
    values = values.reshape(-1, 100)
    edges = np.linspace(0, len(values), steps + 1, dtype=int)
    return np.stack([values[a:max(a+1,b)].mean(0, dtype=np.float32) for a,b in zip(edges,edges[1:])])


def thumbnails(images):
    """Color video[T,64,64,3] -> [T,192] low-resolution color frames."""
    return images.reshape(-1,8,8,8,8,3).mean((2,4)).transpose(0,3,1,2).reshape(-1,192).astype(np.float32)/127.5-1


@lru_cache(maxsize=1)
def tokenizer():
    return tiktoken.get_encoding("gpt2")


@lru_cache(maxsize=32768)
def encode(text):
    return tokenizer().encode(text, disallowed_special=())[:64]


def text_rows(strings):
    rows = [encode(x.strip()) for x in strings]
    lengths = np.array([len(x) for x in rows],np.int64)
    ids = np.zeros((len(rows), max(1, max(lengths))),np.int64)
    for i, row in enumerate(rows):
        ids[i,:len(row)] = row
    return ids, lengths


class JointData:
    def __init__(self, cfg, seed, role="train"):
        self.cfg, self.seed, self.role = cfg, seed, role
        self.batch, self.steps = cfg["batch"], cfg["signal_steps"]
        self.cifar = mapped(Path(cfg["data"])/"sources/cifar_images.npy")
        self.families = [f for f,n in cfg["families"].items() for _ in range(n)]
        assert len(self.families) == 100
        self.rows = {}
        for f in ("coco","speech","ami","clotho","pamap2","droid"):
            path = G2/"robot_rows.json" if f=="droid" else CACHE/f/"rows.json"
            rows = [r for r in read(path) if r["role"]==role]
            if f in ("speech", "ami"):
                rows = [r for r in rows if r["length"] >= (100 if f=="ami" else 4)]
            if f=="droid":
                rows = [r for r in rows if r["steps"]>=self.steps+1]
            self.rows[f] = rows
        self.stats = dict(np.load(G2/"normalizers.npz", allow_pickle=False))
        self.text = UniqueWindows(CORPUS / ("train.tokens.bin" if role=="train" else "validation.tokens.bin"), cfg["text_length"], 3079 + seed)

    @lru_cache(maxsize=8)
    def cycle(self, block):
        a = self.families.copy()
        np.random.default_rng(19429 + self.seed + block).shuffle(a)
        return a

    def position(self, step):
        cycle = self.cycle(step//100)
        family = cycle[step%100]
        occurrence = self.cfg["families"][family]*(step//100) + cycle[:step%100].count(family)
        return family, occurrence

    @lru_cache(maxsize=20)
    def order(self, family, epoch, size):
        key = list(self.cfg["families"]).index(family)
        return np.random.default_rng(1000003*self.seed + 997*key + epoch).permutation(size)

    def indices(self, family, occurrence, size):
        # A single pass covers every eligible row before the next epoch starts.
        absolute = occurrence*self.batch + np.arange(self.batch)
        out = np.empty(self.batch,np.int64)
        for epoch in np.unique(absolute//size):
            take = absolute//size == epoch
            out[take] = self.order(family,int(epoch),size)[absolute[take]%size]
        return out

    def audio(self, values):
        return ((values - self.stats["audio_mean"]) / self.stats["audio_scale"]).clip(-10,10).astype(np.float32)

    def get(self, step):
        family, n = self.position(step)
        if family=="text":
            x,y = self.text.batch(n*self.batch,self.batch,"cpu")
            return family, {"text":x,"next_text":y}, [f"window:{n*self.batch+i}" for i in range(self.batch)]
        rng = np.random.default_rng(7919*self.seed+step)
        if family=="cifar":
            idx = self.indices(family,n,len(self.cifar))
            pixels = np.asarray(self.cifar[idx],np.float32)/127.5-1
            return family, {"image":image_patches(pixels)}, [str(i) for i in idx]
        rows = self.rows[family]
        subset = [rows[i] for i in self.indices(family,n,len(rows))]
        ids = [r["id"] for r in subset]
        if family=="coco":
            pixels = np.stack([mapped(CACHE/family/r["array"])[r["index"]] for r in subset])
            rgb = pixels.reshape(-1,32,2,32,2,3).mean((2,4)).transpose(0,3,1,2).astype(np.float32)/127.5-1
            parts = {"image":image_patches(rgb)}
            captions = [rng.choice(r["captions"]) for r in subset]
        else:
            samples = [self.sequence(family,r,rng) for r in subset]
            parts = {k:np.stack([s[k] for s,_ in samples]) for k in samples[0][0]}
            captions = [text for _,text in samples]
        if any(s.strip() for s in captions):
            parts["caption"], parts["lengths"] = text_rows(captions)
            keys = Counter()
            for s in captions:
                val = tuple(encode(s.strip()))
                if val not in keys:
                    keys[val] = len(keys)
            parts["caption_keys"] = np.array([keys[tuple(encode(s.strip()))] for s in captions])
            parts["record_keys"] = np.unique(ids,return_inverse=True)[1]
        return family,parts,ids

    def sequence(self, family, row, rng):
        t = self.steps
        if family=="speech":
            at, count = row["offset"], row["length"]
            mel = mapped(row["mel_file"])[at:at+count]
            return {"audio":self.audio(frames(mel,t))}, row["transcript"].lower()
        if family=="clotho":
            mel = mapped(CACHE/family/row["file"])
            return {"audio":self.audio(frames(mel,t))}, str(rng.choice(row["captions"]))
        if family=="ami":
            width = 4*t
            start = int(rng.integers(row["length"]-width+1))
            if row["spans"]:
                a,b,_ = row["spans"][int(rng.integers(len(row["spans"])))]
                start = int(np.clip((a+b)*.5/STEP-width//2, 0, row["length"]-width))
            a,b = start*STEP,(start+width)*STEP
            caption = " ".join(w for l,h,w in row["spans"] if l>=a and h<=b and "<" not in w)
            mel = mapped(row["mel_file"])[row["offset"]+start:row["offset"]+start+width]
            pixels = mapped(row["pixel_file"])[start:start+width:4]
            rgb = np.repeat(pixels.reshape(t,64,64,1),3,axis=-1)
            return {"audio":self.audio(frames(mel,t)), "image":thumbnails(rgb)},caption
        if family=="pamap2":
            data = mapped(CACHE/family/row["file"])
            stride = max(1, round(row["rate_hz"]*.04))
            start = int(rng.integers(len(data)-stride*t+1))
            raw = data[start:start+stride*t:stride]
            vals = np.zeros((t,64),np.float32)
            mask = np.zeros((t,64),bool)
            mask[:,:33] = np.isfinite(raw)
            vals[:,:33] = np.where(mask[:,:33],(raw-self.stats["numeric_mean"][:33])/self.stats["numeric_scale"][:33],0)
            return {"numeric":vals.clip(-10,10),"present":mask},""
        folder = CACHE/"droid_g2"/row["id"]
        start = int(rng.integers(row["steps"]-t+1))
        with np.load(folder/"trajectory.npz",allow_pickle=False) as traj:
            vals, mask, _ = observed_robot_history(traj,start,t,self.stats)
        rgb = mapped(folder/"rgb.npy")[start:start+t,0]
        caption = str(rng.choice(row["captions"])) if row["captions"] else ""
        return {"numeric":vals.clip(-10,10),"present":mask,"image":thumbnails(rgb)},caption
