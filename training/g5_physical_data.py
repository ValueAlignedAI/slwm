"""Build disjoint, synchronized contact recordings and replay-checked Push-T interventions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

import numpy as np

from training.g4_data import DT, T, digest, number, quant

CFG = Path("configs/t5/joint_world.json")
SCALE = 64.0


def save_npz(path, arrays):
    with path.with_suffix(".part").open("wb") as out:
        np.savez(out, **arrays)
    path.with_suffix(".part").replace(path)


def simulate(arg):
    seed, target = arg
    out = target / f"{seed}.npz"
    if out.exists():
        return
    import gym_pusht
    from gym_pusht.envs.pusht import PushTEnv

    env = PushTEnv(obs_type="state", render_mode="rgb_array", observation_width=16, observation_height=16)
    rng = np.random.default_rng(seed)
    block = rng.uniform(125, 385, 2)
    angle = rng.uniform(-np.pi, np.pi)
    approach = rng.uniform(0, 2*np.pi)
    direction = np.array([np.cos(approach), np.sin(approach)])
    agent = block - rng.uniform(65, 116)*direction
    start = np.r_[agent, block, angle]
    entry = block - rng.uniform(12, 52)*direction
    prefix = np.linspace(agent, entry, 16)
    branch_angles = approach + rng.uniform(-.32, .32, 3) + np.array([0, 1.9, -1.9])
    outcomes, plans = [], []
    for branch, heading in enumerate(branch_angles):
        env.reset(seed=seed, options={"reset_to_state":start.copy()})
        poses, video = [], []
        for command in prefix:
            pose, *_ = env.step(command)
            poses.append(pose)
            if branch == 0:
                image = env._render(visualize=False)
                video.append(np.rint(image @ np.array([.299, .587, .114])).astype(np.uint8).reshape(-1))
        poses = np.stack(poses)
        if branch == 0:
            history, frames = poses.copy(), np.stack(video)
        else:
            np.testing.assert_allclose(poses, history, atol=1e-7, rtol=0)
        end = np.clip(entry + rng.uniform(72, 130)*np.array([np.cos(heading),np.sin(heading)]), 38, 474)
        plan = np.linspace(entry, end, 17)[1:]
        future = np.stack([env.step(command)[0] for command in plan])
        plans.append(plan)
        outcomes.append((future[:,2:4] - history[-1,2:4])/SCALE)
    env.close()
    history = np.column_stack([history[:,:4]/256-1, np.sin(history[:,4]), np.cos(history[:,4])])
    save_npz(out, {"n":history[None].astype(np.float32), "v":frames[None],
                   "plan":(np.array(plans)[None]/256-1).astype(np.float32),
                   "outcome":np.array(outcomes, dtype=np.float32)[None], "group":np.array([seed],np.int64)})


def impact(arg):
    archive, stem, target = arg
    out = target / (stem+".npz")
    if out.exists():
        return
    import soundfile as sf
    import torch
    import torchaudio
    from vocos.feature_extractors import MelSpectrogramFeatures

    global _mel
    if "_mel" not in globals():
        torch.set_num_threads(1)
        _mel = MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100)
    base = "vis-data-256/"+stem
    with zipfile.ZipFile(archive) as z:
        labels = z.read(base+"_times.txt").decode().splitlines()
        audio, sr = sf.read(io.BytesIO(z.read(base+"_denoised.wav")), dtype="float32")
        with tempfile.NamedTemporaryFile(suffix=".mp4",dir=target) as video:
            video.write(z.read(base+"_mic.mp4"))
            video.flush()
            raw = subprocess.check_output(["ffmpeg","-v","error","-nostdin","-threads","1","-i",video.name,
                        "-an","-filter_threads","1","-vf","fps=375/16,scale=16:16:flags=area,format=gray",
                        "-threads","1","-f","rawvideo","pipe:1"])
    if audio.ndim==2:
        audio = audio.mean(1)
    wave = torch.from_numpy(audio)
    if sr!=24000:
        wave = torchaudio.functional.resample(wave,sr,24000)
    with torch.inference_mode():
        feat = _mel(wave[None])[0].T.numpy()
    frames = np.frombuffer(raw,dtype=np.uint8).reshape(-1,256)
    ticks = min(len(frames),len(wave)//1024,len(feat)//4)
    count = ticks//T
    size = count*T
    event = np.zeros(size,np.uint8)
    for line in labels:
        parts = line.split()
        if len(parts)>=2 and parts[1].lower()!="none":
            ix = math.floor(float(parts[0])/DT)
            if 0<=ix<size:
                event[ix]=1
    day = stem[:10]
    save_npz(out, {"a":quant(feat[:size*4].reshape(size,4,100).mean(1)).reshape(count,T,100),
                   "v":frames[:size].reshape(count,T,256), "event":event.reshape(count,T),
                   "group":np.full(count,number(day)%(2**31-1),np.int64),
                   "recording":np.full(count,number(stem),np.uint64)})


def build(root, family, workers):
    cfg=json.loads(CFG.read_bytes())
    dest=root/family
    dest.mkdir(exist_ok=True,parents=True)
    by_role={}
    if family=="impact":
        archive=root/"sources/vis-data-256.zip"
        if digest(archive)!="a5b0a890c4728cfa39cba95e283f72b4ebe9fd91322016bcad66b25e3e667d17":
            raise ValueError("Impact archive differs from the CRC-verified release")
        with zipfile.ZipFile(archive) as z:
            stems=sorted(Path(n).name.removesuffix("_times.txt") for n in z.namelist() if n.endswith("_times.txt"))
        days=sorted({s[:10] for s in stems},key=lambda s:hashlib.sha256(s.encode()).digest())
        reserve=math.ceil(len(days)*.1)
        role_days={"train":days[:-2*reserve],"dev":days[-2*reserve:-reserve],"test":days[-reserve:]}
        for role,chosen in role_days.items():
            folder=dest/("records-"+role)
            folder.mkdir(exist_ok=True)
            by_role[role]=[(archive,s,folder) for s in stems if s[:10] in chosen]
    else:
        for role,(lo,hi) in cfg["split"]["pusht"].items():
            folder=dest/("records-"+role)
            folder.mkdir(exist_ok=True)
            by_role[role]=[(seed,folder) for seed in range(lo,hi)]
    for role,args in by_role.items():
        folder=dest/role
        if (folder/"card.json").is_file():
            continue
        folder.mkdir(exist_ok=True)
        worker=impact if family=="impact" else simulate
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,_ in enumerate(pool.map(worker,args,chunksize=2)):
                if (i+1)%128==0:
                    print(family,role,i+1,len(args),flush=True)
        records=sorted(args[0][-1].glob("*.npz"))
        if len(records)!=len(args):
            raise ValueError((role,len(records),len(args)))
        with np.load(records[0]) as sample:
            schema={k:(v.shape[1:],v.dtype) for k,v in sample.items()}
        sizes=[]
        for path in records:
            with np.load(path) as data:
                sizes.append(len(data["group"]))
        total=sum(sizes)
        arrays={k:np.lib.format.open_memmap(folder/(k+".npy"),mode="w+",dtype=t,shape=(total,*s))
                for k,(s,t) in schema.items()}
        at=0
        for path,size in zip(records,sizes):
            with np.load(path) as data:
                for k,values in arrays.items():
                    values[at:at+size]=data[k]
            at+=size
        for values in arrays.values():
            values.flush()
        card={"family":family,"split":role,"examples":total,"records":len(records),
              "groups":int(len(np.unique(arrays["group"]))),"sha256":{k:digest(folder/(k+".npy")) for k in arrays}}
        pending=folder/"card.tmp"
        pending.write_text(json.dumps(card,sort_keys=True,indent=2)+"\n")
        pending.replace(folder/"card.json")
        print(json.dumps(card),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("family",choices=("impact","pusht"))
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--workers",default=4,type=int)
    a=p.parse_args()
    build(a.root,a.family,a.workers)
