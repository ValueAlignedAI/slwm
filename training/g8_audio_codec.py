"""Encode and decode mono 24-kHz speech with the released G8 fixed-payload acoustic model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from training.g8_rate_codec import BUDGETS,quantize,pack,unpack
from training.g4_data import quant
from scripts.train.g7_lock_heads import sha

PUBLIC=Path(__file__).resolve().parents[1]/"experiments/multimodal/t8/factorial-transfer-20261110/inference/allocated-acoustic.npz"
MAGIC=b"SLWMAC1\n"


def weights(file):
    with np.load(file,allow_pickle=False) as f:b={k:np.array(v) for k,v in f.items()}
    if b["basis"].shape!=(200,64) or b["centers"].shape!=(64,9,256):raise ValueError("Unsupported acoustic model")
    return b


def encode_audio(path,dest,file,budget):
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from vocos.feature_extractors import MelSpectrogramFeatures
    from training.broad_data import physical_mel
    b=weights(file);pcm,sr=sf.read(path,dtype="float32",always_2d=True)
    signal=pcm.mean(1)
    if not len(signal) or not np.isfinite(signal).all():raise ValueError("A nonempty finite recording is required")
    if sr!=24000:
        gcd=math.gcd(sr,24000);signal=resample_poly(signal,24000//gcd,sr//gcd).astype(np.float32)
    samples=len(signal);pairs=(samples+511)//512
    padded=np.pad(signal,(0,max(1024,pairs*512)-samples),mode="edge").astype(np.float32)
    extractor=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100).eval()
    with torch.inference_mode():
        features=physical_mel(torch.from_numpy(padded)[None],extractor)[:2*pairs].numpy()
    vectors=quant(features).reshape(pairs,200).astype(np.float32)/16-12
    alloc=b["allocated"+str(budget)];parts=[]
    for j in range(0,pairs,2048):
        pc=(vectors[j:j+2048]-b["mean"])@b["basis"]
        codes,_=quantize(pc,b,alloc);parts.append(pack(codes,alloc))
    payload=b"".join(parts)
    if len(payload)!=(budget*pairs+7)//8:raise ValueError("Packed byte count disagrees with rate")
    header={"version":1,"weights_sha256":sha(file),"payload_sha256":hashlib.sha256(payload).hexdigest(),"bits_per_pair":budget,"frame_pairs":pairs,"samples":samples,"sample_rate":24000}
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as f:
        temp=Path(f.name)
        try:f.write(MAGIC+json.dumps(header,separators=(",",":")).encode()+b"\n"+payload);f.flush();os.fsync(f.fileno())
        except BaseException:temp.unlink(missing_ok=True);raise
    temp.replace(dest);return header


def decode_audio(path,dest,file,vocoder,device):
    import soundfile as sf
    from exploration.stream_media import load_renderer,synthesize
    with path.open("rb") as f:
        if f.readline(len(MAGIC)+1)!=MAGIC:raise ValueError("Wrong acoustic stream signature")
        header=json.loads(f.readline(2048));payload=f.read()
    bits=header.get("bits_per_pair");n=header.get("frame_pairs");size=header.get("samples")
    if header.get("version")!=1 or header.get("sample_rate")!=24000 or bits not in BUDGETS or type(size)!=int or size<=0 or n!=(size+511)//512:raise ValueError("Invalid acoustic stream header")
    if sha(file)!=header["weights_sha256"] or hashlib.sha256(payload).hexdigest()!=header["payload_sha256"]:raise ValueError("Acoustic model or payload checksum mismatch")
    b=weights(file);alloc=b["allocated"+str(bits)];block=2048;parts=[]
    for start in range(0,n,block):
        count=min(block,n-start);lo=start*bits//8;hi=lo+(count*bits+7)//8
        codes=unpack(payload[lo:hi],alloc,count)
        pcs=np.column_stack([b["centers"][j,k,codes[:,j]] for j,k in enumerate(alloc)])
        parts.append((pcs@b["basis"].T+b["mean"]).reshape(-1,100).clip(-12,3.9375))
    if len(payload)!=(n*bits+7)//8:raise ValueError("Unexpected trailing or missing audio payload")
    mel=np.concatenate(parts)
    decoder=load_renderer(vocoder,device);signal,_=synthesize(mel,{"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)},decoder,device=device)
    if len(signal)<size:raise ValueError("Decoder did not preserve the physical sample clock")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent,suffix=".wav",delete=False) as f:temp=Path(f.name)
    try:sf.write(temp,signal[:size],24000,format="WAV",subtype="PCM_16");temp.replace(dest)
    finally:temp.unlink(missing_ok=True)
    return header


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("encode","decode"));p.add_argument("input",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--weights",type=Path,default=PUBLIC);p.add_argument("--bits-per-pair",choices=BUDGETS,type=int,default=48)
    p.add_argument("--vocoder",type=Path);p.add_argument("--device",choices=("cpu","cuda"),default="cpu");a=p.parse_args()
    if a.input.resolve()==a.output.resolve():p.error("Input and output must have different paths")
    if a.output.resolve()==a.weights.resolve():p.error("Output would replace the trained model")
    receipt=encode_audio(a.input,a.output,a.weights,a.bits_per_pair) if a.action=="encode" else decode_audio(a.input,a.output,a.weights,a.vocoder,a.device)
    print(json.dumps(receipt,sort_keys=True))
