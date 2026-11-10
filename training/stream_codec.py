"""Explicit audio/video/phoneme edges for continuous shared-field experiments."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
import torchaudio
import cmudict
from vocos import Vocos
from vocos.feature_extractors import MelSpectrogramFeatures

SAMPLE_RATE=24000
HOP=256
PACKET_SAMPLES=1024
PACKET_SECONDS=PACKET_SAMPLES/SAMPLE_RATE
VOCODER=Path("artifacts/decoders/vocos-mel-24khz")


@lru_cache(maxsize=1)
def pronunciation_codec():
    """Fixed external pronunciation dictionary, not a learned held-out vocabulary."""
    dictionary=cmudict.dict(); names=["<sil>","<unk>"]+sorted(p for p,_ in cmudict.phones()); ids={p:i for i,p in enumerate(names)}
    return dictionary,names,ids


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())


def pronounce(word):
    dictionary,_,ids=pronunciation_codec(); entries=dictionary.get(word.lower())
    if not entries: return [1]
    return [ids.get(re.sub(r"\d","",p),1) for p in entries[0]]


def timed_phones(spans,length):
    """Timed words -> coarse phones[length], uniform within each word (declared approximation)."""
    result=np.zeros(length,np.int16); times=(np.arange(length)*4+1.5)*HOP/SAMPLE_RATE
    for start,end,word in spans:
        if end<=start: continue
        phones=pronounce(word); use=(times>=start)&(times<end); offset=((times[use]-start)/(end-start)*len(phones)).astype(int).clip(0,len(phones)-1)
        result[use]=np.asarray(phones)[offset]
    return result


def character_ids(text):
    alphabet=" 'abcdefghijklmnopqrstuvwxyz"; normalized=" ".join(words(text))
    return [alphabet.index(c)+1 for c in normalized if c in alphabet]


def mel_features(sound):
    """Float PCM[B,N] at24kHz -> physical natural-log magnitudes[B,100,F]."""
    extractor=MelSpectrogramFeatures(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=HOP,n_mels=100)
    return extractor.to(sound.device)(sound)


def load_vocoder(device="cpu"):
    model=Vocos.from_hparams(str(VOCODER/"config.yaml")); state=torch.load(VOCODER/"pytorch_model.bin",map_location="cpu",weights_only=True); model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False)


def support_visibility(missing,packets):
    """True waveform mask[N] -> fine/coarse observation masks, respecting FFT support."""
    missing=np.asarray(missing,bool); centers=np.arange(packets*4)*HOP; lo=(centers-512).clip(0,len(missing)); hi=(centers+513).clip(0,len(missing)); integral=np.r_[0,np.cumsum(missing,dtype=np.int64)]
    fine=integral[hi]==integral[lo]; return fine,fine.reshape(packets,4).all(1)


def observed_mel(sound,missing,packets):
    """Erase hidden samples before FFT; no input/phase path can read the erased waveform."""
    erased=np.asarray(sound,np.float32).copy(); erased[missing]=0.; features=mel_features(torch.from_numpy(erased)[None])[0].T.numpy(); features=features[:packets*4].astype(np.float16).astype(np.float32)
    if len(features)<packets*4: features=np.pad(features,((0,packets*4-len(features)),(0,0)),mode="edge")
    fine,coarse=support_visibility(missing,packets); known=np.flatnonzero(fine); unknown=np.flatnonzero(~fine)
    if not len(known): raise ValueError("No observed sound remains")
    for band in range(100): features[unknown,band]=np.interp(unknown,known,features[known,band])
    return features.reshape(packets,400),fine,coarse


def estimated_text(text,duration_model):
    """Plain text -> estimated timed phones, never reference word timings."""
    spans=[]; time=0.
    for word in words(text):
        phones=pronounce(word); x=np.array([1.,np.log(len(phones)),np.log(max(1,len(word)))])
        duration=float(np.exp(x@np.asarray(duration_model["coefficients"])).clip(.08,1.5)); spans.append((time,time+duration,word)); time+=duration+duration_model["interword_gap"]
    length=max(32,int(np.ceil(time/PACKET_SECONDS))); return timed_phones(spans,length),spans,length


def dictionary_hash():
    d,n,_=pronunciation_codec(); return hashlib.sha256(json.dumps({"dictionary":d,"phones":n},sort_keys=True).encode()).hexdigest()
