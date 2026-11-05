"""Observed-waveform constrained inversion; hidden samples are erased first."""
from __future__ import annotations
from pathlib import Path
import subprocess
import wave
import numpy as np
import torch
import torchaudio


def decode_pcm(path: Path,max_seconds=5.12) -> np.ndarray:
    data=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-vn","-ac","1","-ar","16000","-t",str(max_seconds),"-f","s16le","-"],capture_output=True,check=True,timeout=45).stdout
    return np.frombuffer(data,dtype="<i2").copy()


def write_pcm(path: Path,sound: np.ndarray):
    samples=np.rint(np.clip(sound,-1.,32767/32768)*32768).astype("<i2")
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(samples.tobytes())


def mel100(waveform: torch.Tensor) -> torch.Tensor:
    """Recorded [N] at16kHz -> physical log-power [S,64], FFT1024/hop160."""
    transform=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=1024,hop_length=160,n_mels=64,power=2.,center=True).to(waveform.device)
    return transform(waveform).clamp_min(1e-8).log().T


def constrained_audio(predicted: np.ndarray,recorded: np.ndarray,missing: np.ndarray,codec: dict,iterations=64,device="cpu") -> np.ndarray:
    """Predicted normalized [S,64], waveform[N], missing[N] -> repair[N].

    All hidden samples are discarded inside this function before any transform.
    No reference phase or hidden sample is available. Observed samples are hard
    constraints at every iteration and preserved exactly in the returned floats.
    """
    if not missing.any(): return recorded.copy()
    original=torch.as_tensor(recorded,dtype=torch.float32,device=device); hidden=torch.as_tensor(missing,dtype=torch.bool,device=device)
    observed=original.masked_fill(hidden,0.); window=torch.hann_window(1024,device=device)
    def stft(x): return torch.stft(x,1024,hop_length=160,win_length=1024,window=window,center=True,return_complex=True)
    with torch.inference_mode():
        visible_spectrum=stft(observed); frames=visible_spectrum.shape[1]
        features=torch.as_tensor(predicted,dtype=torch.float32,device=device)
        if len(features)<frames: features=torch.cat([features,features[-1:].expand(frames-len(features),-1)])
        physical=features[:frames]*torch.as_tensor(codec["scale"],device=device)+torch.as_tensor(codec["mean"],device=device)
        banks=torchaudio.functional.melscale_fbanks(513,0,8000,64,16000).to(device)
        magnitude=(physical.clamp(-18,15).exp()@torch.linalg.pinv(banks)).clamp_min(1e-9).sqrt().T
        # Mark every FFT window whose support intersects erased samples. The
        # interval can be arbitrary, including more than one true sample gap.
        times=torch.arange(frames,device=device)*160; prefix=torch.cat([hidden.new_zeros(1,dtype=torch.long),hidden.long().cumsum(0)])
        left=(times-512).clamp(0,len(hidden)); right=(times+512).clamp(0,len(hidden))
        unknown=(prefix[right]-prefix[left])>0
        magnitude=torch.where(unknown[None],magnitude,visible_spectrum.abs())
        generator=torch.Generator(device=device).manual_seed(17); random=torch.rand(visible_spectrum.shape,generator=generator,device=device)*2*torch.pi
        phase=torch.where(visible_spectrum.abs()>1e-8,visible_spectrum/visible_spectrum.abs().clamp_min(1e-8),torch.polar(torch.ones_like(random),random))
        result=observed
        for _ in range(iterations):
            candidate=torch.istft(magnitude*phase,1024,hop_length=160,win_length=1024,window=window,center=True,length=len(observed))
            result=torch.where(hidden,candidate.clamp(-1,32767/32768),observed)
            spectrum=stft(result); phase=spectrum/spectrum.abs().clamp_min(1e-8)
        result=torch.where(hidden,result,observed)
    return result.cpu().numpy()
