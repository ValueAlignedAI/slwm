"""Build a speaker-disjoint, original-waveform direction bank from public Speech Commands V2."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import wave
import numpy as np
from g13_acquire import ROOT,file_hash,write

WORDS=("left","right","up","down")
SHA="af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
def features(pcm):
    data=np.zeros(16000,np.float32);n=min(len(pcm),16000);data[:n]=pcm[:n]
    idx=np.arange(64)[:,None]*244+np.arange(400)
    f=np.fft.rfft(data[idx]*np.hanning(400),n=512);p=np.abs(f)**2
    hz=np.fft.rfftfreq(512,1/16000)
    edges=700*(10**(np.linspace(2595*np.log10(1+60/700),2595*np.log10(1+7700/700),42)/2595)-1)
    a,b,c=edges[:-2],edges[1:-1],edges[2:]
    mel=np.maximum(0,np.minimum((hz[:,None]-a)/(b-a),(c-hz[:,None])/(c-b)))
    log=np.log10(np.maximum(p@mel,1e-7))
    return np.clip((log+5.5)/7.0,0,1).astype(np.float16)


def main():
    root=ROOT/"speech";path=root/"speech_commands_v0.02.tar.gz"
    if file_hash(path)!=SHA:raise ValueError("Unexpected original speech archive")
    feats=[];labels=[];names=[];speakers=[];lists={}
    with tarfile.open(path,"r|gz") as tar:
        for item in tar:
            name=item.name.removeprefix("./")
            if name in ("validation_list.txt","testing_list.txt"):
                lists[name]=set(tar.extractfile(item).read().decode().splitlines());continue
            parts=name.split("/")
            if len(parts)!=2 or parts[0] not in WORDS or not name.endswith(".wav"):continue
            raw=tar.extractfile(item).read()
            with wave.open(io.BytesIO(raw)) as wav:
                if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(16000,1,2):raise ValueError("Unknown recording codec")
                pcm=np.frombuffer(wav.readframes(wav.getnframes()),"<i2").astype(np.float32)/32768
            feats.append(features(pcm));labels.append(WORDS.index(parts[0]));names.append(name);speakers.append(parts[1].split("_nohash_")[0])
    roles=np.asarray(["test" if x in lists["testing_list.txt"] else "dev" if x in lists["validation_list.txt"] else "train" for x in names])
    speakers=np.array(speakers);labels=np.array(labels,np.int32);names=np.array(names)
    if any(len(set(roles[speakers==s]))!=1 for s in set(speakers)):raise ValueError("Overlapping speech speaker")
    file=root/"directions.npz";np.savez_compressed(file,x=np.stack(feats),y=labels,names=names,speaker=speakers,role=roles)
    write(root/"directions.json",dict(archive_sha256=SHA,features_sha256=file_hash(file),words=WORDS,
         splits={r:dict(clips=int((roles==r).sum()),speakers=len(set(speakers[roles==r]))) for r in ("train","dev","test")}))
    print("Frozen speech features",len(names),len(set(speakers)),flush=True)

if __name__=="__main__":main()
