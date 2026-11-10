"""Generate forty-three seconds of continuous paired media from one observed edge."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

T=64
DT=1024/24000
PROMPTS=("A stick taps metal repeatedly.","Rain falls on a wooden surface.","Someone drags an object across stone.","Water splashes.")


def predict(root,run,out):
    import jax
    import jax.numpy as jnp
    from models.g4_shared import Spec
    from models.g5_shared import encode
    from training.g4_train import sha_file
    card=json.loads((run/"config.json").read_bytes())
    digest=json.loads((run/"complete.json").read_bytes())["sha256"]["inference.npz"]
    if sha_file(run/"inference.npz")!=digest:
        raise ValueError("Inference hash mismatch")
    spec=Spec(**card["spec"])
    with np.load(run/"inference.npz") as f:
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
    record=None
    for path in sorted((root/"impact/records-dev").glob("*.npz")):
        with np.load(path) as f:
            if len(f["v"])>=16:
                rec={k:np.array(f[k][:16]) for k in ("a","v","event")}
                record=path.stem
                break
    if record is None:
        raise ValueError("No sufficiently long development recording")
    def step(b,src,mem,offset):
        z,m=encode(p,spec,b,src,memory=mem,offset=offset)
        pred=(z@p["readout"].astype(z.dtype)).astype(jnp.float32)
        return pred[0,:,:356],m
    step=jax.jit(step)
    outputs={}
    for view,ix in (("audio",0),("video",1),("text",3),("text_blank",3)):
        predictions=[]
        memory=None
        src=np.eye(4,dtype=np.uint8)[ix:ix+1]
        for j in range(16):
            a=rec["a"][j].astype(np.float32)
            v=rec["v"][j].astype(np.float32)
            last_a=rec["a"][j-1,-1:] if j else a[:1]
            last_v=rec["v"][j-1,-1:] if j else v[:1]
            av=np.clip(a+.65*(a-np.concatenate((last_a,a[:-1]))),0,255)
            vv=np.clip(v+.65*(v-np.concatenate((last_v,v[:-1]))),0,255)
            txt=PROMPTS[j//4] if view=="text" else ""
            ids=np.array([c+3 for c in txt.encode()]+[2],np.int32)
            tx=ids[np.arange(T)*(len(ids)-1)//(T-1)]
            b={"ok":np.ones((1,T),np.uint8),"sense":np.ones((1,T),np.uint8),
               "a":av[None],"v":vv[None],"input_x":tx[None],
               "n":np.zeros((1,T,17),np.float32),"u":np.zeros((1,T,2),np.float32)}
            fields,memory=step(b,src,memory,np.int32(T*j))
            predictions.append(np.asarray(fields))
        outputs[view]=np.concatenate(predictions)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/"predictions.npz",record=record,sha=digest,orig_v=rec["v"].reshape(-1,256),
                        orig_a=rec["a"].reshape(-1,100),**outputs)
    base=outputs["text_blank"]
    stats={"run":card.get("run",run.name),"recording":record,"role":"development","source":"Owens et al., Greatest Hits (CC BY 4.0)",
           "source_url":"https://andrewowens.com/vis/","video_size":[16,16],"ticks":1024,"seconds":1024*DT,
           "readout":"one shared 128-D state per tick","prompts":PROMPTS,"weights_sha256":digest,
           "prompt_effect_rms_mel":float(np.sqrt(np.mean((outputs["text"][:,:100]-base[:,:100])**2))*3),
           "video_from_audio_frame_change":float(np.abs(np.diff(outputs["audio"][:,100:],axis=0)).mean()*127.5),
           "source_frame_change":float(np.abs(np.diff(rec["v"].reshape(-1,256).astype(float),axis=0)).mean())}
    (out/"live.json").write_text(json.dumps(stats,indent=2)+"\n")


def render(root,repo,out):
    import io
    import zipfile
    import math
    import soundfile as sf
    import torch
    import scipy.signal
    from vocos import Vocos
    with np.load(out/"predictions.npz") as f:
        pred={k:np.array(f[k]) for k in f}
    codec=repo/"artifacts/decoders/vocos-mel-24khz"
    torch.set_num_threads(4)
    voc=Vocos.from_hparams(str(codec/"config.yaml"))
    voc.load_state_dict(torch.load(codec/"pytorch_model.bin",map_location="cpu",weights_only=True),strict=True)
    voc.eval()
    length=1024*1024
    def voice(norm):
        feat=np.repeat(np.clip(norm*3-5,-12,4),4,axis=0).T.copy()
        with torch.inference_mode():
            audio=voc.decode(torch.from_numpy(feat[None])).reshape(-1).numpy()
        return np.pad(audio,(0,max(0,length-len(audio))))[:length]
    with zipfile.ZipFile(root/"sources/vis-data-256.zip") as z:
        wave,sr=sf.read(io.BytesIO(z.read("vis-data-256/"+str(pred["record"])+"_denoised.wav")),dtype="float32")
    if wave.ndim==2:
        wave=wave.mean(1)
    if sr!=24000:
        div=math.gcd(sr,24000)
        wave=scipy.signal.resample_poly(wave,24000//div,sr//div)
    frames=lambda x:np.rint(np.clip((x+1)*127.5,0,255)).astype(np.uint8).reshape(-1,16,16)
    media={"video-to-audio":(pred["orig_v"].reshape(-1,16,16),voice(pred["video"][:,:100])),
           "audio-to-video":(frames(pred["audio"][:,100:]),wave[:length]),
           "text-to-audio-video":(frames(pred["text"][:,100:]),voice(pred["text"][:,:100]))}
    for key,(images,audio) in media.items():
        wav=out/(key+".wav")
        sf.write(wav,audio,24000,subtype="PCM_16")
        subprocess.run(["ffmpeg","-y","-v","error","-f","rawvideo","-pixel_format","gray","-video_size","16x16",
            "-framerate","375/16","-i","pipe:0","-i",str(wav),"-vf","scale=256:256:flags=neighbor",
            "-c:v","libopenh264","-pix_fmt","yuv420p","-c:a","aac","-movflags","+faststart","-shortest",str(out/(key+".mp4"))],
            input=np.ascontiguousarray(images).tobytes(),check=True)
    print(json.dumps({k:round((out/(k+".mp4")).stat().st_size/1024) for k in media}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=["predict","render"]); p.add_argument("--data",required=True,type=Path)
    p.add_argument("--run",type=Path); p.add_argument("--repo",type=Path); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="predict":
        predict(a.data,a.run,a.out)
    else:
        render(a.data,a.repo,a.out)
