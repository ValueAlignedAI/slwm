"""X0-M1: actual saved-model conditional media and frozen closed-set probes.

No external text/audio/video generator is called. Waveforms use inverse mel and
Griffin-Lim estimated phase; mouth frames invert training-only PCA. Original
media are reference controls and never substituted into generated outputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import torch
import torch.nn.functional as F

from models.slwm_motion import MotionSLWM
from training.t2_prepare_cremad import SENTENCES,sha256
from training.t2_motion_train import load_data,crop,masked_inputs
from training.t2_real_av import write_json


def inverse_appearance(features: np.ndarray,codec: dict) -> np.ndarray:
    """Appearance [T,96] -> visible grayscale [T,32,48], clamped for display."""
    pixels=(features*codec["appearance_scale"])@codec["appearance_basis"].T+codec["appearance_mean"]
    return np.clip(pixels.reshape(-1,32,48),0,1)


def inverse_motion(features: np.ndarray,codec: dict) -> np.ndarray:
    """Motion [T,64] -> integrated [T,32,48] using a training-mean face prior."""
    delta=(features*codec["motion_scale"])@codec["motion_basis"].T+codec["motion_mean"]
    pixels=codec["appearance_mean"]+np.cumsum(delta,axis=0)
    return np.clip(pixels.reshape(-1,32,48),0,1)


def invert_mel(features: np.ndarray,codec: dict,*,iterations: int=48) -> np.ndarray:
    """Normalized log-mel [T,64] -> waveform [N], with newly estimated phase."""
    import torchaudio
    mel=np.exp(np.clip(features*codec["audio_scale"]+codec["audio_mean"],-18,15))
    banks=torchaudio.functional.melscale_fbanks(513,0,8000,64,16000)
    inverse=torch.linalg.pinv(banks)
    power=(torch.from_numpy(mel.astype(np.float32))@inverse).clamp_min(1e-9).T
    torch.manual_seed(17)
    griffin=torchaudio.transforms.GriffinLim(n_fft=1024,n_iter=iterations,win_length=1024,hop_length=640,power=2.,momentum=.99)
    with torch.no_grad(): sound=griffin(power).numpy()
    sound=np.nan_to_num(sound)
    sound=sound/max(float(np.max(np.abs(sound))),1e-5)*.8
    return sound.astype(np.float32)


def save_wave(path: Path,sound: np.ndarray):
    with wave.open(str(path),"wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
        out.writeframes((np.clip(sound,-1,1)*32767).astype('<i2').tobytes())


def save_video(path: Path,frames: np.ndarray,title: str) -> None:
    """Render only supplied model/codec [T,H,W] frames, with a diagnostic label."""
    from PIL import Image,ImageDraw
    images=[]
    for frame in frames:
        mouth=Image.fromarray((np.clip(frame,0,1)*255).astype(np.uint8)).resize((384,256))
        canvas=Image.new("RGB",(384,296),"#13212b"); canvas.paste(mouth,(0,40))
        ImageDraw.Draw(canvas).text((8,12),title,fill="white")
        images.append(canvas)
    images[0].save(path,save_all=True,append_images=images[1:],duration=40,loop=0,optimize=True)
    rgb=np.stack([np.asarray(x) for x in images])
    # Fedora's ffmpeg-free omits libx264. VP9/WebM is available there and the
    # GIF is the universal inline preview. Never substitute unrelated media.
    video=path.with_suffix(".webm")
    subprocess.run(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","384x296","-r","25","-i","-",
                    "-an","-c:v","libvpx-vp9","-threads","2","-deadline","realtime","-cpu-used","8","-crf","36","-b:v","0",
                    "-pix_fmt","yuv420p",str(video)],input=rgb.tobytes(),check=True,timeout=30)


def predict_sequence(model: MotionSLWM,a: torch.Tensor,v: torch.Tensor,mode: str,window: int=32) -> dict[str,np.ndarray]:
    """Full clip -> overlap-averaged model predictions using trained-size windows."""
    length=len(a); starts=list(range(0,length-window+1,window//2))
    if starts[-1]!=length-window: starts.append(length-window)
    sums={"audio":np.zeros((length,128),np.float32),"video":np.zeros((length,160),np.float32)}
    count=np.zeros((length,1),np.float32)
    with torch.inference_mode(),torch.autocast(device_type=a.device.type,dtype=torch.bfloat16,enabled=a.device.type=="cuda"):
        for start in starts:
            aa,vv=a[None,start:start+window],v[None,start:start+window]
            am,vm=masked_inputs(aa,vv,mode); output=model(aa,vv,am,vm)
            for key in sums: sums[key][start:start+window]+=output[key][0].float().cpu().numpy()
            count[start:start+window]+=1
    return {key:value/count for key,value in sums.items()}


def probe_features(tokens: torch.Tensor) -> np.ndarray:
    """Frozen [B,T,96] -> quarter-pool plus variation [B,480]."""
    parts=[piece.mean(1) for piece in tokens.chunk(4,dim=1)]+[tokens.std(1)]
    return torch.cat(parts,-1).float().cpu().numpy()


def fit_probe(x: np.ndarray,y: np.ndarray,classes: int,regularization: float=10.) -> dict:
    """Training-only ridge class probe [N,F] -> fitted [F,C] and means."""
    x=x.astype(np.float64); targets=np.eye(classes)[y]
    mean=x.mean(0); ym=targets.mean(0); xc=x-mean
    gram=xc.T@xc; gram.flat[::len(gram)+1]+=regularization
    coef=np.linalg.solve(gram,xc.T@(targets-ym))
    return {"coefficient":coef.astype(np.float32),"input_mean":mean.astype(np.float32),"target_mean":ym.astype(np.float32)}


def probe_scores(weights: dict,x: np.ndarray) -> np.ndarray:
    return (x-weights["input_mean"])@weights["coefficient"]+weights["target_mean"]


def probabilities(scores: np.ndarray,temperature: float) -> np.ndarray:
    logits=scores/temperature; logits=logits-logits.max(1,keepdims=True)
    e=np.exp(logits); return e/e.sum(1,keepdims=True)


def calibrate(scores: np.ndarray,y: np.ndarray) -> float:
    choices=[.025,.05,.1,.2,.5,1.,2.]
    return min(choices,key=lambda t:-np.log(probabilities(scores,t)[np.arange(len(y)),y].clip(1e-9)).mean())


def classify_metrics(scores: np.ndarray,y: np.ndarray,temperature: float) -> dict:
    p=probabilities(scores,temperature); chosen=p.argmax(1); confidence=p.max(1); correct=(chosen==y)
    ece=0.
    for lo in np.linspace(0,1,11)[:-1]:
        keep=(confidence>=lo)&(confidence<lo+.1+1e-8)
        if keep.any(): ece+=float(keep.mean()*abs(confidence[keep].mean()-correct[keep].mean()))
    return {"accuracy":float(correct.mean()),"nll":float(-np.log(p[np.arange(len(y)),y].clip(1e-9)).mean()),"ece":ece}


def frozen_probes(model: MotionSLWM,data: dict,out: Path,window: int) -> dict:
    features={}; labels={}; raw={}; batch=32
    sentence_classes=sorted({r["sentence"] for r in data["train"]["rows"]})
    emotion_classes=sorted({r["emotion"] for r in data["train"]["rows"]})
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for split,d in data.items():
            parts={"audio":[],"video":[]}; rawparts={"audio":[],"video":[]}
            for begin in range(0,len(d["rows"]),batch):
                ids=np.arange(begin,min(begin+batch,len(d["rows"]))); starts=(d["lengths"][ids]-window)//2
                a,v=crop(d,ids,starts,window); ar,vr=model.independent(a,v)
                parts["audio"].append(probe_features(ar)); parts["video"].append(probe_features(vr))
                rawparts["audio"].append(probe_features(a.float())); rawparts["video"].append(probe_features(v.float()))
            features[split]={k:np.concatenate(v) for k,v in parts.items()}
            raw[split]={k:np.concatenate(v) for k,v in rawparts.items()}
            labels[split]={"sentence":np.array([sentence_classes.index(r["sentence"]) for r in d["rows"]]),
                           "emotion":np.array([emotion_classes.index(r["emotion"]) for r in d["rows"]])}
    results={"scope":"Frozen closed-set linear recognition, not open text generation",
             "sentence_classes":sentence_classes,"emotion_classes":emotion_classes,"probes":{}}
    weights_out={}
    for modality in ("audio","video"):
        for task,classes in (("sentence",sentence_classes),("emotion",emotion_classes)):
            y=labels["train"][task]; key=modality+"_"+task
            weights=fit_probe(features["train"][modality],y,len(classes))
            val=probe_scores(weights,features["validation"][modality]); temp=calibrate(val,labels["validation"][task])
            learned={s:classify_metrics(probe_scores(weights,features[s][modality]),labels[s][task],temp) for s in ("validation","test")}
            raww=fit_probe(raw["train"][modality],y,len(classes)); rawtemp=calibrate(probe_scores(raww,raw["validation"][modality]),labels["validation"][task])
            shuffled=fit_probe(features["train"][modality],np.random.default_rng(773).permutation(y),len(classes))
            prior=int(np.bincount(y).argmax())
            results["probes"][key]={"classes":classes,"temperature_selected_on_validation":temp,"learned":learned,
                "raw_feature_control_test":classify_metrics(probe_scores(raww,raw["test"][modality]),labels["test"][task],rawtemp),
                "shuffled_training_labels_test":classify_metrics(probe_scores(shuffled,features["test"][modality]),labels["test"][task],temp),
                "prior_test_accuracy":float(np.mean(labels["test"][task]==prior)),
                "fitted_parameters":int(weights["coefficient"].size+weights["target_mean"].size)}
            np.savez_compressed(out/(key+"_probe.npz"),**weights,temperature=np.array(temp),classes=np.array(classes))
            np.savez_compressed(out/(key+"_raw_control.npz"),**raww,temperature=np.array(rawtemp),classes=np.array(classes))
            np.savez_compressed(out/(key+"_shuffled_control.npz"),**shuffled,temperature=np.array(temp),classes=np.array(classes))
            weights_out[key]={**weights,"temperature":temp,"classes":classes}
    write_json(out/"probe_metrics.json",results)
    return weights_out


def render_case(model: MotionSLWM,codec: dict,data: dict,split: str,index: int,folder: Path,weights: dict) -> dict:
    row=data[split]["rows"][index]; length=int(data[split]["lengths"][index])
    a=data[split]["audio"][index,:length]; v=data[split]["video"][index,:length]
    folder.mkdir(parents=True,exist_ok=True)
    arrays={mode:predict_sequence(model,a,v,mode) for mode in ("audio","video","null")}
    target_a=a.cpu().numpy(); target_v=v.cpu().numpy()
    corpus=Path(row["raw_path"]).parent.parent
    with np.load(corpus/"decoded"/(row["sample_id"]+".npz"),allow_pickle=False) as decoded:
        original=decoded["mouth"][:length].astype(np.float32)/255.
    save_video(folder/"reference_mouth.gif",original,"REFERENCE: real recorded mouth")
    save_video(folder/"codec_mouth.gif",inverse_appearance(target_v[:,:96],codec),"CODEC: PCA reconstruction of real video")
    save_video(folder/"audio_to_mouth.gif",inverse_appearance(arrays["audio"]["video"][:,:96],codec),"MODEL: audio -> mouth appearance (coarse)")
    save_video(folder/"audio_to_motion.gif",inverse_motion(arrays["audio"]["video"][:,96:],codec),"MODEL: audio -> motion + mean-face prior")
    save_video(folder/"video_reconstruction.gif",inverse_appearance(arrays["video"]["video"][:,:96],codec),"MODEL: visible video -> reconstructed mouth")
    save_video(folder/"null_mouth.gif",inverse_appearance(arrays["null"]["video"][:,:96],codec),"NULL: no input -> model prior")
    subprocess.run(["ffmpeg","-v","error","-y","-i",row["raw_path"],"-vn","-ac","1","-ar","16000","-t",str(length/25),str(folder/"reference_audio.wav")],check=True,timeout=30)
    for name,features in (("codec_audio",target_a[:,:64]),("audio_reconstruction",arrays["audio"]["audio"][:,:64]),
                          ("video_to_audio",arrays["video"]["audio"][:,:64]),("null_audio",arrays["null"]["audio"][:,:64])):
        save_wave(folder/(name+".wav"),invert_mel(features,codec))
    np.savez_compressed(folder/"model_predictions.npz",**{mode+"_"+name:value for mode,outputs in arrays.items() for name,value in outputs.items()})
    start=(length-32)//2
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        ar,vr=model.independent(a[None,start:start+32],v[None,start:start+32])
    probe_predictions={}
    for mode,tokens in (("audio",ar),("video",vr)):
        features=probe_features(tokens)
        for task in ("sentence","emotion"):
            key=mode+"_"+task; w=weights[key]; probs=probabilities(probe_scores(w,features),w["temperature"])[0]
            label=w["classes"][int(probs.argmax())]
            probe_predictions[key]={"label":label,"confidence":float(probs.max()),
                "text":SENTENCES[label] if task=="sentence" else label,"scope":"closed-set frozen recognition probe"}
    record={"sample_id":row["sample_id"],"actor":row["actor"],"split":split,"actual_sentence":row["transcript"],
        "actual_emotion":row["emotion"],"duration_seconds":length/25,"probes":probe_predictions,"source_sha256":row["raw_sha256"],
        "generation":"Actual saved model outputs. Inverse PCA / Griffin-Lim diagnostics, not realistic media synthesis or open language.",
        "audio_only_identity":"Unknown; appearance is a model average, and integrated motion uses a training-mean face prior",
        "folder":str(folder)}
    write_json(folder/"example.json",record); return record


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"])
    runs=[json.loads((root/r["id"]/"metrics.json").read_text()) for r in cfg["runs"]]
    # Choose seeds/models only by validation. Include small/full and its large
    # counterpart, plus the matched large vanilla baseline for decoder context.
    large=min((r for r in runs if r["run"]["arm"]=="signal_large"),key=lambda r:r["validation"]["selection_score"])
    vanilla=next(r for r in runs if r["run"]["arm"]=="vanilla_large" and r["run"]["seed"]==large["run"]["seed"])
    selected=[runs[0],large,vanilla]
    torch.set_num_threads(4)
    data={s:load_data(Path(cfg["corpus"]),s,torch.device("cuda")) for s in ("train","validation","test")}
    with np.load(Path(cfg["corpus"])/"codec.npz",allow_pickle=False) as archive: codec={k:archive[k] for k in archive.files}
    output=root/"exploration"; output.mkdir(exist_ok=True)
    import shutil
    source_files=("exploration/motion_outputs.py","exploration/motion_infer.py","evals/motion_causal_checks.py","evals/motion_audio_intelligibility.py")
    for source in source_files:
        target=output/"source_snapshot"/source; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    registry={"sprint":"X0-M1","status":"running","selection":"small model plus larger full seed selected by validation and its matched vanilla",
              "selected_models":[r["experiment_id"] for r in selected],"checkpoint_hashes":{r["experiment_id"]:r["checkpoint_sha256"] for r in selected},
              "codec_sha256":sha256(Path(cfg["corpus"])/"codec.npz"),
              "source_hashes":{s:sha256(Path(s)) for s in source_files},
              "claims":"diagnostic media and frozen closed-set probes, not committed behavior, ASR or realistic media generation"}
    write_json(output/"registry.json",registry)
    gallery=[]
    for record in selected:
        folder=output/record["experiment_id"]; folder.mkdir(exist_ok=True)
        checkpoint=torch.load(record["checkpoint"],map_location="cuda",weights_only=False)
        model=MotionSLWM(record["run"]["model"]).cuda().eval(); model.load_state_dict(checkpoint["model"])
        probes=frozen_probes(model,data,folder,cfg["training"]["window"])
        validation=record["validation"]["per_clip"]
        order=sorted(range(len(validation)),key=lambda i:validation[i]["video_input_audio_mel"]+validation[i]["audio_input_video_motion"])
        cases=[("validation",i,label) for i,label in zip((order[0],order[len(order)//2],order[-1]),("lowest_error","median_error","highest_error"))]
        actors=sorted({r["actor"] for r in data["test"]["rows"]})
        for actor in (actors[0],actors[len(actors)//2],actors[-1]):
            index=next(i for i,r in enumerate(data["test"]["rows"]) if r["actor"]==actor)
            cases.append(("test",index,"fixed_test_person"))
        for split,index,selection in cases:
            sample=data[split]["rows"][index]["sample_id"]
            example=render_case(model,codec,data,split,index,folder/(split+"_"+sample),probes)
            gallery.append({"experiment_id":record["experiment_id"],"selection":selection,**example})
            write_json(output/"gallery.json",{"scope":"X0-M1 uncommitted diagnostic outputs","examples":gallery})
            print(json.dumps({"rendered":sample,"model":record["experiment_id"],"split":split}),flush=True)
        del model,checkpoint; torch.cuda.empty_cache()
    registry.update(status="completed",examples=len(gallery)); write_json(output/"registry.json",registry)


if __name__=="__main__": main()
