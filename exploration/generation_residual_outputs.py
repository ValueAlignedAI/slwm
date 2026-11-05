"""Actual audio, rate-matched renderer controls and paired G2 diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch

from models.slwm_generation_residual import SignalResidualAudioHead
from models.slwm_motion import MotionSLWM
from training.t2_generation_residual import codecs
from training.t2_generation_train import load_data
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,cluster_interval
from exploration.generation_outputs import invert100
from exploration.motion_outputs import save_wave,predict_sequence
from evals.generation_controls import interpolate25
from evals.motion_audio_intelligibility import read_wave,word_error


def predict_residual(model: SignalResidualAudioHead,x: torch.Tensor,observed: torch.Tensor,baseline: bool=False) -> np.ndarray:
    """One observed clip [T,D]/[T,128] -> overlap-averaged [4T,64]."""
    length=len(x); starts=list(range(0,length-31,16))
    if starts[-1]!=length-32: starts.append(length-32)
    total=np.zeros((4*length,64),np.float32); count=np.zeros((4*length,1),np.float32)
    with torch.inference_mode(),torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.device.type=="cuda"):
        for s in starts:
            y=(model.baseline(observed[None,s:s+32]) if baseline else model(x[None,s:s+32],observed[None,s:s+32]))[0].float().cpu().numpy()
            total[4*s:4*(s+32)]+=y; count[4*s:4*(s+32)]+=1
    return total/count


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--score-only",action="store_true"); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); g1=json.loads(Path(cfg["g1_config"]).read_text()); root=Path(cfg["artifacts"])/"exploration"
    root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); old,new=codecs(cfg)
    if not args.score_only:
        models={}
        for run in cfg["runs"]:
            checkpoint=torch.load(Path(cfg["artifacts"])/run["id"]/"best.pt",map_location="cpu",weights_only=False)
            model=SignalResidualAudioHead(old,new).cuda().eval(); model.load_state_dict(checkpoint["model"]); models[run["id"]]=model
        old_checkpoint=torch.load(g1["checkpoint"],map_location="cpu",weights_only=False)
        old_model=MotionSLWM(old_checkpoint["config"]["run"]["model"]).cuda().eval(); old_model.load_state_dict(old_checkpoint["model"])
        cases=json.loads((Path(g1["artifacts"])/"exploration/audio_cases.json").read_text()); records=[]
        for split in ("test","external"):
            data=load_data(Path(cfg["corpus"]),split); by_id={r["sample_id"]:i for i,r in enumerate(data["rows"])}
            for case in (c for c in cases if c["split"]==split):
                index=by_id[case["sample_id"]]; length=int(data["lengths"][index]); folder=root/(split+"_"+case["sample_id"]); folder.mkdir(exist_ok=True)
                a=data["audio_coarse"][index,:length].float(); v=data["video_coarse"][index,:length].float()
                for name in ("reference","codec100"): shutil.copyfile(Path(case["folder"])/(name+".wav"),folder/(name+".wav"))
                for run in cfg["runs"]:
                    x=data[run["input"]][index,:length].float(); mel=predict_residual(models[run["id"]],x,a)
                    save_wave(folder/(run["id"]+".wav"),invert100(mel,new))
                model=models[cfg["runs"][0]["id"]]
                interpolated=predict_residual(model,a,a,baseline=True); save_wave(folder/"interpolation.wav",invert100(interpolated,new))
                output=predict_sequence(old_model,a,v,"audio")["audio"][:,:64]
                same_rate=interpolate25(torch.from_numpy(output)[None],old,new)[0].numpy()
                save_wave(folder/"old_audio_100renderer.wav",invert100(same_rate,new))
                records.append({**{k:case[k] for k in ("sample_id","actor","split","transcript")},"folder":str(folder),
                    "conditions":["reference","codec100","interpolation","old_audio_100renderer"]+[r["id"] for r in cfg["runs"]]})
                write_json(root/"cases.json",records)
            del data; torch.cuda.empty_cache()
        del models,old_model; torch.cuda.empty_cache()
    records=json.loads((root/"cases.json").read_text())
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    name="openai/whisper-tiny.en"; revision="87c7102498dcde7456f24cfd30239ca606ed9063"
    processor=WhisperProcessor.from_pretrained(name,revision=revision); recognizer=WhisperForConditionalGeneration.from_pretrained(name,revision=revision).cuda().half().eval()
    results=[]
    for record in records:
        scores={}
        for condition in record["conditions"]:
            path=Path(record["folder"])/(condition+".wav"); inputs=processor(read_wave(path),sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():
                output=recognizer.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=64,num_beams=1,do_sample=False)
            text=processor.batch_decode(output,skip_special_tokens=True)[0].strip()
            scores[condition]={"recognizer_text":text,"wer":word_error(record["transcript"],text),"wave_sha256":sha256(path)}
        results.append({**record,"scores":scores}); write_json(root/"intelligibility_progress.json",results)
        print(json.dumps({"scored":len(results),"sample":record["sample_id"]}),flush=True)
    summary={}
    for split in ("test","external"):
        subset=[r for r in results if r["split"]==split]; conditions=records[0]["conditions"]
        means={c:float(np.mean([r["scores"][c]["wer"] for r in subset])) for c in conditions}; changes={}
        for run in cfg["runs"]:
            delta=np.array([r["scores"]["interpolation"]["wer"]-r["scores"][run["id"]]["wer"] for r in subset])
            changes[run["id"]]={"relative_wer_gain_vs_interpolation":float(delta.mean()/max(means["interpolation"],1e-9)),
                "absolute_wer_reduction_95_person_cluster_interval":cluster_interval(delta,[r["actor"] for r in subset])}
        summary[split]={"samples":len(subset),"mean_clip_wer":means,"paired_changes":changes}
    write_json(root/"intelligibility.json",{"scope":"exploratory G2 diagnostic reuses G1 cases; recognizer text is not generated model text", "summary":summary,
        "recognizer":name,"revision":revision,"external_evaluation_parameters":sum(p.numel() for p in recognizer.parameters()),"cases":results})


if __name__=="__main__": main()
