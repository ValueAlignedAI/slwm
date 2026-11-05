"""Fresh test-other listening comparison; actual predictions only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from training.restoration_other import expanded_config
from training.restoration_content import load_other
from training.restoration_train import load_codec
from evals.content_conservation_probe import load_model
from exploration.restoration_outputs import full_visibility,restore_sequence
from exploration.generation_outputs import invert100
from exploration.motion_outputs import save_wave
from training.t2_real_av import write_json
from evals.motion_audio_intelligibility import read_wave,word_error


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=expanded_config(args.config)
    root=Path(cfg["artifacts"])/"exploration"; root.mkdir(parents=True,exist_ok=True); parent=json.loads(Path(cfg["parent_config"]).read_text()); oldroot=Path(parent["artifacts"])
    torch.set_num_threads(4); codec=load_codec(cfg); data=load_other(cfg); models={r["id"]:load_model(Path(cfg["artifacts"])/r["id"]/"best.pt") for r in cfg["runs"]}
    models.update({key:load_model(oldroot/key/"best.pt") for key in ("EXP-T2-501","EXP-T2-507","EXP-T2-509")})
    with np.load(Path(cfg["generation_corpus"])/"codec100.npz",allow_pickle=False) as d: new={k:d[k] for k in d.files}
    records=[]
    for i,row in enumerate(data["rows"]):
        length=int(data["lengths"][i]); y=data["mel100"][i:i+1,:length].float(); visible=full_visibility(row,length,y.device); obs=codec.observe(y,visible); v=torch.zeros((1,length,160),device=y.device)
        folder=root/row["sample_id"]; folder.mkdir(exist_ok=True); predictions={"codec100":obs["target"][0].cpu().numpy(),"interpolation":obs["base"][0].cpu().numpy()}
        silent=predictions["interpolation"].copy(); silent[~visible[0].cpu().numpy().repeat(4)]=(-18.42068074-new["mean"])/new["scale"]
        predictions["coarse_silence"]=silent
        for key,model in models.items(): predictions[key]=restore_sequence(model,obs,v,False)
        for key,mel in predictions.items(): save_wave(folder/(key+".wav"),invert100(mel,new))
        subprocess.run(["ffmpeg","-v","error","-y","-i",row["raw_path"],"-vn","-ac","1","-ar","16000","-t",str(length/25),str(folder/"reference.wav")],check=True,timeout=45)
        np.save(folder/"visible_mask.npy",visible[0].cpu().numpy()); np.savez_compressed(folder/"predicted_mels.npz",**predictions)
        records.append({"sample_id":row["sample_id"],"actor":row["actor"],"split":"other_test","transcript":row["transcript"],"folder":str(folder),"conditions":["reference"]+list(predictions)})
        write_json(root/"cases.json",records)
    del models,data,codec; torch.cuda.empty_cache()
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error(); revision="87c7102498dcde7456f24cfd30239ca606ed9063"; name="openai/whisper-tiny.en"
    processor=WhisperProcessor.from_pretrained(name,revision=revision); recognizer=WhisperForConditionalGeneration.from_pretrained(name,revision=revision).cuda().half().eval(); results=[]
    for record in records:
        scores={}
        for key in record["conditions"]:
            inputs=processor(read_wave(Path(record["folder"])/(key+".wav")),sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():
                output=recognizer.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=64,num_beams=1,do_sample=False)
            text=processor.batch_decode(output,skip_special_tokens=True)[0].strip(); scores[key]={"recognizer_text":text,"wer":word_error(record["transcript"],text)}
        results.append({**record,"scores":scores}); write_json(root/"intelligibility_progress.json",results); print(json.dumps({"recognized":len(results),"sample":record["sample_id"]}),flush=True)
    means={c:float(np.mean([r["scores"][c]["wer"] for r in results])) for c in records[0]["conditions"]}
    from training.t2_real_av import cluster_interval
    paired={}
    for key in [r["id"] for r in cfg["runs"]]:
        delta=np.array([r["scores"]["interpolation"]["wer"]-r["scores"][key]["wer"] for r in results]); paired[key]={"mean_absolute_wer_reduction":float(delta.mean()),"person_cluster_95_interval":cluster_interval(delta,[r["actor"] for r in results])}
    write_json(root/"intelligibility.json",{"scope":"Fallible external recognizer, fresh 48-clip test-other diagnostic, no training or generation use","recognizer":name,"revision":revision,"evaluation_only_weights":sum(p.numel() for p in recognizer.parameters()),"mean_clip_wer":means,"paired_changes":paired,"cases":results})


if __name__=="__main__": main()
