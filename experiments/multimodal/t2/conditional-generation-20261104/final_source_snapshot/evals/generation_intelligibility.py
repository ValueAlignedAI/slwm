"""Independent, fixed-subset recognizer diagnostic for generated waveforms."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from evals.motion_audio_intelligibility import read_wave,word_error
from training.t2_real_av import write_json,cluster_interval
from training.t2_prepare_cremad import sha256


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"])/"exploration"; torch.set_num_threads(4)
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    name="openai/whisper-tiny.en"; revision="87c7102498dcde7456f24cfd30239ca606ed9063"
    processor=WhisperProcessor.from_pretrained(name,revision=revision)
    model=WhisperForConditionalGeneration.from_pretrained(name,revision=revision).cuda().half().eval()
    cases=json.loads((root/"audio_cases.json").read_text()); results=[]
    for case in cases:
        record={k:case[k] for k in ("sample_id","actor","split","transcript")}; record["conditions"]={}
        for condition in case["conditions"]:
            path=Path(case["folder"])/(condition+".wav"); waveform=read_wave(path)
            inputs=processor(waveform,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():
                generated=model.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),
                    max_new_tokens=64,num_beams=1,do_sample=False)
            text=processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
            record["conditions"][condition]={"recognizer_text":text,"wer":word_error(case["transcript"],text),"wave_sha256":sha256(path)}
        results.append(record); write_json(root/"intelligibility_progress.json",results)
        print(json.dumps({"recognized":len(results),"sample_id":case["sample_id"],"wer_audio401":record["conditions"]["EXP-T2-401"]["wer"],
                          "wer_video402":record["conditions"]["EXP-T2-402"]["wer"]}),flush=True)
    summary={}
    for split in ("test","external"):
        subset=[r for r in results if r["split"]==split]; conditions=cases[0]["conditions"]
        means={c:float(np.mean([r["conditions"][c]["wer"] for r in subset])) for c in conditions}
        gains={}
        for new,old in (("EXP-T2-401","old_audio"),("EXP-T2-402","old_video"),("EXP-T2-402","wrong_video"),("EXP-T2-402","null_video"),("codec100","codec25")):
            delta=np.array([r["conditions"][old]["wer"]-r["conditions"][new]["wer"] for r in subset])
            gains[new+"_vs_"+old]={"absolute_wer_reduction":float(delta.mean()),"relative_wer_reduction":float(delta.mean()/max(means[old],1e-9)),
                "person_cluster_absolute_reduction_95_interval":cluster_interval(delta,[r["actor"] for r in subset])}
        summary[split]={"samples":len(subset),"people":len({r["actor"] for r in subset}),"mean_clip_wer":means,"paired_changes":gains}
    write_json(root/"intelligibility.json",{"scope":"Fallible independent recognizer on preregistered 48-clip subset; not human intelligibility or model-produced text",
        "recognizer":name,"revision":revision,"external_evaluation_parameters":sum(p.numel() for p in model.parameters()),
        "recognizer_never_used_for_generation_or_training":True,"summary":summary,"examples":results})


if __name__=="__main__": main()
