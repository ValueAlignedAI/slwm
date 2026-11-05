"""Pinned external recognizer diagnostic, separate from signal proxy scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from evals.motion_audio_intelligibility import read_wave,word_error
from training.t2_real_av import write_json,cluster_interval
from training.t2_prepare_cremad import sha256


def summarize(results):
    summary={}
    for split in ("crema_test","rav_test"):
        rows=[r for r in results if r["split"]==split]; conditions=rows[0]["conditions"]
        means={c:float(np.mean([r["conditions"][c]["wer"] for r in rows])) for c in conditions}; gains={}
        comparisons=[(f"video{s}",base) for s in ("61","62") for base in (f"audio{s}",f"anchor{s}","silence","interpolation")]
        comparisons += [("video61",c) for c in ("wrong61","shift61","static61")]
        for new,old in comparisons:
            delta=np.array([r["conditions"][old]["wer"]-r["conditions"][new]["wer"] for r in rows]); interval=cluster_interval(delta,[r["actor"] for r in rows])
            relative=float(delta.mean()/max(means[old],1e-9)); gains[new+"_vs_"+old]={"absolute_wer_reduction":float(delta.mean()),"relative_wer_reduction":relative,
                "person_cluster_absolute_reduction_95_interval":interval,"passes_5percent_positive_interval":bool(relative>=.05 and interval[0]>0)}
        summary[split]={"samples":len(rows),"people":len({r["actor"] for r in rows}),"mean_clip_wer":means,"paired_changes":gains}
    return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"])/"exploration"; torch.set_num_threads(4)
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    name="openai/whisper-tiny.en"; revision="87c7102498dcde7456f24cfd30239ca606ed9063"
    processor=WhisperProcessor.from_pretrained(name,revision=revision,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(name,revision=revision,local_files_only=True).cuda().half().eval()
    cases=json.loads((root/"audio_cases.json").read_text()); results=[]
    for case in cases:
        record={k:case[k] for k in ("sample_id","actor","split","transcript")}; record["conditions"]={}
        conditions=case["conditions"]
        for begin in range(0,len(conditions),4):
            names=conditions[begin:begin+4]; waves=[read_wave(Path(case["folder"])/(c+".wav")) for c in names]
            inputs=processor(waves,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():
                generated=model.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=64,num_beams=1,do_sample=False)
            texts=processor.batch_decode(generated,skip_special_tokens=True)
            for c,text in zip(names,texts): record["conditions"][c]={"recognizer_text":text.strip(),"wer":word_error(case["transcript"],text),"wave_sha256":sha256(Path(case["folder"])/(c+".wav"))}
        results.append(record); write_json(root/"intelligibility_progress.json",results)
        print(json.dumps({"recognized":len(results),"sample_id":case["sample_id"],"audio61_wer":record["conditions"]["audio61"]["wer"],"video61_wer":record["conditions"]["video61"]["wer"]}),flush=True)
    summary=summarize(results); practical=all(summary[s]["paired_changes"][f"video{seed}_vs_{base}"]["passes_5percent_positive_interval"] for s in summary for seed in ("61","62") for base in (f"audio{seed}","silence"))
    write_json(root/"intelligibility.json",{"scope":"Fallible external recognizer on 32 preregistered clips; diagnostic WER, not a human intelligibility study",
        "recognizer":name,"revision":revision,"external_evaluation_parameters":sum(p.numel() for p in model.parameters()),"never_used_for_slwm_training_or_generation":True,
        "source_sha256":sha256(Path(__file__)),"practical_video_gate_passes":practical,"summary":summary,"examples":results})


if __name__=="__main__": main()
