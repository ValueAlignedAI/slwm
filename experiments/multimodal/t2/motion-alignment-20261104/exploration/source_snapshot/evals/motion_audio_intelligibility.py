"""Optional external ASR diagnostic, never a component of SLWM generation.

Compare original, codec and actual generated waveforms on the fixed gallery.
Recognizer transcripts are fallible measurements, not model-generated text.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import wave

import numpy as np
import torch

from training.t2_real_av import write_json


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]"," ",text.lower().replace("'","").replace("’","")).split()


def word_error(reference: str,hypothesis: str) -> float:
    """Levenshtein word error including insertions; may exceed one."""
    a,b=words(reference),words(hypothesis)
    previous=list(range(len(b)+1))
    for i,token in enumerate(a,1):
        current=[i]
        for j,other in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(token!=other)))
        previous=current
    return previous[-1]/max(len(a),1)


def read_wave(path: Path) -> np.ndarray:
    with wave.open(str(path),"rb") as stream:
        if stream.getsampwidth()!=2 or stream.getframerate()!=16000 or stream.getnchannels()!=1: raise ValueError("Expected mono PCM16 16kHz")
        return np.frombuffer(stream.readframes(stream.getnframes()),dtype="<i2").astype(np.float32)/32768.


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"])
    out=root/"exploration/intelligibility.json"
    try:
        from transformers import WhisperProcessor,WhisperForConditionalGeneration
        torch.set_num_threads(4)
        name="openai/whisper-tiny.en"
        processor=WhisperProcessor.from_pretrained(name)
        model=WhisperForConditionalGeneration.from_pretrained(name).cuda().half().eval()
        gallery=json.loads((root/"exploration/gallery.json").read_text())["examples"]
        conditions=("reference_audio","codec_audio","audio_reconstruction","video_to_audio","null_audio")
        results=[]
        for example in gallery:
            record={"experiment_id":example["experiment_id"],"sample_id":example["sample_id"],"actual_sentence":example["actual_sentence"],"conditions":{}}
            for condition in conditions:
                waveform=read_wave(Path(example["folder"])/(condition+".wav"))
                inputs=processor(waveform,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
                with torch.inference_mode():
                    generated=model.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),
                        max_new_tokens=64,num_beams=1,do_sample=False)
                text=processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
                record["conditions"][condition]={"recognizer_text":text,"word_error_rate":word_error(example["actual_sentence"],text)}
            results.append(record)
            print(json.dumps({"sample_id":record["sample_id"],"model":record["experiment_id"],"video_to_audio_recognizer":record["conditions"]["video_to_audio"]}),flush=True)
        write_json(out,{"status":"completed","scope":"Exploratory external recognizer on selected diagnostic gallery; not a benchmark or human intelligibility judgment",
            "recognizer":name,"revision":getattr(model.config,"_commit_hash",None),"external_evaluation_parameters":sum(p.numel() for p in model.parameters()),
            "never_used_for_slwm_generation":True,"conditions_mean_wer":{c:float(np.mean([r["conditions"][c]["word_error_rate"] for r in results])) for c in conditions},
            "examples":results,"limitations":"Recognizer can hallucinate on distorted/silent inputs; compare original and codec controls before attributing errors to SLWM."})
    except Exception as exc:
        write_json(out,{"status":"unavailable","error":repr(exc),"scope":"Optional diagnostic only; no main metric or generation artifact depends on it"})
        raise


if __name__=="__main__": main()
