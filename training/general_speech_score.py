"""Pinned independent waveform recognition, never a training teacher."""
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor,WhisperForConditionalGeneration

from exploration.stream_media import read_audio
from exploration.broad_stream_evaluate import edit_distance
from training.general_sources import write
from training.research_sources import file_sha


class Recognizer:
    """Caller owns the shared GPU lock while this evaluator is on CUDA."""
    def __init__(self,recipe,cache_path):
        self.recipe,self.path=recipe,Path(cache_path)
        self.cache=json.loads(self.path.read_text()) if self.path.exists() else {}
        self.processor=WhisperProcessor.from_pretrained(recipe["recognizer"],revision=recipe["recognizer_revision"],local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(recipe["recognizer"],revision=recipe["recognizer_revision"],local_files_only=True).cuda().half().eval().requires_grad_(False)

    def score(self,jobs):
        unique={}
        for job in jobs:
            job["wave_sha256"]=file_sha(job["path"])
            if job["wave_sha256"] not in self.cache:unique[job["wave_sha256"]]=job["path"]
        pending=list(unique.items());batch_size=self.recipe["recognizer_batch"]
        for start in range(0,len(pending),batch_size):
            current=pending[start:start+batch_size]
            signals=[torchaudio.functional.resample(torch.from_numpy(read_audio(path)),24000,16000).numpy() for _,path in current]
            assert all(len(s)<=20*16000 for s in signals),"Use explicit20s chunks for longer diagnostics"
            inputs=self.processor(signals,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():
                ids=self.model.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=self.recipe["recognizer_max_new_tokens"],num_beams=1,do_sample=False)
            for (digest,_),text,tokens in zip(current,self.processor.batch_decode(ids,skip_special_tokens=True),ids.cpu().tolist()):
                eos=self.model.generation_config.eos_token_id
                self.cache[digest]={"text":text,"generated_token_count":tokens.index(eos)+1 if eos in tokens else len(tokens),"token_limit_reached":eos not in tokens}
            write(self.path,self.cache)
        result=[]
        for job in jobs:
            recognized=self.cache[job["wave_sha256"]]
            edits,count=edit_distance(self.processor.tokenizer.normalize(job["reference"]),self.processor.tokenizer.normalize(recognized["text"]))
            result.append({**job,**recognized,"edits":edits,"reference_words":count,"wer":edits/max(1,count),"owner_reference_empty":count==0})
        return result

    def score_long(self,jobs):
        from exploration.stream_media import write_audio
        result=[]
        for job in jobs:
            digest=file_sha(job["path"])
            if digest not in self.cache:
                sound=read_audio(job["path"]);parts=[];folder=Path(job["path"]).with_suffix("").with_name(Path(job["path"]).stem+"_asr_chunks");folder.mkdir(exist_ok=True)
                for start in range(0,len(sound),20*24000):
                    path=folder/(str(start//24000)+".flac");write_audio(path,sound[start:start+20*24000]);parts.append({"path":str(path),"reference":"","start_seconds":start/24000})
                recognized=self.score(parts);text=" ".join(r["text"] for r in recognized)
                self.cache[digest]={"text":text,"chunks":[{k:r[k] for k in ("start_seconds","text","generated_token_count","token_limit_reached")} for r in recognized],"generated_token_count":sum(r["generated_token_count"] for r in recognized),"token_limit_reached":any(r["token_limit_reached"] for r in recognized)}
                write(self.path,self.cache)
            recognized=self.cache[digest];edits,count=edit_distance(self.processor.tokenizer.normalize(job["reference"]),self.processor.tokenizer.normalize(recognized["text"]))
            result.append({**job,**recognized,"wave_sha256":digest,"edits":edits,"reference_words":count,"wer":edits/max(1,count),"owner_reference_empty":count==0})
        return result


def aggregate(rows):
    words=sum(r["reference_words"] for r in rows);edits=sum(r["edits"] for r in rows if r["reference_words"]>0)
    return {"wer":edits/max(1,words),"edits_on_nonempty_references":edits,"reference_words":words,"cases":len(rows),
            "no_owner_reference_cases":sum(r["reference_words"]==0 for r in rows),
            "no_owner_reference_insertions":sum(r["edits"] for r in rows if r["reference_words"]==0),
            "token_limit_cases":sum(r["token_limit_reached"] for r in rows)}
