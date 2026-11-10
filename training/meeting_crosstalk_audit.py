"""Post-test source audit: owner transcript silence does not imply silent audio."""
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from exploration.stream_media import read_audio
from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    torch.set_num_threads(4); cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"])
    rows={r["id"]:r for r in json.loads((Path(cfg["corpus"])/"ami_test/rows.json").read_text())}
    a=rows["ES2012a-A"]; b=rows["ES2012a-B"]; assert a["raw_sha256"]!=b["raw_sha256"] and a["raw_path"]!=b["raw_path"]
    signals=[read_audio(r["raw_path"],80,20) for r in (a,b)]
    centered=[x.astype(np.float64)-float(x.mean()) for x in signals]; n=len(centered[0]); fft=1<<(2*n-1).bit_length()
    correlation=np.fft.irfft(np.fft.rfft(centered[0],fft)*np.fft.rfft(centered[1],fft).conj(),fft)
    limit=1200; lags=np.arange(-limit,limit+1); values=correlation[lags%fft]/np.sqrt(np.square(centered[0]).sum()*np.square(centered[1]).sum())
    at=int(np.argmax(np.abs(values))); revision="87c7102498dcde7456f24cfd30239ca606ed9063"
    processor=WhisperProcessor.from_pretrained("openai/whisper-tiny.en",revision=revision,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en",revision=revision,local_files_only=True).eval()
    sound=torchaudio.functional.resample(torch.from_numpy(signals[1])[None],24000,16000)[0].numpy()
    inputs=processor(sound,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
    with torch.inference_mode(): ids=model.generate(inputs.input_features,attention_mask=inputs.attention_mask,max_new_tokens=256,num_beams=1,do_sample=False)
    text=processor.batch_decode(ids,skip_special_tokens=True)[0]
    owners={r["id"]:" ".join(w for lo,hi,w in r["spans"] if lo>=80 and hi<=100) for r in (a,b)}
    assert not owners[b["id"]]
    original=json.loads((root/"evaluation/recognizer_results.json").read_text())
    batched=next(r for r in original if r["group"]=="long_control" and r["condition"]=="input" and r["id"]==b["id"])
    result={"scope":"Post-test source/readout audit only; no training, filtering, weight/threshold selection or primary gate change",
        "source_sha256":file_sha(__file__),"source_rows":{r["id"]:{k:r[k] for k in ("actor","raw_path","raw_sha256")} for r in (a,b)},
        "interval_seconds":[80,100],"own_speaker_annotations":owners,
        "rms_by_channel":{r["id"]:float(np.sqrt(np.square(wave).mean())) for r,wave in zip((a,b),signals)},
        "maximum_absolute_cross_channel_correlation_within_50ms":float(abs(values[at])),"correlation_lag_samples":int(lags[at]),
        "independent_single_example_cpu_recognition":text,"original_batched_first_chunk":batched["chunks"][0]["text"],
        "external_recognizer_revision":revision,"external_evaluation_parameters":sum(p.numel() for p in model.parameters()),
        "interpretation":"Distinct headset files can contain another participant's audible speech. Cross-channel correlation and independently decoded text support crosstalk here. Empty owner annotations are not acoustic silence; no-reference insertion counts must not automatically be called hallucinations. This is one diagnostic interval, not an audit of all AMI audio."}
    write_json(root/"meeting_crosstalk_audit.json",result); print(json.dumps(result),flush=True)


if __name__=="__main__": main()
