"""Observation-matched silence baseline: same coarse packets as the models.

The original R1 `silence_gaps` diagnostic keeps fine-rate target mel outside
holes, hence has richer observations. Preserve it, but do not use it as a fair
repair baseline. This addendum blanks the COARSE interpolation instead.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from exploration.generation_outputs import invert100
from exploration.motion_outputs import save_wave
from evals.motion_audio_intelligibility import read_wave,word_error
from training.t2_real_av import write_json,cluster_interval
from training.t2_prepare_cremad import sha256


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=json.loads(Path(args.config).read_text())
    root=Path(cfg["artifacts"])/"exploration"; cases=json.loads((root/"cases.json").read_text()); previous=json.loads((root/"intelligibility.json").read_text()); torch.set_num_threads(4)
    with np.load(Path(cfg["generation_corpus"])/"codec100.npz",allow_pickle=False) as d: codec={k:d[k] for k in d.files}
    silence=(-18.42068074-codec["mean"])/codec["scale"]
    for case in cases:
        folder=Path(case["folder"])
        with np.load(folder/"predicted_mels.npz",allow_pickle=False) as d: mel=d["interpolation"].copy()
        visible=np.load(folder/"visible_mask.npy",allow_pickle=False).repeat(4)
        mel[~visible]=silence; save_wave(folder/"coarse_silence.wav",invert100(mel,codec))
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    from transformers.utils import logging
    logging.set_verbosity_error(); revision="87c7102498dcde7456f24cfd30239ca606ed9063"; name="openai/whisper-tiny.en"
    processor=WhisperProcessor.from_pretrained(name,revision=revision); model=WhisperForConditionalGeneration.from_pretrained(name,revision=revision).cuda().half().eval(); results=[]
    for case in cases:
        path=Path(case["folder"])/"coarse_silence.wav"; inputs=processor(read_wave(path),sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
        with torch.inference_mode(): tokens=model.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=64,num_beams=1,do_sample=False)
        text=processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()
        results.append({"sample_id":case["sample_id"],"actor":case["actor"],"split":case["split"],"wer":word_error(case["transcript"],text),"recognizer_text":text,"wave_sha256":sha256(path)})
    summary={}
    for split in ("crema_test","libri_test","ravdess_test"):
        selected=[r for r in results if r["split"]==split]; original={r["sample_id"]:r for r in previous["cases"] if r["split"]==split}; changes={}
        for key in ("EXP-T2-501","EXP-T2-502","EXP-T2-507"):
            delta=np.array([r["wer"]-original[r["sample_id"]]["scores"][key]["wer"] for r in selected])
            changes[key]={"model_absolute_wer_reduction_vs_coarse_silence":float(delta.mean()),"person_cluster_95_interval":cluster_interval(delta,[r["actor"] for r in selected])}
        summary[split]={"mean_clip_wer":float(np.mean([r["wer"] for r in selected])),"paired_model_changes":changes}
    write_json(root/"matched_silence_control.json",{"source_sha256":sha256(Path(__file__)),"scope":"Same 25Hz observations as every model; original fine-rate silence diagnostic is richer and retained separately",
        "recognizer":name,"revision":revision,"summary":summary,"cases":results})


if __name__=="__main__": main()
