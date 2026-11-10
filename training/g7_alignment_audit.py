"""Sealed shared-unit, event and waveform audit on independently held people and rooms."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import wave

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from vocos import Vocos
from training.g4_data import digest
from training.g7_alignment import Readout, Views, combine, evaluate, meta, save
from training.g7_speech import DT, read_clip
from training.g7_units import renderer, style

REV = "87c7102498dcde7456f24cfd30239ca606ed9063"


def fixed_rows(view):
    rng = np.random.default_rng(7719)
    groups = rng.permutation(np.unique(view.groups))
    pools = [list(rng.permutation(np.flatnonzero(view.groups == g))) for g in groups]
    rows = []
    while len(rows) < 64 and any(pools):
        for pool in pools:
            if pool:
                rows.append(pool.pop())
            if len(rows) == 64:
                break
    if len(rows) != 64:
        raise ValueError("The fixed speech audit requires 64 clips")
    return np.array(rows)


def pcm(root, repo, family, d, ix):
    crema = repo / "artifacts/t2_audio_visual/cremad_motion_v3"
    if family == "crema":
        lookup = {r["sample_id"]:r for r in meta(crema / "test.json")}
    out = []
    for row in d.ix[ix]:
        name = str(d.src["id"][row])
        if family == "libri":
            source = (name, "", "", root / "libri/audio" / name.split("-")[0] / (name + ".flac"), "")
        else:
            r = lookup[name]
            source = (name, "", "", repo / r["raw_path"], crema / "decoded" / (name + ".npz"))
        sound, _ = read_clip(source)
        begin, n = int(d.src["start"][row]), int(d.src["length"][row])
        out.append(sound[round(begin * DT * 16000):round((begin+n)*DT*16000)])
    return out


def words(s):
    return re.sub("[^a-z0-9 ]", " ", s.lower().replace("'", "").replace("’", "")).split()


def edits(a, b):
    previous = list(range(len(b)+1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(token != other)))
        previous = current
    return previous[-1]


def dump_wave(file, samples):
    with wave.open(str(file), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(np.rint(np.clip(samples, -1, 1)*32767).astype("<i2").tobytes())


class Synthesis:
    def __init__(self, repo, root):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en", revision=REV, local_files_only=True)
        self.asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", revision=REV, local_files_only=True).cuda().half().eval()
        codec = repo / "artifacts/decoders/vocos-mel-24khz"
        self.vocoder = Vocos.from_hparams(str(codec/"config.yaml"))
        self.vocoder.load_state_dict(torch.load(codec/"pytorch_model.bin", map_location="cpu", weights_only=True))
        self.vocoder.cuda().eval()
        ck = torch.load(root/"unit-renderer/renderer.pt", map_location="cuda", weights_only=True)
        self.render = renderer().cuda().eval()
        self.render.load_state_dict(ck["net"])
        self.style = ck["default_style"].cuda().reshape(1,40)

    def predict(self, recordings):
        out = []
        for j in range(0,len(recordings),16):
            features = self.processor(recordings[j:j+16], sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            ids = self.asr.generate(features.input_features.cuda().half(), attention_mask=features.attention_mask.cuda(),
                                    max_new_tokens=32, num_beams=1, do_sample=False)
            out.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return out

    def waves(self, mel, lengths):
        # All segments are trimmed on their physical frame boundary before scoring.
        out = []
        for j in range(0,len(mel),16):
            values = torch.as_tensor(mel[j:j+16], dtype=torch.float32, device="cuda").clamp(-12,4).transpose(1,2)
            waves = torchaudio.functional.resample(self.vocoder.decode(values),24000,16000).cpu().numpy()
            out.extend(w[:round(int(n)*DT*16000)] for w,n in zip(waves,lengths[j:j+16]))
        return out

    def synth(self, units, voice):
        x = torch.as_tensor(units, device="cuda").reshape(len(units),128).long()
        mean, _ = self.render(x,voice.expand(len(x),-1))
        return (mean*3-5).cpu().numpy()


def audit(a):
    lock = meta(a.ck/"alignment/test-lock.json")
    if digest(a.ck/"unit-renderer/renderer.pt") != lock["renderer"]:
        raise ValueError("Waveform renderer changed")
    folder = a.out / a.tag / a.variant
    folder.mkdir(parents=True, exist_ok=True)
    weight = a.ck / "alignment" / a.tag / a.variant / "weights.pt"
    identity = digest(weight)
    if identity != lock["models"][a.tag+"/"+a.variant]["sha256"]:
        raise ValueError("Unregistered alignment head")
    if (folder/"waves.json").is_file():
        return
    torch.set_num_threads(3)
    head = Readout().cuda().eval()
    head.load_state_dict(torch.load(weight,map_location="cuda",weights_only=True)["weights"])
    views = {f:Views(a.root,a.old,a.tag,f,"test") for f in ("crema","libri","events","oldimpact")}
    with torch.inference_mode():
        if not (folder/"field.json").is_file():
            save(folder/"field.json",{"sha256":identity,"scores":evaluate(head,views,"test",a.variant),
                 "groups":{f:len(np.unique(d.groups)) for f,d in views.items()}})
        synth = Synthesis(a.repo,a.ck)
        scores = {}
        for family in ("crema","libri"):
            d=views[family]; ix=fixed_rows(d); real=d.ix[ix]; lengths=d.length[ix]
            ids = d.src["id"][real].tolist()
            groups = [str(g) for g in d.groups[ix]]
            refs = d.src["phrase"][real].tolist()
            media = folder/family
            media.mkdir(exist_ok=True)
            def score(key, rec):
                hyps=synth.predict(rec)
                e=np.array([edits(words(s),words(h)) for s,h in zip(refs,hyps)])
                n=np.array([len(words(s)) for s in refs])
                unique,g=np.unique(groups,return_inverse=True)
                ge,gn=np.bincount(g,weights=e),np.bincount(g,weights=n)
                sample=np.random.default_rng(779).integers(len(unique),size=(2000,len(unique)))
                ci=np.quantile(ge[sample].sum(1)/gn[sample].sum(1),[.025,.975])
                for j in range(2):
                    dump_wave(media/(f"{j:02d}-"+key+".wav"), rec[j])
                return {"wer":float(e.sum()/n.sum()),"edits":int(e.sum()),"words":int(n.sum()),"speaker_ci95":ci.tolist(),
                        "hypotheses":hyps,"word_edits":e.tolist()}
            # Share the same reference/control waveforms, selection and ASR for every model.
            fixture = a.out/(family+"-oracle.json")
            if not fixture.is_file():
                controls = {"original":score("original",pcm(a.root,a.repo,family,d,ix))}
                fine=d.src["fine"][real].reshape(64,256,100).astype(np.float32)/16-12
                coarse=np.repeat(d.src["a"][real].astype(np.float32)/16-12,4,axis=1)
                oracle=d.units[real]
                ref=d.ref[ix]
                if np.any(ref==ix):raise ValueError("Independent same-speaker enrollment unavailable")
                enrolled=torch.tensor(style(d.src["a"][d.ix[ref]],d.length[ref]),device="cuda")
                for key,mels in (("fine-codec",fine),("coarse-codec",coarse),
                                 ("oracle-neutral",synth.synth(oracle,synth.style)),
                                 ("oracle-enrolled",synth.synth(oracle,enrolled)),
                                 ("constant-units",synth.synth(np.zeros_like(oracle),synth.style))):
                    controls[key]=score(key,synth.waves(mels,lengths))
                save(fixture,{"ids":ids,"speakers":groups,"reference":refs,"conditions":controls,"renderer":lock["renderer"],"recognizer_revision":REV})
            reference=meta(fixture)
            if ids!=reference["ids"] or refs!=reference["reference"] or groups!=reference["speakers"] or reference["renderer"]!=lock["renderer"]:
                raise ValueError("Waveform controls differ across heads")
            outputs={}
            for mode in d.card["views"]:
                name={0:"masked-audio",1:"video",3:"text"}[mode]
                b=combine([d.get(ix,mode,mode==0)])
                units,_,_,mean,_,_,_=head(b["z"],b["voice"],b["texture"])
                direct=np.repeat((mean*3-5).cpu().numpy(),4,axis=1)
                outputs[name+"-direct"]=score(name+"-direct",synth.waves(direct,lengths))
                if a.variant=="units-sync":
                    mel=synth.synth(units.argmax(-1),synth.style)
                    outputs[name+"-units"]=score(name+"-units",synth.waves(mel,lengths))
            scores[family]={"ids":ids,"conditions":outputs}
        save(folder/"waves.json",{"tag":a.tag,"variant":a.variant,"head_sha256":identity,"word_errors_include_insertions":True,"scores":scores})


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for key in ("ck","root","old","repo","out"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--tag",required=True);p.add_argument("--variant",required=True,choices=("reconstruct","units-sync"))
    audit(p.parse_args())
