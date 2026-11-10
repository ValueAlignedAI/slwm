"""Independent full-utterance replication of fixed-rate coding with two frozen recognizers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
import numpy as np
from training.g4_data import number,quant
from training.g8_rate_codec import OUT_RATE,HELD,load_book,quantize,verify_clip_payload
from training.g8_acoustic_codec import DATA,OLD
from training.g7_units import put
from scripts.train.g7_lock_heads import sha,lock

ROOT=DATA.parent/"full-speech";OUTPUT=OUT_RATE.parent/"full-speech"
MD5="32fa31d27d2e1cad72775fee3f4849a9"
SALT="g8-full-codec-81301/"
WHISPER_REV="87c7102498dcde7456f24cfd30239ca606ed9063"
TEACHER_REV="22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def select():
    import soundfile as sf
    ROOT.mkdir(parents=True,exist_ok=True);archive=ROOT/"test-clean.tar.gz"
    if not archive.exists():
        tmp=archive.with_suffix(".partial")
        subprocess.run(["curl","-fLsS","--retry","3","--max-time","1100","https://www.openslr.org/resources/12/test-clean.tar.gz","-o",str(tmp)],check=True)
        with tmp.open("rb") as f:
            if hashlib.file_digest(f,"md5").hexdigest()!=MD5:raise ValueError("Wrong independent source archive")
        tmp.replace(archive)
    with archive.open("rb") as f:
        if hashlib.file_digest(f,"md5").hexdigest()!=MD5:raise ValueError("Full-utterance archive changed")
    if not (ROOT/"extracted").is_file():
        with tarfile.open(archive,"r:gz") as t:
            for m in t:
                parts=Path(m.name).parts
                if parts[:2]!=("LibriSpeech","test-clean"):continue
                if ".." in parts or not (m.isfile() or m.isdir()):raise ValueError("Invalid independent archive")
                t.extract(m,ROOT,filter="data")
        (ROOT/"extracted").touch()
    forbidden=set(np.load(OLD/"speech/libri/train/group.npy").tolist())
    for other in (DATA,HELD):forbidden.update(np.load(other/"confirm/libri/group.npy").tolist())
    choices={};rejected=set()
    for t in sorted((ROOT/"LibriSpeech/test-clean").glob("*/*/*.trans.txt")):
        for line in t.read_text().splitlines():
            uid,transcript=line.split(" ",1);speaker=uid.split("-")[0];path=t.parent/(uid+".flac")
            if number("libri/"+speaker) in forbidden:rejected.add(speaker);continue
            info=sf.info(path)
            if info.samplerate!=16000 or info.channels!=1:raise ValueError("Unexpected source format")
            if 4<=info.frames/16000<=18:choices.setdefault(speaker,[]).append({"id":uid,"speaker":speaker,"reference":transcript,"samples":info.frames,"file":str(path.relative_to(ROOT))})
    chosen=[v for speaker,items in sorted(choices.items()) for v in sorted(items,key=lambda r:number(SALT+r["id"]))[:2]]
    if len(chosen)!=2*len(choices) or not chosen:raise ValueError("Insufficient predefined full utterances")
    d={"source_md5":MD5,"source_sha256":sha(archive),"salt":SALT,"excluded_overlapping_speakers":sorted(rejected),"utterances":chosen,"speakers":len(choices)}
    lock(ROOT/"selection.json",d);return d


def score():
    import soundfile as sf
    import torch
    import torchaudio
    from scipy.signal import resample_poly
    from transformers import WhisperProcessor,WhisperForConditionalGeneration,Wav2Vec2ForCTC
    from vocos.feature_extractors import MelSpectrogramFeatures
    from exploration.stream_media import load_renderer,synthesize
    from training.g7_alignment_audit import words,edits
    from training.g8_speech_score import summarize
    from training.g8_models import read
    from scripts.train.g6_artifacts import roots
    chosen=select();data=chosen["utterances"];teacher=read(OLD/"teacher.json")
    if teacher["revision"]!=TEACHER_REV:raise ValueError("Incorrect frozen second recognizer")
    config={"selection":sha(ROOT/"selection.json"),"codec":sha(OUT_RATE/"weights.npz"),"code":sha(Path("training/g8_full_speech.py")),
        "recognizers":[WHISPER_REV,TEACHER_REV],"samples":len(data),"source":"LibriSpeech test-clean","no_alignment_or_crop":True}
    for disk in roots("g8"):
        dest=disk/"full-speech";dest.mkdir(exist_ok=True);lock(dest/"lock.json",config)
    if (OUTPUT/"complete.json").exists():return
    model=WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en",revision=WHISPER_REV,local_files_only=True).cuda().half().eval()
    proc=WhisperProcessor.from_pretrained("openai/whisper-tiny.en",revision=WHISPER_REV,local_files_only=True)
    ctc=Wav2Vec2ForCTC.from_pretrained(teacher["snapshot"],local_files_only=True).cuda().half().eval()
    vocab={i:k for k,i in read(Path(teacher["snapshot"])/"vocab.json").items()}
    voc=load_renderer(Path("/home/db/dev/slwm-g3/artifacts/decoders/vocos-mel-24khz"),"cuda")
    extractor=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100).cuda().eval();b=load_book()
    conditions=("original","fine","equal48","allocated48","allocated192")
    errors={tag:{c:[] for c in conditions} for tag in ("whisper","ctc")};hyp={tag:{c:[] for c in conditions} for tag in ("whisper","ctc")}
    torch.set_num_threads(3)
    with torch.inference_mode():
        for k in range(0,len(data),8):
            rows=data[k:k+8];streams={c:[] for c in conditions}
            for row in rows:
                wave,sr=sf.read(ROOT/row["file"],dtype="float32");streams["original"].append(wave)
                hi=resample_poly(wave,3,2).astype(np.float32);size=len(hi);count=(size+511)//512
                padded=np.pad(hi,(0,count*512-size),mode="edge")
                spec=extractor(torch.from_numpy(padded)[None].cuda())[0].T[:count*2].cpu().numpy()
                fine=quant(spec).astype(np.float32)/16-12
                x=(fine.reshape(count,200)-b["mean"])@b["basis"]
                mels={"fine":fine}
                for method in conditions[2:]:
                    alloc=b[method];code,pc=quantize(x,b,alloc);payload,decoded=verify_clip_payload(code,b,alloc)
                    np.testing.assert_array_equal(pc,decoded)
                    mels[method]=(decoded@b["basis"].T+b["mean"]).reshape(-1,100).clip(-12,3.9375)
                for name,mel in mels.items():
                    y,_=synthesize(mel,{"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)},voc,device="cuda")
                    if len(y)<size:raise ValueError("Full decoded waveform ended early")
                    streams[name].append(resample_poly(y[:size],2,3).astype(np.float32)[:len(wave)])
            for cond,waves in streams.items():
                w=proc(waves,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
                ids=model.generate(w.input_features.cuda().half(),attention_mask=w.attention_mask.cuda(),max_new_tokens=128,num_beams=1,do_sample=False)
                primary=proc.batch_decode(ids,skip_special_tokens=True)
                tensors=[torch.from_numpy(x) for x in waves];length=torch.tensor([len(x) for x in waves],device="cuda")
                asc=torch.nn.utils.rnn.pad_sequence(tensors,batch_first=True).cuda();mask=torch.arange(asc.shape[1],device="cuda")[None]<length[:,None]
                m=asc.sum(1)/length;v=(((asc-m[:,None])**2)*mask).sum(1)/length
                z=(asc-m[:,None])*(v[:,None]+1e-7).rsqrt()*mask
                raw=ctc(z.half(),attention_mask=mask.long()).logits.argmax(-1).cpu().numpy()
                lens=ctc._get_feat_extract_output_lengths(length).cpu().tolist();second=[]
                for row,stop in zip(raw,lens):
                    sequence=row[:stop];keep=np.r_[True,sequence[1:]!=sequence[:-1]];chars=[vocab.get(int(t),"") for t in sequence[keep] if t!=0]
                    second.append("".join(v for v in chars if not v.startswith("<")).replace("|"," ").strip())
                for name,guesses in (("whisper",primary),("ctc",second)):
                    hyp[name][cond].extend(guesses);errors[name][cond].extend(edits(words(r["reference"]),words(h)) for r,h in zip(rows,guesses))
    groups=np.array([r["speaker"] for r in data]);n=np.array([len(words(r["reference"])) for r in data]);summary={};passes=[];equivalence=[]
    for tag in errors:
        vals={k:np.array(v) for k,v in errors[tag].items()};allocation=summarize(vals["allocated48"]-vals["equal48"],n,groups)
        close=summarize(vals["allocated192"]-vals["original"],n,groups)
        passes.append(allocation["speaker_ci95"][1]<0);equivalence.append(close["speaker_ci95"][0]>-.03 and close["speaker_ci95"][1]<.03)
        summary[tag]={"conditions":{k:summarize(v,n,groups) for k,v in vals.items()},"allocated_minus_equal":allocation,"high_rate_minus_original":close}
    put(OUTPUT/"recognitions.json",{"hypotheses":hyp,"edits":errors,"source_selection":config["selection"]})
    put(OUTPUT/"complete.json",{"panel":config,"scope":"Whole utterances with complete publisher transcripts. No utterance alignment teacher was used for selection.",
        "speakers":len(set(groups)),"clips":len(data),"word_count":int(n.sum()),"seconds":sum(r["samples"] for r in data)/16000,
        "scorers":summary,"allocation_gain_both_recognizers":all(passes),"high_rate_within_three_points_both":all(equivalence),"rows_sha256":sha(OUTPUT/"recognitions.json")})
    print("Whole-utterance replication complete",len(data),len(set(groups)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("select","score"));a=p.parse_args()
    if a.phase=="select":select()
    else:score()
