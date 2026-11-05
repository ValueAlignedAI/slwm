"""Training-only codecs, broad speech and participant-disjoint continuous AMI."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import soundfile as sf
import torch
import torchaudio
import cv2
from vocos.feature_extractors import MelSpectrogramFeatures
from training.stream_codec import PACKET_SECONDS,pronunciation_codec,timed_phones,words,pronounce,dictionary_hash
from training.research_sources import file_sha
from training.t2_real_av import write_json

CFG="configs/t2/broad_stream_data.json"


def stable_order(value): return hashlib.sha256(("B1:"+str(value)).encode()).hexdigest()


def prior_people():
    result=set()
    for name in ("restoration_mixed_v1/libri_train.json","restoration_mixed_v1/libri_validation.json","restoration_mixed_v1/libri_test.json","restoration_other_v1/other_test.json"):
        path=Path("artifacts/t2_audio_visual")/name
        for row in json.loads(path.read_text()): result.add(row["actor"].split("-")[-1])
    return result


def libri_rows(cfg,split):
    root=Path(cfg["libri_root"])
    if split in ("train-clean-100","dev-other"): raw=root/"raw/LibriSpeech"/split
    else: raw=Path("artifacts/t2_audio_visual")/("restoration_mixed_v1" if split=="test-clean" else "restoration_other_v1")/"raw/LibriSpeech"/split
    alignment=root/"alignments"/split
    rows=[]; transcripts={}
    for path in sorted(alignment.glob("**/*.json")):
        if path.name.endswith(".extracted.json"): continue
        data=json.loads(path.read_text()); alignment_digest=file_sha(path)
        for name,item in data.items():
            stem=Path(name).stem; speaker,chapter,_=stem.split("-"); audio=raw/speaker/chapter/(stem+".flac")
            if not audio.exists(): raise FileNotFoundError(audio)
            info=sf.info(audio)
            chapter_id=speaker+"-"+chapter
            if chapter_id not in transcripts:
                transcripts[chapter_id]=dict(line.split(" ",1) for line in (audio.parent/(chapter_id+".trans.txt")).read_text().splitlines() if " " in line)
            original_transcript=transcripts[chapter_id][stem]
            spans=[(float(w["xmin"]),float(w["xmax"]),w["text"].lower()) for w in item["words"].values() if words(w["text"]) and float(w["xmax"])>float(w["xmin"])]
            if info.duration<2 or not spans: continue
            rows.append({"id":stem,"actor":"LIBRI-"+speaker,"source":"libri","subset":split,"raw_path":str(audio),"duration":info.duration,"spans":spans,"transcript":original_transcript,"alignment_words_agree":"".join(words(original_transcript)).replace("'", "")=="".join(words(" ".join(w for _,_,w in spans))).replace("'", ""),"alignment_source_sha256":alignment_digest})
    if split=="dev-other":
        people=sorted({r["actor"] for r in rows},key=stable_order)[:20]; selected=[]
        for person in people: selected+=sorted([r for r in rows if r["actor"]==person],key=lambda r:stable_order(r["id"]))[:12]
        rows=selected
    if split.startswith("test-"):
        used=prior_people(); available=[r for r in rows if r["actor"].split("-")[-1] not in used and 8<=r["duration"]<=18]
        people=sorted({r["actor"] for r in available},key=stable_order)[:12]
        if len(people)!=12: raise ValueError("Insufficient fresh confirmation people")
        rows=[]
        for person in people:
            cases=sorted([r for r in available if r["actor"]==person],key=lambda r:stable_order(r["id"]))[:4]
            if len(cases)!=4: raise ValueError("Insufficient fixed clips for "+person)
            rows+=cases
    return rows


def ami_rows(cfg,role):
    root=Path(cfg["ami_root"]); meetings={m.attrib["observation"]:m for m in ET.parse(root/"annotations/corpusResources/meetings.xml").getroot()}; rows=[]
    for name in cfg["ami_meetings"][role]:
        for speaker in meetings[name]:
            s=speaker.attrib; wordfile=root/"annotations/words"/(name+"."+s["nxt_agent"]+".words.xml"); spans=[]
            for word in ET.parse(wordfile).getroot():
                a=word.attrib
                if word.tag!="w" or a.get("punc")=="true" or not words(word.text or "") or "starttime" not in a: continue
                begin,end=float(a["starttime"]),float(a["endtime"])
                if end>begin: spans.append((begin,end,(word.text or "").lower()))
            audio=root/"sources"/name/(name+".Headset-"+s["channel"]+".wav"); video=root/"sources"/name/(name+"."+s["camera"]+".avi")
            if not audio.exists() or not video.exists(): raise FileNotFoundError("AMI signals incomplete: "+name)
            video_info=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(video)]))
            audio_duration=sf.info(audio).duration; video_duration=float(video_info["format"]["duration"])
            rows.append({"id":name+"-"+s["nxt_agent"],"actor":"AMI-"+s["global_name"],"source":"ami","subset":role,"raw_path":str(audio),"video_path":str(video),"duration":min(audio_duration,video_duration),"audio_duration":audio_duration,"video_duration":video_duration,"shared_timeline_only":True,"spans":spans,"transcript":" ".join(w for _,_,w in spans),"alignment_source_sha256":file_sha(wordfile)})
    return rows


def face_video(row,dest,length):
    """Causal classical face tracking, a frozen fallback, and64×64 gray outputs."""
    path=dest/(row["id"]+"_pixels.npy"); marker=dest/(row["id"]+"_pixels.json")
    if marker.exists(): return json.loads(marker.read_text())
    meta=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,start_time","-of","json",row["video_path"]]))["streams"][0]
    if abs(float(meta.get("start_time",0)))>.05: raise ValueError("Unexpected AMI video timeline")
    w,h=int(meta["width"]),int(meta["height"]); output=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint8,shape=(length,64,64)); cascade_path=Path(cv2.data.haarcascades)/"haarcascade_frontalface_default.xml"; classifier=cv2.CascadeClassifier(str(cascade_path)); profile=cv2.CascadeClassifier(str(Path(cv2.data.haarcascades)/"haarcascade_profileface.xml"))
    process=subprocess.Popen(["ffmpeg","-v","error","-threads","1","-i",row["video_path"],"-an","-vf","setpts=PTS-STARTPTS,fps=375/16","-pix_fmt","gray","-f","rawvideo","-threads","1","-"],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    box=None; detections=0; fallback=0; count=0
    try:
        for i in range(length):
            raw=process.stdout.read(w*h)
            if len(raw)!=w*h: break
            frame=np.frombuffer(raw,np.uint8).reshape(h,w)
            if i%16==0:
                faces=classifier.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4,minSize=(30,30))
                if not len(faces): faces=profile.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4,minSize=(30,30))
                if len(faces):
                    found=max(faces,key=lambda r:r[2]*r[3]); found=np.asarray(found,np.float32); box=found if box is None else .7*box+.3*found; detections+=1
            if box is None: crop=frame; fallback+=1
            else:
                x,y,bw,bh=box; side=min(max(bw,bh)*1.5,w,h); left=int(np.clip(x+bw/2-side/2,0,w-side)); top=int(np.clip(y+bh/2-side/2,0,h-side)); crop=frame[top:top+int(side),left:left+int(side)]
            output[i]=cv2.resize(crop,(64,64),interpolation=cv2.INTER_AREA); count=i+1
    finally:
        process.stdout.close(); process.terminate(); process.wait(); output.flush(); del output
    if count<length-2: raise ValueError("AMI audio/video duration mismatch: "+row["id"])
    if count<length:
        output=np.load(path,mmap_mode="r+"); output[count:]=output[count-1]; output.flush()
    result={"frames":length,"decoded_frames":count,"terminal_repeated_frames":length-count,"face_detections":detections,"fallback_frames":fallback,"detector_sha256":file_sha(cascade_path),"raw_video_sha256":file_sha(row["video_path"]),"video_start_time":float(meta.get("start_time",0))}; write_json(marker,result); return result



def physical_mel(tensor,extractor):
    """Chunked bounded-workspace STFT, equal to the full-recording sample clock."""
    physical=[]; block=24000*32; hop=256
    for begin in range(0,tensor.shape[1],block):
        lo=max(0,begin-1024); hi=min(tensor.shape[1],begin+block+1024); mel=extractor(tensor[:,lo:hi])[0].T; first=(begin-lo)//hop; number=min(block,tensor.shape[1]-begin)//hop
        if begin+block>=tensor.shape[1]: number+=1
        physical.append(mel[first:first+number])
    return torch.cat(physical)


def pack(rows,dest,video=False):
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"manifest.json").exists(): return json.loads((dest/"rows.json").read_text())
    offset=0
    for row in rows:
        frames24=int(np.ceil(sf.info(row["raw_path"]).frames*24000/sf.info(row["raw_path"]).samplerate)); frames24=min(frames24,int(round(row["duration"]*24000))); length=(1+frames24//256)//4; row.update(offset=offset,length=length); offset+=length
    features=np.lib.format.open_memmap(dest/"mel.npy",mode="w+",dtype=np.float16,shape=(offset,400)); labels=np.lib.format.open_memmap(dest/"phones.npy",mode="w+",dtype=np.int16,shape=(offset,)); extractor=MelSpectrogramFeatures(); resample=torchaudio.transforms.Resample(16000,24000); started=time.time()
    def work(row):
        audio,sr=sf.read(row["raw_path"],dtype="float32",always_2d=True); tensor=torch.from_numpy(audio.mean(1))[None]
        if sr!=24000: tensor=resample(tensor) if sr==16000 else torchaudio.functional.resample(tensor,sr,24000)
        tensor=tensor[:,:int(round(row["duration"]*24000))]
        length=row["length"]; mel=physical_mel(tensor,extractor).numpy()[:length*4]; assert len(mel)==length*4
        a=row["offset"]; features[a:a+length]=mel.reshape(length,400).astype(np.float16); labels[a:a+length]=timed_phones(row["spans"],length); row["raw_sha256"]=file_sha(row["raw_path"])
        if video: row["video_decode"]=face_video(row,dest,length)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        done=[]
        for row in pool.map(work,rows):
            done.append(row)
            if len(done)%256==0 or video: print(json.dumps({"packed":str(dest),"records":len(done),"total":len(rows),"seconds":time.time()-started}),flush=True)
    features.flush(); labels.flush(); write_json(dest/"rows.json",done); write_json(dest/"manifest.json",{"records":len(done),"packets":offset,"files":{n:file_sha(dest/n) for n in ("mel.npy","phones.npy","rows.json")},"source_sha256":file_sha(Path(__file__))}); return done


def fit_codecs(lib,ami,root):
    if (root/"codec.npz").exists(): return
    stats=np.load(lib/"mel.npy",mmap_mode="r"); total=0; first=np.zeros(100,np.float64); second=first.copy()
    for begin in range(0,len(stats),65536):
        block=np.asarray(stats[begin:begin+65536],np.float64).reshape(-1,100); first+=block.sum(0); second+=(block*block).sum(0); total+=len(block)
    mean=first/total; scale=np.sqrt(np.maximum(second/total-mean*mean,.01)); rows=json.loads((ami/"rows.json").read_text()); samples=[]
    for row in rows:
        pixels=np.load(ami/(row["id"]+"_pixels.npy"),mmap_mode="r"); ids=np.linspace(0,len(pixels)-1,512).astype(int); samples.append(np.asarray(pixels[ids],np.float32).reshape(512,-1)/255.)
    x=torch.from_numpy(np.concatenate(samples)); pixel_mean=x.mean(0); x-=pixel_mean; torch.manual_seed(93001); u,s,v=torch.pca_lowrank(x,q=72,niter=3,center=False); basis=v[:,:64].T.contiguous(); projected=x@basis.T; video_scale=projected.std(0).clamp_min(.02)
    np.savez(root/"codec.npz",audio_mean=mean.astype(np.float32),audio_scale=scale.astype(np.float32),pixel_mean=pixel_mean.numpy(),video_basis=basis.numpy(),video_scale=video_scale.numpy()); print(json.dumps({"training_codecs_fitted":True,"video_frames":len(x)}),flush=True)


def video_codes(dest,codec):
    rows=json.loads((dest/"rows.json").read_text()); c=np.load(codec,allow_pickle=False)
    for row in rows:
        path=dest/(row["id"]+"_video.npy")
        if path.exists(): continue
        pixels=np.load(dest/(row["id"]+"_pixels.npy"),mmap_mode="r"); out=np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(len(pixels),64))
        for begin in range(0,len(pixels),512):
            block=np.asarray(pixels[begin:begin+512],np.float32).reshape(-1,4096)/255.; out[begin:begin+512]=((block-c["pixel_mean"])@c["video_basis"].T)/c["video_scale"]
        out.flush(); del out


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("plan","prepare","codecs"),required=True); p.add_argument("--family",choices=("all","libri","ami"),default="all"); a=p.parse_args(); cfg=json.loads(Path(CFG).read_text()); root=Path("artifacts/t2_audio_visual/broad_stream_prepared_v1"); root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    if a.stage=="plan":
        plans={}
        if a.family in ("all","libri"):
            for split in ("train-clean-100","dev-other","test-clean","test-other"): plans["libri_"+split]=libri_rows(cfg,split)
        if a.family in ("all","ami"):
            for role in cfg["ami_meetings"]: plans["ami_"+role]=ami_rows(cfg,role)
        people={s:{r["actor"] for r in rows} for s,rows in plans.items()}
        for one in people:
            for two in people:
                if one<two and one.split("_")[0]==two.split("_")[0]: assert not people[one]&people[two],(one,two)
        for name,rows in plans.items(): write_json(root/(name+"_plan.json"),rows)
        write_json(root/("split_registration_"+a.family+".json"),{"roles":{s:{"records":len(r),"people":len(people[s]),"hours":sum(x["duration"] for x in r)/3600} for s,r in plans.items()},"prior_exposed_speakers":sorted(prior_people()),"source_sha256":file_sha(Path(__file__)),"config_sha256":file_sha(CFG),"frozen_before_weights":True}); print(json.dumps({"plans_registered":True}),flush=True); return
    if a.stage=="prepare":
        for path in sorted(root.glob("*_plan.json"),key=lambda p:("train" not in p.name,p.name)):
            name=path.name.removesuffix("_plan.json")
            if a.family!="all" and not name.startswith(a.family+"_"): continue
            pack(json.loads(path.read_text()),root/name,video=name.startswith("ami_"))
        return
    torch.set_num_threads(4)
    fit_codecs(root/"libri_train-clean-100",root/"ami_train",root)
    for role in cfg["ami_meetings"]: video_codes(root/("ami_"+role),root/"codec.npz")
    duration=[]; gaps=[]
    for name in ("libri_train-clean-100","ami_train"):
        for row in json.loads((root/name/"rows.json").read_text()):
            spans=row["spans"]
            for i,(begin,end,word) in enumerate(spans):
                if end>begin and len(pronounce(word))>1: duration.append(([1.,np.log(len(pronounce(word))),np.log(max(1,len(word)))],np.log(np.clip(end-begin,.03,2))))
                if i and 0<=begin-spans[i-1][1]<.3: gaps.append(begin-spans[i-1][1])
    x=np.asarray([a for a,_ in duration]); y=np.asarray([b for _,b in duration]); coef=np.linalg.solve(x.T@x+np.eye(3),x.T@y); write_json(root/"duration_model.json",{"coefficients":coef.tolist(),"interword_gap":float(np.clip(np.mean(gaps),.01,.12)),"training_word_spans":len(x),"phoneme_timing":"CMU first pronunciation, stress removed, uniform within actual word boundaries; not claimed true phone alignment"})
    _,names,_=pronunciation_codec(); write_json(root/"vocabulary.json",names); write_json(root/"corpus_hashes.json",{str(p.relative_to(root)):file_sha(p) for p in root.rglob("*.npy")})
    write_json(root/"dataset_card.json",{"splits":{p.stem:json.loads(p.read_text()) for p in root.glob("split_registration_*.json")},"corpus_hashes_sha256":file_sha(root/"corpus_hashes.json"),"codec_sha256":file_sha(root/"codec.npz"),"duration_model_sha256":file_sha(root/"duration_model.json"),"dictionary_sha256":dictionary_hash(),"sample_rate":24000,"packet_seconds":PACKET_SECONDS,"external_vocoder":"Pinned Vocos waveform renderer shared by all arms, no generator training or test fitting","video":"Causal classical face tracking and64x64gray training-only64D PCA; no pretrained video generator"}); print(json.dumps({"broad_stream_corpus_ready":True}),flush=True)


if __name__=="__main__": main()
