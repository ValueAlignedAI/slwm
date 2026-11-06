"""G1 speech cache: real MFA phones and speakers fresh to every earlier experiment."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

from training.general_data import CACHE,ROOT,order,rows_write,source
from training.general_sources import write
from training.research_sources import file_sha

OLD=Path("artifacts/t2_audio_visual/broad_stream_prepared_v1")
ALIGN=Path("artifacts/t2_audio_visual/broad_speech_v1/alignments/train-clean-100")


def prior_speakers():
    """All earlier role metadata, including former tests, counts as exposure."""
    result=set(); evidence=[]
    for root in (Path("artifacts/t2_audio_visual"),Path("experiments/multimodal/t2")):
        for path in sorted(root.rglob("*.json")):
            if any(p in path.parts for p in ("raw","sources","alignments","exploration","inference_bundle")): continue
            text=path.read_text(errors="replace")
            matches=re.findall(r'"(?:actor|speaker|person)"\s*:\s*"(?:LIBRI|libri)[^"\n]*?-(\d+)"',text)
            matches+=re.findall(r'LibriSpeech/(?:train-clean-100|dev-clean|dev-other|test-clean|test-other)/(\d+)/',text)
            if matches:
                result.update(matches); evidence.append({"path":str(path),"sha256":file_sha(path),"speakers":sorted(set(matches))})
    return result,evidence


def intervals_to_ids(intervals,length,names,unknown_words=()):
    """Actual phone intervals -> int16[T,4], -1 means unavailable supervision.

    Mel frame centers use sample clock i*256/24000. Labels never alter sound.
    Uniform dictionary-derived within-word phones are not used here.
    """
    times=np.arange(length*4)*256/24000; result=np.zeros(length*4,np.int16); lookup={p:i for i,p in enumerate(names)}
    for phone in intervals:
        a,b=float(phone["xmin"]),float(phone["xmax"]); text=phone["text"]
        if b<=a: continue
        value=lookup.get(text,-1)
        if text.lower() in ("spn","<unk>","unk"): value=-1
        result[(times>=a)&(times<b)]=value
    for a,b in unknown_words: result[(times>=a)&(times<b)]=-1
    return result.reshape(length,4)


def unpack(path,dest):
    marker=dest/(path.name+".extracted.json")
    if marker.exists():
        assert json.loads(marker.read_text())["sha256"]==file_sha(path); return
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path) as archive: archive.extractall(dest,filter="data")
    write(marker,{"sha256":file_sha(path)})


def main():
    import soundfile as sf
    import torch
    import torchaudio
    from vocos.feature_extractors import MelSpectrogramFeatures
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("plan","extract","prepare"),required=True); parser.add_argument("--archive-sha256"); args=parser.parse_args()
    folder=CACHE/"speech"; folder.mkdir(parents=True,exist_ok=True); plan_path=folder/"plan.json"
    if args.stage=="plan":
        if plan_path.exists(): print("Existing speech plan is immutable",flush=True); return
        used,evidence=prior_speakers(); write(folder/"prior_speaker_exposure.json",{"speakers":sorted(used),"evidence":evidence})
        metadata=Path("artifacts/t2_audio_visual/broad_speech_v1/raw/LibriSpeech/SPEAKERS.TXT")
        all_people=[line.split("|")[0].strip() for line in metadata.read_text().splitlines() if "|" in line and not line.startswith(";") and line.split("|")[2].strip()=="train-clean-360"]
        people=sorted(set(all_people)-used,key=order)
        if len(people)<256: raise ValueError("Need >=256 previously unexposed train-clean-360 people; got "+str(len(people)))
        roles={p:"selection" if i<16 else "calibration" if i<32 else "test" if i<64 else "train" for i,p in enumerate(people[:256])}
        training=json.loads((OLD/"libri_train-clean-100/rows.json").read_text())
        train_people={r["actor"].split("-")[-1] for r in training}; assert not train_people&set(roles)
        # Only training phone strings define the label vocabulary.
        phone_names={}
        for path in sorted(ALIGN.rglob("*.json")):
            if path.name.endswith(".extracted.json"): continue
            for record in json.loads(path.read_text()).values():
                for phone in record["phones"].values(): phone_names[phone["text"]]=True
        names=["<sil>"]+sorted(set(phone_names)-{"<sil>"})
        write(folder/"phone_vocabulary.json",names)
        write(folder/"fresh_source_revision.json",{"reason":"All40 dev-clean people appeared in previous experiment metadata; zero valid fresh people. No model fitting or final scores preceded this revision.","new_source":"Official train-clean-360, metadata-selected256 new people;192 train,16 selection,16 calibration,32 final", "previous_dev_clean_source_retained":True})
        write(plan_path,{"fresh_people":roles,"excluded_previous_people":sorted(set(all_people)&used),"all_source_people":all_people,"speaker_metadata_sha256":file_sha(metadata),
             "train_people":sorted(train_people),"training_rows_sha256":file_sha(OLD/"libri_train-clean-100/rows.json"),
             "phone_names_sha256":file_sha(folder/"phone_vocabulary.json"),"prior_exposure_sha256":file_sha(folder/"prior_speaker_exposure.json"),
             "source_sha256":file_sha(__file__),"training_count":len(training),"no_final_speaker_used_for_selection":True})
        print(json.dumps({"new_speech_people":dict(Counter(roles.values())),"previous_people_excluded":len(set(all_people)&used),"phone_labels":len(names)}),flush=True); return
    if args.stage=="extract":
        plan=json.loads(plan_path.read_text()); expected=args.archive_sha256
        if not expected or len(expected)!=64: raise ValueError("Expected complete official archive SHA256 required")
        root=ROOT/"fresh_speech_360/raw"; root.mkdir(parents=True,exist_ok=True); digest=hashlib.sha256(); stream=sys.stdin.buffer
        class HashedReader:
            def read(self,size):
                value=stream.read(size); digest.update(value); return value
        reader=HashedReader(); count=0; total=0
        with tarfile.open(fileobj=reader,mode="r|gz") as archive:
            for member in archive:
                path=Path(member.name)
                if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk(): raise ValueError("Unsafe source member")
                if not member.isfile(): continue
                parts=path.parts; take=len(parts)>2 and parts[:2]==("LibriSpeech","train-clean-360") and parts[2] in plan["fresh_people"]
                take=take or (len(parts)==2 and parts[0]=="LibriSpeech" and path.suffix in (".TXT",".txt"))
                if not take: continue
                if member.size>128*1024**2: raise ValueError("Unbounded source member")
                value=archive.extractfile(member).read(); assert len(value)==member.size
                dest=root/path; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(value); count+=1; total+=len(value)
        while reader.read(8*1024**2): pass
        assert digest.hexdigest()==expected,"Whole archive transport SHA256 mismatch"
        write(ROOT/"fresh_speech_360/extraction_manifest.json",{"source_archive_sha256":expected,"selected_members":count,"selected_bytes":total,"selected_people":plan["fresh_people"],"source_sha256":file_sha(__file__),"streamed_complete_original_archive":True})
        print(json.dumps({"speech_extracted_members":count,"bytes":total,"sha256":expected}),flush=True); return
    if (folder/"manifest.json").exists(): return
    plan=json.loads(plan_path.read_text()); names=json.loads((folder/"phone_vocabulary.json").read_text()); torch.set_num_threads(1)
    training=json.loads((OLD/"libri_train-clean-100/rows.json").read_text()); train_index={r["id"]:r for r in training}; train_align={}
    for path in sorted(ALIGN.rglob("*.json")):
        if path.name.endswith(".extracted.json"): continue
        for key,value in json.loads(path.read_text()).items(): train_align[Path(key).stem]=value
    original=np.load(OLD/"libri_train-clean-100/mel.npy",mmap_mode="r"); target=np.lib.format.open_memmap(folder/"train_real_phones.npy",mode="w+",dtype=np.int16,shape=(len(original),4)); target[:]=-1; rows=[]
    for record in training:
        key=record["id"]; aligned=train_align[key]; unknown=[(float(v["xmin"]),float(v["xmax"])) for v in aligned["words"].values() if "<unk>" in v["text"]]
        labels=intervals_to_ids(aligned["phones"].values(),record["length"],names,unknown)
        if not record.get("phone_supervision_usable",True): labels[:]=-1
        a=record["offset"]; target[a:a+record["length"]]=labels
        rows.append({**record,"role":"train","group":record["actor"],"mel_file":str(OLD/"libri_train-clean-100/mel.npy"),"phone_file":str(folder/"train_real_phones.npy"),"phone_unknown_fraction":float(np.mean(labels<0)),"phone_timing":"Actual MFA phone intervals; ambiguous/unknown word intervals excluded"})
    target.flush(); del target,train_align
    extractor=MelSpectrogramFeatures(); transcripts={}; fresh=[]
    assert (ROOT/"fresh_speech_360/extraction_manifest.json").exists()
    unpack(source("fresh_speech_360","LibriSpeech-aligned-train-clean-360.tar.gz"),ROOT/"fresh_speech_360/alignments")
    for path in sorted((ROOT/"fresh_speech_360/alignments").rglob("*.json")):
        if path.name.endswith(".extracted.json"): continue
        for key,aligned in json.loads(path.read_text()).items():
            stem=Path(key).stem; person,chapter,_=stem.split("-")
            if person not in plan["fresh_people"]: continue
            audio=ROOT/"fresh_speech_360/raw/LibriSpeech/train-clean-360"/person/chapter/(stem+".flac")
            chapter_key=person+"-"+chapter
            if chapter_key not in transcripts: transcripts[chapter_key]=dict(line.split(" ",1) for line in (audio.parent/(chapter_key+".trans.txt")).read_text().splitlines())
            signal,sr=sf.read(audio,dtype="float32"); tensor=torch.from_numpy(signal)[None]
            if sr!=24000: tensor=torchaudio.functional.resample(tensor,sr,24000)
            mel=extractor(tensor)[0].T.numpy(); length=len(mel)//4
            unknown=[(float(v["xmin"]),float(v["xmax"])) for v in aligned["words"].values() if "<unk>" in v["text"]]
            units=intervals_to_ids(aligned["phones"].values(),length,names,unknown)
            spans=[(float(w["xmin"]),float(w["xmax"]),w["text"]) for w in aligned["words"].values() if float(w["xmax"])>float(w["xmin"])]
            fresh.append((mel[:length*4].reshape(length,400).astype(np.float16),units,{"id":stem,"actor":"LIBRI-"+person,"group":"LIBRI-"+person,"role":plan["fresh_people"][person],"duration":len(signal)/sr,"length":length,"raw_path":str(audio),"transcript":transcripts[chapter_key][stem],"spans":spans,"raw_sha256":file_sha(audio),"phone_unknown_fraction":float(np.mean(units<0))}))
    count=sum(len(x) for x,_,_ in fresh); features=np.lib.format.open_memmap(folder/"fresh_mel.npy",mode="w+",dtype=np.float16,shape=(count,400)); phones=np.lib.format.open_memmap(folder/"fresh_real_phones.npy",mode="w+",dtype=np.int16,shape=(count,4)); offset=0
    for mel,units,record in fresh:
        features[offset:offset+len(mel)]=mel; phones[offset:offset+len(mel)]=units
        record.update(offset=offset,mel_file=str(folder/"fresh_mel.npy"),phone_file=str(folder/"fresh_real_phones.npy")); rows.append(record); offset+=len(mel)
    features.flush(); phones.flush(); del features,phones
    rows_write(folder,rows,training_hours=sum(r["duration"] for r in rows if r["role"]=="train")/3600,
        fresh_hours={role:sum(r["duration"] for r in rows if r["role"]==role)/3600 for role in ("selection","calibration","test")},
        phone_count=len(names),files={p.name:file_sha(p) for p in folder.glob("*.npy")},external_mel_codec="Fixed100band Vocos feature transform; no acoustic/recognition teacher used in training",source_plan_sha256=file_sha(plan_path))


if __name__=="__main__": main()
