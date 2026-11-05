"""Original GRID AV + aligned words, held-out people AND sentence combinations."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import zipfile
import zlib
import numpy as np
import torch
from exploration.video_repair_audio import mel100
from training.research_sources import file_sha
from training.t2_real_av import stable_hash,write_json


def alignments(archive):
    rows={}
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".align"): continue
            text=tar.extractfile(member).read().decode("ascii"); spans=[]
            for line in text.splitlines():
                start,end,word=line.split(); spans.append((int(start)/25000.,int(end)/25000.,word.lower()))
            words=[w for _,_,w in spans if w not in {"sil","sp"}]; transcript=" ".join(words)
            bucket=int(hashlib.sha256(transcript.encode()).hexdigest()[:12],16)%10
            rows[Path(member.name).stem]={"spans":spans,"transcript":transcript,"bucket":bucket,"alignment_sha256":hashlib.sha256(text.encode()).hexdigest()}
    return rows


def decode_record(path,cfg):
    """Embedded media -> original uint8[T,64,96], synchronized PCM16[N]."""
    meta=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=codec_type,start_time","-of","json",str(path)]))
    starts={r["codec_type"]:float(r.get("start_time",0)) for r in meta["streams"]}; difference=starts["video"]-starts["audio"]
    raw=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-an","-vf",cfg["crop"],"-frames:v",str(cfg["max_packets"]),"-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45).stdout
    pixels=np.frombuffer(raw,np.uint8).reshape(-1,64,96).copy()
    sound=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",str(path),"-vn","-ac","1","-ar","16000","-f","s16le","-"],capture_output=True,check=True,timeout=45).stdout
    pcm=np.frombuffer(sound,"<i2").copy(); trim=int(round(difference*16000))
    pcm=pcm[trim:] if trim>=0 else np.pad(pcm,(-trim,0))
    return pixels,pcm,{"stream_start_seconds":starts,"audio_start_trim_samples":trim}


def prepare_person(person,cfg,vocab):
    root=Path(cfg["root"]); source=root/"sources"/f"s{person:02d}"; zip_path=source/f"s{person}.mpg_vcd.zip"; alignment_path=source/f"s{person}.tar"
    while not zip_path.exists() or not alignment_path.exists(): time.sleep(5)
    labels=alignments(alignment_path)
    role=next((s for s in ("train","validation","calibration","test") if person in cfg[s+"_people"]),None); wanted=set(cfg["sentence_hash_buckets"][role]); selected=[]
    for key,row in labels.items():
        if row["bucket"] in wanted: selected.append((key,role,row))
    if role=="test":
        seen=sorted([(k,r) for k,r in labels.items() if r["bucket"]<8],key=lambda p:hashlib.sha256(p[0].encode()).hexdigest())[:32]
        selected.extend((k,"test_seen",r) for k,r in seen)
    if role=="train":
        seen=sorted([(k,r) for k,r in labels.items() if r["bucket"]==8],key=lambda p:hashlib.sha256(p[0].encode()).hexdigest())[:16]
        selected.extend((k,"train_person_holdout",r) for k,r in seen)
    with np.load(Path(cfg["codec_root"])/"codec100.npz",allow_pickle=False) as c: mean,scale=c["mean"],c["scale"]
    rawdir=root/"raw"/f"s{person:02d}"; rawdir.mkdir(parents=True,exist_ok=True); cache=root/"decoded"/f"s{person:02d}"; cache.mkdir(parents=True,exist_ok=True); records=[]
    with zipfile.ZipFile(zip_path) as archive:
        members={Path(m.filename).stem.split("_")[-1]:m for m in archive.infolist() if m.filename.endswith(".mpg")}
        jobs=[]
        for key,split,row in sorted(selected):
            if key not in members: raise ValueError("Missing GRID video "+key)
            member=members[key]; raw=rawdir/(key+".mpg")
            if not raw.exists() or zlib.crc32(raw.read_bytes())!=member.CRC: raw.write_bytes(archive.read(member))
            jobs.append((key,split,row,raw,member.CRC))
    def decode(job):
        key,split,row,raw,crc=job; path=cache/(key+".npz")
        if not path.exists():
            pixels,pcm,timing=decode_record(raw,cfg); physical=mel100(torch.from_numpy(pcm.astype(np.float32)/32768.)).numpy()
            length=min(len(pixels),(len(physical)+3)//4,cfg["max_packets"]); pad=max(0,4*length-len(physical)); physical=np.pad(physical,((0,pad),(0,0)),mode="edge")
            packed=((physical[:4*length]-mean)/scale).reshape(length,256).astype(np.float16); word_ids=np.zeros(length,np.int16)
            for begin,end,word in row["spans"]:
                use=(np.arange(length)*.04>=begin)&(np.arange(length)*.04<end); word_ids[use]=vocab[word]
            temp=path.with_suffix(".part.npz"); np.savez_compressed(temp,mouth=pixels[:length],mel100=packed,labels=word_ids,pcm=pcm,config_hash=np.array(stable_hash(cfg)),timing=np.array(json.dumps({**timing,"terminal_padded_subframes":pad}))); temp.replace(path)
        with np.load(path,allow_pickle=False) as d:
            if str(d["config_hash"])!=stable_hash(cfg): raise ValueError("Use a new corpus version for crop/timing changes")
            length=len(d["mouth"]); timing=json.loads(str(d["timing"]))
        if length<50: raise ValueError("GRID retained clip too short")
        return {"sample_id":f"GRID-s{person:02d}-"+key,"actor":f"GRID-{person:02d}","split":split,"sentence":row["transcript"],"transcript":row["transcript"],"sentence_bucket":row["bucket"],
            "spans":row["spans"],"raw_path":str(raw),"raw_sha256":file_sha(raw),"zip_member_crc32":crc,"decoded_path":str(path),"frames":length,"timing":timing,"alignment_sha256":row["alignment_sha256"]}
    with ThreadPoolExecutor(max_workers=4) as pool: records=list(pool.map(decode,jobs))
    print(json.dumps({"prepared_person":person,"role":role,"samples":len(records)}),flush=True); return records


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t2/grid_content_data.json"); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["root"]); torch.set_num_threads(1)
    if (root/"dataset_card.json").exists(): return
    words={"sil"}
    for person in cfg["train_people"]:
        path=root/"sources"/f"s{person:02d}"/f"s{person}.tar"
        while not path.exists(): time.sleep(5)
        words.update(w for r in alignments(path).values() if r["bucket"]<8 for _,_,w in r["spans"])
    vocab={word:i for i,word in enumerate(["sil"]+sorted(words-{"sil"}))}; write_json(root/"vocabulary.json",vocab)
    rows=[]
    for person in cfg["train_people"]+cfg["validation_people"]+cfg["calibration_people"]+cfg["test_people"]: rows.extend(prepare_person(person,cfg,vocab))
    splits={}
    for split in ("train","validation","calibration","test","test_seen","train_person_holdout"):
        group=sorted([r for r in rows if r["split"]==split],key=lambda r:r["sample_id"]); n=len(group); paths={}; t=cfg["max_packets"]
        arrays={"mouth":np.lib.format.open_memmap(root/(split+"_mouth.npy"),mode="w+",dtype=np.uint8,shape=(n,t,64,96)),"mel100":np.zeros((n,t,256),np.float16),"labels":np.full((n,t),-100,np.int16),"lengths":np.zeros(n,np.int32)}
        arrays["mouth"][:]=0
        for i,row in enumerate(group):
            length=row["frames"]
            with np.load(row["decoded_path"],allow_pickle=False) as d:
                for name in ("mouth","mel100","labels"): arrays[name][i,:length]=d[name]
            arrays["lengths"][i]=length
        arrays["mouth"].flush()
        for name,value in arrays.items():
            path=root/(split+"_"+name+".npy")
            if name!="mouth": np.save(path,value)
            paths[name]={"path":str(path),"sha256":file_sha(path)}
        del arrays; write_json(root/(split+".json"),group)
        card={"split":split,"samples":n,"people":sorted({r["actor"] for r in group}),"arrays":paths,"config_hash":stable_hash(cfg),"metadata_sha256":file_sha(root/(split+".json")),"retained_seconds":sum(r["frames"] for r in group)*.04}
        write_json(root/(split+"_manifest.json"),card); splits[split]=card; print(json.dumps({"prepared_split":split,"samples":n}),flush=True)
    sets={s:set(r["sentence"] for r in rows if r["split"]==s) for s in splits}; assert not(sets["train"]&sets["validation"] or sets["train"]&sets["test"] or sets["validation"]&sets["test"])
    sources={str(p.relative_to(root)):file_sha(p) for p in sorted((root/"sources").rglob("*.source.json"))}
    write_json(root/"dataset_card.json",{"config":cfg,"config_hash":stable_hash(cfg),"splits":splits,"vocabulary":vocab,"vocabulary_fit":"Only training-person training-sentence alignments","codec_fit":"None; unchanged earlier CREMA train statistics","source_cards":sources,"sentence_combinations_disjoint":True,"primary_people_disjoint":True,
        "scope":"Acted six-word grammar, new speakers and compositional sentences; not open vocabulary or general world modeling","source_sha256":file_sha(Path(__file__))})


if __name__=="__main__": main()
