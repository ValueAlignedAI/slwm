"""Original broad speech/continuous AV sources, metadata and pinned audio decoder."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile
from training.research_sources import fetch,file_sha
from training.t2_real_av import write_json


def unpack_tar(source,dest):
    marker=dest/(source.name+".extracted.json")
    if marker.exists(): return
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(source) as archive: archive.extractall(dest,filter="data")
    write_json(marker,{"source_sha256":file_sha(source)})


def main():
    p=argparse.ArgumentParser(); p.add_argument("--kind",choices=("libri","ami","vocoder","alignments"),required=True); a=p.parse_args(); cfg=json.loads(Path("configs/t2/broad_stream_data.json").read_text())
    if a.kind=="libri":
        root=Path(cfg["libri_root"]); jobs=list(cfg["libri_archives"].items())
        def one(item):
            split,digest=item; source=root/"sources"/(split+".tar.gz"); fetch("https://www.openslr.org/resources/12/"+source.name,source)
            h=hashlib.md5()
            with source.open("rb") as f:
                for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
            if h.hexdigest()!=digest: raise ValueError("Official LibriSpeech MD5 mismatch")
            unpack_tar(source,root/"raw"); return split
        with ThreadPoolExecutor(max_workers=2) as pool:
            for split in pool.map(one,jobs): print(json.dumps({"libri_source_ready":split}),flush=True)
    elif a.kind=="ami":
        root=Path(cfg["ami_root"]); annotation=root/"sources/ami_public_manual_1.6.2.zip"; fetch("https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip",annotation)
        if not (root/"annotations/corpusResources/meetings.xml").exists():
            dest=(root/"annotations").resolve(); dest.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(annotation) as archive:
                for entry in archive.infolist():
                    target=(dest/entry.filename).resolve()
                    if not target.is_relative_to(dest): raise ValueError("Unsafe annotation member")
                archive.extractall(dest)
        jobs=[]
        for meetings in cfg["ami_meetings"].values():
            for meeting in meetings:
                for index in range(4):
                    for kind,name in (("audio",f"{meeting}.Headset-{index}.wav"),("video",f"{meeting}.Closeup{index+1}.avi")):
                        jobs.append((f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/{kind}/{name}",root/"sources"/meeting/name))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(lambda j:fetch(*j),jobs): pass
    elif a.kind=="alignments":
        digests={"train-clean-100":"edaf7f8207b2a9a65d81c8d65a92ced9095d2de70cce71bdccad36f7a9f3800c","dev-other":"f0040c44871b71b5286c94de3969097eca1e765fb2136b1bbf9dfa45e95c75a3","test-clean":"0b361593d13f850d5f69530d270b7ec6b5bf3fef315d68f6632e11b220d92921","test-other":"751fbb81f4fa10208a792c6a6e8f064f07cf4ef9ffa85737358a711597be4e2b"}
        root=Path(cfg["libri_root"])
        for split,digest in digests.items():
            name=f"LibriSpeech-aligned-{split}.tar.gz"; source=root/"sources"/name
            fetch("https://github.com/Tom-Brenner/librispeech-vctk-phone-alignments/releases/download/v1.0.0/"+name,source,digest)
            unpack_tar(source,root/"alignments"/split)
    else:
        root=Path("artifacts/decoders/vocos-mel-24khz"); root.mkdir(parents=True,exist_ok=True)
        for name in ("config.yaml","pytorch_model.bin","README.md"):
            fetch(f"https://huggingface.co/{cfg['vocoder']}/resolve/{cfg['vocoder_revision']}/{name}",root/name)
        write_json(root/"provenance.json",{"model":cfg["vocoder"],"revision":cfg["vocoder_revision"],"role":"External pretrained frozen waveform decoder, shared by all arms; not SLWM-learned linguistic knowledge","files":{n:file_sha(root/n) for n in ("config.yaml","pytorch_model.bin")}})


if __name__=="__main__": main()
