"""Pre-fit revision: restore official words at unambiguous MFA unknown spans."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
from training.stream_codec import pronounce,timed_phones
from training.research_sources import file_sha
from training.t2_real_av import write_json


def canonical(word): return re.sub("[^a-z]","",word.lower())


def restore(row):
    """Exact one-to-one token slots only; ambiguous alignments lose lexical supervision."""
    official=row["transcript"].lower().split(); spans=row["spans"]
    if len(official)!=len(spans): return spans,False,"token_count"
    for word,(_,_,aligned) in zip(official,spans):
        if aligned=="<unk>" or canonical(word)==canonical(aligned): continue
        if "<unk>" in aligned:
            before,after=aligned.split("<unk>",1); a,b=canonical(before),canonical(after); w=canonical(word)
            if w.startswith(a) and w.endswith(b) and len(w)>len(a)+len(b): continue
        if pronounce(word)!=[1] and pronounce(word)==pronounce(aligned): continue
        return spans,False,"nonmatching_known_word"
    return [(a,b,w) for (a,b,_),w in zip(spans,official)],True,"exact_slots"


def main():
    root=Path("artifacts/t2_audio_visual/broad_stream_prepared_v1"); marker=root/"annotation_review_resolved.json"
    if marker.exists(): return
    if not(root/"dataset_card.json").exists(): raise ValueError("Finish and retain original preparation before annotation revision")
    review={"revision":2,"neural_weights_fitted_before_review":False,"source_sha256":file_sha(Path(__file__)),"splits":{}}
    for name in ("dataset_card.json","duration_model.json","corpus_hashes.json"):
        target=root/(name.removesuffix(".json")+"_initial.json")
        if not target.exists(): shutil.copyfile(root/name,target)
    for folder in sorted(root.glob("libri_*")):
        if not folder.is_dir() or not(folder/"manifest.json").exists(): continue
        original=folder/"rows_initial.json"
        if not original.exists(): shutil.copyfile(folder/"rows.json",original)
        if not(folder/"phones_initial.npy").exists(): shutil.copyfile(folder/"phones.npy",folder/"phones_initial.npy")
        if not(folder/"manifest_initial.json").exists(): shutil.copyfile(folder/"manifest.json",folder/"manifest_initial.json")
        rows=json.loads(original.read_text()); units=np.load(folder/"phones.npy",mmap_mode="r+"); changed=0; unresolved=[]; mismatched=0
        for row in rows:
            mismatched+=not row["alignment_words_agree"]; fixed,usable,reason=restore(row)
            if usable:
                changed+=fixed!=[tuple(x) for x in row["spans"]]; row["spans"]=fixed; at=row["offset"]; units[at:at+row["length"]]=timed_phones(fixed,row["length"])
            else: unresolved.append({"id":row["id"],"reason":reason})
            row["ctc_alignment_usable"]=usable; row["phone_supervision_usable"]=usable; row["text_generation_usable"]=usable
        units.flush(); del units; write_json(folder/"rows.json",rows); manifest=json.loads((folder/"manifest_initial.json").read_text()); manifest["files"]={n:file_sha(folder/n) for n in ("mel.npy","phones.npy","rows.json")}; assert manifest["files"]["mel.npy"]==json.loads((folder/"manifest_initial.json").read_text())["files"]["mel.npy"]; manifest["annotation_revision"]=2; manifest["annotation_source_sha256"]=file_sha(Path(__file__)); write_json(folder/"manifest.json",manifest)
        review["splits"][folder.name]={"records":len(rows),"initial_string_disagreements":int(mismatched),"corrected_span_strings":changed,"lexically_usable":len(rows)-len(unresolved),"unresolved_lexical_only_masks":unresolved,"acoustic_examples_removed":0,"mel_values_unchanged":True}
        print(json.dumps({"annotation_review":folder.name,**{k:v for k,v in review["splits"][folder.name].items() if not isinstance(v,list)}}),flush=True)
    # PCA/mel statistics are reused exactly; only duration targets and annotation
    # hashes are recomputed. Original metadata/phone arrays remain beside them.
    subprocess.run([sys.executable,"-m","training.broad_data","--stage","codecs"],check=True)
    durations=[]; gaps=[]
    for name in ("libri_train-clean-100","ami_train"):
        for row in json.loads((root/name/"rows.json").read_text()):
            if not row.get("text_generation_usable",True): continue
            for i,(begin,end,word) in enumerate(row["spans"]):
                phones=pronounce(word)
                if end>begin and phones!=[1]: durations.append(([1.,np.log(len(phones)),np.log(max(1,len(word)))],np.log(np.clip(end-begin,.03,2))))
                if i and 0<=begin-row["spans"][i-1][1]<.3: gaps.append(begin-row["spans"][i-1][1])
    x=np.asarray([a for a,_ in durations]); y=np.asarray([b for _,b in durations]); coefficients=np.linalg.solve(x.T@x+np.eye(3),x.T@y)
    write_json(root/"duration_model.json",{"coefficients":coefficients.tolist(),"interword_gap":float(np.clip(np.mean(gaps),.01,.12)),"training_word_spans":len(x),"annotation_revision":2,"phoneme_timing":"Uniform within official word intervals; ambiguous lexical records excluded from duration supervision"})
    write_json(root/"annotation_review.json",review); card=json.loads((root/"dataset_card.json").read_text()); card["annotation_revision"]=2; card["annotation_review_sha256"]=file_sha(root/"annotation_review.json"); card["duration_model_sha256"]=file_sha(root/"duration_model.json"); assert card["codec_sha256"]==json.loads((root/"dataset_card_initial.json").read_text())["codec_sha256"]; write_json(root/"dataset_card.json",card); write_json(marker,{"resolved":True,"review_sha256":file_sha(root/"annotation_review.json"),"dataset_card_sha256":file_sha(root/"dataset_card.json")})


if __name__=="__main__": main()
