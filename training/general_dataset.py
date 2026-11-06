"""Training-only normalizers and immutable G1 cache/source contract."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import numpy as np
import tiktoken
from training.general_batch import rotation6d,ROBOT_STATE,ROBOT_ACTION
from training.general_data import CACHE,ROOT
from training.general_sources import write
from training.research_sources import file_sha


class Moments:
    def __init__(self,width):self.n=np.zeros(width,np.float64);self.first=self.n.copy();self.second=self.n.copy()
    def add(self,values):
        for begin in range(0,len(values),65536):
            x=np.asarray(values[begin:begin+65536],np.float64);valid=np.isfinite(x);x=np.where(valid,x,0);self.n+=valid.sum(0);self.first+=x.sum(0);self.second+=(x*x).sum(0)
    def finish(self):
        mean=self.first/np.maximum(1,self.n);scale=np.sqrt(np.maximum(self.second/np.maximum(1,self.n)-mean*mean,.0001));return mean.astype(np.float32),scale.astype(np.float32)


def robot_plan(cfg):
    path=ROOT/cfg["robot_family"]/"raw_metadata/scene_role_plan.json";plan=json.loads(path.read_text());assert plan["suitable_for_building_holdout"];return path,plan


def normalizers(cfg):
    path=CACHE/"normalizers.npz"
    if path.exists():return
    audio=Moments(100);numeric=Moments(64);actions=Moments(7)
    # Every provided row is role-filtered before any moment is accumulated.
    for family in ("speech","ami","clotho"):
        rows=[r for r in json.loads((CACHE/family/"rows.json").read_text()) if r["role"]=="train"]
        for row in rows:
            p=Path(row["mel_file"]) if "mel_file" in row else CACHE/family/row["file"];mel=np.load(p,mmap_mode="r")
            a=row.get("offset",0);n=row.get("length",row.get("steps"));audio.add(mel[a:a+n].reshape(-1,100))
    legacy=Path("artifacts/t2_audio_visual/broad_stream_prepared_v1/ami_train/mel.npy");audio.add(np.load(legacy,mmap_mode="r").reshape(-1,100))
    for row in json.loads((CACHE/"pamap2/rows.json").read_text()):
        if row["role"]!="train":continue
        raw=np.load(CACHE/"pamap2"/row["file"],mmap_mode="r")
        for begin in range(0,len(raw),65536):
            values=np.full((min(65536,len(raw)-begin),64),np.nan,np.float32);values[:,:33]=raw[begin:begin+65536];numeric.add(values)
    _,plan=robot_plan(cfg)
    for row in plan["rows"]:
        if row["role"]!="train":continue
        data=np.load(CACHE/cfg["robot_family"]/row["id"]/"trajectory.npz");state=rotation6d(data["observed"],data["gripper"]);values=np.full((len(state),64),np.nan,np.float32);values[:,ROBOT_STATE]=state;numeric.add(values);actions.add(data["action"])
    am,asc=audio.finish();nm,ns=numeric.finish();cm,cs=actions.finish();nm[ROBOT_ACTION]=cm;ns[ROBOT_ACTION]=cs
    np.savez(path,audio_mean=am,audio_scale=asc,numeric_mean=nm,numeric_scale=ns,action_mean=cm,action_scale=cs)
    write(CACHE/"normalizers.json",{"sha256":file_sha(path),"training_only":True,"audio_frames_per_band":audio.n.tolist(),"numeric_samples_per_channel":numeric.n.tolist(),"action_samples":actions.n.tolist(),"unused_numeric_channels":np.flatnonzero(numeric.n==0).tolist(),"source_sha256":file_sha(__file__),"no_mhealth_statistics_used":True})


def freeze(cfg):
    path=CACHE/"dataset_contract.json"
    if path.exists():
        contract=json.loads(path.read_text());assert contract["configuration_sha256"]==file_sha("configs/t2/general_transfer.json")
        return contract
    normalizers(cfg); roles={}; hashes={};people={}
    for family in ("coco","clotho","speech","ami","pamap2","mhealth"):
        folder=CACHE/family;card=json.loads((folder/"manifest.json").read_text());rows=json.loads((folder/"rows.json").read_text());assert card["rows_sha256"]==file_sha(folder/"rows.json")
        role_groups={role:{r["group"] for r in rows if r["role"]==role} for role in ("train","selection","calibration","test")}
        for one in role_groups:
            for two in role_groups:
                if one<two:assert not role_groups[one]&role_groups[two],(family,one,two)
        roles[family]={"records":dict(Counter(r["role"] for r in rows)),"groups":{k:len(v) for k,v in role_groups.items()},"card_sha256":file_sha(folder/"manifest.json")}
        if family in ("speech","ami","pamap2","mhealth"):
            minimum=cfg["context"]+cfg["horizon"]+(cfg["future_audio_guard_packets"] if family in ("speech","ami") else 0)
            roles[family]["minimum_joint_window_packets"]=minimum
            roles[family]["too_short_for_joint_future_windows"]=dict(Counter(r["role"] for r in rows if r.get("length",r.get("steps",0))<minimum))
        if family in ("speech","ami"):
            for role in role_groups:people.setdefault(role,set()).update(r["actor"] for r in rows if r["role"]==role)
            for row in rows:
                p=Path(row["mel_file"])
                if str(p) not in hashes:hashes[str(p)]=file_sha(p)
                if family=="speech" and str(row["raw_path"]).startswith(str(ROOT)):
                    p=Path(row["raw_path"]);assert file_sha(p)==row["raw_sha256"];hashes[str(p)]=row["raw_sha256"]
        # Match each cache to its already registered preparation hash once.
        files=card.get("files",{})
        for name,digest in files.items():
            p=folder/name;assert file_sha(p)==digest,str(p);hashes[str(p)]=digest
        for name in ("rows.json","manifest.json"):hashes[str(folder/name)]=file_sha(folder/name)
        if family=="clotho":
            for row in rows:
                p=folder/row["file"];assert file_sha(p)==row["mel_sha256"];hashes[str(p)]=row["mel_sha256"]
    old=Path("artifacts/t2_audio_visual/broad_stream_prepared_v1/ami_train");legacy=json.loads((old/"rows.json").read_text());people["train"].update(r["actor"] for r in legacy)
    for role in ("selection","calibration","test"):assert not people["train"]&people[role]
    for p in old.glob("*.npy"):hashes[str(p)]=file_sha(p)
    hashes[str(old/"rows.json")]=file_sha(old/"rows.json")
    meetings={r["id"].rsplit("-",1)[0] for r in legacy}
    roles["ami"]["legacy_training_addition"]={"records":len(legacy),"people":len({r["actor"] for r in legacy}),"meetings":len(meetings),"person_hours":sum(r["duration"] for r in legacy)/3600,"independent_meeting_hours":sum(min(r["duration"] for r in legacy if r["id"].rsplit("-",1)[0]==m) for m in meetings)/3600}
    robot_path,robot=robot_plan(cfg);roles["droid"]={"records":dict(Counter(r["role"] for r in robot["rows"])),"groups":robot["group_counts"],"excluded_unverified":len(robot["excluded_unverified_ids"]),"source":cfg["robot_family"]}
    hashes[str(robot_path)]=file_sha(robot_path)
    roles["droid"]["missing_instructions_by_role"]=dict(Counter(r["role"] for r in robot["rows"] if not r["captions"]))
    for row in robot["rows"]:
        for name,digest in row["files"].items():
            p=CACHE/cfg["robot_family"]/row["id"]/name;assert file_sha(p)==digest;hashes[str(p)]=digest
    hashes[str(CACHE/"normalizers.npz")]=file_sha(CACHE/"normalizers.npz")
    cases=CACHE/"evaluation_cases.json";assert cases.exists(),"Freeze metadata-only transfer cases before fitting"
    hashes[str(cases)]=file_sha(cases)
    enc=tiktoken.get_encoding("gpt2");rank_hash=hashlib.sha256(b"".join(token+b":"+str(rank).encode()+b"\n" for token,rank in sorted(enc._mergeable_ranks.items(),key=lambda item:item[1]))).hexdigest()
    expected=json.loads(Path("artifacts/text/fineweb_edu_scale_v1/dataset_card.json").read_text())["tokenizer"]
    assert rank_hash==expected["merge_ranks_sha256"] and tiktoken.__version__==cfg["tokenizer"]["version"]
    write(CACHE/"cache_hashes.json",hashes)
    contract={"roles":roles,"source_sha256":file_sha(__file__),"cache_hashes_sha256":file_sha(CACHE/"cache_hashes.json"),"normalizers_sha256":file_sha(CACHE/"normalizers.npz"),"configuration_sha256":file_sha("configs/t2/general_transfer.json"),"tokenizer":expected,"source_archives_may_be_on_mac":"See source_archival.json; full source bytes were verified before redundant Fedora transport removal", "no_core_pretraining_mhealth":True,"window_sampling":"Uniform connected group, eligible-window-weighted recording; no activity labels in shared pretraining","new_speech_people":json.loads((CACHE/"speech/plan.json").read_text())["fresh_people"],"train_people_including_legacy":len(people["train"]),"frozen_before_shared_core_fit":True}
    contract["evaluation_cases_sha256"]=file_sha(cases)
    write(path,contract);print(json.dumps({"general_dataset_frozen":roles}),flush=True);return contract


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("normalizers","freeze"),required=True);args=parser.parse_args();cfg=json.loads(Path("configs/t2/general_transfer.json").read_text())
    if args.stage=="normalizers":normalizers(cfg)
    else:freeze(cfg)


if __name__=="__main__":main()
