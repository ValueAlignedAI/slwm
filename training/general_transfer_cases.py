"""Freeze metadata-only G1 evaluation cases before representation fitting."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from training.general_data import CACHE,ROOT,order
from training.general_sources import write
from training.research_sources import file_sha

ALIASES={"dog":["dog","dogs","bark","barks","barking"],"cat":["cat","cats","meow","meowing"],"bird":["bird","birds","chirp","chirping","birdsong"],"car":["car","cars","automobile","automobiles"],"motorcycle":["motorcycle","motorcycles","motorbike","motorbikes"],"train":["train","trains","railway"],"bus":["bus","buses"],"horse":["horse","horses","neigh","neighing"],"sheep":["sheep","lamb","lambs","bleat","bleating"],"cow":["cow","cows","cattle","moo","mooing"]}


def choose(key,values,n):return sorted(values,key=lambda x:order(key+":"+str(x)))[:n]


def main():
    target=CACHE/"evaluation_cases.json"
    if target.exists():raise RuntimeError("Case manifest is immutable; inspect existing registration")
    cfg=json.loads(Path("configs/t2/general_transfer.json").read_text()); hashes={};cases={"speech":[],"ami":[],"mhealth":[],"droid":[]}
    for family in ("speech","ami","mhealth"):
        path=CACHE/family/"rows.json";hashes[str(path)]=file_sha(path);rows=json.loads(path.read_text())
        if family=="speech":
            grouped=defaultdict(list)
            for r in rows:
                if r["role"]!="train" and 8<=r["duration"]<=18 and r["ctc_alignment_usable"]:grouped[(r["role"],r["actor"])].append(r)
            for (role,person),current in sorted(grouped.items()):
                ids=choose(person,[r["id"] for r in current],4 if role=="calibration" else 8)
                cases[family].extend({"id":key,"person":person,"role":role,"gap_seconds":.160} for key in ids)
        elif family=="ami":
            for r in rows:
                if r["role"]=="train":continue
                candidates=defaultdict(list);duration=r["duration"]
                for center in [i*4.096 for i in range(1,int(duration/4.096))]:
                    a,b=center-4.096,center+4.096
                    if a<0 or b>duration:continue
                    own=sum(max(0,min(y,b)-max(x,a)) for x,y,_ in r["spans"])/(b-a)
                    other=sum(max(0,min(y,b)-max(x,a)) for x,y in r["other_speaker_spans"])/(b-a)
                    category="own_nonoverlap" if own>=.25 and other<.1 else "own_overlap" if own>=.25 else "little_own_activity"
                    candidates[category].append(round(a,6))
                for category,starts in sorted(candidates.items()):
                    cases[family].extend({"id":r["id"],"person":r["actor"],"group":r["group"],"role":r["role"],"start_seconds":a,"duration_seconds":8.192,"stratum":category} for a in choose(r["id"]+category,starts,3))
        else:
            for r in rows:
                n=10000//3 if r["role"]=="train" else 3000//2 if r["role"]=="selection" else 2000
                starts=choose(r["id"],range(max(0,r["steps"]-cfg["context"]-cfg["horizon"]+1)),n)
                cases[family].append({"id":r["id"],"person":r["group"],"role":r["role"],"starts":starts})
    path=ROOT/cfg["robot_family"]/"raw_metadata/scene_role_plan.json";hashes[str(path)]=file_sha(path)
    for r in json.loads(path.read_text())["rows"]:
        starts=choose(r["id"],range(max(0,r["steps"]-cfg["context"]-cfg["horizon"]+1)),64)
        cases["droid"].append({"id":r["id"],"group":r["group"],"role":r["role"],"starts":starts,"goal_present":bool(r["captions"])})
    gold_path=CACHE/"coco_object_gold.json";gold=json.loads(gold_path.read_text());hashes[str(gold_path)]=file_sha(gold_path)
    semantic={"aliases":ALIASES,"consensus_captions":2,"coco":[],"clotho":[]}
    for family in ("coco","clotho"):
        path=CACHE/family/"rows.json";hashes[str(path)]=file_sha(path)
        for r in json.loads(path.read_text()):
            if family=="coco":labels=gold["labels"][r["id"].split("-",1)[1]]
            else:
                captions=[set(re.findall(r"[a-z]+",s.casefold())) for s in r["captions"]]
                labels=[label for label,names in ALIASES.items() if sum(bool(set(names)&words) for words in captions)>=2]
            if labels:semantic[family].append({"id":r["id"],"role":r["role"],"group":r["group"],"labels":labels})
    semantic["counts"]={family:{role:dict(Counter(label for r in semantic[family] if r["role"]==role for label in r["labels"])) for role in ("train","selection","calibration","test")} for family in ("coco","clotho")}
    counts={family:dict(Counter(r["role"] for r in rows)) for family,rows in cases.items()}
    write(target,{"before_any_g1_optimizer_update":True,"no_predictions_or_final_embeddings_computed":True,"source_sha256":file_sha(__file__),"source_hashes":hashes,"configuration_sha256":file_sha("configs/t2/general_transfer.json"),"evaluation_protocol_sha256":file_sha("docs/research/general_transfer_evaluation_20261106.md"),"cases":cases,"counts":counts,"semantic":semantic})
    print(json.dumps({"case_records":counts,"semantic_class_counts":semantic["counts"]}),flush=True)


if __name__=="__main__":main()
