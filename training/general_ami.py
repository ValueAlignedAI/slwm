"""G1 paired speech expansion, grouped by connected participant identities."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from training.broad_data import decoded_timeline,pack
from training.general_data import ROOT,CACHE,order,rows_write
from training.general_sources import write
from training.research_sources import fetch,file_sha
from training.stream_codec import words

BASE=Path("artifacts/t2_audio_visual/ami_stream_v1")
PLAN=ROOT/"ami/plan.json"
# Selected from official identity metadata, before fitting or held-content scores.
# IS meetings sharing any participant remain in the same connected component.
MEETINGS={
 "train":["IS1000a","IS1003b","IS1004c","TS3003a","TS3003b","TS3004a","TS3004b","TS3010a","TS3010b","ES2015a","ES2015b"],
 "selection":["TS3005a","TS3011a"],"calibration":["TS3006a","IS1007a"],
 "test":["TS3007a","TS3008a","TS3009a","TS3012a","ES2016a","IS1009a"]}


def components(meetings):
    """meeting -> participant set gives transitive person-connected group IDs."""
    remaining=set(meetings); result={}
    while remaining:
        group={min(remaining)}; previous=-1
        while len(group)!=previous:
            previous=len(group); people=set.union(*(meetings[x] for x in group))
            group|={x for x in remaining if people&meetings[x]}
        remaining-=group; name=min(group)
        for meeting in group: result[meeting]=name
    return result


def plan():
    if PLAN.exists(): return json.loads(PLAN.read_text())
    meetings={m.attrib["observation"]:m for m in ET.parse(BASE/"annotations/corpusResources/meetings.xml").getroot()}
    identities={n:{s.attrib["global_name"] for s in m} for n,m in meetings.items()}; groups=components(identities)
    old=json.loads(Path("configs/t2/broad_stream_data.json").read_text())["ami_meetings"]
    prior=set.union(*(identities[n] for names in old.values() for n in names)); rows=[]; jobs=[]
    for role,names in MEETINGS.items():
        for name in names:
            assert not identities[name]&prior,(name,"Earlier person would invalidate fresh-speaker scope")
            for speaker in meetings[name]:
                s=speaker.attrib; assert "camera" in s,(name,"No participant camera in source metadata"); spans=[]
                path=BASE/"annotations/words"/(name+"."+s["nxt_agent"]+".words.xml")
                for word in ET.parse(path).getroot():
                    a=word.attrib
                    if word.tag!="w" or a.get("punc")=="true" or not words(word.text or "") or "starttime" not in a: continue
                    start,end=float(a["starttime"]),float(a["endtime"])
                    if end>start and start>=0 and end<=900: spans.append((start,end,(word.text or "").lower()))
                audio=name+".Headset-"+s["channel"]+".wav"; video=name+"."+s["camera"]+".avi"
                rows.append({"id":name+"-"+s["nxt_agent"],"actor":"AMI-"+s["global_name"],"group":"AMI-COMPONENT-"+groups[name],"role":role,"meeting":name,"source":"ami","raw_path":str(ROOT/"ami/sources"/name/audio),"video_path":str(ROOT/"ami/sources"/name/video),"requested_duration":900,"spans":spans,"transcript":" ".join(w for _,_,w in spans),"alignment_source_sha256":file_sha(path)})
                for family,filename in (("audio",audio),("video",video)):
                    jobs.append({"meeting":name,"file":filename,"url":"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"+name+"/"+family+"/"+filename})
    for one in MEETINGS:
        for two in MEETINGS:
            if one<two:
                assert not {r["actor"] for r in rows if r["role"]==one}&{r["actor"] for r in rows if r["role"]==two}
                assert not {r["group"] for r in rows if r["role"]==one}&{r["group"] for r in rows if r["role"]==two}
    record={"rows":rows,"jobs":jobs,"source_sha256":file_sha(__file__),"component_members":{g:sorted(n for n in groups if groups[n]==g) for g in sorted({groups[r["meeting"]] for r in rows})},
            "old_people_excluded":sorted(prior),"max_duration_seconds_per_person_track":900,"new_selection_people":len({r["actor"] for r in rows if r["role"]=="selection"}),"new_final_people":len({r["actor"] for r in rows if r["role"]=="test"}),
            "policy":"Fail unavailable source jobs explicitly. No substitution based on scored examples; any availability revision is written before fitting. Quiet/overlap tracks remain with stratified evaluation. No new participant appears in any earlier role."}
    write(ROOT/"ami/availability_revision.json",{"before_model_fitting":True,"initial_plan_failure":"EN person records do not supply camera identifiers; the metadata-only plan failed before writing or downloading", "revision":"Use camera-complete IS/TS/ES components only; preserve group roles and24 fresh final people. No content score informed this revision."})
    write(PLAN,record); return record


def acquire():
    record=plan(); results=[]; errors=[]
    def one(job):
        if shutil.disk_usage(ROOT).free<15*1024**3: raise OSError("Archive task-owned verified transport copies before downloading further AMI sources")
        return {**job,"source":fetch(job["url"],ROOT/"ami/sources"/job["meeting"]/job["file"])}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(one,j):j for j in record["jobs"]}
        for future in as_completed(jobs):
            job=jobs[future]
            try: results.append(future.result())
            except Exception as e: errors.append({"job":job,"type":type(e).__name__,"error":str(e)})
            write(ROOT/"ami/acquisition_progress.json",{"verified":results,"errors":errors,"requested":len(record["jobs"])})
            print(json.dumps({"ami_sources":len(results),"failed":len(errors),"requested":len(record["jobs"])}),flush=True)
    if errors: raise RuntimeError("AMI acquisition has explicit failures; revise availability before fitting")


def prepare():
    import soundfile as sf
    import torch
    torch.set_num_threads(1); record=plan(); folder=CACHE/"ami"
    if (folder/"manifest.json").exists(): return
    rows=[]
    for role in MEETINGS:
        current=[]
        for row in (r.copy() for r in record["rows"] if r["role"]==role):
            timeline=decoded_timeline(Path(row["video_path"])); duration=min(900.,sf.info(row["raw_path"]).duration,timeline["decoded_end_seconds"])
            row["duration"]=duration; row["spans"]=[x for x in row["spans"] if x[1]<=duration]; row["transcript"]=" ".join(w for _,_,w in row["spans"]); row["decoded_timeline"]=timeline
            # A person's annotations identify the owner, not every audible voice.
            # All other participant word intervals are kept for overlap strata.
            row["other_speaker_spans"]=[(a,b) for other in record["rows"] if other["meeting"]==row["meeting"] and other["actor"]!=row["actor"] for a,b,_ in other["spans"]]
            current.append(row)
        dest=folder/role
        current=pack(current,dest,video=True)
        for row in current:
            row["mel_file"]=str(dest/"mel.npy"); row["pixel_file"]=str(dest/(row["id"]+"_pixels.npy")); rows.append(row)
    rows_write(folder,rows,person_hours=sum(r["duration"] for r in rows)/3600,
        independent_meeting_hours=sum(min(r["duration"] for r in rows if r["meeting"]==m) for m in {r["meeting"] for r in rows})/3600,
        people={role:len({r["actor"] for r in rows if r["role"]==role}) for role in MEETINGS},
        meetings={role:sorted({r["meeting"] for r in rows if r["role"]==role}) for role in MEETINGS},
        annotations="Actual word intervals; no CMU pseudo-phone labels are used for G1 loss. Headset sound may include other participants.",
        visual_edge="Classical causal face crop64x64 gray, learned CNN in G1. No face PCA or supplied appearance code.",
        plan_sha256=file_sha(PLAN),files={str(p.relative_to(folder)):file_sha(p) for p in folder.rglob("*.npy")})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("plan","download","prepare"),required=True); args=parser.parse_args()
    if args.stage=="plan":
        value=plan(); print(json.dumps({"new_person_tracks":len(value["rows"]),"new_meetings":len(value["jobs"])/8,"fresh_final_people":value["new_final_people"]}),flush=True)
    elif args.stage=="download": acquire()
    else: prepare()


if __name__=="__main__": main()
