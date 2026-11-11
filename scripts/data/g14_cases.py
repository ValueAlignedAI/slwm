"""Assign prospective real speech to independent station layouts without consulting any simulation outcome."""
import hashlib
import json
from pathlib import Path
import numpy as np
from data.g14_grid import labels,SLOTS
from g14_runtime import ROOT,json_file

def decode(pc,pl,pd,addresses):
    weights=np.asarray([pl[a]*pd[b] for a,b in addresses]);scaled=weights/max(1e-10,weights.sum())
    slot=int(weights.argmax());color=int(pc.argmax())
    confidence=float(pc[color]*(.7*scaled[slot]+.3*np.sqrt(weights[slot])))
    return color,slot,confidence

def layout(q,seed):
    rng=np.random.default_rng(seed);a,b=[tuple(x) for x in q["addresses"]]
    if a[0]==b[0] or a[1]==b[1]:raise ValueError("Noncrossable station labels")
    labels=[a,b,(a[0],b[1]),(b[0],a[1])]
    return [labels[i] for i in rng.permutation(4)]

def main(version):
    base=ROOT/"data/utterances";qs=json.loads((base/"quartets.json").read_bytes());meta=json.loads((base/"media.json").read_bytes())
    edge="cue" if version=="base" else json.loads((ROOT/"work/speech-choice.json").read_bytes())["directory"]
    z=np.load(ROOT/"work"/edge/"posterior.npz");dest=ROOT/"data"/("cases-base" if version=="base" else "cases");dest.mkdir(exist_ok=True)
    thresholds={};rows={}
    def probs(role):
        ids=z[role+"_indices"];return {int(i):j for j,i in enumerate(ids)},[z[f"{role}_normal_{k}"] for k in range(3)]
    ci,cp=probs("cal");confidence=[]
    for p,items in qs["cal"].items():
        for j,q in enumerate(items):
            addresses=layout(q,int(p)*100+j)
            for i in q["clips"]:
                a=ci[i];c,slot,conf=decode(*(s[a] for s in cp),addresses);confidence.append(conf)
    threshold=float(np.quantile(confidence,.025)) if confidence else 0.
    json_file(dest/"gate.json",dict(threshold=threshold,calibration_quartet_decodings=len(confidence)))
    for role,total,offset in (("dev",32,240000),("held",256,250000),("fresh",64,260000)):
        pi,pp=probs(role);people=sorted(int(k) for k,v in qs[role].items() if v)
        prepared=[]
        for i in range(total):
            seed=offset+i;shift=i>=total//2;person=people[i%len(people)]
            pool=qs[role][str(person)];q=pool[np.random.default_rng(seed+121).integers(len(pool))];addresses=layout(q,seed+1723)
            commands=[]
            for ident in q["clips"]:
                n=pi[ident];c,s,conf=decode(*(a[n] for a in pp),addresses);truth=labels(meta[ident]["text"])
                commands.append(dict(clip=int(ident),key=meta[ident]["key"],true_color=int(truth[0]),true_station=addresses.index(tuple(truth[1:])),
                                     heard_color=c,heard_station=s,confidence=conf,accept=bool(c<3 and conf>=threshold)))
            prepared.append(dict(seed=seed,shift=bool(shift),speaker=person,addresses=addresses,commands=commands))
        json_file(dest/(role+".json"),prepared)
        rows[role]=dict(scenes=len(prepared),speakers=people,strictly_new_people=role=="fresh",
           decoded_fraction=float(np.mean([x["heard_color"]==x["true_color"] and x["heard_station"]==x["true_station"] for r in prepared for x in r["commands"]])),
           accepted_fraction=float(np.mean([x["accept"] for r in prepared for x in r["commands"]])))
    json_file(dest/"design.json",rows);print("Spoken execution assignments",rows,flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("version",choices=("base","final"));main(p.parse_args().version)
