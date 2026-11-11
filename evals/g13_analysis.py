"""Statistical and matched-control readout for real physical endpoints, object binding and offline decisions."""
import json
from pathlib import Path
import numpy as np
from evals.g13_predict import bank,records,study,arrays
from evals.g13_stats import Groups
from scripts.train.g13_optim import ROOT
from scripts.train.g9_storage import write
from training.g13_physics import group_mean
from training.g13_adaptive import role_prior,apply_prior

def classification(p,y):
    ok=y>=0;pred=p.argmax(1);val=p[np.arange(len(y)),np.maximum(y,0)]
    return dict(accuracy=np.where(ok,pred==y,np.nan),nll=np.where(ok,-np.log(np.maximum(val,1e-7)),np.nan))

def raw_x(r):
    obj=r["geo"].astype("f4");pixel=r["px"][:,-1].reshape(-1,10,16,3).astype("f4")
    image=pixel.reshape(-1,2,5,4,4,3).mean((2,4)).reshape(len(pixel),-1)/255
    return np.concatenate((image,r["goal"].astype("f4")/127,r["bind"].reshape(len(pixel),-1),r["kin"].reshape(len(pixel),-1),
                           obj[:,-1].reshape(len(pixel),-1),(obj[:,-1]-obj[:,0]).reshape(len(pixel),-1)),1)

def raw_fit(r,dev):
    x=raw_x(r);select=r["mover"]>=0;g=r["group"][select];x=x[select].astype("f8");_,inv,n=np.unique(g,return_inverse=True,return_counts=True)
    w=1./n[inv];w/=w.sum();mu=(x*w[:,None]).sum(0);sd=np.maximum(np.sqrt(((x-mu)**2*w[:,None]).sum(0)),.05)
    a=(x-mu)/sd;y=np.eye(5)[r["mover"][select]];ym=(y*w[:,None]).sum(0);mat=a.T@(a*w[:,None]);xy=a.T@((y-ym)*w[:,None])
    target=dev["mover"];dd=(raw_x(dev)-mu)/sd;choices=[]
    for reg in (.04,.2,1.):
        beta=np.linalg.solve(mat+reg*np.eye(len(mu)),xy)
        for t in (1.,3.,6.,12.):
            z=(dd@beta+ym)*t;p=softmax(z);score=group_mean(classification(p,target)["nll"],dev["group"])
            choices.append((score,reg,t,beta))
    _,reg,t,beta=min(choices,key=lambda z:z[:3]);return mu,sd,beta,ym,t

def softmax(z):
    p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)

def suite(r,tr,raw,kind):
    target=r["future"][:,-1];sig=r["plan"].sum(1);m=r["mover"];pr=role_prior(tr)
    y=tr["mover"];ix=np.flatnonzero(y>=0);_,inv,n=np.unique(tr["group"][ix],return_inverse=True,return_counts=True)
    counts=np.bincount(y[ix],weights=1./n[inv],minlength=5)+.01;prior=counts/counts.sum()
    pword=np.zeros((len(m),5),np.float32)+.01;has=r["bind"][:,:,0].sum(1)>0
    chosen=np.where(has,r["bind"][:,:,0].argmax(1)+1,int(prior.argmax()));pword[np.arange(len(m)),chosen]=.96
    mu,sd,beta,ym,t=raw;rr=softmax((((raw_x(r)-mu)/sd)@beta+ym)*t)
    probs=dict(frequency=np.broadcast_to(prior,(len(m),5)),source_word=pword,role_prior=apply_prior(r["bind"],pr),raw=rr)
    metrics={k:classification(v,m) for k,v in probs.items()}
    endpoints=dict(persistence=np.zeros_like(target),velocity=4*r["kin"][:,-1,2:4],integration=sig,
                   linear=r["ref_active"][:,-1],passive_linear=r["ref_passive"][:,-1])
    if kind=="repaired":endpoints["adaptive"]=r["adapt_active"][:,-1]
    for k,v in endpoints.items():metrics[k]=dict(active=np.linalg.norm(v-target,axis=1)*1000)
    return metrics,endpoints

def options(r):
    groups=r["group"];pool=np.flatnonzero(r["first"]);rng=np.random.default_rng(130712)
    ps=r["kin"][:,-1,:2];velocity=r["kin"][:,-1,2:4];remaining=set(groups[pool].tolist());sets=[]
    for g in rng.permutation(sorted(remaining)):
        if g not in remaining:continue
        possible=pool[groups[pool]==g];anchor=int(possible[rng.integers(len(possible))]);selected=[anchor];gs={g}
        d=np.linalg.norm(ps[pool]-ps[anchor],axis=1);dv=np.linalg.norm(velocity[pool]-velocity[anchor],axis=1)
        eligible=np.flatnonzero((d<.04)&(dv<.045));order=eligible[np.argsort((d+.3*dv)[eligible])]
        for j in order:
            i=int(pool[j]);new=int(groups[i])
            if new in gs or new not in remaining:continue
            selected.append(i);gs.add(new)
            if len(selected)==4:break
        if len(selected)==4:sets.append(selected);remaining-=gs
    return np.asarray(sets,np.int32)

def choice(metric,truth,sets):
    if not len(sets):return None
    targets=.055*np.array([[-1,0],[1,0],[0,-1],[0,1]])
    pred=np.linalg.norm(metric[sets][:,None,:,:]-targets[None,:,None,:],axis=-1)
    err=np.linalg.norm(truth[sets][:,None,:,:]-targets[None,:,None,:],axis=-1)
    selected=pred.argmin(2);real=np.take_along_axis(err,selected[:,:,None],2)[:,:,0]
    return dict(regret_mm=(real-err.min(2)).mean(1)*1000,optimal=(selected==err.argmin(2)).mean(1))

def summarize(bs,m):return {k:bs.score(v) for k,v in m.items()}

def gains(bs,a,b,keys=None):
    return {k:bs.score(a[k]-b[k] if k=="accuracy" or k=="optimal" else b[k]-a[k]) for k in (keys or a.keys()) if k in b}

def methods(out,r,mode):
    p=out[mode+"_mover"];d=classification(p,r["mover"]);target=r["future"][:,-1]
    d.update(active=np.linalg.norm(out[mode+"_active"]-target,axis=1)*1000,
             calibrated=np.linalg.norm(out[mode+"_cal"]-target,axis=1)*1000,
             passive=np.linalg.norm(out[mode+"_passive"]-target,axis=1)*1000,
             action=np.linalg.norm(out[mode+"_cmd"]-r["plan"].sum(1),axis=1)*1000)
    return d

def restricted(m,mask):return {k:np.where(mask,v,np.nan) for k,v in m.items()}

def main():
    work=ROOT/"work/score";out=ROOT/"release/metrics";locks=json.loads((work/"model-lock.json").read_bytes())
    paper=dict(runs={},controls={},trials={},population={},comparison={});group_data={};sample={}
    human=json.loads((out/"measurement.json").read_bytes())
    for kind in ("original","repaired"):
        tr=bank(kind,"train");dev=bank(kind,"dev");raw=raw_fit(tr,dev);docs=records(kind)
        for cohort in ("hold","fresh"):
            r=bank(kind,"test",cohort=="fresh");groups=r["group"];bs=Groups(groups);y=r["mover"];target=r["future"][:,-1]
            controls,ends=suite(r,tr,raw,kind);ids=np.unique(groups);idx={a:i for i,a in enumerate(ids)}
            sets=options(r);decision=Groups(np.arange(len(sets)));prefix=kind+"_"+cohort;group_data[prefix+"_groups"]=ids
            paper["controls"][prefix]={k:summarize(bs,v) for k,v in controls.items()}
            paper["trials"][prefix]={k:summarize(decision,choice(v,target,sets)) for k,v in ends.items()}
            paper["population"][prefix]=dict(windows=len(y),episodes=int(r["first"].sum()),groups=len(ids),novel_episodes=int((r["first"]&r["novel"]).sum()),
                measured=int((y>=0).sum()),novel_measured=int(((y>=0)&r["novel"]).sum()),paired_measured=int((r["paired"]&(y>=0)).sum()),
                plans=len(sets),options_per_decision=4,sets=sets.tolist())
            def group_record(key,values):
                for k,a in values.items():
                    mask=np.isfinite(a);gi=np.searchsorted(ids,groups[mask]);num=np.bincount(gi,minlength=len(ids))
                    total=np.bincount(gi,weights=a[mask],minlength=len(ids))
                    group_data[prefix+"_"+key+"_"+k]=np.divide(total,num,out=np.full(len(num),np.nan),where=num>0).astype("f4")
            for name,row in controls.items():group_record(name,row)
            if cohort=="fresh":judged=[v for v in human["fresh"] if v["panel"]=="uniform" and v["final"]!="U"]
            else:judged=[v for v in human["first"] if v["role"]=="test" and v["final"]!="U"]
            by_id={int(r["episode"][i]):i for i in np.flatnonzero(r["first"])};audit_ix=np.array([by_id[v["episode"]] for v in judged],int)
            audit_y=np.asarray(["0RBGY".index(v["final"]) for v in judged]);hb=Groups(groups[audit_ix])
            src=work/kind/cohort;cache={};judgments={}
            for doc in docs:
                tag=doc["tag"];entry=locks[kind+"/"+tag];k0=kind+"/"+tag
                rr=paper["runs"].setdefault(k0,dict(core=doc["core"],recipe=doc["recipe"],seed=doc["seed"],selected=entry["step"],alpha=entry["alpha"],cohorts={}))
                with np.load(src/(tag+".npz")) as file:
                    data=methods(file,r,"correct");cache[tag]=data
                    group_record(tag,data)
                    dd=dict(absolute=summarize(bs,data),novel=summarize(bs,restricted(data,r["novel"])),
                            familiar=summarize(bs,restricted(data,~r["novel"])))
                    alt=methods(file,r,"wrong");none=methods(file,r,"none")
                    keyset=("accuracy","nll","passive","action");msk=r["paired"]&r["first"]
                    dd["speech_text_wrong"]=gains(bs,restricted(data,msk),restricted(alt,msk),keyset)
                    dd["speech_text_none"]=gains(bs,restricted(data,msk),restricted(none,msk),keyset)
                    dd["vs_role"]=gains(bs,data,controls["role_prior"],("accuracy","nll"));dd["vs_raw"]=gains(bs,data,controls["raw"],("accuracy","nll"))
                    dd["vs_source_word"]=gains(bs,data,controls["source_word"],("accuracy",))
                    dd["vs_linear_active"]=bs.score(controls["linear"]["active"]-data["active"])
                    if kind=="repaired":dd["vs_adaptive_active"]=bs.score(controls["adaptive"]["active"]-data["active"])
                    dd["vs_passive_linear"]=bs.score(controls["passive_linear"]["active"]-data["passive"])
                    for mode in ("no_vision","text_only","reset","wrong_plan","zero_plan"):
                        if mode+"_mover" in file:dd[mode]=gains(bs,data,methods(file,r,mode))
                    hits=(file["correct_mover"][audit_ix].argmax(1)==audit_y).astype(float);judgments[tag]=hits
                    dd["visual_review_accuracy"]=hb.score(hits)
                    pp=file["correct_mover"];conf=file["correct_confidence"];gate=.5*(pp.max(1)+conf[:,0])>=entry["thresholds"][0]
                    gate2=conf[:,1]>=entry["thresholds"][1];dd["routing"]=dict(object_coverage=bs.score(np.where(r["first"],gate,np.nan)),
                         conditional_object=bs.score(np.where(gate,data["accuracy"],np.nan)),active_coverage=bs.score(gate2),
                         conditional_active_mm=bs.score(np.where(gate2,data["active"],np.nan)),internal_only=bs.score(~(gate|gate2)))
                    dd["plan_selection"]=summarize(decision,choice(file["correct_cal"],target,sets))
                    if doc["seed"]==1301 and doc["recipe"] in ("residual","anchored"):
                        key=prefix+"_"+tag;sample[key+"_mover"]=pp.astype("f2");sample[key+"_endpoint"]=file["correct_active"].astype("f4")
                rr["cohorts"][cohort]=dd
            # All comparisons below use exactly the same scoring target and the same resampled sources.
            for doc in docs:
                tag=doc["tag"];c=doc["core"];rc=doc["recipe"];seed=doc["seed"];base=f"gpt2-{rc}-{seed}"
                rival=f"{c}-"+("flat" if kind=="original" else "linear")+f"-{seed}"
                info=paper["runs"][kind+"/"+tag]["cohorts"][cohort]
                for label,k in (("vs_dense",base),("vs_process_base",rival)):
                    info[label]=gains(bs,cache[tag],cache[k]);info[label+"_novel"]=gains(bs,restricted(cache[tag],r["novel"]),restricted(cache[k],r["novel"]))
                    info[label+"_review"]=hb.score(judgments[tag]-judgments[k])
                if kind=="repaired" and rc=="anchored":
                    other=f"{c}-adaptive-{seed}";info["vs_unanchored"]=gains(bs,cache[tag],cache[other])
                    oldpath=work/"original"/cohort/f"{c}-residual-{seed}.npz"
                    with np.load(oldpath) as old:
                        old_hits=(old["correct_mover"][audit_ix].argmax(1)==audit_y).astype(float)
                        info["review_change_since_original"]=hb.score(judgments[tag]-old_hits)
            sample[prefix+"_row_group"]=groups;sample[prefix+"_mover"]=y;sample[prefix+"_row_episode"]=r["episode"];sample[prefix+"_future"]=target
            print("Analyzed matched scores",kind,cohort,len(cache),flush=True)
    np.savez_compressed(out/"robot-recording-scores.npz",**group_data);np.savez_compressed(out/"robot-inference-traces.npz",**sample)
    write(out/"robot-results.json",paper);(work/"analyzed").touch()

if __name__=="__main__":main()
