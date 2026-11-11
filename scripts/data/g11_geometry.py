"""Measure hand-excluded, world-registered local surface change from public HoloAssist AHAT depth."""
import argparse
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import threading
import time
import numpy as np
from PIL import Image
import requests
from scipy.spatial import cKDTree
from g10_holo_index import BASE,fetch
from g11_index import PHYSICAL

GRID=10
RADIUS=.10
THRESH=.025
LOCAL=threading.local()
BONES=[(0,1),(0,2),(2,3),(3,4),(4,5)]+[(a,b) for start in (6,11,16,21) for a,b in [(0,start),*[(i,i+1) for i in range(start,start+4)]]]
CV_TO_PSI=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])


def save(path,doc):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(doc,allow_nan=False,separators=(",",":")));tmp.replace(path)


def session():
    if not hasattr(LOCAL,"http"):LOCAL.http=requests.Session()
    return LOCAL.http


def range_bytes(key,off,size):
    return fetch(session(),BASE+key+".tar",int(off),int(off+size),PHYSICAL[key][0])


def member(index,name,suffix):
    key=name+"/Export_py/"+suffix
    if key not in index:raise KeyError(key)
    return index[key]


class DepthFrames:
    """Interpolation search over checksum-valid, chronologically named inner tar members."""
    def __init__(self,entry,directory):
        self.off,self.size=entry;self.directory=directory;directory.mkdir(exist_ok=True,parents=True)
        self.headers={};self.cache={}

    def probe(self,pos):
        pos=max(0,min((int(pos)//512)*512,self.size-512))
        amount=min(256<<10,self.size-pos)
        raw=range_bytes("ahat_depth",self.off+pos,amount)
        found=[]
        at=raw.find(b"ustar")
        while at>=0:
            hpos=at-257
            if hpos>=0 and hpos%512==0 and hpos+512<=len(raw):
                try:h=tarfile.TarInfo.frombuf(raw[hpos:hpos+512],"utf8","strict")
                except (ValueError,tarfile.TarError):h=None
                if h is not None and h.type==tarfile.XHDTYPE and hpos+512+h.size<=len(raw):
                    payload=raw[hpos+512:hpos+512+h.size]
                    name=None
                    for line in payload.splitlines():
                        _,_,field=line.partition(b" ")
                        if field.startswith(b"path="):name=field[5:].decode()
                    if name:
                        pos2=hpos+512+((h.size+511)//512)*512
                        if pos2+512<=len(raw):
                            try:real=tarfile.TarInfo.frombuf(raw[pos2:pos2+512],"utf8","strict")
                            except (ValueError,tarfile.TarError):real=None
                            if real is not None and real.isfile():
                                real.name=name;h=real;hpos=pos2
                if h is not None and h.isfile() and h.name.endswith(".png") and Path(h.name).stem.isdigit():
                    ident=int(Path(h.name).stem);body=pos+hpos+512
                    self.headers[ident]=(body,h.size);found.append(ident)
                    if hpos+512+h.size<=len(raw):
                        self.cache[ident]=raw[hpos+512:hpos+512+h.size]
            at=raw.find(b"ustar",at+5)
        return found

    def get(self,ident,max_id):
        ident=int(ident);file=self.directory/f"{ident:06d}.png"
        if file.exists():return np.asarray(Image.open(file),np.uint16)
        lo_id,lo=-1,0;hi_id,hi=max(max_id+1,ident+1),self.size
        for attempt in range(24):
            if ident in self.headers:break
            lower=[k for k in self.headers if k<ident];higher=[k for k in self.headers if k>ident]
            if lower:
                k=max(lower);o,n=self.headers[k];lo_id=k;lo=o+((n+511)//512)*512
            if higher:
                k=min(higher);o,_=self.headers[k];hi_id=k;hi=max(lo+512,o-512)
            if hi<=lo:raise ValueError("Nonchronological AHAT member index")
            # Target a region just ahead of its estimated header and enumerate every complete neighboring header.
            guess=int(lo+(hi-lo)*(ident-lo_id)/max(1,hi_id-lo_id))-(80<<10)
            if hi-lo<384<<10:guess=lo
            guess=max(lo,min(guess,hi-512))
            known=len(self.headers);self.probe(guess)
            if len(self.headers)==known and ident not in self.headers:
                self.probe(lo if attempt%2==0 else max(lo,guess-(160<<10)))
                if hi-lo<384<<10:self.probe(max(lo,hi-(220<<10)))
        if ident not in self.headers:raise ValueError("AHAT frame not resolved")
        pos,n=self.headers[ident];data=self.cache.get(ident)
        if data is None:data=range_bytes("ahat_depth",self.off+pos,n)
        # Decode before retaining bytes. Pillow verifies the PNG container.
        image=np.asarray(Image.open(io.BytesIO(data)),np.uint16)
        if image.shape!=(512,512):raise ValueError("Unexpected depth grid")
        file.write_bytes(data)
        if len(self.cache)>64:self.cache.clear()
        return image


class SparseHands:
    def __init__(self,entry,duration,start_abs):
        self.off,self.size=entry;self.duration=duration;self.start_abs=start_abs;self.known={}

    def rows(self,pos):
        pos=max(0,min(int(pos),max(0,self.size-(80<<10))))
        raw=range_bytes("hands",self.off+pos,min(80<<10,self.size-pos))
        begin=0 if pos==0 else raw.find(b"\n")+1
        for line in raw[begin:].splitlines(keepends=True):
            if not line.endswith(b"\n"):continue
            parts=line.split(b"\t",2)
            if len(parts)==3:
                t=(int(parts[1])-self.start_abs)/1e7;self.known[t]=(pos+begin,line)
            begin+=len(line)

    def at(self,t):
        for attempt in range(13):
            times=sorted(self.known);idx=bisect_left(times,t)
            if times:
                nearest=min(times[max(0,idx-1):idx+1] or times[-1:],key=lambda v:abs(t-v))
                if nearest>t and idx>0:nearest=times[idx-1]
                if 0<=t-nearest<=.12:
                    row=np.fromstring(self.known[nearest][1].decode(),sep="\t")
                    if len(row) not in (471,473):raise ValueError("Unknown hand schema")
                    if row[2]<.5:return np.empty((0,3)),np.empty((0,3))
                    mats=row[-468:-52].reshape(26,4,4)
                    valid=row[-52:-26]>.5
                    pts=mats[:,:3,3];usable=valid&np.isfinite(pts).all(1)
                    nodes=list(pts[usable])
                    for a,b in BONES:
                        if usable[a] and usable[b]:nodes.append((pts[a]+pts[b])/2)
                    tips=np.array([5,10,15,20,25]);return np.asarray(nodes).reshape(-1,3),pts[tips[usable[tips]]]
            left_t=times[idx-1] if idx>0 else 0.;right_t=times[idx] if idx<len(times) else self.duration
            left=self.known[left_t][0] if left_t in self.known else 0
            right=self.known[right_t][0] if right_t in self.known else self.size
            guess=left+(right-left)*np.clip((t-left_t)/max(.001,right_t-left_t),0,1)
            self.rows(guess-(32<<10) if attempt<9 else guess-(15<<10))
        raise ValueError("No contemporaneous hand sample")


def make_rays(intr):
    k=intr[:9].reshape(3,3);yy,xx=np.mgrid[0:512:2,0:512:2]
    xd=(xx-k[0,2])/k[0,0];yd=(yy-k[1,2])/k[1,1]
    x=xd.copy();y=yd.copy();a,b,c=intr[9:12]
    for _ in range(8):
        r=x*x+y*y;scale=np.maximum(.25,1+a*r+b*r*r+c*r*r*r);x=xd/scale;y=yd/scale
    rays=np.stack([x,y,np.ones_like(x)],-1);rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    return (rays@CV_TO_PSI.T).astype(np.float32)


def cloud(depth,pose,rays,hands):
    d=depth[::2,::2].astype(np.float32)*.001
    valid=(d>=.18)&(d<=.95)&np.isfinite(rays).all(-1)
    pts=(rays[valid]*d[valid,None])@pose[:3,:3].T+pose[:3,3]
    before=len(pts)
    if len(hands):pts=pts[cKDTree(hands).query(pts,k=1)[0]>.045]
    return pts,before-len(pts)


def grid(points,anchor,basis,radius=RADIUS):
    local=(points-anchor)@basis
    valid=(np.abs(local[:,0])<radius)&(np.abs(local[:,1])<radius)&(np.abs(local[:,2])<.30)
    q=local[valid];out=np.full(GRID*GRID,np.nan)
    if len(q):
        xy=np.floor((q[:,:2]+radius)*(GRID/(2*radius))).astype(int).clip(0,GRID-1)
        cells=xy[:,1]*GRID+xy[:,0];order=np.argsort(cells,kind="stable");c=cells[order];z=q[order,2]
        limits=np.r_[0,np.flatnonzero(np.diff(c))+1,len(c)]
        for l,h in zip(limits,limits[1:]):
            if h-l>=3:out[c[l]]=np.median(z[l:h])
    return out


def delta(g):
    support=np.isfinite(g).all(0)
    stable=support&(np.abs(g[0]-g[1])<=.015)&(np.abs(g[2]-g[3])<=.015)
    if stable.sum()<20:return None
    diff=(g[2:,stable].mean(0)-g[:2,stable].mean(0))
    control=np.abs(g[0,support]-g[1,support])
    return dict(cells=int(stable.sum()),common_cells=int(support.sum()),
        changed=float(np.mean(np.abs(diff)>THRESH)),nearer=float(np.mean(diff< -THRESH)),farther=float(np.mean(diff>THRESH)),
        magnitude_mm=float(np.mean(np.minimum(.10,np.abs(diff)))*1000),neutral=float(np.mean(control>THRESH)))


class Recording:
    def __init__(self,root,idx,name,duration):
        self.name=name;self.folder=root/"geometry-cache"/name;self.folder.mkdir(parents=True,exist_ok=True)
        def blob(key,suffix):
            path=self.folder/(key+"-"+suffix.replace("/","--"))
            if not path.exists():
                off,size=member(idx[key],name,suffix);path.write_bytes(range_bytes(key,off,size) if key!="video_compress" else fetch(session(),BASE+"video_compress.tar",off,off+size,'"0x8DC8C9DE268AE31"'))
            return path.read_bytes()
        vt=blob("video_compress","Video/VideoMp4Timing.txt").decode().splitlines();self.start=int(vt[0]);total=(int(vt[-1])-self.start)/1e7
        self.frames=DepthFrames(member(idx["ahat_depth"],name,"AhatDepth_synced.tar"),self.folder/"frames")
        tm=np.loadtxt(io.BytesIO(blob("cam_info","AhatDepth/Timing_sync.txt")),ndmin=2);ps=np.loadtxt(io.BytesIO(blob("cam_info","AhatDepth/Pose_sync.txt")),ndmin=2)
        # Timing_sync lists sensor samples at video rate. The distributed tar contains only the
        # subset named in AhatDepth_synced.txt, often one frame out of every three or four.
        published=blob("cam_info","AhatDepth/AhatDepth_synced.txt").decode().splitlines()
        valid_ids={int(Path(k.strip()).stem) for k in published if Path(k.strip()).stem.isdigit()}
        tm=tm[np.isin(tm[:,1].astype(int),list(valid_ids))]
        if len(tm)<8:raise ValueError("no_published_timed_depth")
        self.t=(tm[:,2]-self.start)/1e7;self.ids=tm[:,1].astype(int);self.pt=(ps[:,1]-self.start)/1e7;self.poses=ps[:,2:18].reshape(-1,4,4)
        self.intr=np.fromstring(blob("cam_info","AhatDepth/Intrinsics.txt").decode(),sep="\t");self.rays=make_rays(self.intr)
        self.eye=np.loadtxt(io.BytesIO(blob("eyes","Eyes/Eyes_sync.txt")),ndmin=2);self.eye_t=(self.eye[:,1]-self.start)/1e7
        self.hand=[SparseHands(member(idx["hands"],name,"Hands/"+side+"_sync.txt"),max(duration,total),self.start) for side in ("Left","Right")]
        self.max_id=int(self.ids.max());self.cache={}
        rgb=np.loadtxt(io.BytesIO(blob("video_compress","Video/Pose_sync.txt")),ndmin=2)
        self.rgb_t=(rgb[:,1]-self.start)/1e7;self.rgb_poses=rgb[:,2:18].reshape(-1,4,4)
        self.rgb_intr=np.fromstring(blob("cam_info","Video/Intrinsics.txt").decode(),sep="\t")
        if np.any(np.diff(self.t)<-.001):raise ValueError("Nonmonotone measured depth clock")

    def view(self,t):
        ix=np.searchsorted(self.t,t,side="right")-1
        if ix<0 or t-self.t[ix]>.23:raise ValueError("depth_age")
        fid=int(self.ids[ix]);ts=float(self.t[ix])
        if fid not in self.cache:
            pi=max(0,int(np.searchsorted(self.pt,ts+1e-6,side="right"))-1);pose=self.poses[pi]
            if abs(self.pt[pi]-ts)>.10 or not np.isfinite(pose).all() or abs(np.linalg.det(pose[:3,:3])-1)>.03:raise ValueError("pose_invalid")
            hands=[h.at(ts) for h in self.hand];hand=np.concatenate([v[0] for v in hands]);tips=np.concatenate([v[1] for v in hands])
            points,removed=cloud(self.frames.get(fid,self.max_id),pose,self.rays,hand)
            if len(points)<80:raise ValueError("insufficient_depth")
            self.cache[fid]=(points,pose,removed,len(hand),tips)
        return fid,ts,self.cache[fid]

    def anchor(self,t,view,contact):
        current,pose,_,_,hand=view[2]
        eye_i=np.searchsorted(self.eye_t,t,side="right")-1;eye=self.eye[max(0,eye_i)]
        gaze=eye_i>=0 and t-self.eye_t[eye_i]<=.15 and len(eye)>=9 and bool(eye[8]) and .7<np.linalg.norm(eye[5:8])<1.3
        if contact and len(hand):
            # Use a surface just beyond the excluded hand envelope. This locator is an outcome
            # measurement, and no post-action depth enters its selection.
            tree=cKDTree(current);dist,where=tree.query(hand,k=1)
            candidates=np.flatnonzero((dist>=.044)&(dist<=.135))
            if len(candidates):
                score=dist[candidates].copy()
                if gaze:
                    direction=eye[5:8]/np.linalg.norm(eye[5:8]);v=current[where[candidates]]-eye[2:5]
                    score+=.6*np.linalg.norm(v-(v@direction)[:,None]*direction,axis=1)
                best=candidates[np.argmin(score)];anchor=current[where[best]]
                neighbors=tree.query_ball_point(anchor,.025)
                if len(neighbors)>=3:anchor=np.median(current[neighbors],axis=0)
                source="hand_adjacent"
            else:contact=False
        else:contact=False
        if not contact:
            if not gaze:raise ValueError("gaze_invalid")
            direction=eye[5:8]/np.linalg.norm(eye[5:8]);v=current-eye[2:5];along=v@direction
            near=(along>.16)&(along<1.)&(np.linalg.norm(v-along[:,None]*direction,axis=1)<.03)
            if near.sum()<5:raise ValueError("no_attended_surface")
            pts=current[near];hit=np.quantile(along[near],.25);close=np.abs(along[near]-hit)<.035
            if close.sum()<3:raise ValueError("no_attended_surface")
            anchor=np.median(pts[close],axis=0);source="gaze"
        forward=anchor-pose[:3,3];forward/=np.linalg.norm(forward)
        right=-pose[:3,1].copy();right-=forward*(right@forward);right/=np.linalg.norm(right)
        return anchor,np.stack([right,np.cross(forward,right),forward],axis=1),source

    def prefix(self,event):
        cut=(event["cut"]+1)/10;v=self.view(cut);anchor,basis,source=self.anchor(cut,v,True)
        obs=grid(v[2][0],anchor,basis);ok=np.isfinite(obs)
        if ok.sum()<20:raise ValueError("prefix_geometry")
        return np.r_[np.where(ok,np.clip(obs/.2,-1.5,1.5),0),ok.astype(float)].astype(np.float32),v[0],v[1]

    def effect(self,event):
        st,en=event["start"],event["end"]
        initial=self.view(st-.12);pose=initial[2][1]
        anchor,basis,anchor_source=self.anchor(st-.12,initial,True)
        times=(st-.45,st-.12,en+.10,en+.42)
        vs=[self.view(t) for t in times]
        if len({v[0] for v in vs})<4:raise ValueError("repeat_depth_frame")
        shifts=[np.linalg.norm(v[2][1][:3,3]-pose[:3,3]) for v in vs]
        angles=[np.rad2deg(np.arccos(np.clip((np.trace(v[2][1][:3,:3].T@pose[:3,:3])-1)/2,-1,1))) for v in vs]
        if max(shifts)>.35 or max(angles)>45:raise ValueError("large_view_change")
        g=np.stack([grid(v[2][0],anchor,basis) for v in vs]);stats=delta(g)
        if stats is None:raise ValueError("stable_support")
        # The surrounding 40 cm patch uses identical measurement rules and excludes its central 20 cm.
        wide=np.stack([grid(v[2][0],anchor,basis,.20) for v in vs]).reshape(4,10,10)
        wide[:,2:8,2:8]=np.nan;bg=delta(wide.reshape(4,-1))
        before=None
        # This fluctuation is a measurement control. It is not passed to the predictor.
        old=grid(vs[0][2][0],anchor,basis);new=grid(vs[1][2][0],anchor,basis)
        observed=np.isfinite(old)&np.isfinite(new)
        if observed.sum()>=20:before=float(np.mean(np.abs(old[observed]-new[observed])>THRESH))
        def rgb_pixel(t):
            i=int(np.argmin(abs(self.rgb_t-t)));xyz=CV_TO_PSI.T@np.linalg.solve(self.rgb_poses[i][:3,:3],anchor-self.rgb_poses[i][:3,3])
            k=self.rgb_intr[:9].reshape(3,3)
            if xyz[2]<=.10:return None
            uv=k@xyz;return [float(uv[0]/uv[2]/self.rgb_intr[-2]),float(uv[1]/uv[2]/self.rgb_intr[-1])]
        stats.update(background_changed=None if bg is None else bg["changed"],pre_motion=before,anchor_world=anchor.tolist(),
            rgb_before=rgb_pixel(st-.12),rgb_after=rgb_pixel(en+.10),frame_ids=[v[0] for v in vs],frame_times=[v[1] for v in vs],
            anchor_source=anchor_source,head_shift_m=max(shifts),head_rotation_deg=max(angles),hand_samples=[v[2][3] for v in vs],
            hand_points_removed=[v[2][2] for v in vs],isolated=event["isolated"])
        return stats,g.astype(np.float32)


def indices(root,partial):
    idx={}
    for k in PHYSICAL:
        p=root/(k+"-index.json")
        if not p.exists() and partial:p=root/(k+"-index.partial.json")
        while not p.exists():time.sleep(3)
        idx[k]=json.loads(p.read_text())["members"]
    idx["video_compress"]=json.loads(Path("/mnt/slwm-data/SLWM/g10/data/holo/video_compress-index.json").read_text())["members"]
    return idx


def one(root,idx,record,folder):
    name=record["name"];final=folder/(name+".json")
    if final.exists():
        previous=json.loads(final.read_text())
        if previous.get("range_reader",1)>=2 or not any("AHAT frame not resolved" in s for s in previous["rejected"].values()):return previous
    wanted=[e for e in record["events"] if e["physical"]];rows=[];feats=[];fids=[];ftimes=[];grids=[];refused={}
    if wanted:
        try:rec=Recording(root,idx,name,record["duration"])
        except (KeyError,ValueError,IndexError) as exc:
            rec=None
            refused.update({str(e["id"]):"recording_unavailable: "+str(exc)[:110] for e in wanted})
        for e in (wanted if rec is not None else []):
            # Capture optional predictor geometry before collecting or gating the future target.
            try:
                feat,_,t=rec.prefix(e);fids.append(e["id"]);ftimes.append(t);feats.append(feat)
            except (ValueError,KeyError,IndexError,np.linalg.LinAlgError):pass
            try:
                stats,g=rec.effect(e);rows.append({"id":e["id"],**stats});grids.append(g)
            except (ValueError,KeyError,IndexError,np.linalg.LinAlgError) as exc:refused[str(e["id"])]=str(exc)[:120]
    np.savez_compressed(folder/(name+".npz"),ids=np.array([v["id"] for v in rows],np.int32),prefix_ids=np.asarray(fids,np.int32),prefix_times=np.asarray(ftimes,np.float64),prefix=np.asarray(feats,np.float32).reshape(-1,200),grids=np.asarray(grids,np.float32).reshape(-1,4,100))
    doc=dict(name=name,range_reader=2,attempted=len(wanted),valid=rows,rejected=refused);save(final,doc)
    return doc


def main(root,mode,workers):
    cat=json.loads((root/"annotations.json").read_text());groups=json.loads((root/"groups.json").read_text())
    partial=mode=="pilot";idx=indices(root,partial);folder=root/("geometry-pilot-final" if partial else "geometry");folder.mkdir(exist_ok=True)
    names=list(cat)
    if partial:
        names=[n for n in groups["folds"]["A"]["train"] if n in groups["folds"]["B"]["train"] and any(e["physical"] for e in cat[n]["events"]) and all(any(k.startswith(n+"/") for k in d) for d in idx.values())]
        names=sorted(names,key=lambda n:hashlib.sha256(("measurement/"+n).encode()).hexdigest())[:10]
    else:names=sorted(names,key=lambda n:hashlib.sha256(("geometry/"+n).encode()).hexdigest())
    completed=0
    with ThreadPoolExecutor(workers) as pool:
        tasks={pool.submit(one,root,idx,cat[n],folder):n for n in names}
        for future in as_completed(tasks):
            n=tasks[future];d=future.result();completed+=1
            if partial or completed%10==0:print(mode,completed,n,d["attempted"],len(d["valid"]),flush=True)
    save(folder/"complete",dict(recordings=len(names),mode=mode))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--mode",choices=("pilot","main"),default="pilot");p.add_argument("--workers",type=int,default=12)
    a=p.parse_args();main(a.root,a.mode,a.workers)
