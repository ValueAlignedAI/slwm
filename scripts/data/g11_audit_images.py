"""Render blinded source views of the pre-action local depth anchor, independent of predictions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def order(key):return hashlib.sha256(("G11-audit-fixed/"+key).encode()).hexdigest()

def frame(path,t):
    cmd=["ffmpeg","-nostdin","-v","error","-ss",str(max(0,t)),"-i",str(path),"-frames:v","1","-vf","scale=352:198","-threads","1","-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
    raw=subprocess.check_output(cmd,timeout=40)
    return Image.fromarray(np.frombuffer(raw,np.uint8).reshape(198,352,3))

def mark(image,uv):
    draw=ImageDraw.Draw(image)
    if uv is None:return
    x,y=round(uv[0]*352),round(uv[1]*198)
    for color,width in (("black",4),("white",2)):
        for ax,ay,bx,by in ((x-13,y,x-5,y),(x+5,y,x+13,y),(x,y-13,x,y-5),(x,y+5,x,y+13)):
            draw.line((ax,ay,bx,by),fill=color,width=width)

def main(root,out):
    cat=json.loads((root/"annotations.json").read_text());group=json.loads((root/"groups.json").read_text());out.mkdir(parents=True,exist_ok=True)
    choices=[]
    for fam,names in group["fresh"].items():
        cases=[]
        for n in names:
            meta=json.loads((root/"geometry"/(n+".json")).read_text());lookup={v["id"]:v for v in cat[n]["events"]}
            for d in meta["valid"]:cases.append((order(n+"/"+str(d["id"])),n,d,lookup[d["id"]]))
        used=set()
        for key,n,d,e in sorted(cases):
            if n in used:continue
            choices.append((key,n,d,e));used.add(n)
            if len(used)==6:break
    choices.sort();audit=[]
    for i,(key,name,stats,event) in enumerate(choices):
        if i%6==0:sheet=Image.new("RGB",(1440,690),"#f4f4f4")
        path=root/"fresh/media"/(name+"-av.mp4")
        x=(i%2)*720;y=((i%6)//2)*230
        for j,(t,uv) in enumerate(((event["start"]-.12,stats["rgb_before"]),(event["end"]+.10,stats["rgb_after"]))):
            im=frame(path,t);mark(im,uv);sheet.paste(im,(x+j*356,y+24))
        ImageDraw.Draw(sheet).text((x+4,y+5),f"{i+1:02d}   before / after",fill="#111")
        audit.append(dict(index=i+1,recording=name,id=event["id"],family=cat[name]["family"],anchor=stats["anchor_source"]))
        if (i+1)%6==0 or i==len(choices)-1:sheet.save(out/f"sheet-{i//6+1}.jpg",quality=91)
    (out/"sample.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("blinded views",len(audit),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("root",type=Path);a.add_argument("out",type=Path);p=a.parse_args();main(p.root,p.out)
