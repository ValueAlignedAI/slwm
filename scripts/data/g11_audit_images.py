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

def main(root,out,extended=False):
    cat=json.loads((root/"annotations.json").read_text());group=json.loads((root/"groups.json").read_text());out.mkdir(parents=True,exist_ok=True)
    choices=[]
    for fam,names in group["fresh"].items():
        cases=[]
        for n in names:
            meta=json.loads((root/"geometry"/(n+".json")).read_text());lookup={v["id"]:v for v in cat[n]["events"]}
            for d in meta["valid"]:cases.append((order(n+"/"+str(d["id"])),n,d,lookup[d["id"]]))
        if extended:
            choices.extend(cases);continue
        used=set()
        for key,n,d,e in sorted(cases):
            if n in used:continue
            choices.append((key,n,d,e));used.add(n)
            if len(used)==6:break
    choices.sort();audit=[]
    for i,(key,name,stats,event) in enumerate(choices):
        if extended:
            if i%16==0:sheet=Image.new("RGB",(1600,740),"#f5f5f5")
            path=root/"fresh/media"/(name+"-av.mp4");image=frame(path,event["start"]-.12)
            x=(i%4)*400;y=((i%16)//4)*185;uv=stats["rgb_before"]
            canvas=image.resize((272,153));tile=Image.new("RGB",(122,122),"#e5e5e5")
            if uv is not None and all(0<=v<=1 for v in uv):
                px,py=int(uv[0]*352),int(uv[1]*198)
                tile=image.crop((px-36,py-36,px+36,py+36)).resize((122,122))
                for pic,xx,yy in ((canvas,int(uv[0]*272),int(uv[1]*153)),(tile,61,61)):
                    d=ImageDraw.Draw(pic)
                    for col,w in (("black",3),("white",1)):
                        for line in ((xx-10,yy,xx-3,yy),(xx+3,yy,xx+10,yy),(xx,yy-10,xx,yy-3),(xx,yy+3,xx,yy+10)):
                            d.line(line,fill=col,width=w)
            else:ImageDraw.Draw(tile).text((10,55),"out of frame",fill="black")
            sheet.paste(canvas,(x,y+20));sheet.paste(tile,(x+274,y+20));ImageDraw.Draw(sheet).text((x+3,y+3),f"{i+1:03d}",fill="black")
        else:
            if i%6==0:sheet=Image.new("RGB",(1440,690),"#f4f4f4")
            path=root/"fresh/media"/(name+"-av.mp4")
            x=(i%2)*720;y=((i%6)//2)*230
            for j,(t,uv) in enumerate(((event["start"]-.12,stats["rgb_before"]),(event["end"]+.10,stats["rgb_after"]))):
                im=frame(path,t);mark(im,uv);sheet.paste(im,(x+j*356,y+24))
            ImageDraw.Draw(sheet).text((x+4,y+5),f"{i+1:02d}   before / after",fill="#111")
        audit.append(dict(index=i+1,recording=name,id=event["id"],family=cat[name]["family"],anchor=stats["anchor_source"]))
        per=16 if extended else 6
        if (i+1)%per==0 or i==len(choices)-1:sheet.save(out/f"sheet-{i//per+1:02d}.jpg",quality=92)
    (out/"sample.json").write_text(json.dumps(audit,indent=2)+"\n")
    print("blinded views",len(audit),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("root",type=Path);a.add_argument("out",type=Path);a.add_argument("--all-pre",action="store_true");p=a.parse_args();main(p.root,p.out,p.all_pre)
