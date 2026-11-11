"""Blind uniform and blue-challenge readback on entirely new parent groups."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from g13_robot import rank
from g13_repair import get_frames,decode
from g13_acquire import ROOT,write

def main():
    r=ROOT/"robot-extra";out=r/"audit";out.mkdir(exist_ok=True)
    ep=json.loads((r/"banks/episodes.json").read_bytes())
    with np.load(r/"banks/test.npz") as z:old=z["mover"];ids=z["episode"];first=z["first"];ticks=z["tick"]
    with np.load(r/"corrected/test.npz") as z:new=z["mover"]
    uniform=sorted(np.flatnonzero(first),key=lambda i:rank("external/uniform/"+ep[ids[i]]["episode"]))[:100]
    chosen=set(uniform);challenge=sorted((i for i in np.flatnonzero(first&((old==2)|(new==2))) if i not in chosen),
                                          key=lambda i:rank("external/blue/"+ep[ids[i]]["episode"]))[:35]
    picked=[dict(episode=int(ids[i]),tick=int(ticks[i]),old=int(old[i]),new=int(new[i]),panel="uniform" if i in chosen else "blue") for i in uniform+challenge]
    order=[(rank("external/first/"+str(v["episode"])),v) for v in picked]
    repeat=sorted(picked,key=lambda v:rank("external/repeat/"+str(v["episode"])))[:10]
    order.extend((rank("external/dup/"+str(v["episode"])),v) for v in repeat);order.sort()
    index=[]
    for i,(opaque,v) in enumerate(order):
        r0=ep[v["episode"]];frames=get_frames(r,r0);t=v["tick"]
        images=[Image.fromarray(decode(frames[j])[:,:,::-1]) for j in (t,-1)]
        reverse=int(opaque[-1],16)%2
        if reverse:images.reverse()
        if i%12==0:sheet=Image.new("RGB",(1310,1190),"#eeeeee")
        x=i%2*655;y=(i%12)//2*198;ImageDraw.Draw(sheet).text((x+3,y+1),f"{i+1:03d}",fill="black")
        sheet.paste(images[0],(x,y+15));sheet.paste(images[1],(x+324,y+15))
        index.append(dict(number=i+1,**v,group=r0["group"],reversed=bool(reverse)))
        if i%12==11 or i==len(order)-1:sheet.save(out/f"extra-{i//12+1:02d}.jpg",quality=90)
    write(out/"index.json",index);print(len(picked),len(order),flush=True)


if __name__=="__main__":main()
