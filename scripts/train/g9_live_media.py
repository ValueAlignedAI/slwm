"""Replay the first held mission from the issued controls, alongside its observation-based oracle."""
import json
import subprocess
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from models.g9_tabletop import initial

MOVES=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[0,0],[0,0]])
LABELS=["hold","move +x","move -x","move +y","move -y","tap +","tap -"]
COLORS=[(208,75,73),(58,151,99),(55,117,194)]
WORDS=["red","green","blue"]


def main(scores,dest):
    panels={}
    for core in ("hybrid","gpt2"):
        tag=f"confirm-{core}-z128-gelu-s912-fresh-4096-test.json"
        d=json.loads((scores/tag).read_text());panels[core]=d["live"]["carry"]["first_mission_actions"]
    start=initial(jnp.array([0]),180013);position=np.array(start["pos"][0]);sign=np.array(start["sign"][0])
    stations=np.array(start["stations"][0]);first=int(start["goal"][0])
    panels["raw"]=[];positions={c:position.copy() for c in panels};seen=False;visit={c:0 for c in panels}
    dest.mkdir(parents=True,exist_ok=True);out=dest/"held-live-control.mp4"
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pixel_format","rgb24","-video_size","660x256","-framerate","10","-i","pipe:0",
         "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(out)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    font=ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",14)
    try:
        for tick in range(512):
            goal=(first+tick//64)%3;delta=stations[goal]-positions["raw"];axis=int(abs(delta[1])>abs(delta[0]))
            act=0 if not delta.any() else 5 if not seen else 1+2*axis+int(delta[axis]*sign[axis]<0)
            seen|=act>=5;panels["raw"].append(act)
            image=Image.new("RGB",(660,256),"white");draw=ImageDraw.Draw(image)
            draw.text((10,5),f"Request: reach {WORDS[goal]} and hold     {tick/10:.1f} s",font=font,fill="#182637")
            for i,core in enumerate(panels):
                x=10+220*i;act=panels[core][tick]
                if tick%5!=2:positions[core]=np.clip(positions[core]+MOVES[act]*sign,0,3)
                visit[core]+=bool((positions[core]==stations[goal]).all())
                title={"hybrid":"SLWM","gpt2":"Dense GPT-2","raw":"Sensor-based controller"}[core]
                draw.text((x,29),title,font=font,fill="#182637")
                for a in range(4):
                    for b in range(4):
                        l=x+38*a;u=54+38*b
                        draw.rectangle((l,u,l+36,u+36),fill="#eff2f4")
                for j,(a,b) in enumerate(stations):
                    l=x+38*a+5;u=54+38*b+5
                    draw.rectangle((l,u,l+25,u+25),outline=COLORS[j],width=6 if j==goal else 2)
                a,b=positions[core];l=x+38*a+12;u=54+38*b+12
                draw.ellipse((l,u,l+13,u+13),fill="#162536")
                draw.text((x,210),LABELS[act],font=font,fill="#182637");draw.text((x,232),f"At goal: {100*visit[core]/(tick+1):.0f}%",font=font,fill="#405463")
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError("Video encoding failed")
    finally:
        if proc.poll() is None:proc.terminate();proc.wait()
    (dest/"live-trace.json").write_text(json.dumps(dict(mission_seed=180013,index=0,source="recorded model actions; deterministic environment replay",hz=10,
        initial_position=position.tolist(),signs=sign.tolist(),stations=stations.tolist(),goal_start=first,actions=panels,at_goal_ticks=visit),indent=2)+"\n")


if __name__=="__main__":main(Path(sys.argv[1]),Path(sys.argv[2]))
