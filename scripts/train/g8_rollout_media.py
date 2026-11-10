"""Render physical replays of the first reserved long-horizon start, checking every decision endpoint."""
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from training.g8_scenes import physics,initial,execute_prefix
from training.g7_units import put
from scripts.train.g7_lock_heads import sha
from scripts.train.g6_artifacts import roots

LABELS=(("hybrid-511-replay","Hybrid, replay"),("hybrid-511-sim","Hybrid, new physics data"),("hybrid-511-other","Hybrid, new non-sim data"),
        ("hybrid-511-fresh","Hybrid, both new"),("gpt2-511-fresh","Dense GPT-2, both new"),("geometry","Raw geometry"))
W=180;TITLE=26;BOTTOM=20


def render(task,root,out):
    import pygame
    import pymunk
    import pymunk.pygame_util
    sources=[];envs=[];max_error=0.;raw={}
    for tag,label in LABELS:
        folder=root/"live"/task/tag
        with np.load(folder/"trajectory.npz") as z:sources.append({k:np.array(z[k][0]) for k in ("seed","goals","object_xy","agent_xy","executed","choice")})
        raw[tag]=sha(folder/"trajectory.npz");r=sources[-1];env=physics(task);_,state,prefix,_,_=initial(task,int(r["seed"]));execute_prefix(env,int(r["seed"]),state,prefix,False)
        env.observation_width=W;env.observation_height=W;envs.append(env)
    row=sources[0];seed=int(row["seed"]);goal=row["goals"]
    for r in sources:
        assert int(r["seed"])==seed;np.testing.assert_array_equal(r["goals"],goal)
    font=ImageFont.load_default(size=12);filename=out/(task+".webm");tmp=out/(task+".partial.webm")
    width,height=W*3,(TITLE+W+BOTTOM)*2+30;fps=envs[0].control_hz
    process=subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r",str(fps),"-i","pipe:0",
         "-an","-c:v","libvpx-vp9","-deadline","realtime","-cpu-used","6","-crf","34","-b:v","400k","-pix_fmt","yuv420p",str(tmp)],stdin=subprocess.PIPE)
    try:
        for t in range(24):
            for h in range(16):
                frame=Image.new("RGB",(width,height),"#f6f7f7");artist=ImageDraw.Draw(frame)
                for j,(env,r,(tag,label)) in enumerate(zip(envs,sources,LABELS)):
                    pose=env.step(r["executed"][t,h])[0];pixels=env._render(visualize=False)
                    xx=(j%3)*W;yy=(j//3)*(TITLE+W+BOTTOM)
                    frame.paste(Image.fromarray(pixels),(xx,yy+TITLE));artist.text((xx+4,yy+4),label,fill="#16373b",font=font)
                    gx,gy=pymunk.pygame_util.to_pygame(pymunk.Vec2d(*goal),pygame.Surface((512,512)));gx=xx+gx*W/512;gy=yy+TITLE+gy*W/512;radius=20*W/512
                    artist.ellipse((gx-radius,gy-radius,gx+radius,gy+radius),outline="#c53641",width=2)
                    distance=np.linalg.norm(pose[2:4]-goal);artist.text((xx+4,yy+TITLE+W+1),f"Distance {distance:.0f} px",fill="#344450",font=font)
                    if h==15:
                        err=np.max(np.abs(pose[2:4]-r["object_xy"][t+1]));max_error=max(max_error,float(err))
                        np.testing.assert_allclose(pose[2:4],r["object_xy"][t+1],atol=1e-3,rtol=0)
                artist.text((5,height-22),f"{task}   |   start {seed}   |   decision {t+1}/24   |   {(16*t+h+1)/fps:.1f} s",fill="#26363d",font=font)
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait(timeout=90):raise RuntimeError("Could not encode rollout preview")
        tmp.replace(filename)
    finally:
        if not process.stdin.closed:process.stdin.close()
        if process.poll() is None:process.terminate();process.wait()
        for env in envs:env.close()
    duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(filename)]))
    if abs(duration-384/fps)>.12:raise ValueError("Incorrect replay length")
    return {"seed":seed,"task":task,"duration_seconds":duration,"fps":fps,"frames":384,"max_replay_endpoint_difference_px":max_error,"file_sha256":sha(filename),
            "source_trajectories":raw,"goal_radius_px":20,"description":"Executed actions in a higher-resolution simulator replay. The red goal overlay is for the viewer."}


if __name__=="__main__":
    r,_=roots("g8");out=r/"public/examples";out.mkdir(exist_ok=True,parents=True)
    data={t:render(t,r,out) for t in ("pusht","puck_barrier")}
    put(r/"public/rollout-examples.json",data);print("Both physical replays verified",flush=True)
