"""Verify duration and generated tracks before committing a continuous inference preview."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main(root):
    files={"audio-to-video.webm":{"video"},"video-to-audio.ogg":{"audio"},"text-to-audio-video.webm":{"audio","video"}}
    manifest={}
    for name,expected in files.items():
        p=root/"previews"/name
        data=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type","-of","json",str(p)]))
        dt=float(data["format"]["duration"])
        if abs(dt-1024*1024/24000)>.1 or {v["codec_type"] for v in data["streams"]}!=expected:raise ValueError("Wrong or incomplete preview: "+name)
        with p.open("rb") as f:blob=hashlib.file_digest(f,"sha256").hexdigest()
        manifest[name]={"seconds":dt,"sha256":blob,"bytes":p.stat().st_size,"generated_tracks":sorted(expected)}
    q=root/"previews.partial";q.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");q.replace(root/"previews.json")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);main(p.parse_args().folder)
