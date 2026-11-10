"""A local listening artifact displaying every fixed case in the same order."""
import argparse
import html
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"])
    summary=json.loads((root/"summary.json").read_text()); cases=json.loads((root/"exploration/audio_cases.json").read_text()); cards=[]
    def player(case,condition,label):
        path=f"exploration/{case['sample_id']}/{condition}.wav"
        return f'<div><span>{html.escape(label)}</span><audio controls preload="none" src="{html.escape(path,quote=True)}"></audio></div>'
    for i,case in enumerate(cases):
        identity=html.escape(case["sample_id"]); begin,end=case["gap_seconds"]
        primary="".join(player(case,c,label) for c,label in (("reference","Recorded speech"),("silence","Speech with missing segment"),("audio61","Audio repair"),("video61","Audio + video repair")))
        controls="".join(player(case,c,label) for c,label in (("interpolation","Spectral interpolation"),("wrong61","Different utterance video"),("shift61","Video shifted +240ms"),("static61","Still video frame"),("audio62","Audio repair, second seed"),("video62","Video repair, second seed"),("oracle_spectrum","Clean spectrum reconstruction · evaluation ceiling")))
        cards.append(f'<article data-split="{case["split"]}"><h2>{i+1:02d} · {identity}</h2><p class="meta">{html.escape(case["split"])} · recorded transcript: “{html.escape(case["transcript"])}” · gap {begin:.2f}–{end:.2f}s</p><div class="players">{primary}</div><details><summary>Controls and second seed</summary><div class="players">{controls}</div></details><details><summary>Recorded mouth frames around the gap</summary><img alt="Twelve actual recorded frames; two before and two after the eight-frame gap" src="exploration/{identity}/recorded_mouth.png"></details></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SLWM speech repair · fixed listening set</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#1d2937;background:#f3f5f7;margin:0}main{max-width:1160px;margin:auto;padding:32px 20px}h1{font-size:32px;line-height:1.2}h2{font-size:18px}header,article{background:white;border:1px solid #d7dee6;border-radius:10px;padding:22px;margin:16px 0}.meta{color:#526174;font-size:14px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:15px;margin-top:16px}.players span{display:block;font-size:14px;margin-bottom:5px}audio{width:100%}details{margin-top:18px}summary{cursor:pointer;color:#225b8f}img{max-width:100%;margin-top:15px}select{font:inherit;padding:8px;border:1px solid #9baabd;border-radius:5px}a{color:#225b8f}.figure{width:100%;margin:15px 0}
</style><main><header><p class="meta">SLWM · T2-V1 · RTX 3090 research artifacts</p><h1>Does video help repair missing speech?</h1><p>DECISION</p><p>All 32 examples were fixed by person and sample ID before scoring. These are actual model outputs. Every condition shares the same missing 320ms and preserves the observed waveform exactly. The recorded transcript is a reference label, not generated model text.</p><p>The main pair uses seed61. More controls, the second seed and the explicitly labeled reconstruction ceiling are available below each example.</p><p><a href="report.md">Full results</a> · <a href="summary.json">Measurements and confidence intervals</a></p><img class="figure" src="video_vs_audio.png" alt="Video versus audio effects and person-cluster confidence intervals"><label>Show <select id="filter"><option value="all">All fixed examples</option><option value="crema_test">CREMA · reused test people</option><option value="rav_test">RAVDESS · fresh test people</option></select></label></header>CARDS</main><script>document.querySelector('#filter').onchange=e=>{document.querySelectorAll('article').forEach(c=>c.hidden=e.target.value!=='all'&&c.dataset.split!==e.target.value)};document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>'''
    (root/"listen.html").write_text(page.replace("DECISION",html.escape(summary["decision"])+".").replace("CARDS","".join(cards)))
    print(json.dumps({"fixed_listening_cases":len(cases),"artifact":str(root/"listen.html")}))


if __name__=="__main__": main()
