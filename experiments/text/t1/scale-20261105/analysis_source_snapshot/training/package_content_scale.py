"""Portable selected model bundles and immutable evidence hash inventories."""
import json
from pathlib import Path
import shutil
import torch
from training.research_sources import file_sha
from training.t2_real_av import write_json


def manifest(root):
    files={str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="delivery_manifest.json" and p.suffix!=".log"}
    write_json(root/"delivery_manifest.json",{"files":files,"source_sha256":file_sha(Path(__file__)),"training_and_evaluation_complete":True}); return len(files)


def main():
    cfg=json.loads(Path("configs/t2/content_sync.json").read_text()); root=Path(cfg["artifacts"]); audit=json.loads((root/"audit.json").read_text()); assert audit["passed"]; bundle=root/"bundle"; bundle.mkdir(exist_ok=True); entries={}
    for name,run in (("content_base","EXP-T2-702"),("audio","EXP-T2-711"),("video","EXP-T2-713")):
        shutil.copyfile(root/run/"best.pt",bundle/(name+".pt")); entries[name]={"run":run,"sha256":file_sha(bundle/(name+".pt"))}
    cal=json.loads((root/"calibration.json").read_text())["EXP-T2-713"]; write_json(bundle/"calibration.json",cal); summary=json.loads((root/"summary_v2.json").read_text()); (bundle/"README.md").write_text('''# Content and synchronization research bundle

These are small research models, validated on reserved GRID people and sentence
combinations. Read [the corrected results](../report_v2.md) and [all fixed audio comparisons](../listen.html)
before interpreting capability. The vocabulary is restricted; this is not a
general world model or production speech system.

- `audio.pt`: matched additional-audio candidate711.
- `video.pt`: video candidate713 plus its exact strong-audio fallback and frozen calibration.
- `content_base.pt`: content-supervised base702, with an explicit timed-word text interface.
- The two fixed NPZ codecs and training-only word vocabulary are included.

From the SLWM repository root:

```bash
python -m exploration.content_sync_infer --checkpoint BUNDLE/video.pt \\
  --codec-dir BUNDLE --calibration BUNDLE/calibration.json \\
  --media recorded_grid.mpg --gaps 1.32:1.64 --device cpu --output /tmp/slwm-repair
```

Replace `BUNDLE` with this directory. Use `audio.pt` for audio-only prediction;
`--audio recording.wav` instead of `--media` supplies an already synchronized
waveform. Original GRID media uses the recorded PTS alignment and fixed crop.
Observed PCM is preserved on that aligned timeline; initial trim/pad is logged.
With the base checkpoint, `--timed-text` explicitly supplies a JSON list of
`{start,end,word}` records. Practical audio/video inference receives no labels.
`--probe-words` requests a separate closed-vocabulary diagnostic. No recognizer
is needed for inference. Other camera views, open vocabulary and causal physical
dynamics remain unvalidated. Keep the source research-use permissions.
''')
    write_json(bundle/"manifest.json",{"entries":entries,"calibration":cal,"gates":{k:v for k,v in summary.items() if k.endswith("gate_passes")},"codecs":{n:file_sha(bundle/n) for n in ("codec.npz","codec100.npz","vocabulary.json")}})
    corpus=Path(cfg["corpus"]); sets={s:set(r["sentence"] for r in json.loads((corpus/(s+".json")).read_text())) for s in ("train","validation","calibration","test","test_seen")}; card=json.loads((corpus/"dataset_card.json").read_text())
    write_json(root/"dataset_scope_audit.json",{"splits":{s:{"clips":d["samples"],"people":len(d["people"]),"retained_hours":d["retained_seconds"]/3600} for s,d in card["splits"].items()},"unique_sentence_combinations":{s:len(v) for s,v in sets.items()},"primary_test_combinations_seen_in_train":len(sets["test"]&sets["train"]),
        "secondary_test_seen_caveat":"This name means training-eligible sentence hash buckets, not guaranteed exact sentence occurrence in training", "secondary_sentence_combinations_actually_seen":len(sets["test_seen"]&sets["train"]),"reviewed_excluded_training_clips":card["exclusions"],"vocabulary_size":len(card["vocabulary"]),"dataset_card_sha256":file_sha(corpus/"dataset_card.json")})
    tcfg=json.loads(Path("configs/t1/fineweb_scale.json").read_text()); text=Path(tcfg["artifacts"]); assert json.loads((text/"audit.json").read_text())["passed"]; tb=text/"bundle"; tb.mkdir(exist_ok=True)
    exports={}
    for name,run in (("gpt2_seed81","EXP-T1-201"),("slwm_seed81","EXP-T1-202")):
        ck=torch.load(text/run/"best.pt",map_location="cpu",weights_only=False); exported={k:v for k,v in ck.items() if k!="optimizer"}; torch.save(exported,tb/(name+".pt")); exports[name]={"run":run,"sha256":file_sha(tb/(name+".pt"))}
    write_json(tb/"manifest.json",{"inference_exports":exports,"full_optimizer_and_rng_checkpoints":"Retained in all four run directories","scope":"16.8M tokens/run early checkpoint; not converged or multimodal"})
    (tb/"README.md").write_text('''# Early GPT-2-scale text checkpoints

`gpt2_seed81.pt` and `slwm_seed81.pt` contain selected inference weights/configs.
All four full optimizer checkpoints remain in the run directories. Both models
start from random weights and see 16.8M tokens from a 2.6B-token prepared corpus.
These are early learning artifacts, not fluent or factual assistants.

```bash
python -m exploration.scale_text_infer --checkpoint BUNDLE/slwm_seed81.pt \\
  --prompt "The purpose of a model is" --tokens 64 --device cpu
```

Replace `BUNDLE` with this directory. No training corpus is needed for inference.
Read [the report](../report.md), [compute diagnostic](../compute_diagnostic.md)
and all fixed unselected continuations before drawing conclusions. Only 512-token
contexts were trained in this stage; a larger positional table is not evidence
of 1024-token quality. The trained speech models are separate weights.
''')
    extra=("exploration/content_sync_gallery.py","exploration/scale_text_infer.py","training/package_content_scale.py","training/scale_text_report.py","evals/scale_compute_diagnostic.py","evals/scale_text_audit.py","scripts/train/package_content_scale_fedora.sh")
    for dest in (root,text):
        prior=dest/"analysis_source_snapshot"
        if prior.exists() and not (dest/"initial_analysis_source_snapshot").exists():
            shutil.copytree(prior,dest/"initial_analysis_source_snapshot")
            shutil.copyfile(dest/"analysis_source_hashes.json",dest/"initial_analysis_source_hashes.json")
        for name in extra:
            path=dest/"analysis_source_snapshot"/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(name,path)
        write_json(dest/"analysis_source_hashes.json",{n:file_sha(n) for n in extra})
    corrections=("evals/content_sync_word_correction.py","training/content_sync_report_v2.py")
    for name in corrections:
        target=root/"word_evaluation_v2/source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_sha(target)!=file_sha(name): raise ValueError("Word evaluation revision changed: "+name)
        shutil.copyfile(name,target)
    write_json(root/"word_evaluation_v2/source_hashes.json",{n:file_sha(n) for n in corrections})
    for dest,source,names in ((root,corpus,("dataset_card.json","vocabulary.json","exclusions.json","source_scope.json")),(text,Path(tcfg["corpus"]),("dataset_card.json","pinned_sources.json"))):
        metadata=dest/"dataset_metadata"; metadata.mkdir(exist_ok=True)
        for name in names: shutil.copyfile(source/name,metadata/name)
    print(json.dumps({"content_files_hashed":manifest(root),"text_files_hashed":manifest(text),"waveforms":audit["waveforms_verified"]}))


if __name__=="__main__": main()
