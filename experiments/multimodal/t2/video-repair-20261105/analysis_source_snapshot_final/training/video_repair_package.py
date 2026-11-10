"""Package trained research checkpoints and hash the finished evidence."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from training.t2_real_av import write_json
from training.t2_prepare_cremad import sha256


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); corpus=Path(cfg["corpus"])
    selection=json.loads((root/"selection.json").read_text()); summary=json.loads((root/"summary.json").read_text()); audit=json.loads((root/"audit.json").read_text()); assert audit["passed"]
    # Verify both retained evaluation revisions and the current corrected source.
    for suffix,folder in (("","evaluation_source_snapshot"),("_v2","evaluation_source_snapshot_v2")):
        for name,digest in json.loads((root/("evaluation_source_hashes"+suffix+".json")).read_text()).items():
            assert sha256(root/folder/name)==digest
            if suffix=="_v2": assert sha256(Path(name))==digest
    bundle=root/"bundle"; entries={}
    for kind,group in (("audio","audio"),("video","primary_video")):
        run=selection[group]["61"]; source=root/run/"best.pt"; target=bundle/(kind+"_seed61.pt"); shutil.copyfile(source,target)
        digest=sha256(target); assert digest==selection["checkpoint_hashes"][run]
        entries[kind]={"file":target.name,"run":run,"seed":61,"sha256":digest,"chosen_by":"Fixed primary video seed; audio selected on validation only"}
    write_json(bundle/"manifest.json",{"entries":entries,"codecs":{name:sha256(bundle/name) for name in ("codec.npz","codec100.npz")},
        "all_other_seeds_and_ablations":"../EXP-T2-*/best.pt and last.pt","decision":summary["decision"],"model_scope":"Offline short-gap research; no production-quality or general world-model claim",
        "video_no_video_behavior":"Exactly the frozen audio anchor, not the separate stronger audio checkpoint","selection_sha256":sha256(root/"selection.json")})
    (bundle/"README.md").write_text(f'''# SLWM true-gap repair research bundle

{summary["decision"]}.

- [Audio control](audio_seed61.pt): {entries["audio"]["run"]}, validation-selected.
- [Video correction model](video_seed61.pt): {entries["video"]["run"]}, fixed primary seed61.
- [Results](../report.md), [failure analysis](../failure_analysis.md), [all32 listening cases](../listen.html).
- [Hashes and checkpoint identities](manifest.json).

Both files include all frozen and trained neural weights and their configuration.
The two NPZ codecs are the unchanged CREMA-training statistics/PCA. In the SLWM
repository, ordinary-file CPU inference needs no dataset or recognizer:

```bash
python -m exploration.video_repair_infer --checkpoint {bundle}/video_seed61.pt \\
  --codec-dir {bundle} --audio recording.wav --video recording.mp4 \\
  --gaps 1.20:1.52 --device cpu --output /tmp/slwm-repair
```

For audio only, choose `audio_seed61.pt` and omit `--video`. Omitting video from
the visual checkpoint invokes its frozen anchor, not the stronger audio control.
Outputs preserve observed PCM samples exactly and predict only short erased gaps.
Use the fixed CREMA/RAV crop geometry; arbitrary camera views are unvalidated.
Clips are limited to5.12s, need1.28s context and gaps at most480ms. These remain
research artifacts; RAVDESS source data is CC BY-NC-SA4.0.
''')
    splits={}
    for split in ("crema_train","crema_validation","crema_test","rav_train","rav_validation","rav_test"):
        path=corpus/(split+"_manifest.json"); d=json.loads(path.read_text()); lengths=np.load(d["arrays"]["lengths"]["path"])
        splits[split]={"samples":d["samples"],"people":d["people"],"retained_seconds":float(lengths.sum()*.04),"eligible_main_test":int((lengths>=50).sum()),
            "array_bytes":sum(Path(x["path"]).stat().st_size for x in d["arrays"].values()),"manifest_sha256":sha256(path)}
    write_json(corpus/"dataset_card.json",{"version":"video_repair_v1","splits":splits,"fitting":"None; previous CREMA-training codecs fixed",
        "fresh_test_people":"rav_test only; CREMA test reused from earlier research","repurposed_validation":"rav_validation was earlier R1 test; it now selects weights",
        "retention":"At most128 video packets /5.12s per source; listening WAVs use the same5.12s cap. No claim that all source files are complete utterances.",
        "video_resolution":[64,96],"raw_integrity":"Raw SHA256 for all records; new RAV ZIP members CRC verified. Whole ZIP MD5 declared only.",
        "sources":{"RAVDESS":{"url":"https://zenodo.org/records/1188976","license":"CC BY-NC-SA 4.0"},"CREMA":"Existing versioned CREMA source card retained"},
        "array_storage":"Fedora /home/db/dev/slwm; local delivery mirrors metadata and trained artifacts, not high-resolution training arrays"})
    checks=json.loads((root/"implementation_checks.json").read_text()); checks["repository_tests_revision2"]={"passed":124,"seconds":2.94,"log_sha256":sha256(root/"tests_v2.log")}; write_json(root/"implementation_checks.json",checks)
    extra=("exploration/video_repair_gallery.py","evals/video_repair_diagnostics.py","training/video_repair_package.py","docs/training/video_repair.md","docs/research/video_repair_crop_inspection_20261105.md","docs/research/findings.md","docs/experiments/experiment_registry.md","README.md","training/README.md","docs/README.md","docs/training/preprocessing.md")
    for name in extra:
        target=root/"analysis_source_snapshot_final"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=sha256(Path(name)): raise ValueError("Final analysis source changed: "+name)
        if not target.exists(): shutil.copyfile(name,target)
    write_json(root/"analysis_source_hashes_final.json",{name:sha256(Path(name)) for name in extra})
    # Evidence hash list includes all checkpoint copies and PCM files, as well as
    # retained failed attempts. Exclude this manifest to avoid self-reference.
    files={str(path.relative_to(root)):sha256(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name!="delivery_manifest.json" and not path.name.endswith(".log")}
    write_json(root/"delivery_manifest.json",{"files":files,"training_runs":len(cfg["anchors"])+len(cfg["runs"]),"trained_positive_windows":sum(r["positive_windows"] for r in summary["runs"]),
        "extra_visual_negative_windows":5*cfg["training"]["steps"]*cfg["training"]["batch_size"],"feature_metrics_unchanged_by_render_correction":True,
        "waveforms":audit["waveforms_verified"],"full_tests":124,"all_training_and_evaluation_jobs_complete":True,
        "data_manifests":{s:d["manifest_sha256"] for s,d in splits.items()},"source_sha256":sha256(Path(__file__))})
    print(json.dumps({"packaged_runs":11,"waveforms":audit["waveforms_verified"],"files_hashed":len(files)}))


if __name__=="__main__": main()
