"""Recount the realized source masks, unique recordings, windows and episode density from locked RNG streams."""
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

root=Path(sys.argv[1]);dest=Path(sys.argv[2])
with np.load(root/"starss-train.npz") as bank:
    lengths=bank["lengths"];names=list(map(str,bank["names"]));tokens=bank["w"]
    offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    starts=np.concatenate([o+np.arange(n-32) for o,n in zip(offsets,lengths)]).astype(np.int32)
    input_stats={"recordings":len(names),"rooms":sorted({n.split("_")[1] for n in names}),"clock_bins":int(lengths.sum()),
                 "recorded_hours":float(lengths.sum()/36000),"event_annotated_clock_fraction":float((tokens[:,0]==24).mean()),
                 "mean_observed_event_words":float((tokens>=32).sum(1).mean()),"valid_training_window_starts":len(starts)}
patterns=np.array([[1]*4]*8+[[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],bool)
out={"training_starss":input_stats,"per_tick":{"spectral_values":64,"quantized_rgb_values":768,"physical_channels":12,"maximum_word_slots":8},"runs":{}}
for seed in (911,912):
    def choices(step):
        pick,drop=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed+31),step))
        return jax.random.randint(pick,(32,),0,len(starts)),jax.random.randint(drop,(128,),0,16)
    selected,mode=jax.jit(jax.vmap(choices))(jnp.arange(4096))
    for n in (384,1024,4096):
        win=starts[np.asarray(selected[:n]).reshape(-1)];mask=patterns[np.asarray(mode[:n])]
        covered=np.zeros(int(lengths.sum())+1,np.int64)
        np.add.at(covered,win,1);np.add.at(covered,win+32,-1)
        out["runs"][f"{seed}-{n}"]={"sim_fresh_episodes":n*96,"sim_replay_unique_episodes":min(2048,n*96),
            "real_windows_sampled":n*32,"real_distinct_window_starts":len(np.unique(win)),"real_distinct_clock_bins":int((covered.cumsum()[:-1]>0).sum()),
            "total_sampled_clock_positions":n*128*32,"sim_sampled_clock_positions":n*96*32,"real_sampled_clock_positions":n*32*32,
            "all_four_fraction":float(mask.all(-1).mean()),"all_four_sim_fraction":float(mask[:,:96].all(-1).mean()),
            "all_four_real_fraction":float(mask[:,96:].all(-1).mean()),"source_fraction_audio_vision_physical_words":mask.mean((0,1)).tolist()}
if (root/"contact/roma.json").exists():
    contact=json.loads((root/"contact/roma.json").read_text());trials=contact["trials"].values()
    out["external_contact"]={"revision":contact["revision"],"selected_scenes":contact["scenes"],"trials":len(contact["trials"]),"failed":contact["failed"],
        "training_use":"external ridge calibration only, six scenes","calibration_ticks":contact["cal"]["ticks"],
        "held_ticks":contact["held"]["ticks"],"max_camera_age_s":max(v["max_camera_age_s"] for v in trials),
        "max_force_age_s":max(v["max_force_age_s"] for v in trials),"verbs":sorted({v["verb"] for v in trials})}
dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
