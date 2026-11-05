"""All-case horizon diagnostics for continuous conditioning versus blind state rollout."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from training.broad_stream_report import interval
from training.stream_codec import PACKET_SECONDS
from training.t2_real_av import write_json


def main():
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"])
    corpus=Path(cfg["corpus"]); rows={r["id"]:r for r in json.loads((corpus/"ami_test/rows.json").read_text())}
    mel=np.load(corpus/"ami_test/mel.npy",mmap_mode="r"); codec=dict(np.load(corpus/"codec.npz"))
    mean=np.tile(codec["audio_mean"],4); scale=np.tile(codec["audio_scale"],4)
    cases=json.loads((root/"evaluation/long_cases.json").read_text()); observations=8*32
    edges=(0,128,256,512,704); details=[]; grouped=defaultdict(list)
    for case in cases:
        row=rows[case["id"]]; first=round(case["segment_start_seconds"]/PACKET_SECONDS)
        pred=dict(np.load(Path(case["folder"])/"proposals.npz"))
        n=len(pred["mel"]); truth=(np.asarray(mel[row["offset"]+first:row["offset"]+first+n],np.float32)-mean)/scale
        video=np.asarray(np.load(corpus/"ami_test"/(row["id"]+"_video.npy"),mmap_mode="r")[first:first+n],np.float32)
        trace=json.loads((Path(case["folder"])/"trace.json").read_text())
        if case["control"]=="prefix":
            assert all(t["observed_audio_fraction"]==t["observed_video_fraction"]==t["provided_text_fraction"]==0 for t in trace[8:])
        for i,(lo,hi) in enumerate(zip(edges,edges[1:])):
            a=observations+lo; z=min(n,observations+hi); assert z>a
            pv=pred["video"][a:z].astype(np.float32); tv=video[a:z]; pa=pred["mel"][a:z].astype(np.float32)
            value={"id":case["id"],"actor":case["actor"],"run":case["run"],"mode":case["mode"],"control":case["control"],"bin":i,
                "seconds_after_prefix_cutoff":[lo*PACKET_SECONDS,(z-observations)*PACKET_SECONDS],
                "audio_mae":float(np.abs(pa-truth[a:z]).mean()),"audio_training_mean_mae":float(np.abs(truth[a:z]).mean()),
                "video_mse":float(np.square(pv-tv).mean()),"video_training_mean_mse":float(np.square(tv).mean()),
                "predicted_video_motion_mse":float(np.square(np.diff(pv,axis=0)).mean()),
                "video_motion_error_mse":float(np.square(np.diff(pv,axis=0)-np.diff(tv,axis=0)).mean()),
                "static_video_motion_error_mse":float(np.square(np.diff(tv,axis=0)).mean())}
            details.append(value); grouped["|".join((case["run"],case["mode"],case["control"],str(i)))].append(value)
    summaries={}
    metrics=("audio_mae","audio_training_mean_mae","video_mse","video_training_mean_mse","predicted_video_motion_mse","video_motion_error_mse","static_video_motion_error_mse")
    for key,values in grouped.items(): summaries[key]={name:interval([r[name] for r in values],[r["actor"] for r in values]) for name in metrics}
    write_json(root/"horizon_diagnostics.json",{"scope":"Pre-test secondary diagnostic, not a primary gate; persisted float16 proposals scored against prepared data; original full-precision case metrics remain separate",
        "observed_prefix_seconds":observations*PACKET_SECONDS,"blind_prefix_continuation_seconds":704*PACKET_SECONDS,
        "after_prefix_edges_seconds":[x*PACKET_SECONDS for x in edges],"continuous_conditioning_is_not_blind_forecasting":True,
        "prefix_rollout_type":"Persistent state with absent sensors; emitted predictions are not fed back as observations","groups":summaries,"all_cases":details})
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for column,mode in enumerate(("audio","video")):
        for run,color in (("EXP-T2-801","#d97706"),("EXP-T2-802","#2563eb")):
            for control,style in (("normal","-"),("reset",":"),("prefix","--")):
                for row,metric in enumerate(("audio_mae","video_motion_error_mse")):
                    y=[summaries["|".join((run,mode,control,str(i)))][metric]["person_mean"] for i in range(4)]
                    axes[row,column].plot([(a+b)*PACKET_SECONDS/2 for a,b in zip(edges,edges[1:])],y,color=color,ls=style,label=run[-3:]+" / "+control)
        axes[0,column].set_title(("Audio" if mode=="audio" else "Video")+" conditioning"); axes[0,column].set_ylabel("Normalized mel MAE"); axes[1,column].set_ylabel("Video motion error MSE")
        for ax in axes[:,column]: ax.set_xlabel("Seconds after the10.9s prefix boundary"); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle("Continuous input versus state-only continuation · all12 reserved people\nPrefix loses all sensors; normal/reset keep receiving input · seed91 diagnostics")
    fig.savefig(root/"horizon_diagnostics.png",dpi=170); plt.close(fig)
    print(json.dumps({"horizon_report_complete":True,"all_case_bins":len(details)}),flush=True)


if __name__=="__main__": main()
