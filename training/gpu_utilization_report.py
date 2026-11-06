"""Descriptive NVIDIA telemetry; utilization is not model FLOP utilization."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np


def stamp(value):
    return datetime.strptime(value.strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--gpu-mib",type=float,default=24576)
    parser.add_argument("--after",help="Inclusive UTC lower bound, YYYY-MM-DDTHH:MM:SS")
    parser.add_argument("--before",help="Inclusive UTC upper bound, YYYY-MM-DDTHH:MM:SS")
    args=parser.parse_args(); root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    # Take one immutable byte snapshot; a concurrently appended partial row is omitted.
    raw=Path(args.input).read_bytes(); raw=raw[:raw.rfind(b"\n")+1]
    rows=list(csv.reader(io.StringIO(raw.decode())))
    lower=datetime.fromisoformat(args.after).replace(tzinfo=timezone.utc) if args.after else None
    upper=datetime.fromisoformat(args.before).replace(tzinfo=timezone.utc) if args.before else None
    parsed=[]; selected=[rows[0]]
    for row in rows[1:]:
        if len(row)!=5: raise ValueError("Unexpected NVIDIA telemetry row")
        at=stamp(row[0])
        if (lower and at<lower) or (upper and at>upper): continue
        values=[float(value.strip().split()[0]) for value in row[1:]]
        parsed.append((at,*values)); selected.append(row)
    if len(parsed)<2: raise ValueError("At least two complete telemetry observations are required")
    array=np.asarray([[r[0].timestamp(),*r[1:]] for r in parsed]); times=array[:,0]-array[0,0]
    if np.any(np.diff(times)<=0): raise ValueError("Telemetry timestamps must increase")
    result={"scope":"Descriptive sampled GPU telemetry, including every workload/queue within the selected interval; no causal attribution to a model",
        "interpretation":"NVIDIA GPU utilization measures sampled kernel activity, not achieved FLOPs, occupancy, useful training throughput, or a quality improvement",
        "first_utc":parsed[0][0].isoformat(),"last_utc":parsed[-1][0].isoformat(),"observations":len(parsed),
        "elapsed_seconds":float(times[-1]),"sample_interval_seconds":{"median":float(np.median(np.diff(times))),"maximum":float(np.diff(times).max())},
        "gpu_utilization_percent":{"sample_mean":float(array[:,1].mean()),"p10_median_p90":np.quantile(array[:,1],[.1,.5,.9]).tolist(),"zero_sample_fraction":float(np.mean(array[:,1]==0)),"at_least_90_percent_sample_fraction":float(np.mean(array[:,1]>=90))},
        "memory_used_mib":{"sample_mean":float(array[:,3].mean()),"peak":float(array[:,3].max()),"device_capacity":args.gpu_mib,"peak_capacity_fraction":float(array[:,3].max()/args.gpu_mib)},
        "power_watts":{"sample_mean":float(array[:,4].mean()),"peak":float(array[:,4].max())},
        "sampled_energy_kwh_trapezoidal_estimate":float(np.sum(np.diff(times)*(array[:-1,4]+array[1:,4])/2)/3600000),
        "data_snapshot":"samples.csv","training_or_test_scores_used":False}
    with (root/"samples.csv").open("w") as stream: csv.writer(stream).writerows(selected)
    (root/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout="constrained")
    axes[0].plot(times/3600,array[:,1],lw=.5,color="#2563eb"); axes[0].set(ylabel="Sampled GPU activity (%)",ylim=(0,100))
    axes[1].plot(times/3600,array[:,3]/1024,lw=.6,color="#d97706"); axes[1].axhline(args.gpu_mib/1024,color="#999",ls="--",label="Device capacity"); axes[1].set(xlabel="Elapsed hours in selected interval",ylabel="Device memory used (GiB)"); axes[1].legend()
    for axis in axes: axis.grid(alpha=.2)
    figure.suptitle("RTX 3090 telemetry · descriptive workload trace\n"+result["first_utc"]+" to "+result["last_utc"])
    figure.savefig(root/"usage.png",dpi=170); plt.close(figure)
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
