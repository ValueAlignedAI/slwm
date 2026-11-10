"""Render registered, two-disk-verified learning curves. The band is the range of the two seeds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def draw(metrics,out):
    rows=[json.loads(f.read_bytes()) for f in metrics.glob("*.json")]
    rows=[v for v in rows if v.get("comparison")=="architecture" and v["split"]=="dev"]
    if not rows: raise ValueError("No verified pairs")
    sizes=sorted({r["arms"][0]["parameters"] for r in rows})
    if len(sizes)!=2: raise ValueError("Expected two registered scales")
    colors={"hybrid":"#087893","gpt2":"#a84c25"}
    fig,axes=plt.subplots(2,2,figsize=(9.2,5.8),sharex=True,sharey="row")
    for col,size in enumerate(sizes):
        size_rows=[r for r in rows if r["arms"][0]["parameters"]==size]; ref=size_rows[0]["arms"][0]
        axes[0,col].set_title(f"{size/1e6:.2f}M parameters",fontsize=12,fontweight="bold",pad=9)
        for row,(field,reference) in enumerate((("action",ref["action"]["ridge"]["endpoint_px"]),
                                                 ("contact",ref["contact"]["average_precision"]["raw_motion"]))):
            ax=axes[row,col]; ax.axhline(reference,linestyle=(0,(4,3)),color="#626d74",linewidth=1)
            for idx,core in enumerate(("hybrid","gpt2")):
                x=[]; mean=[]; low=[]; high=[]
                for step in sorted({d["step"] for d in size_rows}):
                    obs=[d for d in size_rows if d["step"]==step]
                    scores=[(d["arms"][idx]["action"]["joint"]["endpoint_px"] if field=="action" else
                             d["arms"][idx]["contact"]["average_precision"]["correct"]) for d in obs]
                    x.append(obs[0]["positions_per_arm"]/1e6); mean.append(np.mean(scores)); low.append(min(scores)); high.append(max(scores))
                ax.fill_between(x,low,high,color=colors[core],alpha=.15,linewidth=0)
                ax.plot(x,mean,color=colors[core],linewidth=2,marker="o",markersize=4)
            ax.set_xscale("log",base=2)
            ax.set_xlim(1.80,160)
            ax.set_xticks([2.097152,8.388608,33.554432,134.217728],["2.10","8.39","33.55","134.22"])
            ax.grid(axis="y",linewidth=.5,color="#d9dfe2"); ax.set_axisbelow(True)
            for side in ("right","top"): ax.spines[side].set_visible(False)
            ax.tick_params(axis="both",labelsize=9)
        axes[1,col].set_xlabel("Multimodal positions per arm (millions)",fontsize=10)
    axes[0,0].set_ylabel("Endpoint error (pixels)",fontsize=10)
    axes[1,0].set_ylabel("Contact average precision",fontsize=10)
    fig.suptitle("Matched world-state training on Colab v5e",x=.09,y=.985,ha="left",fontsize=17,fontweight="bold")
    items=[Line2D([0],[0],color=colors["hybrid"],lw=2,label="Shared hybrid"),
           Line2D([0],[0],color=colors["gpt2"],lw=2,label="Dense GPT-2"),
           Line2D([0],[0],color="#626d74",lw=1,ls="--",label="Fitted action prior / raw frame motion")]
    fig.legend(handles=items,loc="upper left",bbox_to_anchor=(.08,.945),ncol=3,frameon=False,fontsize=9)
    fig.text(.09,.025,"Lines show the mean at each verified exposure. Shading spans the available training seeds.\nLower endpoint error and higher contact AP are better. Development covers 256 scenes and three recording dates.",ha="left",fontsize=8,color="#465057")
    fig.subplots_adjust(left=.11,right=.965,top=.82,bottom=.18,wspace=.13,hspace=.34)
    fig.savefig(out,dpi=170,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--metrics",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path); a=p.parse_args();draw(a.metrics,a.out)
