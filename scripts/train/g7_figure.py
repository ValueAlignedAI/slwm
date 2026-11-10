"""Learning curves on fresh simulations and rooms at identical exposure, showing the two training seeds."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def draw(metrics,out):
    d=json.loads(metrics.read_bytes())["models"]
    colors={"hybrid":"#075d7a","gpt2":"#a14b26"};x=np.array([8192,10240,12288,16384])
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.2),sharex=True,sharey="row")
    for j,size in enumerate(("anchor","larger")):
        for core,color in colors.items():
            for arm,ls in (("fresh","-"),("replay","--")):
                for i,(metric,mult) in enumerate((("endpoint_px",1),("event_ap",1000))):
                    panel=[]
                    for s in (511,512):
                        parent=d[f"g6-v5e-{size}-{core}-{s}"];fork=d[f"g7-v5e-{size}-{core}-{s}-{arm}"]
                        rr=parent+fork
                        if [r["step"] for r in rr]!=x.tolist():raise ValueError("Incomplete scaling panel")
                        panel.append([r["fresh_dev"][metric]*mult for r in rr])
                    ax=axes[i,j];y=np.array(panel)
                    ax.plot(x,y.mean(0),c=color,ls=ls,lw=1.9,marker="o",ms=3.4)
                    ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=.085,linewidth=0)
        vals=d[f"g6-v5e-{size}-hybrid-511"][0]["fresh_dev"]
        axes[0,j].text(.98,.95,f"Raw ridge {vals['raw_ridge_px']:.1f} px",transform=axes[0,j].transAxes,ha="right",va="top",fontsize=8,color="#454e55")
        axes[1,j].axhline(vals["raw_motion_ap"]*1000,c="#58615d",lw=1,ls=":")
        count=d[f"g6-v5e-{size}-hybrid-511"][0]["params"]
        axes[0,j].set_title(f"{count/1e6:.2f}M parameters",fontsize=13,loc="left",fontweight="bold")
    for ax in axes.ravel():
        ax.grid(axis="y",color="#dbe1e0",lw=.65);ax.spines[["top","right"]].set_visible(False)
        ax.set_xticks(x,["8,192","10,240","12,288","16,384"]);ax.tick_params(labelsize=9)
    axes[0,0].set_ylabel("Push-T endpoint error (pixels)")
    axes[1,0].set_ylabel("Event average precision × 1,000")
    for ax in axes[1]:ax.set_xlabel("Total optimizer updates")
    handles=[Line2D([0],[0],color=v,lw=2,label=k) for k,v in (("Hybrid SLWM",colors["hybrid"]),("Dense GPT-2",colors["gpt2"]))]
    handles.extend([Line2D([0],[0],color="#555",ls=ls,label=name) for name,ls in (("Fresh mixture","-"),("Replay","--"),("Raw frame motion",":"))])
    fig.suptitle("Fresh-domain generalization through continued training",x=.085,ha="left",y=.985,fontsize=16,fontweight="bold")
    fig.legend(handles=handles,ncol=5,loc="upper left",bbox_to_anchor=(.076,.94),frameon=False,fontsize=9)
    fig.text(.085,.018,"Curves show the mean of two independent initializations. Shading spans those seeds. The initial point is the common parent.\nDevelopment uses 256 new simulator starts and five held Finnish rooms. Lower error and higher average precision are favorable.",fontsize=8,color="#37454d")
    fig.subplots_adjust(top=.812,bottom=.17,left=.105,right=.98,hspace=.28,wspace=.13)
    fig.savefig(out,dpi=165,facecolor="white");plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("metrics",type=Path);p.add_argument("out",type=Path);a=p.parse_args();draw(a.metrics,a.out)
