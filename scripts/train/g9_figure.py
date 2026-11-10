"""Plot matched decision, memory, real-signal, and state-width measurements."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS={"hybrid":"#167063","gpt2":"#96562e"}
NAMES={"hybrid":"SLWM","gpt2":"Dense GPT-2"}

def draw(src,out):
    doc=json.loads(src.read_text());rows=doc["rows"];follow=doc["followups"]
    fig,axes=plt.subplots(2,2,figsize=(11.3,7.2),constrained_layout=True)
    def paired(ax,labels,reader,ylabel,suffix=""):
        grid=np.arange(len(labels))
        for core,dx in (("hybrid",-.09),("gpt2",.09)):
            vals=np.array([[reader(core,seed,i) for seed in (911,912)] for i in range(len(labels))])
            for i,v in enumerate(vals):
                ax.plot([grid[i]+dx]*2,v,color=COLORS[core],lw=1.5,alpha=.8)
            ax.scatter(grid+dx,vals[:,0],facecolors="white",edgecolors=COLORS[core],linewidths=1.8,s=43,label=NAMES[core])
            ax.scatter(grid+dx,vals[:,1],c=COLORS[core],s=32)
        ax.set_xticks(grid,labels);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.22);ax.spines[["top","right"]].set_visible(False)
    def row(size,core,field,seed):
        step=384 if size=="pilot" else 4096
        return rows[f"{size}-{core}-z{field}-gelu-s{seed}-fresh/{step}/test"]
    paired(axes[0,0],["13M · 384","13M · 4,096","117M · 4,096"],lambda c,s,i:100*row(("pilot","exposure","confirm")[i],c,128,s)["mean"]["live/carry/occupancy"],"Time at goal (%)")
    axes[0,0].set_title("Training exposure and trunk size · 128-D field",loc="left",fontsize=11)
    axes[0,0].axhline(96.24,lw=1,ls=":",c="#666");axes[0,0].set_ylim(0,104);axes[0,0].legend(loc="center left",frameon=False,fontsize=8)
    paired(axes[0,1],["Native","Full 512","Reset every 32"],lambda c,s,i:100*follow["history_window"][f"confirm-{c}-z512-gelu-s{s}-fresh"][("carry","full","block32")[i]]["occupancy"],"Time at goal (%)")
    axes[0,1].set_title("Inference history · fresh missions, 512-D field",loc="left",fontsize=11);axes[0,1].set_ylim(0,104)
    kinds=["a","v","n","w"]
    def skill(c,s,i):
        k=kinds[i];r=row("confirm",c,512,s)["mean"]
        return 100*(1-r[f"room/{k}/{k}/aligned"]/r[f"prior/{k}/{k}/aligned"])
    paired(axes[1,0],["Audio","RGB","Geometry","Event words"],skill,"Skill vs training prior (%)")
    axes[1,0].axhline(0,lw=.8,c="#555");axes[1,0].set_title("Masked real sensors · held Sony recordings",loc="left",fontsize=11)
    paired(axes[1,1],["128 coordinates","512 coordinates"],lambda c,s,i:row("confirm",c,(128,512)[i],s)["real_field_rank"]["participation_ratio"],"Covariance participation ratio")
    axes[1,1].set_title("Effective state dimension · held room frames",loc="left",fontsize=11);axes[1,1].set_ylim(bottom=0)
    fig.suptitle("G9 matched multimodal world models",fontsize=16,x=.01,ha="left")
    fig.savefig(out,dpi=150,facecolor="white");plt.close(fig)

if __name__=="__main__":draw(Path(sys.argv[1]),Path(sys.argv[2]))
