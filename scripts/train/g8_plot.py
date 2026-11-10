"""Plot paired effects of independently refreshed physical and non-simulator data on decision regret."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw(path,out):
    doc=json.loads(path.read_bytes())["panels"];fig,axes=plt.subplots(1,2,figsize=(9.5,4.5),sharey=True)
    rows=[("hybrid",511),("hybrid",512),("gpt2",511),("gpt2",512)];limits=[]
    for ax,(task,title) in zip(axes,(("pusht","Fresh Push-T starts"),("puck_barrier","Unseen puck and barrier"))):
        ax.axvline(0,color="#606a73",lw=1)
        for i,(core,seed) in enumerate(rows):
            info=doc[task]["series"][f"{core}-{seed}"]["paired_contrasts"]["joint_regret_px"]
            for key,offset,color,marker in (("other_main",.12,"#11687b","o"),("sim_main",-.12,"#a54925","s")):
                z=info[key];mean=z["mean"];lo,hi=z["paired_ci95"];limits.extend([lo,hi])
                ax.errorbar(mean,3-i+offset,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],color=color,marker=marker,capsize=3,lw=1.6,ms=4.5)
        ax.set_yticks(range(4),["Dense, seed 512","Dense, seed 511","Hybrid, seed 512","Hybrid, seed 511"]);ax.tick_params(labelsize=9)
        ax.set_ylim(-.55,3.55);ax.grid(axis="y",color="#e4e8e8",lw=.7);ax.spines[["top","right"]].set_visible(False)
        ax.set_title(title,fontsize=12,loc="left",fontweight="bold");ax.set_xlabel("Change in goal regret (pixels)")
    left,right=min(limits+[0]),max(limits+[0]);pad=max(.02,(right-left)*.08)
    for ax in axes:ax.set_xlim(left-pad,right+pad)
    fig.suptitle("Which new training data improved action choice?",x=.13,y=.98,ha="left",fontweight="bold",fontsize=15)
    fig.legend(handles=[Line2D([0],[0],color="#11687b",marker="o",label="New speech, text and real-event data"),Line2D([0],[0],color="#a54925",marker="s",label="New simulator starts")],
               loc="upper left",bbox_to_anchor=(.12,.915),frameon=False,ncol=2,fontsize=9)
    fig.text(.13,.032,"Negative values mean lower realized regret. Points average the two levels of the other factor. Bars are paired 95% intervals.\nEach panel uses 2,048 starts and four goals per start. The simultaneous primary decision is reported separately.",fontsize=8,color="#43515b")
    fig.subplots_adjust(top=.76,bottom=.22,left=.145,right=.98,wspace=.18);fig.savefig(out,dpi=165,facecolor="white");plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("data",type=Path);p.add_argument("out",type=Path);a=p.parse_args();draw(a.data,a.out)
