"""Plot measured recognition error against actual fixed acoustic payload rate."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(src,dest):
    d=json.loads(src.read_bytes())["allocated"]["cases"];bits=(6,24,48,96,192);x=np.array(bits)*46.875/1000
    fig,axes=plt.subplots(1,2,figsize=(9.3,4.4),sharey=True)
    for ax,(fam,title) in zip(axes,(("libri","New audiobook speakers"),("crema","New recordings of held actors"))):
        controls=d[fam]["controls"]
        for key,color,style,label in (("allocated","#087783","-","Distortion-based allocation"),("equal","#ad4d28","--","Equal allocation")):
            vals=[controls[key+str(b)] for b in bits];m=np.array([z["word_error"] for z in vals])*100;ci=np.array([z["speaker_ci95"] for z in vals])*100
            ax.plot(x,m,style,color=color,marker="o",ms=4,lw=2,label=label);ax.fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.085)
        ax.axhline(100*controls["fine"]["word_error"],color="#515252",ls=":",label="Fine-mel reference")
        ax.set_xscale("log",base=2);ax.set_xticks(x,["0.28","1.13","2.25","4.50","9.00"]);ax.set_title(title,loc="left",fontsize=11,fontweight="bold")
        ax.set_xlabel("Acoustic payload (kbit/s)");ax.spines[["right","top"]].set_visible(False);ax.grid(axis="y",color="#d8dfdf",lw=.6)
    axes[0].set_ylabel("Recognizer word error (%)");fig.suptitle("Word recovery at the same acoustic bit rate",x=.08,y=.995,ha="left",fontweight="bold",fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper left",bbox_to_anchor=(.072,.935),ncol=3,frameon=False,fontsize=9)
    fig.text(.08,.026,"Both curves use the same inputs, fitted scalar quantizers and waveform renderer. Bands are speaker-bootstrap intervals.\nThe 132 audiobook excerpts are from 33 dev-other readers. The 56 acted clips are from 14 actors.",fontsize=8,color="#435453")
    fig.subplots_adjust(top=.77,bottom=.22,left=.095,right=.985,wspace=.15);fig.savefig(dest,dpi=160,facecolor="white");plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("dest",type=Path);a=p.parse_args();draw(a.source,a.dest)
