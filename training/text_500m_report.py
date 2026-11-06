"""Report S5 without changing its frozen trainer or opening tests prematurely."""
import argparse
import json
from pathlib import Path

import numpy as np
from training.research_sources import file_sha
from training.general_sources import write


def status(cfg):
    root=Path(cfg["artifacts"]);runs={};remaining=0
    for run in cfg["runs"]:
        path=root/run["id"]/"curves.json"
        if not path.exists():continue
        curves=json.loads(path.read_text());last=curves[-1]
        earlier=curves[max(0,len(curves)-4)]
        updates=last["new_updates"]-earlier["new_updates"]
        rate=(last["training_validation_seconds"]-earlier["training_validation_seconds"])/updates if updates else last["training_validation_seconds"]/last["new_updates"]
        left=max(0,cfg["additional_updates"]-last["new_updates"]);remaining+=rate*left
        runs[run["id"]]={"variant":run["variant"],"tokens":last["tokens"],"new_updates":last["new_updates"],"validation_nll":last["validation"]["nll"],"recent_seconds_per_update":rate,"remaining_active_hours_estimate":rate*left/3600}
    return {"runs":runs,"estimated_remaining_text_active_hours":remaining/3600,"scope":"Text compute only; excludes initialization, saving, queue time, joint-model work and final evaluation"}


def report(cfg):
    root=Path(cfg["artifacts"]);summary=json.loads((root/"summary.json").read_text());out={};pairs={}
    for run in cfg["runs"]:
        name=run["id"];complete=summary["runs"][name];scores=summary["test"][name]
        assert complete["last_sha256"]==file_sha(root/name/"last.pt")
        curves=json.loads((root/name/"curves.json").read_text())
        out[name]={"variant":run["variant"],"seed":run["seed"],"parameters":complete["parameters"],"tokens":complete["tokens_seen"],"selected_tokens":complete["selected_tokens"],"target_reached":complete["target_reached"],"endpoint":scores["endpoint"],"selected":scores["selected"],"parent":scores["parent"],"continuation_active_hours":complete["training_validation_seconds"]/3600,"endpoint_nll_improvement_over_151m_parent":scores["parent"]["nll"]-scores["endpoint"]["nll"],"sample_file_sha256":file_sha(root/name/"samples.json")}
        pairs.setdefault(run["seed"],{})[run["variant"]]=(name,curves)
    comparisons={}
    for seed,pair in pairs.items():
        a,ac=pair["gpt2"];b,bc=pair["slwm"]
        budget=min(ac[-1]["training_validation_seconds"],bc[-1]["training_validation_seconds"])
        time_matched={}
        for variant,curves in [("gpt2",ac),("slwm",bc)]:
            eligible=[r for r in curves if r["training_validation_seconds"]<=budget]
            if eligible:
                row=eligible[-1];time_matched[variant]={"tokens":row["tokens"],"seconds":row["training_validation_seconds"],"validation_nll":row["validation"]["nll"]}
        ga,sa=out[a],out[b]
        comparisons[str(seed)]={"endpoint_test_nll_gain_gpt2_minus_slwm":ga["endpoint"]["nll"]-sa["endpoint"]["nll"],"perplexity_relative_reduction":1-sa["endpoint"]["perplexity"]/ga["endpoint"]["perplexity"],"continuation_time_ratio_slwm_over_gpt2":sa["continuation_active_hours"]/ga["continuation_active_hours"],"continuation_compute_matched_validation_only":time_matched,"compute_match_scope":"Same151M-token parent exposures, then equal additional active time; parent compute differs. These are validation diagnostics, not a fresh time-matched test."}
    result={"runs":out,"paired_seeds":comparisons,"fresh_test_tokens_per_checkpoint":summary["test_tokens_per_run"],"fresh_test_start_window":summary["test_start_window"],"no_shared_multimodal_weights":True,"warm_restart_recipe":True,"source_sha256":file_sha(__file__)}
    write(root/"findings.json",result)
    lines=["# S5 matched500M-token text continuation","","Fixed endpoints and all seeded raw samples; independent text models, not G1 multimodal weights.","","| Run | Tokens | Params | Fresh test NLL | Perplexity | Extra active h |","|---|---:|---:|---:|---:|---:|"]
    for name,row in out.items():lines.append(f"| {name} ({row['variant']}) | {row['tokens']:,} | {row['parameters']:,} | {row['endpoint']['nll']:.4f} | {row['endpoint']['perplexity']:.2f} | {row['continuation_active_hours']:.2f} |")
    lines += ["","Both architectures resume immutable151M-token parents with optimizer/RNG and unused logical source windows. A registered common warm restart differs from the old uninterrupted recipe. Fresh test positions4096–5119 are scored only after all four weights freeze, including their parents on the same slice.","","The primary comparison is token-matched fresh-test NLL at the fixed endpoint. Additional-time comparisons use the last measured validation point within the common continuation-time budget, with actual token exposure reported. They neither equalize parent compute nor supply a fresh compute-matched test. Queue time, initialization, saving and EMA-free text inference costs remain distinct from active continuation time.","","There are roughly4 training tokens per total parameter at500M. This is a bounded scaling experiment; it does not establish compute-optimal training, a general language assistant, or shared world knowledge. The broad FineWeb-Edu source and domain/byte overlap controls are in its immutable dataset card; no near-duplicate guarantee is added here.","","`findings.json` retains both seeds, endpoint/selected/parent scores and timing tradeoffs. Every fixed greedy and sampled continuation remains in each run's `samples.json`; no examples are selected for fluency.",""]
    (root/"REPORT.md").write_text("\n".join(lines))
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for run in cfg["runs"]:
        rows=json.loads((root/run["id"]/"curves.json").read_text());label=run["variant"]+" "+str(run["seed"])
        axes[0].plot([r["tokens"]/1e6 for r in rows],[r["validation"]["nll"] for r in rows],label=label)
        axes[1].plot([r["training_validation_seconds"]/3600 for r in rows],[r["validation"]["nll"] for r in rows],label=label)
    axes[0].set_xlabel("Total training tokens (millions)");axes[1].set_xlabel("Additional active training/validation hours")
    for axis in axes:axis.set_ylabel("Validation NLL");axis.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(root/"scaling.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("status","report"),required=True);args=parser.parse_args();cfg=json.loads(Path("configs/t1/fineweb_500m.json").read_text())
    if args.stage=="status":print(json.dumps(status(cfg),indent=2))
    else:report(cfg)


if __name__=="__main__":main()
