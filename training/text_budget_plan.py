"""Runtime-only continuation estimates from actual post-ramp training curves."""
import json
from pathlib import Path

from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    cfg=json.loads(Path("configs/t1/fineweb_extend_v2.json").read_text()); root=Path(cfg["artifacts"])
    rates={}; current={}; inputs={}
    for run in cfg["runs"]:
        path=root/run["id"]/"curves.json"; rows=json.loads(path.read_text())
        first=next(row for row in rows if row["extension_step"]==cfg["initial_small_updates"]); last=rows[-1]
        rates[run["id"]]=(last["tokens"]-first["tokens"])/(last["seconds"]-first["seconds"])
        current[run["id"]]=last["tokens"]; inputs[str(path)]=file_sha(path)
    estimates={}
    for goal in (500000000,1000000000,2600000000):
        hours={name:max(0,goal-current[name])/rate/3600 for name,rate in rates.items()}
        estimates[str(goal)]={"additional_hours_by_run":hours,
            "additional_gpu_hours_one_architecture_pair_seed81":hours["EXP-T1-221"]+hours["EXP-T1-222"],
            "additional_gpu_hours_both_architectures_both_seeds":sum(hours.values())}
    result={"method":"Tokens divided by actual post-ramp training/validation timer slope at global/micro batch64; no learning-quality extrapolation",
        "tokens_per_second_by_run":rates,"current_tokens_by_run":current,"input_curve_sha256":inputs,"estimates":estimates,
        "excluded_costs":["GPU lock queues","checkpoint loading/saving","new-stage startup","final evaluation","artifact mirroring"],
        "limits":"Assumes the same512 context, precision, hardware, data pipeline and fused loss. These are runtime estimates, not a scaling law or a promise of quality. New training needs a registered token/LR schedule and fresh uninspected test windows. The2.6B endpoint is a rounded corpus-budget scenario; verify exact prepared unique-window capacity before fitting."}
    write_json(root/"continuation_budget_estimates.json",result); print(json.dumps(result),flush=True)


if __name__=="__main__": main()
