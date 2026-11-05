"""Descriptive wall-time comparison; no new model/step/primary-gate selection."""
import json
from pathlib import Path
import numpy as np
from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    cfg=json.loads(Path("configs/t1/fineweb_scale.json").read_text()); root=Path(cfg["artifacts"]); rows={}
    for seed in (81,82):
        curves={r["variant"]:json.loads((root/r["id"]/"curves.json").read_text()) for r in cfg["runs"] if r["seed"]==seed}; common=min(v[-1]["seconds"] for v in curves.values()); values={}
        for name,points in curves.items(): values[name]={"interpolated_validation_nll":float(np.interp(common,[p["seconds"] for p in points],[p["validation"]["nll"] for p in points])),"interpolated_tokens":float(np.interp(common,[p["seconds"] for p in points],[p["tokens"] for p in points]))}
        rows[str(seed)]={"common_wall_seconds":common,"values":values,"relative_slwm_nll_reduction":1-values["slwm"]["interpolated_validation_nll"]/values["gpt2"]["interpolated_validation_nll"]}
    write_json(root/"compute_diagnostic.json",{"scope":"Post-hoc descriptive interpolation of fixed validation learning curves at equal process wall time, including IO/validation; not equal FLOPs or a measured checkpoint at interpolated steps","seeds":rows,"no_selection_changed":True,"source_sha256":file_sha(Path(__file__))})
    lines=["# Text compute diagnostic","","The primary comparison uses equal tokens. This supplementary analysis linearly interpolates the already fixed validation curves at equal process wall time. It includes validation/IO and can be affected by system load; it is not a matched-FLOP experiment, model-selection rule or directly measured intermediate checkpoint.","", "| Seed | Wall seconds | GPT-2 NLL / tokens | SLWM NLL / tokens | Interpolated NLL reduction |","|---|---:|---:|---:|---:|"]
    for seed,r in rows.items():
        g,s=r["values"]["gpt2"],r["values"]["slwm"]; lines.append(f"| {seed} | {r['common_wall_seconds']:.1f} | {g['interpolated_validation_nll']:.4f} / {g['interpolated_tokens']/1e6:.2f}M | {s['interpolated_validation_nll']:.4f} / {s['interpolated_tokens']/1e6:.2f}M | {100*r['relative_slwm_nll_reduction']:.2f}% |")
    lines += ["","SLWM spends more time per token. Early per-token improvements alone do not establish a compute-efficiency advantage or predict converged behavior. More training and component ablations are required before committing a long scale run."]
    (root/"compute_diagnostic.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__": main()
