"""Readable evidence for early scale learning, with explicit undertraining limits."""
import argparse
import json
from pathlib import Path
from training.research_sources import file_sha


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t1/fineweb_scale.json"); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); summary=json.loads((root/"summary.json").read_text()); card=json.loads((Path(cfg["corpus"])/"dataset_card.json").read_text())
    lines=["# T1-S2 — early GPT-2-scale causal text learning","", "Four from-scratch runs compare GPT-2-style and causal SLWM text processors at the same token exposure in two seeds. This is an early learning-curve experiment; it is not converged pretraining or a jointly trained audio/video/text model.","",
        f"The immutable corpus contains {card['tokens']['train']:,} training tokens, {card['tokens']['validation']:,} validation and {card['tokens']['test']:,} test tokens. Registrable domains are disjoint ({card['domain_counts']}). Normalized exact documents, repeated long prefixes and canonical URLs are removed globally. Residual near duplicates are not guaranteed absent. Source parquet bytes are pinned by revision/SHA256, and the GPT-2 tokenizer is fixed.","",
        "| Run | Variant / seed | Neural weights | Selected / updates | Tokens seen | Validation NLL | Test NLL / PPL | Minutes |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in cfg["runs"]:
        d=summary["runs"][run["id"]]; test=summary["test"][run["id"]]; lines.append(f"| {run['id']} | {run['variant']} / {run['seed']} | {d['parameters']:,} | {d['selected_step']} / {d['updates']} | {d['tokens_seen']:,} | {d['selected_validation_nll']:.4f} | {test['nll']:.4f} / {test['perplexity']:.2f} | {d['wall_seconds']/60:.2f} |")
    lines += ["","| Seed | SLWM / GPT-2 validation NLL | Early guardrail |","|---|---:|---|"]
    for seed,x in summary["early_guardrail"].items(): lines.append(f"| {seed} | {x['slwm_to_gpt2_validation_nll_ratio']:.4f} | {'PASS' if x['passes_early_guardrail'] else 'FAIL'} |")
    lines += ["",f"Total measured training time on the3090: {summary['training_minutes']:.2f} minutes, including validation. The models see16,777,216 tokens each—about0.13 tokens/parameter—while the prepared corpus provides about20.9 available tokens/parameter. The first stage uses less than1% of that corpus per run, without repeating training windows. More data is prepared than consumed; adequate corpus size does not make this short training converged.","",
        "Both have12layers,768width,12heads and tied token embeddings. Shared embeddings/attention/MLPs have identical paired initializers. SLWM adds causal local, frequency-parameterized and long convolutions (~0.56% extra weights). All temporal paths pass future-token isolation tests. Weights allocate1024 positions, but training and these evaluations use512; no1024-context quality claim is allowed. Extra SLWM compute is measured, not called FLOP-matched.","",
        "Each reported validation or test measurement uses the same fixed 131,072-token sample from its reserved stream (32 batches of 8×512); these are not full-corpus perplexities. Checkpoint choice uses validation only; the test stream is read after all four choices are frozen. The >5% validation-rise early-stop rule and any exposure differences are recorded per run. A positive early guardrail means the causal signal path remains viable at this exposure; it does not establish asymptotic superiority, code ability, grounded reasoning or hallucination reduction.","",
        "Every checkpoint retains optimizer/RNG state; final weights, source hashes, actual token counts and learning curves are saved. `samples.json` contains every fixed prompt with greedy and seeded sampling. These are raw model continuations, never selected for fluency or treated as factual answers. No pretrained language-model weights or recognizer outputs train these runs."
    ]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained")
    for ax,seed in zip(axes,(81,82)):
        for run in cfg["runs"]:
            if run["seed"]!=seed: continue
            curves=json.loads((root/run["id"]/"curves.json").read_text()); ax.plot([r["tokens"]/1e6 for r in curves],[r["validation"]["nll"] for r in curves],marker=".",label=run["variant"])
        ax.set_title("Seed "+str(seed)); ax.set_xlabel("Training tokens (millions)"); ax.set_ylabel("Held-domain validation NLL"); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("GPT-2-scale early learning · equal tokens, from scratch\n2.6B-token corpus available; 16.8M consumed per run"); fig.savefig(root/"learning_curves.png",dpi=180); plt.close(fig)
    print(json.dumps({"scale_report_complete":True,"source_sha256":file_sha(Path(__file__))}))


if __name__=="__main__": main()
