# T1-S2 — early GPT-2-scale text comparison

Registered before scientific training. This is a separate text-only artifact set
from the audio/video content experiment. It tests causal signal processing at
GPT-2-small parameter scale, not one jointly trained multimodal124M model.

Prepare at least2.5B actual GPT-2 tokens from pinned FineWeb-Edu parquet files:
revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, source SHA256 verified.
Registrable domains are disjoint across training, validation and test. Globally
remove normalized exact documents, repeated long prefixes and canonical URLs.
Report remaining near-duplicate limitations; never claim complete contamination
prevention. Tokenizer vocabulary is fixed, not fitted. Target2.6B training tokens
provides about21 available tokens per124M parameter. This is corpus capacity,
not the amount trained in this first stage.

Run GPT-2-style124,439,808 weights and causal SLWM125,131,008 weights, strict
text-only totals within1%. Both use12layers,768width,12heads, tied GPT-2 token
embeddings. The extra SLWM local/spectral/long convolution paths are causal.
Initialize shared embeddings, attention and MLP weights identically within each
seed; all weights are random, with no pretrained model imported.

Two seeds81/82, each model1024optimizer updates, global32×512 tokens/update,
16,777,216 training tokens/run. Paired variants consume exactly the same token
windows, without replacement. Context weights are allocated for1024 positions,
but this stage trains/evaluates512 only; no1024-context quality claim is allowed.
BF16, AdamW6e-4,64-update warmup/cosine to10%, weight decay.1, clip1.
Model selection minimizes validation cross-entropy every128updates. Stop a run
after two consecutive validation checks>5% above its previous best; report any
exposure mismatch and compare only equal-exposure curves. Test is read once
all checkpoint choices are frozen. It never selects architecture or decoder.

Engineering timings on the3090 use disposable random tokens and no retained
weights: ~36k GPT-2 /25.5k SLWM tokens/s,5.6/6.35GB peak allocation. These are
performance checks, not language evidence.

Record train/validation/test NLL, perplexity, learning curves, counts, exact
token exposures, wall time and all checkpoints/optimizer/RNG states. Report the
text guardrail as early evidence only: SLWM validation NLL≤1.25×paired GPT-2
in both seeds, with both models improving over initialization and no overfit
stop. This small fraction of the available corpus is not compute-optimal or
converged pretraining. It cannot establish final architecture superiority,
reasoning, code skill, hallucination reduction, or a world model.

Save all six fixed prompt continuations, both greedy and seeded sampling, for
every selected checkpoint. No sample is selected for quality. Samples are raw
diagnostics, not answers or validated factual statements.

Sources: [FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[compute/data scaling study](https://arxiv.org/abs/2203.15556). The scaling study
motivates accounting for both data and parameters; it is not a guarantee that
this dataset or a particular token/parameter ratio is optimal for SLWM.
