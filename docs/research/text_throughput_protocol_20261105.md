# T1-S4: measured GPU batching and a longer matched text trial

Registered after disposable throughput benchmarks, before the new runtime trains.
The original S3 trial saved 512 additional updates for runs211/212/213; no final
test was read. Preserve those sources, weights and logs unchanged. Resume their
model/AdamW/RNG states into new runs221/222/223; run224 starts from the original
S2 parent. First512 small updates use precisely the S3 schedule and plain loss
for every arm, including the not-yet-trained fourth arm. Then all arms use a
global/micro batch64 and the same fused tied-embedding cross-entropy objective.

The disposable numerical checks find absolute loss differences below0.00005 and
embedding-gradient cosine above0.9999 against the original BF16 implementation.
This is numerical agreement, not bitwise identity. Liger0.8.5 is pinned and counted
as a runtime dependency, not a changed model. Validation/testing keep the original
plain-loss batch8 implementation, so reported measurement arithmetic stays fixed.

Each completed run sees **150,994,944 tokens**:
16,777,216 parent +512×16,384 initial extension +3,840×32,768 large-batch tokens.
Counters named `extension_step` are explicitly16,384-token equivalent updates;
actual optimizer-step counts are also saved (5,376 including the parent).
The token-equivalent extension ends at8,192. No original training window repeats.
Paired models/seeds use the same window order, schedule and batch ramp.

Initial512 LR values reproduce S3 exactly. Continue smoothly from the LR at512,
cosine-decaying over the remaining token budget to2e-5. The larger batch changes
the number of optimizer updates per token, so this is a declared batch-ramp
experiment, not an uninterrupted S2 cosine run. Select by validation only every
8,388,608 tokens; stop after two checks>5% above best. Report unequal exposures.
Fresh test remains the preregistered524,288 tokens from windows512 onward, after
all four choices freeze. Corpus/tokenizer/architectures/context remain fixed.

Benchmarks use already-used windows and disposable weights. Batch32 plain loss
improves throughput roughly10–12%; fused batch64 fits both models with useful
headroom, while SLWM128 exhausts memory. Save all raw timings, including slower
and failed settings. Report wall and memory alongside quality; larger-batch or
per-token advantages do not establish matched-FLOP or converged superiority.

This is still approximately1.2 seen tokens per parameter from a prepared2.6B-token
corpus. Keep all samples and checkpoints. Speech/video/action models are separate
artifact families; no unified151M-token multimodal-world-model claim is allowed.
