# Multimodal runtime experiments,2026-11-05

Train-only disposable runs examined the GPU gaps in the24M shared-field jobs.
No scientific checkpoint, positive-window plan, selected model, test case or
gate was changed. All original timings, numerical checks and failed alternatives
are retained. The existing frozen B1 trainer continues unchanged.

Two isolated CUDA-MPS clients gave1.548× total update throughput for a mixed
SLWM/Transformer pair and1.666× for two SLWM seeds. Each client retained its own
model, optimizer, batch128 and RNG. Sixteen measured updates followed three
warmups. The entire benchmark held the shared GPU lock, and the task-specific
MPS service was stopped afterward. No benchmark weights were saved.

The predefined numerical check required trajectory loss differences <.005,
gradient-sketch cosine >.999 and relative gradient-sketch error <.02 after the
19 total updates. Both pair types fail: loss differences reach.026; one mixed
Transformer gradient sketch differs2.70%. This does not establish changed
mathematics or unsafe concurrency; it fails the requested trajectory-agreement
check for continuing this already registered comparison. No test score was read.

A narrower single-context control skips the redundant full-batch copy while
preserving every tensor value. Speedups are1.009× SLWM and1.002× Transformer,
with common CPU data preparation excluded. First-gradient differences are
5.65e-9 and3.67e-5 relative, but subsequent BF16/optimizer trajectories still
diverge (final parameter relative differences.00040/.00061). The latter warns
that exact trajectory checks are sensitive even to an arithmetic-neutral data
alias under this runtime. It does not justify silently relaxing the earlier
gate. The negligible copy speedup provides no reason to revise B1 execution.

Next concurrency work should separately register local objective/gradient
agreement against a higher-precision reference, repeated sequential-run
variation, and longer independent-seed quality comparisons. Keep dedicated-run
compute timing separate from shared-GPU elapsed time. Initial throughput gains
alone do not establish preserved learning quality or an architecture advantage.

The successfully adopted text optimization remains separately registered:
global64 fused output loss instead of repeated micro8 accumulation, with pinned
Liger0.8.5 objective/tied-gradient checks and a recorded128-batch OOM. Measured
SLWM text throughput improves roughly18% over the original batch schedule; both
language architectures use the same revision. This does not make the24M AV jobs
or124/125M language models a unified125M multimodal checkpoint.

Sources: `training/stream_concurrency_benchmark.py`,
`training/stream_copy_benchmark.py`; raw artifact folders
`experiments/multimodal/t2/concurrency-benchmark-20261105` and
`experiments/multimodal/t2/copy-benchmark-20261105`.

## Compilation and short learning controls

The subsequent default-mode compilation probe gives1.458x/1.524x steady
model-step throughput for SLWM/Transformer, with common CPU batch construction
excluded. First gradient relative differences are0.271%/0.248%. The SLWM initial
loss difference.001196 misses the registered.001 gate; the original gate stays
failed. Initial compilation/backward costs79.29s/37.63s. The compiled Transformer
local objective/gradient gate passes; neither local result proves learning
equivalence.

Four separate400-update controls then compare eager and compiled execution with
identical shortened LR horizons, seeds, data, optimizers and losses. An initial
shared-cache variant-limit failure is retained, including the200-update SLWM
state; revision2 resumes it with recorded hashes and isolates caches per model.
At the fixed400 endpoint, compiled selection loss changes+0.103% SLWM and−0.137%
Transformer, passing the predeclared≤1% degradation check. Exact final data RNG
states match their eager controls. All four checkpoints also pass ordinary eager
CPU inference; the compiler is not required to use these training artifacts.

Measured pilot speedups are1.174x/1.290x, equivalent to14.8%/22.5% less elapsed
training/validation time. They reuse a persistent compiler cache warmed by the
preceding probe, and include compilation/cache loading that occurs inside each
pilot slice. The prior79.29s/37.63s startup remains additional. These figures are
not empty-cache end-to-end speedups. A single seed and400 updates give an
encouraging runtime candidate for a registered longer study; they do not establish
fresh-person or converged quality. Main B1 remains eager and unchanged.

Artifacts: `compile-benchmark-20261105`, `compiled-learning-20261105` (preserved
failure) and `compiled-learning-v2-20261105`, under the same T2 experiment parent.

The parallel-filter identity suggests two further execution studies: form the
summed convolution kernel differentiably while retaining the original optimizer
parameters, and batch the four independent alignment views together. Both are
real-arithmetic equivalences under the present per-example normalization; BF16
arithmetic, gradients, cold costs and learning quality still need measurement.
They are proposed experiments, not implemented speedup claims.
