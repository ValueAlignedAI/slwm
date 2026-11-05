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
