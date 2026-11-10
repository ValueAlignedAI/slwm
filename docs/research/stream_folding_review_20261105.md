# Parallel-filter inference reduction

The B1 world block applies local5, frequency-parameterized15 and long31 causal
depthwise filters to the same normalized input, adds their outputs, then applies
attention/MLP processing. These three linear branches can be causally aligned
and summed into a single31-tap kernel and summed bias. This changes neither the
real-arithmetic function nor recurrent update. The Fourier coefficient map for
the15-tap branch has rank15; its imaginary DC coordinate is a null direction.
Multiple branches can still change initialization and optimizer geometry while
training. This inference reduction does not show that their training bias is
irrelevant, and it is not a replacement for a matched time-domain training arm.

The T1 text architecture applies separate normalization/residual updates between
its filters, so that entire block cannot be reduced this way. No T1 or B1 training
source/weight was modified. `models/slwm_stream_folded.py` is inference-only and
freezes all resulting parameters. Original checkpoints and confirmation renders
keep the registered unfused path.

The trained-checkpoint prototype passes full-head/working-memory FP32 checks:
worst differences3.82e-6 CPU and5.72e-6 CUDA. Measured batch16 CPU speedup1.059×;
batch128 CUDA FP32 speedup1.150×. CUDA BF16 gives1.249× but maximum differences
reach0.0625, so FP32 algebraic equivalence is not advertised as identical AMP
quality. Timings compare like precision; the FP32 result does not mean it is
faster than the original BF16 default. These are train-only inference benchmarks,
not improved learned quality or a measured single-agent real-time speedup.

All full-model outputs, memory and heads are compared in
`tests/test_stream_folding.py`; the transformation also removes70,656 redundant
allocated filter/bias coordinates in a512×6 world (original23,980,313 versus
23,909,657 deployed parameters). Trained parameter accounting retains the full
original count. Raw metrics, checkpoint hashes and numerical differences are in
`experiments/multimodal/t2/folded-inference-20261105/`. Recheck final selected
artifacts before treating prototype latency as their measured performance.

This points to a concrete compute study: fuse parallel kernel construction while
retaining the original trainable coefficients/optimizer states, compare local
gradients in higher precision and AMP, then run independent matched quality
trials. It also calls for a single31-tap time-domain control before attributing
a future B1 result specifically to Fourier parameterization.
