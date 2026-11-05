# Compilation throughput probe, 2026-11-05

Before running: compare the current eager B1 implementation with PyTorch 2.13
`torch.compile` in default mode on the repeated `StreamWorld.process` function.
This reduces repeated CPU/kernel dispatch and fuses supported operations without
changing data, model definitions, batch128, the four independent alignment views,
loss coefficients, optimizer or precision. Keep main B1 execution unchanged.

Use only training people, fresh disposable weights with the registered paired
SLWM/Transformer initialization, and data seed718. Compare first objective and
full connected gradients before any updates. Run an identical second eager
backward as a sensitivity control. The local candidate gate is absolute loss
difference <0.001, relative gradient error <0.02 and cosine >0.999. A local pass
does not establish long-run training-quality equivalence.

Afterward run three warmup and sixteen timed updates for each path with the same
plans, preserving both trajectories. Common CPU batch construction is excluded;
report cold compile/first-backward time separately and derive the amortization
cost if there is a speedup. Cold compilation includes both batch128 and batch8
alignment shapes. No model or optimizer checkpoint from this probe is retained.

The runner must take the shared GPU lock and use a six-minute execution timeout
after acquiring it. Retain registration, errors, timeout/OOM results and all
timings. The source and configuration are recorded before compiler execution.
Compilation failure is a result, not a reason to alter frozen primary runs.

See the [official PyTorch compiler guide](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)
for the distinction between cold compilation and steady performance. The current
installed runtime, not the documentation's latest version, defines this probe.
