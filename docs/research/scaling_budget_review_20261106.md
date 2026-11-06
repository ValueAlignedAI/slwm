# Scaling budget review, 2026-11-06

This resource plan uses completed T1 timings and B1 telemetry before B1/P1 final
scores. It does not register another trained candidate or alter the running
comparisons.

The 23:30–00:18 UTC telemetry interval averages 54.05% sampled NVIDIA GPU activity
and peaks at 6,121 MiB device memory on the 24,576 MiB RTX 3090. That establishes
headroom during the observed workload; the activity counter is not achieved
FLOP utilization. The immutable samples and chart are retained under
`broad-stream-20261105/gpu-usage-snapshot-20261106T0020`. Its timestamps, rather
than the approximate directory label, define the measurement interval.

The completed language revision uses global/micro batch64 with fused tied-output
loss. Both architectures failed the batch128 memory fit check, which remains in
the research record. The actual post-ramp training/validation rates are
44.0–45.2k tokens/s for GPT-2 and 29.9–30.5k for SLWM. These exclude checkpoint
loading/saving, queue waits, final tests and mirroring. Reproduce the following
estimates with `python -m training.text_budget_plan`; its JSON records the exact
input-curve hashes.

| Total training tokens per model | Additional hours, one paired seed | Additional hours, both architectures and both seeds |
|---|---:|---:|
| 500M | 5.44 | 10.77 |
| 1B | 13.23 | 26.19 |
| 2.6B | 38.17 | 75.55 |

These are runtime projections, not quality forecasts or scaling laws. The
prepared corpus has exactly 2.6B training tokens; verify the window-boundary
capacity before choosing a final endpoint. A continuation must register a new
learning-rate/token schedule, preserve exact stream order and use fresh,
uninspected test windows. Keep token-matched and timer-matched comparisons
separate. The present 151M-token models use about 1.2 tokens per parameter and
remain repetitive, so increasing token exposure at the same parameter count is
the more informative next language scale study. English web text does not cover
code, physical transitions or general multimodal knowledge.

For multimodal runtime, default-mode compilation gives 1.458x/1.524x steady
SLWM/Transformer model-step throughput; the matched 400-update pilots give
1.174x/1.290x measured training/validation speedups with a warmed disk cache.
Both satisfy the separately registered short-learning quality tolerance, while
the original SLWM local numerical gate remains failed. The original cold
startup of 79.29s/37.63s is additional. A full longer compiled study should keep
its eager comparison, independent seeds and quality gate. Do not substitute it
silently into the current frozen B1 run. MPS concurrency also remains a failed
trajectory-agreement experiment, despite higher aggregate throughput.

Scaling the paired world model needs more independent information as well as
runtime efficiency. The current training set contains 100.58h Libri speech and
8.23h AMI person tracks, but only 2.05h of distinct common meeting time. More
weights cannot create missing active speech, varied scenes or physical action
consequences. Expand participants/sessions and active-speech supervision before
treating a unified 124M multimodal checkpoint as a justified next training
budget. Compare any new temporal vision/content edges through both processors
and count all frozen codecs or teachers. The first DROID source/schema probe is
available separately; two debug episodes establish an input contract, not an
adequate robot-learning dataset or a physical-control result.
