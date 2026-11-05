# T1-S4 — larger-batch GPT-2-scale text continuation

Optimizer-preserving continuation of four from-scratch124/125M-parameter language models. The language weights remain separate from the24M audio/video/phonetic-text world fields. This is a learning-curve study, not converged pretraining.

Available training corpus: 2,600,000,000 tokens, split by registrable domain with global exact/prefix/URL deduplication. These runs consume151M/model at the full target, without repeating the registered windows. Fresh test uses524,288 tokens from reserved windows disjoint from the earlier S2 reported test sample. All choices use validation only.

| Run | Variant / seed | Weights | Tokens seen | Selected token step | Actual optimizer updates | Test NLL / PPL | Extension minutes |
|---|---|---:|---:|---:|---:|---:|---:|
| EXP-T1-221 | gpt2 / 81 | 124,439,808 | 150,994,944 | 9216 | 5376 | 4.3302 / 75.96 | 51.46 |
| EXP-T1-222 | slwm / 81 | 125,131,008 | 150,994,944 | 9216 | 5376 | 4.1712 / 64.79 | 75.52 |
| EXP-T1-223 | gpt2 / 82 | 124,439,808 | 150,994,944 | 9216 | 5376 | 4.3277 / 75.77 | 50.26 |
| EXP-T1-224 | slwm / 82 | 125,131,008 | 150,994,944 | 9216 | 5376 | 4.1703 / 64.74 | 74.68 |

| Seed | Equal-token test NLL reduction | SLWM/GPT extension time | Equal total-time validation reduction | Interpolated total-time diagnostic |
|---|---:|---:|---:|---:|
| 81 | +3.67% | 1.467× | +1.04% | +1.05% |
| 82 | +3.64% | 1.486× | +0.10% | +0.92% |

Compute comparisons include each parent's measured timer plus continuation time, evaluated at the shorter shared total-time budget. The parent's original timer includes checkpoint I/O; continuation counters measure training/validation and exclude loading/saving. Both exclude GPU lock queues. This asymmetry remains explicit rather than retrospectively rewriting frozen clocks. Primary comparison uses the best recorded validation checkpoint available before that budget; interpolation is a separately labeled diagnostic. Parent compute is not free. These are timer-budget comparisons, not exact end-to-end wall or identical-FLOP budgets. Equal-token test results and equal-time validation results answer different questions. The extension-only calculation remains in comparisons.json for auditing.

The initial512 additional updates preserve the original32×512 schedule and optimizer/RNG state. Subsequent updates use64×512 and pinned Liger fused cross entropy, preserving the exact token-window order. Validation keeps the original plain cross-entropy microbatch8. The frozen protocol records the learning-rate extension and >5% two-check overfit stop. The smaller S3 partial experiment remains intact rather than being relabeled as S4.

At the full151M target this is only about1.2 tokens per parameter, while2.6B tokens are prepared. Quality remains undertrained; none of these artifacts establishes factual accuracy, reasoning, grounded actions, or eventual scaling superiority. Two seeds estimate reproducibility at one size, not a scaling law. Every fixed greedy/sampled continuation is retained in the run's samples.json, including poor outputs.

The throughput study retains numerical objective/gradient checks, plain and fused alternatives, slower cases and the128-batch out-of-memory result. Selecting64 uses measured throughput rather than maximizing memory occupancy. Both architectures receive the same runtime revision. Checkpoints retain selected/resume optimizer and RNG states, source snapshots, token counts and final weights.
