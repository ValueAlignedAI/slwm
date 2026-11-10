# S5 matched500M-token text continuation

Fixed endpoints and all seeded raw samples; independent text models, not G1 multimodal weights.

| Run | Tokens | Params | Fresh test NLL | Perplexity | Extra active h |
|---|---:|---:|---:|---:|---:|
| EXP-T1-231 (gpt2) | 500,006,912 | 124,439,808 | 3.9234 | 50.57 | 2.53 |
| EXP-T1-232 (slwm) | 500,006,912 | 125,131,008 | 3.7887 | 44.20 | 3.68 |
| EXP-T1-233 (gpt2) | 500,006,912 | 124,439,808 | 3.9105 | 49.92 | 2.49 |
| EXP-T1-234 (slwm) | 500,006,912 | 125,131,008 | 3.7920 | 44.34 | 3.73 |

Both architectures resume immutable151M-token parents with optimizer/RNG and unused logical source windows. A registered common warm restart differs from the old uninterrupted recipe. Fresh test positions4096–5119 are scored only after all four weights freeze, including their parents on the same slice.

The primary comparison is token-matched fresh-test NLL at the fixed endpoint. Additional-time comparisons use the last measured validation point within the common continuation-time budget, with actual token exposure reported. They neither equalize parent compute nor supply a fresh compute-matched test. Queue time, initialization, saving and EMA-free text inference costs remain distinct from active continuation time.

There are roughly4 training tokens per total parameter at500M. This is a bounded scaling experiment; it does not establish compute-optimal training, a general language assistant, or shared world knowledge. The broad FineWeb-Edu source and domain/byte overlap controls are in its immutable dataset card; no near-duplicate guarantee is added here.

`findings.json` retains both seeds, endpoint/selected/parent scores and timing tradeoffs. Every fixed greedy and sampled continuation remains in each run's `samples.json`; no examples are selected for fluency.
