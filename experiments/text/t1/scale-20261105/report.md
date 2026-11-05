# T1-S2 — early GPT-2-scale causal text learning

Four from-scratch runs compare GPT-2-style and causal SLWM text processors at the same token exposure in two seeds. This is an early learning-curve experiment; it is not converged pretraining or a jointly trained audio/video/text model.

The immutable corpus contains 2,600,000,000 training tokens, 14,573,946 validation and 10,013,422 test tokens. Registrable domains are disjoint ({'test': 3792, 'train': 751997, 'validation': 3794}). Normalized exact documents, repeated long prefixes and canonical URLs are removed globally. Residual near duplicates are not guaranteed absent. Source parquet bytes are pinned by revision/SHA256, and the GPT-2 tokenizer is fixed.

| Run | Variant / seed | Neural weights | Selected / updates | Tokens seen | Validation NLL | Test NLL / PPL | Minutes |
|---|---|---:|---:|---:|---:|---:|---:|
| EXP-T1-201 | gpt2 / 81 | 124,439,808 | 1024 / 1024 | 16,777,216 | 5.6486 | 5.5715 / 262.84 | 7.95 |
| EXP-T1-202 | slwm / 81 | 125,131,008 | 1024 / 1024 | 16,777,216 | 5.4517 | 5.3768 / 216.33 | 11.35 |
| EXP-T1-203 | gpt2 / 82 | 124,439,808 | 1024 / 1024 | 16,777,216 | 5.6449 | 5.5639 / 260.83 | 7.98 |
| EXP-T1-204 | slwm / 82 | 125,131,008 | 1024 / 1024 | 16,777,216 | 5.4688 | 5.3866 / 218.47 | 11.31 |

| Seed | SLWM / GPT-2 validation NLL | Early guardrail |
|---|---:|---|
| 81 | 0.9651 | PASS |
| 82 | 0.9688 | PASS |

Total measured training time on the3090: 38.59 minutes, including validation. The models see16,777,216 tokens each—about0.13 tokens/parameter—while the prepared corpus provides about20.9 available tokens/parameter. The first stage uses less than1% of that corpus per run, without repeating training windows. More data is prepared than consumed; adequate corpus size does not make this short training converged.

Both have12layers,768width,12heads and tied token embeddings. Shared embeddings/attention/MLPs have identical paired initializers. SLWM adds causal local, frequency-parameterized and long convolutions (~0.56% extra weights). All temporal paths pass future-token isolation tests. Weights allocate1024 positions, but training and these evaluations use512; no1024-context quality claim is allowed. Extra SLWM compute is measured, not called FLOP-matched.

Each reported validation or test measurement uses the same fixed 131,072-token sample from its reserved stream (32 batches of 8×512); these are not full-corpus perplexities. Checkpoint choice uses validation only; the test stream is read after all four choices are frozen. The >5% validation-rise early-stop rule and any exposure differences are recorded per run. A positive early guardrail means the causal signal path remains viable at this exposure; it does not establish asymptotic superiority, code ability, grounded reasoning or hallucination reduction.

Every checkpoint retains optimizer/RNG state; final weights, source hashes, actual token counts and learning curves are saved. `samples.json` contains every fixed prompt with greedy and seeded sampling. These are raw model continuations, never selected for fluency or treated as factual answers. No pretrained language-model weights or recognizer outputs train these runs.
