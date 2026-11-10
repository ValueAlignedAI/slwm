# G3 structured-backbone confirmation and six-layer development

**10 November 2026.** Both 2:4 signal models completed 8,000 joint updates. Each received the same ordered tensors as its full-depth comparison models across 512,000 presentations. The confirmation lock binds the two final weight hashes, the primary endpoint lock and the evaluation code. [Registered pruning protocol](../../../../docs/research/joint_structured_20261109.md) · [Confirmation lock](structured/confirmation_lock.json)

## Held-out transfer with 2:4 backbone weights

Balanced accuracy used a frozen readout fitted with 32 labels per class. Paired entries give seeds 81 and 82. Spoken digits cover two held-out speakers. The row for the pruned parent uses the same pruned backbone before joint adaptation, with initialized signal edges. [Structured comparison and paired intervals](structured/structured-8000-test.json) · [Full-depth and raw controls](summary-8000-test.md)

| Readout input | Spoken digits | Fashion images | Banking intents |
| --- | ---: | ---: | ---: |
| Raw features | 66.4% | 74.5% | 69.3% |
| Unadapted 2:4 signal parent | 45.3% / 50.9% | 70.2% / 70.4% | 69.0% / 68.8% |
| Adapted dense GPT-2-style Transformer | 54.0% / 47.3% | 75.7% / 75.8% | 67.3% / 68.9% |
| Adapted unpruned signal model | 61.2% / 55.4% | 76.2% / 75.5% | 69.4% / 69.8% |
| Adapted 2:4 signal model | 54.4% / 54.9% | 75.5% / 76.2% | 69.7% / 69.3% |

Against the unpruned signal model, pruning lowered digit accuracy by **6.8 and 0.5 percentage points**. The speaker-paired 95% intervals for pruned minus unpruned were −9.0 to −4.6 and −3.0 to +2.0 points. Fashion and Banking each moved less than 0.7 points in either direction. The 2:4 model improved on its own pruned parent on all three tasks in both seeds. Raw speech features retained the highest digit accuracy.

Joint adaptation reduced the pruned parents' text NLL from 4.583 / 4.529 to **4.279 / 4.283** on 12,288 validation tokens per seed. The unpruned signal model reached 4.158 / 4.165 and dense GPT-2 reached 4.284 / 4.271. The remaining language loss relative to unpruned signal was about 0.12 NLL in both seeds.

Each pruned backbone retained 28,311,552 of 56,623,104 entries across its 24 feed-forward matrices. The ordinary dense training kernels took 39.9 / 37.4 active minutes versus 38.6 / 37.2 for the unpruned model. The separate 3090 FP16 matrix test measured 0.458 ms packed and 0.279 ms dense for 1,536 rows and a 3,072 × 768 weight matrix. The measured implementation provides no basis for claiming a speed advantage. [Kernel timings and arithmetic check](structured/kernel.json)

## Six-layer development at 1,000 updates

The dense six-layer signal model reached **42.21% / 43.15%** on development Banking77, compared with **70.45% / 70.49%** at twelve layers. Its own six-layer parents scored 43.31% / 43.04%, the twelve-layer dense GPT-2 controls scored 68.54% / 69.08%, and raw features scored 67.71%. The paired six-minus-twelve-layer intervals were −30.63 to −25.81 and −29.87 to −24.98 percentage points. Six-layer Fashion accuracy was 75.5% / 73.0%, and spoken-digit accuracy was 30.0% / 35.0% on one development speaker. Final depth comparisons still require all six 8,000-update checkpoints and their separate confirmation lock. [Matched six-layer development results](depth6/depth-1000-dev.json)

The inference packages, fitted heads, four reduced language parents, two structured language parents and Push-T locks were audited on Fedora and against the independent disk and Drive copies. [Artifact audit](structured-confirmation-audit-20261110.json)
